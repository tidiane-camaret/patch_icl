"""Frozen (or trainable) TotalSegmentator nnU-Net encoder as a PatchSet3D image encoder.

Loads a pretrained nnU-Net `PlainConvUNet` from a TotalSegmentator weights folder
(plans.json + fold_0/checkpoint_final.pth), keeps only its conv `.encoder`, and exposes
a multi-scale feature grid at R^3 — the same contract as ConvEncoder3D / PrimusEncoder:

    forward(B,1,D,H,W) -> (B, out_ch, R, R, R), with .out_ch, .resolution, .train_spacing_mm.

Default target = Dataset297 (total, 3 mm), whose 5 encoder stages at a 128^3 input land at
128/64/32/16/8^3 with 32/64/128/256/320 channels. We concat a configurable subset of stages
(default {2,3,4}: stage 3 is the native R=16 anchor, stage 2 a 2x avg-pool down, stage 4 a 2x
trilinear up) — skipping the low-level, normalization-dominated stages 0-1.

The encoder was pretrained with nnU-Net `CTNormalization` (clip to [p0.5, p99.5], then z-score
with the dataset foreground mean/std), NOT the loader's CT_MEAN/CT_STD. We invert the loader
z-score back to HU, then re-apply CTNormalization from plans.json so the frozen conv sees
in-distribution intensities.
"""
import json
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from dynamic_network_architectures.architectures.unet import PlainConvUNet

from src.models.patchset3d import _down_to
from src.models.primus_encoder import _EncodeCache, _cached_encode
from src.totalseg_dataset import CT_MEAN, CT_STD

_DTYPES = {"bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
           "fp16": torch.float16, "float16": torch.float16,
           "fp32": None, "float32": None}


def resolve_ts_weights_dir(spec) -> Path:
    """Resolve `spec` to a nnU-Net `..._3d_fullres` model folder.

    Accepts either a full path to such a folder (used as-is), or a short TotalSegmentator
    dataset token — a numeric id (`298`) or name fragment (`total_6mm`) — looked up in the TS
    weights cache (`$nnUNet_results`, else `~/.totalsegmentator/nnunet/results`) / `Dataset<token>*`
    / `*__nnUNetPlans__3d_fullres`, exactly like the native pipeline finds its coarse model. Lets
    configs set `arch.nnunet_ts_weights=total_6mm` portably (the cache path is home-dir-specific).
    Mirrors `src.benchmark_models.totalseg._resolve_weights_dir` (Route B eval)."""
    p = Path(str(spec))
    if p.is_dir():
        return p
    token = str(spec)
    roots = [os.environ.get("nnUNet_results"),
             Path.home() / ".totalsegmentator" / "nnunet" / "results"]
    pat = f"Dataset{token}*" if token.isdigit() else f"*{token}*"
    for root in roots:
        if not root:
            continue
        for ds in sorted(Path(root).glob(pat)):
            hits = sorted(ds.glob("*__3d_fullres")) or sorted(ds.glob("3d_fullres"))
            if hits:
                return hits[0]
    raise FileNotFoundError(
        f"Could not resolve nnunet_ts_weights {spec!r}: not a directory and no "
        f"Dataset{token}*/*_3d_fullres under {[str(r) for r in roots if r]}.")


def _build_plainconv_unet(cfg, num_classes):
    """Instantiate a PlainConvUNet matching an (old-format) nnU-Net plans config.

    Features double from UNet_base_num_features, capped at unet_max_num_features. Norm /
    nonlin / conv_bias are the nnUNetTrainer defaults these plans were trained with
    (InstanceNorm3d, LeakyReLU inplace, bias=True) — cf. src/models/encoders/resenc.py.
    """
    n_conv_enc = list(cfg["n_conv_per_stage_encoder"])
    n_stages = len(n_conv_enc)
    base, maxf = int(cfg["UNet_base_num_features"]), int(cfg["unet_max_num_features"])
    features = [min(base * 2 ** i, maxf) for i in range(n_stages)]
    return PlainConvUNet(
        input_channels=1,
        n_stages=n_stages,
        features_per_stage=features,
        conv_op=nn.Conv3d,
        kernel_sizes=[tuple(k) for k in cfg["conv_kernel_sizes"]],
        strides=[tuple(s) for s in cfg["pool_op_kernel_sizes"]],
        n_conv_per_stage=n_conv_enc,
        num_classes=num_classes,
        n_conv_per_stage_decoder=list(cfg["n_conv_per_stage_decoder"]),
        conv_bias=True,
        norm_op=nn.InstanceNorm3d,
        norm_op_kwargs={"eps": 1e-5, "affine": True},
        dropout_op=None,
        nonlin=nn.LeakyReLU,
        nonlin_kwargs={"inplace": True},
    )


class NnUNetTSEncoder(nn.Module):
    def __init__(self, weights_dir, resolution, stages=(2, 3, 4), frozen=True,
                 device="cuda", cache_max=4096, precision="bf16"):
        super().__init__()
        wd = resolve_ts_weights_dir(weights_dir)
        plans = json.load(open(wd / "plans.json"))
        cfg = plans["configurations"]["3d_fullres"]
        num_classes = len(json.load(open(wd / "dataset.json"))["labels"])

        self.precision = str(precision).lower()
        if self.precision not in _DTYPES:
            raise ValueError(f"unknown precision {precision!r} ({'|'.join(_DTYPES)})")
        self.resolution = int(resolution)
        self.frozen = bool(frozen)
        self.stages = tuple(int(s) for s in stages)
        # nnU-Net CTNormalization params (from plans) — applied to HU, not loader z-score.
        fip = plans["foreground_intensity_properties_per_channel"]["0"]
        self.ct_clip = (float(fip["percentile_00_5"]), float(fip["percentile_99_5"]))
        self.ct_mean, self.ct_std = float(fip["mean"]), float(fip["std"])
        self.train_spacing_mm = float(cfg["spacing"][0])  # isotropic pretrain pitch (297: 3 mm)

        net = _build_plainconv_unet(cfg, num_classes)
        sd = torch.load(wd / "fold_0" / "checkpoint_final.pth",
                        map_location="cpu", weights_only=False)["network_weights"]
        missing, unexpected = net.load_state_dict(sd, strict=False)
        enc_missing = [k for k in missing if k.startswith("encoder.")]
        assert not enc_missing, f"encoder keys unfilled by checkpoint: {enc_missing[:4]}"
        self.encoder = net.encoder
        self.encoder.return_skips = True  # forward -> list of per-stage features
        n_ch = _stage_channels(cfg)
        self.out_ch = sum(n_ch[s] for s in self.stages)

        if self.frozen:
            for p in self.encoder.parameters():
                p.requires_grad_(False)
        self.encoder.to(device)
        self._cache = _EncodeCache(int(cache_max))

    def _norm(self, x):
        """Loader z-scored HU -> nnU-Net CTNormalization (invert loader norm, re-normalize)."""
        hu = x.float() * CT_STD + CT_MEAN
        hu = hu.clamp(*self.ct_clip)
        return (hu - self.ct_mean) / self.ct_std

    def _autocast_ctx(self):
        dev = next(self.encoder.parameters()).device
        if dev.type != "cuda":
            return torch.autocast(device_type="cpu", enabled=False)
        dt = _DTYPES[self.precision]
        if dt is None:
            return torch.autocast(device_type="cuda", enabled=False)
        return torch.autocast(device_type="cuda", dtype=dt, enabled=True)

    def _encode_batch(self, x):
        """(B,1,D,H,W) -> (B,out_ch,R,R,R): selected stages resampled to R^3 and concatenated."""
        v = self._norm(x)
        with self._autocast_ctx():
            if self.frozen:
                with torch.no_grad():
                    feats = self.encoder(v)
            else:
                feats = self.encoder(v)
        picked = [_down_to(feats[s].float(), self.resolution) for s in self.stages]
        return torch.cat(picked, dim=1)

    def forward(self, x, spacing=None):  # spacing accepted + ignored (conv net, no RoPE)
        dev = next(self.encoder.parameters()).device
        x = x.to(dev)
        if not (self.frozen and not self.training):
            return self._encode_batch(x)
        return _cached_encode(self._encode_batch, x,
                              lambda xi: _key(xi), self._cache)


def _stage_channels(cfg):
    base, maxf = int(cfg["UNet_base_num_features"]), int(cfg["unet_max_num_features"])
    n = len(cfg["n_conv_per_stage_encoder"])
    return [min(base * 2 ** i, maxf) for i in range(n)]


def _key(xi):
    """Cheap collision-resistant fingerprint of one input row (1,D,H,W); cf. PrimusEncoder."""
    flat = xi.reshape(-1)
    n = flat.numel()
    k = min(n, 512)
    idx = torch.linspace(0, n - 1, steps=k, device=flat.device).long()
    sig = torch.round(flat[idx] * 1000).to(torch.int64).tolist()
    return (tuple(xi.shape), round(float(flat.sum()), 3), hash(tuple(sig)))
