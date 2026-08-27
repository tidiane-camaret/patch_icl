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

`random_init=True` keeps everything plans.json defines (architecture, CTNormalization,
pretrain spacing) but skips the checkpoint and applies nnU-Net's own He init instead — the
from-scratch control for "how much does supervised CT pretraining buy in-context?".
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
from src.totalseg_dataset import resolve_ct_norm

_DTYPES = {"bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
           "fp16": torch.float16, "float16": torch.float16,
           "fp32": None, "float32": None}

_INPUT_NORMS = ("passthrough", "reframe", "zscore")


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
    supports_fine = True          # exposes unpooled per-stage maps (see encode_with_fine)

    def __init__(self, weights_dir, resolution, stages=(2, 3, 4), frozen=True,
                 device="cuda", cache_max=4096, precision="bf16", random_init=False,
                 input_norm="reframe", loader_ct_norm=None):
        super().__init__()
        self.input_norm = str(input_norm)
        if self.input_norm not in _INPUT_NORMS:
            raise ValueError(f"unknown input_norm {input_norm!r} ({'|'.join(_INPUT_NORMS)})")
        self._loader_spec = resolve_ct_norm(loader_ct_norm)
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
        self.random_init = bool(random_init)
        if self.random_init:
            # From-scratch control: keep the ARCHITECTURE, the CTNormalization stats and the
            # pretrain spacing that plans.json defines, and throw away only the trained weights
            # (so a scratch run differs from the finetune run in exactly one thing).
            #
            # The explicit apply() is not optional. PlainConvUNet does NOT initialize in
            # __init__ — nnU-Net does it in the trainer (get_network_from_plans ->
            # network.apply(network.initialize) -> InitWeights_He(1e-2)). Just skipping the
            # load would silently leave PyTorch's Conv3d default (kaiming_uniform, a=sqrt(5)),
            # whose std is ~2.4x SMALLER than He. That matters here because every conv is
            # followed by InstanceNorm3d(affine=True): conv weights are scale-invariant in the
            # forward, so the init scale does not set the signal scale, it sets the EFFECTIVE
            # step size (Adam moves ~lr per element, so the relative move is lr / weight-RMS).
            # Measured on Dataset291, deepest tapped stage (s4, 320x320x3^3): torch-default RMS
            # 0.0062, He 0.0152, converged pretrained 0.0070 — i.e. He starts ~2.2x above where
            # nnU-Net's own training ends up, and moves ~2.4x more slowly per step than the
            # default would. He is both the nnU-Net-faithful and the stabler choice.
            net.apply(net.initialize)
        else:
            sd = torch.load(wd / "fold_0" / "checkpoint_final.pth",
                            map_location="cpu", weights_only=False)["network_weights"]
            missing, unexpected = net.load_state_dict(sd, strict=False)
            enc_missing = [k for k in missing if k.startswith("encoder.")]
            assert not enc_missing, f"encoder keys unfilled by checkpoint: {enc_missing[:4]}"
        self.encoder = net.encoder
        self.encoder.return_skips = True  # forward -> list of per-stage features
        n_ch = _stage_channels(cfg)
        self.out_ch = sum(n_ch[s] for s in self.stages)
        self.stage_ch = n_ch
        # Cumulative stride per stage (stage 0 is stride-1 in these plans) -> the divisor
        # from input side to that stage's native side. Read from plans, not assumed.
        self.stage_div, d = [], 1
        for st in cfg["pool_op_kernel_sizes"]:
            d *= int(st[0])
            self.stage_div.append(d)

        if self.frozen:
            for p in self.encoder.parameters():
                p.requires_grad_(False)
        else:
            # Trainable encoder: nothing downstream ever reads a stage deeper than
            # max(stages) — the forward still runs them, but their output is dropped, so
            # their params only ever see an all-zero grad. (The fine_decode taps are always
            # SHALLOWER than the coarse anchor: a deeper fine stage fails PatchSet3D's
            # `fine_side % resolution` check at construction.) Freeze them so they stay out
            # of the optimizer instead of collecting AdamW state and being weight-decayed for
            # nothing, and so the trainable-param count train.py logs is honest. Dataset291:
            # stage 5 = 5.5M of the encoder's 14.0M params.
            for st in self.encoder.stages[max(self.stages) + 1:]:
                for p in st.parameters():
                    p.requires_grad_(False)
        self.encoder.to(device)
        self._cache = _EncodeCache(int(cache_max))

    def _norm(self, x):
        """passthrough: identity. reframe (default): invert the loader frame to HU, then
        apply this checkpoint's plans CTNormalization. zscore: invert, then per-volume."""
        x = x.float()
        if self.input_norm == "passthrough":
            return x
        hu = x * self._loader_spec.std + self._loader_spec.mean
        if self.input_norm == "reframe":
            return (hu.clamp(*self.ct_clip) - self.ct_mean) / self.ct_std
        flat = hu.reshape(hu.shape[0], -1)                       # zscore
        mu = flat.mean(dim=1).reshape(-1, 1, 1, 1, 1)
        sig = flat.std(dim=1).reshape(-1, 1, 1, 1, 1)
        return (hu - mu) / (sig + 1e-8)

    def _autocast_ctx(self):
        dev = next(self.encoder.parameters()).device
        if dev.type != "cuda":
            return torch.autocast(device_type="cpu", enabled=False)
        dt = _DTYPES[self.precision]
        if dt is None:
            return torch.autocast(device_type="cuda", enabled=False)
        return torch.autocast(device_type="cuda", dtype=dt, enabled=True)

    def _stage_feats(self, x):
        """Frozen-safe encoder forward -> list of per-stage maps at native resolutions."""
        v = self._norm(x)
        with self._autocast_ctx():
            if self.frozen:
                with torch.no_grad():
                    return self.encoder(v)
            return self.encoder(v)

    def _encode_batch(self, x):
        """(B,1,D,H,W) -> (B,out_ch,R,R,R): selected stages resampled to R^3 and concatenated."""
        feats = self._stage_feats(x)
        picked = [_down_to(feats[s].float(), self.resolution) for s in self.stages]
        return torch.cat(picked, dim=1)

    @property
    def n_fine_stages(self) -> int:
        return len(self.stage_ch)

    def fine_stage_channels(self, stage: int) -> int:
        return self.stage_ch[stage]

    def fine_stage_size(self, in_size: int, stage: int) -> int:
        return int(in_size) // self.stage_div[stage]

    def _encode_fine_batch(self, x, fine_rows, fine_stage):
        """One forward -> (coarse (B,out_ch,R,R,R), one fine map per stage in `fine_stage`)."""
        feats = self._stage_feats(x)
        coarse = torch.cat([_down_to(feats[s].float(), self.resolution) for s in self.stages], dim=1)
        return coarse, tuple(feats[st].index_select(0, fine_rows.to(feats[st].device))
                             for st in fine_stage)

    def forward(self, x, spacing=None, fine_rows=None, fine_stage=None):
        # spacing accepted + ignored (conv net, no RoPE)
        dev = next(self.encoder.parameters()).device
        x = x.to(dev)
        if fine_rows is not None:
            # Not cacheable: the encode cache stores only the coarse R^3 map, and a fine map
            # is far too large to keep — so a fine-decode run re-encodes each eval crop.
            return self._encode_fine_batch(x, fine_rows, fine_stage)
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
