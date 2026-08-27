"""From-scratch nnU-Net ResidualEncoderUNet encoder as a PatchSet3D image encoder.

The ResEnc twin of `NnUNetTSEncoder` (src/models/encoders/nnunet_ts.py). Same public
contract — `forward(B,1,D,H,W) -> (B, out_ch, R, R, R)` with `.out_ch`, `.resolution`,
`.train_spacing_mm`, and the `supports_fine` per-stage taps — but the architecture is
specified inline (no plans.json, no checkpoint) and the weights are always He-initialised.

Width follows the nnU-Net ResEnc-M/L/XL recipe: base 32, x2 per stage, capped at 320, with
the residual block schedule (1, 3, 4, 6, 6, 6, ...). nnU-Net's M/L/XL presets differ ONLY
in the planned patch size (which sets how many downsampling stages the net gets), so
`n_stages` is that knob here. n_stages=5 reproduces e2's PlainConvUNet stage geometry
(sides /1 /2 /4 /8 /16, channels 32/64/128/256/320) with residual blocks -> ~10x the params.

Input normalisation (`input_norm`) — the image already arrives in the pipeline's CT frame
(the provider's CtNormSpec, default `fingerprint_1228`), so:
    passthrough — do nothing. The from-scratch default: one frame end-to-end, zero conversions.
    reframe     — invert the loader frame back to HU, then apply `target_ct_norm` (default the
                  `d297` preset). Only for weights pretrained in a different frame.
    zscore      — invert to HU, then per-volume z-score (no clip). nnInteractive "nonCT" style.
"""
import torch
import torch.nn as nn
from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet

from src.models.patchset3d import _down_to
from src.models.primus_encoder import _EncodeCache, _cached_encode
from src.models.encoders.nnunet_ts import _key
from src.totalseg_dataset import resolve_ct_norm

_DTYPES = {"bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
           "fp16": torch.float16, "float16": torch.float16,
           "fp32": None, "float32": None}

# nnU-Net ResEnc encoder block schedule (dynamic_network_architectures ResEncUNetPlanner);
# the first `n_stages` entries are used.
_BLOCKS_PER_STAGE = (1, 3, 4, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6)

_BASE_FEATURES = 32
_MAX_FEATURES = 320

_TRAIN_SPACING_MM = 3.0   # nnU-Net ResEnc recipe pitch (Dataset297); only used for RoPE

_INPUT_NORMS = ("passthrough", "reframe", "zscore")


def _stage_features(n_stages: int) -> list[int]:
    return [min(_BASE_FEATURES * 2 ** i, _MAX_FEATURES) for i in range(n_stages)]


def _build_resenc_unet(n_stages: int, num_classes: int = 2) -> ResidualEncoderUNet:
    """ResidualEncoderUNet matching the nnU-Net ResEnc recipe for `n_stages` stages.

    strides: stage 0 is stride-1, the rest stride-2 (total downsample 2**(n_stages-1)).
    Norm / nonlin / conv_bias are the nnUNetTrainer defaults (InstanceNorm3d, LeakyReLU
    inplace, bias=True), matching src/models/encoders/resenc.py and nnunet_ts.py.
    """
    return ResidualEncoderUNet(
        input_channels=1,
        n_stages=n_stages,
        features_per_stage=_stage_features(n_stages),
        conv_op=nn.Conv3d,
        kernel_sizes=[(3, 3, 3)] * n_stages,
        strides=[(1, 1, 1)] + [(2, 2, 2)] * (n_stages - 1),
        n_blocks_per_stage=list(_BLOCKS_PER_STAGE[:n_stages]),
        num_classes=num_classes,
        n_conv_per_stage_decoder=[1] * (n_stages - 1),
        conv_bias=True,
        norm_op=nn.InstanceNorm3d,
        norm_op_kwargs={"eps": 1e-5, "affine": True},
        dropout_op=None,
        nonlin=nn.LeakyReLU,
        nonlin_kwargs={"inplace": True},
    )


class ResEncTSEncoder(nn.Module):
    supports_fine = True          # exposes unpooled per-stage maps (see fine_stage_*)

    def __init__(self, resolution, n_stages=5, stages=(2, 3, 4),
                 input_norm="passthrough", loader_ct_norm=None, target_ct_norm="d297",
                 frozen=False, device="cuda", cache_max=4096, precision="bf16"):
        super().__init__()
        self.precision = str(precision).lower()
        if self.precision not in _DTYPES:
            raise ValueError(f"unknown precision {precision!r} ({'|'.join(_DTYPES)})")
        self.resolution = int(resolution)
        self.n_stages = int(n_stages)
        self.frozen = bool(frozen)
        self.input_norm = str(input_norm)
        if self.input_norm not in _INPUT_NORMS:
            raise ValueError(f"unknown input_norm {input_norm!r} ({'|'.join(_INPUT_NORMS)})")
        self.stages = tuple(int(s) for s in stages)
        if not all(0 <= s < self.n_stages for s in self.stages):
            raise ValueError(f"stages {self.stages} out of range [0, {self.n_stages})")
        # Frames for the reframe/zscore paths (unused under passthrough).
        self._loader_spec = resolve_ct_norm(loader_ct_norm)
        self._target_spec = resolve_ct_norm(target_ct_norm)
        self.train_spacing_mm = _TRAIN_SPACING_MM

        net = _build_resenc_unet(self.n_stages)
        net.apply(net.initialize)          # nnU-Net He init (not done in __init__)
        self.encoder = net.encoder
        self.encoder.return_skips = True   # forward -> list of per-stage features

        n_ch = _stage_features(self.n_stages)
        self.stage_ch = n_ch
        self.out_ch = sum(n_ch[s] for s in self.stages)
        # Cumulative stride per stage (stage 0 is stride-1) -> input-side divisor.
        self.stage_div, d = [], 1
        for i in range(self.n_stages):
            d *= 1 if i == 0 else 2
            self.stage_div.append(d)

        if self.frozen:
            for p in self.encoder.parameters():
                p.requires_grad_(False)
        else:
            # Nothing downstream reads a stage deeper than max(stages); freeze the tail so
            # it stays out of the optimizer (cf. nnunet_ts.py).
            for st in self.encoder.stages[max(self.stages) + 1:]:
                for p in st.parameters():
                    p.requires_grad_(False)
        self.encoder.to(device)
        self._cache = _EncodeCache(int(cache_max))

    def _norm(self, x):
        """passthrough: identity. reframe/zscore: invert the loader frame to HU first."""
        x = x.float()
        if self.input_norm == "passthrough":
            return x
        hu = x * self._loader_spec.std + self._loader_spec.mean
        if self.input_norm == "reframe":
            t = self._target_spec
            return (hu.clamp(t.clip_lo, t.clip_hi) - t.mean) / t.std
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
        """Encoder forward -> list of per-stage maps at native resolutions."""
        v = self._norm(x)
        with self._autocast_ctx():
            if self.frozen:
                with torch.no_grad():
                    return self.encoder(v)
            return self.encoder(v)

    def _encode_batch(self, x):
        """(B,1,D,H,W) -> (B,out_ch,R,R,R): selected stages resampled to R^3, concatenated."""
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
            return self._encode_fine_batch(x, fine_rows, fine_stage)
        if not (self.frozen and not self.training):
            return self._encode_batch(x)
        return _cached_encode(self._encode_batch, x, lambda xi: _key(xi), self._cache)
