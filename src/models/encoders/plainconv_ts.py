"""From-scratch nnU-Net PlainConvUNet encoder as a PatchSet3D image encoder.

The PlainConv twin of `ResEncTSEncoder` (src/models/encoders/resenc_ts.py) — same
no-plans.json, no-checkpoint, He-initialised recipe, just a plain (non-residual) conv
stack instead of ResEnc's residual blocks. Public contract matches every other PatchSet3D
encoder: `forward(B,1,D,H,W) -> (B, out_ch, R, R, R)` with `.out_ch`, `.resolution`,
`.train_spacing_mm`, and the `supports_fine` per-stage taps.

Unlike `NnUNetTSEncoder` (src/models/encoders/nnunet_ts.py), this reads NO plans.json and
loads NO checkpoint — width, depth and normalization are all specified directly here, so a
run isn't tied to a particular TotalSegmentator dataset's plan. Use `nnunet_ts` instead to
load an actual pretrained checkpoint.

Width: `features_per_stage` (explicit per-stage list) if given, else the same nnU-Net
base=32/x2/cap=320 formula `resenc_ts.py` uses (so the default geometry still matches e2's
current 32/64/128/256/320 @ 5 stages). `n_conv_per_stage=2` throughout (nnU-Net's standard
plain-conv schedule) is the one structural difference from ResEnc's (1,3,4,6,6,...) residual
-block schedule; kernel/stride/norm/nonlin otherwise match resenc_ts.py exactly.

Input normalisation (`input_norm`) reuses resenc_ts.py's `passthrough | reframe | zscore`
enum, but defaults to **zscore** here (per-volume, HU-space) rather than resenc_ts's
`passthrough` — the point of decoupling from plans.json is to also drop the fixed-dataset
CTNormalization stats nnunet_ts.py inherits from its plans file.
"""
import torch
import torch.nn as nn
from dynamic_network_architectures.architectures.unet import PlainConvUNet

from src.models.patchset3d import _down_to
from src.models.primus_encoder import _EncodeCache, _cached_encode
from src.models.encoders.nnunet_ts import _key
from src.models.encoders.resenc_ts import _stage_features, _DTYPES
from src.models.encoders._input_norm import InputRenorm, _INPUT_NORMS
from src.totalseg_dataset import resolve_ct_norm

_N_CONV_PER_STAGE = 2   # nnU-Net's standard plain-conv schedule (vs. ResEnc's block counts)

_TRAIN_SPACING_MM = 3.0   # matches resenc_ts.py; only used for RoPE (unused by this conv net)


def _build_plainconv_ts_unet(n_stages: int, features: list[int],
                             num_classes: int = 2) -> PlainConvUNet:
    """PlainConvUNet with `features` channels per stage.

    strides: stage 0 is stride-1, the rest stride-2 (total downsample 2**(n_stages-1)) —
    same schedule as `resenc_ts._build_resenc_unet`. Norm / nonlin / conv_bias are the
    nnUNetTrainer defaults (InstanceNorm3d, LeakyReLU inplace, bias=True).
    """
    return PlainConvUNet(
        input_channels=1,
        n_stages=n_stages,
        features_per_stage=features,
        conv_op=nn.Conv3d,
        kernel_sizes=[(3, 3, 3)] * n_stages,
        strides=[(1, 1, 1)] + [(2, 2, 2)] * (n_stages - 1),
        n_conv_per_stage=[_N_CONV_PER_STAGE] * n_stages,
        num_classes=num_classes,
        n_conv_per_stage_decoder=[_N_CONV_PER_STAGE] * (n_stages - 1),
        conv_bias=True,
        norm_op=nn.InstanceNorm3d,
        norm_op_kwargs={"eps": 1e-5, "affine": True},
        dropout_op=None,
        nonlin=nn.LeakyReLU,
        nonlin_kwargs={"inplace": True},
    )


class PlainConvTSEncoder(nn.Module):
    supports_fine = True          # exposes unpooled per-stage maps (see fine_stage_*)

    def __init__(self, resolution, n_stages=5, stages=(2, 3, 4), features_per_stage=None,
                 input_norm="zscore", loader_ct_norm=None, target_ct_norm="d297",
                 frozen=False, device="cuda", cache_max=4096, precision="bf16"):
        super().__init__()
        self.precision = str(precision).lower()
        if self.precision not in _DTYPES:
            raise ValueError(f"unknown precision {precision!r} ({'|'.join(_DTYPES)})")
        self.resolution = int(resolution)
        if features_per_stage is not None:
            features = [int(f) for f in features_per_stage]
            n_stages = len(features)
        else:
            n_stages = int(n_stages)
            features = _stage_features(n_stages)
        self.n_stages = n_stages
        self.frozen = bool(frozen)
        self.input_norm = str(input_norm)
        if self.input_norm not in _INPUT_NORMS:
            raise ValueError(f"unknown input_norm {input_norm!r} ({'|'.join(_INPUT_NORMS)})")
        # Shared stem: passthrough/reframe/zscore = the previous inline _norm; instance =
        # modality-agnostic per-sample renorm (no HU inversion). Frames unused for
        # passthrough/instance but cheap to resolve.
        self.input_renorm = InputRenorm(
            self.input_norm,
            loader_spec=resolve_ct_norm(loader_ct_norm),
            target_spec=resolve_ct_norm(target_ct_norm))
        self.stages = tuple(int(s) for s in stages)
        if not all(0 <= s < self.n_stages for s in self.stages):
            raise ValueError(f"stages {self.stages} out of range [0, {self.n_stages})")
        self.train_spacing_mm = _TRAIN_SPACING_MM

        net = _build_plainconv_ts_unet(self.n_stages, features)
        net.apply(net.initialize)          # nnU-Net He init (not done in __init__)
        self.encoder = net.encoder
        self.encoder.return_skips = True   # forward -> list of per-stage features

        self.stage_ch = features
        self.out_ch = sum(features[s] for s in self.stages)
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
            # it stays out of the optimizer (cf. nnunet_ts.py / resenc_ts.py).
            for st in self.encoder.stages[max(self.stages) + 1:]:
                for p in st.parameters():
                    p.requires_grad_(False)
        self.encoder.to(device)
        self._cache = _EncodeCache(int(cache_max))

    def _norm(self, x):
        """Delegates to the shared InputRenorm stem (see _input_norm.py)."""
        return self.input_renorm(x)

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
