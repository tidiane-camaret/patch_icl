"""Frozen `fomofo/tap-ct-b-3d` 3D ViT as a PatchSet3D image encoder.

Mirrors PrimusEncoder's contract so it drops into PatchSet3D unchanged:
    forward(B,1,D,H,W) -> (B, out_ch, R, R, R), with .out_ch and .resolution.

PatchSet3D embeds context masks separately, so the encoder only ever sees the 1-channel
image. This bridges the loader's z-scored-HU / RAS tensor to TAP's raw-HU / LPS input
(de-norm, reorient, TAP processor — the SAME bridge the feature-sim study uses, in
experiments/encoders/tapct_features.py, so the training encoder and the feature-sim probe
are byte-identical), encodes at the ANISOTROPIC native token grid (T/8 in-plane, T/4 depth
from patch (4,8,8)), inverse-reorients LPS->RAS so it aligns with the mask occupancy grid,
then resamples to the isotropic R³ grid PatchSet3D consumes.

Weights are fixed on HF (no sidecar). The model is always frozen (a pretrained foundation
encoder; its forward runs under no_grad in the bridge) and kept in eval mode so features
are deterministic — required for the eval encode-cache reuse across epochs.

Unlike Primus there is no RoPE to rescale by physical spacing: TAP uses learned absolute
pos-embeds that it trilinearly INTERPOLATES to whatever grid the input yields. So it always
runs at the native grid (image_size drives the token count) and is NOT spacing-aware; the
physical cell size is set upstream by data.crop_spacing_mm (cell = patch x spacing).
"""
import contextlib
import pathlib
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse Primus's eval encode-cache machinery (CPU-backed LRU, keyed on an input fingerprint)
# so a frozen encoder pays each distinct val crop once, then every later epoch is head-only.
from src.models.primus_encoder import _EncodeCache, _cached_encode


class TapCTEncoder(nn.Module):
    def __init__(self, resolution, image_size, frozen=True, device="cuda",
                 encoder_stage=None, precision="bf16", to_lps=True,
                 resize_native=True, cache_max=4096):
        super().__init__()
        if not frozen:
            # TAP's bridge forward runs under torch.no_grad (embed()), so grads can't flow;
            # a trainable TAP would need a separate grad-enabled path. Out of scope.
            raise NotImplementedError("TapCTEncoder supports frozen=True only")
        enc_dir = pathlib.Path(__file__).resolve().parents[2] / "experiments" / "encoders"
        if str(enc_dir) not in sys.path:
            sys.path.insert(0, str(enc_dir))
        from tapct_features import load_model, make_processor, dense_features
        self._dense_features = dense_features

        T = int(image_size[-1] if isinstance(image_size, (list, tuple)) else image_size)
        assert T % 8 == 0, f"tap_ct needs image_size divisible by 8, got {T}"
        self.T = T
        self.resolution = int(resolution)
        self.device = device
        self.precision = precision
        self.to_lps = bool(to_lps)
        self.frozen = True

        self.model = load_model(torch.device(device), use_sdpa=True)
        self.proc = make_processor(T)
        if not resize_native:
            self.proc.resize_dims = (224, 224)          # stock in-plane upsample to 224^2

        vit = getattr(self.model, "model", self.model)
        self.out_ch = int(getattr(vit, "embed_dim", 768))
        # Early-exit: physically drop the tail transformer blocks (the block loop runs every
        # block, so lowering n_blocks alone saves nothing). Mid-stack (~7/12) often gives the
        # best correspondence AND ~40% less compute (see docs/logs.md, feature-sim max_layers).
        self.encoder_stage = self._truncate_blocks(vit, encoder_stage)

        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.eval()
        # CPU-backed LRU encode cache, active only in frozen eval mode (deterministic loader,
        # no aug) — the first val encodes each distinct crop, later vals are head-only passes.
        self._cache = _EncodeCache(int(cache_max))

    @staticmethod
    def _truncate_blocks(vit, encoder_stage):
        total = int(getattr(vit, "n_blocks", None) or len(vit.blocks))
        if encoder_stage is None:
            return total
        k = int(encoder_stage)
        if k <= 0 or k >= total:
            return total
        if getattr(vit, "chunked_blocks", False):
            print("  [TapCTEncoder] chunked_blocks=True; skipping encoder_stage truncation")
            return total
        vit.blocks = nn.ModuleList(list(vit.blocks)[:k])
        vit.n_blocks = k
        print(f"  [TapCTEncoder] truncated eva to stage {k}/{total} "
              f"(dropped {total - k} blocks; ~{k/total:.0%} of encoder compute)")
        return k

    def train(self, mode: bool = True):
        """Keep the frozen TAP model in eval mode regardless of the parent's train/eval
        toggle (deterministic features; matches the frozen-encoder contract)."""
        super().train(mode)
        self.model.eval()
        return self

    def reset_cache(self):
        self._cache.clear()

    @staticmethod
    def _key(xi):
        """Cheap collision-resistant fingerprint of one input row (1,D,H,W) — mirrors
        PrimusEncoder._key. No spacing term (TAP is not spacing-aware)."""
        flat = xi.reshape(-1)
        n = flat.numel()
        k = min(n, 512)
        idx = torch.linspace(0, n - 1, steps=k, device=flat.device).long()
        sig = torch.round(flat[idx] * 1000).to(torch.int64).tolist()
        return (tuple(xi.shape), round(float(flat.sum()), 3), hash(tuple(sig)))

    @staticmethod
    def _inv_reorient(g):
        """Inverse of tapct_features.ras_to_lps_axial_first on a (C, gS,gP,gL) grid:
        flip the (P,L) grid axes then transpose(2,1,0) -> (C, gR,gA,gS) RAS order, so the
        grid aligns with the mask occupancy grid PatchSet3D pools in the loader frame."""
        return g.flip(2, 3).permute(0, 3, 2, 1).contiguous()

    def _model_device(self):
        # The frozen model may have been moved by the parent's .to(device) after construction,
        # so read the live param device rather than the constructor's `device` string.
        return next(self.model.parameters()).device

    def _encode_one(self, volume, dev):
        """(1,D,H,W) loader tensor -> (out_ch, gR,gA,gS) native grid in RAS frame."""
        rows, gd = self._dense_features(self.model, self.proc, volume, dev,
                                        to_lps=self.to_lps, precision=self.precision)
        g = rows.reshape(*gd, -1).permute(3, 0, 1, 2)       # (C, gS,gP,gL) LPS token order
        if self.to_lps:
            g = self._inv_reorient(g)                        # -> (C, gR,gA,gS) RAS order
        return g.float()

    def _encode_batch(self, x):
        """(B,1,D,H,W) -> (B, out_ch, R, R, R). Native grid is anisotropic (2:1 axial), so
        resample with F.interpolate to R^3 (NOT _down_to, which keys off shape[-1] only)."""
        dev = self._model_device()
        # dense_features/embed run their OWN precision autocast; disable any enclosing (train)
        # autocast so the fp32 weights aren't left half-cast, then honour self.precision inside.
        actx = (torch.autocast(dev.type, enabled=False) if dev.type == "cuda"
                else contextlib.nullcontext())
        with actx:
            grids = [self._encode_one(x[b], dev) for b in range(x.shape[0])]
        f = torch.stack(grids, 0).to(dev)                    # (B, C, gR,gA,gS)
        R = self.resolution
        if tuple(f.shape[-3:]) != (R, R, R):
            f = F.interpolate(f, size=(R, R, R), mode="trilinear", align_corners=False)
        return f

    def forward(self, x, spacing=None):                      # spacing accepted + ignored
        dev = self._model_device()
        x = x.to(dev)
        # Cache only in frozen eval mode; training (aug -> unique volumes) computes directly.
        if self.training:
            return self._encode_batch(x)
        return _cached_encode(self._encode_batch, x, self._key, self._cache)
