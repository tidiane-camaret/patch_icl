"""Frozen (or trainable) nnUNet Primus ViT as a PatchSet3D image encoder.

PatchSet3D embeds context masks separately, so its encoder only ever sees the
image (1 channel). This wraps the Primus ViT encoder (down_projection + eva, no
segmentation decoder) to the same contract as ConvEncoder3D:
    forward(B,1,D,H,W) -> (B, out_ch, R, R, R), with .out_ch and .resolution.
Weights + arch + HU preprocessing come from the CoLiPri extraction sidecar.
"""
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

# _down_to is defined at module top of patchset3d (before the class), so this import
# resolves even though patchset3d imports PrimusEncoder lazily inside its __init__.
from src.models.patchset3d import _down_to
from src.totalseg_dataset import CT_MEAN, CT_STD


class PrimusEncoder(nn.Module):
    def __init__(self, sidecar_path, resolution, frozen=True, device="cuda"):
        super().__init__()
        from dynamic_network_architectures.architectures.primus import Primus
        with open(sidecar_path) as f:
            meta = json.load(f)
        kw = dict(meta["primus_kwargs"])
        self.input_shape = tuple(kw["input_shape"])
        self.preproc = meta.get("preproc")
        self.resolution = int(resolution)
        self.out_ch = int(kw["embed_dim"])
        self.frozen = bool(frozen)
        self.primus = Primus(**kw)
        weights = meta.get("weights")
        if weights:
            sd = torch.load(weights, map_location="cpu", weights_only=False)
            sd = sd.get("model", sd) if isinstance(sd, dict) else sd
            missing, unexpected = self.primus.load_state_dict(sd, strict=False)
            print(f"[PrimusEncoder] loaded weights: {len(missing)} missing "
                  f"(up_projection decoder, unused), {len(unexpected)} unexpected")
        if self.frozen:
            for p in self.primus.parameters():
                p.requires_grad_(False)
        self.primus.to(device)

    def _preprocess(self, x):
        """(B,1,D,H,W) loader z-scored HU -> resampled to input_shape, encoder-normalised."""
        v = x.float()
        if self.preproc is not None:
            hu = v * CT_STD + CT_MEAN
            hu = hu.clamp(self.preproc["clip_min"], self.preproc["clip_max"])
            v = (hu - self.preproc["mean"]) / self.preproc["std"]
        if tuple(v.shape[-3:]) != self.input_shape:
            v = F.interpolate(v, size=self.input_shape, mode="trilinear", align_corners=False)
        return v

    def _encode(self, x):
        """Primus ViT encoder only (down_projection + eva) -> (B, out_ch, g, g, g)."""
        p = self.primus
        x = p.down_projection(x)
        B, C, W, H, D = x.shape
        x = x.flatten(2).transpose(1, 2)
        if p.register_tokens is not None:
            x = torch.cat([p.register_tokens.expand(B, -1, -1), x], dim=1)
        x, keep = p.eva(x)
        assert keep is None, "patch dropping must be off for dense features"
        if p.register_tokens is not None:
            x = x[:, p.register_tokens.shape[1]:]
        return x.transpose(1, 2).reshape(B, self.out_ch, W, H, D)

    def forward(self, x):
        dev = next(self.primus.parameters()).device
        v = self._preprocess(x.to(dev))
        if self.frozen:
            with torch.no_grad():
                f = self._encode(v)
        else:
            f = self._encode(v)
        return _down_to(f.float(), self.resolution)
