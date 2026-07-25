# experiments/3d/feature_sim/adapters.py
"""Encoder-agnostic feature adapters for the similarity study.

EncoderAdapter maps volumes -> per-cell feature grids at an arbitrary resolution
(dense) or trilinearly-sampled point features (native res). PatchSet3DEncoderAdapter
wraps a loaded PatchSet3D. Future SAM/DINO adapters implement the same interface."""
from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F

from src.models.patchset3d import _down_to


class EncoderAdapter(ABC):
    @property
    @abstractmethod
    def R(self) -> int: ...

    @abstractmethod
    def tiers(self) -> list[str]: ...

    @abstractmethod
    def native_res(self, tier: str, input_res: int) -> int: ...

    @abstractmethod
    def features(self, volumes, tier, res): ...

    @abstractmethod
    def sample_features(self, volumes, tier, coords): ...


class PatchSet3DEncoderAdapter(EncoderAdapter):
    def __init__(self, model):
        self.model = model.eval()
        self.enc = model.encoder
        self._concat_ch = self.enc.out_ch

    @property
    def R(self):
        return self.model.resolution

    @property
    def n_stages(self):
        return len(self.enc.stages)               # excludes the stem

    def tiers(self):
        stages = [f"stage:{i}" for i in range(self.n_stages + 1)]
        return stages + ["concat", "img_embed"]

    def native_res(self, tier, input_res):
        if tier.startswith("stage:"):
            return input_res >> int(tier.split(":")[1])
        if tier in ("concat", "img_embed"):
            return input_res                       # stem-limited, finest genuine
        raise ValueError(f"unknown tier {tier!r}")

    @torch.no_grad()
    def _stage_feats(self, volumes):
        feats = [self.enc.stem(volumes)]
        for stage in self.enc.stages:
            feats.append(stage(feats[-1]))
        return feats                               # [stem, stage1, ...] native res

    @torch.no_grad()
    def _concat_native(self, feats):
        """Concat all stages at the finest (stem) native res — matches encoder semantics
        but keeps native detail instead of pooling to R."""
        r = feats[0].shape[-1]
        return torch.cat([_down_to(f, r) if f.shape[-1] != r else f for f in feats], 1)

    @torch.no_grad()
    def features(self, volumes, tier, res):
        feats = self._stage_feats(volumes)
        if tier.startswith("stage:"):
            f = feats[int(tier.split(":")[1])]
        elif tier == "concat":
            f = self._concat_native(feats)
        elif tier == "img_embed":
            f = self._concat_native(feats)         # projected below at target res
        else:
            raise ValueError(f"unknown tier {tier!r}")
        f = _down_to(f, res)                        # (B,C,res,res,res)
        if tier == "img_embed":
            B, C = f.shape[0], f.shape[1]
            flat = f.flatten(2).transpose(1, 2)     # (B, res^3, C)
            emb = self.model.img_embed(flat)        # (B, res^3, e)
            f = emb.transpose(1, 2).reshape(B, emb.shape[-1], res, res, res)
        return f

    @torch.no_grad()
    def sample_features(self, volumes, tier, coords):
        """coords (B,N,3) normalized in (z,y,x)=(d,h,w) order -> (B,N,C)."""
        feats = self._stage_feats(volumes)
        if tier.startswith("stage:"):
            f = feats[int(tier.split(":")[1])]
        elif tier in ("concat", "img_embed"):
            f = self._concat_native(feats)
        else:
            raise ValueError(f"unknown tier {tier!r}")
        xyz = coords.flip(-1).view(coords.shape[0], coords.shape[1], 1, 1, 3)  # ->(x,y,z)
        s = F.grid_sample(f, xyz, mode="bilinear", align_corners=True)          # (B,C,N,1,1)
        s = s.squeeze(-1).squeeze(-1).transpose(1, 2)                           # (B,N,C)
        if tier == "img_embed":
            s = self.model.img_embed(s)
        return s

    @torch.no_grad()
    def transformer_query(self, image, context_in, context_out):
        """Post-transformer query rep (B,N,e) via a decoder-input hook (res=R only)."""
        captured = {}
        h = self.model.decoder.register_forward_pre_hook(
            lambda mod, args: captured.setdefault("q", args[0]))
        try:
            self.model(image, context_in=context_in, context_out=context_out, mode="train")
        finally:
            h.remove()
        return captured["q"]

    @torch.no_grad()
    def transformer_pair(self, image, context_in, context_out):
        """Post-transformer img-column tokens for BOTH target and context, in the same
        space e (res=R). Unlike transformer_query (target post- vs context PRE-transformer,
        a mismatched probe), this reads the context tokens the transformer actually produced,
        so target<->context correspondence is measured cleanly on the post-transformer rep.

        Returns (target (B,N,e), context (B,K*N,e)). Sequence after ThinkingRows is
        [think(n), context(K*N), target(N)] with sep_t = K*N + n; column 0 is the img token."""
        cap = {}

        def hook(mod, args, output):
            cap["x"], cap["sep_t"] = output, args[1]       # transformer(x, sep_t, ...)

        h = self.model.transformer.register_forward_hook(hook)
        try:
            self.model(image, context_in=context_in, context_out=context_out, mode="train")
        finally:
            h.remove()
        x, sep_t = cap["x"], cap["sep_t"]
        n = self.model.thinking.n
        return x[:, sep_t:, 0, :], x[:, n:sep_t, 0, :]

    @torch.no_grad()
    def transformer_pair_per_layer(self, image, context_in, context_out):
        """Like transformer_pair but returns the (target, context) img-token pair after
        EACH transformer block, so correspondence can be traced layer by layer. Free: the
        forward already runs every block; the hooks only capture its output tensor.

        Token layout + sep_t are invariant across blocks, so the same slicing applies at
        every depth. Returns a list of (target (B,N,e), context (B,K*N,e)), one per block."""
        outs = []
        hs = [b.register_forward_hook(lambda m, a, o: outs.append((o, a[1])))
              for b in self.model.transformer.blocks]
        try:
            self.model(image, context_in=context_in, context_out=context_out, mode="train")
        finally:
            for h in hs:
                h.remove()
        n = self.model.thinking.n
        return [(x[:, s:, 0, :], x[:, n:s, 0, :]) for x, s in outs]
