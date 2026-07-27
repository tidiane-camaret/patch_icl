# experiments/3d/feature_sim/adapters.py
"""Encoder-agnostic feature adapters for the similarity study.

EncoderAdapter maps volumes -> per-cell feature grids at an arbitrary resolution
(dense) or trilinearly-sampled point features (native res). PatchSet3DEncoderAdapter
wraps a loaded PatchSet3D; PrimusEncoderAdapter wraps a frozen nnUNet Primus ViT
(weights-pluggable — e.g. a CoLiPri backbone). Future SAM/DINO adapters implement the
same interface."""
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.patchset3d import _down_to
from src.totalseg_dataset import CT_MEAN, CT_STD


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

    def cost_target(self, input_res):
        """(module, example_inputs) for the encode-cost probe: the encoder stem+stages
        forward on one 1-channel volume at the study input res — the exact path
        features() drives, so cost compares like-for-like against other encoders."""
        adapter = self
        dev = next(self.enc.parameters()).device

        class _EncodeFwd(nn.Module):
            def forward(self, x):
                return adapter._stage_feats(x)

        x = torch.zeros(1, 1, input_res, input_res, input_res, device=dev)
        return _EncodeFwd().to(dev), (x,)

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

    @torch.no_grad()
    def transformer_trace(self, image, context_in, context_out):
        """One forward -> named (target, context) img-token pairs: 'encoder' (the transformer
        INPUT = encoder image features before any attention) then 'L{i}' after each block, all
        at res=R on the same token grid. Lets a training run trace how correspondence evolves
        from the (jointly-trained) encoder through the transformer stack. Free: the forward
        already runs every block; hooks only capture tensors. Returns [(name, tgt, ctx), ...]."""
        outs = []
        hp = self.model.transformer.register_forward_pre_hook(
            lambda m, a: outs.append(("encoder", a[0], a[1])))     # (x_in, sep_t)
        hs = [b.register_forward_hook(
                  lambda m, a, o, i=i: outs.append((f"L{i}", o, a[1])))
              for i, b in enumerate(self.model.transformer.blocks)]
        try:
            self.model(image, context_in=context_in, context_out=context_out, mode="train")
        finally:
            hp.remove()
            for h in hs:
                h.remove()
        n = self.model.thinking.n
        return [(name, x[:, s:, 0, :], x[:, n:s, 0, :]) for name, x, s in outs]


# ---------------------------------------------------------------------------
# Generic frozen Primus (nnUNet ViT) adapter — weights-pluggable.
# CoLiPri's vision tower IS a stock nnUNet Primus, so its backbone weights load
# here directly; weights=None gives an architecture-only (random-init) floor.
# ---------------------------------------------------------------------------


class PrimusEncoderAdapter(EncoderAdapter):
    """Frozen nnUNet Primus ViT as a feature source for the similarity study.

    Only the ViT encoder (`down_projection` -> `eva`) is run; the Primus segmentation
    decoder (`up_projection`) is skipped. Dense features are the eva token grid
    (B, embed_dim, g, g, g) at native token res g = input_shape // patch. Every input
    volume is resampled to `input_shape` so the token grid is fixed (sidesteps rope
    variable-size concerns and matches the pretraining input size).

    preproc (optional): {"clip_min","clip_max","mean","std"} to map the loader's
    z-scored HU back to raw HU and re-normalise the way the pretrained encoder expects.
    None -> feed the loader's z-scored input unchanged (fine for the random-init floor).
    """

    def __init__(self, weights_path=None, primus_kwargs=None, preproc=None,
                 device="cuda"):
        from dynamic_network_architectures.architectures.primus import Primus
        kw = dict(primus_kwargs or {})
        self.input_shape = tuple(kw["input_shape"])
        self.patch = tuple(kw["patch_embed_size"])
        assert all(s % p == 0 for s, p in zip(self.input_shape, self.patch)), \
            f"input_shape {self.input_shape} not divisible by patch {self.patch}"
        self.primus = Primus(**kw).to(device).eval()
        for p in self.primus.parameters():
            p.requires_grad_(False)
        if weights_path is not None:
            sd = torch.load(weights_path, map_location=device)
            sd = sd.get("model", sd) if isinstance(sd, dict) else sd
            missing, unexpected = self.primus.load_state_dict(sd, strict=False)
            if missing or unexpected:
                print(f"[primus] load_state_dict: {len(missing)} missing, "
                      f"{len(unexpected)} unexpected keys")
        self.preproc = preproc
        self.device = device
        self._embed_dim = self.primus.embed_dim
        self._g = self.input_shape[-1] // self.patch[-1]
        # Native-encode cache: the study re-requests the SAME volume at several resolutions
        # (features() encodes to the native grid then _down_to(res)), so cache the native
        # forward and reuse it across resolutions — halves encodes when sweeping 2 res.
        # Keyed by (storage ptr, shape); reset_cache() must be called per task since a later
        # task's tensor can reuse freed storage (same ptr, different data).
        self._native_cache = {}

    # -- interface ---------------------------------------------------------
    @property
    def R(self):
        return self._g                              # native token grid res

    def tiers(self):
        return ["backbone"]

    def native_res(self, tier, input_res):
        return self._g                              # fixed: inputs resampled to input_shape

    # -- preprocessing + encode -------------------------------------------
    def _preprocess(self, volumes):
        """(B,1,D,H,W) loader z-scored HU -> resampled, encoder-normalised (B,1,*input_shape)."""
        v = volumes.float()
        if self.preproc is not None:
            hu = v * CT_STD + CT_MEAN               # undo loader z-score -> ~HU
            hu = hu.clamp(self.preproc["clip_min"], self.preproc["clip_max"])
            v = (hu - self.preproc["mean"]) / self.preproc["std"]
        if tuple(v.shape[-3:]) != self.input_shape:
            v = F.interpolate(v, size=self.input_shape, mode="trilinear",
                              align_corners=False)
        return v

    def _encode(self, x):
        """Run the Primus ViT encoder only -> (B, embed_dim, g, g, g). x already preprocessed."""
        p = self.primus
        x = p.down_projection(x)                    # (B,C,W,H,D)
        B, C, W, H, D = x.shape
        x = x.flatten(2).transpose(1, 2)            # (B,N,C)
        if p.register_tokens is not None:
            x = torch.cat([p.register_tokens.expand(B, -1, -1), x], dim=1)
        x, keep = p.eva(x)                           # keep is None when patch_drop_rate=0
        assert keep is None, "patch dropping must be off for dense features"
        if p.register_tokens is not None:
            x = x[:, p.register_tokens.shape[1]:]
        return x.transpose(1, 2).reshape(B, self._embed_dim, W, H, D)

    def _autocast(self):
        return torch.autocast("cuda", dtype=torch.bfloat16) if self.device == "cuda" \
            else torch.autocast("cpu", dtype=torch.bfloat16)

    def reset_cache(self):
        """Drop cached native encodes. Call once per task (input tensors change)."""
        self._native_cache.clear()

    @torch.no_grad()
    def _encode_native(self, volumes):
        """Preprocess + ViT encode to the native grid (B, embed_dim, g, g, g), cached by
        (storage ptr, shape) so repeated calls on the same volume (different res) hit once."""
        key = (volumes.untyped_storage().data_ptr(), tuple(volumes.shape))
        cached = self._native_cache.get(key)
        if cached is not None:
            return cached
        x = self._preprocess(volumes.to(self.device))
        with self._autocast():
            f = self._encode(x).float()
        self._native_cache[key] = f
        return f

    @torch.no_grad()
    def features(self, volumes, tier, res):
        assert tier == "backbone", f"unknown tier {tier!r}"
        return _down_to(self._encode_native(volumes), res)   # (B, embed_dim, res, res, res)

    @torch.no_grad()
    def sample_features(self, volumes, tier, coords):
        """coords (B,N,3) normalized in (z,y,x) order -> (B,N,C)."""
        assert tier == "backbone", f"unknown tier {tier!r}"
        f = self._encode_native(volumes)
        xyz = coords.to(self.device).flip(-1).view(coords.shape[0], coords.shape[1], 1, 1, 3)
        s = F.grid_sample(f, xyz, mode="bilinear", align_corners=True)   # (B,C,N,1,1)
        return s.squeeze(-1).squeeze(-1).transpose(1, 2)                 # (B,N,C)

    # -- cost probe hook ---------------------------------------------------
    def cost_target(self, input_res):
        """(module, example_inputs) for the encode-cost probe: the ViT encoder forward
        on one preprocessed volume at input_shape (native token grid)."""
        adapter = self

        class _EncodeFwd(nn.Module):
            def forward(self, x):
                return adapter._encode(x)

        x = torch.zeros(1, 1, *self.input_shape, device=self.device)
        return _EncodeFwd().to(self.device), (x,)
