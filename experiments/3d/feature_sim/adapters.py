# experiments/3d/feature_sim/adapters.py
"""Encoder-agnostic feature adapters for the similarity study.

EncoderAdapter maps volumes -> per-cell feature grids at an arbitrary resolution
(dense) or trilinearly-sampled point features (native res). PatchSet3DEncoderAdapter
wraps a loaded PatchSet3D; PrimusEncoderAdapter wraps a frozen nnUNet Primus ViT
(weights-pluggable — e.g. a CoLiPri backbone). Future SAM/DINO adapters implement the
same interface."""
import contextlib
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
        # A Primus encoder emits a single native token grid (no .stem/.stages), so the
        # multi-scale stage/concat tiers don't apply. For a primus-encoder checkpoint we
        # expose 'backbone' (the raw encoder map, == the standalone PrimusEncoderAdapter's
        # features since the encoder is frozen and self-preprocesses) and 'img_embed' (the
        # trainable projection of it); the transformer tiers work unchanged (they hook the
        # full forward, which runs the primus encoder internally).
        self._is_primus = not hasattr(self.enc, "stages")

    @property
    def R(self):
        return self.model.resolution

    @property
    def n_stages(self):
        return len(self.enc.stages)               # excludes the stem

    def tiers(self):
        if self._is_primus:
            return ["backbone", "img_embed"]
        stages = [f"stage:{i}" for i in range(self.n_stages + 1)]
        return stages + ["concat", "img_embed"]

    def native_res(self, tier, input_res):
        if self._is_primus:
            if tier in ("backbone", "img_embed"):
                return self.R                      # encoder emits R^3 regardless of input_res
            raise ValueError(f"unknown primus tier {tier!r}")
        if tier.startswith("stage:"):
            return input_res >> int(tier.split(":")[1])
        if tier in ("concat", "img_embed"):
            return input_res                       # stem-limited, finest genuine
        raise ValueError(f"unknown tier {tier!r}")

    def _apply_img_embed(self, f):
        """(B,C,r,r,r) raw features -> (B,e,r,r,r) via the model's trainable img_embed
        projection (isolated: no per-task norm / pos, matching the tier's intent)."""
        B, r = f.shape[0], f.shape[-1]
        flat = f.flatten(2).transpose(1, 2)         # (B, r^3, C)
        emb = self.model.img_embed(flat)            # (B, r^3, e)
        return emb.transpose(1, 2).reshape(B, emb.shape[-1], r, r, r)

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
    def features(self, volumes, tier, res, spacing=None):
        if self._is_primus:
            if tier not in ("backbone", "img_embed"):
                raise ValueError(f"unknown primus tier {tier!r}")
            # spacing scales the frozen ViT's RoPE in spacing-aware mode; threaded so the
            # study encodes at the crop's physical spacing (matches evaluate.py), not the
            # train-pitch fallback. Ignored by the encoder when not spacing-aware.
            f = _down_to(self.enc(volumes, spacing=spacing), res)   # (B,out_ch,res^3); enc self-caches
            return self._apply_img_embed(f) if tier == "img_embed" else f
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
            f = self._apply_img_embed(f)
        return f

    @torch.no_grad()
    def sample_features(self, volumes, tier, coords, spacing=None):
        """coords (B,N,3) normalized in (z,y,x)=(d,h,w) order -> (B,N,C)."""
        if self._is_primus:
            if tier not in ("backbone", "img_embed"):
                raise ValueError(f"unknown primus tier {tier!r}")
            f = self.enc(volumes, spacing=spacing)  # (B, out_ch, R,R,R); enc self-caches
        else:
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
        features() drives, so cost compares like-for-like against other encoders.

        The primus branch traces `_encode` (down_projection + eva) on an EAGERLY preprocessed
        input, avoiding the cached `enc(x)` forward: (1) the frozen-eval cache path hashes the
        input via `_key` (`round(float(x.sum()))`) — untraceable and it would make the repeated-
        input timing loop measure cache hits, not the real encode; (2) `_preprocess`'s
        native_grid target uses `round(shape/patch)`, which fvcore's jit trace turns into
        `round(Tensor)` (TypeError -> encode_gflops stayed None). Preprocessing eagerly gives
        concrete shapes, so only the tensor-op encode is traced."""
        adapter, is_primus = self, self._is_primus
        dev = next(self.enc.parameters()).device

        class _EncodeFwd(nn.Module):
            def forward(self, x):
                return adapter.enc._encode(x) if is_primus else adapter._stage_feats(x)

        x = torch.zeros(1, 1, input_res, input_res, input_res, device=dev)
        if is_primus:
            x = adapter.enc._preprocess(x)      # eager: strips the untraceable round()/resample
        return _EncodeFwd().to(dev), (x,)

    @torch.no_grad()
    def transformer_query(self, image, context_in, context_out, spacing=None):
        """Post-transformer query rep (B,N,e) via a decoder-input hook (res=R only)."""
        captured = {}
        # fine_decode builds no .decoder — filter_head is then the first module to see q.
        head = getattr(self.model, "decoder", None) or self.model.filter_head
        h = head.register_forward_pre_hook(
            lambda mod, args: captured.setdefault("q", args[0]))
        try:
            self.model(image, context_in=context_in, context_out=context_out,
                       mode="train", spacing=spacing)
        finally:
            h.remove()
        return captured["q"]

    @torch.no_grad()
    def transformer_pair(self, image, context_in, context_out, spacing=None):
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
            self.model(image, context_in=context_in, context_out=context_out,
                       mode="train", spacing=spacing)
        finally:
            h.remove()
        x, sep_t = cap["x"], cap["sep_t"]
        n = self.model.thinking.n
        return x[:, sep_t:, 0, :], x[:, n:sep_t, 0, :]

    @torch.no_grad()
    def transformer_pair_per_layer(self, image, context_in, context_out, spacing=None):
        """Like transformer_pair but returns the (target, context) img-token pair after
        EACH transformer block, so correspondence can be traced layer by layer. Free: the
        forward already runs every block; the hooks only capture its output tensor.

        Token layout + sep_t are invariant across blocks, so the same slicing applies at
        every depth. Returns a list of (target (B,N,e), context (B,K*N,e)), one per block."""
        outs = []
        hs = [b.register_forward_hook(lambda m, a, o: outs.append((o, a[1])))
              for b in self.model.transformer.blocks]
        try:
            self.model(image, context_in=context_in, context_out=context_out,
                       mode="train", spacing=spacing)
        finally:
            for h in hs:
                h.remove()
        n = self.model.thinking.n
        return [(x[:, s:, 0, :], x[:, n:s, 0, :]) for x, s in outs]

    @torch.no_grad()
    def transformer_trace(self, image, context_in, context_out, spacing=None):
        """One forward -> named (target, context) img-token pairs: 'transformer_input' (the
        transformer INPUT — the img token AFTER the trainable img_embed + pos, before any
        attention; NOT the frozen encoder output) then 'L{i}' after each block, all at res=R
        on the same token grid. Lets a training run trace how correspondence evolves from the
        input embedding through the transformer stack. Free: the forward already runs every
        block; hooks only capture tensors. Returns [(name, tgt, ctx), ...]."""
        outs = []
        hp = self.model.transformer.register_forward_pre_hook(
            lambda m, a: outs.append(("transformer_input", a[0], a[1])))   # (x_in, sep_t)
        hs = [b.register_forward_hook(
                  lambda m, a, o, i=i: outs.append((f"L{i}", o, a[1])))
              for i, b in enumerate(self.model.transformer.blocks)]
        try:
            self.model(image, context_in=context_in, context_out=context_out,
                       mode="train", spacing=spacing)
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
                 device="cuda", autocast=True):
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
        # Honour eval.autocast (run.py): True = bf16 encode (fast, default, matches train/eval);
        # False = full fp32 encode (exact reference / reproducibility). Metrics are fp32 either
        # way (_encode_native .float()s the output; _metric_row re-disables autocast).
        self.autocast = bool(autocast)
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

    @property
    def n_layers(self):
        eva = getattr(self.primus.eva, "_orig_mod", self.primus.eva)
        return len(eva.blocks)                       # eva depth (may be truncated)

    def tiers(self):
        # 'backbone' = final eva output (post final-norm). 'backbone_layers' is a fan-out
        # meta-tier: run.py emits one row per eva block (tier 'bb:L{i}'), so correspondence
        # can be traced along transformer depth (the ViT analogue of the conv stage:* sweep).
        return ["backbone", "backbone_layers"]

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

    def _encode_layers(self, x):
        """Run the Primus ViT encoder capturing the token grid AFTER each eva block.
        Returns a list (len n_layers) of (B, embed_dim, g, g, g). x already preprocessed.

        Reimplements eva.forward_features' block loop (rather than forward-hooking) so it is
        robust when eva has been torch.compile-wrapped (hooks on a compiled graph's submodules
        don't fire reliably) — the underlying module is unwrapped via `_orig_mod`. Grids are
        post-block, PRE the final eva `norm` (standard intermediate-layer features); the last
        grid therefore differs slightly from the `backbone` tier (which includes final norm)."""
        p = self.primus
        x = p.down_projection(x)                    # (B,C,W,H,D)
        B, C, W, H, D = x.shape
        x = x.flatten(2).transpose(1, 2)            # (B,N,C)
        n_reg = 0
        if p.register_tokens is not None:
            x = torch.cat([p.register_tokens.expand(B, -1, -1), x], dim=1)
            n_reg = p.register_tokens.shape[1]
        eva = getattr(p.eva, "_orig_mod", p.eva)     # unwrap torch.compile if present
        x, rope, keep = eva._pos_embed(x)
        assert keep is None, "patch dropping must be off for dense features"
        grids = []
        for blk in eva.blocks:
            x = blk(x, rope=rope)
            t = x[:, n_reg:] if n_reg else x         # drop register/prefix tokens
            grids.append(t.transpose(1, 2).reshape(B, self._embed_dim, W, H, D).float())
        return grids

    def _autocast(self):
        if not self.autocast:
            return contextlib.nullcontext()        # fp32 encode (eval.autocast=false)
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

    # -- per-layer (backbone_layers tier) ---------------------------------
    @torch.no_grad()
    def _encode_native_layers(self, volumes):
        """Preprocess + ViT encode, keeping every eva block's native grid. Cached by
        (storage ptr, shape) so the target/context volumes are encoded once even when the
        sweep asks for several resolutions."""
        key = ("layers", volumes.untyped_storage().data_ptr(), tuple(volumes.shape))
        cached = self._native_cache.get(key)
        if cached is not None:
            return cached
        x = self._preprocess(volumes.to(self.device))
        with self._autocast():
            grids = self._encode_layers(x)
        self._native_cache[key] = grids
        return grids

    @torch.no_grad()
    def features_per_layer(self, volumes, res):
        """List (len n_layers) of (B, embed_dim, res, res, res) — one entry per eva block."""
        return [_down_to(g, res) for g in self._encode_native_layers(volumes)]

    @torch.no_grad()
    def sample_features_per_layer(self, volumes, coords):
        """coords (B,N,3) in (z,y,x) -> list (len n_layers) of (B,N,embed_dim)."""
        xyz = coords.to(self.device).flip(-1).view(coords.shape[0], coords.shape[1], 1, 1, 3)
        out = []
        for g in self._encode_native_layers(volumes):
            s = F.grid_sample(g, xyz, mode="bilinear", align_corners=True)   # (B,C,N,1,1)
            out.append(s.squeeze(-1).squeeze(-1).transpose(1, 2))            # (B,N,C)
        return out

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


# ---------------------------------------------------------------------------
# Frozen fomofo/tap-ct-b-3d 3D ViT adapter (weights fixed on HF).
# Native token grid is ANISOTROPIC (T/4, T/8, T/8) from patch (4,8,8); we encode
# once at the native grid, inverse-reorient it back to the loader's RAS frame so it
# aligns with grid_labels, cache it, and _down_to(res) for the sweep. See
# experiments/encoders/tapct_features.py for the dataloader->TAP bridge.
# ---------------------------------------------------------------------------


class TapCTEncoderAdapter(EncoderAdapter):
    """Frozen tap-ct-b-3d ViT as a feature source for the similarity study.

    tiers=['backbone']. Preprocessing bridges the loader's z-scored-HU / RAS tensor to
    TAP's raw-HU / LPS input (de-norm, reorient, TAP processor). The LPS-frame token grid
    is inverse-reoriented back to RAS so it aligns with grid_labels (which pools the mask
    in the loader frame). Encoded once at native res, cached (storage ptr, shape), then
    _down_to(res) — "compute at native anisotropic, resample to res^3".
    """

    def __init__(self, precision="bf16", to_lps=True, resize_native=True,
                 pad_hu=None, image_size=224, max_layers=None, device="cuda"):
        import sys
        import pathlib
        enc_dir = pathlib.Path(__file__).resolve().parents[2] / "encoders"
        if str(enc_dir) not in sys.path:
            sys.path.insert(0, str(enc_dir))
        from tapct_features import (load_model, make_processor, dense_features,
                                     item_to_tap_input)
        self._dense_features = dense_features
        self._item_to_tap = item_to_tap_input
        T = int(image_size)
        assert T % 8 == 0, f"tap_ct needs image_size divisible by 8, got {T}"
        self.T = T
        self.device = device
        self.precision = precision
        self.to_lps = bool(to_lps)
        self.pad_hu = pad_hu
        self.model = load_model(torch.device(device), use_sdpa=True)
        self.proc = make_processor(T)
        if not resize_native:
            self.proc.resize_dims = (224, 224)      # stock in-plane upsample
        self._native_cache = {}
        # Truncate the transformer to the first `max_layers` blocks to cut compute (the block
        # loop always runs every block, so `n` alone doesn't save FLOPs -> physically drop the
        # tail + update n_blocks, else get_intermediate_layers' count assert fails). Mid-stack
        # (~7/12) often gives the best correspondence AND ~40% less compute (see docs/logs.md).
        vit = getattr(self.model, "model", self.model)
        total = getattr(vit, "n_blocks", None) or len(vit.blocks)
        if max_layers is not None and 0 < int(max_layers) < total:
            if getattr(vit, "chunked_blocks", False):
                print("  [tap_ct] chunked_blocks=True; skipping max_layers truncation")
            else:
                k = int(max_layers)
                vit.blocks = nn.ModuleList(list(vit.blocks)[:k])
                vit.n_blocks = k
                print(f"  [tap_ct] truncated to first {k}/{total} transformer blocks")
        # transformer depth (for the backbone_layers tier); robust to attribute path.
        self.n_blocks = getattr(vit, "n_blocks", None) or len(vit.blocks)

    _DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}

    @property
    def R(self):
        return self.T // 8                          # in-plane native grid res (coarse axis)

    @property
    def n_layers(self):
        return self.n_blocks

    def tiers(self):
        return ["backbone", "backbone_layers"]

    def native_res(self, tier, input_res):
        assert tier in ("backbone", "backbone_layers"), f"unknown tap_ct tier {tier!r}"
        return self.T // 8

    def reset_cache(self):
        self._native_cache.clear()

    def _inv_reorient(self, g):
        """Inverse of tapct_features.ras_to_lps_axial_first on a (C, gS,gP,gL) grid:
        flip the (P,L) grid axes then transpose(2,1,0) -> (C, gR,gA,gS) RAS order, so the
        grid aligns with grid_labels' RAS-frame mask pooling."""
        return g.flip(2, 3).permute(0, 3, 2, 1).contiguous()

    def _tokens_to_grid(self, tokens):
        """(1, N, C) patch tokens (row-major gS,gP,gL) -> (C, gR,gA,gS) native grid in RAS
        frame. In-plane grid = resize_dims/8 (patch 8); depth = N/(gh*gw) (patch 4)."""
        rh, rw = self.proc.resize_dims
        gh, gw = rh // 8, rw // 8
        n, c = tokens.shape[1], tokens.shape[2]
        g = tokens[0].float().reshape(n // (gh * gw), gh, gw, c).permute(3, 0, 1, 2)
        return self._inv_reorient(g) if self.to_lps else g

    def _to_res(self, f, res):
        """Resample an anisotropic native grid (B,C,·,·,·) to res^3 (cell-centered, matching
        grid_labels' RAS pooling). _down_to can't be used — it keys off shape[-1] only."""
        if tuple(f.shape[-3:]) == (res, res, res):
            return f
        return F.interpolate(f, size=(res, res, res), mode="trilinear", align_corners=False)

    @torch.no_grad()
    def _encode_native(self, volumes):
        """(B,1,D,H,W) loader tensor -> (B, C, gR,gA,gS) native grid in RAS frame, cached.

        Key on the ORIGINAL caller-retained tensor's storage ptr (not a `.to(device)`
        temporary, whose storage is freed and reused across target/context calls -> cache
        collisions returning the wrong volume's features). dense_features moves data to the
        device internally, so volumes may stay on CPU here.
        """
        key = (volumes.untyped_storage().data_ptr(), tuple(volumes.shape))
        cached = self._native_cache.get(key)
        if cached is not None:
            return cached
        dev = torch.device(self.device)
        grids = []
        for b in range(volumes.shape[0]):
            rows, gd = self._dense_features(self.model, self.proc, volumes[b], dev,
                                            to_lps=self.to_lps, precision=self.precision)
            g = rows.reshape(*gd, -1).permute(3, 0, 1, 2)      # (C, g0,g1,g2)
            if self.to_lps:
                g = self._inv_reorient(g)                      # LPS grid -> RAS order
            grids.append(g.float())
        # dense_features/embed return CPU tensors; keep the grid on-device so the downstream
        # cosine metrics run on GPU (else run.py's TAP rows fall back to slow CPU matmuls).
        f = torch.stack(grids, 0).to(dev)                      # (B, C, ·, ·, ·)
        self._native_cache[key] = f
        return f

    @torch.no_grad()
    def features(self, volumes, tier, res):
        assert tier == "backbone", f"unknown tap_ct tier {tier!r}"
        return self._to_res(self._encode_native(volumes), res)

    def _grid_sample(self, f, coords):
        xyz = coords.to(self.device).flip(-1).view(coords.shape[0], coords.shape[1], 1, 1, 3)
        s = F.grid_sample(f, xyz, mode="bilinear", align_corners=True)   # (B,C,N,1,1)
        return s.squeeze(-1).squeeze(-1).transpose(1, 2)                 # (B,N,C)

    @torch.no_grad()
    def sample_features(self, volumes, tier, coords):
        """coords (B,N,3) normalized in (z,y,x)=(d,h,w) RAS order -> (B,N,C)."""
        assert tier == "backbone", f"unknown tap_ct tier {tier!r}"
        return self._grid_sample(self._encode_native(volumes), coords)

    # -- per-layer (backbone_layers tier) ---------------------------------
    @torch.no_grad()
    def _encode_native_layers(self, volumes):
        """(B,1,D,H,W) -> list (len n_blocks) of (B, C, gR,gA,gS) native grids in RAS frame,
        one per transformer block, cached. Uses model(output_hidden_states=True) so a single
        forward yields every block's token grid."""
        key = ("layers", volumes.untyped_storage().data_ptr(), tuple(volumes.shape))
        cached = self._native_cache.get(key)
        if cached is not None:
            return cached
        dev = torch.device(self.device)
        prec = self.precision
        actx = (contextlib.nullcontext() if prec == "fp32"
                else torch.autocast("cuda", dtype=self._DTYPES[prec]))
        per_vol = []                                       # per_vol[b] = list over layers
        for b in range(volumes.shape[0]):
            pix = self._item_to_tap(volumes[b], self.proc, to_lps=self.to_lps,
                                    pad_hu=self.pad_hu).to(dev)
            with actx:
                hs = self.model(pix, output_hidden_states=True).hidden_states  # tuple (1,N,C)
            per_vol.append([self._tokens_to_grid(h) for h in hs])
        n_l = len(per_vol[0])
        grids = [torch.stack([per_vol[b][l] for b in range(len(per_vol))], 0).to(dev)
                 for l in range(n_l)]                      # list (n_l) of (B,C,·,·,·)
        self._native_cache[key] = grids
        return grids

    @torch.no_grad()
    def features_per_layer(self, volumes, res):
        """List (len n_blocks) of (B, C, res, res, res) — one entry per transformer block."""
        return [self._to_res(g, res) for g in self._encode_native_layers(volumes)]

    @torch.no_grad()
    def sample_features_per_layer(self, volumes, coords):
        """coords (B,N,3) in (z,y,x) RAS order -> list (len n_blocks) of (B,N,C)."""
        return [self._grid_sample(g, coords) for g in self._encode_native_layers(volumes)]

    def cost_target(self, input_res):
        """Encode-cost probe: the TAP forward on a native-size pixel_values built from a zero
        (de-norms to a constant HU) volume. FLOPs via count_encode_flops (fvcore can't trace
        SDPA); timing/VRAM measured by the caller."""
        from tapct_features import item_to_tap_input
        adapter = self
        zero = torch.zeros(1, self.T, self.T, self.T)
        pix = item_to_tap_input(zero, self.proc, to_lps=self.to_lps,
                                pad_hu=self.pad_hu).to(self.device)

        class _EncodeFwd(nn.Module):
            def forward(self, x):
                return adapter.model(x).last_hidden_state

        return _EncodeFwd().to(self.device), (pix,)

    def count_encode_flops(self, module, inputs):
        """GFLOPs of one encode via torch FlopCounterMode, which counts SDPA/flash kernels
        that fvcore can't trace (and doesn't flood stdout). Counted on eager fp32 — FLOPs are
        precision/compile-independent."""
        try:
            from torch.utils.flop_counter import FlopCounterMode
            fc = FlopCounterMode(display=False)
            with torch.no_grad(), fc:
                module(*inputs)
            return fc.get_total_flops() / 1e9
        except Exception as e:                             # honest None on failure
            print(f"  [tap_ct] FLOP count failed: {type(e).__name__}: {e}")
            return None
