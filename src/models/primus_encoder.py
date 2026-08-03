"""Frozen (or trainable) nnUNet Primus ViT as a PatchSet3D image encoder.

PatchSet3D embeds context masks separately, so its encoder only ever sees the
image (1 channel). This wraps the Primus ViT encoder (down_projection + eva, no
segmentation decoder) to the same contract as ConvEncoder3D:
    forward(B,1,D,H,W) -> (B, out_ch, R, R, R), with .out_ch and .resolution.
Weights + arch + HU preprocessing come from the CoLiPri extraction sidecar.
"""
import json
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

# _down_to is defined at module top of patchset3d (before the class), so this import
# resolves even though patchset3d imports PrimusEncoder lazily inside its __init__.
from src.models.patchset3d import _down_to
from src.totalseg_dataset import CT_MEAN, CT_STD


def _native_target_shape(shape, patch):
    """Round each spatial dim to the nearest positive multiple of `patch`.

    In native-grid mode the ViT token grid is input/patch, so the input must be
    divisible by `patch`. Divisible inputs (e.g. 128, 192) pass through unchanged.
    """
    out = []
    for s in shape:
        m = max(1, round(s / patch)) * patch
        out.append(int(m))
    return tuple(out)


def _set_rope_identity_grid(rope, grid):
    """Rebuild a timm RoPE table for `grid` with identity frequencies (ref == feat).

    Identity keeps adjacent tokens exactly 1 apart — the local rotary frequency the
    encoder trained on — so a smaller grid is a sub-block of the training positional
    field (no fractional/stretched positions). update_feat_shape is a no-op when the
    grid is unchanged, so this is cheap to call every forward.
    """
    grid = list(grid)
    if list(rope.feat_shape) == grid and list(rope.ref_feat_shape or []) == grid:
        return
    rope.ref_feat_shape = grid
    rope.update_feat_shape(grid)


class _EncodeCache:
    """LRU, CPU-backed store of encoder outputs, keyed by an input fingerprint.

    Features live on CPU (a cached 864×R³ tensor is ~14 MB; keeping thousands on
    the GPU would blow VRAM shared with training). A cache hit costs one small
    CPU→GPU copy, trivial next to a ViT encode. Sized to hold the whole eval set
    so a frozen encoder pays each distinct crop once — then every later epoch's
    val is a head-only pass.
    """

    def __init__(self, max_entries: int):
        self.max_entries = int(max_entries)
        self._d: "OrderedDict" = OrderedDict()

    def get(self, key):
        t = self._d.get(key)
        if t is not None:
            self._d.move_to_end(key)   # mark most-recently-used
        return t

    def put(self, key, tensor):
        self._d[key] = tensor.detach().to("cpu")
        self._d.move_to_end(key)
        while len(self._d) > self.max_entries:
            self._d.popitem(last=False)   # evict least-recently-used

    def clear(self):
        self._d.clear()

    def __len__(self):
        return len(self._d)


def _cached_encode(encode_fn, x, key_fn, cache: _EncodeCache):
    """Encode each distinct row of x once via encode_fn, reusing `cache`.

    Rows whose key is absent are batched through encode_fn together (one call);
    every row's feature is then read back from the cache and stacked in input
    order, on x's device. Persistent across calls — a later call with the same
    inputs re-encodes nothing.
    """
    keys = [key_fn(x[i]) for i in range(x.shape[0])]
    # Unique missing keys only — dedupe repeats within this batch too, keeping one
    # representative row index per key so a duplicated volume is encoded just once.
    miss: "OrderedDict" = OrderedDict()
    for i, k in enumerate(keys):
        if k not in miss and cache.get(k) is None:
            miss[k] = i
    if miss:
        rows = list(miss.values())
        feats = encode_fn(x[rows])
        for j, k in enumerate(miss):
            cache.put(k, feats[j])
    return torch.stack([cache.get(k).to(x.device) for k in keys], dim=0)


class PrimusEncoder(nn.Module):
    def __init__(self, sidecar_path, resolution, frozen=True, device="cuda",
                 cache_max=4096, encoder_stage=None, native_grid=False):
        super().__init__()
        from dynamic_network_architectures.architectures.primus import Primus
        with open(sidecar_path) as f:
            meta = json.load(f)
        kw = dict(meta["primus_kwargs"])
        self.input_shape = tuple(kw["input_shape"])
        self.patch_size = int(kw["patch_embed_size"][0])
        self.native_grid = bool(native_grid)
        self._warned_resize = False
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
        # Early-exit truncation: keep only the first `encoder_stage` EVA blocks so
        # later blocks are never built into the graph (Eva.forward_features just
        # iterates self.blocks then applies the final norm). Load full weights first
        # so load_state_dict matches, then drop the tail — freeing its params/VRAM.
        self.encoder_stage = self._truncate_blocks(encoder_stage)
        if self.frozen:
            for p in self.primus.parameters():
                p.requires_grad_(False)
        self.primus.to(device)
        # Eval-only encode cache. Active only when frozen AND the module is in eval
        # mode (self.training is False) — where the loader is deterministic and no aug
        # runs, so a given input crop's features are reusable. Because the encoder is
        # frozen, its output for a fixed eval crop is invariant across epochs, so the
        # cache persists across val calls: the first val encodes each distinct crop
        # once, every later val is a head-only pass. CPU-backed + LRU so it can hold the
        # whole eval set without eating VRAM. Never used in training (aug makes each
        # volume unique) or when trainable (grad required). Keyed on a per-row
        # fingerprint. Size the eval set (via eval.n_subjects) to fit cache_max.
        self._cache = _EncodeCache(int(cache_max))

    def _truncate_blocks(self, encoder_stage):
        """Keep only the first `encoder_stage` EVA blocks (early-exit tap).

        None/<=0 or >= depth means no truncation (full encoder). Returns the
        effective stage kept. The final Eva `norm` still runs on the stage-k
        output — the standard normed hidden state at that layer.
        """
        blocks = self.primus.eva.blocks
        depth = len(blocks)
        if encoder_stage is None:
            return depth
        k = int(encoder_stage)
        if k <= 0 or k >= depth:
            return depth
        self.primus.eva.blocks = nn.ModuleList(list(blocks)[:k])  # drop tail → frees VRAM
        print(f"[PrimusEncoder] truncated eva to stage {k}/{depth} "
              f"(dropped {depth - k} blocks; ~{k/depth:.0%} of encoder compute)")
        return k

    def reset_cache(self):
        self._cache.clear()

    @staticmethod
    def _key(xi):
        """Cheap collision-resistant fingerprint of one input row (1,D,H,W)."""
        flat = xi.reshape(-1)
        n = flat.numel()
        k = min(n, 512)
        idx = torch.linspace(0, n - 1, steps=k, device=flat.device).long()
        sig = torch.round(flat[idx] * 1000).to(torch.int64).tolist()
        return (tuple(xi.shape), round(float(flat.sum()), 3), hash(tuple(sig)))

    def _preprocess(self, x):
        """(B,1,D,H,W) loader z-scored HU -> resampled to input_shape, encoder-normalised."""
        v = x.float()
        if self.preproc is not None:
            hu = v * CT_STD + CT_MEAN
            hu = hu.clamp(self.preproc["clip_min"], self.preproc["clip_max"])
            v = (hu - self.preproc["mean"]) / self.preproc["std"]
        target = (_native_target_shape(tuple(v.shape[-3:]), self.patch_size)
                  if self.native_grid else self.input_shape)
        if tuple(v.shape[-3:]) != target:
            if self.native_grid and not self._warned_resize:
                print(f"[PrimusEncoder] native_grid: input {tuple(v.shape[-3:])} not a "
                      f"multiple of patch {self.patch_size}; resampling to {target}")
                self._warned_resize = True
            v = F.interpolate(v, size=target, mode="trilinear", align_corners=False)
        return v

    def _encode(self, x):
        """Primus ViT encoder only (down_projection + eva) -> (B, out_ch, g, g, g)."""
        p = self.primus
        x = p.down_projection(x)
        B, C, W, H, D = x.shape
        if self.native_grid:
            _set_rope_identity_grid(p.eva.rope, (W, H, D))
        x = x.flatten(2).transpose(1, 2)
        if p.register_tokens is not None:
            x = torch.cat([p.register_tokens.expand(B, -1, -1), x], dim=1)
        x, keep = p.eva(x)
        assert keep is None, "patch dropping must be off for dense features"
        if p.register_tokens is not None:
            x = x[:, p.register_tokens.shape[1]:]
        return x.transpose(1, 2).reshape(B, self.out_ch, W, H, D)

    def _encode_batch(self, x):
        """(B,1,D,H,W) -> (B,out_ch,R,R,R), grad only when trainable."""
        v = self._preprocess(x)
        if self.frozen:
            with torch.no_grad():
                f = self._encode(v)
        else:
            f = self._encode(v)
        return _down_to(f.float(), self.resolution)

    def forward(self, x):
        dev = next(self.primus.parameters()).device
        x = x.to(dev)
        # Cache only in frozen eval mode; train / trainable paths compute directly.
        if not (self.frozen and not self.training):
            return self._encode_batch(x)
        return _cached_encode(self._encode_batch, x, self._key, self._cache)
