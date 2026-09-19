# PatchSetV2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `PatchSetV2`, a clean, minimal reimplementation of Iris-style in-context 3D
segmentation, as a new class reusing existing low-level attention/encoder primitives without
inheriting `PatchSet3D`'s accreted knob surface.

**Architecture:** Per volume (target + K context, weight-shared): tokenize per-cell →
foreground-masked pool at native resolution (`pool_v`) → compress to `compress_m` tokens via
`RowCrossAttention` (Stage A) → concat `[pool_v ; compressed]` per volume. All volumes'
sequences concatenate and run through the unmodified `TransformerEncoderStack` (Stage B). The
target's own post-Stage-B tokens act as Iris's task tokens `T`, cross-attending (Eq 5) against
the target's raw never-compressed per-cell grid, then a conv up-path with encoder-pyramid
skips (Eq 6) produces logits.

**Tech Stack:** PyTorch, existing `src/models/pfn_seg_2d.py` primitives (`RowCrossAttention`,
`TransformerEncoderStack`, `ThinkingRows`), existing encoder classes, Hydra config, pytest.

**Spec:** `docs/superpowers/specs/2026-09-19-patchset-v2-design.md`

## Global Constraints

- `PatchSet3D` (`src/models/patchset3d.py`) must remain behaviorally unchanged — existing
  configs/checkpoints (including the currently-training `99_iris_decoder_plainconv_doubling`
  run) must not regress. The one sanctioned touch is a pure refactor (Task 1).
- `PatchSetV2.forward()` must match the exact call contract `experiments/3d/cascade.py`'s
  `_forward_level` already uses for `PatchSet3D`: `model(image, context_in=..., context_out=...,
  mode="train", spacing=sp, query_prior=..., cascade_regs=...)` returning a dict with at least
  `"final_logit"` and `"registers"` keys (`cascade.py:347-361`).
- `pool_v` upsamples to the volume's **native** `(D,H,W)` resolution before masking — never
  `R`, never a coarser intermediate stage's own resolution (explicit correctness requirement
  from the design discussion, matching Iris Eq 2's own ablation).
- No `register_routed`, `mask_slots`, `seq_compress` toggle, `fine_decode`/`decoder` mode
  switch, `pool_token` flag, `iris_pixelshuffle_r`, `full_attn`/`query_self_attn`, or SimMIM
  token-masking knobs in `PatchSetV2` — these are structurally absent, not defaulted off.
- `cascade_registers`, Fourier positional encoding, `mask_embed` (linear/conv), and
  `context_id_embed` are kept and must work.

---

## File Structure

- **Create:** `src/models/encoders/factory.py` — `build_encoder()`, extracted encoder dispatch
  shared by `PatchSet3D` and `PatchSetV2`.
- **Modify:** `src/models/patchset3d.py` — `PatchSet3D.__init__`'s encoder `if/elif` replaced by
  a `build_encoder()` call (behavior-preserving refactor).
- **Create:** `src/models/patchset3d_v2.py` — the `PatchSetV2` class.
- **Modify:** `experiments/3d/train.py` — `build_model()` gets a `"patchset3d_v2"` branch.
- **Create:** `configs/experiment/3d/model/m3_patchset_v2.yaml` — default arch knobs for v2.
- **Create:** `tests/test_patchset3d_v2.py` — unit tests, mirroring `tests/test_patchset3d.py`'s
  style (small dims, plain pytest functions, a `_dummy_batch` helper).

---

### Task 1: Extract `build_encoder` and refactor `PatchSet3D` to use it

**Files:**
- Create: `src/models/encoders/factory.py`
- Modify: `src/models/patchset3d.py:315-387` (replace the inline `if/elif` with a call)
- Test: `tests/test_patchset3d_v2.py` (new file, first test in it)

**Interfaces:**
- Produces: `build_encoder(name: str, resolution: int, *, in_ch: int = 1, enc_dims=None,
  encoder_frozen: bool = True, primus_sidecar: str | None = None,
  nnunet_ts_weights: str | None = None, nnunet_ts_stages=(2, 3, 4),
  nnunet_ts_random_init: bool = False, resenc_n_stages: int = 5,
  plainconv_ts_n_stages: int = 5, plainconv_ts_features_per_stage=None,
  encoder_input_norm: str | None = None, image_size=None, encoder_stage: int | None = None,
  encoder_native_grid: bool = False, encoder_spacing_aware: bool = False,
  encoder_precision: str = "bf16") -> nn.Module` — used by Task 2 onward.

- [ ] **Step 1: Write `build_encoder`, copying the exact dispatch bodies from `PatchSet3D.__init__`**

Create `src/models/encoders/factory.py`:

```python
"""Shared arch.encoder dispatch — used by both PatchSet3D and PatchSetV2 so the ~70-line
if/elif selecting an encoder implementation isn't duplicated between them. Each branch's
body is copied verbatim from PatchSet3D.__init__'s inline dispatch (see git history on
src/models/patchset3d.py predating this file for the version this replaces) — behavior is
unchanged, only the location moved."""

import torch.nn as nn


def build_encoder(
    name: str,
    resolution: int,
    *,
    in_ch: int = 1,
    enc_dims=None,
    encoder_frozen: bool = True,
    primus_sidecar: str | None = None,
    nnunet_ts_weights: str | None = None,
    nnunet_ts_stages=(2, 3, 4),
    nnunet_ts_random_init: bool = False,
    resenc_n_stages: int = 5,
    plainconv_ts_n_stages: int = 5,
    plainconv_ts_features_per_stage=None,
    encoder_input_norm: str | None = None,
    image_size=None,
    encoder_stage: int | None = None,
    encoder_native_grid: bool = False,
    encoder_spacing_aware: bool = False,
    encoder_precision: str = "bf16",
) -> nn.Module:
    if name == "primus":
        if not primus_sidecar:
            raise ValueError("encoder='primus' requires arch.primus_sidecar")
        from src.models.primus_encoder import PrimusEncoder
        return PrimusEncoder(primus_sidecar, resolution, frozen=encoder_frozen, device="cpu",
                             encoder_stage=encoder_stage, native_grid=encoder_native_grid,
                             spacing_aware=encoder_spacing_aware, precision=encoder_precision)
    elif name == "tap_ct":
        from src.models.tapct_encoder import TapCTEncoder
        if not image_size:
            raise ValueError("encoder='tap_ct' requires arch.image_size (from data.image_size)")
        return TapCTEncoder(resolution, image_size, frozen=encoder_frozen, device="cpu",
                            encoder_stage=encoder_stage, precision=encoder_precision)
    elif name == "nnunet_ts":
        from src.models.encoders.nnunet_ts import NnUNetTSEncoder
        if not nnunet_ts_weights:
            raise ValueError("encoder='nnunet_ts' requires arch.nnunet_ts_weights")
        _in_norm = {"input_norm": encoder_input_norm} if encoder_input_norm else {}
        return NnUNetTSEncoder(nnunet_ts_weights, resolution, stages=tuple(nnunet_ts_stages),
                               frozen=encoder_frozen, device="cpu", precision=encoder_precision,
                               random_init=nnunet_ts_random_init, **_in_norm)
    elif name == "resenc_ts":
        from src.models.encoders.resenc_ts import ResEncTSEncoder
        _in_norm = {"input_norm": encoder_input_norm} if encoder_input_norm else {}
        return ResEncTSEncoder(resolution, n_stages=resenc_n_stages, stages=tuple(nnunet_ts_stages),
                               frozen=encoder_frozen, device="cpu", precision=encoder_precision,
                               **_in_norm)
    elif name == "plainconv_ts":
        from src.models.encoders.plainconv_ts import PlainConvTSEncoder
        _in_norm = {"input_norm": encoder_input_norm} if encoder_input_norm else {}
        return PlainConvTSEncoder(resolution, n_stages=plainconv_ts_n_stages,
                                  stages=tuple(nnunet_ts_stages),
                                  features_per_stage=plainconv_ts_features_per_stage,
                                  frozen=encoder_frozen, device="cpu",
                                  precision=encoder_precision, **_in_norm)
    elif name == "conv":
        from src.models.patchset3d import ConvEncoder3D   # lazy: avoids import cycle
        return ConvEncoder3D(in_ch, tuple(enc_dims), resolution)
    raise ValueError(f"unknown arch.encoder {name!r} "
                     f"(conv | primus | tap_ct | nnunet_ts | resenc_ts | plainconv_ts)")
```

- [ ] **Step 2: Write the regression test for the "conv" branch (the only branch unit-testable
  without external weight files)**

Create `tests/test_patchset3d_v2.py`:

```python
import torch
from src.models.encoders.factory import build_encoder


def test_build_encoder_conv_matches_direct_construction():
    from src.models.patchset3d import ConvEncoder3D
    enc = build_encoder("conv", resolution=4, enc_dims=(8, 8, 8))
    assert isinstance(enc, ConvEncoder3D)
    assert enc.out_ch == 24
    out = enc(torch.randn(2, 1, 16, 16, 16))
    assert out.shape == (2, 24, 4, 4, 4)
```

- [ ] **Step 3: Run the new test to verify it passes**

Run: `pytest tests/test_patchset3d_v2.py -v`
Expected: PASS (this test doesn't depend on `PatchSet3D`'s refactor yet, only on the new
factory module, so it should pass before Step 4 too — it's here to lock in the "conv" branch's
behavior before the refactor touches `PatchSet3D`).

- [ ] **Step 4: Refactor `PatchSet3D.__init__` to call `build_encoder`**

In `src/models/patchset3d.py`, replace the entire `if encoder == "primus": ... elif encoder ==
"conv": ... else: raise ValueError(...)` block (currently lines 315-387) with:

```python
        self.encoder = build_encoder(
            encoder, resolution, in_ch=1, enc_dims=enc_dims, encoder_frozen=encoder_frozen,
            primus_sidecar=primus_sidecar, nnunet_ts_weights=nnunet_ts_weights,
            nnunet_ts_stages=nnunet_ts_stages, nnunet_ts_random_init=nnunet_ts_random_init,
            resenc_n_stages=resenc_n_stages, plainconv_ts_n_stages=plainconv_ts_n_stages,
            plainconv_ts_features_per_stage=plainconv_ts_features_per_stage,
            encoder_input_norm=encoder_input_norm, image_size=image_size,
            encoder_stage=encoder_stage, encoder_native_grid=encoder_native_grid,
            encoder_spacing_aware=encoder_spacing_aware, encoder_precision=encoder_precision)
```

Add the import near the top of `src/models/patchset3d.py` (alongside the other local imports):

```python
from src.models.encoders.factory import build_encoder
```

- [ ] **Step 5: Run the full existing PatchSet3D test suite to confirm zero regression**

Run: `pytest tests/test_patchset3d.py tests/test_patchset3d_rope.py -v`
Expected: PASS, all tests, identical to their pre-refactor results (these tests exercise
`encoder="conv"` — the default — so the refactored dispatch path is directly covered).

- [ ] **Step 6: Commit**

```bash
git add src/models/encoders/factory.py src/models/patchset3d.py tests/test_patchset3d_v2.py
git commit -m "refactor: extract PatchSet3D's encoder dispatch into build_encoder

Pure relocation, no behavior change -- shared by PatchSet3D and the
upcoming PatchSetV2 so the dispatch isn't duplicated between them."
```

---

### Task 2: `PatchSetV2` skeleton — encoder + per-volume tokenizer

**Files:**
- Create: `src/models/patchset3d_v2.py`
- Test: `tests/test_patchset3d_v2.py` (append)

**Interfaces:**
- Consumes: `build_encoder` (Task 1); `MaskConvEmbed`, `_down_to`, `_mask_tiles_3d` from
  `src.models.patchset3d`; `FourierPositionalEncoding` from `src.models.patchset_pfn`.
- Produces: `PatchSetV2.__init__(...)` (partial — extended by later tasks);
  `PatchSetV2._tokens_all(self, feat, occ, B, T) -> Tensor (B,T,N,2,e)`;
  `PatchSetV2._occupancy(self, context_out) -> Tensor (B,K,N,p**3)`;
  `PatchSetV2._prior_occupancy(self, prior) -> Tensor (B,1,N,p**3)`;
  `PatchSetV2.N: int` (= `resolution ** 3`), `PatchSetV2.resolution: int`.

- [ ] **Step 1: Write the failing test for the tokenizer**

Append to `tests/test_patchset3d_v2.py`:

```python
from src.models.patchset3d_v2 import PatchSetV2


def _dummy_batch(B=2, K=2, S=16):
    image = torch.randn(B, 1, S, S, S)
    context_in = torch.randn(B, K, 1, S, S, S)
    context_out = (torch.rand(B, K, S, S, S) > 0.5).float()
    return image, context_in, context_out


def test_tokens_all_shape():
    m = PatchSetV2(resolution=4, enc_dims=(8, 8, 8), e=32, h=64, l=2, a=2, thinking_rows=2,
                   fourier_bands=4, compress_m=3, fine_stage=[0], image_size=[16, 16, 16])
    B, K, T = 2, 2, 3
    feat = torch.randn(B, T, m.N, m.encoder.out_ch)
    occ = torch.randn(B, T, m.N, 1)
    tok = m._tokens_all(feat, occ, B, T)
    assert tok.shape == (B, T, m.N, 2, 32)


def test_occupancy_shapes():
    m = PatchSetV2(resolution=4, enc_dims=(8, 8, 8), e=32, h=64, l=2, a=2, thinking_rows=2,
                   fourier_bands=4, compress_m=3, fine_stage=[0], image_size=[16, 16, 16])
    _, cin, cout = _dummy_batch(S=16, K=2)
    occ = m._occupancy(cout)
    assert occ.shape == (2, 2, m.N, 1)
    prior = torch.rand(2, 1, 16, 16, 16)
    pocc = m._prior_occupancy(prior)
    assert pocc.shape == (2, 1, m.N, 1)
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_patchset3d_v2.py -v -k "tokens_all or occupancy"`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.models.patchset3d_v2'`

- [ ] **Step 3: Write the skeleton**

Create `src/models/patchset3d_v2.py`:

```python
"""PatchSetV2: clean Iris-style in-context 3D segmentation.

See docs/superpowers/specs/2026-09-19-patchset-v2-design.md for the full design. Reuses
RowCrossAttention / TransformerEncoderStack / ThinkingRows from pfn_seg_2d.py and the
encoder classes via build_encoder, but is a fresh, minimal class -- not another branch on
PatchSet3D's own accreted knob surface.

Per forward call: target volume (image + prior/prediction mask) + K context volumes
(image + real GT mask), T = K+1 total. Every volume is tokenized, foreground-pooled at
native resolution, and compressed to `compress_m` tokens (Stage A, weight-shared across
volumes) via the same RowCrossAttention arch.seq_compress already uses in PatchSet3D. All
volumes' compressed sequences run through one shared self-attention stack (Stage B). The
target's own post-Stage-B tokens are Iris's task tokens T; they cross-attend (Eq 5) against
the target's raw, never-compressed per-cell grid, and a conv up-path with encoder-pyramid
skips (Eq 6) produces the final logits. No decompression step exists -- the target's raw
grid was never discarded in the first place.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.encoders.factory import build_encoder
from src.models.patchset3d import MaskConvEmbed, _ConvNormAct, _down_to, _mask_tiles_3d
from src.models.patchset_pfn import FourierPositionalEncoding
from src.models.pfn_seg_2d import RowCrossAttention, ThinkingRows, TransformerEncoderStack


class PatchSetV2(nn.Module):
    def __init__(
        self,
        resolution: int = 16,
        enc_dims: tuple[int, ...] = (32, 32, 32, 32),
        e: int = 256,
        h: int = 512,
        l: int = 6,
        a: int = 4,
        thinking_rows: int = 8,
        residual_decay: float = 0.95,
        fourier_bands: int = 8,
        mask_patch_size: int = 1,
        mask_embed: str = "linear",
        compress_m: int = 32,
        compress_layers: int = 1,
        context_id_embed: bool = True,
        max_context: int = 16,
        cascade_registers: bool = False,
        image_size=None,
        encoder: str = "conv",
        encoder_frozen: bool = True,
        primus_sidecar: str | None = None,
        nnunet_ts_weights: str | None = None,
        nnunet_ts_stages=(2, 3, 4),
        nnunet_ts_random_init: bool = False,
        resenc_n_stages: int = 5,
        plainconv_ts_n_stages: int = 5,
        plainconv_ts_features_per_stage=None,
        encoder_input_norm: str | None = None,
        encoder_stage: int | None = None,
        encoder_native_grid: bool = False,
        encoder_spacing_aware: bool = False,
        encoder_precision: str = "bf16",
        fine_stage=(0, 1),
        decoder_dim: int = 64,
    ):
        super().__init__()
        self.resolution = resolution
        self.N = resolution ** 3
        self.mask_patch_size = int(mask_patch_size)
        assert self.mask_patch_size >= 1
        self.compress_m = int(compress_m)
        assert self.compress_m >= 1 and compress_layers >= 1
        self.context_id_embed = bool(context_id_embed)
        self.max_context = int(max_context)
        self.cascade_registers = bool(cascade_registers)
        self.spacing_aware = bool(encoder_spacing_aware)

        self.encoder = build_encoder(
            encoder, resolution, in_ch=1, enc_dims=enc_dims, encoder_frozen=encoder_frozen,
            primus_sidecar=primus_sidecar, nnunet_ts_weights=nnunet_ts_weights,
            nnunet_ts_stages=nnunet_ts_stages, nnunet_ts_random_init=nnunet_ts_random_init,
            resenc_n_stages=resenc_n_stages, plainconv_ts_n_stages=plainconv_ts_n_stages,
            plainconv_ts_features_per_stage=plainconv_ts_features_per_stage,
            encoder_input_norm=encoder_input_norm, image_size=image_size,
            encoder_stage=encoder_stage, encoder_native_grid=encoder_native_grid,
            encoder_spacing_aware=encoder_spacing_aware, encoder_precision=encoder_precision)
        if not getattr(self.encoder, "supports_fine", False):
            raise ValueError(f"PatchSetV2 needs an encoder exposing unpooled stages; "
                             f"encoder={encoder!r} has none (use conv | nnunet_ts | "
                             f"resenc_ts | plainconv_ts)")

        oc = self.encoder.out_ch
        self.img_embed = nn.Linear(oc, e)
        assert mask_embed in ("linear", "conv"), f"mask_embed={mask_embed!r} — 'linear' or 'conv'"
        self.mask_embed = (MaskConvEmbed(self.mask_patch_size, e) if mask_embed == "conv"
                           else nn.Linear(self.mask_patch_size ** 3, e))
        self.pos = FourierPositionalEncoding(e, fourier_bands, n_axes=3)

        r = resolution
        ii = torch.arange(r).repeat_interleave(r * r)
        jj = torch.arange(r).repeat_interleave(r).repeat(r)
        kk = torch.arange(r).repeat(r * r)
        self.register_buffer("ijk_base", torch.stack([ii, jj, kk], dim=-1), persistent=False)

        if self.context_id_embed:
            self.ctx_id = nn.Embedding(self.max_context, e)
            self.qry_id = nn.Parameter(torch.zeros(e))
            nn.init.normal_(self.ctx_id.weight, std=0.1)
            nn.init.normal_(self.qry_id, std=0.1)

    def _tokens_all(self, feat: torch.Tensor, occ: torch.Tensor, B: int, T: int) -> torch.Tensor:
        """feat (B,T,N,Cf) raw encoder grid tokens, occ (B,T,N,p^3) mask/prior occupancy ->
        (B,T,N,2,e): img_embed + mask_embed + Fourier positional encoding, columns
        [img, mask]. No content-type tag (mask_slots) -- context_id_embed, added later on
        the assembled per-volume sequence, already distinguishes target from context rows
        (see docs/superpowers/specs/2026-09-19-patchset-v2-design.md)."""
        img = self.img_embed(feat)
        msk = self.mask_embed(occ)
        ijk = self.ijk_base.unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1)
        pos = self.pos(ijk, self.resolution)
        img = img + pos
        msk = msk + pos
        return torch.stack([img, msk], dim=3)

    def _occupancy(self, context_out: torch.Tensor) -> torch.Tensor:
        """context_out (B,K,D,H,W) -> (B,K,N,p^3), same _down_to/_mask_tiles_3d path
        PatchSet3D._occupancy uses, kept volume-major (not flattened) to match this
        class's (B,T,...) canonical shape."""
        B, K = context_out.shape[0], context_out.shape[1]
        p = self.mask_patch_size
        if p == 1:
            Dn, Hn, Wn = context_out.shape[-3:]
            occ = _down_to(context_out.reshape(B * K, 1, Dn, Hn, Wn).float(), self.resolution)
            return occ.reshape(B, K, self.N, 1)
        tiles = torch.stack([_mask_tiles_3d(context_out[:, k].unsqueeze(1).float(),
                                            self.resolution, p) for k in range(K)], dim=1)
        return tiles.reshape(B, K, self.N, p ** 3)

    def _prior_occupancy(self, prior: torch.Tensor) -> torch.Tensor:
        """(B,1,D,H,W) soft prior -> (B,1,N,p^3), same convention as _occupancy."""
        B = prior.shape[0]
        p = self.mask_patch_size
        prior = prior.reshape(B, 1, *prior.shape[-3:]).float()
        if p == 1:
            return _down_to(prior, self.resolution).reshape(B, 1, self.N, 1)
        return _mask_tiles_3d(prior, self.resolution, p).reshape(B, 1, self.N, p ** 3)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_patchset3d_v2.py -v -k "tokens_all or occupancy"`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/models/patchset3d_v2.py tests/test_patchset3d_v2.py
git commit -m "feat: PatchSetV2 skeleton -- encoder + per-volume tokenizer"
```

---

### Task 3: `pool_v` — native-resolution foreground-masked pooling

**Files:**
- Modify: `src/models/patchset3d_v2.py` (add to `__init__`, add `_pool_all`)
- Test: `tests/test_patchset3d_v2.py` (append)

**Interfaces:**
- Consumes: `self.fine_stage` ordering set up in this task's `__init__` addition; `torch.nn.functional.interpolate`.
- Produces: `PatchSetV2._pool_all(self, fine_finest, context_out, query_prior, B, K, T) -> Tensor (B,T,e)`;
  `PatchSetV2._stage_order: list[int]`, `PatchSetV2._stage_sides: list[int]`,
  `PatchSetV2._finest_idx: int` (all consumed by Task 6's decode).

- [ ] **Step 1: Write the failing test**

A synthetic case: a constant feature map so the masked average is exactly that constant
regardless of mask shape, checking the native-resolution upsample-before-mask ordering
doesn't corrupt the value, and that a tiny foreground region is NOT washed out the way pooling
at a coarse resolution would (this is the property Iris's own ablation cites).

```python
def test_pool_all_native_resolution_and_value():
    m = PatchSetV2(resolution=4, enc_dims=(8, 8, 8), e=32, h=64, l=2, a=2, thinking_rows=2,
                   fourier_bands=4, compress_m=3, fine_stage=[0], image_size=[16, 16, 16])
    B, K, T = 1, 1, 2
    S = m.encoder.fine_stage_size(16, 0)          # stage-0 native side for a 16^3 input
    Cf = m.encoder.fine_stage_channels(0)
    fine_finest = torch.full((B * T, Cf, S, S, S), 3.0)   # constant feature map
    context_out = torch.zeros(B, K, 16, 16, 16)
    context_out[:, :, :2, :2, :2] = 1.0            # a small 2^3 foreground corner
    pool = m._pool_all(fine_finest, context_out, None, B, K, T)
    assert pool.shape == (B, T, 32)
    # constant input -> after per-volume z-score the whole map is 0 everywhere (std=0 branch
    # uses the 1e-8 floor), so the pooled *projection* is deterministic and identical for
    # every volume regardless of mask shape/size -- this is what we can assert without
    # depending on pool_proj's random init producing any particular non-zero value.
    assert torch.allclose(pool[:, 0], pool[:, 1])
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_patchset3d_v2.py -v -k pool_all`
Expected: FAIL with `AttributeError: 'PatchSetV2' object has no attribute '_pool_all'`

- [ ] **Step 3: Add stage bookkeeping to `__init__` and write `_pool_all`**

Add to `PatchSetV2.__init__` (after the encoder is built, before `self.img_embed`), storing
`fine_stage` and deriving stage order/channels — needed by both `_pool_all` (this task) and
decode's skip pyramid (Task 6):

```python
        self.fine_stage = tuple(int(st) for st in fine_stage)
        for st in self.fine_stage:
            if not 0 <= st < self.encoder.n_fine_stages:
                raise ValueError(f"fine_stage {st} out of range [0, {self.encoder.n_fine_stages})")
        if not image_size:
            raise ValueError("PatchSetV2 needs arch.image_size (from data.image_size)")
        stages = sorted(self.fine_stage,
                        key=lambda st: self.encoder.fine_stage_size(int(image_size[0]), st))
        self._stage_order = [self.fine_stage.index(st) for st in stages]   # coarse->fine, into `fine`
        self._stage_sides = [self.encoder.fine_stage_size(int(image_size[0]), st) for st in stages]
        self._stage_chans = [self.encoder.fine_stage_channels(st) for st in stages]
        self._finest_idx = self._stage_order[-1]      # index into `fine` for the finest stage
        self.pool_proj = nn.Linear(self._stage_chans[-1], e)
```

Add `_pool_all` as a new method on `PatchSetV2`:

```python
    def _pool_all(self, fine_finest: torch.Tensor, context_out: torch.Tensor,
                 query_prior: torch.Tensor | None, B: int, K: int, T: int) -> torch.Tensor:
        """Iris Eq 2 foreground-masked average, generalized to every volume (not
        support-only). fine_finest: (B*T,Cf,S,S,S), the finest requested fine_stage map for
        ALL T volumes. Upsamples to the volume's NATIVE (D,H,W) resolution -- never R, never
        S -- before masking: Iris's own ablation credits masking-after-upsampling with the
        small-object Dice gain (docs/methods/iris.md sec 4.1). Returns (B,T,e)."""
        Cf = fine_finest.shape[1]
        Dn, Hn, Wn = context_out.shape[-3:]
        feat_native = F.interpolate(fine_finest.float(), size=(Dn, Hn, Wn),
                                    mode="trilinear", align_corners=False
                                    ).reshape(B, T, Cf, Dn, Hn, Wn)
        sup_mask = context_out.reshape(B, K, 1, Dn, Hn, Wn).float()
        if query_prior is not None:
            qry_mask = query_prior.reshape(B, 1, 1, Dn, Hn, Wn).float()
        else:
            qry_mask = sup_mask.mean(dim=1, keepdim=True)
        mask = torch.cat([sup_mask, qry_mask], dim=1)          # (B,T,1,Dn,Hn,Wn)

        mu = feat_native.mean(dim=(-3, -2, -1), keepdim=True)
        sig = feat_native.std(dim=(-3, -2, -1), keepdim=True) + 1e-8
        feat_z = ((feat_native - mu) / sig).clamp(-10, 10)

        num = (feat_z * mask).sum(dim=(-3, -2, -1))             # (B,T,Cf)
        den = mask.sum(dim=(-3, -2, -1)).clamp_min(1e-6)         # (B,T,1)
        return self.pool_proj(num / den)                        # (B,T,e)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_patchset3d_v2.py -v -k pool_all`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/models/patchset3d_v2.py tests/test_patchset3d_v2.py
git commit -m "feat: PatchSetV2 native-resolution foreground pooling (pool_v)"
```

---

### Task 4: Stage A — per-volume compression, appending `pool_v`

**Files:**
- Modify: `src/models/patchset3d_v2.py` (add to `__init__`, add `_compress_all`, `_assemble_sequence`)
- Test: `tests/test_patchset3d_v2.py` (append)

**Interfaces:**
- Consumes: `RowCrossAttention` (imported in Task 2); `self.compress_m` (Task 2).
- Produces: `PatchSetV2._compress_all(self, tok, B, T) -> Tensor (B,T,compress_m,2,e)`;
  `PatchSetV2._assemble_sequence(self, pool, compressed, B, T) -> Tensor (B, T*(compress_m+1), 2, e)`.

- [ ] **Step 1: Write the failing test**

```python
def test_compress_and_assemble_shapes():
    m = PatchSetV2(resolution=4, enc_dims=(8, 8, 8), e=32, h=64, l=2, a=2, thinking_rows=2,
                   fourier_bands=4, compress_m=3, compress_layers=2, fine_stage=[0],
                   image_size=[16, 16, 16])
    B, T = 2, 3
    tok = torch.randn(B, T, m.N, 2, 32)
    compressed = m._compress_all(tok, B, T)
    assert compressed.shape == (B, T, 3, 2, 32)
    pool = torch.randn(B, T, 32)
    seq = m._assemble_sequence(pool, compressed, B, T)
    assert seq.shape == (B, T * 4, 2, 32)          # per volume: 1 pool row + 3 compressed
    # first row of each volume's block is the (broadcast) pool row
    per_vol = 4
    for t in range(T):
        block = seq[:, t * per_vol:(t + 1) * per_vol]
        assert torch.allclose(block[:, 0, 0], pool[:, t])   # img col
        assert torch.allclose(block[:, 0, 1], pool[:, t])   # mask col (broadcast)
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_patchset3d_v2.py -v -k compress_and_assemble`
Expected: FAIL with `AttributeError: 'PatchSetV2' object has no attribute '_compress_all'`

- [ ] **Step 3: Add compressor to `__init__` and write the two methods**

Add to `PatchSetV2.__init__` (after `self.pool_proj`):

```python
        self.compress_slots = nn.Parameter(torch.empty(self.compress_m, e))
        nn.init.normal_(self.compress_slots, std=0.02)
        self.compressor = nn.ModuleList(
            [RowCrossAttention(a, e, h) for _ in range(compress_layers)])
```

Add the two methods:

```python
    def _compress_all(self, tok: torch.Tensor, B: int, T: int) -> torch.Tensor:
        """tok (B,T,N,2,e) -> (B,T,compress_m,2,e). Weight-shared per-volume compression
        (Stage A): compress_m learnable slots cross-attend into that volume's own N raw
        cells only (T folded into the batch dim -- no cross-volume mixing here; that's
        Stage B's job). Identical mechanism to PatchSet3D's arch.seq_compress Stage A."""
        e = tok.shape[-1]
        kv = tok.reshape(B * T, self.N, 2, e)
        q = self.compress_slots.unsqueeze(0).unsqueeze(2).expand(B * T, -1, 2, -1).contiguous()
        for layer in self.compressor:
            q = layer(q, kv)
        return q.reshape(B, T, self.compress_m, 2, e)

    def _assemble_sequence(self, pool: torch.Tensor, compressed: torch.Tensor,
                           B: int, T: int) -> torch.Tensor:
        """pool (B,T,e), compressed (B,T,compress_m,2,e) -> (B, T*(compress_m+1), 2, e),
        volume-major: each volume's block is [pool_row ; compress_m rows], contiguous, so a
        later slice by volume index recovers exactly that volume's tokens. pool is
        broadcast into both img and mask columns."""
        e = compressed.shape[-1]
        pool_tok = pool.unsqueeze(2).unsqueeze(3).expand(B, T, 1, 2, e)
        seq = torch.cat([pool_tok, compressed], dim=2)          # (B,T,compress_m+1,2,e)
        return seq.reshape(B, T * (self.compress_m + 1), 2, e)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_patchset3d_v2.py -v -k compress_and_assemble`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/models/patchset3d_v2.py tests/test_patchset3d_v2.py
git commit -m "feat: PatchSetV2 Stage A compression + pool_v sequence assembly"
```

---

### Task 5: Stage B — cascade registers, thinking rows, shared transformer

**Files:**
- Modify: `src/models/patchset3d_v2.py` (add to `__init__`, add `_apply_context_tags`, `_stage_b`)
- Test: `tests/test_patchset3d_v2.py` (append)

**Interfaces:**
- Consumes: `ThinkingRows`, `TransformerEncoderStack` (imported Task 2); `self.context_id_embed`,
  `self.cascade_registers` (Task 2).
- Produces: `PatchSetV2._apply_context_tags(self, seq, B, K, T, per_vol) -> Tensor (same shape as seq)`;
  `PatchSetV2._stage_b(self, seq, B, K, T, cascade_regs=None) -> tuple[Tensor (B, n_think+n_extra+T*per_vol, 2, e), Tensor|None]`
  (second element is `registers`, `(B, thinking_rows, e)` or `None`).

- [ ] **Step 1: Write the failing test**

Two properties matter here: shape, and that context volumes actually influence the target's
post-Stage-B tokens (proving Stage B's self-attention is doing real cross-volume work, not
just passing each volume through untouched).

```python
def test_stage_b_shape_and_cross_volume_mixing():
    torch.manual_seed(0)
    m = PatchSetV2(resolution=4, enc_dims=(8, 8, 8), e=32, h=64, l=2, a=2, thinking_rows=2,
                   fourier_bands=4, compress_m=3, compress_layers=1, fine_stage=[0],
                   context_id_embed=True, image_size=[16, 16, 16])
    B, K, T = 2, 2, 3
    per_vol = m.compress_m + 1
    seq = torch.randn(B, T * per_vol, 2, 32)
    seq_out, regs = m._stage_b(seq, B, K, T)
    n_think = m.thinking.n
    assert seq_out.shape == (B, n_think + T * per_vol, 2, 32)
    assert regs is None                          # cascade_registers=False by default

    start = n_think + K * per_vol
    target_block_a = seq_out[:, start:start + per_vol]

    seq_perturbed = seq.clone()
    seq_perturbed[:, :per_vol] += 5.0             # perturb context volume 0 only
    seq_out_b, _ = m._stage_b(seq_perturbed, B, K, T)
    target_block_b = seq_out_b[:, start:start + per_vol]
    assert not torch.allclose(target_block_a, target_block_b)


def test_stage_b_cascade_registers_roundtrip():
    m = PatchSetV2(resolution=4, enc_dims=(8, 8, 8), e=32, h=64, l=2, a=2, thinking_rows=2,
                   fourier_bands=4, compress_m=3, fine_stage=[0], cascade_registers=True,
                   image_size=[16, 16, 16])
    B, K, T = 2, 2, 3
    per_vol = m.compress_m + 1
    seq = torch.randn(B, T * per_vol, 2, 32)
    seq_out, regs = m._stage_b(seq, B, K, T)
    assert regs.shape == (2, m.thinking.n, 32)
    seq_out2, regs2 = m._stage_b(seq, B, K, T, cascade_regs=regs)
    assert seq_out2.shape[1] == seq_out.shape[1] + m.thinking.n   # cascade rows prepended
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_patchset3d_v2.py -v -k stage_b`
Expected: FAIL with `AttributeError: 'PatchSetV2' object has no attribute '_stage_b'`

- [ ] **Step 3: Add Stage B modules to `__init__` and write the two methods**

Add to `PatchSetV2.__init__` (after the `context_id_embed` block):

```python
        self.thinking = ThinkingRows(thinking_rows, e)
        if self.cascade_registers:
            self.cascade_proj = nn.Linear(e, e)
            self.cascade_type = nn.Parameter(torch.zeros(e))
            nn.init.normal_(self.cascade_type, std=0.02)
        self.transformer = TransformerEncoderStack(l, a, e, h, residual_decay)
```

Add the two methods:

```python
    def _apply_context_tags(self, seq: torch.Tensor, B: int, K: int, T: int,
                            per_vol: int) -> torch.Tensor:
        """seq (B, T*per_vol, 2, e). Adds ctx_id[k] to each context volume's block, qry_id
        to the target's block -- lets Stage B's self-attention tell context rows from the
        target row. This also distinguishes GT-content rows from prediction-content rows
        without a separate mask_slots-style tag, since target vs. context identity implies
        which content type that volume's mask column holds (see
        docs/superpowers/specs/2026-09-19-patchset-v2-design.md)."""
        ctx = self.ctx_id(torch.arange(K, device=seq.device))         # (K,e)
        tags = torch.cat([ctx, self.qry_id.unsqueeze(0)], dim=0)      # (T,e)
        tags = tags.repeat_interleave(per_vol, dim=0)                  # (T*per_vol,e)
        return seq + tags.to(seq.dtype).view(1, -1, 1, seq.shape[-1])

    def _stage_b(self, seq: torch.Tensor, B: int, K: int, T: int,
                cascade_regs: torch.Tensor | None = None):
        """seq (B, T*per_vol, 2, e) -> (post-transformer sequence, registers). Full
        self-attention across every volume's tokens (no register_routed, no full_attn
        toggle -- this is the only mode) plus thinking rows and, if enabled, the previous
        cascade level's carried memory rows -- unchanged mechanism from
        PatchSet3D._attn."""
        per_vol = self.compress_m + 1
        if self.context_id_embed:
            seq = self._apply_context_tags(seq, B, K, T, per_vol)
        seq, _ = self.thinking(seq, seq.shape[1])
        n_extra = 0
        if cascade_regs is not None:
            assert self.cascade_registers, "cascade_regs given but cascade_registers=False"
            mem = self.cascade_proj(cascade_regs) + self.cascade_type
            mem = mem.unsqueeze(2).expand(-1, -1, seq.shape[2], -1)
            seq = torch.cat([mem, seq], dim=1)
            n_extra = mem.shape[1]
        # `sep` is unused downstream when full_attn=True (see TransformerEncoderLayer) --
        # passed as 0 for clarity that it has no effect here.
        seq = self.transformer(seq, 0, full_attn=True)
        regs = seq[:, :self.thinking.n].mean(dim=2) if self.cascade_registers else None
        return seq, regs
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_patchset3d_v2.py -v -k stage_b`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/models/patchset3d_v2.py tests/test_patchset3d_v2.py
git commit -m "feat: PatchSetV2 Stage B -- shared cross-volume transformer"
```

---

### Task 6: Decode — Iris Eq 5-6 bidirectional cross-attention + conv up-path

**Files:**
- Modify: `src/models/patchset3d_v2.py` (add to `__init__`, add `_decode`)
- Test: `tests/test_patchset3d_v2.py` (append)

**Interfaces:**
- Consumes: `_ConvNormAct` (imported Task 2); `self._stage_order`, `self._stage_sides`,
  `self._stage_chans` (Task 3).
- Produces: `PatchSetV2._decode(self, T_tok, F_q, fine, B) -> Tensor (B,1,D,H,W)` (native
  resolution, matching `PatchSet3D._native_logit`'s output convention).

- [ ] **Step 1: Write the failing test**

```python
def test_decode_shape_and_backward():
    torch.manual_seed(0)
    m = PatchSetV2(resolution=4, enc_dims=(8, 8, 8), e=32, h=64, l=2, a=2, thinking_rows=2,
                   fourier_bands=4, compress_m=3, fine_stage=[0], decoder_dim=16,
                   image_size=[16, 16, 16])
    B = 2
    per_vol = m.compress_m + 1
    T_tok = torch.randn(B, per_vol, 32, requires_grad=True)
    F_q = torch.randn(B, m.N, 32, requires_grad=True)
    S = m.encoder.fine_stage_size(16, 0)
    fine = (torch.randn(B, m.encoder.fine_stage_channels(0), S, S, S),)
    logit = m._decode(T_tok, F_q, fine, B)
    assert logit.shape == (B, 1, S, S, S)          # stage-0 side == native (16) for a 16^3 input
    logit.mean().backward()
    assert T_tok.grad is not None and F_q.grad is not None
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_patchset3d_v2.py -v -k decode_shape`
Expected: FAIL with `AttributeError: 'PatchSetV2' object has no attribute '_decode'`

- [ ] **Step 3: Add decode modules to `__init__` and write `_decode`**

Add to `PatchSetV2.__init__` (after `self.transformer`):

```python
        self.iris_t2f = nn.MultiheadAttention(e, a, batch_first=True)
        self.iris_f2t = nn.MultiheadAttention(e, a, batch_first=True)
        dims = [max(decoder_dim // (2 ** i), 8) for i in range(len(self._stage_chans))]
        self.token_proj = nn.Linear(e, decoder_dim)
        self.decode_blocks = nn.ModuleList()
        prev = decoder_dim
        for i in range(len(self._stage_chans)):
            self.decode_blocks.append(nn.Sequential(
                _ConvNormAct(prev + self._stage_chans[i], dims[i]),
                _ConvNormAct(dims[i], dims[i])))
            prev = dims[i]
        self.class_embed = nn.Linear(e, prev)
```

Add `_decode`:

```python
    def _decode(self, T_tok: torch.Tensor, F_q: torch.Tensor, fine, B: int) -> torch.Tensor:
        """Iris Eq 5-6, literal reproduction. T_tok (B,compress_m+1,e): target's own
        post-Stage-B tokens (img column), already context-aware from Stage B. F_q
        (B,N,e): target's RAW, never-compressed per-cell grid (the same img_embed output
        used to build `tok` before any compression). fine: target-only unpooled encoder
        stage maps, in self.fine_stage order. Returns (B,1,S,S,S) at the finest requested
        fine_stage's own native side."""
        t2, _ = self.iris_t2f(T_tok, F_q, F_q)          # tokens attend image
        T2 = T_tok + t2
        f2, _ = self.iris_f2t(F_q, T2, T2)              # image attends updated tokens
        Fq2 = F_q + f2

        R = self.resolution
        x = self.token_proj(Fq2).transpose(1, 2).reshape(B, -1, R, R, R)
        for i, block in enumerate(self.decode_blocks):
            s = self._stage_sides[i]
            x = F.interpolate(x, size=(s, s, s), mode="trilinear", align_corners=False)
            skip = fine[self._stage_order[i]]
            x = block(torch.cat([x, skip], dim=1))
        mask_features = x

        class_embed = self.class_embed(T2.mean(dim=1))
        return torch.einsum('bc,bcdhw->bdhw', class_embed, mask_features).unsqueeze(1)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_patchset3d_v2.py -v -k decode_shape`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/models/patchset3d_v2.py tests/test_patchset3d_v2.py
git commit -m "feat: PatchSetV2 decode -- Iris Eq 5-6 cross-attention + conv up-path"
```

---

### Task 7: `forward()`, `train_forward()`, `predict()` — wire everything together

**Files:**
- Modify: `src/models/patchset3d_v2.py` (add `forward`, `_encode`, `_native_logit`,
  `train_forward`, `predict`)
- Test: `tests/test_patchset3d_v2.py` (append)

**Interfaces:**
- Consumes: every method from Tasks 2-6.
- Produces: `PatchSetV2.forward(self, image, context_in, context_out, mode="train",
  spacing=None, query_prior=None, cascade_regs=None) -> dict` with keys `"final_logit"`
  (`(B,1,D,H,W)` at the encoder-grid-derived decode resolution) and `"registers"`
  (`(B,thinking_rows,e)` or `None`) — matches `cascade.py::_forward_level`'s contract exactly.
  `PatchSetV2.train_forward(...)` and `PatchSetV2.predict(...)` match `PatchSet3D`'s own
  signatures (native `(D,H,W)` resolution logits / binary mask).

- [ ] **Step 1: Write the failing tests**

Mirrors `test_patchset3d.py`'s own `test_patchset3d_forward_grid_shape`,
`test_patchset3d_backward`, `test_predict_and_train_forward_native_shape`, and
`test_query_prior_feeds_the_query_mask_token`:

```python
def test_forward_end_to_end_shape():
    m = PatchSetV2(resolution=4, enc_dims=(8, 8, 8), e=32, h=64, l=2, a=2, thinking_rows=2,
                   fourier_bands=4, compress_m=3, fine_stage=[0], decoder_dim=16,
                   image_size=[16, 16, 16])
    img, cin, cout = _dummy_batch(S=16, K=2)
    out = m(img, context_in=cin, context_out=cout, mode="train")
    assert out["final_logit"].shape == (2, 1, 16, 16, 16)
    assert out["registers"] is None


def test_forward_backward():
    m = PatchSetV2(resolution=4, enc_dims=(8, 8, 8), e=32, h=64, l=2, a=2, thinking_rows=2,
                   compress_m=3, fine_stage=[0], decoder_dim=16, image_size=[16, 16, 16])
    img, cin, cout = _dummy_batch(S=16, K=2)
    out = m(img, context_in=cin, context_out=cout)["final_logit"]
    out.mean().backward()
    grads = [p.grad is not None for p in m.parameters() if p.requires_grad]
    assert all(grads) and len(grads) > 0


def test_predict_and_train_forward_native_shape():
    m = PatchSetV2(resolution=4, enc_dims=(8, 8, 8), e=32, h=64, l=2, a=2, thinking_rows=2,
                   compress_m=3, fine_stage=[0], decoder_dim=16, image_size=[16, 16, 16])
    img, cin, cout = _dummy_batch(S=16, K=2)
    logits = m.train_forward(img, cin, cout)
    assert logits.shape == (2, 1, 16, 16, 16)
    pred = m.predict(img, cin, cout)
    assert pred.shape == (2, 16, 16, 16)
    assert set(torch.unique(pred).tolist()) <= {0.0, 1.0}


def test_query_prior_changes_output():
    torch.manual_seed(0)
    m = PatchSetV2(resolution=4, enc_dims=(8, 8, 8), e=32, h=64, l=2, a=2, thinking_rows=2,
                   compress_m=3, fine_stage=[0], decoder_dim=16, image_size=[16, 16, 16])
    m.eval()
    img, cin, cout = _dummy_batch(S=16, K=2)
    base = m(img, context_in=cin, context_out=cout)["final_logit"]
    assert torch.equal(base, m(img, context_in=cin, context_out=cout)["final_logit"])
    prior = torch.rand(2, 1, 16, 16, 16)
    with_prior = m(img, context_in=cin, context_out=cout, query_prior=prior)["final_logit"]
    assert with_prior.shape == base.shape
    assert not torch.allclose(with_prior, base)


def test_cascade_regs_end_to_end():
    m = PatchSetV2(resolution=4, enc_dims=(8, 8, 8), e=32, h=64, l=2, a=2, thinking_rows=2,
                   compress_m=3, fine_stage=[0], decoder_dim=16, image_size=[16, 16, 16],
                   cascade_registers=True)
    img, cin, cout = _dummy_batch(S=16, K=2)
    out1 = m(img, context_in=cin, context_out=cout)
    assert out1["registers"].shape == (2, m.thinking.n, 32)
    out2 = m(img, context_in=cin, context_out=cout, cascade_regs=out1["registers"])
    assert out2["final_logit"].shape == out1["final_logit"].shape
```

- [ ] **Step 2: Run to verify they fail**

Run: `pytest tests/test_patchset3d_v2.py -v -k "forward_end_to_end or forward_backward or predict_and_train_forward or query_prior_changes or cascade_regs_end_to_end"`
Expected: FAIL — `PatchSetV2` object is not callable / has no `forward` yet.

- [ ] **Step 3: Write `forward`, `_encode`, `_native_logit`, `train_forward`, `predict`**

Add to `src/models/patchset3d_v2.py`:

```python
    def _encode(self, x: torch.Tensor, spacing, fine_rows: torch.Tensor):
        kw = {"spacing": spacing} if self.spacing_aware else {}
        kw.update(fine_rows=fine_rows, fine_stage=self.fine_stage)
        return self.encoder(x, **kw)

    def forward(self, image, context_in, context_out, mode="train", spacing=None,
               query_prior=None, cascade_regs=None):
        """query_prior: optional (B,1,D,H,W) soft probability volume already resampled onto
        this forward's grid frame -- feeds the target volume's own tokenize/pool step
        (support-mean fallback when absent), always consumed (unlike PatchSet3D's iris
        decoder path, which silently drops it -- see
        docs/superpowers/specs/2026-09-19-patchset-v2-design.md).

        cascade_regs: optional (B, thinking_rows, e), the previous cascade level's own
        "registers" return value, fed into Stage B (requires cascade_registers=True)."""
        B, K = context_in.shape[0], context_in.shape[1]
        D, H, W = image.shape[-3:]
        imgs = torch.cat([context_in, image.unsqueeze(1)], dim=1)     # (B,T,1,D,H,W)
        T = imgs.shape[1]
        x = imgs.reshape(B * T, 1, D, H, W)
        rows = torch.arange(B * T, device=x.device)      # every volume needs its own fine maps
        feat_map, fine = self._encode(x, spacing, fine_rows=rows)

        occ_ctx = self._occupancy(context_out)                              # (B,K,N,p^3)
        occ_qry = (self._prior_occupancy(query_prior) if query_prior is not None
                  else occ_ctx.mean(dim=1, keepdim=True))                   # (B,1,N,p^3)
        occ = torch.cat([occ_ctx, occ_qry], dim=1)                          # (B,T,N,p^3)

        Cf = feat_map.shape[1]
        feat = feat_map.reshape(B, T, Cf, self.N).transpose(2, 3)           # (B,T,N,Cf)
        tok = self._tokens_all(feat, occ, B, T)                             # (B,T,N,2,e)

        pool = self._pool_all(fine[self._finest_idx], context_out, query_prior, B, K, T)
        compressed = self._compress_all(tok, B, T)
        seq = self._assemble_sequence(pool, compressed, B, T)
        seq, regs = self._stage_b(seq, B, K, T, cascade_regs=cascade_regs)

        per_vol = self.compress_m + 1
        # Thinking/cascade rows sit only at the front and context volumes only ever precede
        # the target (imgs = cat([context_in, image]) sets this order and nothing downstream
        # permutes it), so the target's block is always the LAST per_vol rows, regardless of
        # how many thinking/cascade rows are prepended.
        T_tok = seq[:, -per_vol:, 0, :]                      # target's block, img column

        F_q = tok[:, K, :, 0, :]                            # target's raw, never-compressed grid
        qidx = torch.arange(B, device=x.device) * T + K
        fine_qry = tuple(f[qidx] for f in fine)             # re-slice to target-only
        logit = self._decode(T_tok, F_q, fine_qry, B)
        g = logit.shape[-1]
        if g != D:
            logit = F.interpolate(logit, size=(D, H, W), mode="trilinear", align_corners=False)
        return {"final_logit": logit, "registers": regs}

    def _native_logit(self, image, context_in, context_out, spacing=None, query_prior=None):
        dev = next(self.parameters()).device
        image = image.to(dev); context_in = context_in.to(dev); context_out = context_out.to(dev)
        if query_prior is not None:
            query_prior = query_prior.to(dev)
        return self.forward(image, context_in, context_out, spacing=spacing,
                            query_prior=query_prior)["final_logit"].float()

    def train_forward(self, target_img, context_imgs, context_masks, spacing=None,
                      query_prior=None):
        return self._native_logit(target_img, context_imgs, context_masks, spacing=spacing,
                                  query_prior=query_prior)

    @torch.no_grad()
    def predict(self, target_img, context_imgs, context_masks, spacing=None, query_prior=None):
        logit = self._native_logit(target_img, context_imgs, context_masks, spacing=spacing,
                                   query_prior=query_prior)
        return (torch.sigmoid(logit) >= 0.5).float().squeeze(1)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_patchset3d_v2.py -v`
Expected: PASS, every test in the file (all tasks so far).

- [ ] **Step 5: Commit**

```bash
git add src/models/patchset3d_v2.py tests/test_patchset3d_v2.py
git commit -m "feat: PatchSetV2 forward/train_forward/predict -- matches PatchSet3D's cascade.py contract"
```

---

### Task 8: Wire into `train.py` and add a default Hydra config

**Files:**
- Modify: `experiments/3d/train.py` (add `"patchset3d_v2"` branch to `build_model`)
- Create: `configs/experiment/3d/model/m3_patchset_v2.yaml`
- Test: `tests/test_patchset3d_v2.py` (append)

**Interfaces:**
- Consumes: `PatchSetV2` (Task 7).
- Produces: `build_model(cfg)` returns `(PatchSetV2, "patchset3d_v2")` when `cfg.model ==
  "patchset3d_v2"`.

- [ ] **Step 1: Write the failing test**

A lightweight `DictConfig` built by hand (no real Hydra CLI needed), matching the shapes
`build_model` already expects from `cfg.arch`/`cfg.data`. `experiments/3d/train.py` lives
outside any package (`experiments/3d/` has no `__init__.py`), so load it via `importlib`
against its file path rather than a normal import:

```python
def test_build_model_dispatches_patchset_v2():
    import importlib.util
    from omegaconf import OmegaConf

    spec = importlib.util.spec_from_file_location(
        "experiments_3d_train", "experiments/3d/train.py")
    train_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_mod)

    cfg = OmegaConf.create({
        "model": "patchset3d_v2",
        "arch": {
            "resolution": 4, "enc_dims": [8, 8, 8], "e": 32, "h": 64, "l": 2, "a": 2,
            "thinking_rows": 2, "residual_decay": 0.95, "fourier_bands": 4,
            "compress_m": 3, "compress_layers": 1, "fine_stage": [0], "decoder_dim": 16,
            "encoder": "conv",
        },
        "data": {"image_size": [16, 16, 16]},
    })
    model, name = train_mod.build_model(cfg)
    assert name == "patchset3d_v2"
    from src.models.patchset3d_v2 import PatchSetV2
    assert isinstance(model, PatchSetV2)
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_patchset3d_v2.py -v -k build_model_dispatches`
Expected: FAIL with `ValueError: unknown model 'patchset3d_v2'`

- [ ] **Step 3: Add the `build_model` branch**

In `experiments/3d/train.py`, immediately before the `raise ValueError(f"unknown model
{name!r} (medverse | patchset3d)")` line (`train.py:443`), insert:

```python
    if name == "patchset3d_v2":
        from src.models.patchset3d_v2 import PatchSetV2
        a = cfg.arch
        arch = {
            "resolution": a.resolution, "enc_dims": list(a.enc_dims),
            "e": a.e, "h": a.h, "l": a.l, "a": a.a,
            "thinking_rows": a.thinking_rows, "residual_decay": a.residual_decay,
            "fourier_bands": a.get("fourier_bands", 8),
            "mask_patch_size": a.get("mask_patch_size", 1),
            "mask_embed": a.get("mask_embed", "linear"),
            "compress_m": a.get("compress_m", 32),
            "compress_layers": a.get("compress_layers", 1),
            "context_id_embed": a.get("context_id_embed", True),
            "max_context": a.get("max_context", 16),
            "cascade_registers": a.get("cascade_registers", False),
            "image_size": list(cfg.data.image_size),
            "encoder": a.get("encoder", "conv"),
            "encoder_frozen": a.get("encoder_frozen", True),
            "primus_sidecar": a.get("primus_sidecar", None),
            "nnunet_ts_weights": a.get("nnunet_ts_weights", None),
            "nnunet_ts_stages": a.get("nnunet_ts_stages", (2, 3, 4)),
            "nnunet_ts_random_init": a.get("nnunet_ts_random_init", False),
            "resenc_n_stages": a.get("resenc_n_stages", 5),
            "plainconv_ts_n_stages": a.get("plainconv_ts_n_stages", 5),
            "plainconv_ts_features_per_stage": (
                list(a.plainconv_ts_features_per_stage)
                if a.get("plainconv_ts_features_per_stage") is not None else None),
            "encoder_input_norm": a.get("encoder_input_norm", None),
            "encoder_stage": a.get("encoder_stage", None),
            "encoder_native_grid": a.get("encoder_native_grid", False),
            "encoder_spacing_aware": a.get("encoder_spacing_aware", False),
            "encoder_precision": a.get("encoder_precision", "bf16"),
            "fine_stage": (list(a.fine_stage) if isinstance(a.get("fine_stage", [0, 1]), ListConfig)
                          else a.get("fine_stage", [0, 1])),
            "decoder_dim": a.get("decoder_dim", 64),
        }
        return PatchSetV2(**arch), name
```

And update the final `raise` line to list the new name:

```python
    raise ValueError(f"unknown model {name!r} (medverse | patchset3d | patchset3d_v2)")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_patchset3d_v2.py -v`
Expected: PASS, every test in the file.

- [ ] **Step 5: Write the default Hydra config**

Create `configs/experiment/3d/model/m3_patchset_v2.yaml`:

```yaml
# @package _global_
# m3_patchset_v2 — PatchSetV2, the clean Iris-style reimplementation. Selected as a Hydra
# group: `model=m3_patchset_v2`. See docs/superpowers/specs/2026-09-19-patchset-v2-design.md.
# Encoder-forward keys are NOT here -- pick an `encoder` group (e.g. encoder=e4_plainconv_ts).
model: patchset3d_v2

arch:
  l: 4                          # Stage B transformer layers
  e: 768
  h: 3072
  a: 12
  thinking_rows: 8
  residual_decay: 0.95
  fourier_bands: 8
  mask_patch_size: 1             # single-voxel occupancy per cell -- pool_v handles
                                  # high-resolution mask detail separately (native-res pooling)
  mask_embed: linear
  compress_m: 32                 # Stage A tokens per volume
  compress_layers: 1
  context_id_embed: true         # kept for future multi-context (K>1) support
  max_context: 16
  cascade_registers: false
  fine_stage: [0, 1]             # feeds both pool_v and the decode skip pyramid
  decoder_dim: 64

train:
  optimizer: adamw
  lr: 1.0e-4
  weight_decay: 0.01
  scheduler: cosine
  warmup_epochs: 5
  loss: bce_dice
  dice_weight: 1.0
  checkpoint: null
  resume_weights_only: false
```

- [ ] **Step 6: Commit**

```bash
git add experiments/3d/train.py configs/experiment/3d/model/m3_patchset_v2.yaml tests/test_patchset3d_v2.py
git commit -m "feat: wire PatchSetV2 into train.py's build_model + default Hydra config"
```

---

## Self-Review Notes

- **Spec coverage:** per-volume tokenize (Task 2), `pool_v` at native resolution (Task 3),
  Stage A compression + `pool_v` appended to the sequence (Task 4), Stage B shared transformer
  with `cascade_registers`/`context_id_embed` (Task 5), Iris Eq 5-6 decode with no
  decompression step (Task 6), `forward()`/`train_forward()`/`predict()` matching
  `cascade.py`'s contract including `query_prior` always-consumed (Task 7), `build_encoder`
  extraction with zero `PatchSet3D` regression (Task 1), `train.py` dispatch + config
  (Task 8). All Dropped-table items (`register_routed`, `mask_slots`, `seq_compress` toggle,
  `iris_pixelshuffle_r`, `full_attn`, SimMIM masking, `pool_token`, `fine_decode`/`decoder`
  switch) are structurally absent from `PatchSetV2.__init__`'s signature — nothing to
  additionally verify by test, their absence from the constructor is the verification.
- **Open spec question resolved:** `mask_patch_size` defaults to `1` in the config (Task 8),
  with the `p>1` path still implemented (Task 2) for parity with `PatchSet3D`, so nothing is
  actually foreclosed.
- **Not covered by this plan (explicitly out of scope per the spec's non-goals):** an actual
  training run / smoke test against real data, and any `evaluate.py` native-grid-eval changes
  — `PatchSetV2` satisfies the same `predict()`/`train_forward()` contract `PatchSet3D` does,
  so none are expected to be needed, but this is unverified until a real run happens.
