# PrimusEncoder native-grid mode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in `native_grid` mode to `PrimusEncoder` so the frozen Primus ViT tokenizes at `image_size/8` instead of always resampling to 192³, making encoder FLOPs scale with `data.image_size`.

**Architecture:** Two pure module-level helpers (target-shape rounding + RoPE grid rebuild) are unit-tested in isolation. `PrimusEncoder._preprocess` and `._encode` call them only when the new `native_grid` flag is set; default off preserves today's fixed-192 behavior. The flag threads through `PatchSet3D` and the `train.py` arch dict, with a schema default in the model config. Downstream (`_down_to → resolution`, transformer, decode) is untouched.

**Tech Stack:** PyTorch, `dynamic_network_architectures` Primus/Eva, `timm.layers.RotaryEmbeddingCat`, Hydra configs, pytest.

## Global Constraints

- Env is node-specific and uv-managed; the Primus package lives in the working venv. Run python/pytest via `.venv_thor` on this node: `.venv_thor/bin/pytest`, `.venv_thor/bin/python`. (`.venv` is corrupted — never use it.)
- Patch size for CoLiPri Primus is 8; the ViT token grid is `input/8`. `image_size` in native-grid mode must be divisible by 8 (round to nearest multiple of 8 with a one-time warning otherwise).
- RoPE reference frame is **identity**: `ref_feat_shape = feat_shape = grid`.
- Default `native_grid=False` — existing experiments and `experiments/3d/feature_sim/adapters.py` must be byte-for-byte unaffected.
- Follow repo test style (see `tests/test_fourier_nd.py`): plain `pytest`, import from `src...`.
- Log the change in `docs/logs.md`.
- `experiment=30_colipri_encoder` has `compile: true`, which does `torch.compile(enc.primus.eva, dynamic=True)` (`train.py:519-520`). Native-grid mutates `p.eva.rope.pos_embed` before the eva call; since `image_size` is fixed per run the grid is stable (rope rebuilt once) and `dynamic=True` absorbs the shape at first call. If Task 3 Step 4 hits a compile error, re-run that verification with `arch.compile=false` to isolate — but the encoder path itself must still work.

---

### Task 1: Pure helpers for target shape + RoPE grid rebuild

**Files:**
- Modify: `src/models/primus_encoder.py` (add two module-level functions near the top, after imports)
- Test: `tests/test_primus_native_grid.py` (create)

**Interfaces:**
- Produces:
  - `_native_target_shape(shape: tuple[int,int,int], patch: int) -> tuple[int,int,int]` — rounds each dim to the nearest positive multiple of `patch` (min `patch`).
  - `_set_rope_identity_grid(rope, grid: tuple[int,int,int]) -> None` — sets `rope.ref_feat_shape = list(grid)` then calls `rope.update_feat_shape(list(grid))`, so the rebuilt `rope.pos_embed` has `prod(grid)` positions with identity frequencies. No-op-safe to call repeatedly with the same grid.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_primus_native_grid.py
"""Unit tests for PrimusEncoder native-grid helpers (no model weights loaded)."""
import math

import torch
from timm.layers import RotaryEmbeddingCat

from src.models.primus_encoder import _native_target_shape, _set_rope_identity_grid


def test_native_target_shape_divisible_is_passthrough():
    assert _native_target_shape((128, 128, 128), 8) == (128, 128, 128)
    assert _native_target_shape((192, 192, 192), 8) == (192, 192, 192)


def test_native_target_shape_rounds_to_nearest_multiple():
    # 130 -> 128 (nearest), 132 -> 136 (ties/above), min is one patch
    assert _native_target_shape((130, 130, 130), 8) == (128, 128, 128)
    assert _native_target_shape((4, 4, 4), 8) == (8, 8, 8)


def _make_rope(grid):
    # Mirror Primus' construction: fixed feat_shape (bands=None), identity ref.
    dim = 24  # rope_dim; divisible by 4
    return RotaryEmbeddingCat(dim, in_pixels=False, feat_shape=list(grid),
                              ref_feat_shape=list(grid))


def test_set_rope_identity_grid_rebuilds_for_new_grid():
    rope = _make_rope((24, 24, 24))
    _set_rope_identity_grid(rope, (16, 16, 16))
    assert tuple(rope.feat_shape) == (16, 16, 16)
    assert tuple(rope.ref_feat_shape) == (16, 16, 16)
    # pos_embed rows == number of tokens in the new grid
    assert rope.pos_embed.shape[0] == 16 ** 3


def test_set_rope_identity_grid_stable_recall():
    rope = _make_rope((24, 24, 24))
    _set_rope_identity_grid(rope, (16, 16, 16))
    emb1 = rope.pos_embed.clone()
    _set_rope_identity_grid(rope, (16, 16, 16))  # same grid again
    assert torch.equal(emb1, rope.pos_embed)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv_thor/bin/pytest tests/test_primus_native_grid.py -v`
Expected: FAIL with `ImportError: cannot import name '_native_target_shape'`

- [ ] **Step 3: Write minimal implementation**

Add near the top of `src/models/primus_encoder.py`, after the existing imports:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv_thor/bin/pytest tests/test_primus_native_grid.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/models/primus_encoder.py tests/test_primus_native_grid.py
git commit -m "feat(primus): native-grid helpers (target shape + identity RoPE rebuild)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: Wire `native_grid` into PrimusEncoder preprocessing + encode

**Files:**
- Modify: `src/models/primus_encoder.py` (`__init__`, `_preprocess`, `_encode`)

**Interfaces:**
- Consumes: `_native_target_shape`, `_set_rope_identity_grid` from Task 1.
- Produces: `PrimusEncoder(..., native_grid: bool = False)`. When `native_grid=True`, `_preprocess` feeds the input at `_native_target_shape(input, patch)` (native size if divisible by 8) instead of the fixed `input_shape`, and `_encode` rebuilds the eva RoPE to the resulting token grid before running the blocks. `self.patch_size: int` is stored from `primus_kwargs["patch_embed_size"][0]`.

- [ ] **Step 1: Add the flag + patch_size in `__init__`**

In `PrimusEncoder.__init__`, extend the signature and store state. Change the signature line:

```python
    def __init__(self, sidecar_path, resolution, frozen=True, device="cuda",
                 cache_max=4096, encoder_stage=None, native_grid=False):
```

After `self.input_shape = tuple(kw["input_shape"])` add:

```python
        self.patch_size = int(kw["patch_embed_size"][0])
        self.native_grid = bool(native_grid)
```

- [ ] **Step 2: Update `_preprocess` to honor native_grid**

Replace the resample block in `_preprocess`:

```python
        if tuple(v.shape[-3:]) != self.input_shape:
            v = F.interpolate(v, size=self.input_shape, mode="trilinear", align_corners=False)
        return v
```

with:

```python
        target = (_native_target_shape(tuple(v.shape[-3:]), self.patch_size)
                  if self.native_grid else self.input_shape)
        if tuple(v.shape[-3:]) != target:
            if self.native_grid and not self._warned_resize:
                print(f"[PrimusEncoder] native_grid: input {tuple(v.shape[-3:])} not a "
                      f"multiple of patch {self.patch_size}; resampling to {target}")
                self._warned_resize = True
            v = F.interpolate(v, size=target, mode="trilinear", align_corners=False)
        return v
```

Add `self._warned_resize = False` in `__init__` (next to `self.native_grid`).

- [ ] **Step 3: Rebuild RoPE for the actual grid in `_encode`**

In `_encode`, after `x = p.down_projection(x)` and `B, C, W, H, D = x.shape`, before the `x = x.flatten(2)...` line, insert:

```python
        if self.native_grid:
            _set_rope_identity_grid(p.eva.rope, (W, H, D))
```

- [ ] **Step 4: Manual verification (native grid changes token count + FLOPs)**

The full encoder loads ~1.2 GB weights, so verify by hand rather than in CI. Run:

```bash
.venv_thor/bin/python - <<'PY'
import torch
from src.models.primus_encoder import PrimusEncoder

sc = "results/checkpoints/primus_colipri.json"
for ng in (False, True):
    enc = PrimusEncoder(sc, resolution=24, frozen=True, device="cpu", native_grid=ng)
    enc.eval()
    x = torch.zeros(1, 1, 128, 128, 128)
    v = enc._preprocess(x)
    f = enc._encode(v)               # (B, out_ch, g, g, g)
    print(f"native_grid={ng}: preproc={tuple(v.shape[-3:])} viT_grid={tuple(f.shape[-3:])}")
PY
```

Expected: `native_grid=False: preproc=(192, 192, 192) viT_grid=(24, 24, 24)` and
`native_grid=True: preproc=(128, 128, 128) viT_grid=(16, 16, 16)`.

- [ ] **Step 5: Run the full test suite for this file + a regression check**

Run: `.venv_thor/bin/pytest tests/test_primus_native_grid.py -v`
Expected: PASS (unchanged from Task 1).

- [ ] **Step 6: Commit**

```bash
git add src/models/primus_encoder.py
git commit -m "feat(primus): native_grid mode honors image_size in ViT token grid

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Thread the flag through PatchSet3D, train.py, config + docs

**Files:**
- Modify: `src/models/patchset3d.py` (`__init__` signature + `PrimusEncoder(...)` call, ~lines 90-132)
- Modify: `experiments/3d/train.py` (arch dict, ~lines 244-256)
- Modify: `configs/experiment/3d/model/patchset3d.yaml` (add `arch.encoder_native_grid: false`)
- Modify: `docs/logs.md`

**Interfaces:**
- Consumes: `PrimusEncoder(..., native_grid=...)` from Task 2.
- Produces: `arch.encoder_native_grid` (bool, default false) as a first-class config knob; `PatchSet3D(encoder_native_grid=False)`.

- [ ] **Step 1: Add the param to `PatchSet3D.__init__`**

In `src/models/patchset3d.py`, add to the signature after `encoder_stage: int = None,`:

```python
        encoder_native_grid: bool = False,
```

Change the `PrimusEncoder(...)` construction to pass it:

```python
            self.encoder = PrimusEncoder(primus_sidecar, resolution,
                                         frozen=encoder_frozen, device="cpu",
                                         encoder_stage=encoder_stage,
                                         native_grid=encoder_native_grid)
```

- [ ] **Step 2: Add the key to the train.py arch dict**

In `experiments/3d/train.py`, in the arch dict (after the `"encoder_stage": a.get("encoder_stage", None),` line):

```python
            "encoder_native_grid": a.get("encoder_native_grid", False),
```

- [ ] **Step 3: Add the schema default to the model config**

In `configs/experiment/3d/model/patchset3d.yaml`, under the `arch:` block, add (near `encoder_frozen` / `primus_sidecar` if present, else anywhere in `arch:`):

```yaml
  encoder_native_grid: false   # native_grid: ViT tokenizes at image_size/8 (encoder FLOPs
                               # scale with data.image_size). false = fixed 192^3 resample.
```

- [ ] **Step 4: Verify config composition + FLOPs actually move**

Confirm the key composes and is overridable, and that FLOPs now scale. Run:

```bash
.venv_thor/bin/python experiments/3d/train.py experiment=30_colipri_encoder \
    arch.encoder_native_grid=true data.image_size=[128,128,128] \
    train.epochs=0 train.wandb.project=null 2>&1 | grep -i "gflops\|GFLOP\|size="
```

Expected: prints a GFLOPs value **lower** than the 192³ baseline. Compare against baseline:

```bash
.venv_thor/bin/python experiments/3d/train.py experiment=30_colipri_encoder \
    train.epochs=0 train.wandb.project=null 2>&1 | grep -i "gflops\|GFLOP\|size="
```

Expected: baseline GFLOPs ≈ the original ~192-region value; the native-grid 128³ run is strictly smaller. (If `train.epochs=0` isn't supported by the loop, use `data.max_ds_len_train=1` and Ctrl-C after the GFLOPs line prints — the FLOPs log happens at startup, `train.py:483-486`.)

- [ ] **Step 5: Log the change in docs/logs.md**

Add a dated entry to `docs/logs.md` describing: PrimusEncoder gained an opt-in `arch.encoder_native_grid` flag; default false; when true the frozen ViT tokenizes at `image_size/8` with an identity-RoPE rebuild, so encoder FLOPs scale with `data.image_size` (transformer/decode unchanged via `_down_to`). Reference the design doc path.

- [ ] **Step 6: Commit**

```bash
git add src/models/patchset3d.py experiments/3d/train.py \
    configs/experiment/3d/model/patchset3d.yaml docs/logs.md
git commit -m "feat(patchset3d): arch.encoder_native_grid flag scales encoder FLOPs with image_size

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-Review

- **Spec coverage:** `_preprocess` change (Task 2 Step 2), RoPE rebuild (Task 2 Step 3), identity ref (Task 1 helper), flag default-off wiring (Task 3), docs (Task 3 Step 5), verification that FLOPs move (Task 3 Step 4) + grid shape (Task 2 Step 4). All spec sections mapped.
- **Placeholder scan:** none — all code shown inline.
- **Type consistency:** `_native_target_shape`/`_set_rope_identity_grid` names and `native_grid`/`encoder_native_grid` kwarg names consistent across Tasks 1-3.
