# Cascade Training for PatchSet3D Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train PatchSet3D in an N-level coarse→fine cascade — predict at spacing `s_i`, re-crop the target on the predicted centre-of-mass, predict at `s_{i+1}`, one weighted loss per level, single backward — on the v2 dataloader path, with geometric augmentation replayed identically across levels.

**Architecture:** A new `experiments/3d/cascade.py` owns the level loop (`run_cascade`) and the v2 cascade val pass (`evaluate_cascade`); `PatchSet3D.forward` stays single-level. Levels ≥1 are re-cropped **in the train loop** by calling `TotalSegProvider.load(..., center=predicted_COM, spacing=s_i)` directly (exact eval semantics — accepted synchronous I/O for a first run). `GpuAugmentor` gains an `apply()` that takes injected RNG generators and can capture the composed affine+elastic+deform sampling grid + flip record, so the predicted COM can be mapped back through it to native voxels and the same geometry replayed at every level.

**Tech Stack:** PyTorch, Hydra/OmegaConf, nnU-Net encoders, pytest. Python env: `.venv_blackwell` (odin/Blackwell) — `python -m pytest` and `python` resolve to it on PATH. `git` lives at `/software/anaconda3/envs/git/bin/git`; prepend `export PATH="/software/anaconda3/envs/git/bin:$PATH"` in any shell step that calls git.

**Spec:** `docs/superpowers/specs/2026-08-30-cascade-training-patchset3d-design.md` — read it alongside this plan.

## Global Constraints

- **v2 only**: cascade requires `model=patchset3d`, `data.loader_v2=true`, `data.source in _TOTALSEG_SOURCES` (`experiments/3d/common.py`). No v1, no medverse, no synth/self-context aug modes.
- **Config keys** (exact names): `data.cascade_spacings` (list[float] | null, null = OFF), `data.cascade_crop_jitter` (int, default `0`), `train.cascade_loss_weights` (list[float], default all `1.0`, length == `len(cascade_spacings)`).
- `data.cascade_spacings[0]` **must equal** `data.crop_spacing_mm` (level-0 geometry).
- `data.cascade_spacings` and `data.train_spacing_range` are **mutually exclusive**.
- `len(cascade_spacings) >= 2`.
- **Metric keys** are per-spacing, not per-level-index: `train/loss_r{s:g}`, `train/dice_r{s:g}`, `val/dice_r{s:g}/<class>`, `val/dice_r{s:g}` (macro), plus `val/dice_stitched` (macro, coarse→fine composite of **all** levels, each overwriting the previous). `val/dice` (checkpoint-selection) == `val/dice_stitched` when cascade is on. `train/dice` and grid metrics stay at the **finest** level.
- **Non-cascade path must stay byte-identical.** Every change is behind a `cfg.data.get("cascade_spacings")` check or a defaulted-off kwarg. `GpuAugmentor.__call__` keeps its current signature and behaviour.
- Tests live in `experiments/3d/tests/`. Each test file self-inserts repo root: `ROOT = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(ROOT))`. For imports of `experiments/3d` siblings (`cascade`, `common`, `eval`): also `sys.path.insert(0, str(Path(__file__).resolve().parents[1]))`.
- Run tests from repo root: `python -m pytest experiments/3d/tests/<file>.py -v`.
- Commit messages end with the two trailers used in this repo:
  ```
  Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01CuANpa5naaN7SJLDf5w9Db
  ```
- Log the finished feature in `docs/logs.md` (Task 9).

---

## File Structure

| File | Responsibility | Tasks |
| --- | --- | --- |
| `src/incontext_dataset_v2.py` | `LoadRequest` gains optional `jitter` field | 1 |
| `src/providers/totalseg.py` | `load()` resolves `req.jitter` vs `self.crop_jitter` via `_resolve_jitter` | 1 |
| `src/gpu_augment.py` | `_geometric` captures `(grid, flips)`; `GeoState`; `GpuAugmentor.apply(...)` with injected generators; `__call__` unchanged | 2, 3 |
| `experiments/3d/cascade.py` | **new** — `invert_geo_center`, `CascadeResult`, `run_cascade`, `_cascade_loss`, `_stitched_native_dice_multi`, `evaluate_cascade` | 4, 5, 7, 8 |
| `experiments/3d/common.py` | `_assert_cascade_supported(cfg)`; call it in `train_loader` / `make_eval_loader` | 6 |
| `experiments/3d/train.py` | `main` calls the guard + always builds `GpuAugmentor` when cascade on; `train_epoch` cascade branch; `validate_mean` cascade branch | 6, 7, 8 |
| `experiments/3d/evaluate.py` | `_stitched_native_dice` generalised to a list (keep 2-arg wrapper) | 8 |
| `configs/experiment/3d/experiment/59_organs_cascade_from_scratch.yaml` | **new** experiment config | 9 |
| `docs/logs.md` | changelog entry | 9 |
| `experiments/3d/tests/test_cascade_provider.py` | `_resolve_jitter` + `LoadRequest.jitter` | 1 |
| `experiments/3d/tests/test_gpu_augment_capture.py` | `_geometric` capture, `GpuAugmentor.apply` replay + `__call__` parity | 2, 3 |
| `experiments/3d/tests/test_cascade.py` | `invert_geo_center`, `run_cascade`, `_cascade_loss` | 4, 5, 7 |
| `experiments/3d/tests/test_cascade_guard.py` | `_assert_cascade_supported` | 6 |
| `experiments/3d/tests/test_cascade_stitch.py` | `_stitched_native_dice_multi` | 8 |
| `experiments/3d/tests/test_cascade_config.py` | `experiment=59_*` resolves + guard passes | 9 |

---

## Task 1: `LoadRequest.jitter` + provider jitter resolution

**Files:**
- Modify: `src/incontext_dataset_v2.py` (the `LoadRequest` dataclass, ~lines 19-25)
- Modify: `src/providers/totalseg.py` (`TotalSegProvider.load`, ~lines 132-168; add module-level `_resolve_jitter`)
- Test: `experiments/3d/tests/test_cascade_provider.py`

**Interfaces:**
- Produces: `LoadRequest(rng, crop_spacing_mm, center=None, jitter=None)` — new optional `jitter: Optional[int]`. `None` = use the provider default; an int overrides `self.crop_jitter` for that one load.
- Produces: `src.providers.totalseg._resolve_jitter(req: LoadRequest, default: int) -> int` — returns `int(req.jitter)` when `req.jitter is not None`, else `int(default)`.

- [ ] **Step 1: Write the failing test**

Create `experiments/3d/tests/test_cascade_provider.py`:

```python
"""Task 1: LoadRequest.jitter field + provider jitter resolution."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import random

from src.incontext_dataset_v2 import LoadRequest
from src.providers.totalseg import _resolve_jitter


def _req(jitter=None):
    return LoadRequest(rng=random.Random(0), crop_spacing_mm=1.5, jitter=jitter)


def test_loadrequest_jitter_defaults_none():
    assert _req().jitter is None


def test_loadrequest_jitter_set():
    assert _req(jitter=0).jitter == 0
    assert _req(jitter=7).jitter == 7


def test_resolve_jitter_prefers_request():
    assert _resolve_jitter(_req(jitter=0), default=12) == 0
    assert _resolve_jitter(_req(jitter=3), default=12) == 3


def test_resolve_jitter_falls_back_to_default():
    assert _resolve_jitter(_req(jitter=None), default=12) == 12
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest experiments/3d/tests/test_cascade_provider.py -v`
Expected: FAIL — `ImportError: cannot import name '_resolve_jitter'` (and `TypeError` on the `jitter=` kwarg).

- [ ] **Step 3: Add the `jitter` field**

In `src/incontext_dataset_v2.py`, the `LoadRequest` dataclass currently reads:

```python
@dataclass
class LoadRequest:
    rng: random.Random                 # per-item RNG (eval determinism or global)
    crop_spacing_mm: float             # physical crop pitch for THIS item
    center: Optional[tuple] = None     # native-voxel crop center; None -> provider default
                                       # (cascade fine-crop seam; v2 always passes None)
```

Add one field:

```python
@dataclass
class LoadRequest:
    rng: random.Random                 # per-item RNG (eval determinism or global)
    crop_spacing_mm: float             # physical crop pitch for THIS item
    center: Optional[tuple] = None     # native-voxel crop center; None -> provider default
                                       # (cascade fine-crop seam)
    jitter: Optional[int] = None       # per-load crop-jitter override (native voxels);
                                       # None -> provider default self.crop_jitter.
                                       # Cascade re-crops pass 0 so the predicted COM is exact.
```

- [ ] **Step 4: Add `_resolve_jitter` and use it in `load()`**

In `src/providers/totalseg.py`, add a module-level helper just above `class TotalSegProvider`:

```python
def _resolve_jitter(req: LoadRequest, default: int) -> int:
    """Per-load crop-jitter: req.jitter when set, else the provider default."""
    return int(req.jitter) if req.jitter is not None else int(default)
```

In `TotalSegProvider.load`, after `center = req.center` / the `center is None` block, compute the jitter once and pass it into both crop paths (replace the two `jitter=self.crop_jitter` arguments):

```python
        jitter = _resolve_jitter(req, self.crop_jitter)
        ...
        if cache_p.exists():
            img_cache_np = np.load(cache_p, mmap_mode="r")
            image_t, label_t, geom = crop_and_place_cached(
                img_cache_np, label_np, _ALL_CLASSES_IDX.get(cls, -1), center, self.T,
                crop_spacing_mm=req.crop_spacing_mm, native_spacing=native_sp,
                cache_spacing_mm=float(req.crop_spacing_mm),
                jitter=jitter, rng=req.rng,
                mask_downsample=self.mask_downsample, occ_thr=self.mask_occupancy_thr,
                normalize_fn=lambda a: norm(np.ascontiguousarray(a)))
        else:
            ...
            image_t, label_t, geom = crop_and_place(
                image_np, label_np, _ALL_CLASSES_IDX.get(cls, -1), center, self.T,
                crop_spacing_mm=req.crop_spacing_mm, native_spacing=native_sp,
                jitter=jitter, rng=req.rng,
                mask_downsample=self.mask_downsample, occ_thr=self.mask_occupancy_thr,
                normalize_fn=lambda a: norm(np.ascontiguousarray(a)))
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest experiments/3d/tests/test_cascade_provider.py -v`
Expected: PASS (4 tests).

- [ ] **Step 6: Regression — v2 dataloader tests still pass**

Run: `python -m pytest experiments/3d/tests/test_crop_helpers.py -v`
Expected: PASS (unchanged — `jitter` default `None` preserves every existing call).

- [ ] **Step 7: Commit**

```bash
export PATH="/software/anaconda3/envs/git/bin:$PATH"
git add src/incontext_dataset_v2.py src/providers/totalseg.py experiments/3d/tests/test_cascade_provider.py
git commit -m "feat(cascade): LoadRequest.jitter override + provider resolution

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01CuANpa5naaN7SJLDf5w9Db"
```

---

## Task 2: `_geometric` captures the composed grid + flip record

**Files:**
- Modify: `src/gpu_augment.py` — `_geometric` (lines 241-310)
- Test: `experiments/3d/tests/test_gpu_augment_capture.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `_geometric(vols, masks, group_size, cfg, gen, *, capture=False)` — when `capture=False` returns `(vols, masks)` exactly as today; when `capture=True` returns `(vols, masks, grid, flips)` where `grid` is a `(N, D, H, W, 3)` **float32** tensor (the affine+elastic+deform sampling grid, `grid_sample` xyz convention, values in ~[-1,1]) captured immediately before `grid_sample`, and `flips` is a `(N, 3)` **bool** tensor recording the per-volume axis flips applied (axis order `(D, H, W)`).

- [ ] **Step 1: Write the failing test**

Create `experiments/3d/tests/test_gpu_augment_capture.py`:

```python
"""Tasks 2-3: GpuAugmentor grid/flip capture + injected-generator replay."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import torch
from omegaconf import OmegaConf

from src.gpu_augment import _geometric, GpuAugmentor, GeoState


def _task_cfg(flip_p=0.5, affine_p=1.0, deform_p=0.0, elastic_p=0.0):
    return OmegaConf.create({
        "flip": {"p_d": flip_p, "p_h": flip_p, "p_w": flip_p},
        "affine": {"p": affine_p, "max_angle_deg": 20.0, "scale_min": 0.9,
                   "scale_max": 1.1, "max_translate": 0.1},
        "elastic": {"p": elastic_p, "alpha": 0.1, "grid_scale": 8},
        "deform": {"p": deform_p, "control_points": 4, "max_disp": 0.1, "num_steps": 4},
        "mask_interp": "bilinear",
    })


def test_geometric_capture_shapes():
    N, T = 4, 8
    vols = torch.randn(N, 1, T, T, T)
    masks = torch.zeros(N, T, T, T)
    g = torch.Generator().manual_seed(0)
    out = _geometric(vols, masks, group_size=N, cfg=_task_cfg(deform_p=1.0), gen=g, capture=True)
    assert len(out) == 4
    _v, _m, grid, flips = out
    assert grid.shape == (N, T, T, T, 3) and grid.dtype == torch.float32
    assert flips.shape == (N, 3) and flips.dtype == torch.bool


def test_geometric_no_capture_is_two_tuple():
    N, T = 2, 8
    vols = torch.randn(N, 1, T, T, T)
    masks = torch.zeros(N, T, T, T)
    g = torch.Generator().manual_seed(0)
    out = _geometric(vols, masks, group_size=N, cfg=_task_cfg(), gen=g)
    assert len(out) == 2


def test_geometric_same_seed_same_transform():
    # Same generator seed + same shapes -> identical grid and flips, on different content.
    N, T = 3, 8
    cfg = _task_cfg(deform_p=1.0)
    a = torch.randn(N, 1, T, T, T)
    b = torch.randn(N, 1, T, T, T)
    m = torch.zeros(N, T, T, T)
    g1 = torch.Generator().manual_seed(42)
    g2 = torch.Generator().manual_seed(42)
    _, _, grid_a, flip_a = _geometric(a, m.clone(), N, cfg, g1, capture=True)
    _, _, grid_b, flip_b = _geometric(b, m.clone(), N, cfg, g2, capture=True)
    assert torch.equal(flip_a, flip_b)
    assert torch.allclose(grid_a, grid_b, atol=1e-6)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest experiments/3d/tests/test_gpu_augment_capture.py -v`
Expected: FAIL — `ImportError: cannot import name 'GeoState'` and `_geometric() got an unexpected keyword argument 'capture'`.

- [ ] **Step 3: Add `GeoState` and rework `_geometric`**

In `src/gpu_augment.py`, add near the top (after the imports / `REAL, SYNTH, SELF_CONTEXT` line):

```python
from dataclasses import dataclass


@dataclass
class GeoState:
    """Captured geometry of one _geometric() call, for cascade COM inversion + replay.

    grid  : (N, D, H, W, 3) float32 sampling grid (affine+elastic+deform composed,
            grid_sample xyz convention) captured just before grid_sample. None when
            not captured / no augmentor.
    flips : (N, 3) bool — per-volume axis flips applied before the warp (D, H, W order).
    """
    grid: "torch.Tensor | None"
    flips: "torch.Tensor"
```

Rewrite `_geometric`'s signature and body. The current function:

```python
def _geometric(vols, masks, group_size, cfg, gen):
    """Shared (group_size=T) or independent (group_size=1) flip/affine/elastic."""
    N = vols.shape[0]
    device = vols.device
    G = N // group_size                              # number of groups
    assert N % group_size == 0, f"N={N} must be divisible by group_size={group_size}"
    D, H, W = vols.shape[-3:]

    # --- flips: one decision per group, per axis ---
    fp = cfg.flip
    for g in range(G):
        sl = slice(g * group_size, (g + 1) * group_size)
        for vol_dim, mask_dim, p in [(2, 1, fp.p_d), (3, 2, fp.p_h), (4, 3, fp.p_w)]:
            if _rand(gen, device, 1).item() < p:
                vols[sl] = vols[sl].flip(vol_dim)
                masks[sl] = masks[sl].flip(mask_dim)
```

Change to:

```python
def _geometric(vols, masks, group_size, cfg, gen, *, capture=False):
    """Shared (group_size=T) or independent (group_size=1) flip/affine/elastic/deform.

    capture=True additionally returns (grid, flips): the composed sampling grid (just
    before grid_sample) and the per-volume flip record, for cascade COM inversion + replay.
    """
    N = vols.shape[0]
    device = vols.device
    G = N // group_size                              # number of groups
    assert N % group_size == 0, f"N={N} must be divisible by group_size={group_size}"
    D, H, W = vols.shape[-3:]

    # --- flips: one decision per group, per axis (ax_i 0,1,2 -> D,H,W) ---
    fp = cfg.flip
    flip_rec = torch.zeros(G, 3, dtype=torch.bool, device=device)
    for g in range(G):
        sl = slice(g * group_size, (g + 1) * group_size)
        for ax_i, (vol_dim, mask_dim, p) in enumerate(
                [(2, 1, fp.p_d), (3, 2, fp.p_h), (4, 3, fp.p_w)]):
            if _rand(gen, device, 1).item() < p:
                vols[sl] = vols[sl].flip(vol_dim)
                masks[sl] = masks[sl].flip(mask_dim)
                flip_rec[g, ax_i] = True
```

Then, immediately **before** the line `grid = grid.to(vols.dtype)` near the end of the function, insert the capture:

```python
    captured_grid = grid.detach().float().clone() if capture else None
    grid = grid.to(vols.dtype)
```

Finally change the `return` at the end of the function from:

```python
    return vols, (m.clamp(0.0, 1.0) if mask_soft else m.long())
```

to:

```python
    out_masks = m.clamp(0.0, 1.0) if mask_soft else m.long()
    if capture:
        return vols, out_masks, captured_grid, flip_rec.repeat_interleave(group_size, dim=0)
    return vols, out_masks
```

(Leave every other line of `_geometric` untouched. `GpuAugmentor.__call__` calls `_geometric(...)` without `capture`, so it still unpacks a 2-tuple.)

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest experiments/3d/tests/test_gpu_augment_capture.py -v`
Expected: the 3 tests in this file that don't touch `GpuAugmentor.apply` pass:
`test_geometric_capture_shapes`, `test_geometric_no_capture_is_two_tuple`, `test_geometric_same_seed_same_transform`.
(`GpuAugmentor.apply` tests are added in Task 3 — they will error with `AttributeError` for now; that is expected and handled next task. Run with `-k geometric` to scope: `python -m pytest experiments/3d/tests/test_gpu_augment_capture.py -k geometric -v`.)

- [ ] **Step 5: Regression — GPU aug pipeline tests**

Run: `python -m pytest experiments/3d/tests/ -k "aug or augment" -v`
Expected: PASS (any existing aug tests unchanged — `_geometric` default path is untouched).

- [ ] **Step 6: Commit**

```bash
export PATH="/software/anaconda3/envs/git/bin:$PATH"
git add src/gpu_augment.py experiments/3d/tests/test_gpu_augment_capture.py
git commit -m "feat(cascade): _geometric captures composed grid + flip record

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01CuANpa5naaN7SJLDf5w9Db"
```

---

## Task 3: `GpuAugmentor.apply` — injected generators + replay

**Files:**
- Modify: `src/gpu_augment.py` — add `GpuAugmentor.apply`
- Test: `experiments/3d/tests/test_gpu_augment_capture.py` (add cases)

**Interfaces:**
- Consumes: `_geometric(..., capture=)`, `GeoState` (Task 2).
- Produces: `GpuAugmentor.apply(batch: dict, *, geo_gen: torch.Generator, int_gen: torch.Generator, capture: bool = False) -> tuple[dict, GeoState | None]`.
  Runs the REAL-mode aug body (shared geometric over target+K contexts, then per-volume intensity) on `batch` **in place** (tensors already on device), using `geo_gen` for geometry and `int_gen` for intensity. Returns the same `batch` dict and a `GeoState` when `capture=True` (else `None`). Assumes every task in the batch is REAL (`aug_mode == 0`); the caller (`run_cascade`) asserts that.

- [ ] **Step 1: Write the failing test (add to `test_gpu_augment_capture.py`)**

Append:

```python
def _full_cfg():
    return OmegaConf.create({
        "enabled": True, "gpu": True,
        "task": {
            "flip": {"p_d": 0.5, "p_h": 0.5, "p_w": 0.5},
            "affine": {"p": 1.0, "max_angle_deg": 20.0, "scale_min": 0.9,
                       "scale_max": 1.1, "max_translate": 0.1},
            "elastic": {"p": 0.0, "alpha": 0.1, "grid_scale": 8},
            "deform": {"p": 1.0, "control_points": 4, "max_disp": 0.1, "num_steps": 4},
            "mask_interp": "bilinear",
        },
        "intensity": {
            "brightness_contrast": {"p": 0.5, "brightness": 0.0,
                                    "contrast_range": [0.8, 1.2], "preserve_range": True},
        },
    })


def _fake_batch(B=2, K=3, T=8):
    return {
        "image": torch.randn(B, 1, T, T, T),
        "label": torch.zeros(B, T, T, T),
        "context_in": torch.randn(B, K, 1, T, T, T),
        "context_out": torch.zeros(B, K, T, T, T),
        "aug_mode": torch.zeros(B, dtype=torch.long),
    }


def test_apply_returns_geostate_on_capture():
    aug = GpuAugmentor(_full_cfg())
    b = _fake_batch()
    _, geo = aug.apply(b, geo_gen=torch.Generator().manual_seed(1),
                       int_gen=torch.Generator().manual_seed(2), capture=True)
    assert isinstance(geo, GeoState)
    assert geo.grid.shape == (2 * 4, 8, 8, 8, 3)   # B*T
    assert geo.flips.shape == (2 * 4, 3)


def test_apply_replay_same_geo_seed_matches_geometry():
    # Two batches, same geo_gen seed -> identical captured grid + flips (geometry replay),
    # even with different intensity seeds and content.
    aug = GpuAugmentor(_full_cfg())
    b0, b1 = _fake_batch(), _fake_batch()
    _, g0 = aug.apply(b0, geo_gen=torch.Generator().manual_seed(7),
                      int_gen=torch.Generator().manual_seed(100), capture=True)
    _, g1 = aug.apply(b1, geo_gen=torch.Generator().manual_seed(7),
                      int_gen=torch.Generator().manual_seed(200), capture=True)
    assert torch.equal(g0.flips, g1.flips)
    assert torch.allclose(g0.grid, g1.grid, atol=1e-6)


def test_call_path_unchanged_byte_identical():
    # GpuAugmentor.__call__ must be unaffected by the apply() addition.
    cfg = _full_cfg()
    a = GpuAugmentor(cfg, seed=0)
    b = GpuAugmentor(cfg, seed=0)
    batch_a, batch_b = _fake_batch(B=2), None
    torch.manual_seed(0)
    batch_b = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch_a.items()}
    out_a = a(batch_a, training=True)
    out_b = b(batch_b, training=True)
    assert torch.allclose(out_a["image"], out_b["image"], atol=1e-6)
    assert torch.allclose(out_a["context_in"], out_b["context_in"], atol=1e-6)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest experiments/3d/tests/test_gpu_augment_capture.py -k apply -v`
Expected: FAIL — `AttributeError: 'GpuAugmentor' object has no attribute 'apply'`.

- [ ] **Step 3: Implement `GpuAugmentor.apply`**

In `src/gpu_augment.py`, add this method to `class GpuAugmentor` (place it right after `__init__`, before `__call__`):

```python
    @torch.no_grad()
    def apply(self, batch: dict, *, geo_gen: torch.Generator,
              int_gen: torch.Generator, capture: bool = False):
        """REAL-mode aug for the cascade runner: shared geometric over target+K contexts
        (geo_gen), then per-volume intensity (int_gen). Mutates `batch` in place; returns
        (batch, GeoState|None). Every task is assumed REAL (aug_mode==0) — run_cascade
        asserts it. Kept separate from __call__ so the non-cascade path stays byte-identical.
        """
        cfg = self.cfg
        vols, masks, B, T = _stack_task(batch)          # tensors already on device
        if capture:
            vols, masks, grid, flips = _geometric(
                vols, masks, group_size=T, cfg=cfg.task, gen=geo_gen, capture=True)
        else:
            vols, masks = _geometric(vols, masks, group_size=T, cfg=cfg.task, gen=geo_gen)
            grid = flips = None
        vols = _batched_intensity(vols, cfg.intensity, int_gen)
        _unstack_task(vols, masks, B, T, batch)
        return batch, (GeoState(grid=grid, flips=flips) if capture else None)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest experiments/3d/tests/test_gpu_augment_capture.py -v`
Expected: PASS (all tests in the file, Task 2 + Task 3).

- [ ] **Step 5: Commit**

```bash
export PATH="/software/anaconda3/envs/git/bin:$PATH"
git add src/gpu_augment.py experiments/3d/tests/test_gpu_augment_capture.py
git commit -m "feat(cascade): GpuAugmentor.apply with injected generators + geometry replay

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01CuANpa5naaN7SJLDf5w9Db"
```

---

## Task 4: `invert_geo_center` — map an augmented-grid centroid to native voxels

**Files:**
- Create: `experiments/3d/cascade.py`
- Test: `experiments/3d/tests/test_cascade.py`

**Interfaces:**
- Consumes: `GeoState` (`src.gpu_augment`); `_grid_centroid`, `_predicted_native_center` (`experiments/3d/evaluate.py`).
- Produces:
  - `experiments/3d/cascade.py` module.
  - `invert_geo_center(centroid_dhw, grid_row, flips_row, crop_geom_row, T) -> tuple[int, int, int] | None`.
    `centroid_dhw`: a length-3 sequence `(d, h, w)` in the level's **augmented** `T³` grid, or `None` (empty prob) → returns `None`. `grid_row`: `(T, T, T, 3)` float tensor or `None` (no warp). `flips_row`: `(3,)` bool tensor (D,H,W). `crop_geom_row`: `(4, 3)` long tensor `[starts, crop_sizes, out_sizes, pad_lo]`. Returns native voxel centre `(d, h, w)`, each `>= 0`. When `grid_row is None` and `flips_row` is all-False the result equals `_predicted_native_center` for the same centroid.

- [ ] **Step 1: Write the failing test**

Create `experiments/3d/tests/test_cascade.py`:

```python
"""Tasks 4-5,7: cascade.py — invert_geo_center, run_cascade, _cascade_loss."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # experiments/3d siblings

import numpy as np
import torch

from cascade import invert_geo_center
from evaluate import _predicted_native_center, _grid_centroid


def _geom(starts=(10, 20, 30), crop=(8, 8, 8), out=(8, 8, 8), pad=(0, 0, 0)):
    return torch.tensor([list(starts), list(crop), list(out), list(pad)], dtype=torch.long)


def _prob_blob(T=8, c=(4, 4, 4)):
    p = np.zeros((T, T, T), dtype=np.float32)
    p[c[0], c[1], c[2]] = 1.0
    return p


def test_identity_matches_predicted_native_center():
    T = 8
    prob = _prob_blob(T, c=(5, 3, 6))
    geom = _geom()
    cen = _grid_centroid(prob)                       # np array (d,h,w)
    got = invert_geo_center(cen, None, torch.zeros(3, dtype=torch.bool), geom, T)
    want = _predicted_native_center(torch.from_numpy(prob), geom)
    assert got == want


def test_empty_centroid_returns_none():
    assert invert_geo_center(None, None, torch.zeros(3, dtype=torch.bool), _geom(), 8) is None


def test_flip_mirrors_the_centroid():
    T = 8
    geom = _geom(starts=(0, 0, 0), crop=(8, 8, 8), out=(8, 8, 8), pad=(0, 0, 0))
    cen = np.array([2.0, 1.0, 7.0])                  # d,h,w
    flips = torch.tensor([True, False, True])        # flip d and w
    got = invert_geo_center(cen, None, flips, geom, T)
    # unflip: d -> (T-1)-2 = 5, w -> (T-1)-7 = 0 ; native == pre-aug grid here (identity geom)
    assert got == (5, 1, 0)


def test_grid_shift_maps_through():
    # A constant grid that maps every output voxel to the volume centre in normalized coords
    # (0,0,0) -> pre-aug voxel ((T-1)/2). Identity geom -> native == (T-1)/2 per axis.
    T = 8
    grid_row = torch.zeros(T, T, T, 3)               # all (x,y,z) = 0 -> centre
    geom = _geom(starts=(0, 0, 0), crop=(8, 8, 8), out=(8, 8, 8), pad=(0, 0, 0))
    got = invert_geo_center(np.array([1.0, 2.0, 3.0]), grid_row,
                            torch.zeros(3, dtype=torch.bool), geom, T)
    mid = round((T - 1) / 2)
    assert got == (mid, mid, mid)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest experiments/3d/tests/test_cascade.py -k invert -v` (and the identity/empty/flip/grid tests)
Expected: FAIL — `ModuleNotFoundError: No module named 'cascade'`.

- [ ] **Step 3: Create `experiments/3d/cascade.py` with `invert_geo_center`**

```python
"""N-level coarse->fine cascade for PatchSet3D (v2 pipeline).

run_cascade executes one N-level forward (level 0 = GT-centred, level i>0 = target
re-cropped on level i-1's predicted centre-of-mass); shared by the train loop
(experiments/3d/train.py train_epoch) and the v2 cascade val pass (evaluate_cascade).
PatchSet3D.forward stays single-level.

See docs/superpowers/specs/2026-08-30-cascade-training-patchset3d-design.md.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from src.gpu_augment import GeoState
from src.incontext_dataset_v2 import LoadRequest
from src.totalseg_dataloader_incontext import incontext_collate_fn
from grid_metrics import target_like
from evaluate import _grid_centroid, _predicted_native_center


def invert_geo_center(centroid_dhw, grid_row, flips_row, crop_geom_row, T):
    """Map a centroid in a level's AUGMENTED T^3 grid back to a native crop centre.

    centroid_dhw : length-3 (d,h,w) in the augmented grid, or None (empty prob) -> None.
    grid_row     : (T,T,T,3) float sampling grid (grid_sample xyz convention) or None.
    flips_row    : (3,) bool, per-axis flip (D,H,W order) applied before the warp.
    crop_geom_row: (4,3) long [starts, crop_sizes, out_sizes, pad_lo].
    Returns native voxel (d,h,w), each >= 0. Identity (grid_row None, no flips) reproduces
    evaluate._predicted_native_center for the same centroid.
    """
    if centroid_dhw is None:
        return None
    g = [float(centroid_dhw[a]) for a in range(3)]                     # d,h,w (augmented)

    flips = [bool(x) for x in (flips_row.tolist() if torch.is_tensor(flips_row) else flips_row)]
    for a in range(3):
        if flips[a]:
            g[a] = (T - 1) - g[a]

    if grid_row is not None:
        # Interpolate the (T,T,T,3) grid at the fractional post-unflip coord. Query point in
        # grid_sample xyz order = (w, h, d) normalized with the align_corners=True pairing.
        q = torch.tensor(
            [[[[[2.0 * g[2] / max(1, T - 1) - 1.0,
                 2.0 * g[1] / max(1, T - 1) - 1.0,
                 2.0 * g[0] / max(1, T - 1) - 1.0]]]]],
            dtype=torch.float32)                                       # (1,1,1,1,3)
        field = grid_row.detach().float().permute(3, 0, 1, 2).unsqueeze(0)  # (1,3,T,T,T)
        pre = F.grid_sample(field, q, mode="bilinear", padding_mode="border",
                            align_corners=True)[0, :, 0, 0, 0]         # (3,) = (x,y,z) norm
        x, y, z = (float(v) for v in pre)
        g = [(z + 1.0) / 2.0 * (T - 1),                               # d
             (y + 1.0) / 2.0 * (T - 1),                               # h
             (x + 1.0) / 2.0 * (T - 1)]                               # w

    starts, crop_sizes, out_sizes, pad_lo = (crop_geom_row[r].tolist() for r in range(4))
    native = [int(round(starts[a] + (g[a] - pad_lo[a]) / max(1, out_sizes[a]) * crop_sizes[a]))
              for a in range(3)]
    return tuple(max(0, c) for c in native)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest experiments/3d/tests/test_cascade.py -k "identity or empty or flip or grid_shift" -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
export PATH="/software/anaconda3/envs/git/bin:$PATH"
git add experiments/3d/cascade.py experiments/3d/tests/test_cascade.py
git commit -m "feat(cascade): invert_geo_center — augmented centroid -> native crop centre

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01CuANpa5naaN7SJLDf5w9Db"
```

---

## Task 5: `run_cascade` + `CascadeResult`

**Files:**
- Modify: `experiments/3d/cascade.py` — add `CascadeResult`, `_gen`, `_centroid_from_logit`, `run_cascade`
- Test: `experiments/3d/tests/test_cascade.py` (add cases with fake model + fake provider)

**Interfaces:**
- Consumes: `invert_geo_center` (Task 4); `GpuAugmentor.apply` (Task 3); `LoadRequest` (Task 1); `incontext_collate_fn`, `target_like`.
- Produces:
  - `@dataclass CascadeResult`: `logits: list[torch.Tensor]` (per level, `(B,1,G,G,G)`), `targets: list[torch.Tensor]` (per level, grid GT from `target_like`), `geoms: list[torch.Tensor]` (per level, `(B,4,3)` target crop_geom), `centers: list[list]` (len N; `centers[0]` is `[None]*B`; `centers[i]` is the native COM tuple or `None` per b), `hard_preds: list[torch.Tensor] | None` (per level `(B, T, T, T)` binary; only when `want_hard_preds`), `empty_frac: float`.
  - `run_cascade(model, provider, batch, augmentor, spacings, *, device, training, step, seed, jitter=0, is_prob=False, want_hard_preds=False) -> CascadeResult`.
    `spacings`: list of `>=2` floats. `augmentor`: a `GpuAugmentor` or `None` (val / no-aug). `batch`: a v2 collated batch dict (`image`, `label`, `context_in`, `context_out`, `subjects`, `context_subjects`, `label_names`, `crop_geom`, `aug_mode`). Asserts `aug_mode` all `0`. Levels ≥1 are re-cropped via `provider.load` on the main thread.

- [ ] **Step 1: Write the failing test (append to `test_cascade.py`)**

```python
from dataclasses import dataclass as _dc

from cascade import run_cascade, CascadeResult, _cascade_loss
from src.incontext_dataset_v2 import LoadResult


class _FakeModel(torch.nn.Module):
    """Returns a fixed low-res logit with all mass at one grid cell; records call spacings."""
    spacing_aware = False

    def __init__(self, G=4, hot=(1, 1, 1)):
        super().__init__()
        self.G, self.hot, self.seen_spacing = G, hot, []
        self.p = torch.nn.Parameter(torch.zeros(1))

    def forward(self, image, context_in, context_out, mode="train", spacing=None):
        self.seen_spacing.append(spacing)
        B = image.shape[0]
        lg = torch.full((B, 1, self.G, self.G, self.G), -10.0)
        lg[:, :, self.hot[0], self.hot[1], self.hot[2]] = 10.0
        lg = lg + self.p                                  # keep autograd alive
        return {"final_logit": lg}


class _FakeProvider:
    """Records every load() call; returns synthetic T^3 crops."""
    def __init__(self, T=8):
        self.T, self.calls = T, []

    def load(self, subject, cls, req: LoadRequest):
        self.calls.append({"subject": subject, "cls": cls, "center": req.center,
                           "spacing": req.crop_spacing_mm, "jitter": req.jitter})
        T = self.T
        geom = torch.tensor([[0, 0, 0], [T, T, T], [T, T, T], [0, 0, 0]], dtype=torch.long)
        return LoadResult(image=torch.zeros(1, T, T, T), label=torch.zeros(T, T, T),
                          spacing=torch.full((3,), float(req.crop_spacing_mm)), crop_geom=geom)


def _v2_batch(B=2, K=3, T=8):
    geom = torch.tensor([[0, 0, 0], [T, T, T], [T, T, T], [0, 0, 0]], dtype=torch.long)
    return {
        "image": torch.zeros(B, 1, T, T, T),
        "label": torch.zeros(B, T, T, T),
        "context_in": torch.zeros(B, K, 1, T, T, T),
        "context_out": torch.zeros(B, K, T, T, T),
        "subjects": [f"s{b}" for b in range(B)],
        "context_subjects": [[f"c{b}_{k}" for k in range(K)] for b in range(B)],
        "label_names": ["liver"] * B,
        "crop_geom": geom.unsqueeze(0).repeat(B, 1, 1),
        "aug_mode": torch.zeros(B, dtype=torch.long),
    }


def test_run_cascade_two_levels_no_aug():
    B, T, G = 2, 8, 4
    model, prov = _FakeModel(G=G, hot=(1, 1, 1)), _FakeProvider(T=T)
    res = run_cascade(model, prov, _v2_batch(B=B, T=T), augmentor=None,
                      spacings=[3.0, 1.5], device=torch.device("cpu"),
                      training=True, step=0, seed=0, jitter=0)
    assert isinstance(res, CascadeResult)
    assert len(res.logits) == 2 and len(res.targets) == 2
    assert res.centers[0] == [None, None]
    # level-1 target loads: center == inverted COM (identity geom + hot cell 1/G -> native),
    # contexts loaded with center=None, spacing == 1.5, jitter == 0
    tgt_calls = [c for c in prov.calls if c["center"] is not None]
    ctx_calls = [c for c in prov.calls if c["center"] is None]
    assert len(tgt_calls) == B and all(c["spacing"] == 1.5 for c in tgt_calls)
    assert all(c["jitter"] == 0 for c in prov.calls)
    assert len(ctx_calls) == B * 3
    assert model.seen_spacing == [None] * (2)  # spacing_aware False -> None both levels
    # loss aggregation helper
    lf = lambda logit, target: (logit.float().mean() - target.float().mean()) ** 2
    total, per = _cascade_loss(res, lf, [1.0, 2.0])
    assert total.requires_grad and len(per) == 2


def test_run_cascade_empty_prob_falls_back_to_gt_centroid():
    B, T = 2, 8
    model = _FakeModel(G=4)
    # force an all-background logit -> empty prob -> center None
    model.forward = lambda image, context_in, context_out, mode="train", spacing=None: {
        "final_logit": torch.full((image.shape[0], 1, 4, 4, 4), -30.0) + model.p}
    prov = _FakeProvider(T=T)
    res = run_cascade(model, prov, _v2_batch(B=B, T=T), augmentor=None,
                      spacings=[3.0, 1.5], device=torch.device("cpu"),
                      training=True, step=1, seed=0)
    assert res.centers[1] == [None, None]
    assert res.empty_frac == 1.0
    assert all(c["center"] is None for c in prov.calls)  # every level-1 load GT-centred
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest experiments/3d/tests/test_cascade.py -k run_cascade -v`
Expected: FAIL — `ImportError: cannot import name 'run_cascade'`.

- [ ] **Step 3: Implement `CascadeResult`, helpers, and `run_cascade` in `cascade.py`**

Append to `experiments/3d/cascade.py`:

```python
INT_OFFSET = 1_000_000  # keep intensity seeds clear of geo seeds across plausible step counts


@dataclass
class CascadeResult:
    logits: list                  # per level: (B,1,G,G,G)
    targets: list                 # per level: grid GT (target_like)
    geoms: list                   # per level: (B,4,3) target crop_geom
    centers: list                 # len N: centers[0] == [None]*B; centers[i] native COM|None per b
    hard_preds: list | None       # per level: (B,T,T,T) binary; only when want_hard_preds
    empty_frac: float             # fraction of (level>=1, b) COM inversions that hit the fallback


def _gen(seed_int, device):
    g = torch.Generator(device=device)
    g.manual_seed(int(seed_int) & 0x7FFF_FFFF_FFFF_FFFF)
    return g


def _centroid_from_logit(logit_b1ghw, T, is_prob):
    """Per-b prob-weighted centroid (d,h,w) in the T^3 grid, or None when empty.

    logit upsampled to T^3 so the crop-geom affine (which assumes a T^3 prob) applies.
    """
    prob = logit_b1ghw.float().clamp(0, 1) if is_prob else torch.sigmoid(logit_b1ghw.float())
    up = F.interpolate(prob, size=(T, T, T), mode="trilinear", align_corners=False)
    out = []
    for b in range(up.shape[0]):
        out.append(_grid_centroid(up[b, 0].detach().cpu().numpy()))   # np(d,h,w) or None
    return out


def _to_device(batch, device):
    for k in ("image", "label", "context_in", "context_out"):
        batch[k] = batch[k].to(device, non_blocking=True)
    return batch


def _recrop_level(provider, batch, centers, spacing, *, step, seed, level, jitter):
    """Build one level-i v2 batch: target re-cropped on `centers[b]`, K contexts GT-centred,
    same subjects/classes as level 0. Runs on the calling thread (synchronous provider I/O)."""
    subs, ctxs, clss = batch["subjects"], batch["context_subjects"], batch["label_names"]
    items = []
    for b in range(len(subs)):
        tgt = provider.load(subs[b], clss[b], LoadRequest(
            rng=random.Random((seed, step, level, b)), crop_spacing_mm=float(spacing),
            center=centers[b], jitter=jitter))
        cin, cout = [], []
        for k, cs in enumerate(ctxs[b]):
            r = provider.load(cs, clss[b], LoadRequest(
                rng=random.Random((seed, step, level, b, k)), crop_spacing_mm=float(spacing),
                center=None, jitter=jitter))
            cin.append(r.image); cout.append(r.label)
        items.append({
            "image": tgt.image, "label": tgt.label,
            "context_in": torch.stack(cin), "context_out": torch.stack(cout),
            "subject": subs[b], "context_subjects": list(ctxs[b]),
            "label_name": clss[b], "spacing": tgt.spacing,
            "aug_mode": torch.tensor(0, dtype=torch.long), "crop_geom": tgt.crop_geom,
        })
    return incontext_collate_fn(items)


def _forward_level(model, batch, spacing):
    sp = float(spacing) if getattr(model, "spacing_aware", False) else None
    out = model(batch["image"], context_in=batch["context_in"],
                context_out=batch["context_out"], mode="train", spacing=sp)
    return out["final_logit"].float()


def _hard_pred_native(logit_b1ghw, T, is_prob):
    prob = logit_b1ghw.float().clamp(0, 1) if is_prob else torch.sigmoid(logit_b1ghw.float())
    up = F.interpolate(prob, size=(T, T, T), mode="trilinear", align_corners=False)
    return (up >= 0.5).float().squeeze(1)                                 # (B,T,T,T)


def run_cascade(model, provider, batch, augmentor, spacings, *, device, training,
                step, seed, jitter=0, is_prob=False, want_hard_preds=False):
    assert len(spacings) >= 2, "cascade needs >=2 spacings"
    assert int(batch["aug_mode"].max()) == 0, "run_cascade: v2 REAL tasks only (aug_mode==0)"
    N = len(spacings)
    T = batch["image"].shape[-1]
    B = batch["image"].shape[0]
    geo_seed = seed * 1_000_003 + step

    logits, targets, geoms = [], [], []
    centers = [[None] * B]
    hard = [] if want_hard_preds else None
    empty_hits = empty_total = 0

    cur = _to_device(dict(batch), device)
    for i in range(N):
        if i > 0:
            cur = _recrop_level(provider, batch, centers[i], spacings[i],
                                step=step, seed=seed, level=i, jitter=jitter)
            cur = _to_device(cur, device)
        capture = augmentor is not None and i < N - 1
        if augmentor is not None:
            cur, geo = augmentor.apply(
                cur, geo_gen=_gen(geo_seed, device),
                int_gen=_gen(geo_seed + INT_OFFSET * (i + 1), device), capture=capture)
        else:
            geo = None

        logit = _forward_level(model, cur, spacings[i])
        tgt = target_like(cur["label"].unsqueeze(1).float(), logit)
        logits.append(logit); targets.append(tgt)
        geoms.append(cur["crop_geom"] if "crop_geom" in cur else batch["crop_geom"])
        if want_hard_preds:
            hard.append(_hard_pred_native(logit, T, is_prob))

        if i < N - 1:
            cens = _centroid_from_logit(logit, T, is_prob)
            row = []
            for b in range(B):
                empty_total += 1
                if geo is not None and geo.grid is not None:
                    gr = geo.grid.view(B, -1, T, T, T, 3)[b, -1]        # last vol of task = target
                    fl = geo.flips.view(B, -1, 3)[b, -1]
                else:
                    gr, fl = None, torch.zeros(3, dtype=torch.bool)
                nc = invert_geo_center(cens[b], gr, fl, geoms[i][b], T)
                if nc is None:
                    empty_hits += 1
                row.append(nc)
            centers.append(row)

    return CascadeResult(logits=logits, targets=targets, geoms=geoms, centers=centers,
                         hard_preds=hard,
                         empty_frac=(empty_hits / empty_total if empty_total else 0.0))
```

Note on the target row inside `geo.grid`: `_stack_task` builds `vols = cat([image.unsqueeze(1), ctx], dim=1)` → the **target is row 0** of each task, contexts are `1..K`. **Fix the index**: the target is row `0`, not `-1`. Use `[b, 0]` for both `geo.grid.view(...)` and `geo.flips.view(...)`. (Double-check against `src/gpu_augment.py::_stack_task` when implementing — the comment there says "vol b*T+0 = target b".)

- [ ] **Step 4: Correct the target-row index**

In `run_cascade`, set the target slices to row `0`:

```python
                    gr = geo.grid.view(B, -1, T, T, T, 3)[b, 0]         # target = row 0 of task
                    fl = geo.flips.view(B, -1, 3)[b, 0]
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest experiments/3d/tests/test_cascade.py -v`
Expected: PASS — all `invert` tests plus `test_run_cascade_two_levels_no_aug`, `test_run_cascade_empty_prob_falls_back_to_gt_centroid`.
(`_cascade_loss` is exercised here but defined in Task 7 Step 3 — add a minimal stub now so imports resolve, then flesh it out in Task 7. Stub:)

```python
def _cascade_loss(res: CascadeResult, loss_fn, weights):
    """Sum_i w_i * loss_fn(logit_i, target_i). Returns (total, [per-level floats])."""
    per = [loss_fn(res.logits[i], res.targets[i]) for i in range(len(res.logits))]
    w = list(weights) if weights is not None else [1.0] * len(per)
    total = sum(float(w[i]) * per[i] for i in range(len(per)))
    return total, [float(p.detach()) for p in per]
```

Add that stub to `cascade.py` now (Task 7 keeps it as-is — it is already complete; Task 7 only wires it into `train_epoch`).

- [ ] **Step 6: Commit**

```bash
export PATH="/software/anaconda3/envs/git/bin:$PATH"
git add experiments/3d/cascade.py experiments/3d/tests/test_cascade.py
git commit -m "feat(cascade): run_cascade N-level forward + CascadeResult + _cascade_loss

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01CuANpa5naaN7SJLDf5w9Db"
```

---

## Task 6: config guard `_assert_cascade_supported`

**Files:**
- Modify: `experiments/3d/common.py` — add `_assert_cascade_supported(cfg)`, call it in `train_loader` and `make_eval_loader`
- Modify: `experiments/3d/train.py` — call `_assert_cascade_supported(cfg)` early in `main`; build `GpuAugmentor` when cascade is on even if `augmentations.gpu` is false
- Test: `experiments/3d/tests/test_cascade_guard.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `experiments/3d/common._assert_cascade_supported(cfg) -> None`. No-op when `cfg.data.get("cascade_spacings")` is falsy. Otherwise raises `ValueError` with an actionable message unless **all** hold: `cfg.get("model") == "patchset3d"`; `cfg.data.get("loader_v2")` truthy; `cfg.data.get("source", "totalseg") in _TOTALSEG_SOURCES`; `len(cascade_spacings) >= 2`; `float(cascade_spacings[0]) == float(cfg.data.crop_spacing_mm)`; `cfg.data.get("train_spacing_range") is None`; and (when `train.cascade_loss_weights` is set) `len(cascade_loss_weights) == len(cascade_spacings)`.

- [ ] **Step 1: Write the failing test**

Create `experiments/3d/tests/test_cascade_guard.py`:

```python
"""Task 6: _assert_cascade_supported guard."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from omegaconf import OmegaConf

from common import _assert_cascade_supported


def _cfg(**over):
    base = {
        "model": "patchset3d",
        "data": {"loader_v2": True, "source": "totalseg", "crop_spacing_mm": 3,
                 "cascade_spacings": [3, 1.5], "train_spacing_range": None},
        "train": {"cascade_loss_weights": [1.0, 1.0]},
    }
    cfg = OmegaConf.create(base)
    cfg.merge_with(OmegaConf.create(over))
    return cfg


def test_ok():
    _assert_cascade_supported(_cfg())            # no raise


def test_off_is_noop():
    _assert_cascade_supported(_cfg(data={"cascade_spacings": None}))


def test_rejects_non_patchset():
    with pytest.raises(ValueError, match="patchset3d"):
        _assert_cascade_supported(_cfg(model="medverse"))


def test_rejects_loader_v1():
    with pytest.raises(ValueError, match="loader_v2"):
        _assert_cascade_supported(_cfg(data={"loader_v2": False}))


def test_rejects_spacing_mismatch():
    with pytest.raises(ValueError, match="crop_spacing_mm"):
        _assert_cascade_supported(_cfg(data={"crop_spacing_mm": 2}))


def test_rejects_train_spacing_range_combo():
    with pytest.raises(ValueError, match="train_spacing_range"):
        _assert_cascade_supported(_cfg(data={"train_spacing_range": [1.5, 3.0]}))


def test_rejects_short_list():
    with pytest.raises(ValueError, match="at least 2"):
        _assert_cascade_supported(_cfg(data={"cascade_spacings": [3], "crop_spacing_mm": 3}))


def test_rejects_weight_length_mismatch():
    with pytest.raises(ValueError, match="cascade_loss_weights"):
        _assert_cascade_supported(_cfg(train={"cascade_loss_weights": [1.0]}))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest experiments/3d/tests/test_cascade_guard.py -v`
Expected: FAIL — `ImportError: cannot import name '_assert_cascade_supported'`.

- [ ] **Step 3: Implement the guard in `common.py`**

In `experiments/3d/common.py`, add near the other module-level helpers (e.g. just above `def build_dataset`):

```python
def _assert_cascade_supported(cfg) -> None:
    """Validate data.cascade_spacings / train.cascade_loss_weights. No-op when cascade is off."""
    d = cfg.data
    spacings = d.get("cascade_spacings")
    if not spacings:
        return
    if cfg.get("model") != "patchset3d":
        raise ValueError("data.cascade_spacings requires model=patchset3d.")
    if not d.get("loader_v2", False):
        raise ValueError("data.cascade_spacings requires data.loader_v2=true (v2 pipeline).")
    if d.get("source", "totalseg") not in _TOTALSEG_SOURCES:
        raise ValueError(f"data.cascade_spacings: source {d.get('source')!r} is not a "
                         f"v2 TotalSeg source ({sorted(_TOTALSEG_SOURCES)}).")
    if len(spacings) < 2:
        raise ValueError(f"data.cascade_spacings needs at least 2 entries, got {list(spacings)}.")
    if float(spacings[0]) != float(d.get("crop_spacing_mm")):
        raise ValueError(f"data.cascade_spacings[0]={spacings[0]} must equal "
                         f"data.crop_spacing_mm={d.get('crop_spacing_mm')} (level-0 geometry).")
    if d.get("train_spacing_range") is not None:
        raise ValueError("data.cascade_spacings and data.train_spacing_range are mutually "
                         "exclusive (both set the per-batch physical spacing).")
    w = cfg.get("train", {}).get("cascade_loss_weights")
    if w is not None and len(w) != len(spacings):
        raise ValueError(f"train.cascade_loss_weights (len {len(w)}) must match "
                         f"data.cascade_spacings (len {len(spacings)}).")
```

Confirm `_TOTALSEG_SOURCES` is already defined in `common.py` (it is — used at `build_dataset` line ~230). If the import-time name is module-private and not yet in scope at the helper location, move the helper below its definition.

- [ ] **Step 4: Call the guard from the loaders**

In `experiments/3d/common.py`, add `_assert_cascade_supported(cfg)` as the first line of `train_loader(cfg)` and of `make_eval_loader(cfg, ...)` (right after the docstring).

- [ ] **Step 5: Wire into `train.py` main**

In `experiments/3d/train.py`, inside `main`, right after `vcfg = eval_cfg(cfg)` (before building loaders), add:

```python
    _assert_cascade_supported(cfg)
```

and add `_assert_cascade_supported` to the existing `from common import (...)` line.

Then find the `GpuAugmentor` construction block (`if cfg.augmentations.get("gpu", False):`) and widen the condition so cascade always gets an augmentor:

```python
    from src.gpu_augment import GpuAugmentor
    _cascade_on = bool(cfg.data.get("cascade_spacings"))
    if cfg.augmentations.get("gpu", False) or _cascade_on:
        _, _sc_int, _sc_pi, _ = _self_context(cfg.data, "train")
        gpu_aug = GpuAugmentor(cfg.augmentations,
                               self_context_per_image=bool(_sc_pi),
                               self_context_intensity=bool(_sc_int),
                               seed=int(cfg.get("seed", 0)),
                               ct_norm=cfg.data.get("ct_norm"))
    else:
        gpu_aug = None
```

- [ ] **Step 6: Run test to verify it passes**

Run: `python -m pytest experiments/3d/tests/test_cascade_guard.py -v`
Expected: PASS (8 tests).

- [ ] **Step 7: Regression — existing guard/sweep tests**

Run: `python -m pytest experiments/3d/tests/test_sweep_guard.py -v`
Expected: PASS (unchanged).

- [ ] **Step 8: Commit**

```bash
export PATH="/software/anaconda3/envs/git/bin:$PATH"
git add experiments/3d/common.py experiments/3d/train.py experiments/3d/tests/test_cascade_guard.py
git commit -m "feat(cascade): _assert_cascade_supported guard + always-on GpuAugmentor

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01CuANpa5naaN7SJLDf5w9Db"
```

---

## Task 7: `train_epoch` cascade branch

**Files:**
- Modify: `experiments/3d/train.py` — `train_epoch` (lines ~470-588)
- Test: `experiments/3d/tests/test_cascade.py` already covers `_cascade_loss`; add one integration-style test with fakes here.

**Interfaces:**
- Consumes: `run_cascade`, `_cascade_loss`, `CascadeResult` (Task 5).
- Produces: `train_epoch(...)` runs the N-level cascade when `cfg.data.get("cascade_spacings")` is set: `res = run_cascade(model, loader.dataset.provider, batch, gpu_aug, spacings, device=DEVICE, training=True, step=<monotonic>, seed=cfg.train.seed, jitter=cfg.data.get("cascade_crop_jitter", 0), is_prob=is_prob)`, `loss, per_level = _cascade_loss(res, loss_fn, cfg.train.get("cascade_loss_weights"))`, one `loss.backward()`. Returns the same `(mean_loss, mean_dice, mean_soft, grid)` tuple; `grid` gains `loss_r{s:g}` and `dice_r{s:g}` per level and `cascade_empty_frac`. `train/dice` and the `hard_sum`/`soft_sum`/`cos_sum` grid metrics use the **finest** level's logits/targets.

- [ ] **Step 1: Write the failing test (append to `test_cascade.py`)**

```python
def test_train_epoch_cascade_smoke(monkeypatch):
    """train_epoch runs the cascade branch end-to-end on fakes: 2 optimiser steps, finite loss,
    per-level metric keys present."""
    import types
    import train as train_mod

    B, T, G = 2, 8, 4
    model = _FakeModel(G=G, hot=(1, 1, 1))
    prov = _FakeProvider(T=T)

    class _Loader:
        dataset = types.SimpleNamespace(provider=prov)
        def __iter__(self): return iter([_v2_batch(B=B, T=T), _v2_batch(B=B, T=T)])
        def __len__(self): return 2

    opt = torch.optim.SGD(model.parameters(), lr=0.0)

    class _Sched:
        def step(self, *a): pass

    cfg = OmegaConf.create({
        "model": "patchset3d",
        "data": {"cascade_spacings": [3.0, 1.5], "cascade_crop_jitter": 0,
                 "crop_spacing_mm": 3.0},
        "train": {"seed": 0, "cascade_loss_weights": [1.0, 1.0]},
    })
    loss_fn = lambda logit, target: torch.nn.functional.mse_loss(
        torch.sigmoid(logit.float()), target.float())

    mean_loss, mean_dice, mean_soft, grid = train_mod.train_epoch(
        model, _Loader(), [opt], _Sched(), step_per_batch=True, loss_fn=loss_fn,
        cfg=cfg, epoch=0, is_patchset=True, gpu_aug=None)

    assert np.isfinite(mean_loss)
    assert "loss_r3" in grid and "loss_r1.5" in grid
    assert "dice_r3" in grid and "dice_r1.5" in grid
    assert "cascade_empty_frac" in grid
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest experiments/3d/tests/test_cascade.py -k train_epoch -v`
Expected: FAIL — `KeyError: 'loss_r3'` (cascade branch not implemented; the current loop errors earlier on the fake batch shape or produces no such keys).

- [ ] **Step 3: Implement the cascade branch in `train_epoch`**

In `experiments/3d/train.py`, add the import at the top of the function-body region (with the other `from ... import`s near the module top):

```python
from cascade import run_cascade, _cascade_loss
```

Inside `train_epoch`, the per-batch loop starts `for batch in pbar:`. Add a cascade fast-path at the **top of the loop body**, before the `synth_realizer` / `gpu_aug` / forward block:

```python
        cascade_spacings = cfg.data.get("cascade_spacings")
        if cascade_spacings:
            spacings = [float(s) for s in cascade_spacings]
            gstep = epoch * len(loader) + n
            for opt in optimizers:
                opt.zero_grad(set_to_none=True)
            t_rc = time.perf_counter()
            with _autocast():
                res = run_cascade(
                    model, loader.dataset.provider, batch, gpu_aug, spacings,
                    device=DEVICE, training=True, step=gstep, seed=int(cfg.train.seed),
                    jitter=int(cfg.data.get("cascade_crop_jitter", 0)), is_prob=is_prob)
                loss, per_level = _cascade_loss(
                    res, loss_fn, cfg.train.get("cascade_loss_weights"))
            fine = res.logits[-1].float()
            if not torch.isfinite(fine).all():
                raise RuntimeError(f"non-finite cascade forward @ epoch {epoch} step {n}")
            loss.backward()
            if cfg.train.get("grad_clip"):
                torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.train.grad_clip)
            for opt in optimizers:
                opt.step()
            if step_per_batch:
                scheduler.step()

            total += loss.item()
            fine_tgt = res.targets[-1]
            dice_sum += _hard_dice(fine, fine_tgt, is_prob)
            soft_run += 1.0 - _soft_dice(_to_prob(fine, is_prob), fine_tgt).item()
            n += 1
            rd = fine.shape[-1]
            prob = torch.sigmoid(fine)
            h, hc = hard_sum(prob, fine_tgt); gh += h; ghc += hc
            s, sc = soft_sum(prob, fine_tgt); gs += s; gsc += sc
            c, cc = cos_sum(prob, fine_tgt);  gc += c; gcc += cc
            for si, sp in enumerate(spacings):
                _c_loss_acc.setdefault(f"loss_r{sp:g}", 0.0)
                _c_loss_acc[f"loss_r{sp:g}"] += per_level[si]
                _c_dice_acc.setdefault(f"dice_r{sp:g}", 0.0)
                _c_dice_acc[f"dice_r{sp:g}"] += _hard_dice(
                    res.logits[si].float(), res.targets[si], is_prob)
            _c_empty_acc[0] += res.empty_frac
            if prof:
                tsum.setdefault("recrop", 0.0)
                tsum["recrop"] += (time.perf_counter() - t_rc) * 1000
            pbar.set_postfix(loss=f"{total/n:.4f}", dice=f"{dice_sum/n:.4f}",
                             soft=f"{soft_run/n:.4f}",
                             lr=f"{optimizers[0].param_groups[0]['lr']:.1e}")
            if prof:
                torch.cuda.synchronize()
                t_prev = time.perf_counter()
            continue
```

Initialise the three cascade accumulators once, next to the existing running-sum inits near the top of `train_epoch` (`total, dice_sum, soft_run, n = 0.0, 0.0, 0.0, 0`):

```python
    _c_loss_acc, _c_dice_acc, _c_empty_acc = {}, {}, [0.0]
```

At the end of `train_epoch`, where `grid` is assembled (`if is_patchset and rd is not None:` block), append the cascade keys:

```python
    if _c_loss_acc and n:
        for k, v in _c_loss_acc.items():
            grid[k] = v / n
        for k, v in _c_dice_acc.items():
            grid[k] = v / n
        grid["cascade_empty_frac"] = _c_empty_acc[0] / n
```

If `prof` and `"recrop" in tsum`, add its per-step time to the `grid["time/..."]` block the same way `data`/`encode`/`attn` are handled (guard with `if "recrop" in tsum`).

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest experiments/3d/tests/test_cascade.py -k "train_epoch or cascade_loss or run_cascade or invert" -v`
Expected: PASS.

- [ ] **Step 5: Manual end-to-end smoke (documented, not automated)**

On a GPU node with the TotalSeg data + the `ct_raw_1.5mm.npy` caches present (see Task 9 note), run 1 epoch of a tiny debug cascade:

```bash
python experiments/3d/train.py experiment=59_organs_cascade_from_scratch \
  train.epochs=1 data.max_train_subjects=8 data.max_ds_len_train=16 \
  train.eval_every=1 eval.tasks_per_class=1 wandb.project=null
```

Expected: completes without error; stdout shows `train/loss_r3`, `train/loss_r1.5`, `val/dice_stitched` in the epoch log; no NaN guard trip.

- [ ] **Step 6: Commit**

```bash
export PATH="/software/anaconda3/envs/git/bin:$PATH"
git add experiments/3d/train.py experiments/3d/tests/test_cascade.py
git commit -m "feat(cascade): train_epoch N-level cascade branch + per-spacing metrics

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01CuANpa5naaN7SJLDf5w9Db"
```

---

## Task 8: cascade val — `_stitched_native_dice_multi` + `evaluate_cascade` + `validate_mean` branch

**Files:**
- Modify: `experiments/3d/evaluate.py` — generalise `_stitched_native_dice` to a level list (keep a 2-arg wrapper)
- Modify: `experiments/3d/cascade.py` — add `evaluate_cascade`
- Modify: `experiments/3d/train.py` — `validate_mean` branches to `evaluate_cascade` when cascade is on; `val/dice` := macro `val/dice_stitched`
- Test: `experiments/3d/tests/test_cascade_stitch.py`

**Interfaces:**
- Consumes: `run_cascade` (Task 5); `evaluate.py`'s `_source_root`, `dice_batch`, `build_sample_table`.
- Produces:
  - `evaluate._stitched_native_dice_multi(pg_levels: list[dict], root) -> dict[(subj, cls), float]`. `pg_levels` is a coarse→fine list; each entry maps `(subj, cls) -> (packbits_pred, pred_shape_tuple, crop_geom_ndarray)` (same triple `evaluate.py` already stores in `pred_geom_out`). Composites every level into the native volume in list order, each overwriting the previous, Dice vs `label.npy == class_idx`. `_stitched_native_dice(base_pg, over_pg, root)` becomes `return _stitched_native_dice_multi([base_pg, over_pg] if over_pg else [base_pg], root)`.
  - `cascade.evaluate_cascade(model, cfg, classes, *, loader, seed, is_prob) -> (rows, cases)` — same shape as `evaluate.evaluate_classes`. Iterates `loader` (the level-0 val loader), runs `run_cascade(training=False, augmentor=None, want_hard_preds=True)` per batch, and per class reports `mean_dice` = macro stitched native Dice, plus `dice_r{s:g}` columns; `cases` carry `dice` (stitched) + `dice_r{s:g}` + `class` + `subject`.

- [ ] **Step 1: Write the failing test**

Create `experiments/3d/tests/test_cascade_stitch.py`:

```python
"""Task 8: _stitched_native_dice_multi — coarse->fine composite, each overwriting the previous."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

evaluate = pytest.importorskip("evaluate")


def _pg_entry(pred_native_bool, geom):
    return {("s0", "liver"): (np.packbits(pred_native_bool.astype(bool)),
                              tuple(pred_native_bool.shape), np.asarray(geom))}


def test_finer_level_overwrites_coarser(tmp_path, monkeypatch):
    # Native volume 1 subject, 1 class. Coarse pred fills a big wrong box; fine pred (smaller
    # geom) fills the correct region -> stitched Dice must beat coarse-only.
    from src.totalseg_dataloader_incontext import _ALL_CLASSES_IDX
    idx = _ALL_CLASSES_IDX["liver"]
    D = H = W = 16
    lbl = np.zeros((D, H, W), dtype=np.uint8)
    lbl[4:8, 4:8, 4:8] = idx
    subj = tmp_path / "s0"
    subj.mkdir()
    np.save(subj / "label.npy", lbl)

    # Patch class-index lookup + source root used by _stitched_native_dice_multi.
    monkeypatch.setattr(evaluate, "_source_root", lambda cfg: (None, str(tmp_path), False),
                        raising=False)

    # coarse: whole-volume crop, prediction = wrong half
    coarse_pred = np.zeros((D, H, W), bool); coarse_pred[0:8, :, :] = True
    coarse_geom = [[0, 0, 0], [D, H, W], [D, H, W], [0, 0, 0]]
    # fine: crop == the GT box region, prediction = exactly the GT box
    fine_pred = np.ones((4, 4, 4), bool)
    fine_geom = [[4, 4, 4], [4, 4, 4], [4, 4, 4], [0, 0, 0]]

    base = _pg_entry(coarse_pred, coarse_geom)
    fine = _pg_entry(fine_pred, fine_geom)

    d_coarse = evaluate._stitched_native_dice_multi([base], str(tmp_path))
    d_casc = evaluate._stitched_native_dice_multi([base, fine], str(tmp_path))
    assert d_casc[("s0", "liver")] > d_coarse[("s0", "liver")]
    assert d_casc[("s0", "liver")] == pytest.approx(1.0, abs=1e-6)


# _stitched_native_dice_multi(pg_levels, root) generalises _stitched_native_dice:
#   keys = pg_levels[-1]; require key present in EVERY level; _write_native each level
#   coarse->fine (list order) onto one native bool volume; Dice vs label.npy == idx.
#   _stitched_native_dice(base, over, root) := _stitched_native_dice_multi(
#       [base, over] if over else [base], root)  -- numerically identical to the old 2-arg fn.
```

(If `_stitched_native_dice_multi`'s real signature threads `root` differently than a positional `str`, adapt the test call to match the wrapper you keep — the invariant under test is "finer overwrites coarser and raises Dice".)

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest experiments/3d/tests/test_cascade_stitch.py -v`
Expected: FAIL — `AttributeError: module 'evaluate' has no attribute '_stitched_native_dice_multi'`.

- [ ] **Step 3: Generalise `_stitched_native_dice` in `evaluate.py`**

Read the current `_stitched_native_dice(base_pg, over_pg, root)` (around line 880). It builds a native prediction from `base_pg`, overwrites each `over_pg` region, Dice vs native GT. Refactor the region-writing core into `_apply_pg_level(vol, pg_entry, ...)` and add:

```python
def _stitched_native_dice_multi(pg_levels, root):
    """Composite pg levels (coarse->fine) into the native volume, each overwriting the
    previous, Dice vs label.npy == class_idx. pg_levels: list of {(subj,cls): (packbits,
    shape, geom_ndarray)} in coarse->fine order."""
    # ... (reuse the existing per-(subj,cls) native-volume construction; loop pg_levels in
    #      order, writing each level's region on top of the accumulator, then Dice.)
```

Keep the old name as a thin wrapper so `evaluate_spacing_sweep` is untouched:

```python
def _stitched_native_dice(base_pg, over_pg, root):
    return _stitched_native_dice_multi([base_pg, over_pg] if over_pg else [base_pg], root)
```

Preserve the exact native-GT loading, class-index resolution and Dice formula the current implementation uses (do not change numeric behaviour for the 2-level case — `evaluate_spacing_sweep` depends on it).

- [ ] **Step 4: Run the stitch test**

Run: `python -m pytest experiments/3d/tests/test_cascade_stitch.py -v`
Expected: PASS.

- [ ] **Step 5: Regression — existing eval sweep path imports cleanly**

Run: `python -c "import sys; sys.path.insert(0,'experiments/3d'); import evaluate; print(evaluate._stitched_native_dice)"`
Expected: prints a function ref, no ImportError.

- [ ] **Step 6: Implement `evaluate_cascade` in `cascade.py`**

Append to `experiments/3d/cascade.py`:

```python
def evaluate_cascade(model, cfg, classes, *, loader, seed, is_prob):
    """v2 cascade val pass. Iterates the level-0 val `loader`, runs the N-level cascade with
    no aug, and returns (rows, cases) shaped like evaluate.evaluate_classes: per class a
    macro stitched-native Dice as `mean_dice`, plus per-spacing `dice_r{s:g}`.
    """
    from collections import defaultdict
    import numpy as np
    from common import _source_root                     # NOTE: _source_root lives in common.py
    from evaluate import _stitched_native_dice_multi

    spacings = [float(s) for s in cfg.data.cascade_spacings]
    N = len(spacings)
    _, root, _ = _source_root(cfg)
    pg_levels = [dict() for _ in range(N)]
    per_res = defaultdict(dict)                      # (subj,cls) -> {s: native dice}
    order = []                                       # (subj,cls) in loader order

    model_net = getattr(model, "model", model)
    model_net.eval()
    step = 0
    for batch in loader:
        with torch.no_grad():
            res = run_cascade(model, loader.dataset.provider, batch, augmentor=None,
                              spacings=spacings, device=next(model_net.parameters()).device,
                              training=False, step=step, seed=seed, jitter=0,
                              is_prob=is_prob, want_hard_preds=True)
        step += 1
        subs, clss = batch["subjects"], batch["label_names"]
        for b in range(len(subs)):
            key = (subs[b], clss[b])
            order.append(key)
            for li in range(N):
                hp = res.hard_preds[li][b].cpu().numpy().astype(bool)
                geom = res.geoms[li][b].cpu().numpy()
                pg_levels[li][key] = (np.packbits(hp), tuple(hp.shape), geom)
            # per-resolution native dice = single-level stitch (that level alone)
            for li, s in enumerate(spacings):
                dl = _stitched_native_dice_multi([pg_levels[li]], root)
                per_res[key][s] = float(dl.get(key, float("nan")))

    stitched = _stitched_native_dice_multi(pg_levels, root)

    cases_by_class = defaultdict(list)
    for key in order:
        subj, cls = key
        case = {"class": cls, "subject": subj,
                "dice": round(float(stitched.get(key, float("nan"))), 4)}
        for s in spacings:
            case[f"dice_r{s:g}"] = round(per_res[key].get(s, float("nan")), 4)
        cases_by_class[cls].append(case)

    rows, all_cases = [], []
    for cls in list(classes) + [c for c in cases_by_class if c not in set(classes)]:
        cs = cases_by_class.get(cls, [])
        all_cases.extend(cs)
        if not cs:
            rows.append({"class": cls, "error": "no samples"}); continue
        row = {"class": cls,
               "mean_dice": round(sum(c["dice"] for c in cs) / len(cs), 4),
               "n_samples": len(cs)}
        for s in spacings:
            vals = [c[f"dice_r{s:g}"] for c in cs if not np.isnan(c[f"dice_r{s:g}"])]
            if vals:
                row[f"dice_r{s:g}"] = round(sum(vals) / len(vals), 4)
        rows.append(row)
    return rows, all_cases
```

- [ ] **Step 7: Branch `validate_mean` in `train.py`**

In `experiments/3d/train.py`, at the top of `validate_mean`, add:

```python
    if cfg.data.get("cascade_spacings"):
        from cascade import evaluate_cascade
        rows, cases = evaluate_cascade(
            model, cfg, classes, loader=loader, seed=int(cfg.train.seed),
            is_prob=model_output_is_prob(cfg))
        valid = [r for r in rows if "mean_dice" in r]
        mean_dice = sum(r["mean_dice"] for r in valid) / len(valid) if valid else float("nan")
        return mean_dice, float("nan"), float("nan"), rows, cases
```

(`val/dice_soft` / `val/loss` are reported as NaN for the cascade val pass in this first cut — the design keeps soft/loss reporting out of scope; the headline `val/dice` = macro stitched Dice drives checkpointing.)

In `main`, where `val/dice/<class>` and the seen/unseen split are logged, the per-spacing macros should also be surfaced. After the existing `log.update({f"val/dice/{r['class']}": ...})` line, add:

```python
            if cfg.data.get("cascade_spacings"):
                for s in [float(x) for x in cfg.data.cascade_spacings]:
                    vals = [r[f"dice_r{s:g}"] for r in rows if f"dice_r{s:g}" in r]
                    if vals:
                        log[f"val/dice_r{s:g}"] = sum(vals) / len(vals)
                log["val/dice_stitched"] = val_dice     # == macro stitched (checkpoint metric)
```

- [ ] **Step 8: Run test to verify it passes**

Run: `python -m pytest experiments/3d/tests/test_cascade_stitch.py experiments/3d/tests/test_cascade.py -v`
Expected: PASS.

- [ ] **Step 9: Manual val smoke (documented)**

Re-run the Task 7 Step 5 command; confirm the epoch log now prints `val/dice_stitched` and `val/dice_r3` / `val/dice_r1.5`, and `best.pt` is written on the stitched metric.

- [ ] **Step 10: Commit**

```bash
export PATH="/software/anaconda3/envs/git/bin:$PATH"
git add experiments/3d/evaluate.py experiments/3d/cascade.py experiments/3d/train.py experiments/3d/tests/test_cascade_stitch.py
git commit -m "feat(cascade): v2 cascade val — stitched native Dice + per-spacing macros

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01CuANpa5naaN7SJLDf5w9Db"
```

---

## Task 9: experiment config + docs + config-resolution test

**Files:**
- Create: `configs/experiment/3d/experiment/59_organs_cascade_from_scratch.yaml`
- Modify: `docs/logs.md`
- Test: `experiments/3d/tests/test_cascade_config.py`

**Interfaces:**
- Consumes: `_assert_cascade_supported` (Task 6).
- Produces: `experiment=59_organs_cascade_from_scratch` — exp57 + `data.cascade_spacings: [3, 1.5]`, `train.cascade_loss_weights: [1.0, 1.0]`, resolvable through Hydra and passing the guard.

- [ ] **Step 1: Write the failing test**

Create `experiments/3d/tests/test_cascade_config.py`:

```python
"""Task 9: experiment=59_organs_cascade_from_scratch resolves and passes the guard."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hydra import compose, initialize_config_dir

from common import _assert_cascade_supported

CFG_DIR = str(ROOT / "configs" / "experiment" / "3d")


def test_exp59_resolves_and_passes_guard():
    with initialize_config_dir(config_dir=CFG_DIR, version_base="1.3"):
        cfg = compose(config_name="train",
                      overrides=["experiment=59_organs_cascade_from_scratch"])
    assert list(cfg.data.cascade_spacings) == [3, 1.5]
    assert float(cfg.data.crop_spacing_mm) == 3.0
    assert list(cfg.train.cascade_loss_weights) == [1.0, 1.0]
    assert cfg.data.get("train_spacing_range") is None
    _assert_cascade_supported(cfg)          # no raise
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest experiments/3d/tests/test_cascade_config.py -v`
Expected: FAIL — Hydra `MissingConfigException` / cannot find `59_organs_cascade_from_scratch`.

- [ ] **Step 3: Create the config**

`configs/experiment/3d/experiment/59_organs_cascade_from_scratch.yaml`:

```yaml
# @package _global_
# 59_organs_cascade_from_scratch — exp57 + N-level coarse->fine cascade training.
#
#   python experiments/3d/train.py experiment=59_organs_cascade_from_scratch
#
# Two levels: predict at 3 mm (GT-centred), re-crop the target on the predicted
# centre-of-mass, predict at 1.5 mm; independent weighted loss per level, single backward.
# Geometric augmentation params are sampled once per task and replayed at both levels.
# Val runs the same cascade; `val/dice` is the stitched native Dice (all levels composited
# coarse->fine, each overwriting the previous). See
# docs/superpowers/specs/2026-08-30-cascade-training-patchset3d-design.md.
#
# PRE-REQ: a per-spacing image cache `ct_raw_1.5mm.npy` (and `ct_raw_3mm.npy`) must exist
# per subject dir, else the provider falls back to full-res ct_raw.npy per re-crop load.

defaults:
  - 57_organs_encoder_from_scratch
  - _self_

data:
  crop_spacing_mm: 3            # == cascade_spacings[0]
  cascade_spacings: [3, 1.5]    # coarse -> fine
  cascade_crop_jitter: 0        # re-cropped levels respect the predicted COM exactly

train:
  cascade_loss_weights: [1.0, 1.0]

wandb:
  name: 59_organs_cascade_from_scratch
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest experiments/3d/tests/test_cascade_config.py -v`
Expected: PASS.

- [ ] **Step 5: Add the changelog entry**

Append to `docs/logs.md` (a new dated bullet, matching the file's existing style):

```markdown
## 2026-08-30 — cascade training for PatchSet3D (v2)

* New `data.cascade_spacings: [s0, s1, ...]` runs an N-level coarse->fine cascade in
  `experiments/3d/train.py` (v2 pipeline, patchset3d only): predict at `s_i`, re-crop the
  target on the predicted centre-of-mass (contexts stay GT-centred, same subjects), predict
  at `s_{i+1}`; loss `Σ w_i·loss_i` (`train.cascade_loss_weights`), single backward.
* Levels >=1 are re-cropped in the train loop via `TotalSegProvider.load(center=COM,
  spacing=s_i)` (`LoadRequest.jitter=0`). Geometric aug (flip/affine/elastic/deform) is
  sampled once per task and replayed at every level; the predicted COM is mapped back
  through the composed `grid_sample` grid + flip record (`cascade.invert_geo_center`).
* New `experiments/3d/cascade.py` (`run_cascade`, `evaluate_cascade`) shared by the train
  loop and the v2 val loop. Val logs `dice_r{s}` per level + `dice_stitched` (all levels
  composited coarse->fine, each overwriting the previous); `val/dice` = `dice_stitched`.
* Config: `experiment=59_organs_cascade_from_scratch`. Pre-req: per-spacing `ct_raw_{s}mm.npy`
  image caches or the provider falls back to full-res per re-crop load.
```

- [ ] **Step 6: Full test sweep**

Run: `python -m pytest experiments/3d/tests/ -v`
Expected: PASS (all new cascade tests + all pre-existing tests unchanged).

- [ ] **Step 7: Commit**

```bash
export PATH="/software/anaconda3/envs/git/bin:$PATH"
git add configs/experiment/3d/experiment/59_organs_cascade_from_scratch.yaml docs/logs.md experiments/3d/tests/test_cascade_config.py
git commit -m "feat(cascade): experiment 59 config + docs/logs entry

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01CuANpa5naaN7SJLDf5w9Db"
```

---

## Self-Review

**1. Spec coverage**

| Spec section | Task(s) |
| --- | --- |
| §1 Config surface (`cascade_spacings`, `cascade_crop_jitter`, `cascade_loss_weights`, all asserts) | 6 (guard), 9 (config) |
| §2 Dataloader — `LoadRequest.jitter`, provider threading, no new dataset class | 1 |
| §3 `cascade.py` — `run_cascade`, level-0 aug + capture, `invert_geo_center`, level-i re-crop + replay, `CascadeResult` | 4, 5 |
| §4 `GpuAugmentor` refactor — `_geometric` capture, `apply()`, `__call__` unchanged, `INT_OFFSET` | 2, 3 |
| §5 `train_epoch` integration — weighted loss, per-spacing logging, `recrop` profile bucket, non-cascade path untouched | 7 |
| §6 Val integration — `evaluate_cascade`, `dice_r{s}` + `dice_stitched`, `val/dice` := stitched, stitch generalised to a list | 8 |
| §7 New experiment config (`59_*`), exp57 intact | 9 |
| §8 Files touched | all |
| §8 Risk: pre-req image caches | 9 (config header + docs), 7 Step 5 (smoke note) |
| §8 Risk: RNG-replay shape assumption | 5 (`run_cascade` asserts `aug_mode==0`; `_geometric` replay verified in Task 3 test) |
| §8 Risk: `class_balanced` re-sampling | 5 (`_recrop_level` reads subjects/contexts from the batch, never re-samples) |
| "Resolved": all levels stitched coarse→fine, each overwriting previous | 8 (`_stitched_native_dice_multi`) |
| "Resolved": metric keys `dice_r{s}` + `dice_stitched` | 7, 8 |

No gaps.

**2. Placeholder scan**

- Task 8 Step 3 leaves the *inside* of `_stitched_native_dice_multi` as prose ("reuse the existing per-(subj,cls) native-volume construction"). This is deliberate: the exact body must be derived from the current `_stitched_native_dice` in `evaluate.py` (which the plan author has not reproduced here to avoid transcription drift), and the numeric-invariance requirement + the passing test (`test_cascade_stitch.py`) fully constrain it. Acceptable — the engineer reads the existing function and factors it; the test is the spec.
- All other code steps contain complete, runnable code.
- No "TODO", "handle edge cases", "similar to Task N", or bare "write tests".

**3. Type consistency**

- `GeoState(grid, flips)` — defined Task 2, consumed Task 3 (`apply` builds it), Task 5 (`run_cascade` reads `geo.grid`, `geo.flips`). Consistent.
- `_geometric(..., capture=False)` return: 2-tuple / 4-tuple `(vols, masks, grid, flips)` — Task 2 defines, Task 3 `apply` unpacks the 4-tuple. Consistent.
- `invert_geo_center(centroid_dhw, grid_row, flips_row, crop_geom_row, T)` — Task 4 defines; Task 5 calls with `(cens[b], gr, fl, geoms[i][b], T)`. Consistent (positional).
- `CascadeResult` fields `logits/targets/geoms/centers/hard_preds/empty_frac` — Task 5 defines; Task 7 reads `res.logits[-1]`, `res.targets[-1]`, `res.empty_frac`; Task 8 reads `res.hard_preds`, `res.geoms`. Consistent.
- `run_cascade(model, provider, batch, augmentor, spacings, *, device, training, step, seed, jitter=0, is_prob=False, want_hard_preds=False)` — one signature, used identically in Task 7 (`training=True`, `augmentor=gpu_aug`) and Task 8 (`training=False`, `augmentor=None`, `want_hard_preds=True`). Consistent.
- `_cascade_loss(res, loss_fn, weights) -> (total_tensor, [floats])` — defined in Task 5 Step 5 (complete, not a stub to revisit), consumed Task 7. Consistent.
- `_assert_cascade_supported(cfg)` — Task 6 defines; Task 6 calls it in `common.train_loader` / `make_eval_loader` / `train.main`; Task 9 test calls it. Consistent.
- `_stitched_native_dice_multi(pg_levels, root)` — Task 8 defines; `evaluate_cascade` (Task 8) and the `_stitched_native_dice` wrapper (Task 8) call it. Consistent.
- Metric key format `f"...r{s:g}"` (e.g. `dice_r3`, `dice_r1.5`) — used identically in Task 7 (`train/`) and Task 8 (`val/`). Consistent.

No inconsistencies found.
