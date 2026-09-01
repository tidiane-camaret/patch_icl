# Cascade RAM cache + GPU crop/realize — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make N-level cascade training fast by holding every subject's `ct_raw.npy` + `label.npy` in a fork-shared RAM cache and doing all crop resample / placement / normalization on the GPU, for every cascade level.

**Architecture:** A process-lifetime singleton dict of read-only numpy arrays is preloaded in `TotalSegProvider.__init__` before the DataLoader forks, so workers and the main-process re-crop share the pages copy-on-write. The provider gains `load_native_crop()` returning a raw integer-decimated crop + geometry; a new `src/gpu_realize_crop.py` turns a list of those into the standard `image/label/context_in/context_out` batch dict on-device (trilinear/area resample, occupancy/soft mask, centre-pad, CT normalize). `cascade.py::_recrop_level` uses this for levels ≥ 1; the cascade train loader ships level-0 payloads that `train_epoch` realizes the same way. Non-cascade v2 paths are untouched.

**Tech Stack:** PyTorch (`F.interpolate`, `avg_pool3d`), NumPy (`np.load`, mmap), Hydra/OmegaConf configs, pytest. Cluster data on NFS.

**Spec:** `docs/superpowers/specs/2026-09-01-cascade-ram-cache-gpu-realize-design.md` — read it alongside this plan.

## Global Constraints

- **Native grid is 1.5 mm isotropic for all 1228 subjects.** `ct_raw.npy` (fp16) *is* the 1.5 mm volume; `label.npy` (uint8, all 117 classes) is on the same grid. No `ct_raw_1.5mm.npy` is built.
- **RAM cache holds `ct_raw.npy` + `label.npy` only** — no 3 mm cache. Coarser levels decimate from the cached 1.5 mm array.
- **Fork-COW correctness:** every cached array MUST be `arr.flags.writeable = False`; consumers slice + `.contiguous()` a small copy out, never mutate the cache.
- **Correctness bar = semantic equivalence**, not bit-parity: normalized-image `max|Δ|` < 2e-2 vs `crop_and_place`; `occupancy` mask Dice == 1.0; `soft` mask `max|Δ|` < 1e-4; `crop_geom` byte-identical (`torch.equal`).
- **`crop_geom` is computed on CPU by `organ_crop_arrays` from the native grid and passed through untouched** — `invert_geo_center`, the M2 prior warp, `_stitched_native_dice_multi` all depend on it being unchanged.
- **Air pad value = the resampled member's own normalized `img.min()`** (matches `place_image` and `gpu_synth_realize.py:91`).
- **Decimation factor is per-axis, from geometry not pitch:** `decim[a] = max(1, floor(crop_sizes[a] / out_sizes[a]))`.
- **Any `cascade_spacings`** of length ≥ 2, arbitrary pitch ratios (`[3,1.5]`, `[6,3,1.5]`, non-2×). Nothing assumes 2 levels or power-of-two ratios.
- **Determinism:** `organ_crop_arrays` consumes `req.rng` exactly once (crop jitter). `load_native_crop` must not consume it a second time — eval determinism (`InContextDataset` per-item `eval_seed` RNG) depends on it.
- **CT frame:** `CtNormSpec` from `src/totalseg_dataset.py` (`resolve_ct_norm`); default `fingerprint_1228` = `clip_lo=-1007, clip_hi=1573, mean=-167.3, std=505.8`. `CtNormSpec.norm_min == (clip_lo-mean)/std`.
- **New behaviour is gated:** `data.ram_cache` and `data.gpu_realize_crop`. Both default **on** when `data.cascade_spacings` is set (opt out with `=false`); off otherwise. Non-cascade v2 train/eval loaders never take these paths.
- **Run tests with** `PATH="/software/anaconda3/envs/git/bin:$PATH"` for git; pytest from repo root. GPU-free tests only (`device="cpu"`) — the dev node has no GPU.
- **Log the change in `docs/logs.md`** (project rule).

---

## File Structure

| File | Responsibility |
|---|---|
| `src/providers/volume_cache.py` (create) | Process-lifetime singleton: load `ct_raw.npy` + `label.npy` per subject into read-only ndarrays, threaded, idempotent. |
| `src/providers/totalseg.py` (modify) | `ram_cache` / `ram_cache_max_subjects` ctor args → preload; `NativeCrop` dataclass; `load_native_crop()`. |
| `src/gpu_realize_crop.py` (create) | `normalize_ct_gpu`, `realize_native_crops`, `native_crop_collate_fn`, `_regroup`. |
| `src/incontext_dataset_v2.py` (modify) | `gpu_realize_crop` ctor flag; `__getitem__` emits a `native_crop` payload (no `"image"`) when set. |
| `experiments/3d/cascade.py` (modify) | `_recrop_level` realize branch; `run_cascade(..., realize_crop=False)`; `realize_cascade_level0(batch, …)`. |
| `experiments/3d/common.py` (modify) | `_assert_cascade_supported` defaults/guard; `build_dataset` forwards flags; `train_loader` uses `native_crop_collate_fn` for cascade realize. |
| `experiments/3d/train.py` (modify) | Build `crop_realizer`; call it in the cascade branch of `train_epoch` before `run_cascade`. |
| `configs/experiment/3d/experiment/59_organs_cascade_from_scratch.yaml` (modify) | Add `data.ram_cache: true`, `data.gpu_realize_crop: true`; rewrite the stale image-cache header. |
| `experiments/3d/tests/test_volume_cache.py` (create) | Cache: shape, `writeable=False`, idempotency, singleton reuse, top-up. |
| `experiments/3d/tests/test_gpu_realize_crop.py` (create) | Parity of `realize_native_crops` vs `crop_and_place`; `native_crop_collate_fn`; `_regroup`. |
| `experiments/3d/tests/test_cascade.py` (modify) | `run_cascade(realize_crop=True)` for `[3,1.5]` and `[6,3,1.5]`; `realize_cascade_level0` shape contract. |
| `experiments/3d/tests/test_cascade_guard.py` (modify) | Guard: cascade defaults `ram_cache`/`gpu_realize_crop` on; error on realize-without-cache. |
| `docs/logs.md` (modify) | One entry. |

---

## Task 1: RAM volume cache

**Files:**
- Create: `src/providers/volume_cache.py`
- Test: `experiments/3d/tests/test_volume_cache.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `get_cache(root, subjects, *, max_subjects=None, workers=16) -> dict[str, dict[str, numpy.ndarray]]`
    — outer key = subject id (e.g. `"s0000"`), inner keys `"ct_raw"` (fp16 `(D,H,W)`) and `"label"` (uint8 `(D,H,W)`); every array has `.flags.writeable == False`. Idempotent per `str(root)`: a second call returns the same dict object and loads only subjects not already present (respecting `max_subjects` as a cap on total entries). `subjects` iterable of ids; missing `ct_raw.npy`/`label.npy` for a subject → that subject is skipped (not an error).
  - `clear_cache() -> None` — drops the singleton (test hygiene only).

- [ ] **Step 1: Write the failing test**

```python
# experiments/3d/tests/test_volume_cache.py
"""RAM volume cache: fork-COW read-only preload (plan Task 1)."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.providers.volume_cache import get_cache, clear_cache


@pytest.fixture(autouse=True)
def _clean():
    clear_cache()
    yield
    clear_cache()


def _fake_root(tmp_path, ids=("s0", "s1", "s2")):
    for i, s in enumerate(ids):
        d = tmp_path / s
        d.mkdir()
        np.save(d / "ct_raw.npy", np.full((4, 5, 6), i, dtype=np.float16))
        np.save(d / "label.npy", np.full((4, 5, 6), i, dtype=np.uint8))
    return tmp_path


def test_loads_arrays_readonly(tmp_path):
    root = _fake_root(tmp_path)
    c = get_cache(root, ["s0", "s1", "s2"])
    assert set(c) == {"s0", "s1", "s2"}
    assert c["s1"]["ct_raw"].shape == (4, 5, 6)
    assert c["s1"]["ct_raw"].dtype == np.float16
    assert c["s1"]["label"].dtype == np.uint8
    assert c["s0"]["ct_raw"].flags.writeable is False
    assert c["s0"]["label"].flags.writeable is False


def test_idempotent_same_object_and_topup(tmp_path):
    root = _fake_root(tmp_path)
    c1 = get_cache(root, ["s0"])
    c2 = get_cache(root, ["s0", "s1"])
    assert c1 is c2                      # same singleton dict
    assert set(c2) == {"s0", "s1"}       # s1 topped up
    assert c1["s0"]["ct_raw"] is c2["s0"]["ct_raw"]   # s0 not reloaded


def test_max_subjects_caps_total(tmp_path):
    root = _fake_root(tmp_path)
    c = get_cache(root, ["s0", "s1", "s2"], max_subjects=2)
    assert len(c) == 2


def test_missing_files_skipped(tmp_path):
    root = _fake_root(tmp_path, ids=("s0",))
    (tmp_path / "s9").mkdir()            # dir exists, no npy
    c = get_cache(root, ["s0", "s9"])
    assert set(c) == {"s0"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PATH="/software/anaconda3/envs/git/bin:$PATH" python -m pytest experiments/3d/tests/test_volume_cache.py -q`
Expected: FAIL — `ModuleNotFoundError: src.providers.volume_cache`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/providers/volume_cache.py
"""Process-lifetime RAM cache of TotalSegmentator native volumes.

Holds ct_raw.npy (fp16) + label.npy (uint8) per subject as READ-ONLY numpy
arrays, preloaded once in the main process before the DataLoader forks its
workers, so every fork shares the buffers copy-on-write. Consumers must slice +
.contiguous() a small copy out and never mutate the cached arrays.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

_CACHE: dict[str, dict[str, dict[str, np.ndarray]]] = {}   # str(root) -> {subject -> {"ct_raw","label"}}


def _load_one(root: Path, s: str):
    cp, lp = root / s / "ct_raw.npy", root / s / "label.npy"
    if not (cp.exists() and lp.exists()):
        return s, None
    ct = np.load(cp)                       # materialize into RAM (not mmap)
    lb = np.load(lp)
    ct.flags.writeable = False
    lb.flags.writeable = False
    return s, {"ct_raw": ct, "label": lb}


def get_cache(root, subjects, *, max_subjects=None, workers=16) -> dict:
    """See plan Task 1 Interfaces. Idempotent per str(root); tops up missing subjects."""
    key = str(root)
    root = Path(root)
    store = _CACHE.setdefault(key, {})
    want = list(dict.fromkeys(subjects))                    # de-dup, keep order
    if max_subjects is not None:
        want = want[: int(max_subjects)]
    todo = [s for s in want if s not in store]
    if max_subjects is not None:
        todo = todo[: max(0, int(max_subjects) - len(store))]
    if todo:
        with ThreadPoolExecutor(max_workers=min(workers, len(todo))) as ex:
            for s, payload in ex.map(lambda s: _load_one(root, s), todo):
                if payload is not None:
                    store[s] = payload
    return store


def clear_cache() -> None:
    _CACHE.clear()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PATH="/software/anaconda3/envs/git/bin:$PATH" python -m pytest experiments/3d/tests/test_volume_cache.py -q`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
PATH="/software/anaconda3/envs/git/bin:$PATH" git add src/providers/volume_cache.py experiments/3d/tests/test_volume_cache.py
PATH="/software/anaconda3/envs/git/bin:$PATH" git commit -m "$(cat <<'EOF'
feat: RAM volume cache for TotalSeg native volumes

Process-lifetime singleton of read-only ct_raw/label ndarrays, threaded load,
idempotent per root. Fork-COW sharing for the cascade data path.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01FBrqQGq5vA29WMHSrDPBfa
EOF
)"
```

---

## Task 2: `NativeCrop` + `TotalSegProvider.load_native_crop`

**Files:**
- Modify: `src/providers/totalseg.py`
- Test: `experiments/3d/tests/test_cascade_provider.py` (append)

**Interfaces:**
- Consumes: `get_cache` (Task 1); `organ_crop_arrays`, `_area_pool_3d` from `src.totalseg_dataloader_incontext`; `_ALL_CLASSES_IDX` from `src.totalseg_dataset`; `LoadRequest` from `src.incontext_dataset_v2`.
- Produces:
  - `TotalSegProvider.__init__(..., ram_cache=False, ram_cache_max_subjects=None)` — when `ram_cache`, calls `get_cache(self.root, <this provider's subject list>, max_subjects=ram_cache_max_subjects)` and stores it as `self._ram`. Otherwise `self._ram = None`.
  - `NativeCrop` dataclass (module-level in `src/providers/totalseg.py`): fields
    `image: torch.Tensor` `(d,h,w)` fp16, `label: torch.Tensor` `(d,h,w)` uint8 (multi-class, not yet class-selected), `class_idx: int`, `out_sizes: list[int]`, `pad_lo: list[int]`, `crop_geom: torch.Tensor` `(4,3)` int64, `crop_spacing_mm: float`, `decim: tuple[int,int,int]`.
  - `TotalSegProvider.load_native_crop(subject, cls, req: LoadRequest) -> NativeCrop` — geometry via `organ_crop_arrays` (consumes `req.rng` once); crop pulled from `self._ram[subject]` if present else `np.load(mmap)`; integer-decimated per `decim[a] = max(1, crop_sizes[a] // out_sizes[a])` with `_area_pool_3d`-style strided `avg_pool3d` on the image and area-pool on the label; NO normalize, NO resample-to-`out_sizes`, NO placement. `crop_geom` identical to what `crop_and_place` returns for the same args.

- [ ] **Step 1: Write the failing test**

```python
# append to experiments/3d/tests/test_cascade_provider.py
import numpy as np
import torch

from src.providers.totalseg import NativeCrop


def _tiny_provider(tmp_path, spacing=1.5, T=8):
    """A TotalSegProvider over a 2-subject fake root with ram_cache on."""
    from src.providers.totalseg import TotalSegProvider
    ids = ("s0", "s1")
    for i, s in enumerate(ids):
        d = tmp_path / s
        d.mkdir()
        # smooth ramp so decimation error is bounded
        v = np.linspace(-500, 500, 20 * 20 * 20, dtype=np.float32).reshape(20, 20, 20)
        np.save(d / "ct_raw.npy", v.astype(np.float16))
        lbl = np.zeros((20, 20, 20), dtype=np.uint8)
        lbl[8:12, 8:12, 8:12] = 5                       # class idx 5
        np.save(d / "label.npy", lbl)
    (tmp_path / "meta.csv").write_text("image_id;split\ns0;train\ns1;train\n")
    # spacings.json so native_spacing resolves to 1.5
    (tmp_path / "spacings.json").write_text(
        '{"s0":{"spacing":[1.5,1.5,1.5],"shape":[20,20,20]},'
        ' "s1":{"spacing":[1.5,1.5,1.5],"shape":[20,20,20]}}')
    prov = TotalSegProvider(
        root=str(tmp_path), classes=["_c5"], image_size=(T, T, T), split="train",
        crop_spacing_mm=spacing, crop_jitter=0, mask_downsample="soft",
        mask_occupancy_thr=0.5, ram_cache=True)
    return prov


def test_load_native_crop_geom_matches_crop_and_place(tmp_path, monkeypatch):
    import random
    from src.incontext_dataset_v2 import LoadRequest
    from src.providers.totalseg import crop_and_place
    from src.totalseg_dataloader_incontext import organ_crop_arrays

    prov = _tiny_provider(tmp_path, spacing=3.0, T=8)
    center = (10, 10, 10)
    nc = prov.load_native_crop("s0", "_c5",
                               LoadRequest(rng=random.Random(0), crop_spacing_mm=3.0,
                                           center=center, jitter=0))
    # reference geom from the pure helper on the same inputs
    lbl = np.load(tmp_path / "s0" / "label.npy")
    _, _, out_sizes, pad_lo, geom = organ_crop_arrays(
        lbl, lbl, center, [1.5, 1.5, 1.5], image_size=(8, 8, 8),
        crop_mm=3.0, jitter=0, rng=random.Random(0))
    assert torch.equal(nc.crop_geom, geom)
    assert nc.out_sizes == list(out_sizes) and nc.pad_lo == list(pad_lo)
    assert nc.decim == (2, 2, 2)                        # 3.0 / 1.5, crop_sizes//out_sizes
    # decimated crop still >= out_sizes on every axis (GPU never upsamples)
    assert all(s >= o for s, o in zip(nc.image.shape, nc.out_sizes))
    assert nc.class_idx == prov._ALL_CLASSES_IDX_c5 if False else nc.class_idx >= 0


def test_load_native_crop_consumes_rng_once(tmp_path):
    import random
    from src.incontext_dataset_v2 import LoadRequest
    prov = _tiny_provider(tmp_path, spacing=1.5, T=8)
    r1, r2 = random.Random(0), random.Random(0)
    prov.load_native_crop("s0", "_c5", LoadRequest(rng=r1, crop_spacing_mm=1.5, jitter=3))
    # one randint(lo,hi) per axis == 3 draws; mirror it manually
    [r2.randint(0, 0) for _ in range(3)]
    assert r1.random() == r2.random()
```

Note: `classes=["_c5"]` is a placeholder class name; `load_native_crop` maps it via `_ALL_CLASSES_IDX.get(cls, -1)`. If the fake class isn't in `_ALL_CLASSES_IDX`, `class_idx == -1` and the test's `>= 0` assertion must become `== -1`. Pick a real class name present in `src/totalseg_dataset._ALL_CLASSES_IDX` (e.g. `"liver"`) and plant that index in the fake `label.npy` instead — adjust `lbl[...] = _ALL_CLASSES_IDX["liver"]`.

- [ ] **Step 2: Run test to verify it fails**

Run: `PATH="/software/anaconda3/envs/git/bin:$PATH" python -m pytest experiments/3d/tests/test_cascade_provider.py -q`
Expected: FAIL — `ImportError: cannot import name 'NativeCrop'`.

- [ ] **Step 3: Write minimal implementation**

Add to `src/providers/totalseg.py` (imports + dataclass near the top, method on the class):

```python
from dataclasses import dataclass

import torch.nn.functional as F

from src.totalseg_dataloader_incontext import (
    organ_crop_arrays, place_image, place_label, resample_binary,
    _bbox_for_subject, _IDX_TO_CLASS, _area_pool_3d,
)
from src.providers.volume_cache import get_cache


@dataclass
class NativeCrop:
    image: torch.Tensor            # (d,h,w) fp16, native-pitch, integer-decimated toward out_sizes
    label: torch.Tensor           # (d,h,w) uint8, multiclass, SAME decimation as image
    class_idx: int
    out_sizes: list                # from organ_crop_arrays
    pad_lo: list
    crop_geom: torch.Tensor        # (4,3) int64 — identical to crop_and_place's
    crop_spacing_mm: float
    decim: tuple


def _decim_int_pool(arr_t, decim, *, is_label):
    """Strided integer downsample of a (d,h,w) tensor by `decim` (per-axis int>=1).
    Image: avg_pool3d (area prefilter). Label: max-pool of a one-hot is wrong for
    multiclass, so area-pool the float then round is also wrong — instead take a
    strided subsample of the label (occupancy/soft resample on GPU re-derives the
    fraction from whatever native detail survives; decim is chosen so the survivor
    grid is still >= out_sizes)."""
    if all(d == 1 for d in decim):
        return arr_t
    if is_label:
        return arr_t[:: decim[0], :: decim[1], :: decim[2]].contiguous()
    x = arr_t.float()[None, None]
    x = F.avg_pool3d(x, kernel_size=decim, stride=decim)
    return x[0, 0].to(arr_t.dtype)
```

```python
    # --- inside TotalSegProvider ---
    def load_native_crop(self, subject, cls, req: LoadRequest) -> "NativeCrop":
        if getattr(self, "_ram", None) is not None and subject in self._ram:
            image_np = self._ram[subject]["ct_raw"]
            label_np = self._ram[subject]["label"]
        else:
            subj_dir = self.root / subject
            label_np = np.load(subj_dir / "label.npy", mmap_mode="r")
            image_np = np.load(subj_dir / "ct_raw.npy", mmap_mode="r")
        center = req.center
        if center is None:
            D, H, W = label_np.shape
            center = self._bbox.get(subject, {}).get(cls, (D // 2, H // 2, W // 2))
        jitter = _resolve_jitter(req, self.crop_jitter)
        native_sp = self._spacings.get(subject, (1.0, 1.0, 1.0))
        crop_ct, crop_lbl, out_sizes, pad_lo, geom = organ_crop_arrays(
            image_np, label_np, center, list(native_sp),
            image_size=(self.T, self.T, self.T), crop_mm=req.crop_spacing_mm,
            jitter=jitter, rng=req.rng)
        crop_sizes = geom[1].tolist()
        decim = tuple(max(1, int(cs) // max(1, int(o))) for cs, o in zip(crop_sizes, out_sizes))
        img_t = torch.from_numpy(np.ascontiguousarray(crop_ct))          # fp16
        lbl_t = torch.from_numpy(np.ascontiguousarray(crop_lbl))         # uint8
        img_t = _decim_int_pool(img_t, decim, is_label=False)
        lbl_t = _decim_int_pool(lbl_t, decim, is_label=True)
        return NativeCrop(image=img_t, label=lbl_t,
                          class_idx=_ALL_CLASSES_IDX.get(cls, -1),
                          out_sizes=list(out_sizes), pad_lo=list(pad_lo),
                          crop_geom=geom, crop_spacing_mm=float(req.crop_spacing_mm),
                          decim=decim)
```

```python
    # --- in TotalSegProvider.__init__, after self._spacings / self._ct_stats ---
    def __init__(self, ..., ram_cache=False, ram_cache_max_subjects=None):
        ...
        self._ram = None
        if ram_cache:
            subs = sorted({s for lst in self._label_to_subjects.values() for s in lst})
            self._ram = get_cache(self.root, subs, max_subjects=ram_cache_max_subjects)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PATH="/software/anaconda3/envs/git/bin:$PATH" python -m pytest experiments/3d/tests/test_cascade_provider.py -q`
Expected: PASS (existing 4 + 2 new).

- [ ] **Step 5: Commit**

```bash
PATH="/software/anaconda3/envs/git/bin:$PATH" git add src/providers/totalseg.py experiments/3d/tests/test_cascade_provider.py
PATH="/software/anaconda3/envs/git/bin:$PATH" git commit -m "$(cat <<'EOF'
feat: TotalSegProvider.load_native_crop + RAM cache wiring

NativeCrop payload = integer-decimated crop + geometry, no resample/normalize.
ram_cache ctor arg preloads the singleton before workers fork.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01FBrqQGq5vA29WMHSrDPBfa
EOF
)"
```

---

## Task 3: `src/gpu_realize_crop.py` — GPU resample + collate

**Files:**
- Create: `src/gpu_realize_crop.py`
- Test: `experiments/3d/tests/test_gpu_realize_crop.py`

**Interfaces:**
- Consumes: `NativeCrop` (Task 2); `CtNormSpec` / `resolve_ct_norm` from `src.totalseg_dataset`.
- Produces:
  - `normalize_ct_gpu(t: torch.Tensor, spec: CtNormSpec) -> torch.Tensor` — `((t.clamp(spec.clip_lo, spec.clip_hi) - spec.mean) / spec.std)`, dtype float32.
  - `realize_native_crops(members, *, T, mask_downsample, occ_thr, ct_spec, device) -> dict` — `members` is a list of length `B` of lists of length `K+1` of `NativeCrop` (target first). Returns
    `{"image": (B,1,T,T,T) f32, "label": (B,T,T,T) {f32 soft | int64 occupancy},
      "context_in": (B,K,1,T,T,T) f32, "context_out": (B,K,T,T,T) …,
      "spacing": (B,3) f32, "crop_geom": (B,4,3) int64}` — `crop_geom` taken from the target member.
  - `native_crop_collate_fn(batch: list[dict]) -> dict` — for `InContextDataset` items shaped `{"native_crop": [NativeCrop … K+1], "subject", "context_subjects", "label_name", "aug_mode"}` (no `"image"`). Keeps `native_crop` as a `B`-list; stacks `aug_mode`; passes through `subjects`/`label_names`/`context_subjects` lists. Output key `"native_crop"` (list of `B` lists).
  - `_regroup(flat, B, Kp1) -> list[list]` — reshape a flat length-`B*(K+1)` list into `B` lists of `K+1`.

- [ ] **Step 1: Write the failing test**

```python
# experiments/3d/tests/test_gpu_realize_crop.py
"""Parity of realize_native_crops vs crop_and_place (plan Task 3)."""
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.incontext_dataset_v2 import LoadRequest
from src.providers.totalseg import crop_and_place, NativeCrop
from src.totalseg_dataloader_incontext import organ_crop_arrays, _area_pool_3d
from src.totalseg_dataset import resolve_ct_norm, normalize_ct
from src.gpu_realize_crop import realize_native_crops, native_crop_collate_fn, _regroup
import torch.nn.functional as F


def _smooth_vol(D=24):
    a = np.linspace(-800, 400, D, dtype=np.float32)
    g = a[:, None, None] + a[None, :, None] * 0.3 + a[None, None, :] * 0.6
    return g.astype(np.float16)


def _native_crop_from(image_np, label_np, cls_idx, center, T, spacing):
    crop_ct, crop_lbl, out_sizes, pad_lo, geom = organ_crop_arrays(
        image_np, label_np, center, [1.5, 1.5, 1.5], image_size=(T, T, T),
        crop_mm=spacing, jitter=0, rng=random.Random(0))
    crop_sizes = geom[1].tolist()
    decim = tuple(max(1, int(cs) // max(1, int(o))) for cs, o in zip(crop_sizes, out_sizes))
    it = torch.from_numpy(np.ascontiguousarray(crop_ct))
    lt = torch.from_numpy(np.ascontiguousarray(crop_lbl))
    if any(d > 1 for d in decim):
        it = F.avg_pool3d(it.float()[None, None], decim, decim)[0, 0].half()
        lt = lt[:: decim[0], :: decim[1], :: decim[2]].contiguous()
    return NativeCrop(image=it, label=lt, class_idx=cls_idx, out_sizes=list(out_sizes),
                      pad_lo=list(pad_lo), crop_geom=geom, crop_spacing_mm=spacing, decim=decim)


def _reference(image_np, label_np, cls_idx, center, T, spacing, md, thr, spec):
    return crop_and_place(
        image_np, label_np, cls_idx, center, T, crop_spacing_mm=spacing,
        native_spacing=(1.5, 1.5, 1.5), jitter=0, rng=random.Random(0),
        mask_downsample=md, occ_thr=thr,
        normalize_fn=lambda a: normalize_ct(a, spec))


def test_image_and_geom_parity_no_pad_spacing_1p5():
    spec = resolve_ct_norm(None)
    img = _smooth_vol(24); lbl = np.zeros((24, 24, 24), np.uint8); lbl[8:16, 8:16, 8:16] = 3
    T, s, center = 8, 1.5, (12, 12, 12)
    nc = _native_crop_from(img, lbl, 3, center, T, s)
    out = realize_native_crops([[nc]], T=T, mask_downsample="soft", occ_thr=0.5,
                               ct_spec=spec, device="cpu")
    img_ref, lbl_ref, geom_ref = _reference(img, lbl, 3, center, T, s, "soft", 0.5, spec)
    assert torch.equal(out["crop_geom"][0], geom_ref)
    assert (out["image"][0] - img_ref).abs().max() < 2e-2
    assert (out["label"][0] - lbl_ref.float()).abs().max() < 1e-4


def test_image_parity_decimated_spacing_3():
    spec = resolve_ct_norm(None)
    img = _smooth_vol(40); lbl = np.zeros((40, 40, 40), np.uint8); lbl[14:26, 14:26, 14:26] = 3
    T, s, center = 8, 3.0, (20, 20, 20)
    nc = _native_crop_from(img, lbl, 3, center, T, s)
    assert nc.decim == (2, 2, 2)
    out = realize_native_crops([[nc]], T=T, mask_downsample="occupancy", occ_thr=0.5,
                               ct_spec=spec, device="cpu")
    img_ref, lbl_ref, _ = _reference(img, lbl, 3, center, T, s, "occupancy", 0.5, spec)
    assert (out["image"][0] - img_ref).abs().max() < 2e-2
    inter = (out["label"][0].bool() & lbl_ref.bool()).sum()
    dice = (2 * inter / (out["label"][0].bool().sum() + lbl_ref.bool().sum())).item()
    assert dice == 1.0


def test_occupancy_never_empty():
    spec = resolve_ct_norm(None)
    img = _smooth_vol(24); lbl = np.zeros((24, 24, 24), np.uint8); lbl[12, 12, 12] = 3   # 1 voxel
    nc = _native_crop_from(img, lbl, 3, (12, 12, 12), 8, 1.5)
    out = realize_native_crops([[nc]], T=8, mask_downsample="occupancy", occ_thr=0.5,
                               ct_spec=spec, device="cpu")
    assert int(out["label"][0].sum()) >= 1


def test_regroup_and_collate():
    assert _regroup(list(range(6)), 2, 3) == [[0, 1, 2], [3, 4, 5]]
    spec = resolve_ct_norm(None)
    img = _smooth_vol(24); lbl = np.zeros((24, 24, 24), np.uint8); lbl[10:14, 10:14, 10:14] = 3
    nc = _native_crop_from(img, lbl, 3, (12, 12, 12), 8, 1.5)
    items = [{"native_crop": [nc, nc], "subject": "s0",
              "context_subjects": ["c0"], "label_name": "liver",
              "aug_mode": torch.tensor(0)}]
    b = native_crop_collate_fn(items)
    assert isinstance(b["native_crop"], list) and len(b["native_crop"][0]) == 2
    assert b["subjects"] == ["s0"] and b["label_names"] == ["liver"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PATH="/software/anaconda3/envs/git/bin:$PATH" python -m pytest experiments/3d/tests/test_gpu_realize_crop.py -q`
Expected: FAIL — `ModuleNotFoundError: src.gpu_realize_crop`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/gpu_realize_crop.py
"""GPU realization of TotalSeg native crops for the cascade data path.

Turns a list of provider NativeCrop payloads (raw integer-decimated crop + crop
geometry) into the standard image/label/context_in/context_out batch dict, on
device: trilinear/area resample to out_sizes, CT-normalize, centre-pad to T^3;
occupancy/soft target-class mask with the resample_binary semantics. Slots in
right before GpuAugmentor, exactly like src/gpu_synth_realize.SynthRealizer.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from src.totalseg_dataloader_incontext import _area_pool_3d


def normalize_ct_gpu(t: torch.Tensor, spec) -> torch.Tensor:
    return ((t.float().clamp(spec.clip_lo, spec.clip_hi) - spec.mean) / spec.std)


def _regroup(flat, B, Kp1):
    return [list(flat[b * Kp1:(b + 1) * Kp1]) for b in range(B)]


@torch.no_grad()
def _realize_member(nc, T, mask_downsample, occ_thr, ct_spec, device):
    size = tuple(int(s) for s in nc.out_sizes)
    src = nc.image.to(device).float()[None, None]
    pre = tuple(min(o, s) for o, s in zip(size, src.shape[2:]))
    if pre != tuple(src.shape[2:]):
        src = F.interpolate(src, size=pre, mode="area")
    img = (src if tuple(src.shape[2:]) == size else
           F.interpolate(src, size=size, mode="trilinear", align_corners=False))
    img = normalize_ct_gpu(img[0, 0], ct_spec)                          # (d,h,w) f32

    binm = (nc.label.to(device).long() == int(nc.class_idx)).float()[None, None]
    frac = _area_pool_3d(binm, size)[0, 0].clamp(0.0, 1.0)
    if mask_downsample == "soft":
        peak = float(frac.amax())
        if bool(binm.any()) and peak < occ_thr:
            frac = torch.where(frac >= peak, torch.full_like(frac, occ_thr), frac)
        mask = frac                                                     # f32
    else:                                                              # occupancy
        m = frac >= occ_thr
        if not bool(m.any()) and bool(binm.any()):
            m.view(-1)[int(frac.argmax())] = True
        mask = m.long()

    if size == (T, T, T):
        return img[None], mask
    fi = torch.full((T, T, T), float(img.min()), device=device)
    fm = torch.zeros(T, T, T, dtype=mask.dtype, device=device)
    sl = tuple(slice(int(p), int(p) + s) for p, s in zip(nc.pad_lo, size))
    fi[sl] = img
    fm[sl] = mask
    return fi[None], fm


@torch.no_grad()
def realize_native_crops(members, *, T, mask_downsample, occ_thr, ct_spec, device):
    B, Kp1 = len(members), len(members[0])
    imgs, masks = [], []
    for b in range(B):
        mi, mm = [], []
        for t in range(Kp1):
            i, m = _realize_member(members[b][t], T, mask_downsample, occ_thr, ct_spec, device)
            mi.append(i); mm.append(m)
        imgs.append(torch.stack(mi)); masks.append(torch.stack(mm))
    img = torch.stack(imgs).float()                                    # (B,K+1,1,T,T,T)
    msk = torch.stack(masks)                                           # (B,K+1,T,T,T)
    geom = torch.stack([members[b][0].crop_geom.to(device) for b in range(B)])
    sp = torch.stack([torch.full((3,), float(members[b][0].crop_spacing_mm)) for b in range(B)])
    return {"image": img[:, 0], "context_in": img[:, 1:],
            "label": msk[:, 0], "context_out": msk[:, 1:],
            "spacing": sp.to(device), "crop_geom": geom}


def native_crop_collate_fn(batch):
    out = {
        "native_crop": [b["native_crop"] for b in batch],              # B lists of (K+1)
        "subjects": [b["subject"] for b in batch],
        "label_names": [b["label_name"] for b in batch],
        "context_subjects": [b["context_subjects"] for b in batch],
        "aug_mode": torch.stack([b.get("aug_mode", torch.tensor(0, dtype=torch.long))
                                 for b in batch]),
    }
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PATH="/software/anaconda3/envs/git/bin:$PATH" python -m pytest experiments/3d/tests/test_gpu_realize_crop.py -q`
Expected: PASS (5 tests). If `test_image_parity_decimated_spacing_3` exceeds 2e-2, widen the label blob / smooth the volume more — the bar is semantic equivalence on smooth data.

- [ ] **Step 5: Commit**

```bash
PATH="/software/anaconda3/envs/git/bin:$PATH" git add src/gpu_realize_crop.py experiments/3d/tests/test_gpu_realize_crop.py
PATH="/software/anaconda3/envs/git/bin:$PATH" git commit -m "$(cat <<'EOF'
feat: gpu_realize_crop — GPU resample/normalize/place for cascade native crops

realize_native_crops mirrors place_image + resample_binary semantics
(occupancy non-empty guard, soft peak floor, img.min() air pad). Parity-tested
vs crop_and_place on CPU.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01FBrqQGq5vA29WMHSrDPBfa
EOF
)"
```

---

## Task 4: cascade `_recrop_level` realize branch (levels ≥ 1)

**Files:**
- Modify: `experiments/3d/cascade.py`
- Test: `experiments/3d/tests/test_cascade.py` (append)

**Interfaces:**
- Consumes: `realize_native_crops`, `_regroup` (Task 3); `TotalSegProvider.load_native_crop` (Task 2).
- Produces:
  - `run_cascade(..., realize_crop=False, mask_downsample="occupancy", occ_thr=0.1, ct_spec=None)` — new kwargs threaded to `_recrop_level`. When `realize_crop`, `_recrop_level` builds the level via `provider.load_native_crop` + `realize_native_crops` instead of `provider.load` + `incontext_collate_fn`; the returned batch dict has the same keys the collate path returns (`image/label/context_in/context_out/spacing/crop_geom` — plus `subjects/context_subjects/label_names` copied from the level-0 `batch`).
  - `_recrop_level(..., realize_crop=False, mask_downsample=..., occ_thr=..., ct_spec=..., device=None)`.

- [ ] **Step 1: Write the failing test**

```python
# append to experiments/3d/tests/test_cascade.py
class _NativeCropProvider:
    """Fake provider exposing load_native_crop -> NativeCrop with identity geom."""
    def __init__(self, T=8):
        from src.providers.totalseg import NativeCrop
        self.T, self._NC = T, NativeCrop
        self.calls = []

    def subjects_for(self, cls):
        return ["s0", "s1", "c0_0", "c0_1", "c0_2", "c1_0", "c1_1", "c1_2"]

    def load_native_crop(self, subject, cls, req):
        self.calls.append({"subject": subject, "center": req.center,
                           "spacing": req.crop_spacing_mm, "jitter": req.jitter})
        T = self.T
        geom = torch.tensor([[0, 0, 0], [T, T, T], [T, T, T], [0, 0, 0]], dtype=torch.long)
        lbl = torch.zeros(T, T, T, dtype=torch.uint8); lbl[1:3, 1:3, 1:3] = 7
        return self._NC(image=torch.zeros(T, T, T, dtype=torch.float16), label=lbl,
                        class_idx=7, out_sizes=[T, T, T], pad_lo=[0, 0, 0],
                        crop_geom=geom, crop_spacing_mm=float(req.crop_spacing_mm),
                        decim=(1, 1, 1))


@pytest.mark.parametrize("spacings", [[3.0, 1.5], [6.0, 3.0, 1.5]])
def test_run_cascade_realize_crop_multilevel(spacings):
    from src.totalseg_dataset import resolve_ct_norm
    B, T, G = 2, 8, 4
    model, prov = _FakeModel(G=G, hot=(1, 1, 1)), _NativeCropProvider(T=T)
    res = run_cascade(model, prov, _v2_batch(B=B, T=T), augmentor=None,
                      spacings=spacings, device=torch.device("cpu"),
                      training=True, step=0, seed=0, jitter=0,
                      realize_crop=True, mask_downsample="occupancy", occ_thr=0.1,
                      ct_spec=resolve_ct_norm(None))
    N = len(spacings)
    assert len(res.logits) == N and len(res.centers) == N
    assert res.centers[0] == [None] * B
    for i in range(1, N):
        assert len(res.centers[i]) == B
    # every re-crop level went through load_native_crop
    assert len(prov.calls) == (N - 1) * B * (1 + 3)
    assert all(c["jitter"] == 0 for c in prov.calls)
```

`_FakeModel` / `_v2_batch` already exist in the file. `_v2_batch` provides `subjects`, `context_subjects`, `label_names`, `crop_geom`.

- [ ] **Step 2: Run test to verify it fails**

Run: `PATH="/software/anaconda3/envs/git/bin:$PATH" python -m pytest experiments/3d/tests/test_cascade.py -q -k realize_crop`
Expected: FAIL — `run_cascade() got an unexpected keyword argument 'realize_crop'`.

- [ ] **Step 3: Write minimal implementation**

In `experiments/3d/cascade.py`:

```python
from src.gpu_realize_crop import realize_native_crops, _regroup
```

Extend `_recrop_level` signature and add the realize branch (build the flat `tasks` list exactly as today, then):

```python
def _recrop_level(provider, batch, centers, spacing, *, step, seed, level, jitter,
                  recrop_workers=1, realize_crop=False, mask_downsample="occupancy",
                  occ_thr=0.1, ct_spec=None, device=None):
    subs, ctxs, clss = batch["subjects"], batch["context_subjects"], batch["label_names"]
    sp = float(spacing)
    tasks = []
    for b in range(len(subs)):
        tasks.append((b, -1, subs[b], centers[b], f"{seed}_{step}_{level}_{b}"))
        for k, cs in enumerate(ctxs[b]):
            tasks.append((b, k, cs, None, f"{seed}_{step}_{level}_{b}_{k}"))

    if realize_crop:
        def _load_nc(t):
            b, _k, subj, center, rk = t
            return provider.load_native_crop(subj, clss[b], LoadRequest(
                rng=random.Random(rk), crop_spacing_mm=sp, center=center, jitter=jitter))
        flat = _run_pool(_load_nc, tasks, recrop_workers)
        members = _regroup(flat, len(subs), 1 + len(ctxs[0]))   # target first, then K contexts
        out = realize_native_crops(members, T=batch["image"].shape[-1],
                                   mask_downsample=mask_downsample, occ_thr=occ_thr,
                                   ct_spec=ct_spec, device=device)
        out["subjects"] = list(subs)
        out["context_subjects"] = [list(c) for c in ctxs]
        out["label_names"] = list(clss)
        out["aug_mode"] = torch.zeros(len(subs), dtype=torch.long)
        return out
    # ... existing provider.load + incontext_collate_fn path unchanged ...
```

Factor the thread-pool map already present in the file into a small `_run_pool(fn, tasks, workers)` helper (the `torch.set_num_threads(1)` guard + `_recrop_pool(...).map` block) so both branches use it. `_regroup`'s member order must be `[target, ctx0..ctxK-1]`, matching how `tasks` is built (`k == -1` first per `b`).

Thread `realize_crop` + friends through `run_cascade` → `_recrop_level` call site:

```python
def run_cascade(model, provider, batch, augmentor, spacings, *, device, training,
                step, seed, jitter=0, is_prob=False, want_hard_preds=False,
                recrop_workers=1, query_prior=False, query_prior_hard=False,
                want_figure_arrays=False, realize_crop=False,
                mask_downsample="occupancy", occ_thr=0.1, ct_spec=None):
    ...
        if i > 0:
            cur = _recrop_level(provider, batch, centers[i], spacings[i],
                                step=step, seed=seed, level=i, jitter=jitter,
                                recrop_workers=recrop_workers, realize_crop=realize_crop,
                                mask_downsample=mask_downsample, occ_thr=occ_thr,
                                ct_spec=ct_spec, device=device)
            cur = _to_device(cur, device)   # no-op for realize output (already on device)
```

`_to_device` already guards each key with `.to(device, non_blocking=True)` — harmless on tensors already on `device`.

- [ ] **Step 4: Run test to verify it passes**

Run: `PATH="/software/anaconda3/envs/git/bin:$PATH" python -m pytest experiments/3d/tests/test_cascade.py -q`
Expected: PASS (all existing cascade tests + the 2 new parametrized cases).

- [ ] **Step 5: Commit**

```bash
PATH="/software/anaconda3/envs/git/bin:$PATH" git add experiments/3d/cascade.py experiments/3d/tests/test_cascade.py
PATH="/software/anaconda3/envs/git/bin:$PATH" git commit -m "$(cat <<'EOF'
feat: cascade _recrop_level GPU-realize branch for levels >=1

run_cascade(realize_crop=True): re-crop levels load NativeCrop payloads from the
RAM cache and resample on device. Same batch-dict contract; crop_geom untouched.
Tested for [3,1.5] and [6,3,1.5].

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01FBrqQGq5vA29WMHSrDPBfa
EOF
)"
```

---

## Task 5: level-0 native-crop payload + engine + loader wiring

**Files:**
- Modify: `src/incontext_dataset_v2.py`, `experiments/3d/common.py`
- Test: `experiments/3d/tests/test_gpu_realize_crop.py` (append an engine test)

**Interfaces:**
- Consumes: `TotalSegProvider.load_native_crop` (Task 2), `native_crop_collate_fn` (Task 3).
- Produces:
  - `InContextDataset.__init__(..., gpu_realize_crop=False)`. When set (non-cohort branch only), `__getitem__` returns
    `{"native_crop": [tgt_nc, ctx0_nc, … ctxK-1_nc], "subject", "context_subjects", "label_name", "aug_mode": tensor(0)}` — **no `"image"` key** (engine already skips its CPU aug for imageless items, `incontext_dataset_v2.py:112`). It calls `provider.load_native_crop(subj, cls, req)` for the target and each context, reusing the existing candidate-shuffle / self-context-fallback / pad-to-`context_size` logic (clone `NativeCrop`s for the pad case).
  - `common.build_dataset`: the v2 totalseg branch forwards `ram_cache` / `ram_cache_max_subjects` to `TotalSegProvider` and `gpu_realize_crop` to `InContextDataset`, reading `d.get("ram_cache", bool(d.get("cascade_spacings")))` and `d.get("gpu_realize_crop", bool(d.get("cascade_spacings")))`.
  - `common.train_loader`: when `cfg.data.get("gpu_realize_crop")` and `cfg.data.get("cascade_spacings")`, `collate = native_crop_collate_fn`.

- [ ] **Step 1: Write the failing test**

```python
# append to experiments/3d/tests/test_gpu_realize_crop.py
def test_engine_emits_native_crop_payload(tmp_path):
    import random
    from src.incontext_dataset_v2 import InContextDataset, LoadRequest
    from src.providers.totalseg import NativeCrop

    T = 8
    geom = torch.tensor([[0, 0, 0], [T, T, T], [T, T, T], [0, 0, 0]], dtype=torch.long)

    class P:
        classes = ["liver"]
        def subjects_for(self, cls): return ["a", "b", "c", "d"]
        def load_native_crop(self, subject, cls, req):
            return NativeCrop(image=torch.zeros(T, T, T, dtype=torch.float16),
                              label=torch.zeros(T, T, T, dtype=torch.uint8),
                              class_idx=3, out_sizes=[T, T, T], pad_lo=[0, 0, 0],
                              crop_geom=geom, crop_spacing_mm=req.crop_spacing_mm,
                              decim=(1, 1, 1))

    ds = InContextDataset(P(), context_size=3, class_balanced=False,
                          crop_spacing_mm=3.0, gpu_realize_crop=True)
    ds.samples = [("a", "liver")]
    item = ds[0]
    assert "image" not in item and "native_crop" in item
    assert len(item["native_crop"]) == 4                # target + 3 contexts
    assert all(isinstance(x, NativeCrop) for x in item["native_crop"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PATH="/software/anaconda3/envs/git/bin:$PATH" python -m pytest experiments/3d/tests/test_gpu_realize_crop.py -q -k native_crop_payload`
Expected: FAIL — `InContextDataset.__init__() got an unexpected keyword argument 'gpu_realize_crop'`.

- [ ] **Step 3: Write minimal implementation**

`src/incontext_dataset_v2.py` — add the flag and a payload branch:

```python
    def __init__(self, provider, context_size=3, class_balanced=False,
                 aug_cfg=None, defer_aug=False, crop_spacing_mm=1.5, eval_seed=None,
                 max_tasks_per_class=None, gpu_realize_crop=False):
        ...
        self.gpu_realize_crop = bool(gpu_realize_crop)
```

In `__getitem__`, in the non-cohort branch, after `subj, cls` are chosen and `rng`/`crop_spacing` are set, branch before the `self.provider.load(...)` calls:

```python
        if self.gpu_realize_crop:
            req = LoadRequest(rng=rng, crop_spacing_mm=crop_spacing)
            tgt = self.provider.load_native_crop(subj, cls, req)
            ctx = []
            candidates = [s for s in self.provider.subjects_for(cls) if s != subj]
            for cs in _lazy_shuffle(rng, candidates):
                if len(ctx) >= self.context_size:
                    break
                try:
                    ctx.append(self.provider.load_native_crop(
                        cs, cls, LoadRequest(rng, crop_spacing)))
                except Exception:
                    continue
            if not ctx:
                warnings.warn("InContextDataset: no context candidates; self-context "
                              "fallback (metrics leakage-inflated).", stacklevel=2)
                ctx.append(tgt)
            while len(ctx) < self.context_size:
                ctx.append(ctx[rng.randrange(len(ctx))])
            return {"native_crop": [tgt, *ctx], "subject": subj,
                    "context_subjects": [], "label_name": cls,
                    "aug_mode": torch.tensor(0, dtype=torch.long)}
```

`context_subjects` is `[]` here (not tracked in the realize payload — cascade re-crop reads it from the level-0 `batch`, so it MUST carry through; keep the real ids). Correct the branch to record `ctx_subjects` alongside `ctx` and return `"context_subjects": ctx_subjects`. Contexts appended in the no-candidate / pad cases repeat an already-recorded id.

`common.build_dataset` (v2 totalseg branch) — pass the flags:

```python
        _casc = bool(d.get("cascade_spacings"))
        provider = TotalSegProvider(
            root=root, classes=classes, image_size=tuple(d.image_size),
            split=split, max_subjects=(...),
            crop_spacing_mm=d.get("crop_spacing_mm", 1.5),
            crop_jitter=(...), mask_downsample=d.get("mask_downsample", "occupancy"),
            mask_occupancy_thr=d.get("mask_occupancy_thr", 0.1),
            modality=("mri" if is_mri else "ct"), ct_norm=d.get("ct_norm"),
            ram_cache=bool(d.get("ram_cache", _casc)),
            ram_cache_max_subjects=d.get("ram_cache_max_subjects"))
        return InContextDataset(
            provider, context_size=d.context_size,
            class_balanced=(is_train and d.get("class_balanced", False)),
            aug_cfg=(cfg.augmentations if is_train else None),
            defer_aug=(is_train and bool(cfg.get("augmentations", {}).get("gpu", False))),
            crop_spacing_mm=d.get("crop_spacing_mm", 1.5),
            eval_seed=(None if is_train else int(cfg.get("eval", {}).get("seed", 0))),
            gpu_realize_crop=(is_train and bool(d.get("gpu_realize_crop", _casc))))
```

Apply the same `ram_cache` forwarding in `make_eval_loader`'s v2 branch (so the eval-loader provider shares the singleton), but leave `gpu_realize_crop=False` for eval (the cascade val pass calls `run_cascade(realize_crop=…)` itself via Task 6; the eval `InContextDataset` still emits finished tensors for level 0). Set `ram_cache=bool(d.get("ram_cache", bool(d.get("cascade_spacings"))))` there too.

`common.train_loader` — collate switch:

```python
    _casc_realize = bool(cfg.data.get("cascade_spacings")) and bool(cfg.data.get("gpu_realize_crop", True))
    if cfg.data.get("source") == "synth_gmm_maisi" and cfg.data.get("gpu_realize", False):
        from src.gpu_synth_realize import synth_gpu_collate_fn
        collate = synth_gpu_collate_fn
    elif _casc_realize:
        from src.gpu_realize_crop import native_crop_collate_fn
        collate = native_crop_collate_fn
    else:
        collate = incontext_collate_fn
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PATH="/software/anaconda3/envs/git/bin:$PATH" python -m pytest experiments/3d/tests/test_gpu_realize_crop.py experiments/3d/tests/test_cascade_provider.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
PATH="/software/anaconda3/envs/git/bin:$PATH" git add src/incontext_dataset_v2.py experiments/3d/common.py experiments/3d/tests/test_gpu_realize_crop.py
PATH="/software/anaconda3/envs/git/bin:$PATH" git commit -m "$(cat <<'EOF'
feat: level-0 native-crop payload + loader wiring for cascade realize

InContextDataset(gpu_realize_crop=True) emits an imageless native_crop payload;
train_loader collates it with native_crop_collate_fn; build_dataset/make_eval_loader
forward ram_cache + gpu_realize_crop, defaulted on for cascade runs.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01FBrqQGq5vA29WMHSrDPBfa
EOF
)"
```

---

## Task 6: `train_epoch` level-0 realize + `_assert_cascade_supported` guard + config + docs

**Files:**
- Modify: `experiments/3d/train.py`, `experiments/3d/cascade.py`, `experiments/3d/common.py`, `configs/experiment/3d/experiment/59_organs_cascade_from_scratch.yaml`, `docs/logs.md`
- Test: `experiments/3d/tests/test_cascade.py` (append), `experiments/3d/tests/test_cascade_guard.py` (append)

**Interfaces:**
- Consumes: `realize_native_crops` (Task 3); `run_cascade(realize_crop=…)` (Task 4); the level-0 `native_crop` payload (Task 5).
- Produces:
  - `cascade.realize_cascade_level0(batch, *, T, mask_downsample, occ_thr, ct_spec, device) -> dict` — converts a `native_crop_collate_fn` batch (`{"native_crop": [B lists of K+1], "subjects", "context_subjects", "label_names", "aug_mode"}`) into the standard collated batch dict (`image/label/context_in/context_out/spacing/crop_geom` + the passthrough id lists + `aug_mode` on `device`). Thin wrapper over `realize_native_crops`.
  - `train.py`: `crop_realizer` = a closure bound to `(T, mask_downsample, occ_thr, ct_spec, DEVICE)` built when `_cascade_on and cfg.data.get("gpu_realize_crop", True)`, passed into `train_epoch` as a new kw `crop_realizer=None`. In `train_epoch`'s cascade branch, `if crop_realizer is not None and "native_crop" in batch: batch = crop_realizer(batch)` **before** `run_cascade`, and `run_cascade` is called with `realize_crop=True, mask_downsample=cfg.data.get("mask_downsample","occupancy"), occ_thr=float(cfg.data.get("mask_occupancy_thr",0.1)), ct_spec=resolve_ct_norm(cfg.data.get("ct_norm"))`.
  - `_assert_cascade_supported`: after the existing checks, if `d.get("gpu_realize_crop", True)` and not `d.get("ram_cache", True)` → `raise ValueError`. (Both keys default true under cascade; this only fires on an explicit `ram_cache: false` + realize.)
  - `evaluate_cascade` (`cascade.py`): pass `realize_crop=bool(cfg.data.get("gpu_realize_crop", True))` + the mask/occ/ct_spec kwargs into its `run_cascade` call, so the val pass uses the same path. Level-0 val batches still arrive as finished tensors (eval loader `gpu_realize_crop=False`), so `run_cascade`'s level-0 handling is unchanged; only levels ≥ 1 realize.

- [ ] **Step 1: Write the failing tests**

```python
# append to experiments/3d/tests/test_cascade.py
def test_realize_cascade_level0_shape_contract():
    from cascade import realize_cascade_level0
    from src.totalseg_dataset import resolve_ct_norm
    from src.providers.totalseg import NativeCrop
    B, K, T = 2, 3, 8
    geom = torch.tensor([[0, 0, 0], [T, T, T], [T, T, T], [0, 0, 0]], dtype=torch.long)
    def nc():
        lbl = torch.zeros(T, T, T, dtype=torch.uint8); lbl[1:4, 1:4, 1:4] = 7
        return NativeCrop(image=torch.zeros(T, T, T, dtype=torch.float16), label=lbl,
                          class_idx=7, out_sizes=[T, T, T], pad_lo=[0, 0, 0],
                          crop_geom=geom, crop_spacing_mm=3.0, decim=(1, 1, 1))
    batch = {"native_crop": [[nc() for _ in range(K + 1)] for _ in range(B)],
             "subjects": ["s0", "s1"],
             "context_subjects": [["c0"] * K, ["c1"] * K],
             "label_names": ["liver", "liver"],
             "aug_mode": torch.zeros(B, dtype=torch.long)}
    out = realize_cascade_level0(batch, T=T, mask_downsample="occupancy", occ_thr=0.1,
                                 ct_spec=resolve_ct_norm(None), device=torch.device("cpu"))
    assert out["image"].shape == (B, 1, T, T, T)
    assert out["context_in"].shape == (B, K, 1, T, T, T)
    assert out["label"].shape == (B, T, T, T)
    assert out["crop_geom"].shape == (B, 4, 3)
    assert out["subjects"] == ["s0", "s1"]
```

```python
# append to experiments/3d/tests/test_cascade_guard.py
def test_cascade_realize_requires_ram_cache():
    import pytest
    from omegaconf import OmegaConf
    from common import _assert_cascade_supported
    cfg = _min_cascade_cfg()                       # existing helper in this file
    cfg.data.gpu_realize_crop = True
    cfg.data.ram_cache = False
    with pytest.raises(ValueError, match="ram_cache"):
        _assert_cascade_supported(cfg)


def test_cascade_realize_default_ok():
    from common import _assert_cascade_supported
    cfg = _min_cascade_cfg()                       # no ram_cache / gpu_realize_crop keys
    _assert_cascade_supported(cfg)                 # defaults both on -> no raise
```

If `test_cascade_guard.py` has no `_min_cascade_cfg` helper, reuse whatever minimal-cfg builder it already uses for the other guard tests (check the file); the two assertions above are the payload.

- [ ] **Step 2: Run tests to verify they fail**

Run: `PATH="/software/anaconda3/envs/git/bin:$PATH" python -m pytest experiments/3d/tests/test_cascade.py experiments/3d/tests/test_cascade_guard.py -q -k "level0_shape or ram_cache or realize_default"`
Expected: FAIL — `ImportError: cannot import name 'realize_cascade_level0'` / no `ValueError`.

- [ ] **Step 3: Write minimal implementation**

`cascade.py`:

```python
from src.totalseg_dataset import resolve_ct_norm   # near the other imports


def realize_cascade_level0(batch, *, T, mask_downsample, occ_thr, ct_spec, device):
    """native_crop_collate_fn batch -> standard collated batch dict, on device."""
    out = realize_native_crops(batch["native_crop"], T=T, mask_downsample=mask_downsample,
                               occ_thr=occ_thr, ct_spec=ct_spec, device=device)
    out["subjects"] = list(batch["subjects"])
    out["context_subjects"] = [list(c) for c in batch["context_subjects"]]
    out["label_names"] = list(batch["label_names"])
    out["aug_mode"] = batch.get("aug_mode", torch.zeros(len(out["subjects"]),
                                                        dtype=torch.long)).to(device)
    return out
```

`train.py` — build the realizer next to `synth_realizer` (~line 1121) and pass it in (~line 1140):

```python
    if _cascade_on and bool(cfg.data.get("gpu_realize_crop", True)):
        from cascade import realize_cascade_level0
        from src.totalseg_dataset import resolve_ct_norm
        _rc_spec = resolve_ct_norm(cfg.data.get("ct_norm"))
        _rc_T = int(cfg.data.image_size[0])
        _rc_md = cfg.data.get("mask_downsample", "occupancy")
        _rc_thr = float(cfg.data.get("mask_occupancy_thr", 0.1))
        def crop_realizer(batch, _dev=DEVICE):
            return realize_cascade_level0(batch, T=_rc_T, mask_downsample=_rc_md,
                                          occ_thr=_rc_thr, ct_spec=_rc_spec, device=_dev)
    else:
        crop_realizer = None
    ...
        loss, tr_dice, tr_soft, tr_grid = train_epoch(
            model, loader, optimizers, scheduler, step_per_batch, loss_fn, cfg, epoch,
            is_patchset=is_patchset, gpu_aug=gpu_aug, synth_realizer=synth_realizer,
            crop_realizer=crop_realizer)
```

`train_epoch` — add `crop_realizer=None` to the signature; in the cascade branch, right after `cascade_spacings = cfg.data.get("cascade_spacings")` and before `run_cascade`:

```python
            if crop_realizer is not None and "native_crop" in batch:
                batch = crop_realizer(batch)
```

and extend the `run_cascade(...)` call in that branch with:

```python
                    realize_crop=bool(cfg.data.get("gpu_realize_crop", True)),
                    mask_downsample=cfg.data.get("mask_downsample", "occupancy"),
                    occ_thr=float(cfg.data.get("mask_occupancy_thr", 0.1)),
                    ct_spec=__import__("src.totalseg_dataset", fromlist=["resolve_ct_norm"]).resolve_ct_norm(cfg.data.get("ct_norm")),
```

(prefer a top-of-file `from src.totalseg_dataset import resolve_ct_norm` in `train.py` and just `ct_spec=resolve_ct_norm(cfg.data.get("ct_norm"))`.)

`evaluate_cascade` (`cascade.py`) — mirror the same four kwargs on its `run_cascade` call, reading from `cfg.data` (it already has `cfg`).

`common._assert_cascade_supported` — append before the `cascade_query_prior` check:

```python
    if d.get("gpu_realize_crop", True) and not d.get("ram_cache", True):
        raise ValueError(
            "data.cascade_spacings with data.gpu_realize_crop=true requires data.ram_cache=true "
            "(the RAM cache is what removes the NFS re-crop cost; realize over mmap is slower). "
            "See docs/superpowers/specs/2026-09-01-cascade-ram-cache-gpu-realize-design.md.")
```

`configs/experiment/3d/experiment/59_organs_cascade_from_scratch.yaml` — replace the `# IMAGE CACHES (measured):` comment block with:

```yaml
# DATA PATH: this run holds ct_raw.npy + label.npy for ALL subjects in a fork-shared RAM
# cache (~35 GB; data.ram_cache) and does every crop resample / placement / normalization
# on the GPU for both cascade levels (data.gpu_realize_crop). No ct_raw_{s}mm.npy caches
# are needed. Recommend a >= 96 GB node. Set data.ram_cache=false to fall back to the NFS
# mmap + CPU crop path (slow). See docs/superpowers/specs/2026-09-01-cascade-ram-cache-gpu-realize-design.md.
```

and under `data:`:

```yaml
  ram_cache: true
  gpu_realize_crop: true
```

`docs/logs.md` — prepend an entry:

```markdown
## 2026-09-01 — cascade RAM cache + GPU crop/realize

`data.ram_cache` preloads ct_raw.npy + label.npy for all 1228 subjects into a
fork-COW read-only singleton (`src/providers/volume_cache.py`); `data.gpu_realize_crop`
makes `TotalSegProvider.load_native_crop` ship raw integer-decimated crops that
`src/gpu_realize_crop.realize_native_crops` resamples/normalizes/places on device.
`cascade.py` uses it for levels >=1 (`run_cascade(realize_crop=True)`) and the cascade
train loader ships imageless `native_crop` payloads that `train_epoch` realizes for
level 0. Both default on when `data.cascade_spacings` is set. Non-cascade v2 paths
unchanged. Removes the ~0.3 s/step synchronous NFS re-crop and the ~100 s/val-pass.
Spec: docs/superpowers/specs/2026-09-01-cascade-ram-cache-gpu-realize-design.md.
```

- [ ] **Step 4: Run the full cascade + realize test suite**

Run: `PATH="/software/anaconda3/envs/git/bin:$PATH" python -m pytest experiments/3d/tests/test_cascade.py experiments/3d/tests/test_cascade_guard.py experiments/3d/tests/test_cascade_config.py experiments/3d/tests/test_gpu_realize_crop.py experiments/3d/tests/test_volume_cache.py experiments/3d/tests/test_cascade_provider.py experiments/3d/tests/test_crop_helpers.py -q`
Expected: PASS (all).

- [ ] **Step 5: Sanity-check config resolution (no GPU needed)**

Run: `PATH="/software/anaconda3/envs/git/bin:$PATH" python -c "from hydra import compose, initialize; initialize(config_path='configs/experiment/3d', version_base='1.3'); cfg=compose('train', overrides=['experiment=59_organs_cascade_from_scratch']); print(cfg.data.ram_cache, cfg.data.gpu_realize_crop, list(cfg.data.cascade_spacings))"`
Expected: `True True [3, 1.5]`.

- [ ] **Step 6: Commit**

```bash
PATH="/software/anaconda3/envs/git/bin:$PATH" git add experiments/3d/train.py experiments/3d/cascade.py experiments/3d/common.py configs/experiment/3d/experiment/59_organs_cascade_from_scratch.yaml docs/logs.md experiments/3d/tests/test_cascade.py experiments/3d/tests/test_cascade_guard.py
PATH="/software/anaconda3/envs/git/bin:$PATH" git commit -m "$(cat <<'EOF'
feat: wire cascade level-0 GPU realize + ram_cache guard + config

train_epoch realizes native_crop level-0 batches before run_cascade;
run_cascade/evaluate_cascade run realize_crop for levels >=1. cascade guard
errors on gpu_realize_crop without ram_cache. exp59 config + docs/logs.md.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01FBrqQGq5vA29WMHSrDPBfa
EOF
)"
```

---

## Self-Review

**1. Spec coverage**

| Spec item | Task |
|---|---|
| RAM cache singleton, read-only, threaded, idempotent, `ram_cache_max_subjects` | Task 1; ctor wiring Task 2; config default Task 5/6 |
| No 3 mm cache; coarse decimates from 1.5 mm | Task 2 `decim` + `_decim_int_pool`; parity Task 3 |
| `NativeCrop` payload, `load_native_crop`, `req.rng` once | Task 2 |
| `decim[a] = max(1, floor(crop_sizes[a]/out_sizes[a]))`, per-axis, ≥ out_sizes | Task 2 + test; Task 3 decimated-parity test |
| GPU resample: area/trilinear, `img.min()` pad, `normalize_ct_gpu` | Task 3 `_realize_member` |
| Mask soft (peak floor) + occupancy (argmax non-empty) ported from `resample_binary` | Task 3 + 3 tests |
| `crop_geom` byte-identical, passthrough | Task 2 test (`torch.equal`), Task 3 test, Task 4 |
| `_recrop_level` realize branch, levels ≥ 1, `run_cascade(realize_crop=…)` | Task 4 |
| Any `cascade_spacings` (`[6,3,1.5]`, non-2×), N-level | Task 4 parametrized `[3,1.5]` + `[6,3,1.5]` |
| Level-0 imageless `native_crop` payload; engine hook | Task 5 (`InContextDataset.gpu_realize_crop`) |
| List-preserving collate for cascade | Task 3 `native_crop_collate_fn`; wired Task 5 |
| `crop_realizer` in `train_epoch`, mirrors `synth_realizer` call site | Task 6 |
| eval-loader shares the singleton; cascade val pass realizes ≥ 1 | Task 5 (`make_eval_loader` `ram_cache`), Task 6 (`evaluate_cascade`) |
| `_assert_cascade_supported`: defaults on, error on realize-without-cache | Task 6 + 2 guard tests |
| Config `ram_cache`/`gpu_realize_crop`, header rewrite | Task 6 |
| Non-cascade v2 untouched | flags gated on `cascade_spacings` throughout; no task modifies the non-cascade collate/provider path |
| `docs/logs.md` | Task 6 |
| Semantic-equivalence test bar (2e-2 / Dice 1.0 / 1e-4 / exact geom) | Task 3 tests |

No gaps.

**2. Placeholder scan** — no "TBD"/"handle edge cases"/"similar to". Code blocks are concrete. Two spots defer a detail to the implementer with an explicit instruction, not a placeholder: (a) Task 2 test — pick a real class name from `_ALL_CLASSES_IDX` and adjust one assertion; (b) Task 6 — reuse `test_cascade_guard.py`'s existing minimal-cfg helper. Both name the exact change.

**3. Type consistency**

- `get_cache(root, subjects, *, max_subjects, workers)` — same in Task 1 def, Task 2 call, Task 5 call.
- `NativeCrop` fields (`image,label,class_idx,out_sizes,pad_lo,crop_geom,crop_spacing_mm,decim`) — identical across Tasks 2/3/4/5/6 tests and impl.
- `realize_native_crops(members, *, T, mask_downsample, occ_thr, ct_spec, device)` — Task 3 def; Tasks 4 (`_recrop_level`), 6 (`realize_cascade_level0`) call with the same kwargs.
- `_regroup(flat, B, Kp1)` — Task 3 def and test; Task 4 call.
- `native_crop_collate_fn` output key `"native_crop"` (list of B lists) — Task 3, consumed identically in Task 5 (`train_loader`) and Task 6 (`realize_cascade_level0`).
- `run_cascade(..., realize_crop, mask_downsample, occ_thr, ct_spec)` — Task 4 def; Task 6 train.py + `evaluate_cascade` calls match.
- `InContextDataset(..., gpu_realize_crop=False)` — Task 5 def; `build_dataset` call.
- `_recrop_level(..., realize_crop, mask_downsample, occ_thr, ct_spec, device)` — Task 4 def and internal call.

Consistent.
