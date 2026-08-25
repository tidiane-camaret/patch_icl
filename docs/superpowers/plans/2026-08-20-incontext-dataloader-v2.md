# In-Context DataLoader v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a generic in-context segmentation `Dataset` engine plus a thin totalseg volume-provider adapter, replacing the tangled load/assembly logic of `TotalSegInContextDataset` with a clean, source-agnostic seam.

**Architecture:** Approach A — a `VolumeProvider` (source-specific I/O) feeds a generic `InContextDataset` (task assembly). All per-item state flows through explicit `LoadRequest`/`LoadResult` dataclasses, eliminating the mutable side-channels. The totalseg provider uses a single raw_ct organ-crop load path. The old class is untouched; v2 coexists behind a config flag.

**Tech Stack:** Python, PyTorch, NumPy, pytest, Hydra/OmegaConf (config), existing `src/augmentations.py` and the pure crop helpers in `src/totalseg_dataloader_incontext.py`.

**Spec:** `docs/superpowers/specs/2026-08-20-incontext-dataloader-v2-design.md`

## Global Constraints

- Do **not** modify `src/totalseg_dataloader_incontext.py` (the old class must keep working). Import its pure, module-level helpers only.
- Item dict schema and `incontext_collate_fn` contract are fixed: keys `image (1,T,T,T) f32`, `label (T,T,T) i64`, `context_in (K,1,T,T,T) f32`, `context_out (K,T,T,T) i64`, `subject`, `context_subjects (list[str] len K)`, `label_name`, `spacing (3,) f32`, `aug_mode (scalar i64)`, `crop_geom (4,3) i64`.
- Crop geometry math lives in exactly one function (`crop_and_place`). Never fork it.
- v2 requires native `ct_raw.npy` + `label.npy` + `spacings.json` per subject; missing `ct_raw.npy` is a hard error (no fallback).
- Tests run with the project env: `python -m pytest tests/<file>::<test> -v` (node-specific venv already active; see `feedback_python_env` memory).
- Reported output `spacing` is `crop_spacing_mm` isotropic (the crop of physical extent `T·crop_spacing_mm` is resampled to `T³`).

---

### Task 1: `crop_and_place` — the single crop-geometry function

**Files:**
- Create: `src/providers/__init__.py` (empty)
- Create: `src/providers/totalseg.py` (this task adds `crop_and_place` only)
- Test: `tests/test_incontext_v2_crop.py`

**Interfaces:**
- Consumes (imported from `src.totalseg_dataloader_incontext`): `organ_crop_arrays(ct_mm, label_mm, center, sp, *, image_size, crop_mm, jitter, rng) -> (crop_ct, crop_lbl, out_sizes, pad_lo, crop_geom)`; `place_image(crop_ct, out_sizes, pad_lo, T)`; `place_label(label_small, out_sizes, pad_lo, T)`; `resample_binary(bin_np, size, *, mode, occ_thr)`.
- Produces: `crop_and_place(image_np, label_np, class_idx, center, T, *, crop_spacing_mm, native_spacing, jitter, rng, mask_downsample, occ_thr, normalize_fn=None) -> (image_t (1,T,T,T) f32, label_t (T,T,T) i64, crop_geom (4,3) i64)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_incontext_v2_crop.py
import numpy as np
import random
import torch
from src.providers.totalseg import crop_and_place


def test_crop_and_place_shapes_and_geometry():
    D = 40
    img = (np.random.rand(D, D, D) * 100).astype(np.int16)  # raw-HU-like
    lbl = np.zeros((D, D, D), dtype=np.uint8)
    lbl[15:25, 15:25, 15:25] = 7                            # a blob of class 7
    T = 32
    image_t, label_t, geom = crop_and_place(
        img, lbl, class_idx=7, center=(20, 20, 20), T=T,
        crop_spacing_mm=1.5, native_spacing=(1.5, 1.5, 1.5),
        jitter=0, rng=random.Random(0), mask_downsample="occupancy", occ_thr=0.1)
    assert image_t.shape == (1, T, T, T)
    assert image_t.dtype == torch.float32
    assert label_t.shape == (T, T, T)
    assert label_t.dtype == torch.int64
    assert set(torch.unique(label_t).tolist()) <= {0, 1}
    assert label_t.sum() > 0                                # class 7 present in the crop
    assert geom.shape == (4, 3) and geom.dtype == torch.int64


def test_crop_and_place_thin_structure_survives_occupancy():
    D = 60
    img = np.zeros((D, D, D), dtype=np.int16)
    lbl = np.zeros((D, D, D), dtype=np.uint8)
    lbl[30, :, 30] = 3                                      # 1-voxel-thick line, class 3
    _, label_t, _ = crop_and_place(
        img, lbl, class_idx=3, center=(30, 30, 30), T=16,
        crop_spacing_mm=4.0, native_spacing=(1.0, 1.0, 1.0),
        jitter=0, rng=random.Random(0), mask_downsample="occupancy", occ_thr=0.1)
    assert label_t.sum() > 0                                # thin line not lost on downsample


def test_crop_and_place_applies_normalize_fn():
    D = 20
    img = np.full((D, D, D), 500, dtype=np.int16)
    lbl = np.zeros((D, D, D), dtype=np.uint8); lbl[10, 10, 10] = 1
    image_t, _, _ = crop_and_place(
        img, lbl, class_idx=1, center=(10, 10, 10), T=16,
        crop_spacing_mm=1.0, native_spacing=(1.0, 1.0, 1.0),
        jitter=0, rng=random.Random(0), mask_downsample="nearest", occ_thr=0.5,
        normalize_fn=lambda a: a.astype(np.float32) * 0.0 + 0.25)
    assert torch.allclose(image_t[image_t != image_t.min()],
                          torch.tensor(0.25), atol=1e-5) or (image_t == 0.25).any()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_incontext_v2_crop.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.providers'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/providers/__init__.py  -> empty file
```

```python
# src/providers/totalseg.py
"""TotalSegmentator volume provider for the in-context dataloader v2.

Single raw_ct organ-crop load path. `crop_and_place` is the one place crop
geometry (physical extent -> crop sizes -> resample -> centre-pad) is computed,
reusing the pure helpers extracted in the v1 module.
"""
import numpy as np
import torch

from src.totalseg_dataloader_incontext import (
    organ_crop_arrays, place_image, place_label, resample_binary,
)


def crop_and_place(image_np, label_np, class_idx, center, T, *,
                   crop_spacing_mm, native_spacing, jitter, rng,
                   mask_downsample, occ_thr, normalize_fn=None):
    """Organ-centred crop of physical extent T*crop_spacing_mm around `center`,
    resampled to T^3 and centre-padded. Returns (image (1,T,T,T) f32, label
    (T,T,T) i64 binary for class_idx, crop_geom (4,3) i64).

    `normalize_fn`, when given, maps the cropped raw image slice to model input
    space BEFORE placement (so the air-pad value matches the normalized min)."""
    crop_ct, crop_lbl, out_sizes, pad_lo, geom = organ_crop_arrays(
        image_np, label_np, center, list(native_spacing),
        image_size=(T, T, T), crop_mm=crop_spacing_mm, jitter=jitter, rng=rng)
    crop_ct = np.ascontiguousarray(crop_ct)
    if normalize_fn is not None:
        crop_ct = normalize_fn(crop_ct)
    image_t = place_image(crop_ct, out_sizes, pad_lo, T)
    lbl_small = resample_binary(crop_lbl == class_idx, tuple(out_sizes),
                                mode=mask_downsample, occ_thr=occ_thr)
    label_t = place_label(lbl_small, out_sizes, pad_lo, T).long()
    return image_t, label_t, geom
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_incontext_v2_crop.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/providers/__init__.py src/providers/totalseg.py tests/test_incontext_v2_crop.py
git commit -m "feat(dataloader-v2): add crop_and_place single crop-geometry fn"
```

---

### Task 2: `TotalSegProvider` + `LoadRequest`/`LoadResult`/`VolumeProvider`

**Files:**
- Create: `src/incontext_dataset_v2.py` (this task adds the dataclasses + protocol only)
- Modify: `src/providers/totalseg.py` (add `TotalSegProvider`)
- Test: `tests/test_incontext_v2_provider.py`

**Interfaces:**
- Consumes: `crop_and_place(...)` (Task 1); from `src.totalseg_dataloader_incontext`: `_bbox_for_subject(root, subj) -> (subj, {class: (d,h,w)})`, `_IDX_TO_CLASS: dict[int,str]`; from `src.totalseg_dataset`: `_ALL_CLASSES_IDX: dict[str,int]`, `normalize_ct(arr) -> f32 array`, `normalize_mri(arr, stats)`.
- Produces:
  - `@dataclass LoadRequest{ rng: random.Random; crop_spacing_mm: float; center: tuple|None = None }`
  - `@dataclass LoadResult{ image: Tensor; label: Tensor; spacing: Tensor; crop_geom: Tensor }`
  - `VolumeProvider(Protocol){ classes: list[str]; subjects_for(cls)->list[str]; load(subject, cls, req)->LoadResult }`
  - `TotalSegProvider(root, classes, image_size, split=None, meta_csv=None, max_subjects=None, crop_spacing_mm=1.5, crop_jitter=None, mask_downsample="occupancy", mask_occupancy_thr=0.1, modality="ct")` with attrs `.classes`, `.image_size`, method `subjects_for`, method `load`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_incontext_v2_provider.py
import json
import random
import numpy as np
import pytest
import torch

from src.incontext_dataset_v2 import LoadRequest, LoadResult
from src.providers.totalseg import TotalSegProvider
from src.totalseg_dataset import _ALL_CLASSES_IDX

_CLS, _IDX = next((c, i) for c, i in _ALL_CLASSES_IDX.items() if i > 0)


def _make_tree(root, n_subjects=3, D=48, with_raw=True):
    root.mkdir(parents=True, exist_ok=True)
    spac = {}
    for i in range(n_subjects):
        s = root / f"s{i:04d}"; s.mkdir()
        if with_raw:
            np.save(s / "ct_raw.npy", (np.random.rand(D, D, D) * 200 - 100).astype(np.int16))
        lbl = np.zeros((D, D, D), dtype=np.uint8)
        lbl[10:30, 10:30, 10:30] = _IDX                     # a blob of the target class
        np.save(s / "label.npy", lbl)
        spac[f"s{i:04d}"] = {"spacing": [1.5, 1.5, 1.5], "shape": [D, D, D]}
    json.dump(spac, open(root / "spacings.json", "w"))


def test_provider_load_returns_valid_result(tmp_path):
    root = tmp_path / "ts"; _make_tree(root)
    prov = TotalSegProvider(root=root, classes=[_CLS], image_size=(32, 32, 32),
                            crop_spacing_mm=1.5)
    assert prov.classes == [_CLS]
    assert len(prov.subjects_for(_CLS)) == 3
    req = LoadRequest(rng=random.Random(0), crop_spacing_mm=1.5)
    res = prov.load("s0000", _CLS, req)
    assert isinstance(res, LoadResult)
    assert res.image.shape == (1, 32, 32, 32) and res.image.dtype == torch.float32
    assert res.label.shape == (32, 32, 32) and res.label.dtype == torch.int64
    assert set(torch.unique(res.label).tolist()) <= {0, 1}
    assert res.label.sum() > 0
    assert res.spacing.shape == (3,)
    assert torch.allclose(res.spacing, torch.full((3,), 1.5))
    assert res.crop_geom.shape == (4, 3)


def test_provider_hard_fails_without_ct_raw(tmp_path):
    root = tmp_path / "ts"; _make_tree(root, with_raw=False)
    prov = TotalSegProvider(root=root, classes=[_CLS], image_size=(32, 32, 32))
    req = LoadRequest(rng=random.Random(0), crop_spacing_mm=1.5)
    with pytest.raises((FileNotFoundError, AssertionError)):
        prov.load("s0000", _CLS, req)


def test_provider_request_center_overrides_centroid(tmp_path):
    root = tmp_path / "ts"; _make_tree(root)
    prov = TotalSegProvider(root=root, classes=[_CLS], image_size=(32, 32, 32),
                            crop_jitter=0)
    req_c = LoadRequest(rng=random.Random(0), crop_spacing_mm=1.5, center=(0, 0, 0))
    req_d = LoadRequest(rng=random.Random(0), crop_spacing_mm=1.5)   # centroid
    g_corner = prov.load("s0000", _CLS, req_c).crop_geom
    g_default = prov.load("s0000", _CLS, req_d).crop_geom
    assert not torch.equal(g_corner, g_default)                     # center changed the crop
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_incontext_v2_provider.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.incontext_dataset_v2'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/incontext_dataset_v2.py  (dataclasses + protocol; engine added in Task 3)
"""Generic in-context segmentation dataset engine (v2).

A source-agnostic `InContextDataset` assembles items from a `VolumeProvider`.
Per-item state flows through `LoadRequest`/`LoadResult`, so there is no mutable
instance side-channel (contrast the v1 `_cur_rng`/`_last_crop_geom`).
"""
import random
from dataclasses import dataclass
from typing import Optional, Protocol

import torch


@dataclass
class LoadRequest:
    rng: random.Random                 # per-item RNG (eval determinism or global)
    crop_spacing_mm: float             # physical crop pitch for THIS item
    center: Optional[tuple] = None     # native-voxel crop center; None -> provider default
                                       # (cascade fine-crop seam; v2 always passes None)


@dataclass
class LoadResult:
    image: torch.Tensor                # (1, T, T, T) f32, normalized
    label: torch.Tensor               # (T, T, T) i64, binary {0,1}
    spacing: torch.Tensor              # (3,) mm/voxel of the output
    crop_geom: torch.Tensor            # (4, 3) i64: starts, crop_sizes, out_sizes, pad_lo


class VolumeProvider(Protocol):
    classes: list
    def subjects_for(self, cls: str) -> list: ...
    def load(self, subject: str, cls: str, req: LoadRequest) -> LoadResult: ...
```

```python
# src/providers/totalseg.py  (append TotalSegProvider; keep crop_and_place)
import csv
import hashlib
import os
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from src.incontext_dataset_v2 import LoadRequest, LoadResult
from src.totalseg_dataloader_incontext import _bbox_for_subject, _IDX_TO_CLASS
from src.totalseg_dataset import _ALL_CLASSES_IDX, normalize_ct, normalize_mri


class TotalSegProvider:
    """Source-specific I/O for the totalseg family: scan + bbox caches and a single
    raw_ct organ-crop `load`. Missing ct_raw.npy is a hard error."""

    def __init__(self, root, classes, image_size, split=None, meta_csv=None,
                 max_subjects=None, crop_spacing_mm=1.5, crop_jitter=None,
                 mask_downsample="occupancy", mask_occupancy_thr=0.1, modality="ct"):
        assert modality in ("ct", "mri"), modality
        self.root = Path(root)
        self.classes = list(classes)
        self.image_size = tuple(image_size)
        self.T = self.image_size[0]
        self.crop_spacing_mm = float(crop_spacing_mm)
        self.crop_jitter = (crop_jitter if crop_jitter is not None else self.T // 4)
        self.mask_downsample = mask_downsample
        self.mask_occupancy_thr = float(mask_occupancy_thr)
        self.modality = modality

        subjects = self._subjects(split, meta_csv, max_subjects)
        scan = self._load_or_build_scan()
        cls_set = set(self.classes)
        self._label_to_subjects = {c: [] for c in self.classes}
        for s in subjects:
            for c in scan.get(s, frozenset()):
                if c in cls_set:
                    self._label_to_subjects[c].append(s)
        self._bbox = self._load_or_build_bbox()
        self._spacings = self._load_spacings()
        self._ct_stats = self._load_ct_stats() if modality == "mri" else {}

    # --- public API ---------------------------------------------------------
    def subjects_for(self, cls):
        return self._label_to_subjects.get(cls, [])

    def load(self, subject, cls, req: LoadRequest) -> LoadResult:
        from src.providers.totalseg import crop_and_place  # same module
        subj_dir = self.root / subject
        raw = subj_dir / "ct_raw.npy"
        if not raw.exists():
            raise FileNotFoundError(f"{raw} missing (v2 requires ct_raw.npy)")
        image_np = np.load(raw, mmap_mode="r")
        label_np = np.load(subj_dir / "label.npy", mmap_mode="r")
        center = req.center
        if center is None:
            D, H, W = label_np.shape
            center = self._bbox.get(subject, {}).get(cls, (D // 2, H // 2, W // 2))
        native_sp = self._spacings.get(subject, (1.0, 1.0, 1.0))
        norm = (normalize_ct if self.modality == "ct"
                else (lambda a: normalize_mri(a, self._ct_stats[subject])))
        image_t, label_t, geom = crop_and_place(
            image_np, label_np, _ALL_CLASSES_IDX.get(cls, -1), center, self.T,
            crop_spacing_mm=req.crop_spacing_mm, native_spacing=native_sp,
            jitter=self.crop_jitter, rng=req.rng,
            mask_downsample=self.mask_downsample, occ_thr=self.mask_occupancy_thr,
            normalize_fn=lambda a: norm(np.ascontiguousarray(a)))
        spacing = torch.full((3,), float(req.crop_spacing_mm), dtype=torch.float32)
        return LoadResult(image=image_t, label=label_t, spacing=spacing, crop_geom=geom)

    # --- subjects + caches --------------------------------------------------
    def _subjects(self, split, meta_csv, max_subjects):
        alls = sorted(p.name for p in self.root.iterdir() if p.is_dir())
        if split is not None:
            csv_path = Path(meta_csv) if meta_csv else self.root / "meta.csv"
            valid = set()
            with open(csv_path, encoding="utf-8-sig") as f:
                for row in csv.DictReader(f, delimiter=";"):
                    if row["split"].strip() == split:
                        valid.add(row["image_id"].strip())
            alls = [s for s in alls if s in valid]
        return alls[:max_subjects] if max_subjects is not None else alls

    def _key(self):
        alls = sorted(p.name for p in self.root.iterdir() if p.is_dir())
        return hashlib.sha256("|".join(alls).encode()).hexdigest()[:12]

    def _load_or_build_scan(self):
        path = self.root / f".scan_cache_{self._key()}.pkl"
        if path.exists():
            with open(path, "rb") as f:
                return pickle.load(f)
        cache = {}
        for s in (p.name for p in self.root.iterdir() if p.is_dir()):
            lp = self.root / s / "label.npy"
            if not lp.exists():
                continue
            try:
                idxs = set(np.unique(np.load(lp, mmap_mode="r")))
            except (EOFError, ValueError, OSError):
                continue
            cache[s] = frozenset(_IDX_TO_CLASS[i] for i in idxs if i in _IDX_TO_CLASS)
        with open(path, "wb") as f:
            pickle.dump(cache, f)
        return cache

    def _load_or_build_bbox(self):
        path = self.root / f".bbox_cache_{self._key()}.pkl"
        if path.exists():
            with open(path, "rb") as f:
                return pickle.load(f)
        subs = [p.name for p in self.root.iterdir()
                if p.is_dir() and (p / "label.npy").exists()]
        cache = {}
        with ProcessPoolExecutor(max_workers=min(16, os.cpu_count() or 1)) as ex:
            futs = {ex.submit(_bbox_for_subject, self.root, s): s for s in subs}
            for fut in as_completed(futs):
                s, res = fut.result()
                if res is not None:
                    cache[s] = res
        with open(path, "wb") as f:
            pickle.dump(cache, f)
        return cache

    def _load_spacings(self):
        import json
        path = self.root / "spacings.json"
        if not path.exists():
            return {}
        raw = json.load(open(path))
        return {s: tuple(float(x) for x in m["spacing"]) for s, m in raw.items()}

    def _load_ct_stats(self):
        import json
        path = self.root / "ct_stats.json"
        return json.load(open(path)) if path.exists() else {}
```

Add `import numpy as np` at the top of `src/providers/totalseg.py` if not already present (Task 1 added it).

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_incontext_v2_provider.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/incontext_dataset_v2.py src/providers/totalseg.py tests/test_incontext_v2_provider.py
git commit -m "feat(dataloader-v2): add LoadRequest/LoadResult + TotalSegProvider"
```

---

### Task 3: `InContextDataset` engine

**Files:**
- Modify: `src/incontext_dataset_v2.py` (add the `InContextDataset` class + `_lazy_shuffle` import)
- Test: `tests/test_incontext_v2_engine.py`

**Interfaces:**
- Consumes: `LoadRequest`, `LoadResult`, `VolumeProvider` (Task 2); `TotalSegProvider` (Task 2, for the test); from `src.totalseg_dataloader_incontext`: `_lazy_shuffle(rng, x)`, `incontext_collate_fn(batch)`; from `src.augmentations`: `apply_task_aug(images, masks, cfg)`, `apply_intensity_aug(image, cfg)`.
- Produces: `InContextDataset(provider, context_size=3, class_balanced=False, aug_cfg=None, defer_aug=False, crop_spacing_mm=1.5, eval_seed=None)` — a `torch.utils.data.Dataset` whose `__getitem__` returns the item dict schema in Global Constraints. Attr `.samples: list[(subj, cls)]`, `.active_classes: list[str]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_incontext_v2_engine.py
import json
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.incontext_dataset_v2 import InContextDataset
from src.providers.totalseg import TotalSegProvider
from src.totalseg_dataloader_incontext import incontext_collate_fn
from src.totalseg_dataset import _ALL_CLASSES_IDX

_CLS, _IDX = next((c, i) for c, i in _ALL_CLASSES_IDX.items() if i > 0)


def _make_tree(root, n_subjects=4, D=48):
    root.mkdir(parents=True, exist_ok=True)
    spac = {}
    for i in range(n_subjects):
        s = root / f"s{i:04d}"; s.mkdir()
        np.save(s / "ct_raw.npy", (np.random.rand(D, D, D) * 200 - 100).astype(np.int16))
        lbl = np.zeros((D, D, D), dtype=np.uint8); lbl[10:30, 10:30, 10:30] = _IDX
        np.save(s / "label.npy", lbl)
        spac[f"s{i:04d}"] = {"spacing": [1.5, 1.5, 1.5], "shape": [D, D, D]}
    json.dump(spac, open(root / "spacings.json", "w"))


def _ds(root, **kw):
    prov = TotalSegProvider(root=root, classes=[_CLS], image_size=(32, 32, 32),
                            crop_jitter=0)
    return InContextDataset(prov, context_size=2, **kw)


def test_engine_item_schema(tmp_path):
    root = tmp_path / "ts"; _make_tree(root)
    ds = _ds(root, eval_seed=0)
    assert len(ds.samples) == 4                       # one (subject, _CLS) each
    it = ds[0]
    assert it["image"].shape == (1, 32, 32, 32)
    assert it["label"].shape == (32, 32, 32)
    assert it["context_in"].shape == (2, 1, 32, 32, 32)
    assert it["context_out"].shape == (2, 32, 32, 32)
    assert len(it["context_subjects"]) == 2
    assert it["label_name"] == _CLS
    assert it["spacing"].shape == (3,)
    assert it["crop_geom"].shape == (4, 3)
    assert int(it["aug_mode"]) == 0
    b = incontext_collate_fn([ds[0], ds[1]])
    assert b["image"].shape == (2, 1, 32, 32, 32)
    assert b["context_in"].shape == (2, 2, 1, 32, 32, 32)


def test_engine_eval_seed_reproducible(tmp_path):
    root = tmp_path / "ts"; _make_tree(root)
    ds = _ds(root, eval_seed=0)
    a, b = ds[1], ds[1]
    assert torch.equal(a["image"], b["image"])
    assert a["context_subjects"] == b["context_subjects"]


def test_engine_spacing_tuple_index(tmp_path):
    root = tmp_path / "ts"; _make_tree(root)
    ds = _ds(root, eval_seed=0)
    it = ds[(0, 3.0)]                                  # (idx, spacing) from SpacingBatchSampler
    assert torch.allclose(it["spacing"], torch.full((3,), 3.0))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_incontext_v2_engine.py -v`
Expected: FAIL — `ImportError: cannot import name 'InContextDataset'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/incontext_dataset_v2.py  (append)
import warnings

from torch.utils.data import Dataset

from src.totalseg_dataloader_incontext import _lazy_shuffle
from src.augmentations import apply_task_aug, apply_intensity_aug


class InContextDataset(Dataset):
    """Generic in-context task assembler over a VolumeProvider."""

    def __init__(self, provider, context_size=3, class_balanced=False,
                 aug_cfg=None, defer_aug=False, crop_spacing_mm=1.5, eval_seed=None):
        self.provider = provider
        self.context_size = int(context_size)
        self.class_balanced = bool(class_balanced)
        self.aug_cfg = aug_cfg
        self.defer_aug = bool(defer_aug)
        self.crop_spacing_mm = float(crop_spacing_mm)
        self.eval_seed = eval_seed
        self.samples = [(s, c) for c in provider.classes
                        for s in provider.subjects_for(c)]
        self.active_classes = [c for c in provider.classes if provider.subjects_for(c)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        crop_spacing = self.crop_spacing_mm
        if isinstance(idx, (tuple, list)):
            idx, crop_spacing = int(idx[0]), float(idx[1])
        rng = (random.Random(hash((self.eval_seed, idx)))
               if self.eval_seed is not None else random)

        if self.class_balanced:
            cls = rng.choice(self.active_classes)
            subj = rng.choice(self.provider.subjects_for(cls))
        else:
            subj, cls = self.samples[idx]

        req = LoadRequest(rng=rng, crop_spacing_mm=crop_spacing)
        tgt = self.provider.load(subj, cls, req)
        image_t, label_t = tgt.image, tgt.label

        context_in, context_out, ctx_subjects = [], [], []
        candidates = [s for s in self.provider.subjects_for(cls) if s != subj]
        for cs in _lazy_shuffle(rng, candidates):
            if len(context_in) >= self.context_size:
                break
            try:
                r = self.provider.load(cs, cls, LoadRequest(rng, crop_spacing))
            except Exception:
                continue
            context_in.append(r.image); context_out.append(r.label); ctx_subjects.append(cs)

        if not context_in:
            warnings.warn("InContextDataset: no context candidates; self-context "
                          "fallback (metrics leakage-inflated).", stacklevel=2)
            context_in.append(image_t.clone()); context_out.append(label_t.clone())
            ctx_subjects.append(subj)
        while len(context_in) < self.context_size:
            i = rng.randrange(len(context_in))
            context_in.append(context_in[i].clone())
            context_out.append(context_out[i].clone())
            ctx_subjects.append(ctx_subjects[i])

        if self.aug_cfg is not None and getattr(self.aug_cfg, "enabled", False) and not self.defer_aug:
            imgs = torch.cat([image_t.unsqueeze(0), torch.stack(context_in)], dim=0)
            msks = torch.cat([label_t.unsqueeze(0), torch.stack(context_out)], dim=0)
            imgs, msks = apply_task_aug(imgs, msks, self.aug_cfg.task)
            for i in range(imgs.shape[0]):
                imgs[i] = apply_intensity_aug(imgs[i], self.aug_cfg.intensity)
            image_t, label_t = imgs[0], msks[0]
            context_in, context_out = list(imgs[1:]), list(msks[1:])

        return {
            "image": image_t,
            "label": label_t,
            "context_in": torch.stack(context_in),
            "context_out": torch.stack(context_out),
            "subject": subj,
            "context_subjects": ctx_subjects,
            "label_name": cls,
            "spacing": tgt.spacing,
            "aug_mode": torch.tensor(0, dtype=torch.long),
            "crop_geom": tgt.crop_geom,
        }
```

Note: the `aug_cfg` passed here is expected to already be a `SimpleNamespace` (the v1 `_to_ns` conversion) or a DictConfig with `.enabled`, `.task`, `.intensity`. The wiring task passes the same object v1 uses.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_incontext_v2_engine.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/incontext_dataset_v2.py tests/test_incontext_v2_engine.py
git commit -m "feat(dataloader-v2): add generic InContextDataset engine"
```

---

### Task 4: Wire v2 into `common.py` behind `data.loader_v2`

**Files:**
- Modify: `experiments/3d/common.py` (add a v2 branch in `build_dataset`)
- Test: `tests/test_incontext_v2_wiring.py`

**Interfaces:**
- Consumes: `InContextDataset` (Task 3), `TotalSegProvider` (Task 2), existing `_source_root(cfg)`, `resolve_classes`, `_to_ns` conversion via `cfg.augmentations`.
- Produces: `build_dataset(cfg, split)` returns an `InContextDataset` when `cfg.data.loader_v2` is truthy and `cfg.data.source` is in the totalseg family; unchanged otherwise.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_incontext_v2_wiring.py
import json
import numpy as np
from omegaconf import OmegaConf

from src.incontext_dataset_v2 import InContextDataset
from src.totalseg_dataloader_incontext import TotalSegInContextDataset
from src.totalseg_dataset import _ALL_CLASSES_IDX

import sys, pathlib
sys.path.insert(0, str(pathlib.Path("experiments/3d").resolve()))
from common import build_dataset  # noqa: E402

_CLS, _IDX = next((c, i) for c, i in _ALL_CLASSES_IDX.items() if i > 0)


def _make_tree(root, n=3, D=48):
    root.mkdir(parents=True, exist_ok=True)
    spac = {}
    for i in range(n):
        s = root / f"s{i:04d}"; s.mkdir()
        np.save(s / "ct_raw.npy", (np.random.rand(D, D, D) * 200 - 100).astype(np.int16))
        lbl = np.zeros((D, D, D), dtype=np.uint8); lbl[10:30, 10:30, 10:30] = _IDX
        np.save(s / "label.npy", lbl)
        spac[f"s{i:04d}"] = {"spacing": [1.5, 1.5, 1.5], "shape": [D, D, D]}
    json.dump(spac, open(root / "spacings.json", "w"))


def _cfg(root, loader_v2):
    return OmegaConf.create({
        "data": {"source": "totalseg", "image_size": [32, 32, 32], "context_size": 2,
                 "val_classes": [_CLS], "train_classes": [_CLS], "use_crop": True,
                 "crop_spacing_mm": 1.5, "class_balanced": False,
                 "max_val_subjects": None, "max_train_subjects": None,
                 "loader_v2": loader_v2},
        "paths": {"totalseg": str(root)},
        "eval": {"seed": 0},
    })


def test_build_dataset_v2_flag(tmp_path):
    root = tmp_path / "ts"; _make_tree(root)
    ds = build_dataset(_cfg(root, True), "val")
    assert isinstance(ds, InContextDataset)
    it = ds[0]
    assert it["image"].shape == (1, 32, 32, 32)


def test_build_dataset_default_is_v1(tmp_path):
    root = tmp_path / "ts"; _make_tree(root)
    # v1 needs pre-resized files; just assert the TYPE routing, not a full load
    ds = build_dataset(_cfg(root, False), "val")
    assert isinstance(ds, TotalSegInContextDataset)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_incontext_v2_wiring.py -v`
Expected: FAIL — `test_build_dataset_v2_flag` returns a `TotalSegInContextDataset`, not `InContextDataset`.

- [ ] **Step 3: Write minimal implementation**

In `experiments/3d/common.py`, at the **top of `build_dataset`** (before the existing `omnisynth3d` check), add:

```python
    d = cfg.data
    if d.get("loader_v2", False) and d.get("source", "totalseg") in _TOTALSEG_SOURCES:
        from src.incontext_dataset_v2 import InContextDataset
        from src.providers.totalseg import TotalSegProvider
        _, root, is_mri = _source_root(cfg)
        is_train = split == "train"
        class_spec = d.train_classes if is_train else d.val_classes
        classes = resolve_classes(class_spec, root, is_mri=is_mri)
        provider = TotalSegProvider(
            root=root, classes=classes, image_size=tuple(d.image_size),
            split=split, max_subjects=(d.get("max_train_subjects") if is_train
                                       else d.get("max_val_subjects")),
            crop_spacing_mm=d.get("crop_spacing_mm", 1.5),
            crop_jitter=(None if is_train else cfg.get("eval", {}).get("crop_jitter", 0)),
            mask_downsample=d.get("mask_downsample", "occupancy"),
            mask_occupancy_thr=d.get("mask_occupancy_thr", 0.1),
            modality=("mri" if is_mri else "ct"))
        return InContextDataset(
            provider, context_size=d.context_size,
            class_balanced=(is_train and d.get("class_balanced", False)),
            aug_cfg=(cfg.augmentations if is_train else None),
            defer_aug=(is_train and bool(cfg.get("augmentations", {}).get("gpu", False))),
            crop_spacing_mm=d.get("crop_spacing_mm", 1.5),
            eval_seed=(None if is_train else int(cfg.get("eval", {}).get("seed", 0))))
```

(The existing body of `build_dataset` keeps its own `d = cfg.data` further down; the new block returns before reaching it, so the duplicate local is harmless. If a linter objects, remove the later `d = cfg.data` line since this one now dominates.)

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_incontext_v2_wiring.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Run the full v2 test suite + commit**

Run: `python -m pytest tests/test_incontext_v2_crop.py tests/test_incontext_v2_provider.py tests/test_incontext_v2_engine.py tests/test_incontext_v2_wiring.py -v`
Expected: all PASS.

```bash
git add experiments/3d/common.py tests/test_incontext_v2_wiring.py
git commit -m "feat(dataloader-v2): route build_dataset to v2 behind data.loader_v2"
```

---

### Task 5: Log the change

**Files:**
- Modify: `docs/logs.md`

- [ ] **Step 1: Append a dated entry**

Add to `docs/logs.md` (top of the log, matching existing format):

```markdown
## 2026-08-20 — In-context dataloader v2

Added `src/incontext_dataset_v2.py` (`InContextDataset` engine + `LoadRequest`/
`LoadResult`/`VolumeProvider`) and `src/providers/totalseg.py` (`TotalSegProvider`
+ `crop_and_place`). Generic task-assembly separated from source I/O; single
raw_ct organ-crop load path; per-item state via dataclasses (no more
`_cur_rng`/`_last_crop_geom` side-channels). Gated behind `data.loader_v2`; the
v1 `TotalSegInContextDataset` is untouched. Spec:
docs/superpowers/specs/2026-08-20-incontext-dataloader-v2-design.md.
```

- [ ] **Step 2: Commit**

```bash
git add docs/logs.md
git commit -m "docs: log in-context dataloader v2"
```

---

## Self-Review

**Spec coverage:**
- Single raw_ct load path → Task 1 (`crop_and_place`) + Task 2 (`TotalSegProvider.load`, hard-fail on missing `ct_raw.npy`). ✓
- Provider interface (`LoadRequest`/`LoadResult`/`VolumeProvider`) → Task 2. ✓
- Pure single-source crop function → Task 1. ✓
- Generic engine (sampling, context, aug, packaging, eval determinism, `(idx,spacing)`) → Task 3. ✓
- Item schema + collate reuse → Task 3 test asserts schema + `incontext_collate_fn`. ✓
- Cascade anticipation hooks: `center` in `LoadRequest` (Task 2, tested), one pure `crop_and_place` (Task 1), `LoadResult` extensible (dataclass) → covered; buffer variant is documented future work, no task (correctly out of scope). ✓
- Coexistence behind `data.loader_v2`, old class untouched → Task 4 (both routing tests). ✓
- Non-goals (synth/self_context/multi-label/cascade/fast-slow paths) → not implemented, by construction. ✓

**Placeholder scan:** No TBD/TODO; all steps have runnable code and explicit expected results.

**Type consistency:** `LoadRequest(rng, crop_spacing_mm, center)` and `LoadResult(image, label, spacing, crop_geom)` are used identically in Tasks 2–4. `crop_and_place(...)` signature matches between Task 1 definition and Task 2 call. `InContextDataset(provider, context_size, class_balanced, aug_cfg, defer_aug, crop_spacing_mm, eval_seed)` matches between Task 3 and Task 4. ✓

One risk to verify during execution: `apply_task_aug`/`apply_intensity_aug` expect config attribute access (`cfg.task`, `cfg.flip`, ...). When `aug_cfg` is a raw OmegaConf `DictConfig` (as passed by `common.py`), attribute access works; the v1 dataset converts to `SimpleNamespace` via `_to_ns` for speed. v2 leaves that optimization out initially (YAGNI); if the augment path shows omegaconf overhead in profiling, port `_to_ns` into the engine `__init__`.
