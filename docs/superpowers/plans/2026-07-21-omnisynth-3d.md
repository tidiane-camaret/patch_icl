# omniSynth 3D Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend omniSynth to compose 3D in-context scenes by painting bbox-cropped TotalSegmentator organs at random 3D positions on a `D×H×W` canvas, feeding the existing 3D training pipeline unchanged.

**Architecture:** A one-time build script crops every organ from the pre-resized TotalSeg volumes into per-class fp16 tile caches on disk. `TotalSegObjectBank` reads those small caches at train time (LRU across classes — no full-volume reads in the hot path). A parallel `render3d.py` composites tiles onto a 3D canvas with contour-based anti-overlap. `OmniSynth3DICLDataset` orchestrates query + K contexts and emits the `TotalSegInContextDataset` contract, wired into `experiments/3d/common.py` as `source=omnisynth3d`.

**Tech Stack:** Python, NumPy, SciPy (`ndimage.zoom`), PyTorch, Hydra/OmegaConf, pytest.

## Global Constraints

- **Design spec:** `docs/superpowers/specs/2026-07-21-omnisynth-3d-design.md` — the source of truth.
- **Output contract** (must match `src/totalseg_dataloader_incontext.py`, consumed by `incontext_collate_fn`): `image` `(1,D,H,W)` float32; `label` `(D,H,W)` int64 (no channel dim); `context_in` `(K,1,D,H,W)` float32; `context_out` `(K,D,H,W)` int64; `subject` str; `label_name` str; `spacing` `(3,)` float32.
- **Rendition tile:** `[2, T, T, T]` fp16 — ch0 = intensity (zeroed outside mask), ch1 = binary mask {0,1}.
- **Contour-accurate:** compositing, label union, and anti-overlap all operate on `mask > 0`, never the bbox rectangle.
- **v1 scope:** free placement only; `target_mode ∈ {identical, class}` (no `aug`, no 3D rotation); background black or noise-field.
- **Class labels:** `from data.totalseg_classes import ALL_CLASSES` — a `list[str]`; a class's integer label value in `label_*.npy` is `ALL_CLASSES.index(name) + 1`.
- **2D path untouched:** do not modify `render.py`, `dataset.py`'s `OmniSynthICLDataset`, or any existing 2D test.
- **Run tests with:** `cd /home/dpxuser/dev/patch_icl && python -m pytest <path> -v` (repo root on `sys.path` via the `sys.path.insert(0, ".")` line each test file already uses).

---

## File Structure

| File | Responsibility |
|---|---|
| `src/datasets/omniSynth/bank_common3d.py` (create) | Pure 3D tile builders: `make_object_tile_3d`, `crop_to_tile_3d` |
| `scripts/synth3d/build_totalseg_tiles.py` (create) | One-time offline build of per-class tile caches from pre-resized volumes |
| `src/datasets/omniSynth/bank_totalseg.py` (create) | `TotalSegObjectBank` — reads tile caches, `task_ids`/`get`/`alphabet` |
| `src/datasets/omniSynth/render3d.py` (create) | `render_scene_3d` + 3D compositing/placement helpers |
| `src/datasets/omniSynth/config.py` (modify) | Add `OmniTotalSegConfig` dataclass |
| `src/datasets/omniSynth/dataset3d.py` (create) | `OmniSynth3DICLDataset` — orchestration, emits the 3D contract |
| `src/datasets/omniSynth/__init__.py` (modify) | Export the new config + dataset |
| `experiments/3d/common.py` (modify) | `build_dataset` branch for `source=omnisynth3d` |
| `configs/experiment/3d/omnisynth3d.yaml` (create) | Hydra config selecting the source |
| `src/datasets/omniSynth/test_bank_common3d.py` (create) | Tests for tile builders |
| `src/datasets/omniSynth/test_bank_totalseg.py` (create) | Tests for the bank (synthetic cache fixture) |
| `src/datasets/omniSynth/test_render3d.py` (create) | Tests for 3D scene composition |
| `src/datasets/omniSynth/test_dataset3d.py` (create) | Integration test: contract shapes/dtypes + eval determinism |

---

## Task 1: 3D tile builders (`bank_common3d.py`)

Pure functions that turn an (intensity, mask) volume crop into a `[2,T,T,T]` fp16 tile — the 3D twin of `bank_common.py`. No I/O, fully unit-testable.

**Files:**
- Create: `src/datasets/omniSynth/bank_common3d.py`
- Test: `src/datasets/omniSynth/test_bank_common3d.py`

**Interfaces:**
- Consumes: nothing (leaf module).
- Produces:
  - `make_object_tile_3d(vol_crop, m_crop, *, source_size, image_size, size_scale=1.0, min_tile=2) -> np.ndarray` shape `[2,T,T,T]` dtype `float16`. `vol_crop` float in [0,1] `(d,h,w)`; `m_crop` bool `(d,h,w)`. Scales by `r=(image_size/source_size)*size_scale`, centers into a cube `T=max(d2,h2,w2)`, zeroes intensity outside the mask.
  - `crop_to_tile_3d(vol01, mask_bool, min_vox, **sizing) -> np.ndarray | None` — bbox-crops the object then calls `make_object_tile_3d(**sizing)`; returns `None` if `mask.sum() < min_vox` or the mask vanishes under resize.

- [ ] **Step 1: Write the failing test**

Create `src/datasets/omniSynth/test_bank_common3d.py`:

```python
import sys; sys.path.insert(0, ".")
import numpy as np
from src.datasets.omniSynth.bank_common3d import make_object_tile_3d, crop_to_tile_3d


def _blob(shape=(8, 10, 6)):
    m = np.zeros(shape, dtype=bool)
    m[2:6, 3:8, 1:5] = True
    v = np.zeros(shape, dtype=np.float32)
    v[m] = 0.5
    return v, m


def test_tile_shape_and_channels_no_resize():
    v, m = _blob()
    t = make_object_tile_3d(v, m, source_size=64, image_size=64, size_scale=1.0)
    assert t.ndim == 4 and t.shape[0] == 2 and t.dtype == np.float16
    D, H, W = t.shape[1:]
    assert D == H == W                       # centered in a cube
    assert D == max(v.shape)                 # r==1 -> tile = max bbox dim


def test_mask_binary_and_intensity_masked():
    v, m = _blob()
    t = make_object_tile_3d(v, m, source_size=64, image_size=64)
    intensity, mask = t[0].astype(np.float32), t[1].astype(np.float32)
    assert set(np.unique(mask)).issubset({0.0, 1.0})
    assert mask.sum() > 0
    assert float(intensity[mask == 0].max(initial=0.0)) == 0.0   # no texture outside mask


def test_size_scale_shrinks_tile():
    v, m = _blob()
    big = make_object_tile_3d(v, m, source_size=64, image_size=64, size_scale=1.0)
    small = make_object_tile_3d(v, m, source_size=64, image_size=64, size_scale=0.5)
    assert small.shape[1] < big.shape[1]


def test_crop_to_tile_rejects_tiny_mask():
    v = np.zeros((8, 8, 8), dtype=np.float32)
    m = np.zeros((8, 8, 8), dtype=bool)
    m[0, 0, 0] = True                        # 1 voxel
    assert crop_to_tile_3d(v, m, min_vox=8, source_size=64, image_size=64) is None


def test_crop_to_tile_crops_to_bbox():
    v = np.zeros((16, 16, 16), dtype=np.float32)
    m = np.zeros((16, 16, 16), dtype=bool)
    m[4:8, 4:9, 4:7] = True                  # bbox dims (4,5,3)
    v[m] = 0.7
    t = crop_to_tile_3d(v, m, min_vox=4, source_size=64, image_size=64)
    assert t is not None and t.shape[1] == 5     # tile = max bbox dim


if __name__ == "__main__":
    test_tile_shape_and_channels_no_resize()
    test_mask_binary_and_intensity_masked()
    test_size_scale_shrinks_tile()
    test_crop_to_tile_rejects_tiny_mask()
    test_crop_to_tile_crops_to_bbox()
    print("ALL BANK_COMMON3D TESTS PASSED")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest src/datasets/omniSynth/test_bank_common3d.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.datasets.omniSynth.bank_common3d'`

- [ ] **Step 3: Write minimal implementation**

Create `src/datasets/omniSynth/bank_common3d.py`:

```python
"""3D tile builders for the TotalSegmentator object bank — the volumetric twin of
bank_common.py. Turn an (intensity, mask) volume crop into a [2, T, T, T] fp16
rendition (ch0 = intensity zeroed outside the mask, ch1 = binary mask), centered in
a cube. Pure functions, no I/O."""

import numpy as np
from scipy.ndimage import zoom as nd_zoom


def make_object_tile_3d(vol_crop, m_crop, *, source_size, image_size,
                        size_scale=1.0, min_tile=2):
    """(intensity crop [0,1] (d,h,w), bool mask (d,h,w)) -> [2,T,T,T] float16.

    Scales by r = (image_size/source_size)*size_scale so the object keeps its size
    relative to the canvas (aspect preserved), then centers it in a cube of side
    T = max(scaled dims). Intensity is zeroed outside the mask."""
    r = (float(image_size) / float(source_size)) * float(size_scale)
    d, h, w = m_crop.shape
    d2 = int(min(image_size, max(min_tile, round(d * r))))
    h2 = int(min(image_size, max(min_tile, round(h * r))))
    w2 = int(min(image_size, max(min_tile, round(w * r))))

    if (d2, h2, w2) != (d, h, w):
        zf = (d2 / d, h2 / h, w2 / w)
        m_res = nd_zoom(m_crop.astype(np.float32), zf, order=1)
        v_res = nd_zoom(vol_crop.astype(np.float32), zf, order=1)
    else:
        m_res = m_crop.astype(np.float32)
        v_res = vol_crop.astype(np.float32)

    mb = m_res >= 0.5
    if not mb.any():                      # thin mask blurred below 0.5 under resize
        mb = m_res > 0                    # keep any coverage (stay non-empty)
    v_res = np.clip(v_res, 0.0, 1.0) * mb

    tile = max(d2, h2, w2)
    off = ((tile - d2) // 2, (tile - h2) // 2, (tile - w2) // 2)
    out = np.zeros((2, tile, tile, tile), dtype=np.float16)
    sl = tuple(slice(o, o + s) for o, s in zip(off, (d2, h2, w2)))
    out[(0,) + sl] = v_res.astype(np.float16)
    out[(1,) + sl] = mb.astype(np.float16)
    return out


def crop_to_tile_3d(vol01, mask_bool, min_vox, **sizing):
    """Bbox-crop an organ from a full (vol01, bool mask) and build its tile.
    Returns None when the mask is smaller than `min_vox` or vanishes under resize."""
    if int(mask_bool.sum()) < min_vox:
        return None
    zs, ys, xs = np.nonzero(mask_bool)
    z0, z1 = zs.min(), zs.max() + 1
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    tile = make_object_tile_3d(vol01[z0:z1, y0:y1, x0:x1],
                               mask_bool[z0:z1, y0:y1, x0:x1], **sizing)
    if tile[1].sum() == 0:
        return None
    return tile
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest src/datasets/omniSynth/test_bank_common3d.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add src/datasets/omniSynth/bank_common3d.py src/datasets/omniSynth/test_bank_common3d.py
git commit -m "feat(omnisynth3d): 3D object-tile builders (bank_common3d)"
```

---

## Task 2: Tile-cache build script (`build_totalseg_tiles.py`)

Offline script: for a split, read pre-resized `label_{D}x{H}x{W}.npy` + `ct_{D}x{H}x{W}.npy`, crop each organ once, and write per-class fp16 tile caches plus an index. A small helper `build_tiles_for_split(...)` holds the logic so it is testable on a synthetic subject directory.

**Files:**
- Create: `scripts/synth3d/build_totalseg_tiles.py`
- Test: `src/datasets/omniSynth/test_build_totalseg_tiles.py`

**Interfaces:**
- Consumes: `make_object_tile_3d`, `crop_to_tile_3d` (Task 1).
- Produces:
  - `subjects_for_split(root, split) -> list[str]` — subject dir names in `meta.csv` for `split` (semicolon-delimited, `utf-8-sig`, columns `image_id`,`split`); `split=None` → all subject dirs.
  - `build_tiles_for_split(root, out_root, size, split, *, max_renditions=200, min_vox=8, size_scale=1.0, classes=None) -> Path` — writes `<out_root>/T{size[0]}/{split}/class_{lv}.pkl` (`{"name": str, "tiles": list[np.float16 [2,T,T,T]]}`) and `index.pkl` (`{lv: name}`); returns the split dir `Path`. `size` is `(D,H,W)`; reads `label_{D}x{H}x{W}.npy` / `ct_{D}x{H}x{W}.npy`. `classes` = optional list of class names to restrict to.
  - CLI: `python scripts/synth3d/build_totalseg_tiles.py --root R --out O --size D H W --split train [--max-renditions N --min-vox M --size-scale S --classes a b ...]`

- [ ] **Step 1: Write the failing test**

Create `src/datasets/omniSynth/test_build_totalseg_tiles.py`:

```python
import sys; sys.path.insert(0, ".")
import pickle
from pathlib import Path

import numpy as np

from scripts.synth3d.build_totalseg_tiles import (
    subjects_for_split, build_tiles_for_split)
from data.totalseg_classes import ALL_CLASSES


def _make_fake_root(tmp_path, size=(16, 16, 16)):
    """Two subjects, each a pre-resized ct/label cube with two organs (label values
    1 and 3). meta.csv puts both in the train split."""
    D, H, W = size
    lv_a = 1                                 # ALL_CLASSES[0]
    lv_b = 3                                 # ALL_CLASSES[2]
    for i, subj in enumerate(("s0000", "s0001")):
        d = tmp_path / subj
        d.mkdir()
        lab = np.zeros(size, dtype=np.uint8)
        lab[2:7, 2:7, 2:7] = lv_a
        lab[9:13, 9:13, 9:13] = lv_b
        ct = (np.random.default_rng(i).random(size) * 255).astype(np.float16)
        np.save(d / f"label_{D}x{H}x{W}.npy", lab)
        np.save(d / f"ct_{D}x{H}x{W}.npy", ct)
    (tmp_path / "meta.csv").write_text(
        "image_id;split\ns0000;train\ns0001;train\n", encoding="utf-8")
    return lv_a, lv_b


def test_subjects_for_split(tmp_path):
    _make_fake_root(tmp_path)
    assert subjects_for_split(tmp_path, "train") == ["s0000", "s0001"]
    assert subjects_for_split(tmp_path, "val") == []


def test_build_writes_index_and_class_files(tmp_path):
    lv_a, lv_b = _make_fake_root(tmp_path)
    out = tmp_path / "tiles"
    split_dir = build_tiles_for_split(tmp_path, out, (16, 16, 16), "train",
                                      max_renditions=10, min_vox=4)
    index = pickle.loads((split_dir / "index.pkl").read_bytes())
    assert set(index) == {lv_a, lv_b}
    assert index[lv_a] == ALL_CLASSES[lv_a - 1]
    data = pickle.loads((split_dir / f"class_{lv_a}.pkl").read_bytes())
    assert data["name"] == ALL_CLASSES[lv_a - 1]
    assert len(data["tiles"]) == 2                       # one rendition per subject
    t = data["tiles"][0]
    assert t.shape[0] == 2 and t.dtype == np.float16
    assert set(np.unique(t[1].astype(np.float32))).issubset({0.0, 1.0})


if __name__ == "__main__":
    import tempfile
    for fn in (test_subjects_for_split, test_build_writes_index_and_class_files):
        with tempfile.TemporaryDirectory() as d:
            fn(Path(d))
    print("ALL BUILD TESTS PASSED")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest src/datasets/omniSynth/test_build_totalseg_tiles.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.synth3d.build_totalseg_tiles'`

- [ ] **Step 3: Write minimal implementation**

Create `scripts/synth3d/__init__.py` (empty) and `scripts/synth3d/build_totalseg_tiles.py`:

```python
"""Precompute per-class organ tile caches for omniSynth 3D.

For a split, crop every organ once from the pre-resized label_{D}x{H}x{W}.npy +
ct_{D}x{H}x{W}.npy and write <out>/T{D}/{split}/class_{lv}.pkl (fp16 [2,T,T,T]
tiles) + index.pkl ({lv: class_name}). Built once; TotalSegObjectBank reads these
small files at train time (no full-volume reads in the hot path)."""

import argparse
import csv
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from data.totalseg_classes import ALL_CLASSES
from src.datasets.omniSynth.bank_common3d import crop_to_tile_3d


def subjects_for_split(root, split):
    root = Path(root)
    subs = sorted(p.name for p in root.iterdir() if p.is_dir())
    if split is None:
        return subs
    valid = set()
    with open(root / "meta.csv", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f, delimiter=";"):
            if row["split"].strip() == split:
                valid.add(row["image_id"].strip())
    return [s for s in subs if s in valid]


def build_tiles_for_split(root, out_root, size, split, *, max_renditions=200,
                          min_vox=8, size_scale=1.0, classes=None):
    root, out_root = Path(root), Path(out_root)
    D, H, W = size
    suffix = f"{D}x{H}x{W}"
    src_size = max(size)                       # canvas-relative sizing reference
    allowed_lv = (None if not classes
                  else {ALL_CLASSES.index(c) + 1 for c in classes})

    subs = subjects_for_split(root, split)
    per_class: dict[int, dict] = {}            # lv -> {"name", "tiles"}
    for subj in subs:
        lab_p = root / subj / f"label_{suffix}.npy"
        ct_p = root / subj / f"ct_{suffix}.npy"
        if not lab_p.exists() or not ct_p.exists():
            continue
        lab = np.load(lab_p)
        ct = np.clip(np.load(ct_p).astype(np.float32), 0, None)
        ct = ct / (ct.max() + 1e-6)            # -> [0,1] for the intensity channel
        for lv in np.unique(lab):
            lv = int(lv)
            if lv == 0 or lv > len(ALL_CLASSES):
                continue
            if allowed_lv is not None and lv not in allowed_lv:
                continue
            entry = per_class.setdefault(
                lv, {"name": ALL_CLASSES[lv - 1], "tiles": []})
            if len(entry["tiles"]) >= max_renditions:
                continue
            tile = crop_to_tile_3d(ct, lab == lv, min_vox,
                                   source_size=src_size, image_size=src_size,
                                   size_scale=size_scale)
            if tile is not None:
                entry["tiles"].append(tile)

    split_dir = out_root / f"T{D}" / split
    split_dir.mkdir(parents=True, exist_ok=True)
    index = {}
    for lv, entry in per_class.items():
        if not entry["tiles"]:
            continue
        (split_dir / f"class_{lv}.pkl").write_bytes(pickle.dumps(entry))
        index[lv] = entry["name"]
    (split_dir / "index.pkl").write_bytes(pickle.dumps(index))
    print(f"[{split}] wrote {len(index)} classes -> {split_dir}", flush=True)
    return split_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, nargs=3, default=[64, 64, 64])
    ap.add_argument("--split", default="train")
    ap.add_argument("--max-renditions", type=int, default=200)
    ap.add_argument("--min-vox", type=int, default=8)
    ap.add_argument("--size-scale", type=float, default=1.0)
    ap.add_argument("--classes", nargs="*", default=None)
    a = ap.parse_args()
    build_tiles_for_split(a.root, a.out, tuple(a.size), a.split,
                          max_renditions=a.max_renditions, min_vox=a.min_vox,
                          size_scale=a.size_scale, classes=a.classes)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest src/datasets/omniSynth/test_build_totalseg_tiles.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add scripts/synth3d/__init__.py scripts/synth3d/build_totalseg_tiles.py src/datasets/omniSynth/test_build_totalseg_tiles.py
git commit -m "feat(omnisynth3d): build script for per-class organ tile caches"
```

---

## Task 3: `TotalSegObjectBank` (`bank_totalseg.py`)

Reads the tile caches from Task 2 and exposes the bank interface (`task_ids`/`get`/`alphabet`), matching `MedSegObjectBank`. Loads each class file once, LRU-cached across classes.

**Files:**
- Create: `src/datasets/omniSynth/bank_totalseg.py`
- Test: `src/datasets/omniSynth/test_bank_totalseg.py`

**Interfaces:**
- Consumes: tile cache layout from Task 2 (`<tiles_root>/T{D}/{split}/index.pkl` + `class_{lv}.pkl`).
- Produces:
  - `get_or_build_totalseg_bank(tiles_root, size, split, classes=(), lru_classes=64) -> TotalSegObjectBank` (process-shared cache, mirrors `get_or_build_medseg_bank`).
  - `TotalSegObjectBank(tiles_root, size, split, classes=(), lru_classes=64)` with:
    - `task_ids(split=None) -> list[int]` — class label values present (filtered to `classes` names if given).
    - `get(class_id) -> list[np.ndarray]` — that class's `[2,T,T,T]` fp16 tiles (loaded + LRU-cached).
    - `alphabet(class_id) -> str` — the class name.

- [ ] **Step 1: Write the failing test**

Create `src/datasets/omniSynth/test_bank_totalseg.py`:

```python
import sys; sys.path.insert(0, ".")
import pickle
from pathlib import Path

import numpy as np
import pytest

from src.datasets.omniSynth.bank_totalseg import TotalSegObjectBank


def _fixture_cache(tmp_path, size=16):
    """Write a minimal T{size} train cache with 2 classes, 3 tiles each."""
    split_dir = tmp_path / f"T{size}" / "train"
    split_dir.mkdir(parents=True)
    index = {1: "adrenal_gland_left", 3: "aorta"}
    for lv, name in index.items():
        tiles = []
        for _ in range(3):
            t = np.zeros((2, 8, 8, 8), dtype=np.float16)
            t[0, 2:6, 2:6, 2:6] = 0.5
            t[1, 2:6, 2:6, 2:6] = 1.0
            tiles.append(t)
        (split_dir / f"class_{lv}.pkl").write_bytes(
            pickle.dumps({"name": name, "tiles": tiles}))
    (split_dir / "index.pkl").write_bytes(pickle.dumps(index))
    return tmp_path


def test_interface_parity(tmp_path):
    root = _fixture_cache(tmp_path)
    bank = TotalSegObjectBank(root, (16, 16, 16), "train")
    ids = bank.task_ids()
    assert set(ids) == {1, 3}
    assert bank.alphabet(1) == "adrenal_gland_left"
    r = bank.get(3)
    assert len(r) == 3 and r[0].shape == (2, 8, 8, 8) and r[0].dtype == np.float16


def test_class_subset_filter(tmp_path):
    root = _fixture_cache(tmp_path)
    bank = TotalSegObjectBank(root, (16, 16, 16), "train", classes=("aorta",))
    assert bank.task_ids() == [3]


def test_get_is_cached(tmp_path):
    root = _fixture_cache(tmp_path)
    bank = TotalSegObjectBank(root, (16, 16, 16), "train")
    assert bank.get(1) is bank.get(1)                    # same list object (LRU hit)


def test_missing_cache_raises(tmp_path):
    with pytest.raises((FileNotFoundError, ValueError)):
        TotalSegObjectBank(tmp_path, (16, 16, 16), "train")


if __name__ == "__main__":
    import tempfile
    for fn in (test_interface_parity, test_class_subset_filter,
               test_get_is_cached, test_missing_cache_raises):
        with tempfile.TemporaryDirectory() as d:
            fn(Path(d))
    print("ALL TOTALSEG BANK TESTS PASSED")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest src/datasets/omniSynth/test_bank_totalseg.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.datasets.omniSynth.bank_totalseg'`

- [ ] **Step 3: Write minimal implementation**

Create `src/datasets/omniSynth/bank_totalseg.py`:

```python
"""TotalSegObjectBank: real TotalSegmentator organs as an omniSynth object source.

Mirrors MedSegObjectBank's interface (task_ids / get / alphabet) so it drops into
the render_scene + target_mode machinery. It reads the precomputed per-class tile
caches written by scripts/synth3d/build_totalseg_tiles.py — no full-volume reads at
train time. A class is a TotalSeg organ label value; a rendition is one subject's
organ as a [2, T, T, T] fp16 tile. Each class file is loaded once and LRU-cached."""

import pickle
from collections import OrderedDict
from pathlib import Path

_BANK_CACHE: dict = {}


def get_or_build_totalseg_bank(tiles_root, size, split="train", classes=(),
                               lru_classes=64):
    key = (str(tiles_root), int(size[0]), str(split), tuple(classes), int(lru_classes))
    if key not in _BANK_CACHE:
        _BANK_CACHE[key] = TotalSegObjectBank(tiles_root, size, split, classes,
                                              lru_classes)
    return _BANK_CACHE[key]


class TotalSegObjectBank:
    def __init__(self, tiles_root, size, split="train", classes=(), lru_classes=64):
        self.split_dir = Path(tiles_root) / f"T{int(size[0])}" / split
        index_path = self.split_dir / "index.pkl"
        if not index_path.exists():
            raise FileNotFoundError(f"no tile cache at {index_path} — run "
                                    "scripts/synth3d/build_totalseg_tiles.py first")
        self._index: dict[int, str] = pickle.loads(index_path.read_bytes())
        if classes:
            wanted = set(classes)
            self._index = {lv: n for lv, n in self._index.items() if n in wanted}
        if not self._index:
            raise ValueError(f"empty class pool for split {split!r} at {self.split_dir}")
        self._pool = sorted(self._index)
        self._lru_classes = int(lru_classes)
        self._loaded: "OrderedDict[int, list]" = OrderedDict()

    def task_ids(self, split=None) -> list[int]:
        return list(self._pool)

    def get(self, class_id: int) -> list:
        cid = int(class_id)
        if cid in self._loaded:
            self._loaded.move_to_end(cid)
            return self._loaded[cid]
        data = pickle.loads((self.split_dir / f"class_{cid}.pkl").read_bytes())
        tiles = data["tiles"]
        self._loaded[cid] = tiles
        if len(self._loaded) > self._lru_classes:
            self._loaded.popitem(last=False)
        return tiles

    def alphabet(self, class_id: int) -> str:
        return self._index[int(class_id)]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest src/datasets/omniSynth/test_bank_totalseg.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/datasets/omniSynth/bank_totalseg.py src/datasets/omniSynth/test_bank_totalseg.py
git commit -m "feat(omnisynth3d): TotalSegObjectBank reading per-class tile caches"
```

---

## Task 4: 3D scene composition (`render3d.py`)

The volumetric twin of `render.py`, free-placement path only. Pure and bank-free (takes sampler callables), so it is unit-testable with trivial samplers.

**Files:**
- Create: `src/datasets/omniSynth/render3d.py`
- Test: `src/datasets/omniSynth/test_render3d.py`

**Interfaces:**
- Consumes: nothing (bank-free; samplers passed in).
- Produces:
  - `render_scene_3d(rng, canvas, n_objects, k_min, k_max, target_sampler, distractor_sampler, *, tries=1, max_overlap=1.0, background="black", bg_kwargs=None) -> (image, mask, k, info)`. `canvas=(D,H,W)`; `image`/`mask` float32 `(D,H,W)`; `mask` binary; `info={"k": int, "target_centroids": [(z,y,x) in [0,1], ...]}`. Samplers return `[2,T,T,T]` tiles (or bare 3D bitmaps). `k ~ U[k_min,k_max]` clamped to `[1, n_objects]`. Targets composited last (over distractors); anti-overlap on `mask > 0`.

- [ ] **Step 1: Write the failing test**

Create `src/datasets/omniSynth/test_render3d.py`:

```python
import sys; sys.path.insert(0, ".")
import numpy as np
from src.datasets.omniSynth.render3d import render_scene_3d

CANVAS = (16, 16, 16)


def _obj(intensity, mask_val, t=6):
    def s(rng):
        arr = np.zeros((2, t, t, t), dtype=np.float32)
        arr[0] = intensity
        arr[1] = mask_val
        return arr
    return s


def test_shapes_and_k_range():
    rng = np.random.default_rng(0)
    for _ in range(30):
        img, mask, k, info = render_scene_3d(
            rng, CANVAS, n_objects=5, k_min=2, k_max=4,
            target_sampler=_obj(0.5, 1.0), distractor_sampler=_obj(0.9, 1.0))
        assert img.shape == CANVAS and img.dtype == np.float32
        assert mask.shape == CANVAS
        assert 2 <= k <= 4
        assert len(info["target_centroids"]) >= 1


def test_mask_binary_and_only_targets():
    # distractors paint intensity 0.9 but must never enter the label mask.
    rng = np.random.default_rng(1)
    img, mask, k, _ = render_scene_3d(
        rng, CANVAS, n_objects=6, k_min=2, k_max=2,
        target_sampler=_obj(0.5, 1.0), distractor_sampler=_obj(0.9, 1.0))
    assert set(np.unique(mask)).issubset({0.0, 1.0})
    assert mask.sum() > 0
    # every masked voxel shows the target intensity, not a distractor's
    assert np.allclose(img[mask > 0], 0.5)


def test_k_clamped_to_n_objects():
    rng = np.random.default_rng(2)
    _, _, k, _ = render_scene_3d(
        rng, CANVAS, n_objects=3, k_min=99, k_max=99,
        target_sampler=_obj(0.5, 1.0), distractor_sampler=_obj(0.9, 1.0))
    assert k == 3


def test_centroids_in_unit_range():
    rng = np.random.default_rng(3)
    _, _, _, info = render_scene_3d(
        rng, CANVAS, n_objects=4, k_min=1, k_max=1,
        target_sampler=_obj(0.5, 1.0), distractor_sampler=_obj(0.9, 1.0))
    for (z, y, x) in info["target_centroids"]:
        assert 0.0 <= z <= 1.0 and 0.0 <= y <= 1.0 and 0.0 <= x <= 1.0


def test_anti_overlap_reduces_union_deficit():
    big = _obj(1.0, 1.0, t=10)               # oversized tiles -> overlap likely

    def occupancy(tries):
        rng = np.random.default_rng(5)
        areas = unions = 0
        for _ in range(15):
            img, _, _, _ = render_scene_3d(
                rng, CANVAS, n_objects=6, k_min=1, k_max=1,
                target_sampler=big, distractor_sampler=big,
                tries=tries, max_overlap=0.0)
            unions += int((img > 0).sum())
            areas += 6 * 10 ** 3
        return areas / max(unions, 1)         # higher => more overlap
    assert occupancy(16) < occupancy(1)


def test_black_background_zero_off_object():
    rng = np.random.default_rng(6)
    img, mask, _, _ = render_scene_3d(
        rng, CANVAS, n_objects=1, k_min=1, k_max=1,
        target_sampler=_obj(0.5, 1.0), distractor_sampler=_obj(0.5, 1.0))
    assert (img[mask == 0] == 0).all()


if __name__ == "__main__":
    test_shapes_and_k_range()
    test_mask_binary_and_only_targets()
    test_k_clamped_to_n_objects()
    test_centroids_in_unit_range()
    test_anti_overlap_reduces_union_deficit()
    test_black_background_zero_off_object()
    print("ALL RENDER3D TESTS PASSED")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest src/datasets/omniSynth/test_render3d.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.datasets.omniSynth.render3d'`

- [ ] **Step 3: Write minimal implementation**

Create `src/datasets/omniSynth/render3d.py`:

```python
"""Pure 3D scene composition for omniSynth 3D — the volumetric twin of render.py
(free-placement path only). Bank-free (samplers passed in), so it is unit-testable
with trivial samplers. Objects are pasted by their true contour (ch1 mask), and
anti-overlap operates on mask>0, never the bbox rectangle."""

import numpy as np


def render_scene_3d(rng, canvas, n_objects, k_min, k_max, target_sampler,
                    distractor_sampler, *, tries=1, max_overlap=1.0,
                    background="black", bg_kwargs=None):
    """Compose a free-placement 3D scene. Returns (image (D,H,W) float32,
    mask (D,H,W) float32 binary, k, info). info["target_centroids"] holds each
    target's mask centre-of-mass in [0,1] (z,y,x)."""
    D, H, W = canvas
    n_obj = max(1, int(n_objects))
    k = int(rng.integers(k_min, k_max + 1))
    k = max(1, min(k, n_obj))
    is_target = np.zeros(n_obj, dtype=bool)
    is_target[rng.permutation(n_obj)[:k]] = True

    image = _make_background_3d(D, H, W, background, rng, bg_kwargs)
    mask = np.zeros((D, H, W), dtype=np.float32)
    occ = np.zeros((D, H, W), dtype=bool)
    centroids = []
    target_paints = []
    for i in range(n_obj):
        if is_target[i]:
            res = target_sampler(rng)
            tile = res[0] if isinstance(res, tuple) else res
        else:
            tile = distractor_sampler(rng)
        vol_t, mask_t = _split_3d(tile)
        td = vol_t.shape[0]
        cz, cy, cx = _place_random_3d(occ, mask_t, td, D, H, W, rng, tries, max_overlap)
        _occupy_3d(occ, mask_t, cz, cy, cx)
        if is_target[i]:
            target_paints.append((vol_t, mask_t, cz, cy, cx))
            pasted = _paste_3d(mask, mask_t, cz, cy, cx)
            centroids.append(_paste_centroid_3d(pasted, cz, cy, cx, D, H, W))
        else:
            _composite_3d(image, vol_t, mask_t, cz, cy, cx)
    for vol_t, mask_t, cz, cy, cx in target_paints:      # targets over distractors
        _composite_3d(image, vol_t, mask_t, cz, cy, cx)
    return image, mask, k, {"k": k, "target_centroids": centroids}


def _split_3d(tile):
    """(vol, mask) from a [2,T,T,T] rendition or a bare 3D bitmap (vol==mask)."""
    if tile.ndim == 4:
        return tile[0].astype(np.float32), tile[1].astype(np.float32)
    t = tile.astype(np.float32)
    return t, t


def _slices_3d(td, th, tw, cz, cy, cx, D, H, W):
    oz, oy, ox = cz - td // 2, cy - th // 2, cx - tw // 2
    dz0, dy0, dx0 = max(0, oz), max(0, oy), max(0, ox)
    dz1, dy1, dx1 = min(D, oz + td), min(H, oy + th), min(W, ox + tw)
    if dz0 >= dz1 or dy0 >= dy1 or dx0 >= dx1:
        return None
    return ((slice(dz0, dz1), slice(dy0, dy1), slice(dx0, dx1)),
            (slice(dz0 - oz, dz1 - oz), slice(dy0 - oy, dy1 - oy),
             slice(dx0 - ox, dx1 - ox)))


def _composite_3d(canvas, vol_t, mask_t, cz, cy, cx):
    sl = _slices_3d(*vol_t.shape, cz, cy, cx, *canvas.shape)
    if sl is None:
        return
    cs, ts = sl
    canvas[cs] = canvas[cs] * (1.0 - mask_t[ts]) + vol_t[ts] * mask_t[ts]


def _paste_3d(label, mask_t, cz, cy, cx):
    """Union-paste a mask; returns (offset, sub_mask) of the written region or None."""
    sl = _slices_3d(*mask_t.shape, cz, cy, cx, *label.shape)
    if sl is None:
        return None
    cs, ts = sl
    sub = (mask_t[ts] > 0).astype(np.float32)
    np.maximum(label[cs], sub, out=label[cs])
    return cs, sub


def _occupy_3d(occ, mask_t, cz, cy, cx):
    sl = _slices_3d(*mask_t.shape, cz, cy, cx, *occ.shape)
    if sl is None:
        return
    cs, ts = sl
    np.logical_or(occ[cs], mask_t[ts] > 0, out=occ[cs])


def _overlap_frac_3d(occ, mask_t, cz, cy, cx):
    sl = _slices_3d(*mask_t.shape, cz, cy, cx, *occ.shape)
    if sl is None:
        return 1.0
    cs, ts = sl
    m = mask_t[ts] > 0
    tot = int(m.sum())
    if tot == 0:
        return 0.0
    return float(np.logical_and(m, occ[cs]).sum()) / tot


def _clamp_center_3d(cz, cy, cx, t, D, H, W):
    lo = t // 2
    return (min(max(cz, lo), D - (t - lo)),
            min(max(cy, lo), H - (t - lo)),
            min(max(cx, lo), W - (t - lo)))


def _place_random_3d(occ, mask_t, t, D, H, W, rng, tries, max_overlap):
    best, best_ov = None, 2.0
    for _ in range(max(1, tries)):
        cz, cy, cx = _clamp_center_3d(int(rng.integers(0, D)), int(rng.integers(0, H)),
                                      int(rng.integers(0, W)), t, D, H, W)
        if tries <= 1:
            return cz, cy, cx
        ov = _overlap_frac_3d(occ, mask_t, cz, cy, cx)
        if ov < best_ov:
            best, best_ov = (cz, cy, cx), ov
        if ov <= max_overlap:
            return cz, cy, cx
    return best


def _paste_centroid_3d(pasted, cz, cy, cx, D, H, W):
    if pasted is not None:
        (sz, sy, sx), sub = pasted
        zs, ys, xs = np.nonzero(sub)
        if zs.size:
            return ((sz.start + float(zs.mean())) / D,
                    (sy.start + float(ys.mean())) / H,
                    (sx.start + float(xs.mean())) / W)
    return (cz / D, cy / H, cx / W)


def _make_background_3d(D, H, W, background, rng, bg_kwargs):
    """"black" -> zeros (no rng touched). "noise" -> a base grey level + gaussian
    noise, so a dark object painted over it stays visible."""
    if background != "noise":
        return np.zeros((D, H, W), dtype=np.float32)
    kw = bg_kwargs or {}
    lo, hi = kw.get("bg_intensity", (0.2, 0.6))
    img = np.full((D, H, W), float(rng.uniform(lo, hi)), dtype=np.float32)
    noise = kw.get("bg_noise", 0.03)
    if noise > 0:
        img = img + rng.normal(0.0, noise, size=(D, H, W)).astype(np.float32)
    return np.clip(img, 0.0, 1.0).astype(np.float32)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest src/datasets/omniSynth/test_render3d.py -v`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add src/datasets/omniSynth/render3d.py src/datasets/omniSynth/test_render3d.py
git commit -m "feat(omnisynth3d): 3D scene composition (render3d)"
```

---

## Task 5: Config + `OmniSynth3DICLDataset` (`config.py`, `dataset3d.py`)

Add the `OmniTotalSegConfig` dataclass and the dataset that orchestrates query + K contexts into the 3D-pipeline contract. Reuses the parent's deterministic-seeding + target-mode helpers, and `render.make_target_sampler`/`make_distractor_sampler` (dimension-agnostic for `identical`/`class`).

**Files:**
- Modify: `src/datasets/omniSynth/config.py` (append dataclass)
- Create: `src/datasets/omniSynth/dataset3d.py`
- Modify: `src/datasets/omniSynth/__init__.py` (exports)
- Test: `src/datasets/omniSynth/test_dataset3d.py`

**Interfaces:**
- Consumes: `TotalSegObjectBank` (Task 3), `render_scene_3d` (Task 4), `OmniSynthICLDataset._subject_rngs/_item_rng/_resolve_target_mode` (existing), `render.make_target_sampler/make_distractor_sampler` (existing).
- Produces:
  - `OmniTotalSegConfig` — dataclass with fields: `tiles_root: str`, `size: tuple = (64,64,64)`, `classes: tuple = ()`, `n_objects: int = 4`, `k_min: int = 1`, `k_max: int = 2`, `placement_tries: int = 4`, `placement_max_overlap: float = 0.1`, `target_mode: str = "class"`, `background: str = "black"`, `lru_classes: int = 64`, `eval_seed_namespace: int = 0`, `eval_subjects_per_task: int = 4`, `epoch_length: int = 10000`.
  - `OmniSynth3DICLDataset(split="train", context_size=3, cfg=None, deterministic=None)` — a `torch.utils.data.Dataset` returning the contract in Global Constraints.

- [ ] **Step 1: Write the failing test**

Create `src/datasets/omniSynth/test_dataset3d.py`:

```python
import sys; sys.path.insert(0, ".")
import pickle
from pathlib import Path

import numpy as np
import torch

from src.datasets.omniSynth.config import OmniTotalSegConfig
from src.datasets.omniSynth.dataset3d import OmniSynth3DICLDataset


def _fixture_cache(tmp_path, size=16, splits=("train", "val")):
    for split in splits:
        split_dir = tmp_path / f"T{size}" / split
        split_dir.mkdir(parents=True)
        index = {1: "adrenal_gland_left", 3: "aorta", 5: "autochthon_left"}
        for lv, name in index.items():
            tiles = []
            for j in range(3):
                t = np.zeros((2, 6, 6, 6), dtype=np.float16)
                t[0] = 0.3 + 0.1 * j
                t[1, 1:5, 1:5, 1:5] = 1.0
                tiles.append(t)
            (split_dir / f"class_{lv}.pkl").write_bytes(
                pickle.dumps({"name": name, "tiles": tiles}))
        (split_dir / "index.pkl").write_bytes(pickle.dumps(index))
    return tmp_path


def _cfg(root):
    return OmniTotalSegConfig(tiles_root=str(root), size=(16, 16, 16),
                              n_objects=4, k_min=1, k_max=2)


def test_contract_shapes_and_dtypes(tmp_path):
    root = _fixture_cache(tmp_path)
    ds = OmniSynth3DICLDataset(split="train", context_size=3, cfg=_cfg(root))
    item = ds[0]
    assert item["image"].shape == (1, 16, 16, 16) and item["image"].dtype == torch.float32
    assert item["label"].shape == (16, 16, 16) and item["label"].dtype == torch.int64
    assert item["context_in"].shape == (3, 1, 16, 16, 16)
    assert item["context_out"].shape == (3, 16, 16, 16) and item["context_out"].dtype == torch.int64
    assert item["spacing"].shape == (3,)
    assert isinstance(item["subject"], str) and isinstance(item["label_name"], str)
    assert item["label"].max() <= 1                       # binary target


def test_eval_is_deterministic(tmp_path):
    root = _fixture_cache(tmp_path)
    ds = OmniSynth3DICLDataset(split="val", context_size=2, cfg=_cfg(root))
    a, b = ds[0], ds[0]
    assert torch.equal(a["image"], b["image"]) and torch.equal(a["label"], b["label"])
    assert torch.equal(a["context_in"], b["context_in"])


def test_collate_compatible(tmp_path):
    from src.totalseg_dataloader_incontext import incontext_collate_fn
    root = _fixture_cache(tmp_path)
    ds = OmniSynth3DICLDataset(split="train", context_size=2, cfg=_cfg(root))
    batch = incontext_collate_fn([ds[0], ds[1]])
    assert batch["image"].shape == (2, 1, 16, 16, 16)
    assert batch["context_in"].shape == (2, 2, 1, 16, 16, 16)


if __name__ == "__main__":
    import tempfile
    for fn in (test_contract_shapes_and_dtypes, test_eval_is_deterministic,
               test_collate_compatible):
        with tempfile.TemporaryDirectory() as d:
            fn(Path(d))
    print("ALL DATASET3D TESTS PASSED")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest src/datasets/omniSynth/test_dataset3d.py -v`
Expected: FAIL — `ImportError: cannot import name 'OmniTotalSegConfig'`

- [ ] **Step 3: Write minimal implementation**

Append to `src/datasets/omniSynth/config.py`:

```python
@dataclass
class OmniTotalSegConfig:
    """3D omniSynth: TotalSegmentator organs on a 3D canvas. Reads the precomputed
    per-class tile cache under {tiles_root}/T{size[0]}/{split}/ (built by
    scripts/synth3d/build_totalseg_tiles.py). Free placement only; target_mode is
    restricted to identical | class in v1 (no 3D affine aug)."""
    tiles_root: str = ""
    size: tuple = (64, 64, 64)
    classes: tuple = ()               # () = all classes present; else subset of names
    n_objects: int = 4                # total organs placed per scene (targets + distractors)
    k_min: int = 1                    # target count ~ U[k_min, k_max], clamped to [1, n_objects]
    k_max: int = 2
    placement_tries: int = 4          # anti-overlap: candidates tried per object (>1 = rejection)
    placement_max_overlap: float = 0.1
    target_mode: str = "class"        # identical | class (aug deferred)
    background: str = "black"         # black | noise
    lru_classes: int = 64             # #class tile-files kept in RAM
    eval_seed_namespace: int = 0
    eval_subjects_per_task: int = 4
    epoch_length: int = 10000
```

Create `src/datasets/omniSynth/dataset3d.py`:

```python
"""OmniSynth3DICLDataset: paints bbox-cropped TotalSegmentator organs at random 3D
positions onto a D×H×W canvas, emitting the TotalSegInContextDataset contract
(image/label/context_in/context_out/subject/label_name/spacing) so the existing 3D
pipeline + incontext_collate_fn consume it unchanged.

Reuses OmniSynthICLDataset's deterministic RNG seeding + target-mode resolution, and
render.make_target_sampler/make_distractor_sampler (dimension-agnostic for the
identical|class modes used here). Free placement only; no per-item scipy warps."""

import numpy as np
import torch

from .bank_totalseg import get_or_build_totalseg_bank
from .config import OmniTotalSegConfig
from .dataset import OmniSynthICLDataset
from .render import make_distractor_sampler, make_target_sampler
from .render3d import render_scene_3d

_TARGET_MODES_3D = ("identical", "class")


class OmniSynth3DICLDataset(OmniSynthICLDataset):
    def __init__(self, split="train", context_size=3, cfg=None, deterministic=None):
        self.split = split
        self.context_size = context_size
        self.cfg = cfg or OmniTotalSegConfig()
        if self.cfg.target_mode not in _TARGET_MODES_3D:
            raise ValueError(f"3D target_mode must be identical|class, got "
                             f"{self.cfg.target_mode!r}")
        # The reused parent helpers read self.sampling.eval_seed_namespace and
        # self.scene.target_mode — point both at the single 3D config.
        self.sampling = self.cfg
        self.scene = self.cfg
        self.canvas = tuple(int(v) for v in self.cfg.size)
        self.deterministic = (split != "train") if deterministic is None else deterministic

        self.bank = get_or_build_totalseg_bank(self.cfg.tiles_root, self.cfg.size,
                                               split, tuple(self.cfg.classes),
                                               self.cfg.lru_classes)
        self.pool = self.bank.task_ids(split)
        if not self.pool:
            raise ValueError(f"empty class pool for split {split!r}")

        if self.deterministic:
            self._eval_index = []
            self.samples = []
            for class_id in self.pool:
                for s in range(self.cfg.eval_subjects_per_task):
                    self.samples.append(len(self._eval_index))
                    self._eval_index.append((class_id, s))
        else:
            self._eval_index = None
            self.samples = list(range(self.cfg.epoch_length))

    def __len__(self):
        return len(self.samples)

    def _render(self, rng, target_sampler, distractor_sampler):
        return render_scene_3d(
            rng, self.canvas, self.cfg.n_objects, self.cfg.k_min, self.cfg.k_max,
            target_sampler, distractor_sampler,
            tries=self.cfg.placement_tries, max_overlap=self.cfg.placement_max_overlap,
            background=self.cfg.background)

    def __getitem__(self, idx):
        if self.deterministic:
            class_id, sample_index = self._eval_index[idx]
        else:
            class_id = int(self.pool[np.random.default_rng().integers(len(self.pool))])
            sample_index = idx

        rngs = self._subject_rngs(class_id, sample_index)     # inherited
        base_rng = self._item_rng(class_id, sample_index)     # inherited
        mode = self._resolve_target_mode(base_rng)            # inherited

        target_sampler = make_target_sampler(self.bank, class_id, self.scene,
                                             base_rng, mode=mode)
        distractor_sampler = make_distractor_sampler(self.bank, self.pool, class_id)

        t_img, t_seg, _, _ = self._render(rngs[0], target_sampler, distractor_sampler)
        ctx = [self._render(rngs[1 + i], target_sampler, distractor_sampler)
               for i in range(self.context_size)]

        def _img(a):   # (D,H,W) float -> (1,D,H,W) float32
            return torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)).unsqueeze(0)

        def _lbl(a):   # (D,H,W) float -> (D,H,W) int64 binary
            return torch.from_numpy((np.ascontiguousarray(a) > 0).astype(np.int64))

        return {
            "image":       _img(t_img),
            "label":       _lbl(t_seg),
            "context_in":  torch.stack([_img(c[0]) for c in ctx]),
            "context_out": torch.stack([_lbl(c[1]) for c in ctx]),
            "subject":     f"omni_{int(class_id)}_{int(sample_index)}",
            "label_name":  self.bank.alphabet(class_id),
            "spacing":     torch.ones(3, dtype=torch.float32),
        }
```

Update `src/datasets/omniSynth/__init__.py` — add to the imports and `__all__`:

```python
from .config import (OmniDiversityConfig, OmniMedSegConfig, OmniSceneConfig,
                     OmniSamplingConfig, OmniTotalSegConfig)
from .dataset import OmniSynthICLDataset
from .dataset3d import OmniSynth3DICLDataset

__all__ = [
    "OmniDiversityConfig",
    "OmniMedSegConfig",
    "OmniSceneConfig",
    "OmniSamplingConfig",
    "OmniTotalSegConfig",
    "OmniSynthICLDataset",
    "OmniSynth3DICLDataset",
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest src/datasets/omniSynth/test_dataset3d.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Run the full omniSynth suite to confirm the 2D path is untouched**

Run: `python -m pytest src/datasets/omniSynth/ -v`
Expected: PASS (all new + existing tests; medseg/biomedparse tests may `SKIP` if their data isn't mounted)

- [ ] **Step 6: Commit**

```bash
git add src/datasets/omniSynth/config.py src/datasets/omniSynth/dataset3d.py src/datasets/omniSynth/__init__.py src/datasets/omniSynth/test_dataset3d.py
git commit -m "feat(omnisynth3d): OmniTotalSegConfig + OmniSynth3DICLDataset"
```

---

## Task 6: Pipeline integration (`experiments/3d/common.py`, config)

Wire `source=omnisynth3d` into the 3D `build_dataset` and add a Hydra config selecting it, so `train_loader` / `make_eval_loader` (which already use `incontext_collate_fn`) drive the new dataset.

**Files:**
- Modify: `experiments/3d/common.py` (`build_dataset`)
- Create: `configs/experiment/3d/omnisynth3d.yaml`
- Test: `src/datasets/omniSynth/test_integration3d.py`

**Interfaces:**
- Consumes: `OmniSynth3DICLDataset`, `OmniTotalSegConfig` (Task 5); `cfg.paths.totalseg`; a `cfg.synth3d` config block.
- Produces: `build_dataset(cfg, split)` returns `OmniSynth3DICLDataset` when `cfg.data.source == "omnisynth3d"`.

- [ ] **Step 1: Write the failing test**

Create `src/datasets/omniSynth/test_integration3d.py`:

```python
import sys; sys.path.insert(0, ".")
import pickle
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from experiments.threed_common_shim import build_dataset  # see Step 3 note
```

> Note: `experiments/3d` is not an importable package name (starts with a digit). The test imports `build_dataset` via a tiny shim module created in Step 3. The rest of the test:

```python

def _fixture_cache(tmp_path, size=16):
    split_dir = tmp_path / f"T{size}" / "train"
    split_dir.mkdir(parents=True)
    index = {1: "adrenal_gland_left", 3: "aorta"}
    for lv, name in index.items():
        tiles = [np.pad(np.ones((2, 4, 4, 4), dtype=np.float16), 0) for _ in range(2)]
        (split_dir / f"class_{lv}.pkl").write_bytes(
            pickle.dumps({"name": name, "tiles": tiles}))
    (split_dir / "index.pkl").write_bytes(pickle.dumps(index))
    return tmp_path


def test_build_dataset_dispatches_omnisynth3d(tmp_path):
    root = _fixture_cache(tmp_path)
    cfg = OmegaConf.create({
        "data": {"source": "omnisynth3d", "context_size": 2, "image_size": [16, 16, 16]},
        "paths": {"totalseg": str(root)},
        "synth3d": {"tiles_root": str(root), "size": [16, 16, 16], "n_objects": 3,
                    "k_min": 1, "k_max": 1, "target_mode": "class"},
    })
    ds = build_dataset(cfg, "train")
    item = ds[0]
    assert item["image"].shape == (1, 16, 16, 16)
    assert item["label"].shape == (16, 16, 16)


if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        test_build_dataset_dispatches_omnisynth3d(Path(d))
    print("INTEGRATION3D TEST PASSED")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest src/datasets/omniSynth/test_integration3d.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'experiments.threed_common_shim'`

- [ ] **Step 3: Write minimal implementation**

Create `experiments/threed_common_shim.py` (lets tests import from the digit-prefixed dir):

```python
"""Import shim: `experiments/3d` cannot be imported as a package (dir name starts
with a digit), so expose build_dataset by path for tests and callers that need it."""
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "_threed_common", Path(__file__).resolve().parent / "3d" / "common.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

build_dataset = _mod.build_dataset
```

In `experiments/3d/common.py`, add the `omnisynth3d` branch at the **top** of `build_dataset` (before `_source_root`, which rejects unknown sources):

```python
def build_dataset(cfg, split: str):
    """Construct the 3D in-context dataset for `split`, dispatching on cfg.data.source."""
    if cfg.data.get("source", "totalseg") == "omnisynth3d":
        from src.datasets.omniSynth.dataset3d import OmniSynth3DICLDataset
        from src.datasets.omniSynth.config import OmniTotalSegConfig
        s = cfg.synth3d
        tiles_root = s.get("tiles_root", None) or cfg.paths.get("totalseg")
        cfg3d = OmniTotalSegConfig(
            tiles_root=tiles_root,
            size=tuple(s.get("size", cfg.data.image_size)),
            classes=tuple(s.get("classes", ()) or ()),
            n_objects=int(s.get("n_objects", 4)),
            k_min=int(s.get("k_min", 1)), k_max=int(s.get("k_max", 2)),
            placement_tries=int(s.get("placement_tries", 4)),
            placement_max_overlap=float(s.get("placement_max_overlap", 0.1)),
            target_mode=s.get("target_mode", "class"),
            background=s.get("background", "black"),
            lru_classes=int(s.get("lru_classes", 64)),
            eval_seed_namespace=int(s.get("eval_seed_namespace", 0)),
            eval_subjects_per_task=int(s.get("eval_subjects_per_task", 4)),
            epoch_length=int(s.get("epoch_length", 10000)),
        )
        return OmniSynth3DICLDataset(split=split, context_size=cfg.data.context_size,
                                     cfg=cfg3d)
    d = cfg.data
    # ... existing body unchanged ...
```

> The existing return-type annotation `-> TotalSegInContextDataset` becomes inaccurate; change the signature to `def build_dataset(cfg, split: str):` (drop the annotation) to avoid a misleading type.

Create `configs/experiment/3d/omnisynth3d.yaml`:

```yaml
# @package _global_
# 3D omniSynth: TotalSegmentator organs on a 3D canvas.
# Prerequisite: build the tile cache once, e.g.
#   python scripts/synth3d/build_totalseg_tiles.py \
#     --root ${paths.totalseg} --out ${paths.totalseg}/omni_tiles \
#     --size 64 64 64 --split train
#   (repeat with --split val)
data:
  source: omnisynth3d
  image_size: [64, 64, 64]
  context_size: 3

synth3d:
  tiles_root: ${paths.totalseg}/omni_tiles
  size: [64, 64, 64]
  classes: []            # [] = all classes present in the cache
  n_objects: 4
  k_min: 1
  k_max: 2
  placement_tries: 4
  placement_max_overlap: 0.1
  target_mode: class     # identical | class
  background: black       # black | noise
  eval_subjects_per_task: 4
  epoch_length: 10000
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest src/datasets/omniSynth/test_integration3d.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Run the whole new suite**

Run: `python -m pytest src/datasets/omniSynth/test_bank_common3d.py src/datasets/omniSynth/test_build_totalseg_tiles.py src/datasets/omniSynth/test_bank_totalseg.py src/datasets/omniSynth/test_render3d.py src/datasets/omniSynth/test_dataset3d.py src/datasets/omniSynth/test_integration3d.py -v`
Expected: PASS (all)

- [ ] **Step 6: Commit**

```bash
git add experiments/3d/common.py experiments/threed_common_shim.py configs/experiment/3d/omnisynth3d.yaml src/datasets/omniSynth/test_integration3d.py
git commit -m "feat(omnisynth3d): wire source=omnisynth3d into 3D build_dataset"
```

---

## Post-implementation: real-data smoke (manual, not a unit test)

After the tasks pass, validate against the real store on a dev node (documented for the operator, not part of the TDD loop):

```bash
# 1. Build a tiny cache (a few classes, few subjects) at 64³
python scripts/synth3d/build_totalseg_tiles.py \
  --root <cfg.paths.totalseg> --out /tmp/omni_tiles --size 64 64 64 \
  --split train --max-renditions 20 --classes aorta liver spleen
python scripts/synth3d/build_totalseg_tiles.py \
  --root <cfg.paths.totalseg> --out /tmp/omni_tiles --size 64 64 64 \
  --split val --max-renditions 20 --classes aorta liver spleen

# 2. Plot a few items (reuse experiments/3d/plot_dataset_items.py if it accepts a dataset)
#    or a 5-line script instantiating OmniSynth3DICLDataset and saving mid-slices.
```

Log the outcome in `docs/logs.md` (per repo convention).

---

## Self-Review

**1. Spec coverage:**
- Parallel `render3d.py`, 2D untouched → Task 4 + "2D path untouched" constraint + Task 5 Step 5. ✓
- Free 3D placement, native sizes, contour anti-overlap → Task 4 (`_overlap_frac_3d`, `_place_random_3d`). ✓
- Precompute-once tile cache, LRU, no hot-path full-volume reads → Tasks 2–3. ✓
- `identical`/`class` only, no aug/rotation → Task 5 (`_TARGET_MODES_3D` guard). ✓
- Contour pasting (mask not bbox) → Task 4 `_composite_3d`/`_paste_3d`/`_occupy_3d` all use `mask_t`. ✓
- Thin subclass reusing seeding/target-mode helpers → Task 5 (`OmniSynth3DICLDataset(OmniSynthICLDataset)`). ✓
- 3D-pipeline output contract → Task 5 test `test_contract_shapes_and_dtypes` + `test_collate_compatible`. ✓
- Integration `source=omnisynth3d` + config → Task 6. ✓
- Tests: render3d / bank / dataset / integration → Tasks 1–6. ✓
- Deferred (grid 3D, image bg, biomedparse-3D, 3D rotation) → not implemented, matching spec YAGNI. ✓

**2. Placeholder scan:** No TBD/TODO; every code step shows complete code; every command has expected output. ✓

**3. Type consistency:**
- `make_object_tile_3d(vol_crop, m_crop, *, source_size, image_size, size_scale, min_tile)` — same call in `crop_to_tile_3d`, build script, and its test. ✓
- `crop_to_tile_3d(..., source_size=, image_size=, size_scale=)` — build script passes exactly these. ✓
- `TotalSegObjectBank(tiles_root, size, split, classes, lru_classes)` — Task 3 signature matches `get_or_build_totalseg_bank` and Task 5's call. ✓
- `render_scene_3d(rng, canvas, n_objects, k_min, k_max, target_sampler, distractor_sampler, *, tries, max_overlap, background, bg_kwargs)` — Task 5 `_render` calls with matching kwargs. ✓
- Tile cache layout `T{size[0]}/{split}/class_{lv}.pkl` + `index.pkl` written in Task 2, read identically in Task 3. ✓
- `OmniTotalSegConfig` fields consumed in Task 5 dataset + Task 6 `build_dataset` match the dataclass. ✓
