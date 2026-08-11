# ChemoTox in-context eval — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Evaluate the 3D in-context models (patchset3d, medverse, …) on the ChemoTox cohort for both label schemes — 117 TotalSegmentator classes and 4 body-composition tissues — reusing the existing TotalSeg eval harness.

**Architecture:** Generalize `scripts/convert_to_npy.py` into a single tool with a `--source` seam that converts the ChemoTox cohort (paths from a JSON) into an uncompressed, 1.5 mm-iso `.npy` tree byte-compatible with the base `use_crop` dataset. `source=chemotox` (totalseg labels) then reuses `TotalSegInContextDataset` unchanged; `source=chemotox_bc` (tissues) uses a thin subclass reading `bc.npy`.

**Tech Stack:** Python, NumPy, nibabel, scipy.ndimage, PyTorch, Hydra/OmegaConf, pytest.

## Global Constraints

- **Cache spacing is 1.5 mm isotropic, non-negotiable.** The base `use_crop` path
  (`TotalSegInContextDataset._load_spacings`, lines ~600–610) hardcodes native crop
  data to 1.5 mm/voxel; any other cache spacing silently mis-sizes crops. Every
  converted subject MUST have a `spacings.json` entry (value overridden to 1.5 under
  use_crop, but presence is required — missing subjects fall back to 1.0 mm).
- **TS→project label remap is by NAME, mandatory.** ChemoTox `total_seg_total.nii.gz`
  uses TotalSegmentator v2 `total` numbering (`5=liver, 51=heart, 52=aorta`); the
  project's `ALL_CLASSES` is alphabetical (`3=aorta, 5=autochthon_left`). Never treat
  the raw TS integers as project label ids.
- **bclabels is 4-D `(...,2)`; use channel 0 only.** Channel 1 is instance IDs — discard.
- **Do not change the default `totalseg` behavior of `convert_to_npy.py`.** `--source`
  defaults to `totalseg`, `--out` defaults to `--data` (in-place), `--target-spacing`
  defaults to `None` (full native). Existing runs must be byte-identical.
- **bc classes:** `muscle=1, sat=2, vat=3, imat=4` (the `bc.npy` voxel values).
- **`.npy`, not `.npz`** (mmap crop access).
- Cohort JSON: `experiments/3d/universal_coords/coords_paths_chemotox.json` (366 entries,
  keys `patientID#date`; fields `img`, `totalseg`, `bclabels`).
- `paths.chemotox` default: `/nfs/data/nii/data1/Analysis/camaret___in_context_segmentation/ANALYSIS_20251122/data/chemotox`.

---

### Task 1: Authoritative TS-v2 `total` map + remap

**Files:**
- Create: `data/totalseg_total_map.py`
- Test: `tests/test_totalseg_total_map.py`

**Interfaces:**
- Produces:
  - `TOTALSEG_V2_TOTAL: list[str]` — 117 names in TS id order (id = index+1).
  - `build_ts_to_project_lut() -> np.ndarray` — uint8 LUT, `lut[ts_id] = project_idx`
    (project idx from `data.totalseg_classes.ALL_CLASSES`, 1-indexed), `lut[0]=0`.
  - `remap_ts_total(arr: np.ndarray) -> np.ndarray` — vectorized `lut[arr]`, returns uint8.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_totalseg_total_map.py
import json, os
import numpy as np
import pytest
from data.totalseg_total_map import (
    TOTALSEG_V2_TOTAL, build_ts_to_project_lut, remap_ts_total,
)
from data.totalseg_classes import ALL_CLASSES

_CLASS_TO_IDX = {c: i + 1 for i, c in enumerate(ALL_CLASSES)}

def test_map_has_117_known_anchors():
    assert len(TOTALSEG_V2_TOTAL) == 117
    # TS ids are 1-indexed positions in the ordered list
    assert TOTALSEG_V2_TOTAL[0] == "spleen"       # ts id 1
    assert TOTALSEG_V2_TOTAL[4] == "liver"        # ts id 5
    assert TOTALSEG_V2_TOTAL[50] == "heart"       # ts id 51
    assert TOTALSEG_V2_TOTAL[51] == "aorta"       # ts id 52

def test_all_ts_names_exist_in_project():
    assert all(n in _CLASS_TO_IDX for n in TOTALSEG_V2_TOTAL)

def test_remap_translates_by_name():
    arr = np.array([[0, 5, 52, 51]], dtype=np.int16)  # bg, liver, aorta, heart (TS ids)
    out = remap_ts_total(arr)
    assert out.dtype == np.uint8
    assert out[0, 0] == 0
    assert out[0, 1] == _CLASS_TO_IDX["liver"]
    assert out[0, 2] == _CLASS_TO_IDX["aorta"]
    assert out[0, 3] == _CLASS_TO_IDX["heart"]

def test_matches_cohort_stats_file_if_present():
    # The cohort's per-subject stats json is keyed by name in TS id order; if reachable,
    # assert our embedded list matches it exactly (transcription guard).
    p = ("/nfs/data/nii/data1/jungm___ChemoTox/10116066/20220316122148/"
         "ML/total_seg_total_stats_recomp.json")
    if not os.path.exists(p):
        pytest.skip("cohort stats file not reachable")
    names = list(json.load(open(p)).keys())
    assert names == TOTALSEG_V2_TOTAL
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_totalseg_total_map.py -v`
Expected: FAIL with `ModuleNotFoundError: data.totalseg_total_map`.

- [ ] **Step 3: Write the module**

```python
# data/totalseg_total_map.py
"""Official TotalSegmentator v2 `total` label ordering (id = index+1) and a remap to
the project's alphabetical ALL_CLASSES numbering. ChemoTox total_seg_total.nii.gz uses
the TS ordering; the project's label.npy uses ALL_CLASSES — so remap by NAME."""
import numpy as np
from data.totalseg_classes import ALL_CLASSES

# TS v2 `total` names in label-id order (verified against the cohort's
# total_seg_total_stats_recomp.json key order — see test_matches_cohort_stats_file).
TOTALSEG_V2_TOTAL: list[str] = [
    "spleen", "kidney_right", "kidney_left", "gallbladder", "liver", "stomach",
    "pancreas", "adrenal_gland_right", "adrenal_gland_left", "lung_upper_lobe_left",
    "lung_lower_lobe_left", "lung_upper_lobe_right", "lung_middle_lobe_right",
    "lung_lower_lobe_right", "esophagus", "trachea", "thyroid_gland", "small_bowel",
    "duodenum", "colon", "urinary_bladder", "prostate", "kidney_cyst_left",
    "kidney_cyst_right", "sacrum", "vertebrae_S1", "vertebrae_L5", "vertebrae_L4",
    "vertebrae_L3", "vertebrae_L2", "vertebrae_L1", "vertebrae_T12", "vertebrae_T11",
    "vertebrae_T10", "vertebrae_T9", "vertebrae_T8", "vertebrae_T7", "vertebrae_T6",
    "vertebrae_T5", "vertebrae_T4", "vertebrae_T3", "vertebrae_T2", "vertebrae_T1",
    "vertebrae_C7", "vertebrae_C6", "vertebrae_C5", "vertebrae_C4", "vertebrae_C3",
    "vertebrae_C2", "vertebrae_C1", "heart", "aorta", "pulmonary_vein",
    "brachiocephalic_trunk", "subclavian_artery_right", "subclavian_artery_left",
    "common_carotid_artery_right", "common_carotid_artery_left",
    "brachiocephalic_vein_left", "brachiocephalic_vein_right", "atrial_appendage_left",
    "superior_vena_cava", "inferior_vena_cava", "portal_vein_and_splenic_vein",
    "iliac_artery_left", "iliac_artery_right", "iliac_vena_left", "iliac_vena_right",
    "humerus_left", "humerus_right", "scapula_left", "scapula_right", "clavicula_left",
    "clavicula_right", "femur_left", "femur_right", "hip_left", "hip_right",
    "spinal_cord", "gluteus_maximus_left", "gluteus_maximus_right",
    "gluteus_medius_left", "gluteus_medius_right", "gluteus_minimus_left",
    "gluteus_minimus_right", "autochthon_left", "autochthon_right", "iliopsoas_left",
    "iliopsoas_right", "brain", "skull", "rib_left_1", "rib_left_2", "rib_left_3",
    "rib_left_4", "rib_left_5", "rib_left_6", "rib_left_7", "rib_left_8", "rib_left_9",
    "rib_left_10", "rib_left_11", "rib_left_12", "rib_right_1", "rib_right_2",
    "rib_right_3", "rib_right_4", "rib_right_5", "rib_right_6", "rib_right_7",
    "rib_right_8", "rib_right_9", "rib_right_10", "rib_right_11", "rib_right_12",
    "sternum", "costal_cartilages",
]

_CLASS_TO_IDX = {c: i + 1 for i, c in enumerate(ALL_CLASSES)}


def build_ts_to_project_lut() -> np.ndarray:
    """uint8 LUT: lut[ts_id] = project label idx (0 for background / unknown)."""
    lut = np.zeros(len(TOTALSEG_V2_TOTAL) + 1, dtype=np.uint8)
    for ts_id, name in enumerate(TOTALSEG_V2_TOTAL, start=1):
        lut[ts_id] = _CLASS_TO_IDX[name]  # every TS name is in ALL_CLASSES (asserted in tests)
    return lut


def remap_ts_total(arr: np.ndarray) -> np.ndarray:
    """Translate a TS-v2 `total` label volume to project ALL_CLASSES numbering."""
    lut = build_ts_to_project_lut()
    flat = np.asarray(arr).astype(np.int64)
    flat = np.clip(flat, 0, len(lut) - 1)  # guard stray ids
    return lut[flat].astype(np.uint8)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_totalseg_total_map.py -v`
Expected: PASS (the cohort-stats test may `skip` off-cluster).

- [ ] **Step 5: Commit**

```bash
git add data/totalseg_total_map.py tests/test_totalseg_total_map.py
git commit -m "feat(data): TS-v2 total label map + name-based remap to project numbering"
```

---

### Task 2: `convert_to_npy.py` — general scaffolding (`--source/--out/--target-spacing`, resample helper)

Add the general options and the resample-to-spacing helper WITHOUT yet adding the
chemotox source. Keep `totalseg` behavior byte-identical.

**Files:**
- Modify: `scripts/convert_to_npy.py`
- Test: `tests/test_convert_generalize.py`

**Interfaces:**
- Produces:
  - `_resample_to_spacing(vol: np.ndarray, native_sp: list[float], target_sp: float, order: int) -> np.ndarray`
  - argparse gains `--source {totalseg,chemotox}` (default `totalseg`), `--out DIR`
    (default = `--data`), `--target-spacing FLOAT` (default `None`),
    `--limit INT` (default `None`, convert only the first N subjects — smoke test).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_convert_generalize.py
import numpy as np
from scripts.convert_to_npy import _resample_to_spacing

def test_resample_halves_shape_when_target_double_native():
    vol = np.random.rand(20, 20, 20).astype(np.float32)
    out = _resample_to_spacing(vol, native_sp=[1.0, 1.0, 1.0], target_sp=2.0, order=1)
    assert out.shape == (10, 10, 10)

def test_resample_anisotropic_native():
    vol = np.zeros((20, 20, 10), dtype=np.uint8)
    out = _resample_to_spacing(vol, native_sp=[1.5, 1.5, 3.0], target_sp=1.5, order=0)
    # x,y already 1.5 -> unchanged; z at 3.0 -> doubles to 20
    assert out.shape == (20, 20, 20)
    assert out.dtype == np.uint8
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_convert_generalize.py -v`
Expected: FAIL with `ImportError: cannot import name '_resample_to_spacing'`.

- [ ] **Step 3: Add the helper**

Add near `_iso_resize` in `scripts/convert_to_npy.py`:

```python
def _resample_to_spacing(vol: np.ndarray, native_sp, target_sp: float,
                         order: int = 1) -> np.ndarray:
    """Resample `vol` from native voxel spacing (mm, per axis) to `target_sp` mm
    isotropic. order=1 (trilinear) for images, order=0 (nearest) for label maps.
    out_shape[i] = round(shape[i] * native_sp[i] / target_sp)."""
    zoom = [float(ns) / float(target_sp) for ns in native_sp]
    out = ndi.zoom(vol, zoom, order=order)
    return out.astype(vol.dtype, copy=False)
```

- [ ] **Step 4: Add argparse options (do not wire chemotox behavior yet)**

In `main()`'s `ArgumentParser`, add:

```python
    parser.add_argument("--source", choices=["totalseg", "chemotox"], default="totalseg",
                        help="dataset source: totalseg (dir tree, default) or chemotox (JSON of paths)")
    parser.add_argument("--out", default=None,
                        help="output root; defaults to --data (in-place for totalseg)")
    parser.add_argument("--target-spacing", type=float, default=None, dest="target_spacing",
                        help="resample the native outputs to this mm-isotropic spacing "
                             "(default: keep full native)")
    parser.add_argument("--limit", type=int, default=None,
                        help="convert only the first N subjects (smoke test)")
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_convert_generalize.py -v`
Expected: PASS.

- [ ] **Step 6: Verify totalseg path still imports/parses cleanly**

Run: `python scripts/convert_to_npy.py --help`
Expected: help text lists the new flags; no error.

- [ ] **Step 7: Commit**

```bash
git add scripts/convert_to_npy.py tests/test_convert_generalize.py
git commit -m "feat(convert): add --source/--out/--target-spacing/--limit + resample-to-spacing helper"
```

---

### Task 3: `convert_to_npy.py` — chemotox source (enumerate + load_raw + multi-label save + meta.csv)

**Files:**
- Modify: `scripts/convert_to_npy.py`
- Test: `tests/test_convert_chemotox.py`

**Interfaces:**
- Consumes: `remap_ts_total` (Task 1), `_resample_to_spacing` (Task 2), `_normalise_ct`.
- Produces:
  - `enumerate_subjects(source, data, out, limit) -> list[dict]` — each task dict has
    `subj_id`, `out_dir`, `inputs` (source-specific path dict), plus a `labels` name list.
  - `load_raw(task) -> tuple[np.ndarray, list[float], dict[str, np.ndarray]]`
    returning `(raw_ct_f32, native_spacing, {label_name: array})`.
  - `convert_subject(task)` rewritten to consume a task dict and write
    `ct.npy` + one `.npy` per label name; returns `(subj_id, status, spacing, shape, stats)`.
  - chemotox output tree: `out/{subj_id}/{ct.npy,label.npy,bc.npy}`, `out/spacings.json`,
    `out/meta.csv` (`image_id;split`, all `test`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_convert_chemotox.py
import json
import numpy as np
import nibabel as nib
import pytest
from scripts.convert_to_npy import load_raw
from data.totalseg_classes import ALL_CLASSES

_CLASS_TO_IDX = {c: i + 1 for i, c in enumerate(ALL_CLASSES)}


def _write_nii(path, arr, spacing):
    aff = np.diag(list(spacing) + [1.0])
    nib.save(nib.Nifti1Image(arr, aff), str(path))


def test_load_raw_chemotox_remaps_and_takes_bc_channel0(tmp_path):
    D = (6, 6, 4)
    img = (np.random.rand(*D) * 100).astype(np.float32)
    ts = np.zeros(D, dtype=np.int16); ts[0, 0, 0] = 5   # TS liver
    bc = np.zeros(D + (2,), dtype=np.int16)
    bc[..., 0][1, 1, 1] = 1                             # muscle (channel 0)
    bc[..., 1][1, 1, 1] = 9999                          # instance id (channel 1, ignored)
    _write_nii(tmp_path / "img.nii", img, (1.5, 1.5, 3.0))
    _write_nii(tmp_path / "ts.nii", ts, (1.5, 1.5, 3.0))
    _write_nii(tmp_path / "bc.nii", bc, (1.5, 1.5, 3.0))

    task = {"source": "chemotox",
            "inputs": {"img": str(tmp_path / "img.nii"),
                       "totalseg": str(tmp_path / "ts.nii"),
                       "bclabels": str(tmp_path / "bc.nii")}}
    raw, sp, labels = load_raw(task)

    assert raw.shape == D and raw.dtype == np.float32
    assert sp == pytest.approx([1.5, 1.5, 3.0])
    assert set(labels) == {"label", "bc"}
    assert labels["label"][0, 0, 0] == _CLASS_TO_IDX["liver"]
    assert labels["bc"][1, 1, 1] == 1
    assert labels["bc"].max() == 1          # channel 1's 9999 must not leak in
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_convert_chemotox.py -v`
Expected: FAIL with `ImportError: cannot import name 'load_raw'`.

- [ ] **Step 3: Implement enumerate + load_raw + rewrite convert_subject**

Add imports at top of `scripts/convert_to_npy.py`:

```python
import csv
from data.totalseg_total_map import remap_ts_total
```

Add the source seam:

```python
COHORT_JSON = ROOT / "experiments/3d/universal_coords/coords_paths_chemotox.json"

# label channels each source emits (written as {name}.npy; "label" is the primary mask)
SOURCE_LABELS = {"totalseg": ["label"], "chemotox": ["label", "bc"]}


def enumerate_subjects(source: str, data, out, limit=None) -> list[dict]:
    """Return a list of per-subject task dicts (subj_id, out_dir, inputs)."""
    tasks: list[dict] = []
    if source == "totalseg":
        for s in sorted(p for p in Path(data).iterdir() if p.is_dir()):
            tasks.append({"subj_id": s.name, "out_dir": str(Path(out) / s.name),
                          "inputs": {"subj_dir": str(s)}})
    elif source == "chemotox":
        cohort = json.load(open(data)) if str(data).endswith(".json") else json.load(open(COHORT_JSON))
        for key, rec in cohort.items():
            subj_id = key.replace("#", "_")
            tasks.append({"subj_id": subj_id, "out_dir": str(Path(out) / subj_id),
                          "inputs": {"img": rec["img"], "totalseg": rec["totalseg"],
                                     "bclabels": rec["bclabels"]}})
    else:
        raise ValueError(f"unknown source {source!r}")
    if limit is not None:
        tasks = tasks[:limit]
    return tasks


def load_raw(task: dict):
    """(raw_ct f32, native_spacing [3], {label_name: array}) for a chemotox subject.

    All three volumes share one native grid, so no canonicalization is needed — read
    raw dataobj and take spacing from the img affine. (The totalseg source does its own
    CT+segmentations reading inside _convert_totalseg to stay byte-identical.)"""
    assert task["source"] == "chemotox", "load_raw serves the chemotox source only"
    p = task["inputs"]
    img = nib.load(p["img"])
    raw = np.asanyarray(img.dataobj).astype(np.float32)
    sp = [abs(float(x)) for x in nib.affines.voxel_sizes(img.affine)[:3]]
    ts = np.asanyarray(nib.load(p["totalseg"]).dataobj)
    label = remap_ts_total(ts)
    bc = np.asanyarray(nib.load(p["bclabels"]).dataobj)[..., 0].astype(np.uint8)
    return raw, sp, {"label": label, "bc": bc}
```

**Preserve the totalseg path byte-identically via source dispatch.** Do NOT fold
totalseg into a new generalized body — that would drop `ct_raw.npy`/`--store-raw` and
the granular native/raw/sized skip logic, violating the "totalseg byte-identical"
Global Constraint. Instead:

1. **Rename the EXISTING `convert_subject` body to `_convert_totalseg(task: dict)`**,
   changing ONLY how it reads its inputs at the top (from the task dict instead of the
   old tuple). Everything else — `need_native`/`need_raw`/`need_sized`, `ct_raw.npy`
   int16/float16, `ct_stats.json` MRI stats, sized variants, skip logic, the returned
   `(subj, status, native_spacing, native_shape, stats)` tuple — stays verbatim. Read:
   `subj_dir = Path(task["inputs"]["subj_dir"])`, `overwrite = task["overwrite"]`,
   `size = task["size"]`, `modality = task["modality"]`, `store_raw = task["store_raw"]`.
   It writes in place to `subj_dir` exactly as today (totalseg ignores `--out`).

2. **Add `_convert_chemotox(task: dict)`** — the new multi-label / target-spacing path
   (no `ct_raw`, no MRI, writes a fresh tree to `out_dir`):

```python
def _convert_chemotox(task: dict):
    """Convert one chemotox subject to the out tree. Returns (subj_id, status, sp, shape, None)."""
    subj_id = task["subj_id"]
    out_dir = Path(task["out_dir"]); out_dir.mkdir(parents=True, exist_ok=True)
    overwrite = task["overwrite"]; size = task["size"]; target_sp = task["target_spacing"]
    label_names = SOURCE_LABELS["chemotox"]
    ct_out = out_dir / "ct.npy"
    label_outs = {n: out_dir / f"{n}.npy" for n in label_names}
    if (ct_out.exists() and all(p.exists() for p in label_outs.values())
            and not overwrite and size is None):
        return subj_id, "skip", None, None, None
    try:
        raw, native_sp, labels = load_raw(task)
        vol = _normalise_ct(raw)
        if target_sp is not None:
            vol = _resample_to_spacing(vol, native_sp, target_sp, order=1)
            labels = {n: _resample_to_spacing(a, native_sp, target_sp, order=0)
                      for n, a in labels.items()}
            out_sp = [float(target_sp)] * 3
        else:
            out_sp = native_sp
        out_shape = list(vol.shape)
        np.save(ct_out, vol.astype(np.float16))
        for n, a in labels.items():
            np.save(label_outs[n], a.astype(np.uint8))
        if size is not None:  # optional fixed-cube sized variants (primary label only)
            size_str = f"{size[0]}x{size[1]}x{size[2]}"
            sp = tuple(out_sp)
            np.save(out_dir / f"ct_{size_str}.npy",
                    _iso_resize(vol.astype(np.float32), size, order=1, aa=True, spacing=sp).astype(np.float16))
            np.save(out_dir / f"label_{size_str}.npy",
                    _iso_resize(labels["label"], size, order=0, aa=False, spacing=sp))
    except Exception:
        return subj_id, traceback.format_exc(), None, None, None
    return subj_id, "ok", out_sp, out_shape, None
```

3. **Add the dispatcher** `convert_subject`:

```python
def convert_subject(task: dict):
    if task["source"] == "totalseg":
        return _convert_totalseg(task)
    return _convert_chemotox(task)
```

Note: `load_raw`'s totalseg branch is used only by `_convert_chemotox`'s counterpart
tests; `_convert_totalseg` keeps its own original inline CT+segmentations reading so
its output stays byte-identical (do not reroute it through `load_raw`).

- [ ] **Step 4: Run the load_raw test to verify it passes**

Run: `pytest tests/test_convert_chemotox.py -v`
Expected: PASS.

- [ ] **Step 5: Rewire `main()` to build task dicts, write meta.csv, keep totalseg identical**

Replace the task construction + pool loop in `main()`:

```python
    data_dir = args.data
    if data_dir is None:
        data_dir = str(COHORT_JSON) if args.source == "chemotox" else _default_data_dir()
    out_root = Path(args.out) if args.out else Path(data_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    subjects = enumerate_subjects(args.source, data_dir, out_root, args.limit)
    total = len(subjects)
    size = tuple(args.size) if args.size else None
    for t in subjects:
        t.update(overwrite=args.overwrite, size=size, target_spacing=args.target_spacing,
                 source=args.source, modality=args.modality, store_raw=args.store_raw)
    print(f"source={args.source} | {total} subjects | out={out_root} | "
          f"target_spacing={args.target_spacing} | size={size}")

    spacings_path = out_root / "spacings.json"
    spacings = json.load(open(spacings_path)) if spacings_path.exists() else {}
    stats_path = out_root / "ct_stats.json"
    ct_stats = json.load(open(stats_path)) if stats_path.exists() else {}

    done = ok = skipped = errors = 0
    t0 = time.time()
    with mp.Pool(processes=args.workers) as pool:
        for subj, status, sp, shape, subj_stats in pool.imap_unordered(
            convert_subject, subjects, chunksize=1
        ):
            done += 1
            if status == "ok":
                ok += 1
                if sp is not None and shape is not None:
                    spacings[subj] = {"spacing": sp, "shape": shape}
                if subj_stats is not None:
                    ct_stats[subj] = subj_stats
            elif status == "skip":
                skipped += 1
            else:
                errors += 1; print(f"\n[ERROR] {subj}:\n{status}")
            elapsed = time.time() - t0; rate = done / elapsed if elapsed else 0
            print(f"\r  {done}/{total} ok={ok} skip={skipped} err={errors} "
                  f"{rate:.1f} subj/s", end="", flush=True)

    if spacings:
        json.dump(spacings, open(spacings_path, "w"))
        print(f"\nSpacings -> {spacings_path} ({len(spacings)})")
    if ct_stats:
        json.dump(ct_stats, open(stats_path, "w"))
    # meta.csv for sources with no native split (chemotox): all subjects -> test
    if args.source == "chemotox":
        with open(out_root / "meta.csv", "w", newline="") as f:
            w = csv.writer(f, delimiter=";"); w.writerow(["image_id", "split"])
            for s in sorted(spacings): w.writerow([s, "test"])
        print(f"meta.csv -> {out_root/'meta.csv'} ({len(spacings)} test)")
    print(f"\nDone in {(time.time()-t0)/60:.1f} min — ok={ok} skip={skipped} err={errors}")
```

Also ensure `_CLASS_TO_IDX` is module-level (it already is) and reachable by `load_raw`.

- [ ] **Step 6: Smoke-convert 2 real chemotox subjects**

Run:
```bash
python scripts/convert_to_npy.py --source chemotox \
    --out /tmp/chemotox_smoke --target-spacing 1.5 --workers 2 --limit 2
```
Expected: 2 subject dirs each with `ct.npy`, `label.npy`, `bc.npy`; `spacings.json`
(2 entries, spacing `[1.5,1.5,1.5]`); `meta.csv` with 2 `test` rows. Verify:
```bash
python -c "
import numpy as np, json, glob
d=sorted(glob.glob('/tmp/chemotox_smoke/*/'))[0]
ct=np.load(d+'ct.npy'); lab=np.load(d+'label.npy'); bc=np.load(d+'bc.npy')
print('ct', ct.shape, ct.dtype, 'label ids', np.unique(lab)[:8], 'bc ids', np.unique(bc))
print(json.load(open('/tmp/chemotox_smoke/spacings.json')))
"
```
Expected: `bc ids` ⊆ `[0 1 2 3 4]`; label ids are project indices; ct is 1.5 mm-shaped.

- [ ] **Step 7: Verify totalseg source unaffected (dry parse)**

Run: `python scripts/convert_to_npy.py --source totalseg --limit 0 --help` then
`python -c "from scripts.convert_to_npy import enumerate_subjects; print('ok')"`
Expected: imports and totalseg enumeration path unchanged (no chemotox-only assumptions).

- [ ] **Step 8: Commit**

```bash
git add scripts/convert_to_npy.py tests/test_convert_chemotox.py
git commit -m "feat(convert): chemotox source — JSON enumerate, TS remap, bc ch0, multi-label npy + meta.csv"
```

---

### Task 4: `ChemoToxBCDataset` (4-class body-composition, use_crop)

**Files:**
- Create: `src/chemotox_dataset.py`
- Test: `tests/test_chemotox_dataset.py`

**Interfaces:**
- Consumes: `TotalSegInContextDataset` and its helpers (`_organ_crop_arrays`,
  `_place_image`, `_place_label`, `_resample_binary`, `_get_spacing`).
- Produces:
  - `BC_NAMES = ["muscle", "sat", "vat", "imat"]`, `BC_ID = {name: i+1}`.
  - `ChemoToxBCDataset(root, classes=BC_NAMES, image_size, split="test",
    context_size=1, max_subjects=None, eval_seed=0, use_crop=True,
    crop_spacing_mm=1.5, crop_jitter=None)` — reads `ct.npy` + `bc.npy`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_chemotox_dataset.py
import json
import numpy as np
import torch
from src.chemotox_dataset import ChemoToxBCDataset, BC_NAMES


def _make_tree(root, n_subjects=2, D=48):
    root.mkdir(parents=True, exist_ok=True)
    spac = {}
    for i in range(n_subjects):
        s = root / f"subj_{i}"; s.mkdir()
        ct = (np.random.rand(D, D, D).astype(np.float16))
        bc = np.zeros((D, D, D), dtype=np.uint8)
        bc[5:20, 5:20, 5:20] = 1     # muscle
        bc[25:40, 5:20, 5:20] = 2    # sat
        bc[5:20, 25:40, 5:20] = 3    # vat
        bc[25:40, 25:40, 5:20] = 4   # imat
        np.save(s / "ct.npy", ct); np.save(s / "bc.npy", bc)
        spac[f"subj_{i}"] = {"spacing": [1.5, 1.5, 1.5], "shape": [D, D, D]}
    json.dump(spac, open(root / "spacings.json", "w"))


def test_bc_dataset_items(tmp_path):
    root = tmp_path / "chemo"
    _make_tree(root)
    ds = ChemoToxBCDataset(root=root, classes=BC_NAMES, image_size=(32, 32, 32),
                           split="test", context_size=1, use_crop=True,
                           crop_spacing_mm=1.5, eval_seed=0)
    assert len(ds) == 2 * len(BC_NAMES)   # (subject, class) pairs
    item = ds[0]
    assert item["image"].shape == (1, 32, 32, 32)
    assert item["label"].shape == (32, 32, 32)
    assert item["context_in"].shape == (1, 1, 32, 32, 32)
    assert set(torch.unique(item["label"]).tolist()) <= {0, 1}
    assert item["label"].sum() > 0        # foreground present for the cropped tissue
    assert item["label_name"] in BC_NAMES
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_chemotox_dataset.py -v`
Expected: FAIL with `ModuleNotFoundError: src.chemotox_dataset`.

- [ ] **Step 3: Implement the subclass**

```python
# src/chemotox_dataset.py
"""In-context EVAL dataset over ChemoTox body-composition tissues (4 classes).

Reuses TotalSegInContextDataset for context sampling, eval-seed determinism, the
single-label __getitem__ path, and the collate contract. Reads the converted cache
tree (ct.npy + bc.npy + spacings.json, built at 1.5 mm iso by convert_to_npy
--source chemotox). use_crop-only, eval-only."""
import hashlib
import os
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from src.totalseg_dataloader_incontext import TotalSegInContextDataset

BC_NAMES = ["muscle", "sat", "vat", "imat"]
BC_ID = {n: i + 1 for i, n in enumerate(BC_NAMES)}


def _bc_centroids_for_subject(args) -> tuple[str, dict | None]:
    """Per-class centroid (native bc.npy voxel space) for one subject."""
    root, subj = args
    try:
        arr = np.load(Path(root) / subj / "bc.npy", mmap_mode="r")
        D, H, W = arr.shape
        d_g = np.arange(D, dtype=np.float32)[:, None, None]
        h_g = np.arange(H, dtype=np.float32)[None, :, None]
        w_g = np.arange(W, dtype=np.float32)[None, None, :]
        out: dict[str, tuple[int, int, int]] = {}
        for name, lid in BC_ID.items():
            m = (arr == lid)
            n = int(m.sum())
            if n == 0:
                continue
            out[name] = (int((d_g * m).sum() / n), int((h_g * m).sum() / n),
                         int((w_g * m).sum() / n))
        return subj, out
    except Exception:
        return subj, None


class ChemoToxBCDataset(TotalSegInContextDataset):
    def __init__(self, root, classes=BC_NAMES, image_size=(128, 128, 128),
                 split: Optional[str] = "test", context_size: int = 1,
                 max_subjects: Optional[int] = None, eval_seed: int = 0,
                 use_crop: bool = True, crop_spacing_mm: float = 1.5,
                 crop_jitter: Optional[int] = None):
        assert use_crop, "ChemoToxBCDataset is use_crop-only"
        super().__init__(
            root=root, classes=list(classes), image_size=image_size, split=split,
            context_size=context_size, max_subjects=max_subjects, aug_cfg=None,
            synth_method=None, p_synth=0.0, class_balanced=False, use_crop=True,
            crop_spacing_mm=crop_spacing_mm, crop_jitter=crop_jitter,
            num_labels_per_sample=1, eval_seed=eval_seed, raw_ct=False, modality="ct")

    # --- overrides -----------------------------------------------------------
    def _get_subjects(self, split, meta_csv, max_subjects) -> list[str]:
        assert split in (None, "test"), f"eval-only (split={split!r})"
        subs = sorted(p.name for p in self.root.iterdir()
                      if p.is_dir() and (p / "bc.npy").exists())
        return subs[:max_subjects] if max_subjects is not None else subs

    def _load_or_build_cache(self) -> dict:
        """Every subject carries all 4 diffuse tissues -> trivial subject->classes."""
        return {s: frozenset(BC_NAMES) for s in
                (p.name for p in self.root.iterdir()
                 if p.is_dir() and (p / "bc.npy").exists())}

    def _load_or_build_bbox_cache(self) -> dict:
        subs = sorted(p.name for p in self.root.iterdir()
                      if p.is_dir() and (p / "bc.npy").exists())
        key = hashlib.sha256(("bc_centroid|" + "|".join(subs)).encode()).hexdigest()[:12]
        cache_path = self.root / f".bc_centroid_cache_{key}.pkl"
        if cache_path.exists():
            with open(cache_path, "rb") as f:
                return pickle.load(f)
        n_workers = min(16, os.cpu_count() or 1)
        cache: dict = {}
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futs = {ex.submit(_bc_centroids_for_subject, (str(self.root), s)): s for s in subs}
            for fut in as_completed(futs):
                subj, res = fut.result()
                if res is not None:
                    cache[subj] = res
        with open(cache_path, "wb") as f:
            pickle.dump(cache, f)
        return cache

    def _load_crop(self, subj: str, cls: str):
        """Organ-centred native crop of fixed physical extent (T*crop_spacing_mm),
        resampled to T³. Crops ct.npy + (bc.npy == BC_ID[cls])."""
        subj_dir = self.root / subj
        local_id = BC_ID[cls]
        label_mm = np.load(subj_dir / "bc.npy", mmap_mode="r")
        D, H, W = label_mm.shape
        center = self._bbox_cache.get(subj, {}).get(cls) or (D // 2, H // 2, W // 2)
        sp = self._get_spacing(subj).tolist()
        crop_ct, crop_lbl, out_sizes, pad_lo = self._organ_crop_arrays(
            subj_dir, label_mm, center, sp)
        image_t = self._place_image(crop_ct, out_sizes, pad_lo)
        label_t = self._place_label(
            self._resample_binary(crop_lbl == local_id, tuple(out_sizes)), out_sizes, pad_lo)
        return image_t, label_t

    def _load(self, subj: str, cls: str):
        return self._load_crop(subj, cls)
```

Note: `_organ_crop_arrays` calls `self._load_native_ct_mmap(subj_dir)`, which with
`raw_ct=False` reads `ct.npy` — exactly our cache file. No override needed there.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_chemotox_dataset.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/chemotox_dataset.py tests/test_chemotox_dataset.py
git commit -m "feat(dataset): ChemoToxBCDataset — 4-class body-composition in-context eval"
```

---

### Task 5: Wire sources into `experiments/3d/common.py`

**Files:**
- Modify: `experiments/3d/common.py`
- Test: `tests/test_chemotox_routing.py`

**Interfaces:**
- Consumes: `ChemoToxBCDataset` (Task 4), `TotalSegInContextDataset`.
- Produces: `build_dataset(cfg, "test")` returns `TotalSegInContextDataset` for
  `source=chemotox` and `ChemoToxBCDataset` for `source=chemotox_bc`;
  `make_eval_loader` routes both correctly (chemotox via the direct-totalseg branch,
  chemotox_bc via the subclass special-case).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_chemotox_routing.py
import json
import numpy as np
from omegaconf import OmegaConf
from experiments_common_shim import build_dataset  # see Step 3 note
from src.chemotox_dataset import ChemoToxBCDataset, BC_NAMES
from src.totalseg_dataloader_incontext import TotalSegInContextDataset


def _make_bc_tree(root, D=48):
    root.mkdir(parents=True, exist_ok=True); spac = {}
    for i in range(2):
        s = root / f"subj_{i}"; s.mkdir()
        np.save(s / "ct.npy", np.random.rand(D, D, D).astype(np.float16))
        bc = np.zeros((D, D, D), np.uint8)
        bc[5:20, 5:20, 5:20] = 1; bc[25:40, 25:40, 5:20] = 2
        bc[5:20, 25:40, 5:20] = 3; bc[25:40, 5:20, 5:20] = 4
        np.save(s / "bc.npy", bc)
        spac[f"subj_{i}"] = {"spacing": [1.5, 1.5, 1.5], "shape": [D, D, D]}
    json.dump(spac, open(root / "spacings.json", "w"))


def _cfg(source, root):
    return OmegaConf.create({
        "data": {"source": source, "image_size": [32, 32, 32], "context_size": 1,
                 "use_crop": True, "crop_spacing_mm": 1.5, "val_classes": "benchmark",
                 "train_classes": "benchmark", "max_val_subjects": None,
                 "max_train_subjects": None},
        "paths": {"chemotox": str(root)},
        "eval": {"seed": 0, "crop_jitter": 0},
    })


def test_build_dataset_chemotox_bc(tmp_path):
    root = tmp_path / "chemo"; _make_bc_tree(root)
    ds = build_dataset(_cfg("chemotox_bc", root), "test")
    assert isinstance(ds, ChemoToxBCDataset)
    assert len(ds) == 2 * len(BC_NAMES)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_chemotox_routing.py -v`
Expected: FAIL (chemotox_bc not routed / import shim missing).

- [ ] **Step 3: Implement wiring**

In `experiments/3d/common.py`:

1. Import at top: `from src.chemotox_dataset import ChemoToxBCDataset, BC_NAMES`.

2. Add `chemotox` to the totalseg source set so `_source_root` resolves it:

```python
# near _TOTALSEG_SOURCES definition (~line 29)
_TOTALSEG_SOURCES = {"totalseg", "totalsegmri", "chemotox"}
```

Confirm `_source_root` returns `is_mri = source == "totalsegmri"` (so chemotox → CT).

3. In `_source_root`, before the `_TOTALSEG_SOURCES` check, special-case chemotox_bc
   (its root is the same chemotox tree):

```python
    if source == "chemotox_bc":
        root = cfg.paths.get("chemotox")
        if root is None:
            raise ValueError("cfg.paths.chemotox is not set (needed for source=chemotox_bc)")
        return source, root, False
```

4. In `build_dataset`, add a branch alongside the `totalseg_more_labels` branch:

```python
    if cfg.data.get("source") == "chemotox_bc":
        d = cfg.data
        root = cfg.paths.get("chemotox")
        return ChemoToxBCDataset(
            root=root, classes=BC_NAMES, image_size=tuple(d.image_size),
            split=split, context_size=d.context_size,
            max_subjects=d.get("max_val_subjects"),
            eval_seed=int(cfg.get("eval", {}).get("seed", 0)),
            use_crop=True, crop_spacing_mm=d.get("crop_spacing_mm", 1.5),
            crop_jitter=cfg.get("eval", {}).get("crop_jitter", None))
```

5. In `make_eval_loader`, add `chemotox_bc` to the subclass special-case tuple so it
   is routed through `build_dataset` and honors the spacing sampler:

```python
    if d.get("source") in ("omnisynth3d", "anchor_synth3d", "totalseg_more_labels",
                            "chemotox_bc"):
```

(`chemotox` needs no change here — it falls through to the direct-totalseg branch,
which now accepts it because it is in `_TOTALSEG_SOURCES`.)

6. For the test's import shim, add `tests/experiments_common_shim.py`:

```python
# tests/experiments_common_shim.py — expose experiments/3d/common.py under a stable name
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "3d"))
from common import build_dataset, make_eval_loader  # noqa: E402,F401
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_chemotox_routing.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add experiments/3d/common.py tests/test_chemotox_routing.py tests/experiments_common_shim.py
git commit -m "feat(3d/common): route source=chemotox (base) and chemotox_bc (subclass)"
```

---

### Task 6: Dataset configs + `paths.chemotox`, end-to-end smoke

**Files:**
- Create: `configs/experiment/3d/dataset/chemotox.yaml`
- Create: `configs/experiment/3d/dataset/chemotox_bc.yaml`
- Modify: `configs/cluster/nfs.yaml`

**Interfaces:**
- Consumes: everything above. Produces two Hydra `dataset=` groups and a cluster path.

- [ ] **Step 1: Add the cluster path**

In `configs/cluster/nfs.yaml`, under `paths:` add:

```yaml
  chemotox:    /nfs/data/nii/data1/Analysis/camaret___in_context_segmentation/ANALYSIS_20251122/data/chemotox
```

- [ ] **Step 2: Write `chemotox.yaml` (117 totalseg classes)**

```yaml
# @package _global_
# ChemoTox cohort, TotalSegmentator labels (117 classes). Eval-only. Reuses the base
# TotalSegInContextDataset use_crop path over the 1.5mm-iso cache written by
# `convert_to_npy.py --source chemotox --target-spacing 1.5`.
# NOTE: class presence is FOV-driven across the 366 subjects (thorax vs abdomen);
# sparse classes simply contribute fewer samples.
defaults:
  - override /augmentations: multiverseg_v2
data:
  source: chemotox
  image_size: [128, 128, 128]
  context_size: 1
  use_crop: true
  crop_spacing_mm: 1.5          # MUST be >= cache spacing (1.5mm); base use_crop assumes 1.5mm native
  raw_ct: false
  mask_downsample: nearest
  mask_occupancy_thr: 0.5
  num_labels_per_sample: 1
  max_val_subjects: null
  val_classes: benchmark
  train_classes: benchmark      # unused (eval-only) but resolve_classes needs a value
```

- [ ] **Step 3: Write `chemotox_bc.yaml` (4 tissues)**

```yaml
# @package _global_
# ChemoTox cohort, body-composition tissues (muscle/sat/vat/imat). Eval-only.
# Served by ChemoToxBCDataset over bc.npy in the same 1.5mm-iso cache.
defaults:
  - override /augmentations: multiverseg_v2
data:
  source: chemotox_bc
  image_size: [128, 128, 128]
  context_size: 1
  use_crop: true
  crop_spacing_mm: 1.5
  raw_ct: false
  num_labels_per_sample: 1
  max_val_subjects: null
  val_classes: [muscle, sat, vat, imat]
  train_classes: [muscle, sat, vat, imat]
```

- [ ] **Step 4: Hydra compose smoke (no data needed)**

Run:
```bash
python -c "
from hydra import initialize_config_dir, compose
from pathlib import Path
cfgdir = str(Path('configs').resolve())
with initialize_config_dir(config_dir=cfgdir, version_base='1.3'):
    for ds in ['chemotox', 'chemotox_bc']:
        cfg = compose(config_name='eval', overrides=[f'dataset={ds}', 'cluster=nfs'])
        print(ds, cfg.data.source, cfg.paths.chemotox, list(cfg.data.image_size))
"
```
Expected: prints each source, the chemotox path, and `[128,128,128]`; no compose error.

- [ ] **Step 5: End-to-end eval smoke on the 2 smoke-converted subjects**

(Reuses `/tmp/chemotox_smoke` from Task 3 Step 6; converts one more so context exists —
`--limit 3`.) Run a tiny eval with a cheap model and few subjects:
```bash
python scripts/convert_to_npy.py --source chemotox --out /tmp/chemotox_smoke \
    --target-spacing 1.5 --workers 3 --limit 3
python experiments/3d/eval.py dataset=chemotox    cluster=nfs \
    paths.chemotox=/tmp/chemotox_smoke eval.model=medverse eval.n_subjects=3 \
    eval.batch_size=1 eval.workers=0 eval.save_figures=false
python experiments/3d/eval.py dataset=chemotox_bc cluster=nfs \
    paths.chemotox=/tmp/chemotox_smoke eval.model=medverse eval.n_subjects=3 \
    eval.batch_size=1 eval.workers=0 eval.save_figures=false
```
Expected: both runs complete and print per-class Dice rows (numbers may be poor on 3
subjects — this checks the pipeline end to end, not accuracy).

- [ ] **Step 6: Commit**

```bash
git add configs/experiment/3d/dataset/chemotox.yaml \
        configs/experiment/3d/dataset/chemotox_bc.yaml configs/cluster/nfs.yaml
git commit -m "feat(configs): chemotox + chemotox_bc dataset groups and paths.chemotox"
```

---

### Task 7: Full conversion (operator step)

Not a code task — the one-time cache build. Run when ready to eval for real.

- [ ] **Step 1: Build the full 1.5 mm cache (~35 GB, 366 subjects)**

```bash
python scripts/convert_to_npy.py --source chemotox \
    --out /nfs/data/nii/data1/Analysis/camaret___in_context_segmentation/ANALYSIS_20251122/data/chemotox \
    --target-spacing 1.5 --workers 32
```
Expected: `ok=366` (or `ok + err` accounting for any unreadable subjects, logged),
`spacings.json` with 366 entries, `meta.csv` with 366 `test` rows.

- [ ] **Step 2: Sanity-check the built tree**

```bash
python -c "
import json, glob, numpy as np
root='/nfs/.../ANALYSIS_20251122/data/chemotox'  # fill in
sp=json.load(open(root+'/spacings.json')); print('subjects', len(sp))
d=sorted(glob.glob(root+'/*/'))[0]
print('bc ids', np.unique(np.load(d+'bc.npy')))
print('label ids', np.unique(np.load(d+'label.npy'))[:10])
"
```
Expected: 366 subjects; bc ids ⊆ {0,1,2,3,4}; label ids in project numbering.

- [ ] **Step 3: Real eval runs**

```bash
python experiments/3d/eval.py dataset=chemotox \
    eval.model=<model> eval.checkpoint=<ckpt> eval.n_subjects=50
python experiments/3d/eval.py dataset=chemotox_bc \
    eval.model=<model> eval.checkpoint=<ckpt> eval.n_subjects=50
```
Expected: per-class Dice / time / GFLOPs written to `${paths.results}/3d_eval`.

---

## Self-Review

**Spec coverage:** Component 1 (generalize convert) → Tasks 2–3; component 2 (chemotox
reuses base) → Task 5 (`_TOTALSEG_SOURCES`) + Task 6 config; component 3 (ChemoToxBCDataset)
→ Task 4; component 4 (wiring) → Task 5; component 5 (configs) → Task 6; the TS-numbering
risk → Task 1 (remap-by-name + cohort-stats guard). Full conversion + real eval → Task 7.
All spec sections covered.

**Placeholder scan:** No TBD/TODO. `<model>`/`<ckpt>`/`root='/nfs/...'` in Task 7 are
operator-supplied runtime values (that task is explicitly a manual operator step), not
code placeholders.

**Type consistency:** `load_raw`/`enumerate_subjects`/`convert_subject` all consume the
same task dict shape (`source`, `inputs`, `out_dir`, `subj_id`, `overwrite`, `size`,
`target_spacing`, `modality`). `BC_NAMES`/`BC_ID` consistent across Tasks 4–6.
`remap_ts_total` name consistent Tasks 1/3. `_resample_to_spacing(vol, native_sp,
target_sp, order)` consistent Tasks 2/3.
