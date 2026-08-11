# Coords-Function Synthetic Labels Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a coords-driven synthetic-label mode to `TotalSegInContextDataset` that emits K+1 position-corresponding (image, label) volumes across *different* subjects, using the `coords.npy` canonical body frame.

**Architecture:** A synthetic label is a smooth field `f(coords)→[0,1]` with random params, evaluated per subject on its `coords.npy` — so correspondence is by construction. A pure field/task library (`src/coords_synth.py`) is unit-tested standalone; the dataloader wires it in as a new `p_coords` synth mode. Phase A ships hard (binary) labels that drop into the existing integer pipeline unchanged; Phase B adds soft (float) labels behind a flag.

**Tech Stack:** Python, NumPy, PyTorch, Hydra/OmegaConf configs, pytest.

## Global Constraints

- Log changes to `docs/logs.md` (project rule, CLAUDE.md).
- Write understandable code with short docstrings; write tests only when necessary (CLAUDE.md).
- Run everything on loki with `.venv_thor_fresh/bin/python` (numpy/nibabel/torch available); data root `/nfs/data/nii/data1/Analysis/camaret___in_context_segmentation/ANALYSIS_20251122/data/totalseg`.
- Every TotalSeg scan has `coords.npy` (native 1.5mm grid, float16, shape (X,Y,Z,3)), voxel-aligned with `ct.npy` / `label.npy`.
- Field families are **localized only**: `gaussian`, `ellipsoid`, `cyl_capped`. Scale floor ≈ 40 mm. Edge width `EDGE = 4.0` mm.
- Guards: `min_mass` (in-crop label mass) and `min_hi` ≈ 0.15 (cross-subject anatomy consistency).
- Phase A labels are `int64` binary (mirrors existing `_get_synth_item`); no loss/metric/collate changes.

---

### Task 1: Field & consistency library (`src/coords_synth.py`)

Canonical home for the field primitives (currently duplicated in `experiments/3d/universal_coords/coords_synth_consistency.py`). Pure NumPy, no torch, no I/O.

**Files:**
- Create: `src/coords_synth.py`
- Test: `tests/test_coords_synth.py`
- Modify: `experiments/3d/universal_coords/coords_synth_consistency.py` (import from `src.coords_synth` instead of local defs)
- Modify: `experiments/3d/universal_coords/plot_coords_synth.py` (import from `src.coords_synth`)

**Interfaces:**
- Produces:
  - `EDGE: float = 4.0`, `LOCALIZED: tuple = ("gaussian", "ellipsoid", "cyl_capped")`
  - `sample_field(family: str, scale: float, ref_co: np.ndarray, rng) -> dict` — params dict, always includes keys `"family"` and `"mu"` (3,).
  - `eval_field(p: dict, co: np.ndarray) -> np.ndarray` — `co` is `(N,3)`, returns soft `(N,)` in [0,1]; dispatches on `p["family"]`.
  - `coords_aabb(co_flat: np.ndarray, lab_flat: np.ndarray) -> tuple[np.ndarray, np.ndarray]` — `(lo(3,), hi(3,))` over labelled voxels.
  - `soft_hist(lab_flat: np.ndarray, w: np.ndarray) -> np.ndarray | None` — bg-excluded, normalised length-256 histogram, or None.
  - `pairwise_hi(hists: list[np.ndarray]) -> float` — mean pairwise `min`-intersection; 0.0 if <2.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_coords_synth.py
import numpy as np
from src.coords_synth import (sample_field, eval_field, coords_aabb,
                              soft_hist, pairwise_hi, LOCALIZED, EDGE)


def _grid(n=20):
    g = np.stack(np.meshgrid(*[np.arange(n)] * 3, indexing="ij"), -1).reshape(-1, 3).astype(np.float32)
    return g  # coords == voxel index here (fine for math tests)


def test_gaussian_peaks_at_mu_and_decays():
    co = _grid()
    p = {"family": "gaussian", "mu": np.array([10., 10., 10.]),
         "Sinv": np.diag([1 / 9.] * 3)}  # sigma=3
    w = eval_field(p, co)
    at_mu = w[np.argmin(np.linalg.norm(co - p["mu"], axis=1))]
    far = w[np.argmin(np.linalg.norm(co - np.array([0., 0., 0.]), axis=1))]
    assert at_mu > 0.99 and far < 0.01


def test_ellipsoid_hard_inside_outside():
    co = _grid()
    p = {"family": "ellipsoid", "mu": np.array([10., 10., 10.]),
         "R": np.eye(3), "radii": np.array([4., 4., 4.])}
    w = eval_field(p, co)
    inside = w[np.argmin(np.linalg.norm(co - np.array([11., 10., 10.]), axis=1))]
    outside = w[np.argmin(np.linalg.norm(co - np.array([17., 10., 10.]), axis=1))]
    assert inside > 0.5 and outside < 0.5


def test_cyl_capped_bounded_along_axis():
    co = _grid()
    p = {"family": "cyl_capped", "mu": np.array([10., 10., 10.]),
         "axis": np.array([0., 0., 1.]), "r": 3.0, "L": 4.0}
    w = eval_field(p, co)
    on_axis_in = w[np.argmin(np.linalg.norm(co - np.array([10., 10., 13.]), axis=1))]
    on_axis_out = w[np.argmin(np.linalg.norm(co - np.array([10., 10., 18.]), axis=1))]
    assert on_axis_in > 0.5 and on_axis_out < 0.5


def test_sample_field_localized_has_family_and_mu():
    rng = np.random.default_rng(0)
    co = _grid()
    for fam in LOCALIZED:
        p = sample_field(fam, 40.0, co, rng)
        assert p["family"] == fam and p["mu"].shape == (3,)
        assert eval_field(p, co).shape == (len(co),)


def test_coords_aabb_over_labelled_only():
    co = _grid(4)
    lab = np.zeros(len(co), np.uint8)
    lab[0] = 1; lab[-1] = 2  # corners (0,0,0) and (3,3,3)
    lo, hi = coords_aabb(co, lab)
    assert np.allclose(lo, [0, 0, 0]) and np.allclose(hi, [3, 3, 3])


def test_hist_and_pairwise_hi():
    lab = np.array([0, 1, 1, 2, 2, 2])
    w = np.ones(6, np.float32)
    h = soft_hist(lab, w)
    assert h is not None and abs(h.sum() - 1.0) < 1e-6 and h[0] == 0.0
    assert abs(pairwise_hi([h, h]) - 1.0) < 1e-6
    h2 = soft_hist(np.array([0, 3, 3, 3, 3, 3]), w)
    assert pairwise_hi([h, h2]) < 1e-6
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv_thor_fresh/bin/python -m pytest tests/test_coords_synth.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.coords_synth'`

- [ ] **Step 3: Write `src/coords_synth.py`**

```python
"""Coords-function synthetic labels: a synthetic label is a smooth field
f(coords)->[0,1] evaluated per subject on its coords.npy, so the same field
yields position-corresponding labels across subjects. Localized (bounded,
anchored) families only — unbounded primitives fail on heterogeneous FOVs.
Pure NumPy; the dataloader wires this in as a p_coords synth mode.
"""
import numpy as np

EDGE = 4.0                                        # hard-edge width (mm)
LOCALIZED = ("gaussian", "ellipsoid", "cyl_capped")


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def f_gaussian(co, p):
    d = co - p["mu"]
    return np.exp(-0.5 * np.einsum("ni,ij,nj->n", d, p["Sinv"], d))


def f_ellipsoid(co, p):
    d = (co - p["mu"]) @ p["R"].T
    dn = np.sqrt(((d / p["radii"]) ** 2).sum(1))
    return _sigmoid((1.0 - dn) * 20.0)            # steep -> ~hard at 0.5


def f_cyl_capped(co, p):
    d = co - p["mu"]
    a = d @ p["axis"]
    rd = np.linalg.norm(d - np.outer(a, p["axis"]), axis=1)
    return _sigmoid((p["r"] - rd) / EDGE) * _sigmoid((p["L"] - np.abs(a)) / EDGE)


FIELDS = {"gaussian": f_gaussian, "ellipsoid": f_ellipsoid, "cyl_capped": f_cyl_capped}


def eval_field(p, co):
    return FIELDS[p["family"]](co, p)


def rand_unit(rng):
    v = rng.normal(size=3); return v / np.linalg.norm(v)


def rand_rot(rng):
    Q, R = np.linalg.qr(rng.normal(size=(3, 3)))
    return Q * np.sign(np.diag(R))


def sample_field(family, scale, ref_co, rng):
    """Sample field params anchored at a real canonical location (a coords value
    of a random labelled voxel). scale is the characteristic size in mm."""
    mu = ref_co[rng.integers(len(ref_co))].astype(np.float64)
    if family == "gaussian":
        aniso = rng.uniform(0.6, 1.6, 3)
        return {"family": family, "mu": mu, "Sinv": np.diag(1.0 / (scale * aniso) ** 2)}
    if family == "ellipsoid":
        return {"family": family, "mu": mu, "R": rand_rot(rng),
                "radii": scale * rng.uniform(0.6, 1.6, 3)}
    if family == "cyl_capped":
        return {"family": family, "mu": mu, "axis": rand_unit(rng),
                "r": scale, "L": scale * rng.uniform(1.0, 2.5)}
    raise ValueError(f"unknown/unbounded family: {family}")


def coords_aabb(co_flat, lab_flat):
    """Coords bounding box over labelled voxels = the canonical body region a
    scan covers. Cheap FOV pre-filter for subject grouping."""
    c = co_flat[lab_flat > 0]
    return c.min(0), c.max(0)


def soft_hist(lab_flat, w):
    """bg-excluded, soft-weighted, normalised anatomy histogram (length 256)."""
    h = np.bincount(lab_flat, weights=w, minlength=256)[:256].astype(np.float64)
    h[0] = 0.0
    s = h.sum()
    return None if s <= 0 else h / s


def pairwise_hi(hists):
    if len(hists) < 2:
        return 0.0
    return float(np.mean([np.minimum(hists[i], hists[j]).sum()
                          for i in range(len(hists)) for j in range(i + 1, len(hists))]))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv_thor_fresh/bin/python -m pytest tests/test_coords_synth.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Point the experiment scripts at the library (DRY)**

In `experiments/3d/universal_coords/coords_synth_consistency.py`, replace the local `f_*`, `sample_params`, `FIELDS`, `coords_aabb`, `soft_hist`, `LOCALIZED`, `rand_unit`, `rand_rot` defs with:
```python
from src.coords_synth import (EDGE, FIELDS, LOCALIZED, eval_field, coords_aabb,
                              soft_hist, rand_unit, rand_rot)
from src.coords_synth import sample_field as sample_params  # keep call sites working
```
In `experiments/3d/universal_coords/plot_coords_synth.py`, update its import line to pull the same names from `src.coords_synth` (drop the import from `coords_synth_consistency` for those symbols; keep `TS, DS, MIN_MASS, subject_ids, load` there).

- [ ] **Step 6: Verify the montage still renders (regression)**

Run: `.venv_thor_fresh/bin/python experiments/3d/universal_coords/plot_coords_synth.py`
Expected: `saved .../figs/coords_synth_examples.png`, no import errors.

- [ ] **Step 7: Commit**

```bash
git add src/coords_synth.py tests/test_coords_synth.py \
  experiments/3d/universal_coords/coords_synth_consistency.py \
  experiments/3d/universal_coords/plot_coords_synth.py
git commit -m "feat(coords-synth): field & consistency library (src/coords_synth.py)"
```

---

### Task 2: FOV-aware task builder (`src/coords_synth.py`)

Add the multi-subject assembly: sample a field, FOV-filter subjects by AABB, apply the mass + consistency guards, return the K+1 subjects with per-subject crop centers.

**Files:**
- Modify: `src/coords_synth.py`
- Test: `tests/test_coords_synth.py`

**Interfaces:**
- Consumes: `sample_field`, `eval_field`, `soft_hist`, `pairwise_hi`, `LOCALIZED` (Task 1).
- Produces:
  - `build_coords_task(pool, get_coords, aabb, K, rng, *, families=LOCALIZED, scale_lo=40.0, scale_hi=140.0, min_mass=40.0, min_hi=0.15, retries=120) -> dict | None`
    where `get_coords(sid) -> (co_flat (N,3) float32, lab_flat (N,) int, shape (3,) int)` are strided arrays, and `aabb: dict[sid]->(lo,hi)`.
    Returns `{"params": dict, "family": str, "picks": [(sid, center_ijk (3,) int)], "hi": float}` (len(picks)==K+1) or `None`.
    `center_ijk` is the voxel index (in the STRIDED grid the loader passes) of the region centroid; the caller rescales by its stride.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_coords_synth.py
from src.coords_synth import build_coords_task


def _fake_subject(offset, n=24):
    """coords == voxel index + offset; a labelled 6^3 block near the center."""
    idx = np.stack(np.meshgrid(*[np.arange(n)] * 3, indexing="ij"), -1).reshape(-1, 3)
    co = (idx + offset).astype(np.float32)
    lab = np.zeros(len(idx), np.uint8)
    c = (idx >= 9) & (idx <= 14)
    lab[c.all(1)] = 5
    return co, lab, np.array([n, n, n])


def test_build_task_groups_fov_matching_subjects():
    subs = {"a": _fake_subject(np.zeros(3)), "b": _fake_subject(np.zeros(3)),
            "c": _fake_subject(np.zeros(3))}
    from src.coords_synth import coords_aabb
    aabb = {s: coords_aabb(subs[s][0], subs[s][1]) for s in subs}
    get = lambda s: subs[s]
    rng = np.random.default_rng(0)
    task = build_coords_task(list(subs), get, aabb, K=2, rng=rng,
                             families=("ellipsoid",), scale_lo=3.0, scale_hi=4.0,
                             min_mass=1.0, min_hi=0.0)
    assert task is not None and len(task["picks"]) == 3
    for sid, center in task["picks"]:
        assert center.shape == (3,)


def test_build_task_fov_filter_excludes_disjoint():
    # 'far' subject's labelled region sits at coords far from a/b's region.
    subs = {"a": _fake_subject(np.zeros(3)), "b": _fake_subject(np.zeros(3)),
            "far": _fake_subject(np.array([500, 500, 500]))}
    from src.coords_synth import coords_aabb
    aabb = {s: coords_aabb(subs[s][0], subs[s][1]) for s in subs}
    # Force the anchor to come from 'a' by making it the only ref; check 'far' never picked.
    rng = np.random.default_rng(1)
    seen_far = False
    for _ in range(10):
        task = build_coords_task(["a", "b", "far"], lambda s: subs[s], aabb, K=1, rng=rng,
                                 families=("gaussian",), scale_lo=3.0, scale_hi=3.0,
                                 min_mass=1.0, min_hi=0.0)
        if task:
            seen_far |= any(sid == "far" for sid, _ in task["picks"])
    assert not seen_far
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv_thor_fresh/bin/python -m pytest tests/test_coords_synth.py -k build_task -v`
Expected: FAIL — `ImportError: cannot import name 'build_coords_task'`

- [ ] **Step 3: Implement `build_coords_task`**

```python
# add to src/coords_synth.py
def _centroid_ijk(w, shape):
    """Voxel index of the soft field's centroid on a (flattened) grid of `shape`."""
    idx = np.arange(w.size)
    tot = w.sum()
    flat = int(round((idx * w).sum() / tot)) if tot > 0 else w.argmax()
    return np.array(np.unravel_index(flat, tuple(shape)))


def build_coords_task(pool, get_coords, aabb, K, rng, *, families=LOCALIZED,
                      scale_lo=40.0, scale_hi=140.0, min_mass=40.0, min_hi=0.15,
                      retries=120):
    """One position-corresponding task: sample a localized field anchored on a
    reference subject, keep only subjects whose coords-AABB contains the anchor
    (FOV filter), apply the mass guard, and require the K+1 picks to agree on
    anatomy (pairwise HI >= min_hi). Returns the field + picks or None."""
    for _ in range(retries):
        ref = pool[rng.integers(len(pool))]
        ref_co, _, _ = get_coords(ref)
        family = families[rng.integers(len(families))]
        scale = float(rng.uniform(scale_lo, scale_hi))
        p = sample_field(family, scale, ref_co, rng)
        mu = p["mu"]
        cand = [s for s in pool
                if (aabb[s][0] <= mu).all() and (mu <= aabb[s][1]).all()]
        picks, hists = [], []
        for s in rng.permutation(cand):
            co, lab, shape = get_coords(s)
            w = eval_field(p, co)
            if w.sum() < min_mass:
                continue
            h = soft_hist(lab, w)
            if h is None:
                continue
            picks.append((s, _centroid_ijk(w, shape))); hists.append(h)
            if len(picks) == K + 1:
                break
        if len(picks) == K + 1 and pairwise_hi(hists) >= min_hi:
            return {"params": p, "family": family, "picks": picks,
                    "hi": pairwise_hi(hists)}
    return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv_thor_fresh/bin/python -m pytest tests/test_coords_synth.py -k build_task -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/coords_synth.py tests/test_coords_synth.py
git commit -m "feat(coords-synth): FOV-aware multi-subject task builder"
```

---

### Task 3: Dataloader init — coords pool + AABB cache + config knobs

Wire the coords pool and a disk-cached AABB index into `TotalSegInContextDataset.__init__`, mirroring the existing synth/scan caches.

**Files:**
- Modify: `src/totalseg_dataloader_incontext.py` (`__init__`, roughly around lines 213–360)

**Interfaces:**
- Consumes: `coords_aabb` (Task 1).
- Produces: instance attrs `self.p_coords: float`, `self.coords_fname: str`, `self._coords_subjects: list[str]`, `self._coords_aabb: dict[str, tuple[np.ndarray,np.ndarray]]`, plus field-sampling attrs `self._coords_cfg: dict`. New `__init__` kwargs: `p_coords: float = 0.0`, `coords_fname: str = "coords.npy"`, `coords_families=("gaussian","ellipsoid","cyl_capped")`, `coords_scale_mm=(40.0,140.0)`, `coords_min_mass=40.0`, `coords_min_hi=0.15`, `coords_ds=4` (stride for center-finding).

- [ ] **Step 1: Add constructor kwargs**

In the `__init__` signature (near `p_synth: float = 0.5,`), add:
```python
        p_coords: float = 0.0,
        coords_fname: str = "coords.npy",
        coords_families: tuple = ("gaussian", "ellipsoid", "cyl_capped"),
        coords_scale_mm: tuple = (40.0, 140.0),
        coords_min_mass: float = 40.0,
        coords_min_hi: float = 0.15,
        coords_ds: int = 4,
```

- [ ] **Step 2: Store config + build the coords pool/AABB cache**

After the existing synth-cache block (after `self._synth_subjects`/`_synth_sv_ids` are set, ~line 358), add:
```python
        self.p_coords = p_coords
        self.coords_fname = coords_fname
        self._coords_cfg = dict(families=tuple(coords_families),
                                scale_lo=float(coords_scale_mm[0]),
                                scale_hi=float(coords_scale_mm[1]),
                                min_mass=float(coords_min_mass),
                                min_hi=float(coords_min_hi), ds=int(coords_ds))
        self._coords_subjects, self._coords_aabb = (
            self._load_or_build_coords_cache(subjects) if p_coords > 0 else ([], {}))
        if p_coords > 0:
            print(f"Coords synth: p_coords={p_coords} | "
                  f"{len(self._coords_subjects)} subjects | "
                  f"families={self._coords_cfg['families']} "
                  f"scale={coords_scale_mm}mm", flush=True)
```

- [ ] **Step 3: Implement the AABB cache builder**

Add this method next to `_load_or_build_synth_cache` (~line 411). It reads each subject's `coords.npy` + `label.npy` on the `coords_ds` stride once and pickles the AABB index, keyed by a SHA of the subject list (mirrors `_cache_path`).

```python
    def _load_or_build_coords_cache(self, subjects):
        """Return (coords_subjects, {subject: (lo, hi)}) — the coords AABB over
        labelled voxels, for the FOV pre-filter. Disk-cached like the SV cache."""
        import hashlib, pickle
        from src.coords_synth import coords_aabb
        subs = [s for s in subjects if (self.root / s / self.coords_fname).exists()]
        key = hashlib.sha1((self.coords_fname + "|".join(subs)).encode()).hexdigest()[:16]
        cache_path = self.root / f".coords_aabb_cache_{key}.pkl"
        if cache_path.exists():
            with open(cache_path, "rb") as f:
                aabb = pickle.load(f)
            print(f"Loaded coords AABB cache ({len(aabb)} subjects)", flush=True)
            return subs, aabb
        print(f"Building coords AABB cache for {len(subs)} subjects...", flush=True)
        ds = self._coords_cfg["ds"]
        aabb = {}
        for i, s in enumerate(subs):
            co = np.load(self.root / s / self.coords_fname, mmap_mode="r")[::ds, ::ds, ::ds]
            lab = np.load(self.root / s / "label.npy", mmap_mode="r")[::ds, ::ds, ::ds]
            aabb[s] = coords_aabb(np.asarray(co, np.float32).reshape(-1, 3),
                                  np.asarray(lab).reshape(-1))
            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(subs)}", flush=True)
        with open(cache_path, "wb") as f:
            pickle.dump(aabb, f)
        return subs, aabb
```

- [ ] **Step 4: Smoke-verify init builds the cache**

Run:
```bash
.venv_thor_fresh/bin/python -c "
import sys; sys.path.insert(0, '.')
from src.totalseg_dataloader_incontext import TotalSegInContextDataset
from data.totalseg_classes import BALANCED_CLASSES
r='/nfs/data/nii/data1/Analysis/camaret___in_context_segmentation/ANALYSIS_20251122/data/totalseg'
ds=TotalSegInContextDataset(root=r, classes=BALANCED_CLASSES[:5], image_size=(128,128,128),
    split='train', context_size=1, use_crop=True, p_synth=0.0, p_coords=1.0,
    max_subjects=40, class_balanced=True)
print('coords subjects:', len(ds._coords_subjects), 'aabb keys:', len(ds._coords_aabb))
assert len(ds._coords_subjects) > 0
"
```
Expected: prints a non-zero subject count; builds/loads `.coords_aabb_cache_*.pkl`.

- [ ] **Step 5: Commit**

```bash
git add src/totalseg_dataloader_incontext.py
git commit -m "feat(coords-synth): dataloader coords pool + AABB cache + config knobs"
```

---

### Task 4: `_get_coords_item` + routing (Phase A, hard labels)

Add the coords synth item builder and route to it from `__getitem__`. Reuse the existing native-crop machinery; emit binary `int64` labels so the rest of the pipeline is untouched.

**Files:**
- Modify: `src/totalseg_dataloader_incontext.py` (new method `_get_coords_item`; routing in `__getitem__` ~line 875)

**Interfaces:**
- Consumes: `build_coords_task`, `eval_field` (Tasks 1–2); `self._coords_subjects`, `self._coords_aabb`, `self._coords_cfg`; existing `self.image_size`, `self.use_crop`, `self.crop_jitter`, `self._load_native_ct_mmap`, `self._normalize_native`, `self.raw_ct`, `apply_synth_aug`.
- Produces: an item dict identical in shape/dtype to `_get_synth_item` (`image` (1,T,T,T) float32, `label` (T,T,T) int64, `context_in` (K,1,T,T,T), `context_out` (K,T,T,T) int64, `subject`, `label_name="coords_<family>"`, `spacing`).

- [ ] **Step 1: Add routing in `__getitem__`**

Immediately after the existing synth route (`if self._synth_subjects and random.random() < self.p_synth: return self._get_synth_item()`, ~line 875), add:
```python
        if self._coords_subjects and random.random() < self.p_coords:
            item = self._get_coords_item()
            if item is not None:
                return item
            # fall through to a normal real item if a task couldn't be assembled
```

- [ ] **Step 2: Implement `_get_coords_item`**

Add near `_get_synth_item` (~line 720). It: builds a task on the strided grid, then for each picked subject slices a native T³ crop centered on that subject's region centroid (rescaled by stride), evaluates the field on the crop's coords, binarizes, and augments K+1 copies.

```python
    def _get_coords_item(self):
        """Coords-driven synth: K+1 DIFFERENT subjects, one shared field f(coords)
        -> position-corresponding binary labels. Hard (int64) labels; Phase A."""
        from src.coords_synth import build_coords_task, eval_field
        c = self._coords_cfg
        ds = c["ds"]
        T = self.image_size[0]

        def get_coords(sid):
            co = np.asarray(np.load(self.root / sid / self.coords_fname,
                                    mmap_mode="r")[::ds, ::ds, ::ds], np.float32)
            lab = np.asarray(np.load(self.root / sid / "label.npy",
                                     mmap_mode="r")[::ds, ::ds, ::ds])
            shape = np.array(lab.shape)
            return co.reshape(-1, 3), lab.reshape(-1), shape

        task = build_coords_task(
            self._coords_subjects, get_coords, self._coords_aabb,
            self.context_size, self._cur_rng if self.eval_seed is not None else np.random.default_rng(),
            families=c["families"], scale_lo=c["scale_lo"], scale_hi=c["scale_hi"],
            min_mass=c["min_mass"], min_hi=c["min_hi"])
        if task is None:
            return None
        p = task["params"]

        items = []
        for sid, center_ds in task["picks"]:
            subj_dir = self.root / sid
            ct_mm = self._load_native_ct_mmap(subj_dir)                 # (D,H,W)
            co_mm = np.load(subj_dir / self.coords_fname, mmap_mode="r")  # (D,H,W,3)
            D, H, W = ct_mm.shape
            center = (center_ds * ds).astype(int)                       # strided -> native
            j = self.crop_jitter or 0
            starts = []
            for cc, s in zip(center, (D, H, W)):
                ideal = int(cc) - T // 2
                lo = max(0, ideal - j)
                hi = max(lo, min(max(0, s - T), ideal + j))
                starts.append(random.randint(lo, hi))
            d0, h0, w0 = starts
            crop_ct = ct_mm[d0:d0+T, h0:h0+T, w0:w0+T]
            crop_co = np.asarray(co_mm[d0:d0+T, h0:h0+T, w0:w0+T], np.float32)
            s = crop_ct.shape
            if self.raw_ct:
                crop_ct = self._normalize_native(sid, np.ascontiguousarray(crop_ct))

            img = np.zeros((T, T, T), np.float32)
            msk = np.zeros((T, T, T), np.uint8)
            img[:s[0], :s[1], :s[2]] = crop_ct.astype(np.float32)
            w = eval_field(p, crop_co.reshape(-1, 3)).reshape(crop_co.shape[:3])
            msk[:s[0], :s[1], :s[2]] = (w[:s[0], :s[1], :s[2]] >= 0.5).astype(np.uint8)

            image_t = torch.from_numpy(img).unsqueeze(0)
            mask_t = torch.from_numpy(msk).long()
            if self.aug_cfg is not None and self.aug_cfg.enabled:
                image_t, mask_t = apply_synth_aug(image_t, mask_t, self.aug_cfg.synth)
            items.append((image_t, mask_t))

        image_out, label_out = items[0]
        ctx = items[1:]
        item = {
            "image": image_out,
            "label": label_out,
            "context_in": torch.stack([it[0] for it in ctx]),
            "context_out": torch.stack([it[1] for it in ctx]),
            "subject": task["picks"][0][0],
            "label_name": f"coords_{task['family']}",
            "spacing": self._reported_spacing(task["picks"][0][0]),
        }
        if self.random_coloring:
            item["label_palette"] = self._sample_palette(
                label_out, [it[1] for it in ctx], self.num_labels_per_sample)
        return item
```

- [ ] **Step 3: Smoke-verify an item is well-formed**

Run:
```bash
.venv_thor_fresh/bin/python -c "
import sys, torch; sys.path.insert(0, '.')
from src.totalseg_dataloader_incontext import TotalSegInContextDataset
from data.totalseg_classes import BALANCED_CLASSES
r='/nfs/data/nii/data1/Analysis/camaret___in_context_segmentation/ANALYSIS_20251122/data/totalseg'
ds=TotalSegInContextDataset(root=r, classes=BALANCED_CLASSES[:5], image_size=(128,128,128),
    split='train', context_size=1, use_crop=True, p_synth=0.0, p_coords=1.0,
    max_subjects=60, class_balanced=True, crop_jitter=0)
it=ds._get_coords_item()
assert it is not None
print('image', tuple(it['image'].shape), it['image'].dtype)
print('label', tuple(it['label'].shape), it['label'].dtype, 'uniq', torch.unique(it['label']).tolist())
print('context_out', tuple(it['context_out'].shape), 'name', it['label_name'])
assert it['image'].shape==(1,128,128,128) and it['label'].dtype==torch.long
assert set(torch.unique(it['label']).tolist()) <= {0,1}
assert it['label'].sum()>0 and it['context_out'].sum()>0
print('OK')
"
```
Expected: prints shapes; label dtype long, values ⊆ {0,1}, non-empty target + context; `OK`.

- [ ] **Step 4: Commit**

```bash
git add src/totalseg_dataloader_incontext.py
git commit -m "feat(coords-synth): _get_coords_item + routing (hard labels, Phase A)"
```

---

### Task 5: Config wiring + training smoke run

Forward the coords knobs through `build_dataset` and the `get_incontext_loader` factory, expose them in the dataset config, and confirm a couple of training steps run.

**Files:**
- Modify: `experiments/3d/common.py` (`build_dataset`, ~lines 161–183)
- Modify: `src/totalseg_dataloader_incontext.py` (`get_incontext_loader`, ~lines 1264–1310)
- Modify: `configs/experiment/3d/dataset/totalseg.yaml`

**Interfaces:**
- Consumes: the `__init__` kwargs from Task 3.
- Produces: cfg keys `data.p_coords`, `data.coords_families`, `data.coords_scale_mm`, `data.coords_min_mass`, `data.coords_min_hi`, `data.coords_ds`, all reaching the dataset (train split only, like `p_synth`).

- [ ] **Step 1: Forward through `build_dataset`**

In `experiments/3d/common.py`, in the final `TotalSegInContextDataset(...)` return (~line 161), add after `p_synth=(d.p_synth if is_train else 0.0),`:
```python
        p_coords=(d.get("p_coords", 0.0) if is_train else 0.0),
        coords_families=tuple(d.get("coords_families", ("gaussian", "ellipsoid", "cyl_capped"))),
        coords_scale_mm=tuple(d.get("coords_scale_mm", (40.0, 140.0))),
        coords_min_mass=float(d.get("coords_min_mass", 40.0)),
        coords_min_hi=float(d.get("coords_min_hi", 0.15)),
        coords_ds=int(d.get("coords_ds", 4)),
```

- [ ] **Step 2: Forward through the factory**

In `get_incontext_loader` (`src/totalseg_dataloader_incontext.py`), add `p_coords: float = 0.0,` to the signature and `p_coords=p_coords,` to the `TotalSegInContextDataset(...)` call.

- [ ] **Step 3: Expose config keys**

In `configs/experiment/3d/dataset/totalseg.yaml`, under `data:`, add:
```yaml
  p_coords: 0.0                 # fraction of synth items drawn from the coords-function path (0 = off)
  coords_families: [gaussian, ellipsoid, cyl_capped]  # localized fields only
  coords_scale_mm: [40.0, 140.0]  # characteristic size band (floor ~40mm; below that correspondence degrades)
  coords_min_mass: 40.0         # min in-crop label mass for a subject to count
  coords_min_hi: 0.15           # min cross-subject anatomy consistency for a task
  coords_ds: 4                  # stride for center-finding / AABB
```

- [ ] **Step 4: Training smoke run (2 steps, coords on)**

Run:
```bash
.venv_thor_fresh/bin/python experiments/3d/train.py \
  data.p_synth=1 data.p_coords=1.0 data.max_train_subjects=60 \
  train.epochs=1 data.max_ds_len_train=8 train.wandb_project=null 2>&1 | tail -n 20
```
Expected: training starts, prints the `Coords synth: p_coords=1.0 | ... subjects` line, completes a few steps without shape/dtype errors and with a finite loss.

- [ ] **Step 5: Visual QA of dataset items (optional but recommended)**

Run `experiments/3d/plot_dataset_items.py` (builds via `build_dataset`) with `data.p_synth=1 data.p_coords=1.0` to render a few coords-synth items and eyeball target/context correspondence.

- [ ] **Step 6: Log + commit**

Append a `docs/logs.md` entry summarizing the coords synth mode (path, config keys, Phase A hard labels). Then:
```bash
git add experiments/3d/common.py src/totalseg_dataloader_incontext.py \
  configs/experiment/3d/dataset/totalseg.yaml docs/logs.md
git commit -m "feat(coords-synth): config wiring + training smoke (Phase A)"
```

---

### Task 6: Phase B — soft (float) labels

Add a soft-label mode: emit the continuous field value as a float target. Gate behind a flag; verify the loss accepts soft targets.

**Files:**
- Modify: `src/totalseg_dataloader_incontext.py` (`_get_coords_item`, `__init__`)
- Modify: `experiments/3d/common.py`, `configs/experiment/3d/dataset/totalseg.yaml`
- Verify (read, maybe modify): `experiments/3d/train.py` loss/metric sites (`context_out` at ~309–315, ~405)

**Interfaces:**
- Consumes: everything from Tasks 1–5.
- Produces: `__init__` kwarg + cfg key `coords_soft: bool = False`; when true, `label`/`context_out` are `float32` in [0,1].

- [ ] **Step 1: Check whether the loss binarizes the target**

Run:
```bash
grep -n "bce_dice\|def .*loss\|\.long()\|> *0\.5\|argmax\|context_out\|label" experiments/3d/train.py | sed -n '1,40p'
```
Inspect the loss/metric functions the grep points to. Confirm the target is used as a probability (BCE/soft-Dice accept floats) and note any `.long()`/threshold/`argmax` on `label` or `context_out` that would break soft targets. Record findings in the commit message.

- [ ] **Step 2: Add the `coords_soft` knob**

In `__init__` add `coords_soft: bool = False,` and store `self.coords_soft = coords_soft`. In `_get_coords_item`, replace the mask build with:
```python
            if self.coords_soft:
                msk = np.zeros((T, T, T), np.float32)
                msk[:s[0], :s[1], :s[2]] = w[:s[0], :s[1], :s[2]].astype(np.float32)
                mask_t = torch.from_numpy(msk)              # float32 [0,1]
            else:
                msk = np.zeros((T, T, T), np.uint8)
                msk[:s[0], :s[1], :s[2]] = (w[:s[0], :s[1], :s[2]] >= 0.5).astype(np.uint8)
                mask_t = torch.from_numpy(msk).long()
```
Guard augmentation: `apply_synth_aug` on a float mask must not nearest-cast it to int — if it does, skip mask aug in soft mode (apply only intensity aug to the image). Set `item["label_name"] = f"coords_{task['family']}" + ("_soft" if self.coords_soft else "")`. Disable `random_coloring` in soft mode (palette assumes integer labels): only attach `label_palette` when `not self.coords_soft`.

- [ ] **Step 3: Forward the flag**

Add `coords_soft=(d.get("coords_soft", False) if is_train else False),` in `build_dataset`, and `coords_soft: false` in the dataset yaml.

- [ ] **Step 4: Smoke-verify soft item + loss**

Run:
```bash
.venv_thor_fresh/bin/python -c "
import sys, torch; sys.path.insert(0, '.')
from src.totalseg_dataloader_incontext import TotalSegInContextDataset
from data.totalseg_classes import BALANCED_CLASSES
r='/nfs/data/nii/data1/Analysis/camaret___in_context_segmentation/ANALYSIS_20251122/data/totalseg'
ds=TotalSegInContextDataset(root=r, classes=BALANCED_CLASSES[:5], image_size=(128,128,128),
    split='train', context_size=1, use_crop=True, p_synth=0.0, p_coords=1.0,
    coords_soft=True, max_subjects=60, class_balanced=True, crop_jitter=0)
it=ds._get_coords_item()
assert it['label'].dtype==torch.float32
assert float(it['label'].max())<=1.0 and float(it['label'].min())>=0.0 and float(it['label'].max())>0.5
print('soft label OK', float(it['label'].min()), float(it['label'].max()))
"
```
Expected: `soft label OK` with values in [0,1] and a peak > 0.5.

- [ ] **Step 5: Training smoke run (soft)**

Run:
```bash
.venv_thor_fresh/bin/python experiments/3d/train.py \
  data.p_synth=1 data.p_coords=1.0 data.coords_soft=true data.max_train_subjects=60 \
  train.epochs=1 data.max_ds_len_train=8 train.wandb_project=null 2>&1 | tail -n 20
```
Expected: runs with a finite loss. If the grep in Step 1 found a hard threshold on the target, fix that site to accept soft targets before this passes; note the fix in the commit.

- [ ] **Step 6: Log + commit**

```bash
git add -A
git commit -m "feat(coords-synth): Phase B soft float labels behind coords_soft flag"
```

---

## Self-Review

**Spec coverage:**
- Field vocabulary (localized ellipsoid/cyl_capped/gaussian) → Task 1. ✓
- Correspondence by construction / eval per subject → Tasks 1–2, 4. ✓
- Scale floor 40mm → config `coords_scale_mm` default (40,140) Task 5. ✓
- FOV-aware grouping (coords AABB), mass guard, HI backstop → Tasks 2–3. ✓
- Multi-subject assembly + crop-centering (matches use_crop) → Task 4. ✓
- Integration as `p_coords` sharing synth budget → Tasks 4–5. ✓
- Phase A hard (integer pipeline unchanged) then Phase B soft (float plumbing) → Tasks 4 vs 6. ✓
- Complementary to supervoxel path (different subjects vs aug copies) → realized by routing coexisting with `_get_synth_item`. ✓
- Performance (strided center-finding, AABB cache) → Tasks 2–4. ✓
- Validation artifacts (montage, smoke run) → Task 1 Step 6, Task 4 Step 3, Task 5 Step 4. ✓

**Placeholder scan:** No TBD/TODO; every code step has concrete code; every run step has an exact command + expected result.

**Type consistency:** `sample_field`/`eval_field`/`coords_aabb`/`soft_hist`/`pairwise_hi`/`build_coords_task` signatures match across Tasks 1–4; `build_coords_task` returns `{"params","family","picks","hi"}` consumed consistently in Task 4; `get_coords(sid)->(co_flat,lab_flat,shape)` matches its use in the builder and the dataloader closure.

## Notes / risks

- **Throughput:** `_get_coords_item` loads `coords.npy` (strided) for candidate subjects each item; the AABB pre-filter narrows candidates and center-finding uses stride `coords_ds=4`. If workers bottleneck, add a pre-resized `coords_{T}.npy` (non-crop fast path) or a small on-disk low-res coords per subject — deferred until measured (spec Performance section).
- **Soft aug:** `apply_synth_aug` may nearest-cast masks; Task 6 Step 2 guards this. Confirm before enabling soft in production.
