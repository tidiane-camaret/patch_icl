# anchor_synth3d Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an `anchor_synth3d` in-context dataset that draws synthetic blobs at a consistent anatomical position relative to a shared anchor organ, on real TotalSegmentator CT backgrounds.

**Architecture:** A new package `src/datasets/anchor_synth/` with pure, analytic shape generators (`shapes.py`), placement/compositing helpers (`draw.py`), and a dataset (`dataset3d.py`) that subclasses `TotalSegInContextDataset` to reuse its scan cache, class-balanced sampling, and pre-resized fast-path loading. Wiring is added to the 3D harness (`common.py`, `eval.py`) plus a Hydra config.

**Tech Stack:** Python, NumPy, PyTorch, scipy (opt-in only), pytest, Hydra/OmegaConf.

## Global Constraints

- Ride the pre-resized fast path: subjects must have `ct_{D}x{H}x{W}.npy` and `label_{D}x{H}x{W}.npy`; the scan cache reads native `label.npy`.
- Default shape path is **analytic** — no scipy `rotate`/`zoom`/`gaussian_filter`. scipy is used only in the opt-in `roughen()` (`boundary_complexity > 0`).
- The anchor organ is a landmark only: it is **never** written into the label. Only drawn objects are labeled (IDs `1..n_objects`).
- The per-item task spec (per-object offset, geometry, contrast) is drawn **once** and shared across the K+1 scenes; only small scale/rotation jitter differs per scene.
- Deterministic (val/test) seeding: `np.random.SeedSequence([eval_seed_namespace, _ALL_CLASSES_IDX[anchor_cls], sample_index])`.
- v1 implements `object_source="blob"` only; `"organ"` raises `NotImplementedError` (follow-up).
- Tests live next to code as `src/datasets/anchor_synth/test_*.py`; each test file starts with `import sys; sys.path.insert(0, ".")`. Run from repo root `/home/dpxuser/dev/patch_icl`.

---

### Task 1: Shape generators (`shapes.py`)

**Files:**
- Create: `src/datasets/anchor_synth/__init__.py`
- Create: `src/datasets/anchor_synth/shapes.py`
- Test: `src/datasets/anchor_synth/test_shapes.py`

**Interfaces:**
- Produces:
  - `sample_object_spec(rng, shape="blob", eccentricity=3.0, n_harmonics=4, harmonic_amp=0.30, edge_blur=0.08) -> dict` with keys `axes (np.ndarray (3,))`, `R0 (np.ndarray (3,3))`, `terms (list[(u (3,), a float)])`, `edge_blur (float)`.
  - `render_object(size:int, spec:dict, R_extra=None) -> np.ndarray (size,size,size) float32 in [0,1]`.
  - `small_rotation(rng, max_deg:float) -> np.ndarray (3,3)`.
  - `roughen(alpha:np.ndarray, c:float, rng) -> np.ndarray float32` (scipy; opt-in).

- [ ] **Step 1: Write the failing test**

```python
# src/datasets/anchor_synth/test_shapes.py
import sys; sys.path.insert(0, ".")
import numpy as np

from src.datasets.anchor_synth.shapes import (
    sample_object_spec, render_object, small_rotation, roughen,
)


def test_render_object_shape_and_range():
    rng = np.random.default_rng(0)
    spec = sample_object_spec(rng, shape="blob")
    a = render_object(24, spec)
    assert a.shape == (24, 24, 24)
    assert a.dtype == np.float32
    assert a.min() >= 0.0 and a.max() <= 1.0
    assert (a > 0.5).sum() > 0                      # non-empty object


def test_render_object_is_irregular():
    # harmonics + random orientation => not mirror-symmetric on axis 0
    rng = np.random.default_rng(1)
    spec = sample_object_spec(rng, shape="blob", harmonic_amp=0.30)
    a = render_object(32, spec)
    assert not np.allclose(a, np.flip(a, axis=0))


def test_render_object_deterministic_for_spec():
    spec = sample_object_spec(np.random.default_rng(7))
    a = render_object(20, spec)
    b = render_object(20, spec)
    assert np.array_equal(a, b)                      # spec fully determines shape


def test_elongated_is_anisotropic():
    rng = np.random.default_rng(2)
    spec = sample_object_spec(rng, shape="elongated", eccentricity=4.0)
    a = render_object(40, spec) > 0.5
    sides = []
    for ax in range(3):
        proj = a.any(axis=tuple(i for i in range(3) if i != ax))
        idx = np.nonzero(proj)[0]
        sides.append(idx[-1] - idx[0] + 1)
    assert max(sides) >= 1.6 * min(sides)            # clearly elongated


def test_small_rotation_is_near_identity():
    R = small_rotation(np.random.default_rng(3), max_deg=10.0)
    assert R.shape == (3, 3)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-6)   # orthonormal
    assert np.trace(R) > 2.5                             # small angle


def test_roughen_changes_boundary():
    spec = sample_object_spec(np.random.default_rng(4))
    a = render_object(28, spec)
    r = roughen(a, c=0.6, rng=np.random.default_rng(5))
    assert r.shape == a.shape
    assert not np.array_equal(r > 0.5, a > 0.5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest src/datasets/anchor_synth/test_shapes.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.datasets.anchor_synth'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/datasets/anchor_synth/__init__.py
"""anchor_synth3d: synthetic objects at anchor-relative positions on real CT."""
```

```python
# src/datasets/anchor_synth/shapes.py
"""Analytic 3D object generators for anchor_synth3d.

Follows controlSynth/shapes/blob.py extended to 3D: a base ellipsoid whose
radius is modulated by low-frequency angular bumps gives organic, irregular
(non-spherical) shapes. Fully analytic — no scipy in the default path. The
object geometry is fixed by `sample_object_spec` so it is reproducible and can
be shared across the K+1 scenes; `render_object` rasterizes it (optionally with a
small extra rotation for per-scene jitter). `roughen` (scipy) is opt-in.
"""

import numpy as np


def _unit_grid(size):
    """Centered coordinate grids in [-1, 1] on a size^3 cube (z, y, x)."""
    half = max(1.0, (size - 1) / 2.0)
    lin = (np.arange(size) - (size - 1) / 2.0) / half
    z, y, x = np.meshgrid(lin, lin, lin, indexing="ij")
    return z, y, x


def _rand_rotation(rng):
    """A random rotation matrix via QR of a Gaussian matrix (sign-fixed)."""
    q, r = np.linalg.qr(rng.standard_normal((3, 3)))
    q = q * np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def small_rotation(rng, max_deg):
    """A rotation of a random axis by a small angle in [-max_deg, max_deg]."""
    ax = rng.standard_normal(3)
    ax = ax / (np.linalg.norm(ax) + 1e-9)
    ang = np.radians(rng.uniform(-max_deg, max_deg))
    K = np.array([[0, -ax[2], ax[1]],
                  [ax[2], 0, -ax[0]],
                  [-ax[1], ax[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1.0 - np.cos(ang)) * (K @ K)


def sample_object_spec(rng, shape="blob", eccentricity=3.0, n_harmonics=4,
                       harmonic_amp=0.30, edge_blur=0.08):
    """Draw a reproducible object geometry (axes, orientation, angular bumps)."""
    if shape == "mix":
        shape = str(rng.choice(("blob", "elongated")))
    axes = np.array([rng.uniform(0.85, 1.0) for _ in range(3)], dtype=np.float64)
    if shape == "elongated":
        axes[:] = 1.0 / np.sqrt(float(eccentricity))
        axes[int(rng.integers(3))] = 1.0
    R0 = _rand_rotation(rng)
    terms = []
    for _ in range(int(n_harmonics)):
        u = rng.standard_normal(3)
        u = u / (np.linalg.norm(u) + 1e-9)
        terms.append((u, float(rng.uniform(-harmonic_amp, harmonic_amp))))
    return {"axes": axes, "R0": R0, "terms": terms, "edge_blur": float(edge_blur)}


def render_object(size, spec, R_extra=None):
    """Rasterize a spec to a soft alpha tile (size^3) in [0, 1]. R_extra applies a
    small per-scene rotation on top of the spec's base orientation."""
    z, y, x = _unit_grid(size)
    pts = np.stack([z.ravel(), y.ravel(), x.ravel()], 0)          # (3, N)
    R = spec["R0"] if R_extra is None else (R_extra @ spec["R0"])
    pr = (R @ pts) / spec["axes"][:, None]                        # ellipsoid frame
    rr = np.sqrt((pr ** 2).sum(0))                               # radius (N,)
    dirs = pr / (rr + 1e-6)                                      # unit dirs (3, N)
    r_mod = np.ones_like(rr)
    for u, a in spec["terms"]:
        r_mod = r_mod + a * (dirs.T @ u) ** 2                    # low-freq bumps
    r_mod = np.clip(r_mod, 0.5, 1.7)
    base_r = 0.72
    blur = max(1e-3, float(spec["edge_blur"]) * base_r)
    alpha = np.clip((base_r * r_mod - rr) / blur + 0.5, 0.0, 1.0)
    return alpha.reshape(z.shape).astype(np.float32)


def roughen(alpha, c, rng):
    """Perturb the alpha boundary via SDF + smoothed noise (opt-in, uses scipy).
    Mirrors controlSynth/shapes/boundary.py. Returns a hard {0,1} float32 mask."""
    from scipy.ndimage import distance_transform_edt, gaussian_filter

    c = float(c)
    m = alpha > 0.5
    if c <= 0.0 or not m.any():
        return alpha
    sdf = distance_transform_edt(m) - distance_transform_edt(~m)
    char = float(np.cbrt(m.sum()))
    noise = gaussian_filter(rng.standard_normal(m.shape), sigma=max(1.0, char * 0.3))
    std = noise.std()
    if std > 1e-8:
        noise = noise / std
    rough = (sdf + c * char * 0.5 * noise) > 0.0
    return rough.astype(np.float32)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest src/datasets/anchor_synth/test_shapes.py -q`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add src/datasets/anchor_synth/__init__.py src/datasets/anchor_synth/shapes.py src/datasets/anchor_synth/test_shapes.py
git commit -m "feat(anchor-synth3d): analytic 3D object shape generators"
```

---

### Task 2: Placement & compositing helpers (`draw.py`)

**Files:**
- Create: `src/datasets/anchor_synth/draw.py`
- Test: `src/datasets/anchor_synth/test_draw.py`

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces:
  - `anchor_stats(mask:np.ndarray) -> (centroid (3,) float64, extent (3,) float64, (lo (3,) int, hi (3,) int)) | None`.
  - `offset_to_center(centroid, extent, offset_norm, tile_size:int, vol_shape) -> np.ndarray (3,) float64`.
  - `place_object(image:np.ndarray, alpha:np.ndarray, center, contrast_delta:float, label=None, label_id=1) -> np.ndarray bool` (footprint, same shape as image; mutates `image` and `label` in place).

- [ ] **Step 1: Write the failing test**

```python
# src/datasets/anchor_synth/test_draw.py
import sys; sys.path.insert(0, ".")
import numpy as np

from src.datasets.anchor_synth.draw import (
    anchor_stats, offset_to_center, place_object,
)


def test_anchor_stats_bbox_and_centroid():
    m = np.zeros((20, 20, 20), dtype=np.uint8)
    m[4:10, 6:14, 8:12] = 1
    centroid, extent, (lo, hi) = anchor_stats(m)
    assert list(lo) == [4, 6, 8]
    assert list(hi) == [9, 13, 11]
    assert list(extent) == [6, 8, 4]
    assert np.allclose(centroid, [6.5, 9.5, 9.5])


def test_anchor_stats_empty_returns_none():
    assert anchor_stats(np.zeros((8, 8, 8), dtype=np.uint8)) is None


def test_offset_to_center_uses_extent_and_clamps():
    centroid = np.array([10.0, 10.0, 10.0])
    extent = np.array([8.0, 8.0, 8.0])
    c = offset_to_center(centroid, extent, [0.5, 0.0, -0.5], tile_size=6,
                         vol_shape=(20, 20, 20))
    assert np.allclose(c, [14.0, 10.0, 6.0])
    # push far out-of-bounds -> clamped so a size-6 tile stays fully inside
    c2 = offset_to_center(centroid, extent, [10.0, -10.0, 0.0], tile_size=6,
                          vol_shape=(20, 20, 20))
    assert np.allclose(c2, [17.0, 3.0, 10.0])       # [20-3, 3, 10]


def test_place_object_blends_and_writes_label():
    image = np.full((16, 16, 16), 0.4, dtype=np.float32)
    label = np.zeros((16, 16, 16), dtype=np.int64)
    alpha = np.zeros((6, 6, 6), dtype=np.float32)
    alpha[1:5, 1:5, 1:5] = 1.0                       # solid core
    foot = place_object(image, alpha, center=[8, 8, 8], contrast_delta=0.2,
                        label=label, label_id=1)
    assert foot.shape == image.shape
    assert foot.sum() == (alpha > 0.5).sum()
    # interior intensity == local bg (0.4) + delta (0.2)
    assert np.allclose(image[foot], 0.6, atol=1e-5)
    assert (label == 1).sum() == foot.sum()
    assert np.array_equal(label > 0, foot)


def test_place_object_clips_at_border():
    image = np.zeros((10, 10, 10), dtype=np.float32)
    alpha = np.ones((6, 6, 6), dtype=np.float32)
    foot = place_object(image, alpha, center=[0, 0, 0], contrast_delta=1.0)
    assert foot[:3, :3, :3].all()                    # in-bounds octant written
    assert foot.sum() == 27
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest src/datasets/anchor_synth/test_draw.py -q`
Expected: FAIL — `ModuleNotFoundError` / `cannot import name`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/datasets/anchor_synth/draw.py
"""Placement and compositing helpers for anchor_synth3d.

Pure numpy (no dataset/IO deps), so unit-testable with trivial arrays. The
object is alpha-composited toward the local background mean plus a small contrast
delta with soft edges, so it blends into the CT and is only findable via the
anchor-relative position the K contexts demonstrate.
"""

import numpy as np


def anchor_stats(mask):
    """(centroid, extent, (lo, hi)) of a binary mask via axis projections.

    centroid = bbox centre; extent = per-axis bbox side length. None if empty.
    Cheap (no full nonzero / scipy) — projects the mask onto each axis.
    """
    m = mask > 0
    if not m.any():
        return None
    lo = np.empty(3, dtype=np.int64)
    hi = np.empty(3, dtype=np.int64)
    for ax in range(3):
        proj = m.any(axis=tuple(a for a in range(3) if a != ax))
        idx = np.nonzero(proj)[0]
        lo[ax], hi[ax] = int(idx[0]), int(idx[-1])
    centroid = (lo + hi) / 2.0
    extent = (hi - lo + 1).astype(np.float64)
    return centroid, extent, (lo, hi)


def offset_to_center(centroid, extent, offset_norm, tile_size, vol_shape):
    """Voxel centre for an object at centroid + offset_norm * extent, clamped so a
    `tile_size` cube stays fully inside `vol_shape`."""
    center = (np.asarray(centroid, dtype=np.float64)
              + np.asarray(offset_norm, dtype=np.float64)
              * np.asarray(extent, dtype=np.float64))
    half = tile_size / 2.0
    return np.clip(center, half, np.asarray(vol_shape, dtype=np.float64) - half)


def _slices_3d(t, cz, cy, cx, D, H, W):
    """(canvas_slices, tile_slices) for a t^3 tile centred at (cz,cy,cx) clipped to
    a D×H×W volume; None if fully out of bounds."""
    oz, oy, ox = cz - t // 2, cy - t // 2, cx - t // 2
    dz0, dy0, dx0 = max(0, oz), max(0, oy), max(0, ox)
    dz1, dy1, dx1 = min(D, oz + t), min(H, oy + t), min(W, ox + t)
    if dz0 >= dz1 or dy0 >= dy1 or dx0 >= dx1:
        return None
    return ((slice(dz0, dz1), slice(dy0, dy1), slice(dx0, dx1)),
            (slice(dz0 - oz, dz1 - oz), slice(dy0 - oy, dy1 - oy),
             slice(dx0 - ox, dx1 - ox)))


def place_object(image, alpha, center, contrast_delta, label=None, label_id=1):
    """Alpha-composite `alpha` into `image` at voxel `center`, blending toward the
    local background mean + contrast_delta. Writes alpha>0.5 into `label` with
    `label_id` when given. Mutates `image`/`label`; returns the bool footprint."""
    t = alpha.shape[0]
    c = np.round(np.asarray(center)).astype(int)
    footprint = np.zeros(image.shape, dtype=bool)
    sl = _slices_3d(t, int(c[0]), int(c[1]), int(c[2]), *image.shape)
    if sl is None:
        return footprint
    cs, ts = sl
    a = alpha[ts]
    core = a > 0.5
    region = image[cs]
    bg = float(region[core].mean()) if core.any() else float(region.mean())
    target_val = bg + float(contrast_delta)
    region[:] = region * (1.0 - a) + target_val * a
    footprint[cs] = core
    if label is not None:
        lreg = label[cs]
        lreg[core] = label_id
    return footprint
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest src/datasets/anchor_synth/test_draw.py -q`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add src/datasets/anchor_synth/draw.py src/datasets/anchor_synth/test_draw.py
git commit -m "feat(anchor-synth3d): anchor stats + object placement/compositing"
```

---

### Task 3: Dataset (`dataset3d.py`)

**Files:**
- Create: `src/datasets/anchor_synth/dataset3d.py`
- Test: `src/datasets/anchor_synth/test_dataset3d.py`

**Interfaces:**
- Consumes: `shapes.sample_object_spec`, `shapes.render_object`, `shapes.small_rotation`, `shapes.roughen`; `draw.anchor_stats`, `draw.offset_to_center`, `draw.place_object`; base `TotalSegInContextDataset` (`self._load`, `self.active_classes`, `self.label_to_subjects`, `self._get_spacing`); `src.totalseg_dataset._ALL_CLASSES_IDX`.
- Produces: `AnchorSynth3DICLDataset(root, classes, image_size, split, context_size, object_source="blob", shape="blob", n_objects=1, offset_range=0.6, scale_frac=0.4, scale_jitter=0.15, rotate_jitter=12.0, contrast_delta=0.15, edge_blur=0.08, boundary_complexity=0.0, harmonic_amp=0.30, eccentricity=3.0, n_harmonics=4, deterministic=None, eval_seed_namespace=0, eval_subjects_per_task=4, epoch_length=10000, max_subjects=None)` emitting the standard in-context contract dict.

- [ ] **Step 1: Write the failing test**

```python
# src/datasets/anchor_synth/test_dataset3d.py
import sys; sys.path.insert(0, ".")
import numpy as np
import torch

from src.datasets.anchor_synth.dataset3d import AnchorSynth3DICLDataset
from src.totalseg_dataset import _ALL_CLASSES_IDX

SIZE = 16
ANCHOR = "aorta"


def _make_root(tmp_path, n=5):
    """Fake TotalSeg root: each subject has native label.npy + pre-resized
    ct_/label_ npy (native == resized at SIZE), all containing the anchor organ."""
    idx = _ALL_CLASSES_IDX[ANCHOR]
    rows = ["image_id;split"]
    for i in range(n):
        subj = f"s{i:04d}"
        d = tmp_path / subj
        d.mkdir()
        label = np.zeros((SIZE, SIZE, SIZE), dtype=np.uint8)
        # anchor block, position jittered per subject so it is a real landmark
        z0 = 3 + (i % 3)
        label[z0:z0 + 6, 5:11, 6:10] = idx
        ct = (0.3 + 0.01 * i) * np.ones((SIZE, SIZE, SIZE), dtype=np.float16)
        np.save(d / "label.npy", label)
        np.save(d / f"label_{SIZE}x{SIZE}x{SIZE}.npy", label)
        np.save(d / f"ct_{SIZE}x{SIZE}x{SIZE}.npy", ct)
        rows.append(f"{subj};val")
    (tmp_path / "meta.csv").write_text("\n".join(rows) + "\n")
    return tmp_path


def _ds(root, **kw):
    return AnchorSynth3DICLDataset(
        root=root, classes=[ANCHOR], image_size=(SIZE, SIZE, SIZE),
        split="val", context_size=2, eval_subjects_per_task=2,
        offset_range=0.2, scale_frac=0.4, contrast_delta=0.3, **kw)


def test_contract_shapes_and_object_drawn(tmp_path):
    ds = _ds(_make_root(tmp_path))
    assert len(ds) == 2                              # 1 anchor class * 2 subjects/task
    item = ds[0]
    assert item["image"].shape == (1, SIZE, SIZE, SIZE)
    assert item["label"].shape == (SIZE, SIZE, SIZE)
    assert item["label"].dtype == torch.int64
    assert item["context_in"].shape == (2, 1, SIZE, SIZE, SIZE)
    assert item["context_out"].shape == (2, SIZE, SIZE, SIZE)
    assert item["label"].sum() > 0                   # a blob was drawn
    assert item["label_name"] == ANCHOR


def test_anchor_not_emitted_as_label(tmp_path):
    ds = _ds(_make_root(tmp_path))
    item = ds[0]
    idx = _ALL_CLASSES_IDX[ANCHOR]
    anchor_mask = (np.load(tmp_path / f"{item['subject']}/label_{SIZE}x{SIZE}x{SIZE}.npy") == idx)
    # the label is the drawn object, not the anchor organ
    assert not np.array_equal(item["label"].numpy() > 0, anchor_mask)


def test_deterministic_across_instances(tmp_path):
    root = _make_root(tmp_path)
    a = _ds(root)[0]
    b = _ds(root)[0]
    assert torch.equal(a["label"], b["label"])
    assert torch.equal(a["image"], b["image"])


def test_organ_source_not_implemented(tmp_path):
    import pytest
    with pytest.raises(NotImplementedError):
        _ds(_make_root(tmp_path), object_source="organ")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest src/datasets/anchor_synth/test_dataset3d.py -q`
Expected: FAIL — `ModuleNotFoundError: ... dataset3d`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/datasets/anchor_synth/dataset3d.py
"""AnchorSynth3DICLDataset: draws synthetic objects at a consistent position
relative to a shared anchor organ, on real TotalSegmentator CT backgrounds.

Subclasses TotalSegInContextDataset to reuse its scan cache, class-balanced
anchor/subject sampling, and pre-resized fast-path loading. The anchor organ is a
landmark only (never labeled); the label is the drawn object(s). The per-item task
spec (per-object offset/geometry/contrast) is drawn once and shared across the K+1
scenes; only small scale/rotation jitter varies per scene. See
docs/superpowers/specs/2026-07-22-anchor-synth3d-design.md.
"""

import numpy as np
import torch

from src.totalseg_dataloader_incontext import TotalSegInContextDataset
from src.totalseg_dataset import _ALL_CLASSES_IDX
from .shapes import sample_object_spec, render_object, small_rotation, roughen
from .draw import anchor_stats, offset_to_center, place_object


class AnchorSynth3DICLDataset(TotalSegInContextDataset):
    def __init__(self, root, classes, image_size=(128, 128, 128), split="train",
                 context_size=1, object_source="blob", shape="blob", n_objects=1,
                 offset_range=0.6, scale_frac=0.4, scale_jitter=0.15,
                 rotate_jitter=12.0, contrast_delta=0.15, edge_blur=0.08,
                 boundary_complexity=0.0, harmonic_amp=0.30, eccentricity=3.0,
                 n_harmonics=4, deterministic=None, eval_seed_namespace=0,
                 eval_subjects_per_task=4, epoch_length=10000, max_subjects=None):
        if object_source != "blob":
            raise NotImplementedError(
                f"object_source={object_source!r} not implemented in v1 (blob only)")
        super().__init__(root=root, classes=classes, image_size=image_size,
                         split=split, context_size=context_size,
                         max_subjects=max_subjects, class_balanced=True)
        self.object_source = object_source
        self.shape = shape
        self.n_objects = int(n_objects)
        self.offset_range = float(offset_range)
        self.scale_frac = float(scale_frac)
        self.scale_jitter = float(scale_jitter)
        self.rotate_jitter = float(rotate_jitter)
        self.contrast_delta = float(contrast_delta)
        self.edge_blur = float(edge_blur)
        self.boundary_complexity = float(boundary_complexity)
        self.harmonic_amp = float(harmonic_amp)
        self.eccentricity = float(eccentricity)
        self.n_harmonics = int(n_harmonics)
        self.eval_seed_namespace = int(eval_seed_namespace)
        self.eval_subjects_per_task = int(eval_subjects_per_task)
        self.epoch_length = int(epoch_length)
        self.anchor_deterministic = (split != "train") if deterministic is None else deterministic

        if self.anchor_deterministic:
            self._eval_index = [(cls, s) for cls in self.active_classes
                                for s in range(self.eval_subjects_per_task)]
            self._n = len(self._eval_index)
        else:
            self._eval_index = None
            self._n = self.epoch_length

    def __len__(self):
        return self._n

    def _draw_specs(self, rng):
        """Per-item task spec (shared across the K+1 scenes)."""
        specs = []
        for _ in range(self.n_objects):
            specs.append({
                "geom": sample_object_spec(
                    rng, shape=self.shape, eccentricity=self.eccentricity,
                    n_harmonics=self.n_harmonics, harmonic_amp=self.harmonic_amp,
                    edge_blur=self.edge_blur),
                "offset": rng.uniform(-self.offset_range, self.offset_range, size=3),
                "contrast": float(rng.uniform(-1.0, 1.0) * self.contrast_delta),
            })
        return specs

    def _render_subject(self, subj, anchor_cls, specs, scene_rng):
        image_t, anchor_t = self._load(subj, anchor_cls)          # fast path
        img = image_t.squeeze(0).numpy().astype(np.float32).copy()  # (D,H,W)
        label = np.zeros(img.shape, dtype=np.int64)
        stats = anchor_stats(anchor_t.numpy())
        if stats is not None:
            centroid, extent, _ = stats
            base = self.scale_frac * float(np.mean(extent))
            for lid, spec in enumerate(specs, 1):
                jit = 1.0 + scene_rng.uniform(-self.scale_jitter, self.scale_jitter)
                size = max(3, int(round(base * jit)))
                alpha = render_object(size, spec["geom"],
                                      R_extra=small_rotation(scene_rng, self.rotate_jitter))
                if self.boundary_complexity > 0.0:
                    alpha = roughen(alpha, self.boundary_complexity, scene_rng)
                center = offset_to_center(centroid, extent, spec["offset"],
                                          size, img.shape)
                place_object(img, alpha, center, spec["contrast"], label, lid)
        return torch.from_numpy(img).unsqueeze(0), torch.from_numpy(label)

    def __getitem__(self, idx):
        if self.anchor_deterministic:
            anchor_cls, sample_index = self._eval_index[idx]
            item_rng = np.random.default_rng(np.random.SeedSequence(
                [self.eval_seed_namespace, _ALL_CLASSES_IDX[anchor_cls], sample_index]))
        else:
            item_rng = np.random.default_rng()
            anchor_cls = self.active_classes[item_rng.integers(len(self.active_classes))]

        subs = self.label_to_subjects[anchor_cls]
        order = item_rng.permutation(len(subs))
        chosen = [subs[i] for i in order[:self.context_size + 1]]
        while len(chosen) < self.context_size + 1:
            chosen.append(chosen[int(item_rng.integers(len(chosen)))])

        specs = self._draw_specs(item_rng)
        scene_seeds = item_rng.integers(0, 2 ** 32, size=len(chosen))
        scenes = [self._render_subject(subj, anchor_cls, specs,
                                       np.random.default_rng(int(s)))
                  for subj, s in zip(chosen, scene_seeds)]

        image_t, label_t = scenes[0]
        ctx = scenes[1:]
        return {
            "image":       image_t,
            "label":       label_t,
            "context_in":  torch.stack([c[0] for c in ctx]),
            "context_out": torch.stack([c[1] for c in ctx]),
            "subject":     chosen[0],
            "label_name":  anchor_cls,
            "spacing":     self._get_spacing(chosen[0]),
            "meta": {"anchor": anchor_cls,
                     "n_objects": self.n_objects,
                     "offsets": [spec["offset"].tolist() for spec in specs],
                     "contrasts": [spec["contrast"] for spec in specs]},
        }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest src/datasets/anchor_synth/test_dataset3d.py -q`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/datasets/anchor_synth/dataset3d.py src/datasets/anchor_synth/test_dataset3d.py
git commit -m "feat(anchor-synth3d): AnchorSynth3DICLDataset on real CT backgrounds"
```

---

### Task 4: Config + harness wiring

**Files:**
- Create: `configs/experiment/3d/dataset/anchor_synth3d.yaml`
- Modify: `experiments/3d/common.py` (`build_dataset`, `make_eval_loader`)
- Modify: `experiments/3d/eval.py` (class-listing branch, ~lines 67-81)
- Test: `src/datasets/anchor_synth/test_wiring.py`

**Interfaces:**
- Consumes: `AnchorSynth3DICLDataset` (Task 3); `data.totalseg_classes.resolve_classes`.
- Produces: `build_dataset(cfg, split)` returns an `AnchorSynth3DICLDataset` when `cfg.data.source == "anchor_synth3d"`.

- [ ] **Step 1: Write the config**

```yaml
# configs/experiment/3d/dataset/anchor_synth3d.yaml
# @package _global_
# anchor_synth3d: synthetic objects drawn at a consistent position relative to a
# shared anchor organ, on real TotalSegmentator CT backgrounds. Composed as
# `dataset=anchor_synth3d`. Rides the pre-resized fast path (ct_/label_ npy).
data:
  source: anchor_synth3d
  image_size: [128, 128, 128]
  context_size: 1

anchor_synth:
  object_source: blob        # blob (analytic) | organ (not implemented in v1)
  shape: mix                 # blob | elongated | mix
  n_objects: 1               # objects (labels) per task; all share the one anchor
  anchor_classes: []         # [] = all; or a list / "benchmark" / "not_benchmark"
  offset_range: 0.6          # per-axis offset ~ U[-r, r], in units of anchor extent
  scale_frac: 0.4            # object size = scale_frac * mean(anchor extent)
  scale_jitter: 0.15         # per-scene multiplicative size jitter
  rotate_jitter: 12.0        # per-scene small rotation (deg)
  contrast_delta: 0.15       # |object - local background| intensity (blends in)
  edge_blur: 0.08            # soft-edge width (fraction of object radius)
  boundary_complexity: 0.0   # >0 = scipy SDF-noise roughening (heavier path)
  eval_subjects_per_task: 4
  eval_seed_namespace: 0
  epoch_length: 10000
```

- [ ] **Step 2: Write the failing test**

```python
# src/datasets/anchor_synth/test_wiring.py
import sys; sys.path.insert(0, ".")
sys.path.insert(0, "experiments/3d")
import numpy as np
from omegaconf import OmegaConf

from src.totalseg_dataset import _ALL_CLASSES_IDX

SIZE = 16
ANCHOR = "aorta"


def _make_root(tmp_path, n=4):
    idx = _ALL_CLASSES_IDX[ANCHOR]
    rows = ["image_id;split"]
    for i in range(n):
        subj = f"s{i:04d}"
        d = tmp_path / subj
        d.mkdir()
        label = np.zeros((SIZE, SIZE, SIZE), dtype=np.uint8)
        label[3:9, 5:11, 6:10] = idx
        ct = 0.3 * np.ones((SIZE, SIZE, SIZE), dtype=np.float16)
        np.save(d / "label.npy", label)
        np.save(d / f"label_{SIZE}x{SIZE}x{SIZE}.npy", label)
        np.save(d / f"ct_{SIZE}x{SIZE}x{SIZE}.npy", ct)
        rows.append(f"{subj};train")
    (tmp_path / "meta.csv").write_text("\n".join(rows) + "\n")
    return tmp_path


def test_build_dataset_dispatches_anchor_synth3d(tmp_path):
    from common import build_dataset
    from src.datasets.anchor_synth.dataset3d import AnchorSynth3DICLDataset

    root = str(_make_root(tmp_path))
    cfg = OmegaConf.create({
        "paths": {"totalseg": root},
        "data": {"source": "anchor_synth3d", "image_size": [SIZE, SIZE, SIZE],
                 "context_size": 1, "max_train_subjects": None},
        "anchor_synth": {"object_source": "blob", "shape": "blob", "n_objects": 1,
                         "anchor_classes": [ANCHOR], "offset_range": 0.2,
                         "scale_frac": 0.4, "scale_jitter": 0.15,
                         "rotate_jitter": 12.0, "contrast_delta": 0.3,
                         "edge_blur": 0.08, "boundary_complexity": 0.0,
                         "eval_subjects_per_task": 2, "eval_seed_namespace": 0,
                         "epoch_length": 5},
    })
    ds = build_dataset(cfg, "train")
    assert isinstance(ds, AnchorSynth3DICLDataset)
    assert len(ds) == 5                              # epoch_length (train)
    item = ds[0]
    assert item["image"].shape == (1, SIZE, SIZE, SIZE)
    assert item["label"].sum() > 0
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python -m pytest src/datasets/anchor_synth/test_wiring.py -q`
Expected: FAIL — `build_dataset` raises `unknown data.source 'anchor_synth3d'`.

- [ ] **Step 4: Add the `build_dataset` branch**

In `experiments/3d/common.py`, immediately after the `if cfg.data.get("source", "totalseg") == "omnisynth3d":` block (before the `d = cfg.data` line ~76), insert:

```python
    if cfg.data.get("source") == "anchor_synth3d":
        from src.datasets.anchor_synth.dataset3d import AnchorSynth3DICLDataset
        a = cfg.get("anchor_synth")
        if a is None:
            raise ValueError("data.source=anchor_synth3d requires an `anchor_synth` block")
        root = cfg.paths.get("totalseg")
        classes = resolve_classes(a.get("anchor_classes") or (), totalseg_root=root)
        is_train = split == "train"
        return AnchorSynth3DICLDataset(
            root=root, classes=classes, image_size=tuple(cfg.data.image_size),
            split=split, context_size=cfg.data.context_size,
            object_source=a.get("object_source", "blob"),
            shape=a.get("shape", "blob"), n_objects=int(a.get("n_objects", 1)),
            offset_range=float(a.get("offset_range", 0.6)),
            scale_frac=float(a.get("scale_frac", 0.4)),
            scale_jitter=float(a.get("scale_jitter", 0.15)),
            rotate_jitter=float(a.get("rotate_jitter", 12.0)),
            contrast_delta=float(a.get("contrast_delta", 0.15)),
            edge_blur=float(a.get("edge_blur", 0.08)),
            boundary_complexity=float(a.get("boundary_complexity", 0.0)),
            eval_subjects_per_task=int(a.get("eval_subjects_per_task", 4)),
            eval_seed_namespace=int(a.get("eval_seed_namespace", 0)),
            epoch_length=int(a.get("epoch_length", 10000)),
            deterministic=(split != "train"),
            max_subjects=(cfg.data.get("max_train_subjects") if is_train
                          else cfg.data.get("max_val_subjects")))
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest src/datasets/anchor_synth/test_wiring.py -q`
Expected: PASS (1 passed)

- [ ] **Step 6: Add the `make_eval_loader` + `eval.py` branches**

In `experiments/3d/common.py` `make_eval_loader`, extend the existing omnisynth condition (`if d.get("source") == "omnisynth3d":`) to also cover the new source:

```python
    if d.get("source") in ("omnisynth3d", "anchor_synth3d"):
```

In `experiments/3d/eval.py`, extend the class-listing branch (line ~68 `if source == "omnisynth3d":`) by adding, before it:

```python
    if source == "anchor_synth3d":
        from data.totalseg_classes import resolve_classes
        a = cfg.anchor_synth
        classes = resolve_classes(a.get("anchor_classes") or (),
                                  totalseg_root=cfg.paths.get("totalseg"))
    elif source == "omnisynth3d":
```

(convert the existing `if source == "omnisynth3d":` to `elif`; keep its body unchanged).

- [ ] **Step 7: Verify nothing regressed**

Run: `python -m pytest src/datasets/anchor_synth/ -q && python -c "import sys; sys.path.insert(0,'.'); sys.path.insert(0,'experiments/3d'); import common, eval"`
Expected: all anchor_synth tests pass; imports succeed with no error.

- [ ] **Step 8: Commit**

```bash
git add configs/experiment/3d/dataset/anchor_synth3d.yaml experiments/3d/common.py experiments/3d/eval.py src/datasets/anchor_synth/test_wiring.py
git commit -m "feat(anchor-synth3d): config + 3D harness wiring (build/eval)"
```

---

### Task 5: Docs

**Files:**
- Modify: `docs/logs.md` (append)

- [ ] **Step 1: Append a log entry**

Append to `docs/logs.md`:

```markdown
## anchor_synth3d dataset

Added `data.source=anchor_synth3d` (`dataset=anchor_synth3d`): pulls K+1 real CT
scans that share an anchor organ and draws a synthetic blob at a consistent
anchor-relative position (offset normalized to anchor extent, small per-scene
scale/rotation jitter, contrast blended to local background). Anchor is a
landmark only — the label is the drawn object(s). New package
`src/datasets/anchor_synth/` (analytic shapes + placement); subclasses
`TotalSegInContextDataset` for the scan cache + fast-path loading. v1 = blob
objects only; organ objects and multi-anchor deferred. Spec:
docs/superpowers/specs/2026-07-22-anchor-synth3d-design.md.
```

- [ ] **Step 2: Commit**

```bash
git add docs/logs.md
git commit -m "docs(anchor-synth3d): log dataset addition"
```

---

## Self-Review Notes

- **Spec coverage:** shapes/irregularity (Task 1), anchor stats + blend (Task 2), subclass dataset + determinism + anchor-unlabeled + multi-object same-anchor (Task 3), config + `build_dataset`/`make_eval_loader`/`eval.py` wiring (Task 4), logs (Task 5). Real-organ `object_source` deferred with an explicit `NotImplementedError` per Global Constraints.
- **Determinism:** eval seed uses `_ALL_CLASSES_IDX[anchor_cls]` (integer), matching the spec fix.
- **Types:** `render_object`/`sample_object_spec`/`small_rotation`/`roughen` (Task 1) consumed with identical signatures in Task 3; `anchor_stats`/`offset_to_center`/`place_object` (Task 2) likewise.
