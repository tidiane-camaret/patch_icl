# Cohort-Consistent Synthetic Shapes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `p_shape`-gated cohort mode to `SynthGmmProvider` that stamps a
procedurally generated geometric shape (blob/splatter/disk/cylinder) into a real
gmm_bank host organ crop, with four independently controllable per-axis consistency
knobs (shape, size, position, intensity) across the K+1 cohort members.

**Architecture:** A new pure-numpy `src/shapes3d/` package (primitives + per-axis
instantiation, no torch/provider dependencies, independently testable) plugs into
`SynthGmmProvider._build_nc` at one insertion point. The shape's own cohort/per-member
randomness reuses the provider's existing two-level, cascade-reproducible RNG scheme
(`gmm_seed` for cohort-shared draws, `(gmm_seed, member_idx)` for each member's
instantiation) — the same scheme `_draw_gmm`/`mu_e` already use for intensity — so
cascade re-crop consistency requires no new plumbing beyond one extra subject-string
field.

**Tech Stack:** Python, NumPy (shape rasterization), PyTorch (unchanged downstream
paint/resample pipeline), pytest.

**Spec:** `docs/superpowers/specs/2026-09-14-cohort-consistent-synthetic-shapes-design.md`

## Global Constraints

- `p_shape` defaults to `0.0` — no existing run (including `92_multisource_synth`) may
  change behavior until a config explicitly sets `data.gmm.p_shape > 0`.
- Shape rasterization uses **loose containment**: the shape's *position* is drawn from
  inside the host organ, but the rasterized mask is never clipped or retried against the
  organ boundary (spec §2.3). No reject-until-valid loop, ever.
- All four consistency knobs (`shape_between_ratio`, `size_between_ratio`,
  `position_between_ratio`, `intensity_between_ratio`) are continuous `[0,1]`, `0` =
  cohort-identical, `1` = fully independent per member (spec §2.2), except shape
  *family*, which is a discrete per-cohort choice (spec §2.2).
- Each of the K+1 cohort members gets its own independently-sampled host donor entry
  (spec §2.6); host class is drawn unrestricted via the existing
  `GmmCohortSampler.sample_cohort(rng)` (spec §2.7).
- A shape overwrites the host label at its footprint under a new pseudo-class id (spec
  §2.5) — registered in `data/maisi_classes.py` the same way `200: 'body'` was, **not**
  `201`–`204` as the spec's design doc illustratively suggested: those numbers are free
  in the current vocabulary but so are `133`–`199`, and `mu`/`sd` arrays are sized
  `maxid + 1` (default `maxid=200`, `src/synth_gmm_maisi_dataset.py:35`) — an id above
  `maxid` would index out of bounds. Use `195`–`198` (inside the default `maxid=200`
  bound, no config change required).

---

### Task 1: Shape primitives (`src/shapes3d/primitives.py`)

Pure NumPy geometry generators. No torch, no provider/dataset imports. Each
`make_<family>` takes the grid shape, a center voxel, a params dict, and an
`np.random.Generator`, and returns `(mask: np.ndarray[bool], realized_meta: dict)`.

**Files:**
- Create: `src/shapes3d/__init__.py` (empty)
- Create: `src/shapes3d/primitives.py`
- Test: `src/shapes3d/test_primitives.py`

**Interfaces:**
- Produces: `make_blob(shape, center, params, rng)`, `make_splatter(shape, center,
  params, rng)`, `make_disk(shape, center, params, rng)`, `make_cylinder(shape, center,
  params, rng)`, `make_shape(family, shape, center, params, rng)` (dispatcher) — all
  return `(mask: np.ndarray[bool] of shape `shape`, meta: dict)`. `params` always
  includes `"size_frac"` (target fraction of `shape`'s total voxel count); family-specific
  keys documented per function below.

- [ ] **Step 1: Write the failing tests**

```python
# src/shapes3d/test_primitives.py
import numpy as np
import pytest

from src.shapes3d.primitives import make_blob, make_cylinder, make_disk, make_shape, make_splatter

SHAPE = (40, 40, 40)
CENTER = (20.0, 20.0, 20.0)


def test_make_blob_hits_requested_size_frac_within_tolerance():
    rng = np.random.default_rng(0)
    mask, meta = make_blob(SHAPE, CENTER, {"size_frac": 0.05, "roughness": 0.15}, rng)
    assert mask.shape == SHAPE
    assert mask.dtype == np.uint8 or mask.dtype == bool
    realized = mask.astype(bool).mean()
    assert 0.02 < realized < 0.09          # harmonic perturbation moves it off-target some
    assert meta["family"] == "blob"
    assert meta["realized_size_frac"] == pytest.approx(realized, abs=1e-9)


def test_make_blob_is_deterministic_given_the_same_rng_state():
    mask1, _ = make_blob(SHAPE, CENTER, {"size_frac": 0.05, "roughness": 0.15},
                         np.random.default_rng(42))
    mask2, _ = make_blob(SHAPE, CENTER, {"size_frac": 0.05, "roughness": 0.15},
                         np.random.default_rng(42))
    np.testing.assert_array_equal(mask1, mask2)


def test_make_blob_roughness_zero_is_a_sphere():
    rng = np.random.default_rng(0)
    mask, _ = make_blob(SHAPE, CENTER, {"size_frac": 0.05, "roughness": 0.0}, rng)
    coords = np.argwhere(mask.astype(bool))
    dist = np.linalg.norm(coords - np.array(CENTER), axis=1)
    assert dist.std() < 0.75               # every surface voxel ~equidistant from center


def test_make_splatter_has_multiple_components():
    from scipy.ndimage import label as cc_label
    rng = np.random.default_rng(1)
    mask, meta = make_splatter(
        SHAPE, CENTER, {"size_frac": 0.05, "n_components": 5, "spread_frac": 0.3,
                       "roughness": 0.15}, rng)
    n_components, count = cc_label(mask.astype(bool))
    assert count >= 2                       # "several disjoint components", not one blob
    assert meta["family"] == "splatter"


def test_make_disk_is_flattened_along_its_axis():
    rng = np.random.default_rng(2)
    mask, meta = make_disk(
        SHAPE, CENTER, {"size_frac": 0.05, "aspect_ratio": 0.2, "flatten_axis": 0}, rng)
    coords = np.argwhere(mask.astype(bool))
    extent = coords.max(0) - coords.min(0)
    assert extent[0] < extent[1] * 0.6      # flattened axis visibly shorter than the others
    assert extent[0] < extent[2] * 0.6


def test_make_cylinder_is_elongated_along_its_axis():
    rng = np.random.default_rng(3)
    mask, meta = make_cylinder(
        SHAPE, CENTER,
        {"size_frac": 0.05, "radius_frac": 0.2, "length_frac": 0.7,
         "azimuth": 0.0, "elevation": 0.0}, rng)
    coords = np.argwhere(mask.astype(bool))
    extent = coords.max(0) - coords.min(0)
    assert extent.max() > extent.min() * 1.5   # visibly elongated, not isotropic
    assert meta["family"] == "cylinder"


def test_make_shape_dispatches_by_family_name():
    rng = np.random.default_rng(0)
    mask, meta = make_shape("disk", SHAPE, CENTER,
                            {"size_frac": 0.05, "aspect_ratio": 0.2, "flatten_axis": 1}, rng)
    assert meta["family"] == "disk"


def test_make_shape_rejects_unknown_family():
    with pytest.raises(ValueError):
        make_shape("not_a_family", SHAPE, CENTER, {"size_frac": 0.05}, np.random.default_rng(0))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/dpxuser/dev/patch_icl && python -m pytest src/shapes3d/test_primitives.py -v`
Expected: `ModuleNotFoundError: No module named 'src.shapes3d'`

- [ ] **Step 3: Write the implementation**

```python
# src/shapes3d/__init__.py
```

```python
# src/shapes3d/primitives.py
"""Procedural 3D shape primitives for synth_gmm's shape-mode cohorts (see
docs/superpowers/specs/2026-09-14-cohort-consistent-synthetic-shapes-design.md).

Pure NumPy: no torch, no provider/dataset imports. Every make_<family> function takes
the target grid shape, a center voxel (float coords, may be non-integer), a params
dict, and an np.random.Generator, and returns (mask uint8[*shape], realized_meta dict).
Shapes are NOT anatomically plausible by design (spec: "do not need to be plausible
lesions") -- they exist to teach general geometric reasoning, not organ realism.

Volume targeting is closed-form (radius solved from the target voxel count), not an
iterative search -- cheap, and "realized" (measured from the rasterized mask) is always
recorded rather than trusting the request, since harmonic/angular perturbation and
voxel discretization both move the actual volume off-target."""

import numpy as np

_EPS = 1e-9


def _grid_dist_angle(shape, center):
    """(rr, theta, phi): spherical radius / polar angle (from axis 0) / azimuth (in the
    axis1-axis2 plane) of every grid voxel relative to `center`."""
    D, H, W = shape
    dd, hh, ww = np.mgrid[0:D, 0:H, 0:W].astype(np.float64)
    dz, dy, dx = dd - center[0], hh - center[1], ww - center[2]
    rr = np.sqrt(dz * dz + dy * dy + dx * dx)
    theta = np.arccos(np.divide(dz, rr, out=np.zeros_like(rr), where=rr > _EPS))
    phi = np.arctan2(dy, dx)
    return rr, theta, phi


def _harmonics(theta, phi, rng, amp, terms=((2, 1), (3, 2), (4, 1))):
    """Low-frequency multiplicative angular perturbation -> organic (non-spherical)
    surface. amp=0 -> exactly 1.0 everywhere (a perfect sphere)."""
    out = np.ones_like(theta)
    if amp <= 0.0:
        return out
    for l, m in terms:
        a = rng.uniform(-amp, amp)
        d_theta = rng.uniform(0, 2 * np.pi)
        d_phi = rng.uniform(0, 2 * np.pi)
        out = out + a * np.cos(l * theta + d_theta) * np.cos(m * phi + d_phi)
    return np.clip(out, 0.4, 1.6)


def _sphere_radius_for_size_frac(shape, size_frac):
    """Equivalent-sphere radius hitting `size_frac` of the grid's total voxel count."""
    n_vox = shape[0] * shape[1] * shape[2]
    target_vox = max(1.0, float(size_frac) * n_vox)
    return (3.0 * target_vox / (4.0 * np.pi)) ** (1.0 / 3.0)


def make_blob(shape, center, params, rng):
    """Roughly round organic blob: a harmonic-perturbed sphere. params: size_frac,
    roughness (angular perturbation amplitude, 0=perfect sphere, ~0.3=quite irregular)."""
    base_r = _sphere_radius_for_size_frac(shape, params["size_frac"])
    rr, theta, phi = _grid_dist_angle(shape, center)
    r_dir = base_r * _harmonics(theta, phi, rng, params.get("roughness", 0.15))
    mask = (rr <= r_dir).astype(np.uint8)
    return mask, {"family": "blob", "realized_size_frac": float(mask.mean())}


def make_splatter(shape, center, params, rng):
    """Scattered cluster of several small blobs under one label -- 3D analog of
    controlSynth's shapes/scattered.py. A distinct failure mode from a single blob: the
    model has to find ALL components, not just the nearest one. params: size_frac,
    n_components, spread_frac (cluster radius as a fraction of the mean grid extent),
    roughness (per-component)."""
    n = max(1, int(round(params.get("n_components", 4))))
    spread = float(params.get("spread_frac", 0.25)) * float(np.mean(shape))
    n_vox = shape[0] * shape[1] * shape[2]
    target_vox = max(1.0, float(params["size_frac"]) * n_vox)
    per_component_vox = target_vox / n
    comp_r = (3.0 * per_component_vox / (4.0 * np.pi)) ** (1.0 / 3.0)
    mask = np.zeros(shape, dtype=bool)
    upper = np.array(shape, dtype=np.float64) - 1.0
    for _ in range(n):
        offset = rng.normal(0.0, spread, size=3)
        c = np.clip(np.asarray(center, dtype=np.float64) + offset, 0.0, upper)
        rr, theta, phi = _grid_dist_angle(shape, c)
        r_dir = comp_r * _harmonics(theta, phi, rng, params.get("roughness", 0.15))
        mask |= (rr <= r_dir)
    mask = mask.astype(np.uint8)
    return mask, {"family": "splatter", "realized_size_frac": float(mask.mean()),
                  "n_components": n}


def make_disk(shape, center, params, rng):
    """Flattened ellipsoid: one axis scaled by `aspect_ratio` (<1 flattens it). params:
    size_frac, aspect_ratio (flattened-axis semi-length / other-axis semi-length),
    flatten_axis (0/1/2, which grid axis is flattened)."""
    axis = int(params.get("flatten_axis", 0)) % 3
    aspect = float(np.clip(params.get("aspect_ratio", 0.25), 0.05, 0.95))
    n_vox = shape[0] * shape[1] * shape[2]
    target_vox = max(1.0, float(params["size_frac"]) * n_vox)
    # ellipsoid volume = 4/3 pi * r^2 * (aspect*r) = aspect * sphere_volume(r)
    r = (3.0 * target_vox / (4.0 * np.pi * aspect)) ** (1.0 / 3.0)
    semi = [r, r, r]
    semi[axis] = r * aspect
    D, H, W = shape
    dd, hh, ww = np.mgrid[0:D, 0:H, 0:W].astype(np.float64)
    dz, dy, dx = dd - center[0], hh - center[1], ww - center[2]
    val = (dz / semi[0]) ** 2 + (dy / semi[1]) ** 2 + (dx / semi[2]) ** 2
    mask = (val <= 1.0).astype(np.uint8)
    return mask, {"family": "disk", "realized_size_frac": float(mask.mean()),
                  "flatten_axis": axis, "aspect_ratio": aspect}


def make_cylinder(shape, center, params, rng):
    """Capsule: voxels within `radius` of a line segment of length `length` through
    `center`, oriented by (azimuth, elevation). params: size_frac, radius_frac
    (unused directly -- radius is solved from size_frac and length so the two knobs
    don't fight; radius_frac is accepted for API symmetry with the spec but the solved
    radius is what's realized, always recorded in meta), length_frac (segment length as
    a fraction of the grid diagonal), azimuth, elevation (radians)."""
    D, H, W = shape
    az = float(params.get("azimuth", rng.uniform(0, 2 * np.pi)))
    el = float(params.get("elevation", rng.uniform(-np.pi / 2, np.pi / 2)))
    direction = np.array([np.sin(el), np.cos(el) * np.sin(az), np.cos(el) * np.cos(az)])
    diag = float(np.sqrt(D * D + H * H + W * W))
    length = max(1.0, float(params.get("length_frac", 0.6)) * diag)
    n_vox = D * H * W
    target_vox = max(1.0, float(params["size_frac"]) * n_vox)
    # cylinder-only approx (end-cap volume is a small correction at these aspect ratios;
    # the REALIZED fraction below is measured from the actual rasterized mask, not this).
    radius = max(0.75, float(np.sqrt(target_vox / (np.pi * length))))

    c = np.asarray(center, dtype=np.float64)
    p0, p1 = c - 0.5 * length * direction, c + 0.5 * length * direction
    dd, hh, ww = np.mgrid[0:D, 0:H, 0:W].astype(np.float64)
    pts = np.stack([dd, hh, ww], axis=-1)
    seg = p1 - p0
    seg_len2 = float(seg @ seg) or _EPS
    t = np.clip(((pts - p0) @ seg) / seg_len2, 0.0, 1.0)
    closest = p0 + t[..., None] * seg
    dist = np.linalg.norm(pts - closest, axis=-1)
    mask = (dist <= radius).astype(np.uint8)
    return mask, {"family": "cylinder", "realized_size_frac": float(mask.mean()),
                  "radius": radius, "length": length}


_FAMILIES = {"blob": make_blob, "splatter": make_splatter, "disk": make_disk,
            "cylinder": make_cylinder}


def make_shape(family, shape, center, params, rng):
    """Dispatch to make_<family>. Raises ValueError for an unknown family name."""
    fn = _FAMILIES.get(family)
    if fn is None:
        raise ValueError(f"unknown shape family {family!r} (expected one of {sorted(_FAMILIES)})")
    return fn(shape, center, params, rng)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/dpxuser/dev/patch_icl && python -m pytest src/shapes3d/test_primitives.py -v`
Expected: all PASS. If `test_make_splatter_has_multiple_components` fails because scipy
isn't available, confirm `scipy` is already a project dependency (`grep scipy
pyproject.toml`) before troubleshooting further — it is used elsewhere in this repo
(`scripts/augmentation.py`-style code from NV-Generate-CTMR uses it too, and patch_icl's
own `src/synth_gmm_maisi_dataset.py` / eval code depend on scipy already).

- [ ] **Step 5: Commit**

```bash
git add src/shapes3d/__init__.py src/shapes3d/primitives.py src/shapes3d/test_primitives.py
git commit -m "feat(shapes3d): procedural blob/splatter/disk/cylinder primitives

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01AfbPWj3NvbKfBFCN2CKGZ6"
```

---

### Task 2: Cohort spec + per-axis consistency instantiation

**Files:**
- Create: `src/shapes3d/spec.py`
- Create: `src/shapes3d/instantiate.py`
- Test: `src/shapes3d/test_instantiate.py`

**Interfaces:**
- Consumes: `make_shape` from Task 1 (`src/shapes3d/primitives.py`).
- Produces: `ShapeCohortSpec` (dataclass, `src/shapes3d/spec.py`); `CohortHyperparams`,
  `MemberShapeDraw` (dataclasses), `draw_cohort_hyperparams(rng: np.random.Generator,
  spec: ShapeCohortSpec) -> CohortHyperparams`, `draw_member_shape(member_rng:
  np.random.Generator, cohort_hp: CohortHyperparams, spec: ShapeCohortSpec) ->
  MemberShapeDraw`, `rasterize_shape_in_crop(crop_lbl: np.ndarray, host_cls_id: int,
  shape_id: int, member_draw: MemberShapeDraw, rng: np.random.Generator) -> None`
  (mutates `crop_lbl` in place) — all in `src/shapes3d/instantiate.py`.

- [ ] **Step 1: Write the failing tests**

```python
# src/shapes3d/test_instantiate.py
import numpy as np
import pytest

from src.shapes3d.instantiate import (draw_cohort_hyperparams, draw_member_shape,
                                      rasterize_shape_in_crop)
from src.shapes3d.spec import ShapeCohortSpec


def test_draw_cohort_hyperparams_is_deterministic():
    spec = ShapeCohortSpec()
    hp1 = draw_cohort_hyperparams(np.random.default_rng(7), spec)
    hp2 = draw_cohort_hyperparams(np.random.default_rng(7), spec)
    assert hp1.family == hp2.family
    assert hp1.size_frac == hp2.size_frac
    np.testing.assert_array_equal(hp1.position_uvw, hp2.position_uvw)
    assert hp1.shape_params == hp2.shape_params


def test_draw_cohort_hyperparams_family_always_one_of_the_four():
    spec = ShapeCohortSpec()
    for seed in range(20):
        hp = draw_cohort_hyperparams(np.random.default_rng(seed), spec)
        assert hp.family in ("blob", "splatter", "disk", "cylinder")


def test_between_ratio_zero_reuses_the_cohort_value_exactly():
    """size_between_ratio=0 -> every member's size_frac equals the cohort's, regardless
    of what the member's own RNG stream would have drawn independently."""
    spec = ShapeCohortSpec(size_between_ratio=0.0, position_between_ratio=0.0,
                          shape_between_ratio=0.0)
    cohort_hp = draw_cohort_hyperparams(np.random.default_rng(1), spec)
    for seed in (2, 3, 4):
        member = draw_member_shape(np.random.default_rng(seed), cohort_hp, spec)
        assert member.size_frac == pytest.approx(cohort_hp.size_frac)
        np.testing.assert_allclose(member.position_uvw, cohort_hp.position_uvw)


def test_between_ratio_one_draws_independently_per_member():
    """size_between_ratio=1 -> members drawn with different seeds get different
    size_frac (statistically -- not guaranteed for any single pair, so assert the
    cohort's 5 members aren't all identical, which between_ratio=0 always would be)."""
    spec = ShapeCohortSpec(size_between_ratio=1.0)
    cohort_hp = draw_cohort_hyperparams(np.random.default_rng(1), spec)
    sizes = [draw_member_shape(np.random.default_rng(s), cohort_hp, spec).size_frac
             for s in range(5)]
    assert len(set(sizes)) > 1


def test_between_ratio_blends_linearly_between_cohort_and_fresh_draw():
    spec_identical = ShapeCohortSpec(size_between_ratio=0.0)
    spec_half = ShapeCohortSpec(size_between_ratio=0.5)
    spec_full = ShapeCohortSpec(size_between_ratio=1.0)
    cohort_hp = draw_cohort_hyperparams(np.random.default_rng(9), spec_identical)
    m0 = draw_member_shape(np.random.default_rng(3), cohort_hp, spec_identical).size_frac
    m_half = draw_member_shape(np.random.default_rng(3), cohort_hp, spec_half).size_frac
    m1 = draw_member_shape(np.random.default_rng(3), cohort_hp, spec_full).size_frac
    assert m0 == pytest.approx(cohort_hp.size_frac)
    assert m_half == pytest.approx((m0 + m1) / 2, abs=1e-6)


def test_cylinder_orientation_blends_via_shortest_angle_not_raw_value():
    """Guards against the 0/2*pi wrap making a small between_ratio look like a huge
    orientation jump."""
    spec = ShapeCohortSpec(family_weights={"cylinder": 1.0}, shape_between_ratio=0.1)
    cohort_hp = draw_cohort_hyperparams(np.random.default_rng(5), spec)
    cohort_hp.shape_params["azimuth"] = 0.05    # pin near the wrap boundary
    member = draw_member_shape(np.random.default_rng(11), cohort_hp, spec)
    # a small between_ratio must keep azimuth close to the cohort value going the SHORT
    # way around the circle, never off by ~2*pi
    diff = abs((member.shape_params["azimuth"] - cohort_hp.shape_params["azimuth"] + np.pi)
              % (2 * np.pi) - np.pi)
    assert diff < 1.0


def test_rasterize_shape_in_crop_stamps_the_new_class_id_into_crop_lbl():
    crop_lbl = np.zeros((30, 30, 30), dtype=np.uint8)
    crop_lbl[5:25, 5:25, 5:25] = 7          # host organ footprint (class 7)
    spec = ShapeCohortSpec(family_weights={"blob": 1.0})
    cohort_hp = draw_cohort_hyperparams(np.random.default_rng(0), spec)
    member = draw_member_shape(np.random.default_rng(1), cohort_hp, spec)
    rasterize_shape_in_crop(crop_lbl, host_cls_id=7, shape_id=195,
                            member_draw=member, rng=np.random.default_rng(2))
    assert (crop_lbl == 195).any()


def test_rasterize_shape_in_crop_falls_back_to_whole_crop_when_host_absent():
    """Loose containment (spec sec 2.3): if the host class has no voxels in this local
    crop, the shape is still stamped somewhere sane (whole-crop fallback), never raises."""
    crop_lbl = np.zeros((20, 20, 20), dtype=np.uint8)   # host class 7 not present at all
    spec = ShapeCohortSpec(family_weights={"blob": 1.0})
    cohort_hp = draw_cohort_hyperparams(np.random.default_rng(0), spec)
    member = draw_member_shape(np.random.default_rng(1), cohort_hp, spec)
    rasterize_shape_in_crop(crop_lbl, host_cls_id=7, shape_id=195,
                            member_draw=member, rng=np.random.default_rng(2))
    assert (crop_lbl == 195).any()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/dpxuser/dev/patch_icl && python -m pytest src/shapes3d/test_instantiate.py -v`
Expected: `ModuleNotFoundError: No module named 'src.shapes3d.spec'`

- [ ] **Step 3: Write the implementation**

```python
# src/shapes3d/spec.py
"""Config surface for synth_gmm's shape-mode cohorts. See
docs/superpowers/specs/2026-09-14-cohort-consistent-synthetic-shapes-design.md."""

from dataclasses import dataclass, field


@dataclass
class ShapeCohortSpec:
    """Every *_between_ratio is in [0,1]: 0 = every cohort member reuses the exact
    cohort-drawn value, 1 = every member redraws independently within the ranges below.
    `intensity_between_ratio=None` (default) means the shape's pseudo-class uses
    whatever the provider's own gmm.between_ratio setting already gives every other
    class -- set it to override just the shape's intensity consistency independently
    of that global toggle."""
    family_weights: dict = field(default_factory=lambda: {
        "blob": 1.0, "splatter": 1.0, "disk": 1.0, "cylinder": 1.0})
    size_frac_range: tuple = (0.02, 0.15)
    shape_between_ratio: float = 0.3
    size_between_ratio: float = 0.3
    position_between_ratio: float = 0.5
    intensity_between_ratio: float | None = None

    blob_roughness_range: tuple = (0.05, 0.3)
    splatter_n_components_range: tuple = (2, 6)
    splatter_spread_frac_range: tuple = (0.15, 0.4)
    splatter_roughness_range: tuple = (0.1, 0.2)
    disk_aspect_ratio_range: tuple = (0.15, 0.4)
    cylinder_radius_frac_range: tuple = (0.15, 0.35)
    cylinder_length_frac_range: tuple = (0.5, 0.8)
```

```python
# src/shapes3d/instantiate.py
"""Per-axis cohort-consistency instantiation for shape-mode cohorts: a cohort-level
draw (shared, like synth_gmm's mu/sd) blended toward each member's independent draw by
that axis's *_between_ratio (like synth_gmm's mu_e). See
docs/superpowers/specs/2026-09-14-cohort-consistent-synthetic-shapes-design.md."""

from dataclasses import dataclass, field

import numpy as np

from .primitives import make_shape

# (family, param key) -> ShapeCohortSpec attribute holding that param's [lo, hi] prior
# range -- used to draw both the cohort baseline and each member's fresh redraw.
_CONTINUOUS_RANGE = {
    ("blob", "roughness"): "blob_roughness_range",
    ("splatter", "spread_frac"): "splatter_spread_frac_range",
    ("splatter", "roughness"): "splatter_roughness_range",
    ("disk", "aspect_ratio"): "disk_aspect_ratio_range",
    ("cylinder", "radius_frac"): "cylinder_radius_frac_range",
    ("cylinder", "length_frac"): "cylinder_length_frac_range",
}


@dataclass
class CohortHyperparams:
    family: str
    size_frac: float
    position_uvw: np.ndarray            # (3,) in [0,1], cohort baseline
    shape_params: dict = field(default_factory=dict)


@dataclass
class MemberShapeDraw:
    family: str
    size_frac: float
    position_uvw: np.ndarray
    shape_params: dict = field(default_factory=dict)


def _blend(cohort_value, fresh_value, ratio):
    """ratio=0 -> cohort_value; ratio=1 -> fresh_value; linear in between."""
    return cohort_value + ratio * (fresh_value - cohort_value)


def _angle_diff(a, b):
    """Shortest signed a-b, wrapped to [-pi, pi] -- so blending an angle never jumps
    the long way around the circle."""
    return float((a - b + np.pi) % (2 * np.pi) - np.pi)


def draw_cohort_hyperparams(rng, spec):
    """Once per cohort (caller keys `rng` off the cohort's own seed, e.g.
    np.random.default_rng([gmm_seed, -1])) -- family is a discrete per-cohort choice
    (never blended); every continuous param gets a baseline draw here."""
    families = list(spec.family_weights)
    weights = np.array([float(spec.family_weights[f]) for f in families], dtype=float)
    family = families[int(rng.choice(len(families), p=weights / weights.sum()))]

    size_frac = float(rng.uniform(*spec.size_frac_range))
    position_uvw = rng.uniform(0.15, 0.85, size=3)

    if family == "blob":
        shape_params = {"roughness": float(rng.uniform(*spec.blob_roughness_range))}
    elif family == "splatter":
        lo, hi = spec.splatter_n_components_range
        shape_params = {
            "n_components": int(rng.integers(int(lo), int(hi) + 1)),
            "spread_frac": float(rng.uniform(*spec.splatter_spread_frac_range)),
            "roughness": float(rng.uniform(*spec.splatter_roughness_range)),
        }
    elif family == "disk":
        shape_params = {
            "aspect_ratio": float(rng.uniform(*spec.disk_aspect_ratio_range)),
            "flatten_axis": int(rng.integers(0, 3)),
        }
    elif family == "cylinder":
        shape_params = {
            "radius_frac": float(rng.uniform(*spec.cylinder_radius_frac_range)),
            "length_frac": float(rng.uniform(*spec.cylinder_length_frac_range)),
            "azimuth": float(rng.uniform(0, 2 * np.pi)),
            "elevation": float(rng.uniform(-np.pi / 2, np.pi / 2)),
        }
    else:
        raise ValueError(f"unknown shape family {family!r}")

    return CohortHyperparams(family=family, size_frac=size_frac,
                             position_uvw=position_uvw, shape_params=shape_params)


def draw_member_shape(member_rng, cohort_hp, spec):
    """Once per cohort member (caller keys `member_rng` off (gmm_seed, member_idx), the
    same stream synth_gmm's own per-member intensity draw already uses)."""
    family = cohort_hp.family

    fresh_size = float(member_rng.uniform(*spec.size_frac_range))
    size_frac = _blend(cohort_hp.size_frac, fresh_size, spec.size_between_ratio)

    fresh_uvw = member_rng.uniform(0.15, 0.85, size=3)
    position_uvw = _blend(cohort_hp.position_uvw, fresh_uvw, spec.position_between_ratio)

    shape_params = dict(cohort_hp.shape_params)
    for (fam, key), range_attr in _CONTINUOUS_RANGE.items():
        if fam != family:
            continue
        fresh = float(member_rng.uniform(*getattr(spec, range_attr)))
        shape_params[key] = _blend(shape_params[key], fresh, spec.shape_between_ratio)

    if family == "splatter":
        lo, hi = spec.splatter_n_components_range
        fresh_n = int(member_rng.integers(int(lo), int(hi) + 1))
        shape_params["n_components"] = int(round(
            _blend(shape_params["n_components"], fresh_n, spec.shape_between_ratio)))

    if family == "cylinder":
        fresh_az = float(member_rng.uniform(0, 2 * np.pi))
        shape_params["azimuth"] = float(
            shape_params["azimuth"]
            + spec.shape_between_ratio * _angle_diff(fresh_az, shape_params["azimuth"]))
        fresh_el = float(member_rng.uniform(-np.pi / 2, np.pi / 2))
        shape_params["elevation"] = _blend(shape_params["elevation"], fresh_el,
                                           spec.shape_between_ratio)

    return MemberShapeDraw(family=family, size_frac=size_frac,
                           position_uvw=position_uvw, shape_params=shape_params)


def rasterize_shape_in_crop(crop_lbl, host_cls_id, shape_id, member_draw, rng):
    """Stamp one member's shape into `crop_lbl` (uint8, native crop grid) under
    `shape_id`, positioned at `member_draw.position_uvw` relative to the HOST class's
    own bounding box within this crop. Loose containment (design doc sec 2.3): if the
    host class has no voxels here, falls back to the whole crop as the placement region
    -- the shape's rasterization itself is never clipped to the host boundary either
    way. Mutates `crop_lbl` in place."""
    shape = crop_lbl.shape
    host_fg = np.argwhere(crop_lbl == host_cls_id)
    if host_fg.size > 0:
        bbox_lo, bbox_hi = host_fg.min(0).astype(np.float64), host_fg.max(0).astype(np.float64)
    else:
        bbox_lo, bbox_hi = np.zeros(3), np.array(shape, dtype=np.float64) - 1.0
    center = bbox_lo + member_draw.position_uvw * (bbox_hi - bbox_lo)

    params = dict(member_draw.shape_params)
    params["size_frac"] = member_draw.size_frac
    mask, _meta = make_shape(member_draw.family, shape, center, params, rng)
    crop_lbl[mask.astype(bool)] = shape_id
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/dpxuser/dev/patch_icl && python -m pytest src/shapes3d/test_instantiate.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/shapes3d/spec.py src/shapes3d/instantiate.py src/shapes3d/test_instantiate.py
git commit -m "feat(shapes3d): per-axis cohort-consistency instantiation

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01AfbPWj3NvbKfBFCN2CKGZ6"
```

---

### Task 3: Register shape pseudo-classes

**Files:**
- Modify: `data/maisi_classes.py`
- Test: `data/test_maisi_classes.py` (create if it doesn't already cover this; check
  first with `ls data/test_maisi_classes.py`)

**Interfaces:**
- Produces: `MAISI_IDX_TO_CLASS[195..198]` = `'shape_blob' | 'shape_splatter' |
  'shape_disk' | 'shape_cylinder'`; `SHAPE_ID_TO_FAMILY: dict[int, str]` mapping those
  same 4 ids to the bare family name (`"blob"` etc., matching `ShapeCohortSpec`'s
  `family_weights` keys and `make_shape`'s family argument) — new symbol in
  `data/maisi_classes.py`, consumed by Task 4.

- [ ] **Step 1: Write the failing test**

```python
# data/test_maisi_classes.py  (append if the file already exists)
from data.maisi_classes import MAISI_CLASS_TO_IDX, MAISI_IDX_TO_CLASS, SHAPE_ID_TO_FAMILY


def test_shape_pseudo_classes_are_registered_and_round_trip():
    expected = {195: "shape_blob", 196: "shape_splatter", 197: "shape_disk",
               198: "shape_cylinder"}
    for idx, name in expected.items():
        assert MAISI_IDX_TO_CLASS[idx] == name
        assert MAISI_CLASS_TO_IDX[name] == idx


def test_shape_id_to_family_matches_the_bare_family_names():
    assert SHAPE_ID_TO_FAMILY == {195: "blob", 196: "splatter", 197: "disk",
                                  198: "cylinder"}


def test_shape_ids_stay_within_the_default_maxid_bound():
    """mu/sd arrays are sized maxid+1 (default maxid=200,
    src/synth_gmm_maisi_dataset.py:35) -- a shape id above maxid would index out of
    bounds when SynthGmmProvider indexes mu[crop_lbl]."""
    assert max(SHAPE_ID_TO_FAMILY) <= 200
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/dpxuser/dev/patch_icl && python -m pytest data/test_maisi_classes.py -v`
Expected: `ImportError: cannot import name 'SHAPE_ID_TO_FAMILY'`

- [ ] **Step 3: Register the pseudo-classes**

In `data/maisi_classes.py`, add the four new entries to the existing
`MAISI_IDX_TO_CLASS` dict, right after `200: 'body',` (or before it — position in the
dict doesn't matter, `MAISI_CLASS_TO_IDX`/`MAISI_CLASSES` are both derived from it):

```python
    200: 'body',
    195: 'shape_blob',
    196: 'shape_splatter',
    197: 'shape_disk',
    198: 'shape_cylinder',
}

MAISI_CLASS_TO_IDX = {v: k for k, v in MAISI_IDX_TO_CLASS.items()}
MAISI_CLASSES = [MAISI_IDX_TO_CLASS[k] for k in sorted(MAISI_IDX_TO_CLASS)]

# Shape-mode pseudo-class ids (docs/superpowers/specs/
# 2026-09-14-cohort-consistent-synthetic-shapes-design.md) -> the bare family name
# ShapeCohortSpec.family_weights and shapes3d.primitives.make_shape use.
SHAPE_ID_TO_FAMILY = {195: 'blob', 196: 'splatter', 197: 'disk', 198: 'cylinder'}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/dpxuser/dev/patch_icl && python -m pytest data/test_maisi_classes.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add data/maisi_classes.py data/test_maisi_classes.py
git commit -m "feat(maisi_classes): register shape pseudo-classes 195-198

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01AfbPWj3NvbKfBFCN2CKGZ6"
```

---

### Task 4: Wire `SynthGmmProvider`

This is the integration task: `p_shape`/`shape` config, `assemble_task`'s shape-mode
branch, `_build_nc`'s shape-stamping step, and `load_native_crop`'s reconstruction —
using the exact same `_make_bank`/`SynthGmmMaisiDataset` fixture pattern already
established in `src/providers/test_synth_gmm.py`.

**Files:**
- Modify: `src/providers/synth_gmm.py`
- Modify: `src/providers/test_synth_gmm.py`

**Interfaces:**
- Consumes: `ShapeCohortSpec`, `draw_cohort_hyperparams`, `draw_member_shape`,
  `rasterize_shape_in_crop` (Tasks 1-2, `src/shapes3d/`); `SHAPE_ID_TO_FAMILY` (Task 3,
  `data/maisi_classes.py`).
- Produces: `SynthGmmProvider(dataset, cascade=True, p_shape: float = 0.0,
  shape_spec: ShapeCohortSpec | None = None)` — new constructor kwargs, both optional
  and backward compatible (existing call sites that only pass `dataset`/`cascade`
  continue to work unchanged, `p_shape=0.0` means shape mode never fires).

- [ ] **Step 1: Write the failing tests**

Add to `src/providers/test_synth_gmm.py` (new imports at the top, new test functions
appended):

```python
from data.maisi_classes import SHAPE_ID_TO_FAMILY
from src.shapes3d.spec import ShapeCohortSpec
```

```python
def _make_shape_provider(tmp_path, p_shape=1.0, shape_spec=None):
    bank_dir = _make_bank(tmp_path)
    ds = SynthGmmMaisiDataset(bank_dir, image_size=(T, T, T), context_size=1,
                              crop_spacing_mm=3.0, classes=[CLS], maxid=256)
    return SynthGmmProvider(ds, cascade=True, p_shape=p_shape,
                            shape_spec=shape_spec or ShapeCohortSpec())


def test_assemble_task_shape_mode_returns_a_shape_pseudo_class(tmp_path):
    provider = _make_shape_provider(tmp_path, p_shape=1.0)
    rng = random.Random(0)
    task = provider.assemble_task(rng, crop_spacing_mm=3.0)
    ncs = task["native_crop"]
    assert len(ncs) == 2                      # context_size=1 -> target + 1 context
    for nc in ncs:
        assert nc.class_idx in SHAPE_ID_TO_FAMILY
        assert nc.has_fg                      # the shape was actually stamped and survived paint
    assert task["label_name"].startswith("shape_")


def test_assemble_task_p_shape_zero_never_returns_a_shape_class(tmp_path):
    provider = _make_shape_provider(tmp_path, p_shape=0.0)
    rng = random.Random(0)
    for _ in range(10):
        task = provider.assemble_task(rng, crop_spacing_mm=3.0)
        for nc in task["native_crop"]:
            assert nc.class_idx not in SHAPE_ID_TO_FAMILY


def test_load_native_crop_reproduces_the_same_shape_as_assemble_task(tmp_path):
    """Cascade re-crop consistency: re-deriving from the subject string alone must
    reproduce the identical shape mask assemble_task originally built."""
    provider = _make_shape_provider(tmp_path, p_shape=1.0)
    rng = random.Random(0)
    task = provider.assemble_task(rng, crop_spacing_mm=3.0)
    original = task["native_crop"][0]

    req = LoadRequest(rng=random.Random(0), crop_spacing_mm=3.0, center=None,
                      center_mode="com")
    rebuilt = provider.load_native_crop(task["subject"], task["label_name"], req)

    assert rebuilt.class_idx == original.class_idx
    np.testing.assert_array_equal(rebuilt.label_frac.numpy(), original.label_frac.numpy())


def test_shape_mode_subject_string_carries_the_host_class_as_a_fourth_field(tmp_path):
    provider = _make_shape_provider(tmp_path, p_shape=1.0)
    rng = random.Random(0)
    task = provider.assemble_task(rng, crop_spacing_mm=3.0)
    assert task["subject"].count("|") == 3       # filename|gmm_seed|member_idx|host<id>
    assert task["subject"].split("|")[3].startswith("host")


def test_between_ratio_zero_still_paints_a_nonempty_shape_in_every_member(tmp_path):
    """shape/size/position_between_ratio=0 pins the DRAWN PARAMETERS identical across
    the cohort (already exercised precisely, at the pure-function level, by
    src/shapes3d/test_instantiate.py::test_between_ratio_zero_reuses_the_cohort_value_exactly).
    Here we only need the provider-level plumbing check: with those params pinned, every
    member still ends up with a real, non-empty painted shape (has_fg) -- i.e. the pinned
    values survive _build_nc's crop/rasterize/paint pipeline instead of e.g. silently
    landing outside every member's own crop bounds."""
    spec = ShapeCohortSpec(shape_between_ratio=0.0, size_between_ratio=0.0,
                          position_between_ratio=0.0)
    provider = _make_shape_provider(tmp_path, p_shape=1.0, shape_spec=spec)
    rng = random.Random(1)
    task = provider.assemble_task(rng, crop_spacing_mm=3.0)
    for nc in task["native_crop"]:
        assert nc.has_fg
```

Add `import pytest` to the test file's existing imports (`numpy` and `random` are
already imported, per the file's current header).

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/dpxuser/dev/patch_icl && python -m pytest src/providers/test_synth_gmm.py -v -k shape`
Expected: `TypeError: SynthGmmProvider.__init__() got an unexpected keyword argument 'p_shape'`

- [ ] **Step 3: Implement the provider changes**

In `src/providers/synth_gmm.py`, update the imports and `SynthGmmProvider.__init__`:

```python
from data.maisi_classes import MAISI_CLASS_TO_IDX, MAISI_IDX_TO_CLASS, SHAPE_ID_TO_FAMILY
from src.gpu_gmm_intensity import sample_grouped_uniform
from src.providers.totalseg import _resolve_center
from src.shapes3d.instantiate import draw_cohort_hyperparams, draw_member_shape, rasterize_shape_in_crop
from src.shapes3d.spec import ShapeCohortSpec
from src.totalseg_dataloader_incontext import organ_crop_arrays
```

```python
    def __init__(self, dataset, *, cascade=False, p_shape=0.0, shape_spec=None):
        self.ds = dataset
        self.epoch_length = len(dataset)
        self.classes = [MAISI_IDX_TO_CLASS.get(c, str(c)) for c in dataset.cs.classes]
        self.cascade = cascade
        self.p_shape = float(p_shape)
        self.shape_spec = shape_spec or ShapeCohortSpec()
        if cascade:
            self._entry_by_file = {e["file"]: e for e in dataset.cs.entries}
            from src.providers.totalseg import NativeCrop
            from src.totalseg_dataset import CtNormSpec
            self._SYNTH_NORM = CtNormSpec(clip_lo=-3.0, clip_hi=3.0, mean=0.0, std=1.0)
            self._NativeCrop = NativeCrop
```

Update `_build_nc`'s signature and body (the diff below shows the full new function —
only the parts marked `# NEW` change; everything else is unchanged from the current
file):

```python
    def _build_nc(self, e, cls_id, rng, crop_mm, mu, sd, gmm_seed, member_idx, *,
                  center=None, jitter=None, center_mode="com",
                  shape_hp=None, shape_id=None):  # NEW
        """Crop + paint one MAISI bank entry -> NativeCrop. cascade=True required.

        Shape mode (shape_hp/shape_id both set): `cls_id` still drives crop placement
        against `e`'s real anatomy (the host); the painted/returned class becomes
        `shape_id` instead."""  # NEW docstring, extends the original one-liner
        member_nrng = np.random.default_rng([int(gmm_seed), int(member_idx)])
        n = self.ds.maxid + 1
        mu_e = (mu + self.ds.between_ratio * sd * member_nrng.standard_normal(n).astype(np.float32)
                if self.ds.between_ratio is not None else mu)

        arr = np.squeeze(np.load(self.ds.cs.dir / "masks" / e["file"], mmap_mode="r"))
        cents = e["cents"].get(cls_id)
        fallback = tuple(cents[:3]) if cents is not None else None
        fg_samples = e.get("fg_samples", {}).get(cls_id)
        center = _resolve_center(SimpleNamespace(center=center, center_mode=center_mode, rng=rng),
                                  arr, cls_id, fallback, fg_samples=fg_samples)
        if jitter is None:
            jitter = self.ds.jitter
        _, crop_lbl, out_sizes, pad_lo, geom = organ_crop_arrays(
            arr, arr, center, list(e["spacing"]),
            image_size=(self.ds.T,) * 3, crop_mm=crop_mm, jitter=jitter, rng=rng)
        cap = self.ds.gpu_realize_max_native
        if cap and max(crop_lbl.shape) > cap:
            step = tuple(-(-s // cap) for s in crop_lbl.shape)
            crop_lbl = crop_lbl[::step[0], ::step[1], ::step[2]]
        crop_lbl = np.ascontiguousarray(crop_lbl, dtype=np.uint8)

        target_cls = cls_id                                            # NEW
        if shape_hp is not None:                                       # NEW
            member_draw = draw_member_shape(member_nrng, shape_hp, self.shape_spec)  # NEW
            rasterize_shape_in_crop(crop_lbl, cls_id, shape_id, member_draw, member_nrng)  # NEW
            target_cls = shape_id                                      # NEW
            if self.shape_spec.intensity_between_ratio is not None:    # NEW
                fresh_noise = member_nrng.standard_normal()             # NEW
                mu_e[shape_id] = (mu[shape_id] + self.shape_spec.intensity_between_ratio  # NEW
                                  * sd[shape_id] * fresh_noise)          # NEW

        img, mask = self.ds._resample_paint_mask(
            crop_lbl, out_sizes, pad_lo, target_cls, mu_e, sd, member_nrng)  # target_cls, was cls_id

        T = self.ds.T
        return self._NativeCrop(
            image=img[0].half(),
            label_frac=mask.float().half(),
            class_idx=target_cls,                                      # target_cls, was cls_id
            has_fg=bool(mask.any()),
            out_sizes=[T, T, T],
            pad_lo=[0, 0, 0],
            crop_geom=geom,
            crop_spacing_mm=float(crop_mm),
            decim=(1, 1, 1),
            modality="synth",
            norm=self._SYNTH_NORM,
        )
```

Update `assemble_task`:

```python
    def assemble_task(self, rng, crop_spacing_mm):
        """Engine cohort hook: build one in-context item from the engine's per-item RNG."""
        if not self.cascade:
            nrng = np.random.default_rng(rng.getrandbits(64))
            return self.ds.assemble(rng, nrng, float(crop_spacing_mm))

        gmm_seed = rng.getrandbits(64)
        mu, sd = self._draw_gmm(gmm_seed)
        host_cls_id, cohort = self.ds.cs.sample_cohort(rng)

        shape_hp, shape_id = None, None                                # NEW
        if self.p_shape > 0.0 and rng.random() < self.p_shape:         # NEW
            shape_hp = draw_cohort_hyperparams(                        # NEW
                np.random.default_rng([int(gmm_seed), -1]), self.shape_spec)  # NEW
            family_by_shape_name = {v: k for k, v in SHAPE_ID_TO_FAMILY.items()}  # NEW
            shape_id = family_by_shape_name[shape_hp.family]            # NEW

        ncs = [self._build_nc(e, host_cls_id, rng, float(crop_spacing_mm), mu, sd, gmm_seed, i,
                              shape_hp=shape_hp, shape_id=shape_id)      # shape_hp/shape_id NEW
               for i, e in enumerate(cohort)]

        if shape_hp is not None:                                        # NEW
            name = MAISI_IDX_TO_CLASS[shape_id]                          # NEW
            host_suffix = f"|host{host_cls_id}"                          # NEW
        else:
            name = MAISI_IDX_TO_CLASS.get(host_cls_id, str(host_cls_id))
            host_suffix = ""                                             # NEW

        return {
            "native_crop": ncs,
            "subject": f"{cohort[0]['file']}|{gmm_seed}|0{host_suffix}",           # host_suffix NEW
            "context_subjects": [f"{e['file']}|{gmm_seed}|{i + 1}{host_suffix}"    # host_suffix NEW
                                  for i, e in enumerate(cohort[1:])],
            "label_name": name,
            "aug_mode": torch.tensor(0, dtype=torch.long),
            "tgt_modality": "synth",
            "ctx_modality": "synth",
        }
```

Update `load_native_crop`:

```python
    def load_native_crop(self, subject, cls, req):
        """Cascade re-crop: re-derive same GMM + member paint nrng from subject string."""
        if not self.cascade:
            raise RuntimeError("SynthGmmProvider.load_native_crop requires cascade=True")
        parts = subject.split("|")                                      # NEW (was rsplit("|", 2))
        if len(parts) == 4:                                              # NEW
            filename, gmm_seed_str, member_idx_str, host_str = parts     # NEW
            host_cls_id = int(host_str[len("host"):])                    # NEW
        else:                                                            # NEW
            filename, gmm_seed_str, member_idx_str = parts               # NEW
            host_cls_id = None                                           # NEW
        gmm_seed = int(gmm_seed_str)
        member_idx = int(member_idx_str)
        mu, sd = self._draw_gmm(gmm_seed)
        e = self._entry_by_file[filename]
        cls_id = MAISI_CLASS_TO_IDX.get(cls)
        if cls_id is None:
            try:
                cls_id = int(cls)
            except (ValueError, TypeError):
                raise ValueError(f"SynthGmmProvider.load_native_crop: unknown class {cls!r}")

        shape_hp, shape_id = None, None                                  # NEW
        if host_cls_id is not None:                                      # NEW
            shape_hp = draw_cohort_hyperparams(                          # NEW
                np.random.default_rng([int(gmm_seed), -1]), self.shape_spec)  # NEW
            shape_id = cls_id                                            # NEW
            cls_id = host_cls_id       # resolve center/crop against the real host anchor  # NEW

        # no jitter for cascade recrops (center is predicted, not default centroid)
        jitter = 0 if req.center is not None else self.ds.jitter
        return self._build_nc(e, cls_id, req.rng, req.crop_spacing_mm, mu, sd,
                              gmm_seed, member_idx, center=req.center, jitter=jitter,
                              center_mode=getattr(req, "center_mode", "com"),
                              shape_hp=shape_hp, shape_id=shape_id)        # shape_hp/shape_id NEW
```

Also update the module docstring at the top of `src/providers/synth_gmm.py` (currently
lines 1-14) to mention the new mode — append a paragraph:

```python
Shape mode (p_shape > 0, cascade=True only): a `p_shape` coin flip per cohort swaps the
sampled cohort's target for a procedurally generated geometric shape (blob/splatter/
disk/cylinder, see src/shapes3d/) stamped into the sampled hosts' real anatomy under a
new pseudo-class id (data.maisi_classes.SHAPE_ID_TO_FAMILY). The host's real class
still drives crop placement (center resolution against real anatomy); only the painted
target and its label change. See docs/superpowers/specs/
2026-09-14-cohort-consistent-synthetic-shapes-design.md.
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/dpxuser/dev/patch_icl && python -m pytest src/providers/test_synth_gmm.py -v`
Expected: all PASS, including every pre-existing test in the file (the constructor
change is backward compatible: `p_shape` and `shape_spec` both default such that
existing call sites `SynthGmmProvider(ds, cascade=True)` are unaffected).

- [ ] **Step 5: Commit**

```bash
git add src/providers/synth_gmm.py src/providers/test_synth_gmm.py
git commit -m "feat(synth_gmm): p_shape cohort mode stamping procedural shapes into hosts

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01AfbPWj3NvbKfBFCN2CKGZ6"
```

---

### Task 5: Log the change

**Files:**
- Modify: `docs/logs.md`

Per this repo's CLAUDE.md convention ("Log changes to docs/logs.md"), append an entry
in the file's existing `## YYYY-MM-DD — title` prose style.

- [ ] **Step 1: Append the log entry**

Add to the end of `docs/logs.md`:

```markdown

## 2026-09-14 — synth_gmm shape-mode cohorts (procedural blob/splatter/disk/cylinder)

New `src/shapes3d/` package (`primitives.py`, `spec.py`, `instantiate.py`) generates
procedurally-defined 3D shapes -- not anatomically plausible by design -- for a new
`SynthGmmProvider` cohort mode (`data.gmm.p_shape`, default `0.0`, off unless enabled).
When it fires, a shape is stamped into each cohort member's independently-sampled real
gmm_bank host organ crop under a new pseudo-class id (`data/maisi_classes.py`, ids
195-198, `SHAPE_ID_TO_FAMILY`), loosely positioned inside the host (no reject-until-
valid containment). Four independent `*_between_ratio` knobs (shape/size/position/
intensity, `ShapeCohortSpec`) control how identical vs. independent each axis is across
the K+1 cohort members -- generalizing the cohort-shared-mu/sd-plus-per-member-
between_ratio pattern `_draw_gmm`/`_build_nc` already used for intensity to shape, size,
and position too. Cascade re-crop consistency reuses the existing `gmm_seed`/
`member_idx`-keyed RNG scheme with one added subject-string field (the host's real
class id, needed to re-resolve the shape's position at any re-crop level). Motivated by
OOD eval (tumors/sub-regions) tending to fall back to known-organ shapes -- see
docs/superpowers/specs/2026-09-14-cohort-consistent-synthetic-shapes-design.md for the
full design and the NV-Generate-CTMR/controlSynth/omniSynth prior art it builds on.
Not yet wired into any training config or calibrated against eval -- that's the
deliberately deferred next step (see spec sec 7).
```

- [ ] **Step 2: Commit**

```bash
git add docs/logs.md
git commit -m "docs: log synth_gmm shape-mode cohorts addition

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01AfbPWj3NvbKfBFCN2CKGZ6"
```

---

## Explicitly out of scope for this plan (see spec sec 7)

- Wiring `p_shape > 0` into any actual training config (`92_multisource_synth` or a new
  experiment file) — this plan only makes the capability available and off by default.
- Calibrating whether shape-mode cohorts measurably improve OOD (tumor/sub-region) eval.
- Tuning the default parameter ranges in `ShapeCohortSpec` against visual inspection or
  a difficulty study (controlSynth-style) — the ranges here are reasonable starting
  points, not calibrated.
- Any curriculum/scheduling of `p_shape` or the `*_between_ratio` knobs over training.
