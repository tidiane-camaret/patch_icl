"""Per-axis cohort-consistency instantiation for shape-mode cohorts: a cohort-level
draw (shared, like synth_gmm's mu/sd) blended toward each member's independent draw by
that axis's *_between_ratio (like synth_gmm's mu_e). See
docs/superpowers/specs/2026-09-14-cohort-consistent-synthetic-shapes-design.md.

Size and position are drawn in PHYSICAL units (mm) so the same member_draw resolves to
the same physical object at every cascade level -- see rasterize_shape_in_crop and
docs/logs.md 2026-09-15."""

from dataclasses import dataclass, field

import numpy as np

from .primitives import make_shape, reach_vox

_EPS = 1e-9

# (family, param key) -> ShapeCohortSpec attribute holding that param's [lo, hi] prior
# range -- used to draw both the cohort baseline and each member's fresh redraw.
_CONTINUOUS_RANGE = {
    ("blob", "roughness"): "blob_roughness_range",
    ("splatter", "spread_mm"): "splatter_spread_mm_range",
    ("splatter", "roughness"): "splatter_roughness_range",
    ("disk", "aspect_ratio"): "disk_aspect_ratio_range",
    ("cylinder", "length_mm"): "cylinder_length_mm_range",
}


@dataclass
class CohortHyperparams:
    family: str
    size_mm: float
    position_offset_mm: np.ndarray      # (3,) mm offset from the host's centroid, cohort baseline
    shape_params: dict = field(default_factory=dict)


@dataclass
class MemberShapeDraw:
    family: str
    size_mm: float
    position_offset_mm: np.ndarray
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

    size_mm = float(rng.uniform(*spec.size_mm_range))
    position_offset_mm = rng.uniform(*spec.position_offset_mm_range, size=3)

    if family == "blob":
        shape_params = {"roughness": float(rng.uniform(*spec.blob_roughness_range))}
    elif family == "splatter":
        lo, hi = spec.splatter_n_components_range
        shape_params = {
            "n_components": int(rng.integers(int(lo), int(hi) + 1)),
            "spread_mm": float(rng.uniform(*spec.splatter_spread_mm_range)),
            "roughness": float(rng.uniform(*spec.splatter_roughness_range)),
        }
    elif family == "disk":
        shape_params = {
            "aspect_ratio": float(rng.uniform(*spec.disk_aspect_ratio_range)),
            "flatten_axis": int(rng.integers(0, 3)),
        }
    elif family == "cylinder":
        shape_params = {
            "length_mm": float(rng.uniform(*spec.cylinder_length_mm_range)),
            "azimuth": float(rng.uniform(0, 2 * np.pi)),
            "elevation": float(rng.uniform(-np.pi / 2, np.pi / 2)),
        }
    else:
        raise ValueError(f"unknown shape family {family!r}")

    return CohortHyperparams(family=family, size_mm=size_mm,
                             position_offset_mm=position_offset_mm, shape_params=shape_params)


def draw_member_shape(member_rng, cohort_hp, spec):
    """Once per cohort member (caller keys `member_rng` off (gmm_seed, member_idx), the
    same stream synth_gmm's own per-member intensity draw already uses)."""
    family = cohort_hp.family

    fresh_size = float(member_rng.uniform(*spec.size_mm_range))
    size_mm = _blend(cohort_hp.size_mm, fresh_size, spec.size_between_ratio)

    fresh_offset = member_rng.uniform(*spec.position_offset_mm_range, size=3)
    position_offset_mm = _blend(cohort_hp.position_offset_mm, fresh_offset,
                                spec.position_between_ratio)

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

    if family == "disk":
        # discrete param: the natural analog of "blend" is a probabilistic redraw
        # (shape_between_ratio=0 -> never redraw, =1 -> always redraw) -- a linear
        # blend doesn't make sense for a categorical axis choice. Drawn unconditionally
        # (not just when the redraw fires) so RNG consumption is deterministic.
        redraw = member_rng.random() < spec.shape_between_ratio
        fresh_axis = int(member_rng.integers(0, 3))
        if redraw:
            shape_params["flatten_axis"] = fresh_axis

    if family == "cylinder":
        fresh_az = float(member_rng.uniform(0, 2 * np.pi))
        shape_params["azimuth"] = float(
            shape_params["azimuth"]
            + spec.shape_between_ratio * _angle_diff(fresh_az, shape_params["azimuth"]))
        fresh_el = float(member_rng.uniform(-np.pi / 2, np.pi / 2))
        shape_params["elevation"] = _blend(shape_params["elevation"], fresh_el,
                                           spec.shape_between_ratio)

    return MemberShapeDraw(family=family, size_mm=size_mm,
                           position_offset_mm=position_offset_mm, shape_params=shape_params)


def rasterize_shape_in_crop(crop_lbl, shape_id, member_draw, mm_per_voxel, center_local, rng):
    """Stamp one member's shape into `crop_lbl` (uint8) under `shape_id`.

    `center_local`: the shape's center already resolved into crop_lbl's own local array
    coordinates by the caller (bank/crop-geometry glue -- mapping a fixed native
    coordinate through the crop's native-array origin and any cap-stride decimation --
    deliberately kept out of this pure module; see src/providers/synth_gmm.py).

    `mm_per_voxel`: this crop's CURRENT effective voxel size (a 3-tuple/array, e.g. the
    host's native spacing times any cap-stride decimation). Converts
    `member_draw.size_mm` (and, for cylinder, `shape_params["length_mm"]`; for
    splatter, `shape_params["spread_mm"]`) from physical mm into this crop's local
    voxel units. Because these physical mm values are resolved fresh from whatever
    `mm_per_voxel` is current, the SAME member_draw produces a shape of the SAME
    PHYSICAL size and position at any cascade level, however that level's crop
    resolution differs -- this is what makes shape-mode cohorts cascade-consistent
    (replacing an earlier crop-relative size_frac/position_uvw design that was not; see
    docs/logs.md 2026-09-15).

    Rasterizes into a local sub-box around `center_local` sized by
    `primitives.reach_vox` (not the full crop) for performance, then copies the result
    into `crop_lbl` at the right offset -- clipped to the crop's own array bounds
    (loose containment: never clipped against any organ boundary, only the crop's own
    bounds; if the shape's reach doesn't touch this crop at all, nothing is stamped).
    Mutates `crop_lbl` in place -- caller must ensure it is writeable (a zero-copy view
    of an mmap_mode="r" array is not)."""
    mm_per_voxel = np.asarray(mm_per_voxel, dtype=np.float64)
    vox_mm3 = float(np.prod(mm_per_voxel))
    mm_per_voxel_iso = vox_mm3 ** (1.0 / 3.0)   # isotropic approx (see spec docstring)

    target_mm3 = (4.0 / 3.0) * np.pi * (member_draw.size_mm / 2.0) ** 3
    size_vox = target_mm3 / max(vox_mm3, _EPS)

    params = dict(member_draw.shape_params)
    params["size_vox"] = size_vox
    if member_draw.family == "cylinder":
        params["length_vox"] = params.pop("length_mm") / mm_per_voxel_iso
    if member_draw.family == "splatter":
        params["spread_vox"] = params.pop("spread_mm") / mm_per_voxel_iso

    crop_shape = np.array(crop_lbl.shape, dtype=np.int64)
    center_local = np.asarray(center_local, dtype=np.float64)
    reach = reach_vox(member_draw.family, params) * 1.2   # small extra margin
    lo = np.clip(np.floor(center_local - reach).astype(np.int64), 0, crop_shape)
    hi = np.clip(np.ceil(center_local + reach).astype(np.int64) + 1, 0, crop_shape)
    sub_shape = tuple((hi - lo).tolist())
    if any(s <= 0 for s in sub_shape):
        return   # shape's reach doesn't touch this crop at all -- nothing to stamp
    sub_center = tuple((center_local - lo).tolist())

    mask, _meta = make_shape(member_draw.family, sub_shape, sub_center, params, rng)
    sl = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    sub_view = crop_lbl[sl]
    sub_view[mask.astype(bool)] = shape_id
