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
