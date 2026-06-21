"""
Task definition and base-geometry compositing (build-time).

A *task* is (task_seed, fg_label, build_difficulty); its base geometry is
deterministic in (master_seed, task_id) and identical whether produced here or by
an offline precompute (spec ss6). `make_base_geometry` composites:

    foreground morphology -> enforce area -> roughen boundary
    -> Voronoi background (num_labels regions) -> inject shape-distractors
    -> paint foreground on top

and records realized difficulty + heuristic axis_loadings for analysis.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

from .config import DifficultyBuildSpec, map_region_size
from .shapes import make_foreground
from .shapes.area import enforce_area_fraction
from .shapes.boundary import set_boundary_complexity
from .shapes.distractors import inject_distractors


@dataclass
class Task:
    task_id: int
    fg_label: int
    morphology: str
    geo_params: dict
    label_map: np.ndarray            # uint8 [H,W]
    meta: dict = field(default_factory=dict)


def resolve_difficulty(spec: DifficultyBuildSpec, task_id: int, rng):
    """Return (morphology, geo_params) for one task.

    mode='fixed'            -> every task shares the spec's scalar difficulty.
    mode='per_task_sampled' -> factors listed in spec.sampled are drawn per task
                               from their [lo, hi] range (the rest stay fixed), so
                               one run spans a difficulty range. (binned: deferred.)
    """
    if spec.mode not in ("fixed", "per_task_sampled"):
        raise NotImplementedError(
            f"build mode {spec.mode!r} not in V1 (use 'fixed' or 'per_task_sampled'; "
            "'binned' is deferred to the eval-grid sub-project)")

    morphology = spec.morphology
    if isinstance(morphology, Mapping):              # {type: weight} mixture
        types = list(morphology)                     # works for dict and OmegaConf DictConfig
        weights = np.array([float(morphology[t]) for t in types], dtype=float)
        morphology = types[int(rng.choice(len(types), p=weights / weights.sum()))]

    geo_params = {
        "thinness": spec.thinness,
        "tortuosity": spec.tortuosity,
        "branching_density": spec.branching_density,
        "region_size": spec.region_size,
        "boundary_complexity": spec.boundary_complexity,
        "scattered_count": spec.scattered_count,
        "scattered_clustering": spec.scattered_clustering,
        "task_ambiguity": spec.task_ambiguity,
    }

    if spec.mode == "per_task_sampled":
        for factor, rng_range in dict(spec.sampled).items():
            lo, hi = float(rng_range[0]), float(rng_range[1])
            value = float(rng.uniform(lo, hi))
            if isinstance(geo_params.get(factor), int):   # e.g. scattered_count
                value = int(round(value))
            geo_params[factor] = value

    # Realism knobs are global build settings (not per-task difficulty factors);
    # pass them through geo_params so make_base_geometry can read them without a
    # signature change. Defaults reproduce the original behavior.
    geo_params["boundary_amp_scale"] = spec.boundary_amp_scale
    geo_params["boundary_sigma_frac"] = spec.boundary_sigma_frac
    geo_params["boundary_keep_largest"] = spec.boundary_keep_largest
    geo_params["position_jitter"] = spec.position_jitter

    return morphology, geo_params


def _voronoi_background(image_size, num_labels, rng):
    """Partition the frame into num_labels regions (labels 1..num_labels)."""
    seeds = rng.uniform(0, image_size, size=(num_labels, 2))
    yy, xx = np.mgrid[0:image_size, 0:image_size].astype(np.float64)
    d = ((yy[None] - seeds[:, 0, None, None]) ** 2 +
         (xx[None] - seeds[:, 1, None, None]) ** 2)
    return (d.argmin(axis=0) + 1).astype(np.uint8)    # 1..num_labels


def place_foreground(mask, jitter, rng, image_size):
    """Translate `mask` so its centroid lands at N(0.5, jitter) per axis.

    The blob/vessel/scattered generators all place the shape near the image centre
    (margins, area-zoom about the centroid), which makes synth masks far more
    centered than real ones (biomedparse centroid std ~0.14 vs ~0.09). This shifts
    the finished shape to a sampled centroid (zero-fill, so part may fall off-frame
    -> realistic ~6% border contact). jitter<=0 -> unchanged (original centered).
    """
    jitter = float(jitter)
    m = mask.astype(bool)
    if jitter <= 0.0 or not m.any():
        return mask.astype(np.uint8)
    ys, xs = np.nonzero(m)
    cy, cx = ys.mean(), xs.mean()
    H = W = image_size
    ty = float(np.clip(rng.normal(0.5, jitter), 0.05, 0.95)) * (H - 1)
    tx = float(np.clip(rng.normal(0.5, jitter), 0.05, 0.95)) * (W - 1)
    dy, dx = int(round(ty - cy)), int(round(tx - cx))
    # Clamp the shift so the shape's bounding box stays inside the frame (with a small
    # inset, since the per-subject deformation later nudges it by a few px). This keeps
    # the shape off-centre without slicing it at the border -- real masks touch the
    # frame edge only ~6% of the time; an unclamped shift made it ~48%.
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    inset = 2
    lo_y, hi_y = -y0 + inset, (H - 1) - y1 - inset    # bounds invert if shape ~ frame-sized
    lo_x, hi_x = -x0 + inset, (W - 1) - x1 - inset
    dy = int(np.clip(dy, lo_y, hi_y)) if lo_y <= hi_y else 0
    dx = int(np.clip(dx, lo_x, hi_x)) if lo_x <= hi_x else 0
    out = np.zeros_like(m)
    sy0, sy1 = max(0, dy), min(H, H + dy)          # dst rows
    sx0, sx1 = max(0, dx), min(W, W + dx)
    gy0, gy1 = max(0, -dy), min(H, H - dy)         # src rows
    gx0, gx1 = max(0, -dx), min(W, W - dx)
    out[sy0:sy1, sx0:sx1] = m[gy0:gy1, gx0:gx1]
    return out.astype(np.uint8)


def _axis_loadings(morphology, geo_params, n_distractors):
    """Heuristic [0,1] estimates of identification vs segmentation difficulty.

    Recorded only for analysis; the calibration protocol (spec ss12) measures the
    true loadings via the oracle-vs-in-context gap.
    """
    amb = float(geo_params["task_ambiguity"])
    ident = min(1.0, amb + (0.4 if morphology == "scattered" else 0.0))
    seg = float(np.clip(
        0.5 * (1.0 - map_region_size(geo_params["region_size"]) / 0.30)   # small=hard
        + 0.3 * geo_params["boundary_complexity"]
        + (0.3 if morphology in ("tubular", "scattered") else 0.0)
        + (0.3 * geo_params["thinness"] if morphology == "tubular" else 0.0),
        0.0, 1.0))
    return {"identification": ident, "segmentation": seg}


def make_base_geometry(image_size, morphology, geo_params, num_labels, rng):
    """Return (label_map uint8 [H,W], fg_label int, realized_meta dict)."""
    fg_mask, fg_meta = make_foreground(morphology, image_size, geo_params, rng)

    # Blob family scales via the generic zoom; vessel/scattered scale internally.
    if morphology in ("blob", "elongated", "annular"):
        fg_mask = enforce_area_fraction(fg_mask, map_region_size(geo_params["region_size"]))
    # keep_largest only for simply connected morphologies; scattered is multi-component
    # by design and tubular branches (a largest-CC would eat its thin branches).
    keep_largest = (bool(geo_params.get("boundary_keep_largest", False))
                    and morphology in ("blob", "elongated", "annular"))
    fg_mask = set_boundary_complexity(
        fg_mask, geo_params["boundary_complexity"], rng,
        amp_scale=geo_params.get("boundary_amp_scale", 1.0),
        sigma_frac=geo_params.get("boundary_sigma_frac", 0.0),
        keep_largest=keep_largest)

    if fg_mask.sum() == 0:                            # never emit an empty foreground
        fg_mask[image_size // 2 - 3:image_size // 2 + 3,
                image_size // 2 - 3:image_size // 2 + 3] = 1

    # Realistic placement: shift the finished shape off-centre (after the empty guard,
    # so we never translate a fallback patch out of frame).
    fg_mask = place_foreground(fg_mask, geo_params.get("position_jitter", 0.0),
                               rng, image_size)

    label_map = _voronoi_background(image_size, num_labels, rng)

    distractor_start = num_labels + 1
    distractor_labels = inject_distractors(
        label_map, fg_mask, geo_params["task_ambiguity"], num_labels,
        distractor_start, rng)

    fg_label = distractor_start + num_labels         # stable, above all distractors
    label_map[fg_mask > 0] = fg_label                # foreground painted last (on top)

    meta = {
        "morphology": morphology,
        "geo_params": dict(geo_params),
        "n_distractors": len(distractor_labels),
        "distractor_labels": list(map(int, distractor_labels)),
        "realized_area": float((label_map == fg_label).sum()) / label_map.size,
        # Task-level fg intensity side, shared across the task's subjects so the
        # foreground has a consistent appearance for context-matching (see gmm_fill).
        "appearance_sign": int(rng.choice([-1, 1])),
        "axis_loadings": _axis_loadings(morphology, geo_params, len(distractor_labels)),
        **{f"fg_{k}": v for k, v in fg_meta.items() if k != "morphology"},
    }
    return label_map.astype(np.uint8), int(fg_label), meta
