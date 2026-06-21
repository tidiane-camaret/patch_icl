"""
Configuration objects for controlSynth (spec ss5).

Three independent dataclasses keep the experimental axes separable in code:
  - DiversityConfig      -> number of distinct tasks (the "diversity" axis)
  - DifficultyBuildSpec  -> build-time difficulty (frozen into base geometry)
  - DifficultyLiveConfig -> live difficulty (applied per subject in the loader)
  - SamplingConfig       -> quantity (epoch_length) + determinism (train vs eval)

Every [0,1] factor has a documented MONOTONE mapping to its underlying generation
parameter, gathered here as `map_*` functions so calibration (spec ss12) tweaks a
single place. Midpoints/endpoints are anchored so defaults reproduce baseline
behavior (e.g. noise_level: 0 -> sigma=0.0, 1 -> sigma=0.25).
"""

from dataclasses import dataclass, field


# -- [0,1] -> generation-parameter mappings (monotone; hard direction = larger) --

def map_noise_sigma(level: float) -> float:
    """noise_level -> additive-noise std.  0 -> 0.0, 1 -> 0.25 (linear)."""
    return 0.25 * _clip01(level)


def map_region_size(frac: float) -> float:
    """region_size -> target foreground area fraction, log-mapped.

    0 -> ~0.3% area, 1 -> ~30% area. Small regions (hard direction) get fine
    resolution because Dice variance explodes there (spec ss11).
    """
    import math
    lo, hi = math.log(0.003), math.log(0.30)
    return math.exp(lo + _clip01(frac) * (hi - lo))


def map_caliber_px(thinness: float, image_size: int) -> float:
    """thinness -> min (leaf) caliber radius, px, of tubular structures.

    0 (thick) -> image_size/42 (~3px @128), 1 (thin) -> image_size/210 (~0.6px @128).
    Hard direction = thinner. Kept small so trees read as vessels, not blobs.
    """
    thick = image_size / 42.0
    thin = image_size / 210.0
    return thick - _clip01(thinness) * (thick - thin)


def map_boundary_roughness(c: float) -> float:
    """boundary_complexity -> radial-perturbation amplitude (fraction of radius)."""
    return 0.35 * _clip01(c)


def map_deform_sigma(shift: float, image_size: int) -> float:
    """support_query_shift -> elastic displacement magnitude (px).

    0 -> 0 px, 1 -> image_size * 0.10 px peak displacement.
    """
    return _clip01(shift) * image_size * 0.10


def map_contrast_gap(contrast: float) -> float:
    """foreground_contrast -> distance of the fg mean from the background cluster
    centre (0.5), in intensity units (regions live on ~[0,1]).  0 -> 0.05 (fg sits
    inside the background cluster, intensity-invisible), 1 -> 0.45 (fg near the [0,1]
    edge but clear of the background band [0.25,0.75], so it stays separable rather
    than saturating into the background's noise tail).  Low contrast is the hard
    direction.  Used by appearance.gmm_fill, which keeps the background in a central
    band so the fg owns the extremes (see that docstring)."""
    return 0.05 + _clip01(contrast) * 0.40


def map_texture_std(t: float) -> float:
    """texture_heterogeneity -> within-region intensity std.  0 -> 0.0, 1 -> 0.2."""
    return 0.2 * _clip01(t)


def map_ambiguity_n_distractors(ambiguity: float, num_labels: int) -> int:
    """task_ambiguity (geometry side) -> how many of the num_labels background
    regions are made to share the foreground's shape/attribute. Monotone in
    ambiguity; capped at num_labels."""
    return int(round(_clip01(ambiguity) * num_labels))


def _clip01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else float(x)


# -- Config dataclasses (spec ss5) ----------------------------------------------

@dataclass
class DiversityConfig:
    num_tasks: int = 1000
    num_labels: int = 16          # distractor/background regions behind the foreground
    context_size: int = 3
    master_seed: int = 42
    splits: tuple = (0.8, 0.1, 0.1)  # train / val / test task pools


@dataclass
class DifficultyBuildSpec:
    """Build-time difficulty (frozen into base geometry at init)."""
    mode: str = "fixed"               # "fixed" | "per_task_sampled" (binned: deferred)
    morphology: object = "blob"       # type str, or {type: weight} mixture
    thinness: float = 0.5             # [0,1] -> min caliber (tubular)
    tortuosity: float = 0.5           # [0,1] -> angular perturbation (tubular)
    branching_density: float = 0.5    # [0,1] -> p_branch x max_depth (tubular)
    region_size: float = 0.15         # [0,1] -> foreground area fraction (log-mapped)
    boundary_complexity: float = 0.3  # [0,1] contour roughness
    scattered_count: int = 8          # n components (scattered)
    scattered_clustering: float = 0.0 # [0,1] Poisson(0) -> clustered(1)
    task_ambiguity: float = 0.0       # [0,1] geometry side: n + similarity of distractors
    # Realism knobs (backward-compatible: defaults reproduce the original centered,
    # high-fragmentation behavior). See shapes/boundary.set_boundary_complexity and
    # task.place_foreground; tuned to match biomedparse mask stats in hard_diverse.
    boundary_amp_scale: float = 1.0   # <1 -> gentler roughening (less shattering)
    boundary_sigma_frac: float = 0.0  # >0 -> roughness blur = sqrt(area)*frac (low-freq); 0 = fixed sigma
    boundary_keep_largest: bool = False  # keep largest CC after roughening (blob/elongated/annular only)
    position_jitter: float = 0.0      # std of normalized centroid placement; 0 = centered (original)
    # per_task_sampled: {factor: [lo, hi]} ranges each task draws from (others stay fixed).
    sampled: dict = field(default_factory=dict)
    # eval-grid binning of `bin_factor` (used by the val sample names -> difficulty curve).
    n_bins: int = 1
    bin_factor: str = "task_ambiguity"


@dataclass
class DifficultyLiveConfig:
    """Live difficulty (applied per subject in the dataloader)."""
    support_query_shift: float = 0.3
    # Per-SUBJECT pose jitter -> within-(target+context)-set position/size spread.
    # support_query_shift only deforms one shared base (boundary wobble); real sets vary
    # far more in location/size across instances. 0 -> off (original). See dataset.jitter_pose
    # and scripts/context_mask_distance.py (real targets: centroid_dist 0.16, area_logratio 1.2).
    support_query_translate: float = 0.0  # std of per-subject centroid shift (fraction of image)
    support_query_scale: float = 0.0      # std of per-subject log2 zoom factor
    foreground_contrast: float = 0.5
    texture_heterogeneity: float = 0.2
    noise_level: float = 0.3
    context_copy_fraction: float = 0.0   # fraction of contexts that are pristine (low-shift) exemplars -> easier
    context_consistency: float = 1.0
    task_ambiguity_intensity: float = 0.0  # live side of ambiguity


@dataclass
class SamplingConfig:
    epoch_length: int = 10000
    deterministic: bool = False        # False=train (infinite subjects), True=eval
    eval_seed_namespace: int = 0
    eval_subjects_per_task: int = 4    # val grid: subjects sampled per held-out task
