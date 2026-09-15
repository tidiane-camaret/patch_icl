"""Config surface for synth_gmm's shape-mode cohorts. See
docs/superpowers/specs/2026-09-14-cohort-consistent-synthetic-shapes-design.md and
docs/logs.md 2026-09-15 (world-space redesign)."""

from dataclasses import dataclass, field


@dataclass
class ShapeCohortSpec:
    """Every *_between_ratio is in [0,1]: 0 = every cohort member reuses the exact
    cohort-drawn value, 1 = every member redraws independently within the ranges below.
    `intensity_between_ratio=None` (default) means the shape's pseudo-class uses
    whatever the provider's own gmm.between_ratio setting already gives every other
    class -- set it to override just the shape's intensity consistency independently
    of that global toggle.

    Size and position are PHYSICAL (mm), not crop-relative: `size_mm_range` draws a
    target diameter, `position_offset_mm_range` draws a per-axis mm offset from the
    host's own centroid. Both are resolved into the CURRENT crop's local voxel units
    fresh at every cascade re-crop level (see instantiate.rasterize_shape_in_crop), so
    the same member draw is the SAME physical object at any level, however that
    level's crop_spacing_mm differs -- this replaced an earlier crop-relative
    (size_frac/position_uvw) design that was NOT consistent across cascade levels."""
    family_weights: dict = field(default_factory=lambda: {
        "blob": 1.0, "splatter": 1.0, "disk": 1.0, "cylinder": 1.0})
    size_mm_range: tuple = (10.0, 50.0)               # target diameter, mm
    position_offset_mm_range: tuple = (-30.0, 30.0)   # per-axis offset from host centroid, mm
    shape_between_ratio: float = 0.3
    size_between_ratio: float = 0.3
    position_between_ratio: float = 0.5
    intensity_between_ratio: float | None = None

    blob_roughness_range: tuple = (0.05, 0.3)
    splatter_n_components_range: tuple = (2, 6)
    splatter_spread_mm_range: tuple = (5.0, 20.0)
    splatter_roughness_range: tuple = (0.1, 0.2)
    disk_aspect_ratio_range: tuple = (0.15, 0.4)
    cylinder_length_mm_range: tuple = (30.0, 100.0)
