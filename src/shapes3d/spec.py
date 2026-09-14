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
    of that global toggle.

    KNOWN LIMITATION: `size_frac_range` and every position/shape param below are
    resolved relative to the CROP GRID a shape is rasterized into (see
    rasterize_shape_in_crop), not in world/mm space -- unlike mu/sd (physically
    resolution-independent scalars), a shape's absolute physical size/position is NOT
    guaranteed consistent across cascade levels that use a different crop_spacing_mm.
    Safe for single-level (or single-crop_spacing_mm) use; needs a world-space
    redesign before combining shape mode with multi-level cascade training."""
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
