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

    # host_anchored: False (default) = today's behavior, the shape's mu is an INDEPENDENT
    # draw from the same U(0,255) domain as every other class -- confirmed by
    # experiments/3d/synth_task_generation/plot_shape_intensity.py to be visually and
    # numerically unrelated to the host organ it's stamped inside (docs/logs.md 2026-09-23
    # "shape intensity vs host organ"). True = the shape's mu is drawn RELATIVE to its host
    # class's own mu, SCALED BY THE HOST'S OWN sd (mu[host_cls_id] + U(*host_contrast_ratio_range)
    # * sd[host_cls_id], clipped to [0,255]) -- a ratio, not an absolute offset, matching how
    # sd_between_ratio/intensity_between_ratio already scale by sd rather than using a flat
    # constant (an earlier absolute-offset version, (-60,60) mu-units, was ~7-12x too extreme
    # relative to sd's own ~[0,9] range at var_max=80 -- caught by calibrating against real
    # data, see below). Drawn ONCE per cohort (same seed-derived rng pattern as shape_hp
    # itself), so target+context share the same host-relative contrast; intensity_between_ratio
    # (above) still layers its own per-member jitter on top of whichever value (anchored or
    # not) ends up in mu[shape_id], unchanged.
    #
    # host_contrast_ratio_range calibrated from REAL target-vs-immediately-surrounding-tissue
    # contrast (docs/logs.md 2026-09-23 "target-surround contrast calibration",
    # experiments/3d/synth_task_generation/analyze_target_surround_contrast.py), measured as
    # (target_mean - ring_mean)/ring_std (a ring-std-normalized effect size, comparable across
    # CT/MRI) on the 7 integrated OOD sources, 749 cases: per-source medians range +2.38
    # (isles22 stroke lesion, DWI -- classically hyperintense) to -0.69 (atlas_v2 chronic
    # stroke lesion, T1 -- classically hypointense) to ~0 (hu_lwk1 -- an internal vertebra ROI,
    # not a real lesion boundary); pooled [p10,p90]=[-0.87,2.00]. (-2.5,2.5) covers this
    # measured range with margin on both signs (real lesion types are usually consistently
    # hyper- OR hypo-intense, not symmetric per-instance, but which sign applies to an
    # arbitrary synthetic host class isn't knowable in advance, so both signs are drawn with
    # equal probability here -- a coarser approximation than per-source-calibrated signed
    # ranges, deferred).
    host_anchored: bool = False
    host_contrast_ratio_range: tuple = (-2.5, 2.5)   # multiplies sd[host_cls_id]

    blob_roughness_range: tuple = (0.05, 0.3)
    splatter_n_components_range: tuple = (2, 6)
    splatter_spread_mm_range: tuple = (5.0, 20.0)
    splatter_roughness_range: tuple = (0.1, 0.2)
    disk_aspect_ratio_range: tuple = (0.15, 0.4)
    cylinder_length_mm_range: tuple = (30.0, 100.0)

    # scatter_field: many-instance analog of splatter, added 2026-09-27 to close a
    # measured gap -- real multi-focal lesion fields (Shifts-MS MS plaques, ISLES22
    # multi-focal emboli) carry 10-100+ components/subject and span 20-150mm, far
    # beyond splatter's (2-6, 5-20mm) design point (docs/logs.md 2026-09-26). Kept as
    # its own family rather than widening splatter in place, so existing splatter-tuned
    # configs/tests are unaffected.
    scatter_field_n_components_range: tuple = (10, 120)
    scatter_field_spread_mm_range: tuple = (20.0, 140.0)
    scatter_field_roughness_range: tuple = (0.1, 0.3)

    # vessel: thin branching tubular network (cheap L-system-like tapering capsule
    # tree, see shapes3d/primitives.py::_vessel_segments) -- a vein/vessel stand-in
    # targeting the repo's own "thickness, not identity" OOD driver finding (thin
    # structures are the model's known weak point) with genuine branching topology
    # rather than another straight tube.
    vessel_trunk_length_mm_range: tuple = (40.0, 120.0)
    vessel_branch_depth_range: tuple = (2, 4)
    vessel_radius_falloff_range: tuple = (0.6, 0.85)

    # torus: non-convex, genus-1 ring -- deliberately non-anatomical (per spec: shapes
    # need not be plausible lesions), added per direct user request ("segments, torus,
    # etc.") for topological diversity cheap SDF families don't otherwise cover.
    # ratio = major/minor radius (>1 always; larger = thinner ring).
    torus_ratio_range: tuple = (2.0, 5.0)

    # family_host_classes: {family_name: (maisi_class_id, ...)} -- when a family has an
    # entry here, its cohort's host organ is restricted to a uniform random choice among
    # those ids (via CohortSampler.sample_cohort's target_class) instead of the default
    # uniform-over-every-class host. Added 2026-09-27 to test the sim-to-real gap: a
    # scattered lesion-field family stamped into a RANDOM organ (liver, lung, ...) never
    # specifically taught the model "find scattered small hyperintense regions WITHIN
    # BRAIN tissue", the actual real-world domain (ISLES22/Shifts-MS are brain-only)
    # (docs/logs.md 2026-09-27 "sim-to-real gap"). Families with no entry are unaffected.
    family_host_classes: dict = field(default_factory=dict)
