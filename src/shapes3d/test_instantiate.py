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
    assert hp1.size_mm == hp2.size_mm
    np.testing.assert_array_equal(hp1.position_offset_mm, hp2.position_offset_mm)
    assert hp1.shape_params == hp2.shape_params


def test_draw_cohort_hyperparams_family_always_one_of_the_four():
    spec = ShapeCohortSpec()
    for seed in range(20):
        hp = draw_cohort_hyperparams(np.random.default_rng(seed), spec)
        assert hp.family in ("blob", "splatter", "disk", "cylinder")


def test_between_ratio_zero_reuses_the_cohort_value_exactly():
    """size_between_ratio=0 -> every member's size_mm equals the cohort's, regardless
    of what the member's own RNG stream would have drawn independently."""
    spec = ShapeCohortSpec(size_between_ratio=0.0, position_between_ratio=0.0,
                          shape_between_ratio=0.0)
    cohort_hp = draw_cohort_hyperparams(np.random.default_rng(1), spec)
    for seed in (2, 3, 4):
        member = draw_member_shape(np.random.default_rng(seed), cohort_hp, spec)
        assert member.size_mm == pytest.approx(cohort_hp.size_mm)
        np.testing.assert_allclose(member.position_offset_mm, cohort_hp.position_offset_mm)


def test_between_ratio_one_draws_independently_per_member():
    """size_between_ratio=1 -> members drawn with different seeds get different
    size_mm (statistically -- not guaranteed for any single pair, so assert the
    cohort's 5 members aren't all identical, which between_ratio=0 always would be)."""
    spec = ShapeCohortSpec(size_between_ratio=1.0)
    cohort_hp = draw_cohort_hyperparams(np.random.default_rng(1), spec)
    sizes = [draw_member_shape(np.random.default_rng(s), cohort_hp, spec).size_mm
             for s in range(5)]
    assert len(set(sizes)) > 1


def test_between_ratio_blends_linearly_between_cohort_and_fresh_draw():
    spec_identical = ShapeCohortSpec(size_between_ratio=0.0)
    spec_half = ShapeCohortSpec(size_between_ratio=0.5)
    spec_full = ShapeCohortSpec(size_between_ratio=1.0)
    cohort_hp = draw_cohort_hyperparams(np.random.default_rng(9), spec_identical)
    m0 = draw_member_shape(np.random.default_rng(3), cohort_hp, spec_identical).size_mm
    m_half = draw_member_shape(np.random.default_rng(3), cohort_hp, spec_half).size_mm
    m1 = draw_member_shape(np.random.default_rng(3), cohort_hp, spec_full).size_mm
    assert m0 == pytest.approx(cohort_hp.size_mm)
    assert m_half == pytest.approx((m0 + m1) / 2, abs=1e-6)


def test_flatten_axis_redraws_per_member_when_shape_between_ratio_is_one():
    """Regression: flatten_axis (disk's one discrete shape param) used to never redraw
    per member regardless of shape_between_ratio, silently leaking cohort consistency
    on that axis even at ratio=1 ('fully independent'). ratio=1 -> the redraw always
    fires (member_rng.random() < 1.0 is always True), so across several member seeds
    at least one flatten_axis must differ from the cohort baseline."""
    spec = ShapeCohortSpec(family_weights={"disk": 1.0}, shape_between_ratio=1.0)
    cohort_hp = draw_cohort_hyperparams(np.random.default_rng(2), spec)
    axes = [draw_member_shape(np.random.default_rng(s), cohort_hp, spec)
            .shape_params["flatten_axis"] for s in range(8)]
    assert len(set(axes)) > 1 or axes[0] != cohort_hp.shape_params["flatten_axis"]


def test_flatten_axis_never_redraws_when_shape_between_ratio_is_zero():
    spec = ShapeCohortSpec(family_weights={"disk": 1.0}, shape_between_ratio=0.0)
    cohort_hp = draw_cohort_hyperparams(np.random.default_rng(2), spec)
    for seed in range(8):
        member = draw_member_shape(np.random.default_rng(seed), cohort_hp, spec)
        assert member.shape_params["flatten_axis"] == cohort_hp.shape_params["flatten_axis"]


def test_angle_diff_is_the_shortest_signed_wraparound():
    from src.shapes3d.instantiate import _angle_diff
    # going from near 2*pi to near 0 the short way should be a SMALL step, not the
    # ~6.18 long way around
    d = _angle_diff(0.05, 6.23)
    assert abs(d) < 0.2
    assert d == pytest.approx((0.05 - 6.23 + np.pi) % (2 * np.pi) - np.pi, abs=1e-9)


def test_cylinder_orientation_blends_via_shortest_angle_not_raw_value(monkeypatch):
    """draw_member_shape must route azimuth blending through _angle_diff (shortest-path
    wraparound), not a raw linear blend of the two angle values -- proven here by
    monkeypatching _angle_diff to a distinctive fixed return and confirming it flows
    straight through into the output, decoupled from whatever fresh_az the RNG drew."""
    import src.shapes3d.instantiate as instantiate_mod
    monkeypatch.setattr(instantiate_mod, "_angle_diff", lambda a, b: 10.0)
    spec = ShapeCohortSpec(family_weights={"cylinder": 1.0}, shape_between_ratio=0.1)
    cohort_hp = draw_cohort_hyperparams(np.random.default_rng(5), spec)
    cohort_hp.shape_params["azimuth"] = 0.05
    member = draw_member_shape(np.random.default_rng(11), cohort_hp, spec)
    assert member.shape_params["azimuth"] == pytest.approx(0.05 + 0.1 * 10.0)


def test_rasterize_shape_in_crop_stamps_the_new_class_id_into_crop_lbl():
    crop_lbl = np.zeros((30, 30, 30), dtype=np.uint8)
    spec = ShapeCohortSpec(family_weights={"blob": 1.0})
    cohort_hp = draw_cohort_hyperparams(np.random.default_rng(0), spec)
    member = draw_member_shape(np.random.default_rng(1), cohort_hp, spec)
    rasterize_shape_in_crop(crop_lbl, shape_id=195, member_draw=member,
                            mm_per_voxel=(1.0, 1.0, 1.0), center_local=(15.0, 15.0, 15.0),
                            rng=np.random.default_rng(2))
    assert (crop_lbl == 195).any()


def test_rasterize_shape_in_crop_does_not_raise_when_reach_is_entirely_outside_the_crop():
    """Loose containment: a shape's fixed physical position can legitimately fall
    outside a narrow/distant crop window (e.g. a bad cascade center prediction) -- that
    must silently stamp nothing, never raise or clip against anything else."""
    crop_lbl = np.zeros((10, 10, 10), dtype=np.uint8)
    spec = ShapeCohortSpec(family_weights={"blob": 1.0})
    cohort_hp = draw_cohort_hyperparams(np.random.default_rng(0), spec)
    member = draw_member_shape(np.random.default_rng(1), cohort_hp, spec)
    rasterize_shape_in_crop(crop_lbl, shape_id=195, member_draw=member,
                            mm_per_voxel=(1.0, 1.0, 1.0), center_local=(1000.0, 1000.0, 1000.0),
                            rng=np.random.default_rng(2))
    assert not (crop_lbl == 195).any()


def test_rasterize_shape_in_crop_is_physically_consistent_across_crop_resolutions():
    """The core cascade-consistency claim this whole module exists for: the SAME
    member_draw, rasterized into crops at DIFFERENT resolutions but covering the SAME
    physical field of view (like two cascade levels with different crop_spacing_mm),
    produces a shape of the SAME physical (mm) size and position -- unlike the earlier
    crop-relative (size_frac/position_uvw) design this replaced. Uses roughness=0 (a
    perfect sphere) so physical volume/centroid are unambiguous."""
    spec = ShapeCohortSpec(family_weights={"blob": 1.0}, blob_roughness_range=(0.0, 0.0),
                          size_between_ratio=0.0, position_between_ratio=0.0)
    cohort_hp = draw_cohort_hyperparams(np.random.default_rng(3), spec)
    member = draw_member_shape(np.random.default_rng(4), cohort_hp, spec)

    # Level A: 1mm voxels, 60^3 crop -> 60mm physical FOV. Level B: 2mm voxels, 30^3
    # crop -> the SAME 60mm physical FOV at half the resolution (like a coarser
    # cascade level). Shape centered in both.
    crop_a = np.zeros((60, 60, 60), dtype=np.uint8)
    rasterize_shape_in_crop(crop_a, shape_id=195, member_draw=member,
                            mm_per_voxel=(1.0, 1.0, 1.0), center_local=(30.0, 30.0, 30.0),
                            rng=np.random.default_rng(9))
    crop_b = np.zeros((30, 30, 30), dtype=np.uint8)
    rasterize_shape_in_crop(crop_b, shape_id=195, member_draw=member,
                            mm_per_voxel=(2.0, 2.0, 2.0), center_local=(15.0, 15.0, 15.0),
                            rng=np.random.default_rng(9))

    vol_a_mm3 = float((crop_a == 195).sum()) * 1.0 ** 3
    vol_b_mm3 = float((crop_b == 195).sum()) * 2.0 ** 3
    assert vol_a_mm3 > 0 and vol_b_mm3 > 0
    assert vol_b_mm3 == pytest.approx(vol_a_mm3, rel=0.3)   # coarser grid -> more discretization error

    centroid_a_mm = np.argwhere(crop_a == 195).mean(0) * 1.0
    centroid_b_mm = np.argwhere(crop_b == 195).mean(0) * 2.0
    np.testing.assert_allclose(centroid_a_mm, centroid_b_mm, atol=3.0)   # within ~1.5 coarse voxels
