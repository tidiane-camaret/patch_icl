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
