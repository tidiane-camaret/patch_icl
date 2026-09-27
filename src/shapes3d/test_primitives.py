import numpy as np
import pytest

from src.shapes3d.primitives import (make_blob, make_cylinder, make_disk, make_scatter_field,
                                     make_shape, make_splatter, make_torus, make_vessel,
                                     reach_vox)

SHAPE = (40, 40, 40)
CENTER = (20.0, 20.0, 20.0)
SIZE_VOX = 3200.0   # matches the old size_frac=0.05 on a 40^3=64000-voxel grid


def test_make_blob_hits_requested_size_within_tolerance():
    rng = np.random.default_rng(0)
    mask, meta = make_blob(SHAPE, CENTER, {"size_vox": SIZE_VOX, "roughness": 0.15}, rng)
    assert mask.shape == SHAPE
    assert mask.dtype == np.uint8 or mask.dtype == bool
    realized_vox = float(mask.astype(bool).sum())
    assert 0.4 * SIZE_VOX < realized_vox < 1.8 * SIZE_VOX   # harmonic perturbation moves it off-target some
    assert meta["family"] == "blob"
    assert meta["realized_size_frac"] == pytest.approx(mask.astype(bool).mean(), abs=1e-9)


def test_make_blob_is_deterministic_given_the_same_rng_state():
    mask1, _ = make_blob(SHAPE, CENTER, {"size_vox": SIZE_VOX, "roughness": 0.15},
                         np.random.default_rng(42))
    mask2, _ = make_blob(SHAPE, CENTER, {"size_vox": SIZE_VOX, "roughness": 0.15},
                         np.random.default_rng(42))
    np.testing.assert_array_equal(mask1, mask2)


def test_make_blob_roughness_zero_is_a_sphere():
    from scipy.ndimage import binary_erosion
    rng = np.random.default_rng(0)
    mask, _ = make_blob(SHAPE, CENTER, {"size_vox": SIZE_VOX, "roughness": 0.0}, rng)
    m = mask.astype(bool)
    surface = m & ~binary_erosion(m)
    coords = np.argwhere(surface)
    dist = np.linalg.norm(coords - np.array(CENTER), axis=1)
    assert dist.std() < 0.75               # every surface voxel ~equidistant from center


def test_make_blob_size_is_independent_of_the_grid_it_is_rasterized_into():
    """The whole point of the size_vox (absolute) API: the SAME size_vox produces the
    SAME realized voxel count regardless of the grid's own size -- unlike the old
    size_frac (fraction-of-grid) design this replaced."""
    rng1 = np.random.default_rng(7)
    mask_small, _ = make_blob((30, 30, 30), (15.0, 15.0, 15.0),
                              {"size_vox": SIZE_VOX, "roughness": 0.0}, rng1)
    rng2 = np.random.default_rng(7)
    mask_large, _ = make_blob((80, 80, 80), (40.0, 40.0, 40.0),
                              {"size_vox": SIZE_VOX, "roughness": 0.0}, rng2)
    assert mask_small.astype(bool).sum() == mask_large.astype(bool).sum()


def test_make_splatter_has_multiple_components():
    from scipy.ndimage import label as cc_label
    rng = np.random.default_rng(1)
    mask, meta = make_splatter(
        SHAPE, CENTER, {"size_vox": SIZE_VOX, "n_components": 5, "spread_vox": 12.0,
                       "roughness": 0.15}, rng)
    n_components, count = cc_label(mask.astype(bool))
    assert count >= 2                       # "several disjoint components", not one blob
    assert meta["family"] == "splatter"


def test_make_disk_is_flattened_along_its_axis():
    rng = np.random.default_rng(2)
    mask, meta = make_disk(
        SHAPE, CENTER, {"size_vox": SIZE_VOX, "aspect_ratio": 0.2, "flatten_axis": 0}, rng)
    coords = np.argwhere(mask.astype(bool))
    extent = coords.max(0) - coords.min(0)
    assert extent[0] < extent[1] * 0.6      # flattened axis visibly shorter than the others
    assert extent[0] < extent[2] * 0.6


def test_make_cylinder_is_elongated_along_its_axis():
    rng = np.random.default_rng(3)
    mask, meta = make_cylinder(
        SHAPE, CENTER,
        {"size_vox": SIZE_VOX, "length_vox": 28.0, "azimuth": 0.0, "elevation": 0.0}, rng)
    coords = np.argwhere(mask.astype(bool))
    extent = coords.max(0) - coords.min(0)
    assert extent.max() > extent.min() * 1.5   # visibly elongated, not isotropic
    assert meta["family"] == "cylinder"


def test_make_scatter_field_has_many_components():
    from scipy.ndimage import label as cc_label
    rng = np.random.default_rng(4)
    mask, meta = make_scatter_field(
        (100, 100, 100), (50.0, 50.0, 50.0),
        {"size_vox": SIZE_VOX, "n_components": 40, "spread_vox": 30.0, "roughness": 0.15}, rng)
    _, count = cc_label(mask.astype(bool))
    assert count >= 10                      # many disjoint components, not a handful
    assert meta["family"] == "scatter_field"
    assert meta["n_components"] == 40


def test_make_scatter_field_is_deterministic():
    kwargs = dict(shape=(60, 60, 60), center=(30.0, 30.0, 30.0),
                  params={"size_vox": SIZE_VOX, "n_components": 20, "spread_vox": 15.0})
    mask1, _ = make_scatter_field(**kwargs, rng=np.random.default_rng(3))
    mask2, _ = make_scatter_field(**kwargs, rng=np.random.default_rng(3))
    np.testing.assert_array_equal(mask1, mask2)


def test_make_vessel_is_thin_and_branching():
    rng = np.random.default_rng(5)
    mask, meta = make_vessel(
        (100, 100, 100), (50.0, 50.0, 50.0),
        {"size_vox": 400.0, "length_vox": 40.0, "azimuth": 0.3, "elevation": 0.2,
         "branch_depth": 3, "radius_falloff": 0.75}, rng)
    assert mask.astype(bool).sum() > 0
    assert meta["family"] == "vessel"
    assert meta["n_segments"] > 1           # actually branched, not just the trunk
    # thin: total foreground volume should be much less than a sphere of the same
    # bounding-box extent would need (a branching tube fills its bbox sparsely).
    coords = np.argwhere(mask.astype(bool))
    bbox_vol = np.prod(coords.max(0) - coords.min(0) + 1)
    assert mask.astype(bool).sum() < 0.3 * bbox_vol


def test_make_vessel_branch_depth_zero_is_just_the_trunk():
    rng = np.random.default_rng(6)
    mask, meta = make_vessel(
        (60, 60, 60), (30.0, 30.0, 30.0),
        {"size_vox": 400.0, "length_vox": 30.0, "azimuth": 0.0, "elevation": 0.0,
         "branch_depth": 0, "radius_falloff": 0.75}, rng)
    assert meta["n_segments"] == 1


def test_make_torus_has_a_hole_in_the_middle():
    rng = np.random.default_rng(7)
    center = (50.0, 50.0, 50.0)
    mask, meta = make_torus(
        (100, 100, 100), center,
        {"size_vox": SIZE_VOX, "ratio": 3.0, "flatten_axis": 0}, rng)
    assert mask.astype(bool).sum() > 0
    assert meta["family"] == "torus"
    # the exact center voxel must be background -- that's the defining feature of a torus
    assert not mask[int(center[0]), int(center[1]), int(center[2])]


def test_make_torus_ring_lies_in_the_plane_perpendicular_to_flatten_axis():
    rng = np.random.default_rng(8)
    center = (50.0, 50.0, 50.0)
    mask, _ = make_torus((100, 100, 100), center,
                         {"size_vox": SIZE_VOX, "ratio": 4.0, "flatten_axis": 0}, rng)
    coords = np.argwhere(mask.astype(bool))
    extent = coords.max(0) - coords.min(0)
    assert extent[0] < extent[1] * 0.6       # flattened along axis 0, same test as make_disk
    assert extent[0] < extent[2] * 0.6


def test_make_shape_dispatches_by_family_name():
    rng = np.random.default_rng(0)
    mask, meta = make_shape("disk", SHAPE, CENTER,
                            {"size_vox": SIZE_VOX, "aspect_ratio": 0.2, "flatten_axis": 1}, rng)
    assert meta["family"] == "disk"


def test_make_shape_rejects_unknown_family():
    with pytest.raises(ValueError):
        make_shape("not_a_family", SHAPE, CENTER, {"size_vox": SIZE_VOX}, np.random.default_rng(0))


def test_reach_vox_bounds_every_foreground_voxel_for_each_family():
    """reach_vox must be a genuine upper bound: every foreground voxel of a shape
    rasterized on a grid FAR larger than its reach must lie within reach_vox of
    center (otherwise instantiate.rasterize_shape_in_crop's sub-box would clip it)."""
    big_shape = (120, 120, 120)
    big_center = (60.0, 60.0, 60.0)
    cases = [
        ("blob", {"size_vox": SIZE_VOX, "roughness": 0.3}),
        ("splatter", {"size_vox": SIZE_VOX, "n_components": 5, "spread_vox": 15.0,
                      "roughness": 0.2}),
        ("disk", {"size_vox": SIZE_VOX, "aspect_ratio": 0.15, "flatten_axis": 1}),
        ("cylinder", {"size_vox": SIZE_VOX, "length_vox": 40.0, "azimuth": 0.7,
                      "elevation": 0.3}),
        ("scatter_field", {"size_vox": SIZE_VOX, "n_components": 30, "spread_vox": 25.0,
                           "roughness": 0.2}),
        ("vessel", {"size_vox": 400.0, "length_vox": 30.0, "azimuth": 0.7, "elevation": 0.3,
                   "branch_depth": 4, "radius_falloff": 0.8}),
        ("torus", {"size_vox": SIZE_VOX, "ratio": 3.0, "flatten_axis": 1}),
    ]
    for family, params in cases:
        mask, _ = make_shape(family, big_shape, big_center, params, np.random.default_rng(0))
        coords = np.argwhere(mask.astype(bool))
        assert coords.size > 0, family
        dist = np.linalg.norm(coords - np.array(big_center), axis=1)
        bound = reach_vox(family, params)
        assert dist.max() <= bound, (family, dist.max(), bound)


def test_reach_vox_rejects_unknown_family():
    with pytest.raises(ValueError):
        reach_vox("not_a_family", {"size_vox": SIZE_VOX})
