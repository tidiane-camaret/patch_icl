import numpy as np
import pytest

from src.shapes3d.primitives import (make_blob, make_cylinder, make_disk, make_shape,
                                     make_splatter, reach_vox)

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
