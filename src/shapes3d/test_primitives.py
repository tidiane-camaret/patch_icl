import numpy as np
import pytest

from src.shapes3d.primitives import make_blob, make_cylinder, make_disk, make_shape, make_splatter

SHAPE = (40, 40, 40)
CENTER = (20.0, 20.0, 20.0)


def test_make_blob_hits_requested_size_frac_within_tolerance():
    rng = np.random.default_rng(0)
    mask, meta = make_blob(SHAPE, CENTER, {"size_frac": 0.05, "roughness": 0.15}, rng)
    assert mask.shape == SHAPE
    assert mask.dtype == np.uint8 or mask.dtype == bool
    realized = mask.astype(bool).mean()
    assert 0.02 < realized < 0.09          # harmonic perturbation moves it off-target some
    assert meta["family"] == "blob"
    assert meta["realized_size_frac"] == pytest.approx(realized, abs=1e-9)


def test_make_blob_is_deterministic_given_the_same_rng_state():
    mask1, _ = make_blob(SHAPE, CENTER, {"size_frac": 0.05, "roughness": 0.15},
                         np.random.default_rng(42))
    mask2, _ = make_blob(SHAPE, CENTER, {"size_frac": 0.05, "roughness": 0.15},
                         np.random.default_rng(42))
    np.testing.assert_array_equal(mask1, mask2)


def test_make_blob_roughness_zero_is_a_sphere():
    from scipy.ndimage import binary_erosion
    rng = np.random.default_rng(0)
    mask, _ = make_blob(SHAPE, CENTER, {"size_frac": 0.05, "roughness": 0.0}, rng)
    m = mask.astype(bool)
    surface = m & ~binary_erosion(m)
    coords = np.argwhere(surface)
    dist = np.linalg.norm(coords - np.array(CENTER), axis=1)
    assert dist.std() < 0.75               # every surface voxel ~equidistant from center


def test_make_splatter_has_multiple_components():
    from scipy.ndimage import label as cc_label
    rng = np.random.default_rng(1)
    mask, meta = make_splatter(
        SHAPE, CENTER, {"size_frac": 0.05, "n_components": 5, "spread_frac": 0.3,
                       "roughness": 0.15}, rng)
    n_components, count = cc_label(mask.astype(bool))
    assert count >= 2                       # "several disjoint components", not one blob
    assert meta["family"] == "splatter"


def test_make_disk_is_flattened_along_its_axis():
    rng = np.random.default_rng(2)
    mask, meta = make_disk(
        SHAPE, CENTER, {"size_frac": 0.05, "aspect_ratio": 0.2, "flatten_axis": 0}, rng)
    coords = np.argwhere(mask.astype(bool))
    extent = coords.max(0) - coords.min(0)
    assert extent[0] < extent[1] * 0.6      # flattened axis visibly shorter than the others
    assert extent[0] < extent[2] * 0.6


def test_make_cylinder_is_elongated_along_its_axis():
    rng = np.random.default_rng(3)
    mask, meta = make_cylinder(
        SHAPE, CENTER,
        {"size_frac": 0.05, "radius_frac": 0.2, "length_frac": 0.7,
         "azimuth": 0.0, "elevation": 0.0}, rng)
    coords = np.argwhere(mask.astype(bool))
    extent = coords.max(0) - coords.min(0)
    assert extent.max() > extent.min() * 1.5   # visibly elongated, not isotropic
    assert meta["family"] == "cylinder"


def test_make_shape_dispatches_by_family_name():
    rng = np.random.default_rng(0)
    mask, meta = make_shape("disk", SHAPE, CENTER,
                            {"size_frac": 0.05, "aspect_ratio": 0.2, "flatten_axis": 1}, rng)
    assert meta["family"] == "disk"


def test_make_shape_rejects_unknown_family():
    with pytest.raises(ValueError):
        make_shape("not_a_family", SHAPE, CENTER, {"size_frac": 0.05}, np.random.default_rng(0))
