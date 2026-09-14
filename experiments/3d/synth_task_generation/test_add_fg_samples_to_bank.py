import numpy as np

from add_fg_samples_to_bank import fg_samples_for_mask


def test_fg_samples_for_mask_returns_valid_coords_per_present_class():
    arr = np.zeros((6, 6, 6), dtype=np.uint8)
    arr[1, 1, 1] = 1
    arr[2, 2, 2] = 1
    arr[4, 4, 4] = 2
    out = fg_samples_for_mask(arr, n_samples=64, seed=0)
    assert set(out.keys()) == {1, 2}
    for coord in out[1]:
        assert arr[tuple(coord)] == 1
    for coord in out[2]:
        assert arr[tuple(coord)] == 2
    assert 0 not in out


def test_fg_samples_for_mask_returns_all_voxels_when_fewer_than_n_samples():
    arr = np.zeros((6, 6, 6), dtype=np.uint8)
    arr[1, 1, 1] = 5
    arr[2, 2, 2] = 5
    out = fg_samples_for_mask(arr, n_samples=64, seed=0)
    assert out[5].shape == (2, 3)
    coords = {tuple(c) for c in out[5]}
    assert coords == {(1, 1, 1), (2, 2, 2)}


def test_fg_samples_for_mask_caps_at_n_samples_without_duplicates():
    arr = np.zeros((10, 10, 10), dtype=np.uint8)
    arr[:, :, :] = 0
    arr[0:5, :, :] = 3           # 500 voxels of class 3
    out = fg_samples_for_mask(arr, n_samples=8, seed=0)
    assert out[3].shape == (8, 3)
    coords = {tuple(c) for c in out[3]}
    assert len(coords) == 8      # no duplicates
    for coord in coords:
        assert arr[coord] == 3


def test_fg_samples_for_mask_is_reproducible_with_same_seed():
    arr = np.zeros((10, 10, 10), dtype=np.uint8)
    arr[0:5, :, :] = 3
    a = fg_samples_for_mask(arr, n_samples=8, seed=42)
    b = fg_samples_for_mask(arr, n_samples=8, seed=42)
    assert np.array_equal(a[3], b[3])


def test_fg_samples_for_mask_empty_array_returns_no_labels():
    arr = np.zeros((4, 4, 4), dtype=np.uint8)
    out = fg_samples_for_mask(arr, n_samples=8, seed=0)
    assert out == {}
