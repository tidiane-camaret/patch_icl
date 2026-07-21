import sys; sys.path.insert(0, ".")
import numpy as np
from src.datasets.omniSynth.bank_common3d import make_object_tile_3d, crop_to_tile_3d


def _blob(shape=(8, 10, 6)):
    m = np.zeros(shape, dtype=bool)
    m[2:6, 3:8, 1:5] = True
    v = np.zeros(shape, dtype=np.float32)
    v[m] = 0.5
    return v, m


def test_tile_shape_and_channels_no_resize():
    v, m = _blob()
    t = make_object_tile_3d(v, m, source_size=64, image_size=64, size_scale=1.0)
    assert t.ndim == 4 and t.shape[0] == 2 and t.dtype == np.float16
    D, H, W = t.shape[1:]
    assert D == H == W                       # centered in a cube
    assert D == max(v.shape)                 # r==1 -> tile = max bbox dim


def test_mask_binary_and_intensity_masked():
    v, m = _blob()
    t = make_object_tile_3d(v, m, source_size=64, image_size=64)
    intensity, mask = t[0].astype(np.float32), t[1].astype(np.float32)
    assert set(np.unique(mask)).issubset({0.0, 1.0})
    assert mask.sum() > 0
    assert float(intensity[mask == 0].max(initial=0.0)) == 0.0   # no texture outside mask


def test_size_scale_shrinks_tile():
    v, m = _blob()
    big = make_object_tile_3d(v, m, source_size=64, image_size=64, size_scale=1.0)
    small = make_object_tile_3d(v, m, source_size=64, image_size=64, size_scale=0.5)
    assert small.shape[1] < big.shape[1]


def test_crop_to_tile_rejects_tiny_mask():
    v = np.zeros((8, 8, 8), dtype=np.float32)
    m = np.zeros((8, 8, 8), dtype=bool)
    m[0, 0, 0] = True                        # 1 voxel
    assert crop_to_tile_3d(v, m, min_vox=8, source_size=64, image_size=64) is None


def test_crop_to_tile_crops_to_bbox():
    v = np.zeros((16, 16, 16), dtype=np.float32)
    m = np.zeros((16, 16, 16), dtype=bool)
    m[4:8, 4:9, 4:7] = True                  # bbox dims (4,5,3)
    v[m] = 0.7
    t = crop_to_tile_3d(v, m, min_vox=4, source_size=64, image_size=64)
    assert t is not None and t.shape[1] == 5     # tile = max bbox dim


if __name__ == "__main__":
    test_tile_shape_and_channels_no_resize()
    test_mask_binary_and_intensity_masked()
    test_size_scale_shrinks_tile()
    test_crop_to_tile_rejects_tiny_mask()
    test_crop_to_tile_crops_to_bbox()
    print("ALL BANK_COMMON3D TESTS PASSED")
