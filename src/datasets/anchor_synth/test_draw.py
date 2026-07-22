import sys; sys.path.insert(0, ".")
import numpy as np

from src.datasets.anchor_synth.draw import (
    anchor_stats, offset_to_center, place_object,
)


def test_anchor_stats_bbox_and_centroid():
    m = np.zeros((20, 20, 20), dtype=np.uint8)
    m[4:10, 6:14, 8:12] = 1
    centroid, extent, (lo, hi) = anchor_stats(m)
    assert list(lo) == [4, 6, 8]
    assert list(hi) == [9, 13, 11]
    assert list(extent) == [6, 8, 4]
    assert np.allclose(centroid, [6.5, 9.5, 9.5])


def test_anchor_stats_empty_returns_none():
    assert anchor_stats(np.zeros((8, 8, 8), dtype=np.uint8)) is None


def test_offset_to_center_uses_extent_and_clamps():
    centroid = np.array([10.0, 10.0, 10.0])
    extent = np.array([8.0, 8.0, 8.0])
    c = offset_to_center(centroid, extent, [0.5, 0.0, -0.5], tile_size=6,
                         vol_shape=(20, 20, 20))
    assert np.allclose(c, [14.0, 10.0, 6.0])
    # push far out-of-bounds -> clamped so a size-6 tile stays fully inside
    c2 = offset_to_center(centroid, extent, [10.0, -10.0, 0.0], tile_size=6,
                          vol_shape=(20, 20, 20))
    assert np.allclose(c2, [17.0, 3.0, 10.0])       # [20-3, 3, 10]


def test_place_object_blends_and_writes_label():
    image = np.full((16, 16, 16), 0.4, dtype=np.float32)
    label = np.zeros((16, 16, 16), dtype=np.int64)
    alpha = np.zeros((6, 6, 6), dtype=np.float32)
    alpha[1:5, 1:5, 1:5] = 1.0                       # solid core
    foot = place_object(image, alpha, center=[8, 8, 8], contrast_delta=0.2,
                        label=label, label_id=1)
    assert foot.shape == image.shape
    assert foot.sum() == (alpha > 0.5).sum()
    # interior intensity == local bg (0.4) + delta (0.2)
    assert np.allclose(image[foot], 0.6, atol=1e-5)
    assert (label == 1).sum() == foot.sum()
    assert np.array_equal(label > 0, foot)


def test_place_object_clips_at_border():
    image = np.zeros((10, 10, 10), dtype=np.float32)
    alpha = np.ones((6, 6, 6), dtype=np.float32)
    foot = place_object(image, alpha, center=[0, 0, 0], contrast_delta=1.0)
    assert foot[:3, :3, :3].all()                    # in-bounds octant written
    assert foot.sum() == 27
