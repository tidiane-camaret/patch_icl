import numpy as np
from src.datasets.anchor_synth.draw import (
    affine_weights, frame_length, barycentric_center,
    anchor_stats, place_object,
)

TETRA = np.array([[0., 0., 0.], [10., 0., 0.], [0., 10., 0.], [0., 0., 10.]])


def test_affine_weights_sum_to_one_and_convex_when_no_extrapolation():
    rng = np.random.default_rng(0)
    for _ in range(50):
        w = affine_weights(rng, 4, extrapolation=0.0, concentration=1.0)
        assert w.shape == (4,)
        assert abs(w.sum() - 1.0) < 1e-9
        assert (w >= -1e-12).all()          # convex: inside the hull


def test_affine_weights_extrapolation_allows_negative():
    rng = np.random.default_rng(1)
    saw_negative = any(
        affine_weights(rng, 4, extrapolation=1.0).min() < 0 for _ in range(200)
    )
    assert saw_negative                     # mild extrapolation can leave the hull


def test_frame_length_is_rotation_invariant():
    rng = np.random.default_rng(2)
    q, _ = np.linalg.qr(rng.standard_normal((3, 3)))     # random rotation
    L0 = frame_length(TETRA)
    L1 = frame_length((q @ TETRA.T).T + np.array([5., -3., 2.]))  # rotate + translate
    assert abs(L0 - L1) < 1e-6
    assert L0 > 0


def test_barycentric_center_barycenter_and_onehot():
    vol = (100, 100, 100)
    bc = barycentric_center(TETRA, np.full(4, 0.25), tile_size=4, vol_shape=vol)
    assert np.allclose(bc, TETRA.mean(0))
    # One-hot on point in-bounds: should return that point
    oh = barycentric_center(TETRA, np.array([0., 1., 0., 0.]), tile_size=4, vol_shape=vol)
    assert np.allclose(oh, [10., 2., 2.])  # [10., 0., 0.] clamped by tile_size/2


def test_barycentric_center_clamped_in_bounds():
    vol = (32, 32, 32)
    # Low-side clamp: all centroids at origin, weights push negative → clamped to half tile.
    far = np.array([[0., 0., 0.]] * 4)
    bc = barycentric_center(far, np.array([2., -1., 0., 0.]), tile_size=8, vol_shape=vol)
    assert (bc >= 4).all() and (bc <= 28).all()          # tile_size/2 .. vol - tile_size/2
    # High-side clamp: centroids near the far corner, weight > 1 pushes past the upper edge.
    corner = np.array([[30., 30., 30.]] * 4)
    bc_hi = barycentric_center(corner, np.array([2., -1., 0., 0.]), tile_size=8, vol_shape=vol)
    upper = np.array(vol, dtype=float) - 8 / 2.0        # vol - tile_size/2 = 28
    assert np.allclose(bc_hi, upper)                     # clamped to upper bound on all axes
    assert (bc_hi >= 4).all()                            # also above lower bound


# ---------------------------------------------------------------------------
# anchor_stats
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# place_object
# ---------------------------------------------------------------------------

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
