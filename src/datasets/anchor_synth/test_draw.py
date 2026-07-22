import numpy as np
from src.datasets.anchor_synth.draw import (
    affine_weights, frame_length, barycentric_center,
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
    far = np.array([[0., 0., 0.]] * 4)
    bc = barycentric_center(far, np.array([2., -1., 0., 0.]), tile_size=8, vol_shape=vol)
    assert (bc >= 4).all() and (bc <= 28).all()          # tile_size/2 .. vol - tile_size/2
