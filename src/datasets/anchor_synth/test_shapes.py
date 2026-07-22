import sys; sys.path.insert(0, ".")
import numpy as np

from src.datasets.anchor_synth.shapes import (
    sample_object_spec, render_object, small_rotation, roughen,
)


def test_render_object_shape_and_range():
    rng = np.random.default_rng(0)
    spec = sample_object_spec(rng, shape="blob")
    a = render_object(24, spec)
    assert a.shape == (24, 24, 24)
    assert a.dtype == np.float32
    assert a.min() >= 0.0 and a.max() <= 1.0
    assert (a > 0.5).sum() > 0                      # non-empty object


def test_render_object_is_irregular():
    # harmonics + random orientation => not mirror-symmetric on axis 0
    rng = np.random.default_rng(1)
    spec = sample_object_spec(rng, shape="blob", harmonic_amp=0.30)
    a = render_object(32, spec)
    assert not np.allclose(a, np.flip(a, axis=0))


def test_render_object_deterministic_for_spec():
    spec = sample_object_spec(np.random.default_rng(7))
    a = render_object(20, spec)
    b = render_object(20, spec)
    assert np.array_equal(a, b)                      # spec fully determines shape


def test_elongated_is_anisotropic():
    rng = np.random.default_rng(2)
    spec = sample_object_spec(rng, shape="elongated", eccentricity=4.0)
    a = render_object(40, spec) > 0.5
    sides = []
    for ax in range(3):
        proj = a.any(axis=tuple(i for i in range(3) if i != ax))
        idx = np.nonzero(proj)[0]
        sides.append(idx[-1] - idx[0] + 1)
    assert max(sides) >= 1.6 * min(sides)            # clearly elongated


def test_small_rotation_is_near_identity():
    R = small_rotation(np.random.default_rng(3), max_deg=10.0)
    assert R.shape == (3, 3)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-6)   # orthonormal
    assert np.trace(R) > 2.5                             # small angle


def test_roughen_changes_boundary():
    spec = sample_object_spec(np.random.default_rng(4))
    a = render_object(28, spec)
    r = roughen(a, c=0.6, rng=np.random.default_rng(5))
    assert r.shape == a.shape
    assert not np.array_equal(r > 0.5, a > 0.5)
