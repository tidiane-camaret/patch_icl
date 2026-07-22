"""Analytic 3D object generators for anchor_synth3d.

Follows controlSynth/shapes/blob.py extended to 3D: a base ellipsoid whose
radius is modulated by low-frequency angular bumps gives organic, irregular
(non-spherical) shapes. Fully analytic — no scipy in the default path. The
object geometry is fixed by `sample_object_spec` so it is reproducible and can
be shared across the K+1 scenes; `render_object` rasterizes it (optionally with a
small extra rotation for per-scene jitter). `roughen` (scipy) is opt-in.
"""

import numpy as np


def _unit_grid(size):
    """Centered coordinate grids in [-1, 1] on a size^3 cube (z, y, x)."""
    half = max(1.0, (size - 1) / 2.0)
    lin = (np.arange(size) - (size - 1) / 2.0) / half
    z, y, x = np.meshgrid(lin, lin, lin, indexing="ij")
    return z, y, x


def _rand_rotation(rng):
    """A random rotation matrix via QR of a Gaussian matrix (sign-fixed)."""
    q, r = np.linalg.qr(rng.standard_normal((3, 3)))
    q = q * np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def small_rotation(rng, max_deg):
    """A rotation of a random axis by a small angle in [-max_deg, max_deg]."""
    ax = rng.standard_normal(3)
    ax = ax / (np.linalg.norm(ax) + 1e-9)
    ang = np.radians(rng.uniform(-max_deg, max_deg))
    K = np.array([[0, -ax[2], ax[1]],
                  [ax[2], 0, -ax[0]],
                  [-ax[1], ax[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1.0 - np.cos(ang)) * (K @ K)


def sample_object_spec(rng, shape="blob", eccentricity=3.0, n_harmonics=4,
                       harmonic_amp=0.30, edge_blur=0.08):
    """Draw a reproducible object geometry (axes, orientation, angular bumps)."""
    if shape == "mix":
        shape = str(rng.choice(("blob", "elongated")))
    axes = np.array([rng.uniform(0.85, 1.0) for _ in range(3)], dtype=np.float64)
    if shape == "elongated":
        axes[:] = 1.0 / np.sqrt(float(eccentricity))
        axes[int(rng.integers(3))] = 1.0
    R0 = _rand_rotation(rng)
    terms = []
    for _ in range(int(n_harmonics)):
        u = rng.standard_normal(3)
        u = u / (np.linalg.norm(u) + 1e-9)
        terms.append((u, float(rng.uniform(-harmonic_amp, harmonic_amp))))
    return {"axes": axes, "R0": R0, "terms": terms, "edge_blur": float(edge_blur)}


def render_object(size, spec, R_extra=None):
    """Rasterize a spec to a soft alpha tile (size^3) in [0, 1]. R_extra applies a
    small per-scene rotation on top of the spec's base orientation."""
    z, y, x = _unit_grid(size)
    pts = np.stack([z.ravel(), y.ravel(), x.ravel()], 0)          # (3, N)
    R = spec["R0"] if R_extra is None else (R_extra @ spec["R0"])
    pr = (R @ pts) / spec["axes"][:, None]                        # ellipsoid frame
    rr = np.sqrt((pr ** 2).sum(0))                               # radius (N,)
    dirs = pr / (rr + 1e-6)                                      # unit dirs (3, N)
    r_mod = np.ones_like(rr)
    for u, a in spec["terms"]:
        r_mod = r_mod + a * (dirs.T @ u) ** 2                    # low-freq bumps
    r_mod = np.clip(r_mod, 0.5, 1.7)
    base_r = 0.72
    blur = max(1e-3, float(spec["edge_blur"]) * base_r)
    alpha = np.clip((base_r * r_mod - rr) / blur + 0.5, 0.0, 1.0)
    return alpha.reshape(z.shape).astype(np.float32)


def roughen(alpha, c, rng):
    """Perturb the alpha boundary via SDF + smoothed noise (opt-in, uses scipy).
    Mirrors controlSynth/shapes/boundary.py. Returns a hard {0,1} float32 mask."""
    from scipy.ndimage import distance_transform_edt, gaussian_filter

    c = float(c)
    m = alpha > 0.5
    if c <= 0.0 or not m.any():
        return alpha
    sdf = distance_transform_edt(m) - distance_transform_edt(~m)
    char = float(np.cbrt(m.sum()))
    noise = gaussian_filter(rng.standard_normal(m.shape), sigma=max(1.0, char * 0.3))
    std = noise.std()
    if std > 1e-8:
        noise = noise / std
    rough = (sdf + c * char * 0.5 * noise) > 0.0
    return rough.astype(np.float32)
