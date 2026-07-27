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


def _bezier(P0, P1, P2, m):
    """m points along the quadratic Bézier through P0, P1(control), P2. (m, 3)."""
    t = np.linspace(0.0, 1.0, m)[:, None]
    return (1 - t) ** 2 * P0 + 2 * (1 - t) * t * P1 + t ** 2 * P2


def sample_object_spec(rng, shape="blob", eccentricity=3.0, n_harmonics=4,
                       harmonic_amp=0.30, edge_blur=0.08):
    """Draw a reproducible object geometry. `kind`="ellipsoid" (blob/elongated:
    axes + angular bumps) or "tube" (tubular: a swept sphere along a curved
    centerline). Both carry `R0` (base orientation) and `edge_blur`."""
    if shape == "mix":
        shape = str(rng.choice(("blob", "elongated", "tubular")))

    if shape == "tubular":
        # Capsule / swept-sphere along a gently curved centerline (cheap: a handful
        # of sample points; render unions spheres via point-to-centerline distance).
        R0 = _rand_rotation(rng)
        axis = np.array([1.0, 0.0, 0.0])
        half = float(rng.uniform(0.55, 0.72))                    # half-length in [-1,1]
        perp = rng.standard_normal(3)
        perp = perp - (perp @ axis) * axis
        perp = perp / (np.linalg.norm(perp) + 1e-9)
        bend = float(rng.uniform(0.0, 0.5)) * half               # curvature offset
        m = 24
        curve = _bezier(-half * axis, bend * perp, half * axis, m)   # (m, 3)
        r0 = float(rng.uniform(0.10, 0.20))                      # tube caliber
        taper = float(rng.uniform(0.0, 0.6))                     # thin toward one end
        radii = np.maximum(r0 * (1.0 - taper * np.linspace(0.0, 1.0, m)), 0.03)
        return {"kind": "tube", "shape": shape, "curve": curve, "radii": radii,
                "R0": R0, "edge_blur": float(edge_blur)}

    # ellipsoid path — original draw order preserved (axes, R0, terms) so existing
    # blob/elongated specs stay byte-identical.
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
    return {"kind": "ellipsoid", "shape": shape, "axes": axes, "R0": R0, "terms": terms,
            "edge_blur": float(edge_blur)}


def render_object(size, spec, R_extra=None):
    """Rasterize a spec to a soft alpha tile (size^3) in [0, 1]. R_extra applies a
    small per-scene rotation on top of the spec's base orientation."""
    z, y, x = _unit_grid(size)
    pts = np.stack([z.ravel(), y.ravel(), x.ravel()], 0)          # (3, N)
    R = spec["R0"] if R_extra is None else (R_extra @ spec["R0"])

    if spec.get("kind") == "tube":
        curve = (R @ spec["curve"].T).T                          # (M, 3) world coords
        radii = spec["radii"]                                    # (M,)
        # squared distance from every centerline point to every voxel, via
        # ||c-p||^2 = |c|^2 - 2 c·p + |p|^2 (no (M,3,N) temporary).
        csq = (curve ** 2).sum(1)[:, None]                       # (M, 1)
        psq = (pts ** 2).sum(0)[None, :]                         # (1, N)
        dist = np.sqrt(np.clip(csq - 2.0 * (curve @ pts) + psq, 0.0, None))  # (M, N)
        val = (radii[:, None] - dist).max(0)                     # union of spheres (N,)
        blur = max(1e-3, float(spec["edge_blur"]) * float(radii.max()))
        alpha = np.clip(val / blur + 0.5, 0.0, 1.0)
        return alpha.reshape(z.shape).astype(np.float32)

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
