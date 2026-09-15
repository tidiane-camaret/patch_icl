"""Procedural 3D shape primitives for synth_gmm's shape-mode cohorts (see
docs/superpowers/specs/2026-09-14-cohort-consistent-synthetic-shapes-design.md).

Pure NumPy: no torch, no provider/dataset imports. Every make_<family> function takes
the target grid shape, a center voxel (float coords, may be non-integer, relative to
`shape`'s own origin), a params dict, and an np.random.Generator, and returns
(mask uint8[*shape], realized_meta dict). Shapes are NOT anatomically plausible by
design (spec: "do not need to be plausible lesions") -- they exist to teach general
geometric reasoning, not organ realism.

Target sizes are ABSOLUTE local-voxel quantities (`size_vox`, `length_vox`,
`spread_vox`), not fractions of `shape` -- the caller (src/shapes3d/instantiate.py)
resolves a member's physical (mm) size into these local-voxel absolutes using the
CURRENT crop's own voxel size, so the same physical shape can be rasterized into a
small sub-box rather than the full crop (see `reach_vox`, used by
instantiate.rasterize_shape_in_crop for exactly this). This is what makes shape-mode
cohorts consistent across cascade levels with different crop_spacing_mm -- see
docs/logs.md 2026-09-15 (this replaced an earlier size_frac/length_frac/spread_frac
design that was relative to whatever `shape` happened to be passed in, and therefore
NOT physically consistent across crop resolutions).

Volume targeting is closed-form (radius solved from the target voxel count), not an
iterative search -- cheap, and "realized" (measured from the rasterized mask) is always
recorded rather than trusting the request, since harmonic/angular perturbation and
voxel discretization both move the actual volume off-target."""

import numpy as np

_EPS = 1e-9


def _radius_for_volume(target_vox):
    """Equivalent-sphere radius (voxels) for an absolute target volume (voxels)."""
    target_vox = max(1.0, float(target_vox))
    return (3.0 * target_vox / (4.0 * np.pi)) ** (1.0 / 3.0)


def _grid_dist_angle(shape, center):
    """(rr, theta, phi): spherical radius / polar angle (from axis 0) / azimuth (in the
    axis1-axis2 plane) of every grid voxel relative to `center`."""
    D, H, W = shape
    dd, hh, ww = np.mgrid[0:D, 0:H, 0:W].astype(np.float64)
    dz, dy, dx = dd - center[0], hh - center[1], ww - center[2]
    rr = np.sqrt(dz * dz + dy * dy + dx * dx)
    theta = np.arccos(np.divide(dz, rr, out=np.zeros_like(rr), where=rr > _EPS))
    phi = np.arctan2(dy, dx)
    return rr, theta, phi


def _harmonics(theta, phi, rng, amp, terms=((2, 1), (3, 2), (4, 1))):
    """Low-frequency multiplicative angular perturbation -> organic (non-spherical)
    surface. amp=0 -> exactly 1.0 everywhere (a perfect sphere)."""
    out = np.ones_like(theta)
    if amp <= 0.0:
        return out
    for l, m in terms:
        a = rng.uniform(-amp, amp)
        d_theta = rng.uniform(0, 2 * np.pi)
        d_phi = rng.uniform(0, 2 * np.pi)
        out = out + a * np.cos(l * theta + d_theta) * np.cos(m * phi + d_phi)
    return np.clip(out, 0.4, 1.6)


def make_blob(shape, center, params, rng):
    """Roughly round organic blob: a harmonic-perturbed sphere. params: size_vox
    (absolute target volume, voxels), roughness (angular perturbation amplitude,
    0=perfect sphere, ~0.3=quite irregular)."""
    base_r = _radius_for_volume(params["size_vox"])
    rr, theta, phi = _grid_dist_angle(shape, center)
    r_dir = base_r * _harmonics(theta, phi, rng, params.get("roughness", 0.15))
    mask = (rr <= r_dir).astype(np.uint8)
    return mask, {"family": "blob", "realized_size_frac": float(mask.mean())}


def make_splatter(shape, center, params, rng):
    """Scattered cluster of several small blobs under one label -- 3D analog of
    controlSynth's shapes/scattered.py. A distinct failure mode from a single blob: the
    model has to find ALL components, not just the nearest one. params: size_vox
    (absolute total target volume, voxels), n_components, spread_vox (cluster jitter
    std, voxels), roughness (per-component)."""
    n = max(1, int(round(params.get("n_components", 4))))
    spread = max(0.0, float(params.get("spread_vox", 0.0)))
    per_component_vox = float(params["size_vox"]) / n
    comp_r = _radius_for_volume(per_component_vox)
    mask = np.zeros(shape, dtype=bool)
    upper = np.array(shape, dtype=np.float64) - 1.0
    for _ in range(n):
        offset = rng.normal(0.0, spread, size=3) if spread > 0 else np.zeros(3)
        c = np.clip(np.asarray(center, dtype=np.float64) + offset, 0.0, upper)
        rr, theta, phi = _grid_dist_angle(shape, c)
        r_dir = comp_r * _harmonics(theta, phi, rng, params.get("roughness", 0.15))
        mask |= (rr <= r_dir)
    mask = mask.astype(np.uint8)
    return mask, {"family": "splatter", "realized_size_frac": float(mask.mean()),
                  "n_components": n}


def make_disk(shape, center, params, rng):
    """Flattened ellipsoid: one axis scaled by `aspect_ratio` (<1 flattens it). params:
    size_vox (absolute target volume, voxels), aspect_ratio (flattened-axis semi-length
    / other-axis semi-length), flatten_axis (0/1/2, which grid axis is flattened)."""
    axis = int(params.get("flatten_axis", 0)) % 3
    aspect = float(np.clip(params.get("aspect_ratio", 0.25), 0.05, 0.95))
    # ellipsoid volume = 4/3 pi * r^2 * (aspect*r) = aspect * sphere_volume(r)
    r = (3.0 * max(1.0, float(params["size_vox"])) / (4.0 * np.pi * aspect)) ** (1.0 / 3.0)
    semi = [r, r, r]
    semi[axis] = r * aspect
    D, H, W = shape
    dd, hh, ww = np.mgrid[0:D, 0:H, 0:W].astype(np.float64)
    dz, dy, dx = dd - center[0], hh - center[1], ww - center[2]
    val = (dz / semi[0]) ** 2 + (dy / semi[1]) ** 2 + (dx / semi[2]) ** 2
    mask = (val <= 1.0).astype(np.uint8)
    return mask, {"family": "disk", "realized_size_frac": float(mask.mean()),
                  "flatten_axis": axis, "aspect_ratio": aspect}


def make_cylinder(shape, center, params, rng):
    """Capsule: voxels within `radius` of a line segment of length `length_vox` through
    `center`, oriented by (azimuth, elevation). params: size_vox (absolute target
    volume, voxels; radius is solved from size_vox and length_vox so the two knobs
    don't fight), length_vox (absolute segment length, voxels), azimuth, elevation
    (radians)."""
    D, H, W = shape
    az = float(params["azimuth"]) if "azimuth" in params else float(rng.uniform(0, 2 * np.pi))
    el = float(params["elevation"]) if "elevation" in params else float(rng.uniform(-np.pi / 2, np.pi / 2))
    direction = np.array([np.sin(el), np.cos(el) * np.sin(az), np.cos(el) * np.cos(az)])
    length = max(1.0, float(params.get("length_vox", 1.0)))
    # cylinder-only approx (end-cap volume is a small correction at these aspect ratios;
    # the REALIZED fraction below is measured from the actual rasterized mask, not this).
    radius = max(0.75, float(np.sqrt(max(1.0, float(params["size_vox"])) / (np.pi * length))))

    c = np.asarray(center, dtype=np.float64)
    p0, p1 = c - 0.5 * length * direction, c + 0.5 * length * direction
    dd, hh, ww = np.mgrid[0:D, 0:H, 0:W].astype(np.float64)
    pts = np.stack([dd, hh, ww], axis=-1)
    seg = p1 - p0
    seg_len2 = float(seg @ seg) or _EPS
    t = np.clip(((pts - p0) @ seg) / seg_len2, 0.0, 1.0)
    closest = p0 + t[..., None] * seg
    dist = np.linalg.norm(pts - closest, axis=-1)
    mask = (dist <= radius).astype(np.uint8)
    return mask, {"family": "cylinder", "realized_size_frac": float(mask.mean()),
                  "radius": radius, "length": length}


def reach_vox(family, params):
    """Generous local-voxel bounding radius for this shape's rasterized footprint,
    given the SAME `params` a make_<family> call would use -- independent of any
    specific `shape` array (unlike each make_<family>'s own internal math, this only
    needs an upper bound, not the exact mask). Used by
    instantiate.rasterize_shape_in_crop to size a rasterization sub-box instead of
    materializing the full crop -- the shape only ever occupies a small fraction of a
    typical crop, so this is a large, cheap win. Margins are generous on purpose (a
    too-small sub-box silently clips the shape; a too-large one only costs a bit of
    wasted compute)."""
    size_vox = float(params["size_vox"])
    if family == "blob":
        return _radius_for_volume(size_vox) * 1.6          # harmonic clip max (see _harmonics)
    if family == "splatter":
        n = max(1, int(round(params.get("n_components", 4))))
        comp_r = _radius_for_volume(size_vox / n)
        spread = max(0.0, float(params.get("spread_vox", 0.0)))
        return 3.0 * spread + comp_r * 1.6                  # 3-sigma jitter + harmonic clip
    if family == "disk":
        aspect = float(np.clip(params.get("aspect_ratio", 0.25), 0.05, 0.95))
        return (3.0 * max(1.0, size_vox) / (4.0 * np.pi * aspect)) ** (1.0 / 3.0)
    if family == "cylinder":
        length = max(1.0, float(params.get("length_vox", 1.0)))
        radius = max(0.75, float(np.sqrt(max(1.0, size_vox) / (np.pi * length))))
        return length / 2.0 + radius
    raise ValueError(f"unknown shape family {family!r}")


_FAMILIES = {"blob": make_blob, "splatter": make_splatter, "disk": make_disk,
            "cylinder": make_cylinder}


def make_shape(family, shape, center, params, rng):
    """Dispatch to make_<family>. Raises ValueError for an unknown family name."""
    fn = _FAMILIES.get(family)
    if fn is None:
        raise ValueError(f"unknown shape family {family!r} (expected one of {sorted(_FAMILIES)})")
    return fn(shape, center, params, rng)
