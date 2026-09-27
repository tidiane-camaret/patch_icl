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


def make_scatter_field(shape, center, params, rng):
    """Many small independent blobs scattered over a WIDE region -- the many-instance
    analog of make_splatter, added to close a measured real-vs-synthetic gap: real
    multi-focal lesion fields (MS plaques, scattered emboli) carry 10-100+ components
    per subject (docs/logs.md 2026-09-26 "inspecting the 4 worst 135b OOD sources"),
    far more than splatter_n_components_range's cap of 6. Unlike make_splatter (which
    computes the FULL local `shape` grid once per component -- fine at n<=6, expensive
    at n~100), each component here gets its OWN small local sub-box sized to its own
    radius, cheap even at large n since per-component cost is O(comp_r^3) not
    O(prod(shape)). params: size_vox (total target volume, voxels, split
    log-normally across components so instance sizes vary -- real fields aren't
    uniform microscopic copies), n_components, spread_vox (jitter std, voxels; meant
    to be much larger than splatter's -- a real scattered field spans a good fraction
    of an organ, not a tight local cluster), roughness (per-component)."""
    n = max(1, int(round(params.get("n_components", 20))))
    spread = max(0.0, float(params.get("spread_vox", 0.0)))
    total_vox = max(1.0, float(params["size_vox"]))
    roughness = params.get("roughness", 0.15)
    # log-normal split: heterogeneous component sizes (a few larger, a long tail of
    # smaller ones) rather than n identical copies -- cheap realism, sigma fixed (not
    # cohort-configurable, this is a within-shape microstructure detail, not a
    # cohort-level axis the way n_components/spread/roughness are).
    raw = rng.lognormal(mean=0.0, sigma=0.6, size=n)
    per_component_vox = total_vox * raw / raw.sum()

    shape_arr = np.array(shape, dtype=np.int64)
    upper = shape_arr.astype(np.float64) - 1.0
    center = np.asarray(center, dtype=np.float64)
    mask = np.zeros(shape, dtype=bool)
    for i in range(n):
        offset = rng.normal(0.0, spread, size=3) if spread > 0 else np.zeros(3)
        c = center + offset
        comp_r = _radius_for_volume(per_component_vox[i])
        reach = comp_r * 1.6 + 1.0
        lo = np.clip(np.floor(c - reach), 0, upper).astype(np.int64)
        hi = np.clip(np.ceil(c + reach) + 1, 0, shape_arr).astype(np.int64)
        sub_shape = tuple((hi - lo).tolist())
        if any(s <= 0 for s in sub_shape):
            continue   # this component's local box misses the crop entirely
        local_center = tuple((c - lo).tolist())
        rr, theta, phi = _grid_dist_angle(sub_shape, local_center)
        r_dir = comp_r * _harmonics(theta, phi, rng, roughness)
        sl = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
        mask[sl] |= (rr <= r_dir)
    mask = mask.astype(np.uint8)
    return mask, {"family": "scatter_field", "realized_size_frac": float(mask.mean()),
                  "n_components": n}


def _rotate_axis_angle(v, axis, angle):
    """Rodrigues' rotation of unit vector `v` by `angle` radians around unit `axis`."""
    axis = axis / (np.linalg.norm(axis) + _EPS)
    return (v * np.cos(angle) + np.cross(axis, v) * np.sin(angle)
            + axis * np.dot(axis, v) * (1.0 - np.cos(angle)))


def _perturb_direction(rng, direction, max_angle):
    """A new unit direction within `max_angle` radians of `direction`, rotated around a
    random perpendicular axis -- cheap way to get a plausible-looking branch direction
    without a full 3D orientation parameterization."""
    # any vector not parallel to `direction` works to build a perpendicular axis
    helper = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(helper, direction)) > 0.9:
        helper = np.array([0.0, 1.0, 0.0])
    perp = np.cross(direction, helper)
    perp /= np.linalg.norm(perp) + _EPS
    swing_angle = rng.uniform(0.0, max_angle)
    swirl = rng.uniform(0, 2 * np.pi)
    axis = _rotate_axis_angle(perp, direction, swirl)
    return _rotate_axis_angle(direction, axis, swing_angle)


_VESSEL_MAX_LENGTH_FALLOFF = 0.8   # matches the rng.uniform(0.55, 0.8) upper bound below;
                                    # kept as a named constant so reach_vox's geometric-series
                                    # bound stays a genuine upper bound if that range ever moves


def _vessel_segments(rng, direction, length, radius, depth, radius_falloff):
    """[(p0, p1, radius), ...] segments of a recursively-bifurcating tapering tube
    tree, rooted at the origin -- cheap L-system-like vessel/vein generator (no mesh,
    no marching cubes, just line segments later rasterized as capsules). Recursion
    depth is small (spec default 2-4) so the segment count stays tiny (<= 2**(depth+1)
    - 1, e.g. 31 at depth=4)."""
    segments = [(np.zeros(3), direction * length, radius)]
    if depth <= 0 or radius < 0.6:
        return segments
    tip = direction * length
    for _ in range(2):   # bifurcation
        child_dir = _perturb_direction(rng, direction, max_angle=0.7)
        child_length = length * rng.uniform(0.55, _VESSEL_MAX_LENGTH_FALLOFF)
        child_radius = radius * radius_falloff * rng.uniform(0.85, 1.15)
        for p0, p1, r in _vessel_segments(rng, child_dir, child_length, child_radius,
                                          depth - 1, radius_falloff):
            segments.append((tip + p0, tip + p1, r))
    return segments


def make_vessel(shape, center, params, rng):
    """Thin branching tubular network -- a cheap vein/vessel stand-in built from a
    recursively-bifurcating, tapering tree of capsule segments (see _vessel_segments),
    each rasterized into its OWN small local sub-box (like make_scatter_field) rather
    than the shared full `shape` grid, so branch count/depth stays cheap. params:
    size_vox (trunk target volume; trunk radius solved from size_vox/length exactly
    like make_cylinder), length_vox (trunk length), azimuth/elevation (trunk
    direction), branch_depth (recursion depth), radius_falloff (child/parent radius
    ratio per generation, Murray's-law-like tapering)."""
    az = float(params.get("azimuth", 0.0))
    el = float(params.get("elevation", 0.0))
    direction = np.array([np.sin(el), np.cos(el) * np.sin(az), np.cos(el) * np.cos(az)])
    length = max(1.0, float(params.get("length_vox", 1.0)))
    trunk_radius = max(0.75, float(np.sqrt(max(1.0, float(params["size_vox"])) / (np.pi * length))))
    depth = max(0, int(round(params.get("branch_depth", 3))))
    radius_falloff = float(np.clip(params.get("radius_falloff", 0.75), 0.4, 0.95))

    center = np.asarray(center, dtype=np.float64)
    # root the tree so its trunk is centered on `center` (matches every other family's
    # convention of `center` being the object's own centroid, not one endpoint).
    root = center - 0.5 * length * direction
    segments = _vessel_segments(rng, direction, length, trunk_radius, depth, radius_falloff)

    shape_arr = np.array(shape, dtype=np.int64)
    upper = shape_arr.astype(np.float64) - 1.0
    mask = np.zeros(shape, dtype=bool)
    for p0_local, p1_local, r in segments:
        p0, p1 = root + p0_local, root + p1_local
        seg_lo = np.minimum(p0, p1) - r - 1.0
        seg_hi = np.maximum(p0, p1) + r + 1.0
        lo = np.clip(np.floor(seg_lo), 0, upper).astype(np.int64)
        hi = np.clip(np.ceil(seg_hi) + 1, 0, shape_arr).astype(np.int64)
        sub_shape = tuple((hi - lo).tolist())
        if any(s <= 0 for s in sub_shape):
            continue
        p0_l, p1_l = p0 - lo, p1 - lo
        D, H, W = sub_shape
        dd, hh, ww = np.mgrid[0:D, 0:H, 0:W].astype(np.float64)
        pts = np.stack([dd, hh, ww], axis=-1)
        seg = p1_l - p0_l
        seg_len2 = float(seg @ seg) or _EPS
        t = np.clip(((pts - p0_l) @ seg) / seg_len2, 0.0, 1.0)
        closest = p0_l + t[..., None] * seg
        dist = np.linalg.norm(pts - closest, axis=-1)
        sl = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
        mask[sl] |= (dist <= r)
    mask = mask.astype(np.uint8)
    return mask, {"family": "vessel", "realized_size_frac": float(mask.mean()),
                  "n_segments": len(segments)}


def make_torus(shape, center, params, rng):
    """Ring / donut: axis-aligned torus SDF (axis = `flatten_axis`, matching make_disk's
    convention of picking one grid axis rather than a general 3D orientation -- cheap
    and sufficient for shape-diversity purposes). Non-convex, genus-1 topology -- a
    deliberately NON-anatomical shape (spec: shapes need not be plausible lesions) that
    exercises "does the model handle a hole in the middle", cheap to add per the user's
    own "segments, torus, etc." suggestion. params: size_vox (target volume; major/
    minor radius solved in closed form from size_vox and `ratio`), ratio (major/minor
    radius ratio -- >1 always, larger = thinner ring), flatten_axis (0/1/2, the torus's
    symmetry axis)."""
    axis = int(params.get("flatten_axis", 0)) % 3
    ratio = max(1.05, float(params.get("ratio", 3.0)))
    # torus volume = 2*pi^2 * R * r^2, r = R/ratio -> R^3 = size_vox*ratio^2/(2*pi^2)
    size_vox = max(1.0, float(params["size_vox"]))
    R = (size_vox * ratio ** 2 / (2.0 * np.pi ** 2)) ** (1.0 / 3.0)
    r = R / ratio
    D, H, W = shape
    dd, hh, ww = np.mgrid[0:D, 0:H, 0:W].astype(np.float64)
    d = [dd, hh, ww]
    dz = d[axis] - center[axis]
    plane_axes = [a for a in range(3) if a != axis]
    du = d[plane_axes[0]] - center[plane_axes[0]]
    dv = d[plane_axes[1]] - center[plane_axes[1]]
    q = np.sqrt(du * du + dv * dv) - R
    dist = np.sqrt(q * q + dz * dz)
    mask = (dist <= r).astype(np.uint8)
    return mask, {"family": "torus", "realized_size_frac": float(mask.mean()),
                  "flatten_axis": axis, "ratio": ratio, "R": R, "r": r}


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
    if family == "scatter_field":
        n = max(1, int(round(params.get("n_components", 20))))
        # generous per-component upper bound (~p99 of the lognormal(sigma=0.6) split
        # used by make_scatter_field, not the mean share) so a rare large-share draw
        # doesn't silently clip.
        comp_r = _radius_for_volume(5.0 * size_vox / n)
        spread = max(0.0, float(params.get("spread_vox", 0.0)))
        return 3.0 * spread + comp_r * 1.6
    if family == "vessel":
        length = max(1.0, float(params.get("length_vox", 1.0)))
        radius = max(0.75, float(np.sqrt(max(1.0, size_vox) / (np.pi * length))))
        depth = max(0, int(round(params.get("branch_depth", 3))))
        # geometric series bound on how far a leaf branch can extend beyond the trunk
        # tip, using the max per-generation length_falloff make_vessel actually draws
        # from (_VESSEL_MAX_LENGTH_FALLOFF) -- a genuine upper bound, not a guess.
        extra = length * sum(_VESSEL_MAX_LENGTH_FALLOFF ** k for k in range(1, depth + 1))
        return length / 2.0 + extra + radius
    if family == "torus":
        ratio = max(1.05, float(params.get("ratio", 3.0)))
        R = (max(1.0, size_vox) * ratio ** 2 / (2.0 * np.pi ** 2)) ** (1.0 / 3.0)
        r = R / ratio
        return (R + r) * 1.1
    raise ValueError(f"unknown shape family {family!r}")


_FAMILIES = {"blob": make_blob, "splatter": make_splatter, "disk": make_disk,
            "cylinder": make_cylinder, "scatter_field": make_scatter_field,
            "vessel": make_vessel, "torus": make_torus}


def make_shape(family, shape, center, params, rng):
    """Dispatch to make_<family>. Raises ValueError for an unknown family name."""
    fn = _FAMILIES.get(family)
    if fn is None:
        raise ValueError(f"unknown shape family {family!r} (expected one of {sorted(_FAMILIES)})")
    return fn(shape, center, params, rng)
