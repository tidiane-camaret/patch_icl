"""
Scattered morphology (build-time): many small components from a point process.

`clustering` interpolates the point process:
  0.0 -> hardcore  (min-distance rejection, regular layout)
  0.5 -> Poisson   (uniform random)
  1.0 -> clustered (Thomas process: few parents, children scattered nearby)

Loads on BOTH difficulty axes (spec ss3.3): many small components raise
segmentation difficulty, and "all such components everywhere" raises
identification difficulty -> meta flags this.
"""

import numpy as np

from ..config import map_region_size


def _centers(image_size, count, clustering, rng):
    margin = 0.12 * image_size
    lo, hi = margin, image_size - margin
    if clustering < 0.5:
        # hardcore <-> Poisson: shrink the enforced min-distance toward 0.
        t = clustering / 0.5
        min_dist = (1.0 - t) * (image_size * 0.22)
        pts = []
        tries = 0
        while len(pts) < count and tries < count * 40:
            c = rng.uniform(lo, hi, size=2)
            if all(np.linalg.norm(c - p) >= min_dist for p in pts):
                pts.append(c)
            tries += 1
        return np.array(pts) if pts else rng.uniform(lo, hi, size=(count, 2))
    # Poisson <-> clustered: Thomas process with tightening child spread.
    t = (clustering - 0.5) / 0.5
    n_parents = max(1, int(round(count * (1.0 - 0.8 * t))))
    parents = rng.uniform(lo, hi, size=(n_parents, 2))
    spread = image_size * (0.18 * (1.0 - t) + 0.03)
    pts = []
    for j in range(count):
        par = parents[j % n_parents]
        pts.append(np.clip(par + rng.normal(0, spread, size=2), lo, hi))
    return np.array(pts)


def make_scattered(image_size, params, rng):
    """Stamp `count` small blobs/ellipses; scale to region_size. (mask, meta)."""
    count = int(params.get("scattered_count", 8))
    clustering = float(params.get("scattered_clustering", 0.0))
    size_dispersion = float(params.get("size_dispersion", 0.4))

    centers = _centers(image_size, count, clustering, rng)
    H = W = image_size
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    mask = np.zeros((H, W), bool)

    base_r = image_size * 0.030
    for c in centers:
        r = base_r * np.exp(rng.normal(0, size_dispersion))
        ecc = rng.uniform(1.0, 1.8)
        rot = rng.uniform(0, np.pi)
        ca, sa = np.cos(rot), np.sin(rot)
        dy, dx = yy - c[0], xx - c[1]
        xr = ca * dx + sa * dy
        yr = -sa * dx + ca * dy
        mask |= ((xr / (r * ecc)) ** 2 + (yr / (r / ecc)) ** 2) <= 1.0

    target = map_region_size(params.get("region_size", 0.15)) * mask.size
    mask = _scale_components(mask, centers, image_size, target, rng,
                             base_r, size_dispersion)

    realized_area = float(mask.sum()) / mask.size
    return mask.astype(np.uint8), {
        "morphology": "scattered", "n_components": int(len(centers)),
        "clustering": clustering, "realized_area": realized_area,
        "dual_axis": True,   # flags segmentation + identification loading (spec ss3.3)
    }


def _scale_components(mask, centers, image_size, target_px, rng, base_r, disp,
                      max_iter=6):
    """Grow/shrink component radii multiplicatively to hit target area."""
    from scipy.ndimage import binary_dilation, binary_erosion
    m = mask.astype(bool)
    for _ in range(max_iter):
        cur = m.sum()
        if cur == 0 or 0.85 * target_px <= cur <= 1.18 * target_px:
            break
        if cur < target_px:
            m = binary_dilation(m)
        else:
            e = binary_erosion(m)
            if e.sum() == 0:
                break
            m = e
    return m
