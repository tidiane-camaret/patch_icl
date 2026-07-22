"""Placement and compositing helpers for anchor_synth3d.

Pure numpy (no dataset/IO deps), so unit-testable with trivial arrays. The
object is alpha-composited toward the local background mean plus a small contrast
delta with soft edges, so it blends into the CT and is only findable via the
anchor-relative position the K contexts demonstrate.
"""

import numpy as np


def anchor_stats(mask):
    """(centroid, extent, (lo, hi)) of a binary mask via axis projections.

    centroid = bbox centre; extent = per-axis bbox side length. None if empty.
    Cheap (no full nonzero / scipy) — projects the mask onto each axis.
    """
    m = mask > 0
    if not m.any():
        return None
    lo = np.empty(3, dtype=np.int64)
    hi = np.empty(3, dtype=np.int64)
    for ax in range(3):
        proj = m.any(axis=tuple(a for a in range(3) if a != ax))
        idx = np.nonzero(proj)[0]
        lo[ax], hi[ax] = int(idx[0]), int(idx[-1])
    centroid = (lo + hi) / 2.0
    extent = (hi - lo + 1).astype(np.float64)
    return centroid, extent, (lo, hi)


def affine_weights(rng, n, extrapolation=0.0, concentration=1.0):
    """`n` barycentric weights summing to 1. Base convex `u ~ Dirichlet`, expanded
    around the barycenter 1/n by (1+extrapolation) so weights may go mildly negative
    (extrapolation=0 -> strictly inside the hull)."""
    u = rng.dirichlet([float(concentration)] * int(n))
    b = 1.0 / int(n)
    return b + (1.0 + float(extrapolation)) * (u - b)


def frame_length(centroids):
    """Mean pairwise Euclidean distance of centroids (n,3) — an orientation- and
    translation-invariant characteristic length of the landmark frame."""
    c = np.asarray(centroids, dtype=np.float64)
    n = len(c)
    if n < 2:
        return 0.0
    diffs = c[:, None, :] - c[None, :, :]
    d = np.sqrt((diffs ** 2).sum(-1))
    iu = np.triu_indices(n, k=1)
    return float(d[iu].mean())


def barycentric_center(centroids, weights, tile_size, vol_shape):
    """Voxel centre = Σ wᵢ·centroidᵢ, clamped so a `tile_size` cube stays fully
    inside `vol_shape`."""
    c = np.asarray(centroids, dtype=np.float64)          # (n, 3)
    w = np.asarray(weights, dtype=np.float64)            # (n,)
    center = (w[:, None] * c).sum(0)                     # (3,)
    half = tile_size / 2.0
    return np.clip(center, half, np.asarray(vol_shape, dtype=np.float64) - half)


def _slices_3d(t, cz, cy, cx, D, H, W):
    """(canvas_slices, tile_slices) for a t^3 tile centred at (cz,cy,cx) clipped to
    a D×H×W volume; None if fully out of bounds."""
    oz, oy, ox = cz - t // 2, cy - t // 2, cx - t // 2
    dz0, dy0, dx0 = max(0, oz), max(0, oy), max(0, ox)
    dz1, dy1, dx1 = min(D, oz + t), min(H, oy + t), min(W, ox + t)
    if dz0 >= dz1 or dy0 >= dy1 or dx0 >= dx1:
        return None
    return ((slice(dz0, dz1), slice(dy0, dy1), slice(dx0, dx1)),
            (slice(dz0 - oz, dz1 - oz), slice(dy0 - oy, dy1 - oy),
             slice(dx0 - ox, dx1 - ox)))


def place_object(image, alpha, center, contrast_delta, label=None, label_id=1):
    """Alpha-composite `alpha` into `image` at voxel `center`, blending toward the
    local background mean + contrast_delta. Writes alpha>0.5 into `label` with
    `label_id` when given. Mutates `image`/`label`; returns the bool footprint."""
    t = alpha.shape[0]
    c = np.round(np.asarray(center)).astype(int)
    footprint = np.zeros(image.shape, dtype=bool)
    sl = _slices_3d(t, int(c[0]), int(c[1]), int(c[2]), *image.shape)
    if sl is None:
        return footprint
    cs, ts = sl
    a = alpha[ts]
    core = a > 0.5
    region = image[cs]
    bg = float(region[core].mean()) if core.any() else float(region.mean())
    target_val = bg + float(contrast_delta)
    region[:] = region * (1.0 - a) + target_val * a
    footprint[cs] = core
    if label is not None:
        lreg = label[cs]
        lreg[core] = label_id
    return footprint
