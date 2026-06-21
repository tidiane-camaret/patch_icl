"""
Per-subject elastic deformation (the cheap CPU path; runs in dataloader workers).

A smoothed random displacement field warps the base geometry, giving each subject
(target + each context) a distinct realization. `support_query_shift` sets the
magnitude of the context<->query gap (spec ss10.7). Masks use nearest-neighbour
interpolation (order=0) so labels stay integer-valued.
"""

import numpy as np

from .config import map_deform_sigma


def deform(label_map, shift, rng, image_size=None):
    """Warp a uint8 label map by a smoothed random displacement field.

    Returns a uint8 array of the same shape. `shift` in [0,1] -> displacement
    magnitude via map_deform_sigma. shift<=0 returns the input unchanged.
    """
    from scipy.ndimage import gaussian_filter, map_coordinates

    H, W = label_map.shape
    mag = map_deform_sigma(shift, image_size or H)
    if mag <= 0.0:
        return label_map.astype(np.uint8)

    smooth = max(H, W) * 0.10                         # field smoothness (coherent warp)
    dy = gaussian_filter(rng.standard_normal((H, W)), smooth)
    dx = gaussian_filter(rng.standard_normal((H, W)), smooth)
    # normalize each field to unit std, then scale to the requested magnitude
    dy = dy / (dy.std() + 1e-8) * mag
    dx = dx / (dx.std() + 1e-8) * mag

    yy, xx = np.mgrid[0:H, 0:W]
    coords = np.array([yy + dy, xx + dx])
    warped = map_coordinates(label_map, coords, order=0, mode="nearest")
    return warped.astype(np.uint8)


def jitter_pose(label_map, translate, scale, fg_label, rng, image_size):
    """Per-subject translation + isotropic scale of the label map about the fg centroid.

    Adds WITHIN-(target+context)-set position/size spread: `deform` only wobbles one
    shared base shape, so synth context sets are far more self-similar than real ones
    (real centroid_dist 0.16 / area_logratio 1.2 vs synth 0.10 / 0.57). `translate` is the
    std of the centroid shift (fraction of image); `scale` is the std of the log2 zoom.
    Both <= 0 -> unchanged. The shift is clamped to keep the (scaled) fg bbox in frame, so
    the foreground never lands off-frame (cf. the same concern in task.place_foreground).
    """
    from scipy.ndimage import affine_transform

    translate, scale = float(translate), float(scale)
    fg = label_map == fg_label
    if (translate <= 0.0 and scale <= 0.0) or not fg.any():
        return label_map
    H = W = image_size
    ys, xs = np.nonzero(fg)
    cy, cx = ys.mean(), xs.mean()
    s = float(np.clip(2.0 ** rng.normal(0.0, scale), 0.6, 1.6)) if scale > 0.0 else 1.0
    ty = rng.normal(0.0, translate) * H if translate > 0.0 else 0.0
    tx = rng.normal(0.0, translate) * W if translate > 0.0 else 0.0
    # Output fg bbox = centroid + shift + scaled half-extents; clamp shift to keep it in frame.
    inset = 2
    hy0, hy1 = (ys.min() - cy) * s, (ys.max() - cy) * s
    hx0, hx1 = (xs.min() - cx) * s, (xs.max() - cx) * s
    lo_y, hi_y = inset - cy - hy0, (H - 1) - inset - cy - hy1
    lo_x, hi_x = inset - cx - hx0, (W - 1) - inset - cx - hx1
    ty = float(np.clip(ty, lo_y, hi_y)) if lo_y <= hi_y else 0.0
    tx = float(np.clip(tx, lo_x, hi_x)) if lo_x <= hi_x else 0.0
    # affine_transform maps output->input: in = (1/s)·out + offset, scaling about centroid.
    inv = 1.0 / s
    matrix = np.array([[inv, 0.0], [0.0, inv]])
    offset = np.array([cy - (cy + ty) * inv, cx - (cx + tx) * inv])
    out = affine_transform(label_map, matrix, offset=offset, order=0,
                           output_shape=(H, W), mode="nearest")
    return out.astype(label_map.dtype)
