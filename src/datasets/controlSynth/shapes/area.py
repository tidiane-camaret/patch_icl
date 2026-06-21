"""
Area enforcement (spec ss10.5): scale a shape to a target foreground fraction
BEFORE deformation, so size and morphology don't couple through layout.

Generic fallback used by the blob family (vessel/scattered scale internally).
Zooms the mask about its centroid to the target area, then fine-tunes with a few
dilation/erosion steps.
"""

import numpy as np


def enforce_area_fraction(mask, target_frac, max_tune=4):
    """Scale `mask` (uint8 [H,W]) so foreground area ~= target_frac. Returns uint8."""
    from scipy.ndimage import affine_transform, binary_dilation, binary_erosion

    m = mask.astype(bool)
    cur = m.sum()
    H, W = m.shape
    target_px = float(target_frac) * m.size
    if cur == 0 or target_px <= 0:
        return mask.astype(np.uint8)

    # Zoom about centroid by s = sqrt(target/current).
    s = float(np.sqrt(target_px / cur))
    s = float(np.clip(s, 0.2, 5.0))
    cy, cx = np.argwhere(m).mean(axis=0)
    # affine_transform maps output->input: input = M @ out + offset, with M = 1/s I.
    inv = 1.0 / s
    matrix = np.array([[inv, 0.0], [0.0, inv]])
    offset = np.array([cy - inv * cy, cx - inv * cx])
    zoomed = affine_transform(m.astype(np.float32), matrix, offset=offset,
                              order=0, output_shape=(H, W)) > 0.5

    # Fine-tune residual area mismatch.
    for _ in range(max_tune):
        c = zoomed.sum()
        if c == 0 or 0.9 * target_px <= c <= 1.12 * target_px:
            break
        if c < target_px:
            zoomed = binary_dilation(zoomed)
        else:
            e = binary_erosion(zoomed)
            if e.sum() == 0:
                break
            zoomed = e
    return zoomed.astype(np.uint8)
