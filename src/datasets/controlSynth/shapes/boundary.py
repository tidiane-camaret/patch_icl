"""
Boundary roughness, DECOUPLED from morphology (spec ss10.5).

`set_boundary_complexity` perturbs any mask's signed-distance field with smoothed
noise and re-thresholds, so the same knob roughens a blob, a vessel, or a
scattered set identically. Roughness frequency is fixed; amplitude scales with c.
"""

import numpy as np

from ..config import map_boundary_roughness


def set_boundary_complexity(mask, c, rng, amp_scale=1.0, sigma_frac=0.0,
                            keep_largest=False):
    """Return a roughened copy of `mask` (uint8). c in [0,1]; 0 -> unchanged.

    The boundary is perturbed by adding smoothed noise to the signed-distance field
    and re-thresholding. Realism knobs (all default to the original behavior):
      `amp_scale` (<1) shrinks the perturbation amplitude so the contour ripples
        instead of detaching islands -- the original `amp = 0.35*c*sqrt(area)` is
        large enough to carve a small/thin shape into many pieces.
      `sigma_frac` (>0) sets the blur to `sqrt(area)*sigma_frac` (low-frequency
        relative to the object) instead of the fixed sigma=1.6, so the perturbation
        shifts the whole boundary in/out coherently rather than punching holes.
      `keep_largest` keeps only the largest connected component after thresholding
        (cleans up any residual detached specks). Caller restricts this to simply
        connected morphologies (blob/elongated/annular) -- never scattered/tubular.
    """
    from scipy.ndimage import distance_transform_edt, gaussian_filter, label

    c = float(c)
    if c <= 0.0 or mask.sum() == 0:
        return mask.astype(np.uint8)

    m = mask.astype(bool)
    sdf = distance_transform_edt(m) - distance_transform_edt(~m)

    char = np.sqrt(m.sum())                          # characteristic radius
    sigma = max(1.6, char * sigma_frac) if sigma_frac > 0.0 else 1.6
    noise = rng.standard_normal(m.shape)
    noise = gaussian_filter(noise, sigma=sigma)      # roughness frequency
    std = noise.std()
    if std > 1e-8:
        noise = noise / std

    # Amplitude relative to object scale (sqrt area ~ characteristic radius).
    amp = map_boundary_roughness(c) * char * float(amp_scale)
    rough = (sdf + amp * noise) > 0.0

    if keep_largest and rough.any():
        lbl, n = label(rough)
        if n > 1:
            largest = 1 + int(np.argmax([(lbl == k).sum() for k in range(1, n + 1)]))
            rough = lbl == largest
    return rough.astype(np.uint8)
