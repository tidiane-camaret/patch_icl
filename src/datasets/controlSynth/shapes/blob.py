"""
Blob family: blob, elongated, annular. Cheap, fully on-the-fly.

All three are perturbed-ellipse fields rasterized at a moderate radius; final
area is fixed later by area.enforce_area_fraction, so radius here only needs to
be in a sane range. Each returns (mask uint8 [H,W], realized_meta dict).
"""

import numpy as np


def _grid(image_size, center, rotation):
    """Rotated (xr, yr) coordinates and polar (radius, angle) about `center`."""
    H = W = image_size
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    dy, dx = yy - center[0], xx - center[1]
    ca, sa = np.cos(rotation), np.sin(rotation)
    xr = ca * dx + sa * dy
    yr = -sa * dx + ca * dy
    rr = np.sqrt(xr * xr + yr * yr)
    ang = np.arctan2(yr, xr)
    return xr, yr, rr, ang


def _radial_harmonics(ang, rng, n=(2, 3, 4), amp=0.12):
    """Low-frequency multiplicative radial perturbation -> organic (not circular)."""
    out = np.ones_like(ang)
    for k in n:
        a = rng.uniform(-amp, amp)
        phi = rng.uniform(0, 2 * np.pi)
        out = out + a * np.cos(k * ang + phi)
    return np.clip(out, 0.4, 1.6)


def _center(image_size, rng, margin=0.28):
    """Random center kept away from the border so shapes usually fit."""
    lo, hi = margin * image_size, (1 - margin) * image_size
    return (rng.uniform(lo, hi), rng.uniform(lo, hi))


def make_blob(image_size, params, rng):
    """Roughly round organic blob (low-harmonic radial perturbation)."""
    center = _center(image_size, rng)
    base_r = image_size * 0.18
    _, _, rr, ang = _grid(image_size, center, rng.uniform(0, 2 * np.pi))
    r_theta = base_r * _radial_harmonics(ang, rng)
    mask = (rr <= r_theta).astype(np.uint8)
    return mask, {"morphology": "blob", "center": center}


def make_elongated(image_size, params, rng):
    """Eccentric ellipse at a random orientation."""
    center = _center(image_size, rng)
    rotation = rng.uniform(0, 2 * np.pi)
    ecc = rng.uniform(2.5, 4.5)
    ax = image_size * 0.22
    by = ax / ecc
    xr, yr, _, ang = _grid(image_size, center, rotation)
    scale = _radial_harmonics(ang, rng, amp=0.08)
    mask = (((xr / (ax * scale)) ** 2 + (yr / (by * scale)) ** 2) <= 1.0).astype(np.uint8)
    return mask, {"morphology": "elongated", "center": center, "eccentricity": ecc}


def make_annular(image_size, params, rng):
    """Ring / shell. The interior hole is a built-in distractor (spec ss10.5)."""
    center = _center(image_size, rng)
    _, _, rr, ang = _grid(image_size, center, rng.uniform(0, 2 * np.pi))
    r_out = image_size * 0.20 * _radial_harmonics(ang, rng, amp=0.06)
    thickness = rng.uniform(0.30, 0.55)          # fraction of outer radius
    r_in = r_out * (1.0 - thickness)
    mask = ((rr <= r_out) & (rr >= r_in)).astype(np.uint8)
    return mask, {"morphology": "annular", "center": center,
                  "ring_thickness": float(thickness)}
