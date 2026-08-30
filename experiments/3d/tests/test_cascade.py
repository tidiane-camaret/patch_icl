"""Tasks 4-5,7: cascade.py — invert_geo_center, run_cascade, _cascade_loss."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # experiments/3d siblings

import numpy as np
import torch

from cascade import invert_geo_center
from evaluate import _predicted_native_center, _grid_centroid


def _geom(starts=(10, 20, 30), crop=(8, 8, 8), out=(8, 8, 8), pad=(0, 0, 0)):
    return torch.tensor([list(starts), list(crop), list(out), list(pad)], dtype=torch.long)


def _prob_blob(T=8, c=(4, 4, 4)):
    p = np.zeros((T, T, T), dtype=np.float32)
    p[c[0], c[1], c[2]] = 1.0
    return p


def test_identity_matches_predicted_native_center():
    T = 8
    prob = _prob_blob(T, c=(5, 3, 6))
    geom = _geom()
    cen = _grid_centroid(prob)                       # np array (d,h,w)
    got = invert_geo_center(cen, None, torch.zeros(3, dtype=torch.bool), geom, T)
    want = _predicted_native_center(torch.from_numpy(prob), geom)
    assert got == want


def test_empty_centroid_returns_none():
    assert invert_geo_center(None, None, torch.zeros(3, dtype=torch.bool), _geom(), 8) is None


def test_flip_mirrors_the_centroid():
    T = 8
    geom = _geom(starts=(0, 0, 0), crop=(8, 8, 8), out=(8, 8, 8), pad=(0, 0, 0))
    cen = np.array([2.0, 1.0, 7.0])                  # d,h,w
    flips = torch.tensor([True, False, True])        # flip d and w
    got = invert_geo_center(cen, None, flips, geom, T)
    # unflip: d -> (T-1)-2 = 5, w -> (T-1)-7 = 0 ; native == pre-aug grid here (identity geom)
    assert got == (5, 1, 0)


def test_grid_shift_maps_through():
    # A constant grid that maps every output voxel to the volume centre in normalized coords
    # (0,0,0) -> pre-aug voxel ((T-1)/2). Identity geom -> native == (T-1)/2 per axis.
    T = 8
    grid_row = torch.zeros(T, T, T, 3)               # all (x,y,z) = 0 -> centre
    geom = _geom(starts=(0, 0, 0), crop=(8, 8, 8), out=(8, 8, 8), pad=(0, 0, 0))
    got = invert_geo_center(np.array([1.0, 2.0, 3.0]), grid_row,
                            torch.zeros(3, dtype=torch.bool), geom, T)
    mid = round((T - 1) / 2)
    assert got == (mid, mid, mid)


def test_directional_grid_maps_x_channel_to_w():
    """Verify that grid's x-channel (normalized) maps to the w output axis, not d.

    Build a grid with pure translation along W only in normalized coords.
    grid[..., 0] = 0.5 means x-channel = +0.5, which denormalizes to w-component.
    With identity geometry and centroid at (3.0, 3.0, 3.0):
    - w should denormalize to approximately 0.5 → (0.5+1)/2 * (T-1) ≈ 6.125
    - d and h should denormalize to 0 → (0+1)/2 * (T-1) ≈ 3.5
    This proves x maps to w (not d).
    """
    T = 8
    grid_row = torch.zeros(T, T, T, 3)
    grid_row[..., 0] = 0.5  # x-channel = +0.5, y and z = 0
    geom = _geom(starts=(0, 0, 0), crop=(8, 8, 8), out=(8, 8, 8), pad=(0, 0, 0))
    centroid = np.array([3.0, 3.0, 3.0])

    d, h, w = invert_geo_center(centroid, grid_row, torch.zeros(3, dtype=torch.bool), geom, T)

    # Denormalization: (x + 1) / 2 * (T - 1) where x is the grid value
    # For w (x-channel): (0.5 + 1) / 2 * 7 = 0.75 * 7 = 5.25 → rounds to 5
    # For d (z-channel): (0 + 1) / 2 * 7 = 0.5 * 7 = 3.5 → rounds to 4
    # For h (y-channel): (0 + 1) / 2 * 7 = 0.5 * 7 = 3.5 → rounds to 4
    expected_w = round((0.5 + 1.0) / 2.0 * (T - 1))
    expected_d = round((0.0 + 1.0) / 2.0 * (T - 1))
    expected_h = round((0.0 + 1.0) / 2.0 * (T - 1))

    assert w == expected_w, f"w={w}, expected {expected_w}"
    assert d == expected_d, f"d={d}, expected {expected_d}"
    assert h == expected_h, f"h={h}, expected {expected_h}"
    # Key assertion: w should differ from d and h in the expected direction
    assert w > d, f"w ({w}) should be greater than d ({d}) due to +0.5 grid offset"
    assert w > h, f"w ({w}) should be greater than h ({h}) due to +0.5 grid offset"
