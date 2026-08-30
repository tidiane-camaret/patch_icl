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

    align_corners=False denormalization: voxel = ((norm + 1) * T - 1) / 2.
    With grid[..., 0] = 0.5 (x-channel), centroid (3,3,3), identity geom, T=8:
    - w (x-channel 0.5): ((0.5+1)*8-1)/2 = 5.5 → rounds to 6
    - d,h (z,y-channels 0): ((0+1)*8-1)/2 = 3.5 → round to 4
    Assertion: w > d and w > h (x pulls w toward +x, not d).
    """
    T = 8
    grid_row = torch.zeros(T, T, T, 3)
    grid_row[..., 0] = 0.5  # x-channel = +0.5, y and z = 0
    geom = _geom(starts=(0, 0, 0), crop=(8, 8, 8), out=(8, 8, 8), pad=(0, 0, 0))
    centroid = np.array([3.0, 3.0, 3.0])

    d, h, w = invert_geo_center(centroid, grid_row, torch.zeros(3, dtype=torch.bool), geom, T)

    # With align_corners=False: ((norm + 1) * T - 1) / 2
    expected_w = round(((0.5 + 1.0) * T - 1.0) / 2.0)
    expected_d = round(((0.0 + 1.0) * T - 1.0) / 2.0)
    expected_h = round(((0.0 + 1.0) * T - 1.0) / 2.0)

    assert w == expected_w, f"w={w}, expected {expected_w}"
    assert d == expected_d, f"d={d}, expected {expected_d}"
    assert h == expected_h, f"h={h}, expected {expected_h}"
    # Key assertion: w should differ from d and h in the expected direction
    assert w > d, f"w ({w}) should be greater than d ({d}) due to +0.5 grid offset"
    assert w > h, f"w ({w}) should be greater than h ({h}) due to +0.5 grid offset"


def test_flip_then_grid_inversion_order():
    """Discriminates correct order (grid-lookup-then-unflip) from wrong (unflip-then-lookup).

    grid[..., 2] = 0.5 displaces along d (z-channel). With flip on d:
    Correct: interp grid at g_aug (2,3,3) -> d in flipped vol = ((0.5+1)*8-1)/2 = 5.5 ->
    unflip d -> (8-1)-5.5 = 1.5 -> round 2 -> native (2,4,4).
    Wrong order would give round(5.5)=6 (no 2nd unflip).
    """
    T = 8
    grid_row = torch.zeros(T, T, T, 3)
    grid_row[..., 2] = 0.5                                   # z-channel -> d output axis
    geom = _geom(starts=(0, 0, 0), crop=(8, 8, 8), out=(8, 8, 8), pad=(0, 0, 0))
    flips = torch.tensor([True, False, False])               # flip d only
    got = invert_geo_center(np.array([2.0, 3.0, 3.0]), grid_row, flips, geom, T)
    assert got == (2, 4, 4)
