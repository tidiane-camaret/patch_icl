"""N-level coarse->fine cascade for PatchSet3D (v2 pipeline).

run_cascade executes one N-level forward (level 0 = GT-centred, level i>0 = target
re-cropped on level i-1's predicted centre-of-mass); shared by the train loop
(experiments/3d/train.py train_epoch) and the v2 cascade val pass (evaluate_cascade).
PatchSet3D.forward stays single-level.

See docs/superpowers/specs/2026-08-30-cascade-training-patchset3d-design.md.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from src.gpu_augment import GeoState
from src.incontext_dataset_v2 import LoadRequest
from src.totalseg_dataloader_incontext import incontext_collate_fn
from grid_metrics import target_like
from evaluate import _grid_centroid, _predicted_native_center


def invert_geo_center(centroid_dhw, grid_row, flips_row, crop_geom_row, T):
    """Map a centroid in a level's AUGMENTED T^3 grid back to a native crop centre.

    centroid_dhw : length-3 (d,h,w) in the augmented grid, or None (empty prob) -> None.
    grid_row     : (T,T,T,3) float sampling grid (grid_sample xyz convention) or None.
    flips_row    : (3,) bool, per-axis flip (D,H,W order) applied before the warp.
    crop_geom_row: (4,3) long [starts, crop_sizes, out_sizes, pad_lo].
    Returns native voxel (d,h,w), each >= 0. Identity (grid_row None, no flips) reproduces
    evaluate._predicted_native_center for the same centroid.
    """
    if centroid_dhw is None:
        return None
    g = [float(centroid_dhw[a]) for a in range(3)]                     # d,h,w (augmented)

    flips = [bool(x) for x in (flips_row.tolist() if torch.is_tensor(flips_row) else flips_row)]
    for a in range(3):
        if flips[a]:
            g[a] = (T - 1) - g[a]

    if grid_row is not None:
        # Interpolate the (T,T,T,3) grid at the fractional post-unflip coord. Query point in
        # grid_sample xyz order = (w, h, d) normalized with the align_corners=True pairing.
        q = torch.tensor(
            [[[[[2.0 * g[2] / max(1, T - 1) - 1.0,
                 2.0 * g[1] / max(1, T - 1) - 1.0,
                 2.0 * g[0] / max(1, T - 1) - 1.0]]]]],
            dtype=torch.float32)                                       # (1,1,1,1,3)
        field = grid_row.detach().float().permute(3, 0, 1, 2).unsqueeze(0)  # (1,3,T,T,T)
        pre = F.grid_sample(field, q, mode="bilinear", padding_mode="border",
                            align_corners=True)[0, :, 0, 0, 0]         # (3,) = (x,y,z) norm
        x, y, z = (float(v) for v in pre)
        g = [(z + 1.0) / 2.0 * (T - 1),                               # d
             (y + 1.0) / 2.0 * (T - 1),                               # h
             (x + 1.0) / 2.0 * (T - 1)]                               # w

    starts, crop_sizes, out_sizes, pad_lo = (crop_geom_row[r].tolist() for r in range(4))
    native = [int(round(starts[a] + (g[a] - pad_lo[a]) / max(1, out_sizes[a]) * crop_sizes[a]))
              for a in range(3)]
    return tuple(max(0, c) for c in native)
