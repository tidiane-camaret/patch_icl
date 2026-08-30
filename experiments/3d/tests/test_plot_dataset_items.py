"""_overlay / _best_slice must not render a soft (float) occupancy mask with interior
holes: a bilinear-warped binary mask has interior voxels at 0.9999x (float32), which an
exact `== label_id` test drops. See docs/logs.md 2026-08-30."""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "3d"))

from plot_dataset_items import _overlay, _best_slice  # noqa: E402


def _soft_warped_square(n=32):
    """A solid square whose interior, after a sub-voxel shift + bilinear resample, sits at
    ~0.9999 not exactly 1.0 — the shape the GPU mask warp produces for a soft mask."""
    m = torch.zeros(1, 1, n, n)
    m[0, 0, 8:24, 8:24] = 1.0
    grid = torch.nn.functional.affine_grid(
        torch.tensor([[[1.0, 0.0, 0.013], [0.0, 1.0, -0.021]]]), (1, 1, n, n),
        align_corners=False)
    w = torch.nn.functional.grid_sample(m, grid, mode="bilinear",
                                        padding_mode="zeros", align_corners=False)
    return w[0, 0].numpy()


def test_overlay_fills_soft_mask_interior():
    soft = _soft_warped_square()
    assert (soft == 1).sum() < (soft >= 0.5).sum()          # interior really is < 1.0
    img = np.zeros_like(soft)
    rgb = _overlay(img, soft, {1: [1.0, 0.2, 0.2]}, alpha=0.5)
    coloured = (rgb[..., 0] > rgb[..., 2] + 0.1)            # red channel lifted
    # every >=0.5 voxel is overlaid (no interior holes), and no <0.5 voxel is
    assert coloured.sum() == int((soft >= 0.5).sum())


def test_overlay_integer_multilabel_unchanged():
    mask = np.zeros((16, 16), dtype=np.int64)
    mask[2:8, 2:8] = 1
    mask[9:14, 9:14] = 2
    rgb = _overlay(np.zeros_like(mask, dtype=float), mask,
                   {1: [1.0, 0.0, 0.0], 2: [0.0, 1.0, 0.0]})
    assert (rgb[2:8, 2:8, 0] > 0.4).all() and (rgb[9:14, 9:14, 1] > 0.4).all()


def test_best_slice_picks_soft_foreground():
    m = torch.zeros(5, 16, 16)
    m[3] = torch.from_numpy(_soft_warped_square(16))        # slice 3 carries the object
    _, msl = _best_slice(torch.zeros(1, 5, 16, 16), m)
    assert (msl >= 0.5).sum() > 0                           # picked the slice with the object
