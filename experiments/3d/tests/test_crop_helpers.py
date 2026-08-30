"""Unit tests for the extracted pure crop/resample helpers (Task 1)."""
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.totalseg_dataloader_incontext import (  # noqa: E402
    organ_crop_arrays, place_image, place_label, resample_binary,
)


def test_organ_crop_centered_no_jitter():
    # 20^3 volume, isotropic 1mm, T=8, crop_mm=1 -> target extent 8 voxels, centered crop.
    ct = np.arange(20 ** 3, dtype=np.float32).reshape(20, 20, 20)
    lbl = np.zeros((20, 20, 20), dtype=np.uint8)
    lbl[9:11, 9:11, 9:11] = 1
    rng = random.Random(0)
    crop_ct, crop_lbl, out_sizes, pad_lo, geom = organ_crop_arrays(
        ct, lbl, center=(10, 10, 10), sp=[1.0, 1.0, 1.0],
        image_size=(8, 8, 8), crop_mm=1.0, jitter=0, rng=rng)
    assert crop_ct.shape == (8, 8, 8)          # extent 8, fits in 20
    assert out_sizes == [8, 8, 8]              # full T, no padding
    assert pad_lo == [0, 0, 0]
    # starts = center - cs//2 = 10 - 4 = 6
    assert geom[0].tolist() == [6, 6, 6]
    assert geom.shape == (4, 3)


def test_organ_crop_thin_axis_padded():
    # A thin axis (size 4 < extent 8) is captured whole and maps to <T with centre pad.
    ct = np.zeros((4, 20, 20), dtype=np.float32)
    lbl = np.zeros((4, 20, 20), dtype=np.uint8)
    rng = random.Random(0)
    _, _, out_sizes, pad_lo, geom = organ_crop_arrays(
        ct, lbl, center=(2, 10, 10), sp=[1.0, 1.0, 1.0],
        image_size=(8, 8, 8), crop_mm=1.0, jitter=0, rng=rng)
    assert geom[1].tolist()[0] == 4            # crop_sizes[0] clamped to native 4
    assert out_sizes[0] == 4 and pad_lo[0] == 2  # 4 maps to 4/8, centred -> pad 2


def test_place_image_pads_with_air():
    crop = np.full((4, 8, 8), -3.0, dtype=np.float32)
    img = place_image(crop, out_sizes=[4, 8, 8], pad_lo=[2, 0, 0], T=8)
    assert img.shape == (1, 8, 8, 8)
    assert float(img[0, 0, 0, 0]) == -3.0      # padded region filled with crop.min()


def test_resample_binary_occupancy_keeps_thin():
    m = np.zeros((8, 8, 8), dtype=bool)
    m[0, 0, 0] = True                          # single voxel
    out = resample_binary(m, (2, 2, 2), mode="occupancy", occ_thr=0.1)
    assert out.sum() >= 1                       # densest voxel kept (non-empty guarantee)
    out_near = resample_binary(m, (2, 2, 2), mode="nearest", occ_thr=0.5)
    assert out_near.shape == (2, 2, 2)


def test_resample_binary_soft_returns_fraction():
    # Half of the 8^3 volume filled -> each 4^3-footprint output cell is 50% covered.
    m = np.zeros((8, 8, 8), dtype=bool)
    m[:4] = True
    out = resample_binary(m, (2, 2, 2), mode="soft", occ_thr=0.5)
    assert out.dtype == torch.float32
    assert out.min() >= 0.0 and out.max() <= 1.0
    assert torch.allclose(out[0], torch.full((2, 2), 1.0))   # fully-inside slab
    assert torch.allclose(out[1], torch.full((2, 2), 0.0))   # fully-outside slab


def test_resample_binary_soft_partial_cell():
    # A 2-voxel-thick sheet in a 4^3 footprint -> fraction 2/4 = 0.5.
    m = np.zeros((8, 8, 8), dtype=bool)
    m[0:2, 0:4, 0:4] = True
    out = resample_binary(m, (2, 2, 2), mode="soft", occ_thr=0.5)
    assert abs(float(out[0, 0, 0]) - 0.5) < 1e-6


def test_resample_binary_soft_nonempty_guard():
    # A single native voxel -> true fraction 1/64, below occ_thr; peak cell floored so
    # the structure never vanishes.
    m = np.zeros((8, 8, 8), dtype=bool)
    m[0, 0, 0] = True
    out = resample_binary(m, (2, 2, 2), mode="soft", occ_thr=0.5)
    assert out.dtype == torch.float32
    assert float(out.max()) >= 0.5
    assert float((out > 0).sum()) == 1          # only the peak cell is lifted


def test_place_label_preserves_float_dtype():
    small = torch.full((4, 8, 8), 0.5, dtype=torch.float32)
    lab = place_label(small, out_sizes=[4, 8, 8], pad_lo=[2, 0, 0], T=8)
    assert lab.dtype == torch.float32
    assert lab[0].sum() == 0 and abs(float(lab[2].sum()) - 32.0) < 1e-4


def test_place_label_centers():
    small = torch.ones(4, 8, 8, dtype=torch.long)
    lab = place_label(small, out_sizes=[4, 8, 8], pad_lo=[2, 0, 0], T=8)
    assert lab.shape == (8, 8, 8)
    assert lab[0].sum() == 0 and lab[2].sum() == 64   # padded slices 0, body slices set
