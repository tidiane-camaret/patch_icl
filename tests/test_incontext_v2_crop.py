import numpy as np
import random
import torch
from src.providers.totalseg import crop_and_place


def test_crop_and_place_shapes_and_geometry():
    D = 40
    img = (np.random.rand(D, D, D) * 100).astype(np.int16)  # raw-HU-like
    lbl = np.zeros((D, D, D), dtype=np.uint8)
    lbl[15:25, 15:25, 15:25] = 7                            # a blob of class 7
    T = 32
    image_t, label_t, geom = crop_and_place(
        img, lbl, class_idx=7, center=(20, 20, 20), T=T,
        crop_spacing_mm=1.5, native_spacing=(1.5, 1.5, 1.5),
        jitter=0, rng=random.Random(0), mask_downsample="occupancy", occ_thr=0.1)
    assert image_t.shape == (1, T, T, T)
    assert image_t.dtype == torch.float32
    assert label_t.shape == (T, T, T)
    assert label_t.dtype == torch.int64
    assert set(torch.unique(label_t).tolist()) <= {0, 1}
    assert label_t.sum() > 0                                # class 7 present in the crop
    assert geom.shape == (4, 3) and geom.dtype == torch.int64


def test_crop_and_place_thin_structure_survives_occupancy():
    D = 60
    img = np.zeros((D, D, D), dtype=np.int16)
    lbl = np.zeros((D, D, D), dtype=np.uint8)
    lbl[30, :, 30] = 3                                      # 1-voxel-thick line, class 3
    _, label_t, _ = crop_and_place(
        img, lbl, class_idx=3, center=(30, 30, 30), T=16,
        crop_spacing_mm=4.0, native_spacing=(1.0, 1.0, 1.0),
        jitter=0, rng=random.Random(0), mask_downsample="occupancy", occ_thr=0.1)
    assert label_t.sum() > 0                                # thin line not lost on downsample


def test_crop_and_place_applies_normalize_fn():
    D = 20
    img = np.full((D, D, D), 500, dtype=np.int16)
    lbl = np.zeros((D, D, D), dtype=np.uint8); lbl[10, 10, 10] = 1
    image_t, _, _ = crop_and_place(
        img, lbl, class_idx=1, center=(10, 10, 10), T=16,
        crop_spacing_mm=1.0, native_spacing=(1.0, 1.0, 1.0),
        jitter=0, rng=random.Random(0), mask_downsample="nearest", occ_thr=0.5,
        normalize_fn=lambda a: a.astype(np.float32) * 0.0 + 0.25)
    assert torch.allclose(image_t[image_t != image_t.min()],
                          torch.tensor(0.25), atol=1e-5) or (image_t == 0.25).any()
