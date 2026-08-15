"""Unit tests for the synthetic-ellipsoid target-label generator used by
data.self_context.synth_masks (see src/totalseg_dataloader_incontext.make_ellipsoid_label)."""

import random
import numpy as np
import torch

from src.totalseg_dataloader_incontext import make_ellipsoid_label


def _body_image(D=48, thresh_val=1.0):
    """Normalised-CT-like volume: an air shell (very low) around a body cube (above 0)."""
    img = torch.full((1, D, D, D), -1.6)          # air (below any body threshold)
    img[0, 12:36, 12:36, 12:36] = thresh_val      # body block
    return img


def test_shape_dtype_nonempty():
    img = _body_image()
    label, ctr, radii_mm = make_ellipsoid_label(
        img, spacing=[1.0, 1.0, 1.0], rng=random.Random(0), min_mm=3, max_mm=8)
    assert label.shape == img.shape[1:]
    assert label.dtype == torch.uint8
    assert label.sum() > 0                          # never empty
    assert label[ctr] == 1                          # centroid is set
    assert radii_mm.shape == (3,)                   # generative size returned
    assert ((radii_mm >= 3) & (radii_mm <= 8)).all()


def test_centroid_inside_body():
    img = _body_image()
    body = img[0] > -0.46
    for seed in range(20):
        _, ctr, _ = make_ellipsoid_label(
            img, spacing=[1.0, 1.0, 1.0], rng=random.Random(seed), min_mm=2, max_mm=6)
        assert bool(body[ctr]), f"centroid {ctr} landed in air (seed {seed})"


def test_radii_scale_with_spacing():
    """A fixed mm radius should cover FEWER voxels at coarser (larger mm/voxel) spacing."""
    img = _body_image(D=64)
    # Big body so the ellipsoid fits; fixed radius via min==max.
    img[0] = 1.0
    fine = make_ellipsoid_label(
        img, spacing=[1.0, 1.0, 1.0], rng=random.Random(3), min_mm=20, max_mm=20)[0].sum()
    coarse = make_ellipsoid_label(
        img, spacing=[4.0, 4.0, 4.0], rng=random.Random(3), min_mm=20, max_mm=20)[0].sum()
    assert coarse < fine


def test_deterministic_with_seed():
    img = _body_image()
    a = make_ellipsoid_label(img, spacing=[1.0, 1.0, 1.0], rng=random.Random(7))[0]
    b = make_ellipsoid_label(img, spacing=[1.0, 1.0, 1.0], rng=random.Random(7))[0]
    assert torch.equal(a, b)


def test_tiny_radius_not_empty():
    """Sub-voxel radius (1mm at 4mm spacing) still yields exactly the centroid voxel."""
    img = _body_image()
    label, ctr, _ = make_ellipsoid_label(
        img, spacing=[4.0, 4.0, 4.0], rng=random.Random(1), min_mm=1, max_mm=1)
    assert label.sum() >= 1
    assert label[ctr] == 1
