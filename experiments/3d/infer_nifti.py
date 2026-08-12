"""Nifti in-context cascade inference — predict a target organ mask from context
(image, binary-mask) nifti pairs via the 4mm->1.5mm cascade, GT-free for the target.

See docs/superpowers/specs/2026-08-12-nifti-incontext-cascade-inference-design.md.
"""
import sys
import warnings
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from nibabel.affines import voxel_sizes

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling eval/evaluate/common

from src.totalseg_dataset import normalize_ct
from src.totalseg_dataloader_incontext import (
    organ_crop_arrays, place_image, place_label, resample_binary,
)


def load_nifti(path):
    """Load a nifti -> (array, affine). Array is the stored data with scaling applied."""
    img = nib.load(str(path))
    return np.asanyarray(img.dataobj), img.affine


def voxel_spacing(affine):
    """Per-axis mm/voxel (3,) from the affine, aligned with the array axes."""
    return [float(v) for v in voxel_sizes(affine)]


def mask_centroid(mask):
    """Integer centre-of-mass (d,h,w) of a binary mask; volume centre + warn if empty."""
    fg = np.asarray(mask) > 0
    if not fg.any():
        warnings.warn("infer_nifti: empty context/target mask — using volume centre.",
                      stacklevel=2)
        return tuple(s // 2 for s in fg.shape)
    idx = np.nonzero(fg)
    return tuple(int(a.mean()) for a in idx)


def prep_target(ct, sp, center, *, T, crop_mm):
    """Native CT (normalised) + centre -> (img_t (1,T,T,T), crop_geom (4,3)).

    No target label, so ct doubles as the label array for organ_crop_arrays (its
    crop_lbl output is discarded). rng is unused at jitter=0 (centred crop)."""
    import random
    crop_ct, _, out_sizes, pad_lo, geom = organ_crop_arrays(
        ct, ct, center, sp, image_size=(T, T, T), crop_mm=crop_mm,
        jitter=0, rng=random.Random(0))
    return place_image(crop_ct, out_sizes, pad_lo, T), geom


def prep_context(ct, mask, sp, center, *, T, crop_mm, mask_downsample, occ_thr):
    """Native (CT, binary mask) + centre -> (img_t (1,T,T,T), mask_t (T,T,T) long)."""
    import random
    assert ct.shape == mask.shape, f"context ct {ct.shape} != mask {mask.shape}"
    crop_ct, crop_mask, out_sizes, pad_lo, _ = organ_crop_arrays(
        ct, mask, center, sp, image_size=(T, T, T), crop_mm=crop_mm,
        jitter=0, rng=random.Random(0))
    img_t = place_image(crop_ct, out_sizes, pad_lo, T)
    mask_small = resample_binary(np.asarray(crop_mask) > 0, tuple(out_sizes),
                                 mode=mask_downsample, occ_thr=occ_thr)
    return img_t, place_label(mask_small, out_sizes, pad_lo, T)
