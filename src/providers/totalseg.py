"""TotalSegmentator volume provider for the in-context dataloader v2.

Single raw_ct organ-crop load path. `crop_and_place` is the one place crop
geometry (physical extent -> crop sizes -> resample -> centre-pad) is computed,
reusing the pure helpers extracted in the v1 module.
"""
import numpy as np
import torch

from src.totalseg_dataloader_incontext import (
    organ_crop_arrays, place_image, place_label, resample_binary,
)


def crop_and_place(image_np, label_np, class_idx, center, T, *,
                   crop_spacing_mm, native_spacing, jitter, rng,
                   mask_downsample, occ_thr, normalize_fn=None):
    """Organ-centred crop of physical extent T*crop_spacing_mm around `center`,
    resampled to T^3 and centre-padded. Returns (image (1,T,T,T) f32, label
    (T,T,T) i64 binary for class_idx, crop_geom (4,3) i64).

    `normalize_fn`, when given, maps the cropped raw image slice to model input
    space BEFORE placement (so the air-pad value matches the normalized min)."""
    crop_ct, crop_lbl, out_sizes, pad_lo, geom = organ_crop_arrays(
        image_np, label_np, center, list(native_spacing),
        image_size=(T, T, T), crop_mm=crop_spacing_mm, jitter=jitter, rng=rng)
    crop_ct = np.ascontiguousarray(crop_ct)
    if normalize_fn is not None:
        crop_ct = normalize_fn(crop_ct)
    image_t = place_image(crop_ct, out_sizes, pad_lo, T)
    lbl_small = resample_binary(crop_lbl == class_idx, tuple(out_sizes),
                                mode=mask_downsample, occ_thr=occ_thr)
    label_t = place_label(lbl_small, out_sizes, pad_lo, T).long()
    return image_t, label_t, geom
