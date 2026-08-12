"""Nifti in-context cascade inference — predict a target organ mask from context
(image, binary-mask) nifti pairs via the 4mm->1.5mm cascade, GT-free for the target.

See docs/superpowers/specs/2026-08-12-nifti-incontext-cascade-inference-design.md.
"""
import random
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
from eval import _build_model, _warn_uninherited_data
from evaluate import _write_native, _predicted_native_center, dice_binary
from common import DEVICE


def load_nifti(path):
    """Load a nifti -> (array, affine), reoriented to closest-canonical (RAS) to match
    the training preprocessing (scripts/convert_to_npy.py uses nib.as_closest_canonical)."""
    img = nib.as_closest_canonical(nib.load(str(path)))
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
    crop_ct, _, out_sizes, pad_lo, geom = organ_crop_arrays(
        ct, ct, center, sp, image_size=(T, T, T), crop_mm=crop_mm,
        jitter=0, rng=random.Random(0))
    return place_image(crop_ct, out_sizes, pad_lo, T), geom


def prep_context(ct, mask, sp, center, *, T, crop_mm, mask_downsample, occ_thr):
    """Native (CT, binary mask) + centre -> (img_t (1,T,T,T), mask_t (T,T,T) long)."""
    assert ct.shape == mask.shape, f"context ct {ct.shape} != mask {mask.shape}"
    crop_ct, crop_mask, out_sizes, pad_lo, _ = organ_crop_arrays(
        ct, mask, center, sp, image_size=(T, T, T), crop_mm=crop_mm,
        jitter=0, rng=random.Random(0))
    img_t = place_image(crop_ct, out_sizes, pad_lo, T)
    mask_small = resample_binary(np.asarray(crop_mask) > 0, tuple(out_sizes),
                                 mode=mask_downsample, occ_thr=occ_thr)
    return img_t, place_label(mask_small, out_sizes, pad_lo, T)


def _resample_gt(gt, shape):
    """Nearest-resample a binary GT to `shape` if it differs (bool)."""
    if gt.shape == shape:
        return gt.astype(bool)
    t = torch.from_numpy(gt.astype(np.float32))[None, None]
    return (torch.nn.functional.interpolate(t, size=shape, mode="nearest")[0, 0] > 0.5).numpy()


def _to_original_orientation(pred_canon, target_path):
    """Reorient a canonical-space (RAS) prediction back to target_path's on-disk orientation.

    load_nifti feeds the model RAS-canonical arrays (to match training), so predictions come
    out on the RAS grid. This inverts that reorientation using the target's original affine,
    returning (array, affine) whose voxel grid + affine match the input target file exactly —
    so the saved mask overlays the original image by voxel index, not just in world space.
    A target that is already RAS makes the transform the identity (no-op)."""
    orig = nib.load(str(target_path))
    ras_ornt = nib.orientations.axcodes2ornt(("R", "A", "S"))
    transform = nib.orientations.ornt_transform(ras_ornt, nib.io_orientation(orig.affine))
    return nib.orientations.apply_orientation(pred_canon, transform), orig.affine


def predict_nifti(cfg, target_path, context_pairs, gt_path=None, out_path=None):
    """Run the coarse->fine in-context cascade on nifti files (GT-free target).

    cfg            : OmegaConf cfg (same surface as experiments/3d/eval.py). Uses
                     data.image_size / mask_downsample / mask_occupancy_thr and
                     eval.model / eval.checkpoint / eval.spacing_sweep.
    target_path    : target CT .nii.gz.
    context_pairs  : list[(image_path, binary_mask_path)] for the same organ (K = len).
    gt_path        : optional target GT (binary) .nii.gz -> Dice + coarse-only Dice.
    out_path       : optional -> write the predicted mask as .nii.gz on the target grid.
                     The model runs in RAS-canonical space (load_nifti canonicalises to match
                     training), but the returned/saved mask is reoriented back to the target
                     file's original orientation + affine, so it overlays the input by voxel
                     index (see _to_original_orientation).

    Returns {"pred", "affine", "dice", "coarse_only_dice", "pred_path"}.
    """
    if not context_pairs:
        raise ValueError("predict_nifti needs at least one context pair (in-context model)")

    _warn_uninherited_data(cfg)
    model = _build_model(cfg)
    T = int(cfg.data.image_size[0])
    crop_ds = cfg.data.get("mask_downsample", "occupancy")
    crop_thr = float(cfg.data.get("mask_occupancy_thr", 0.1))
    spacings = [float(s) for s in cfg.eval.spacing_sweep]

    # --- load target + contexts once (arrays reused across passes) --------------
    tgt_ct, affine = load_nifti(target_path)
    tgt_ct = normalize_ct(tgt_ct)
    tgt_sp = voxel_spacing(affine)
    shape = tgt_ct.shape

    contexts = []  # (ct_norm, mask_bool, spacing, centroid)
    for img_p, msk_p in context_pairs:
        c_ct, c_aff = load_nifti(img_p)
        c_msk, _ = load_nifti(msk_p)
        c_msk = np.asarray(c_msk) > 0
        contexts.append((normalize_ct(c_ct), c_msk, voxel_spacing(c_aff),
                         mask_centroid(c_msk)))

    native = np.zeros(shape, dtype=bool)   # stitched (coarse then fine overwrite)
    coarse_native = None
    center = tuple(s // 2 for s in shape)  # coarse: volume centre
    prev_pred = prev_geom = None

    for i, s in enumerate(spacings):
        if i > 0:
            # Hard-predict centroid (not soft prob like eval's cascade) — one forward per
            # pass, model-agnostic; intentional divergence documented in the design spec.
            c = _predicted_native_center(
                torch.from_numpy(prev_pred.astype(np.float32)),
                torch.from_numpy(prev_geom.astype(np.int64)))
            center = tuple(s2 // 2 for s2 in shape) if c == "volume_center" else c

        tgt_img, geom = prep_target(tgt_ct, tgt_sp, center, T=T, crop_mm=s)
        ctx_in, ctx_out = [], []
        for c_ct, c_msk, c_sp, c_center in contexts:
            im, mk = prep_context(c_ct, c_msk, c_sp, c_center, T=T, crop_mm=s,
                                  mask_downsample=crop_ds, occ_thr=crop_thr)
            ctx_in.append(im)
            ctx_out.append(mk)
        target_b = tgt_img.unsqueeze(0).to(DEVICE)                      # (1,1,T,T,T)
        ctx_in_b = torch.stack(ctx_in).unsqueeze(0).to(DEVICE)         # (1,K,1,T,T,T)
        ctx_out_b = torch.stack(ctx_out).unsqueeze(0).to(DEVICE)       # (1,K,T,T,T)

        kw = {"spacing": s} if getattr(model, "spacing_aware", False) else {}
        with torch.no_grad():
            pred = model.predict(target_b, ctx_in_b, ctx_out_b, **kw)   # (1,T,T,T)
        pred = pred.squeeze(0).cpu().numpy()

        geom_np = geom.numpy()
        _write_native(native, pred, geom_np)
        if i == 0:
            coarse_native = native.copy()
        prev_pred, prev_geom = pred, geom_np

    # --- metrics (canonical space; both pred and GT are RAS so Dice is orientation-safe) ---
    dice = coarse_only = None
    if gt_path is not None:
        gt, _ = load_nifti(gt_path)
        gt = _resample_gt(np.asarray(gt) > 0, shape)
        gt_t = torch.from_numpy(gt)
        dice = float(dice_binary(torch.from_numpy(native), gt_t))
        coarse_only = float(dice_binary(torch.from_numpy(coarse_native), gt_t))

    # --- output on the ORIGINAL target grid -------------------------------------
    # Reorient the RAS prediction back to the target file's stored orientation + affine so
    # the mask shares the input image's voxel grid (not a permuted RAS grid).
    pred_native, out_affine = _to_original_orientation(native, target_path)
    pred_path = None
    if out_path is not None:
        nib.save(nib.Nifti1Image(pred_native.astype(np.uint8), out_affine), str(out_path))
        pred_path = Path(out_path)

    return {"pred": pred_native, "affine": out_affine, "dice": dice,
            "coarse_only_dice": coarse_only, "pred_path": pred_path}
