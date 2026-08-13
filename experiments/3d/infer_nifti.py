"""Nifti in-context cascade inference — predict a target organ mask from context
(image, binary-mask) nifti pairs via the 4mm->1.5mm cascade, GT-free for the target.

See docs/superpowers/specs/2026-08-12-nifti-incontext-cascade-inference-design.md.
"""
import random
import re
import sys
import warnings
import xml.etree.ElementTree as ET
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


def _iter_chunks(seq, n):
    """Yield successive length-<=n slices of seq."""
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def _native_from_passes(shape, passes):
    """Stitch a per-label bool native volume from its cascade passes.

    `passes` is the ordered [(pred_grid, geom), ...] for one label; finer (later) passes
    overwrite coarser ones — the same semantics as the original single-pass stitch. Kept as
    small T³ grids until here so we never hold one full-size volume per label at once."""
    nat = np.zeros(shape, dtype=bool)
    for pred, geom in passes:
        _write_native(nat, pred, geom)
    return nat


def _read_caret_label_table(path):
    """Raw Caret <LabelTable> extension bytes embedded in a nifti (code-0 header extension),
    or None if the file has none. TotalSegmentator masks store organ names + colors here."""
    try:
        exts = nib.load(str(path)).header.extensions
    except Exception:
        return None
    for e in exts:
        c = e.get_content()
        if isinstance(c, (bytes, bytearray)) and b"<CaretExtension" in c:
            return bytes(c).rstrip(b"\x00")           # strip nifti 16-byte null padding
    return None


def _caret_label_names(content):
    """{id: name} from Caret LabelTable extension bytes; {} if unparseable."""
    names = {}
    try:
        root = ET.fromstring(content)
    except ET.ParseError:
        return names
    for lab in root.iter("Label"):
        try:
            names[int(lab.get("Key"))] = (lab.text or "").strip()
        except (TypeError, ValueError):
            pass
    return names


_LABEL_RE = re.compile(rb'\s*<Label\b[^>]*?\bKey="(\d+)"[^>]*?>.*?</Label>', re.DOTALL)


def _subset_caret_label_table(content, keep_ids):
    """Drop the <Label> entries whose Key isn't in keep_ids (background 0 always kept),
    leaving the rest of the original extension bytes byte-for-byte intact.

    Deliberately NOT an XML re-serialization: the Caret/atlas format carries structure a
    viewer needs to open the volume as an atlas (the `<?xml?>` declaration, CDATA wrapping,
    and the `<VolumeType>Label</VolumeType>` marker). ElementTree round-tripping strips those;
    surgical byte deletion preserves them so the prediction stays viewer-identical to the
    source atlas except for the removed labels. Returns None if no <Label> is found."""
    keep = {int(i) for i in keep_ids} | {0}
    found = False

    def repl(m):
        nonlocal found
        found = True
        return m.group(0) if int(m.group(1)) in keep else b""

    out = _LABEL_RE.sub(repl, content)
    return out if found else None


def _output_label_table(gt_path, context_pairs, keep_ids):
    """(Nifti1Extension, {id: name}) for the predicted multi-label mask, subset to keep_ids.

    Prefer the target GT's embedded LabelTable (its names/colors), else the first context
    mask that has one — so a viewer shows organ names on the prediction. (None, {}) if no
    source carries a table."""
    sources = ([gt_path] if gt_path is not None else []) + [m for _, m in context_pairs]
    keep = {int(i) for i in keep_ids}
    for src in sources:
        content = _read_caret_label_table(src)
        if content is None:
            continue
        sub = _subset_caret_label_table(content, keep_ids)
        if sub is None:
            continue
        names = {k: v for k, v in _caret_label_names(content).items() if k in keep}
        return nib.nifti1.Nifti1Extension(0, sub), names
    return None, {}


def predict_nifti(cfg, target_path, context_pairs, label_ids=None, batch_size=8,
                  gt_path=None, out_path=None):
    """Run the coarse->fine in-context cascade on nifti files (GT-free target).

    cfg            : OmegaConf cfg (same surface as experiments/3d/eval.py). Uses
                     data.image_size / mask_downsample / mask_occupancy_thr and
                     eval.model / eval.checkpoint / eval.spacing_sweep.
    target_path    : target CT .nii.gz.
    context_pairs  : list[(image_path, mask_path)] for the same organ(s), K = len. Each
                     mask is binarized (>0) in single-label mode, or read as an id-valued
                     multi-label mask when label_ids is given.
    label_ids      : None -> single binary organ (pred is a bool mask; dice/coarse_only are
                     floats — the original behavior). A list of ints (or "all" for every
                     non-zero id present in the context masks) -> multi-label: each context
                     mask supplies label L's context as (mask == L), the cascade runs per
                     label batched `batch_size` labels per model forward, and the output is
                     one id-valued uint8 mask (smaller organs win overlaps). dice/coarse_only
                     become {label: dice} dicts and macro_dice is added.
    batch_size     : label-tasks per model forward in multi-label mode (default 8).
    gt_path        : optional target GT .nii.gz (binary in single mode, id-valued in
                     multi mode) -> Dice + coarse-only Dice.
    out_path       : optional -> write the predicted mask as .nii.gz on the target grid.
                     The model runs in RAS-canonical space (load_nifti canonicalises to match
                     training), but the returned/saved mask is reoriented back to the target
                     file's original orientation + affine, so it overlays the input by voxel
                     index (see _to_original_orientation).

    Returns {"pred", "affine", "dice", "coarse_only_dice", "pred_path"}; multi-label adds
    "labels" and "macro_dice" and makes dice/coarse_only per-label dicts.
    """
    if not context_pairs:
        raise ValueError("predict_nifti needs at least one context pair (in-context model)")

    _warn_uninherited_data(cfg)
    model = _build_model(cfg)
    T = int(cfg.data.image_size[0])
    crop_ds = cfg.data.get("mask_downsample", "occupancy")
    crop_thr = float(cfg.data.get("mask_occupancy_thr", 0.1))
    spacings = [float(s) for s in cfg.eval.spacing_sweep]

    # --- load target + contexts once (arrays reused across passes/labels) -------
    tgt_ct, affine = load_nifti(target_path)
    tgt_ct = normalize_ct(tgt_ct)
    tgt_sp = voxel_spacing(affine)
    shape = tgt_ct.shape

    contexts = []  # (ct_norm, id_mask, spacing) shared across labels
    for img_p, msk_p in context_pairs:
        c_ct, c_aff = load_nifti(img_p)
        c_msk, _ = load_nifti(msk_p)
        contexts.append((normalize_ct(c_ct), np.asarray(c_msk), voxel_spacing(c_aff)))

    multilabel = label_ids is not None
    if not multilabel:
        labels = [None]                                    # single binary organ
    elif isinstance(label_ids, str) and label_ids.lower() == "all":
        present = set()
        for _, c_ml, _ in contexts:
            present |= {int(v) for v in np.unique(c_ml) if v != 0}
        labels = sorted(present)
    else:
        labels = [int(l) for l in label_ids]
    if not labels:
        raise ValueError("no labels to segment (empty label_ids / no non-zero context ids)")

    # per-label context: binary mask (== id, or >0 for single) + its centroid, once.
    tasks = []
    for lab in labels:
        ctx = []
        for c_ct, c_ml, c_sp in contexts:
            b = (c_ml > 0) if lab is None else (c_ml == lab)
            ctx.append((c_ct, b, c_sp, mask_centroid(b)))
        tasks.append({"label": lab, "ctx": ctx, "passes": []})

    # --- coarse->fine cascade, batched over labels ------------------------------
    for i, s in enumerate(spacings):
        prepped = []                                       # (task, tgt_img, geom, ctx_in, ctx_out)
        for task in tasks:
            if i == 0:
                center = tuple(sz // 2 for sz in shape)    # coarse: volume centre
            else:
                # Hard-predict centroid (not soft prob like eval's cascade) — one forward per
                # pass, model-agnostic; intentional divergence documented in the design spec.
                prev_pred, prev_geom = task["passes"][-1]
                c = _predicted_native_center(
                    torch.from_numpy(prev_pred.astype(np.float32)),
                    torch.from_numpy(prev_geom.astype(np.int64)))
                center = tuple(sz // 2 for sz in shape) if c == "volume_center" else c

            tgt_img, geom = prep_target(tgt_ct, tgt_sp, center, T=T, crop_mm=s)
            ctx_in, ctx_out = [], []
            for c_ct, c_bin, c_sp, c_center in task["ctx"]:
                im, mk = prep_context(c_ct, c_bin, c_sp, c_center, T=T, crop_mm=s,
                                      mask_downsample=crop_ds, occ_thr=crop_thr)
                ctx_in.append(im)
                ctx_out.append(mk)
            prepped.append((task, tgt_img, geom, ctx_in, ctx_out))

        for chunk in _iter_chunks(prepped, batch_size):
            target_b = torch.stack([p[1] for p in chunk]).to(DEVICE)             # (B,1,T,T,T)
            ctx_in_b = torch.stack([torch.stack(p[3]) for p in chunk]).to(DEVICE)   # (B,K,1,T,T,T)
            ctx_out_b = torch.stack([torch.stack(p[4]) for p in chunk]).to(DEVICE)  # (B,K,T,T,T)
            kw = {"spacing": s} if getattr(model, "spacing_aware", False) else {}
            with torch.no_grad():
                preds = model.predict(target_b, ctx_in_b, ctx_out_b, **kw)        # (B,T,T,T)
            preds = preds.cpu().numpy()
            for j, (task, _, geom, _, _) in enumerate(chunk):
                task["passes"].append((preds[j], geom.numpy()))

    gt_ml = None
    if gt_path is not None:
        gt_arr, _ = load_nifti(gt_path)
        gt_ml = np.asarray(gt_arr)

    if not multilabel:
        # --- single binary organ: original return contract (bool pred, scalar dice) ---
        task = tasks[0]
        native = _native_from_passes(shape, task["passes"])
        dice = coarse_only = None
        if gt_ml is not None:
            gt = _resample_gt(gt_ml > 0, shape)
            gt_t = torch.from_numpy(gt)
            coarse = _native_from_passes(shape, task["passes"][:1])
            dice = float(dice_binary(torch.from_numpy(native), gt_t))
            coarse_only = float(dice_binary(torch.from_numpy(coarse), gt_t))
        pred_native, out_affine = _to_original_orientation(native, target_path)
        pred_path = None
        if out_path is not None:
            nib.save(nib.Nifti1Image(pred_native.astype(np.uint8), out_affine), str(out_path))
            pred_path = Path(out_path)
        return {"pred": pred_native, "affine": out_affine, "dice": dice,
                "coarse_only_dice": coarse_only, "pred_path": pred_path}

    # --- multi-label: per-label metrics + small-organ-wins id-valued stitch ------
    # Bounded memory: materialize one label's native at a time (never all at once).
    dice = coarse_only = None
    if gt_ml is not None:
        dice, coarse_only = {}, {}
    sizes = {}
    for task in tasks:
        lab = task["label"]
        native = _native_from_passes(shape, task["passes"])
        sizes[lab] = int(native.sum())
        if gt_ml is not None:
            gt = _resample_gt(gt_ml == lab, shape)
            gt_t = torch.from_numpy(gt)
            coarse = _native_from_passes(shape, task["passes"][:1])
            dice[lab] = float(dice_binary(torch.from_numpy(native), gt_t))
            coarse_only[lab] = float(dice_binary(torch.from_numpy(coarse), gt_t))
    macro = float(np.mean(list(dice.values()))) if dice else None

    # Write larger organs first so smaller ones win where predictions overlap.
    combined = np.zeros(shape, dtype=np.uint8)
    for task in sorted(tasks, key=lambda t: sizes[t["label"]], reverse=True):
        combined[_native_from_passes(shape, task["passes"])] = task["label"]

    pred_native, out_affine = _to_original_orientation(combined, target_path)
    # Carry organ names/colors onto the prediction (GT table preferred, else context).
    ext, label_names = _output_label_table(gt_path, context_pairs, [t["label"] for t in tasks])
    pred_path = None
    if out_path is not None:
        out_img = nib.Nifti1Image(pred_native.astype(np.uint8), out_affine)
        if ext is not None:
            out_img.header.extensions.append(ext)
        nib.save(out_img, str(out_path))
        pred_path = Path(out_path)

    return {"pred": pred_native, "affine": out_affine, "dice": dice,
            "coarse_only_dice": coarse_only, "macro_dice": macro,
            "labels": [t["label"] for t in tasks], "label_names": label_names,
            "pred_path": pred_path}
