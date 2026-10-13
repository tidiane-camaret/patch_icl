"""Nifti in-context cascade inference — predict a target organ mask from context
(image, binary-mask) nifti pairs via the coarse->fine cascade, GT-free for the target.

Runs the SAME N-level cascade as training / eval: cascade.run_cascade over a small
in-memory NiftiProvider (each level re-crops the target on the previous level's predicted
centre-of-mass and, when data.cascade_query_prior is set, feeds the previous mask as the
query prior). The hard per-level predictions are stitched coarse->fine into the target's
native volume and written back on the target's on-disk grid.

See docs/superpowers/specs/2026-08-12-nifti-incontext-cascade-inference-design.md.
"""
import random
import re
import sys
import time
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

from src.totalseg_dataset import normalize_ct, resolve_ct_norm
from src.totalseg_dataloader_incontext import (
    organ_crop_arrays, place_image, place_label, resample_binary,
)
from src.incontext_dataset_v2 import LoadRequest, LoadResult
from src.providers.totalseg import NativeCrop, build_native_crop  # noqa: F401 (NativeCrop: return-type annotation)
from eval import _build_model, _warn_uninherited_data
from evaluate import _write_native, dice_binary
from cascade import run_cascade, _recrop_level
from common import DEVICE


def load_nifti(path):
    """Load a nifti -> (array, affine), reoriented to closest-canonical (RAS) to match
    the training preprocessing (scripts/convert_to_npy.py uses nib.as_closest_canonical)."""
    img = nib.as_closest_canonical(nib.load(str(path)))
    return np.asanyarray(img.dataobj), img.affine


def voxel_spacing(affine):
    """Per-axis mm/voxel (3,) from the affine, aligned with the array axes."""
    return [float(v) for v in voxel_sizes(affine)]


def _resolve_context_center(req, binmask):
    """Context crop centre per req.center_mode, mirroring providers/totalseg.py
    _resolve_center: "com" (default) -> the mask centroid; "random_fg" -> a uniformly
    random in-mask voxel drawn via req.rng (falls back to the centroid if the mask is
    empty). An explicit req.center always wins."""
    if req.center is not None:
        return req.center
    if req.center_mode == "random_fg":
        coords = np.argwhere(np.asarray(binmask))
        if coords.shape[0] > 0:
            return tuple(int(v) for v in coords[req.rng.randrange(coords.shape[0])])
    return mask_centroid(binmask)


def mask_centroid(mask):
    """Integer centre-of-mass (d,h,w) of a binary mask; volume centre + warn if empty."""
    fg = np.asarray(mask) > 0
    if not fg.any():
        warnings.warn("infer_nifti: empty context/target mask — using volume centre.",
                      stacklevel=2)
        return tuple(s // 2 for s in fg.shape)
    idx = np.nonzero(fg)
    return tuple(int(a.mean()) for a in idx)


def prep_target(ct, sp, center, *, T, crop_mm, normalize_fn=None):
    """Native CT + centre -> (img_t (1,T,T,T), crop_geom (4,3)).

    No target label, so ct doubles as the label array for organ_crop_arrays (its
    crop_lbl output is discarded). rng is unused at jitter=0 (centred crop).
    `normalize_fn`, when given, maps the raw cropped slice to model input space
    BEFORE placement (crop_and_place's own pattern) -- `ct` is raw HU, not
    pre-normalized, so the GPU realize path (which normalizes its own un-normalized
    NativeCrop.image) sees the exact same un-normalized source."""
    crop_ct, _, out_sizes, pad_lo, geom = organ_crop_arrays(
        ct, ct, center, sp, image_size=(T, T, T), crop_mm=crop_mm,
        jitter=0, rng=random.Random(0))
    crop_ct = np.ascontiguousarray(crop_ct)
    if normalize_fn is not None:
        crop_ct = normalize_fn(crop_ct)
    return place_image(crop_ct, out_sizes, pad_lo, T), geom


def prep_context(ct, mask, sp, center, *, T, crop_mm, mask_downsample, occ_thr,
                 normalize_fn=None):
    """Native (CT, binary mask) + centre -> (img_t (1,T,T,T), mask_t (T,T,T) long,
    crop_geom (4,3)). `ct` is raw HU; see prep_target's normalize_fn note."""
    assert ct.shape == mask.shape, f"context ct {ct.shape} != mask {mask.shape}"
    crop_ct, crop_mask, out_sizes, pad_lo, geom = organ_crop_arrays(
        ct, mask, center, sp, image_size=(T, T, T), crop_mm=crop_mm,
        jitter=0, rng=random.Random(0))
    crop_ct = np.ascontiguousarray(crop_ct)
    if normalize_fn is not None:
        crop_ct = normalize_fn(crop_ct)
    img_t = place_image(crop_ct, out_sizes, pad_lo, T)
    mask_small = resample_binary(np.asarray(crop_mask) > 0, tuple(out_sizes),
                                 mode=mask_downsample, occ_thr=occ_thr)
    return img_t, place_label(mask_small, out_sizes, pad_lo, T), geom


_FG = "__fg__"   # single-binary-organ label key (multi-label uses str(id))


class NiftiProvider:
    """In-memory VolumeProvider for cascade.run_cascade over already-loaded nifti arrays.

    subject keys: 'tgt' (the target CT, GT-free -> zero label) and 'ctx{k}' (context k).
    `cls` is the label id as a string, or _FG for single binary mode. Level-0 target crop
    falls back to the volume centre; run_cascade passes an explicit native-voxel centre for
    every finer level. Contexts always self-centre on their own mask centroid."""

    def __init__(self, tgt_ct, tgt_sp, contexts, *, T, mask_downsample, occ_thr, ct_spec=None):
        self.tgt_ct = tgt_ct                           # RAW HU -- see load()/load_native_crop
        self.tgt_sp = list(tgt_sp)
        self.contexts = contexts                      # [(ct_raw, id_mask, spacing), ...] -- RAW HU
        self.T = int(T)
        self.mask_downsample = mask_downsample
        self.occ_thr = float(occ_thr)
        self.ct_spec = resolve_ct_norm(ct_spec)
        self.tgt_center0 = tuple(s // 2 for s in tgt_ct.shape)
        # Dummy all-zero "label" for the GPU path's organ_crop_arrays call on the target (it
        # needs SOME label array to crop alongside the image) -- built once, reused for every
        # target load_native_crop call. Unlike the CPU prep_target's trick of reusing `ct`
        # itself as a throwaway label array (safe there because its crop_lbl output is
        # discarded outright), build_native_crop DOES check `crop_lbl == class_idx` to build
        # has_fg/label_frac, so it needs a label array that is genuinely always zero, not one
        # that merely goes unused.
        self._tgt_zero_label = np.zeros(tgt_ct.shape, dtype=np.uint8)
        # Multi-label mode calls load() for EVERY label in a chunk, but the target's level-0
        # crop (center=None -> self.tgt_center0, independent of `cls`) is byte-identical
        # across all of them -- without this, a 20-label run was re-running the native CPU
        # resample of the full target volume 20 times for what is provably the same output
        # (prep_target's result depends only on (tgt_ct, tgt_sp, center, T, sp), none of
        # which vary by cls at level 0; inference also never jitters -- jitter=0 throughout
        # predict_nifti -- so there's no RNG dependence to worry about either). Keyed on the
        # RESOLVED center (post center_mode resolution upstream), not cls, so it also catches
        # the (rarer) case of two different labels' predicted centroids at level>0 landing on
        # the exact same voxel. Cloned on return so no caller can mutate the cached tensors.
        self._load_cache: dict = {}
        # (subject, cls) -> binmask / centroid, spacing- and level-independent (see the
        # context branch of load() below for why these are cached separately from
        # _load_cache, which is keyed on the resolved center and so can't dedupe the work
        # that PRODUCES that center in the first place).
        self._binmask_cache: dict = {}
        self._centroid_cache: dict = {}

    def subjects_for(self, cls):                      # unused on the run_cascade path
        return ["tgt"]

    def _resolve_ctx_binmask_center(self, subject, cls, c_idmask, req):
        """Shared by load() and load_native_crop(): context always self-centres on its OWN
        mask (req.center is always None for context tasks, every level -- see
        _recrop_level), so binmask/centroid are recomputed from scratch at EVERY cascade
        level even though they don't depend on level/spacing at all -- cache both, keyed on
        (subject, cls) alone. The centroid itself is only cached when center_mode=="com"
        (the deterministic path); "random_fg" draws a fresh random in-mask voxel each call
        on purpose, so it's deliberately excluded (binmask is still safe to reuse either
        way -- it's just the fixed (c_idmask==cls) comparison)."""
        bkey = (subject, cls)
        binmask = self._binmask_cache.get(bkey)
        if binmask is None:
            binmask = (c_idmask > 0) if cls == _FG else (c_idmask == int(cls))
            self._binmask_cache[bkey] = binmask
        if req.center is None and req.center_mode != "random_fg":
            center = self._centroid_cache.get(bkey)
            if center is None:
                center = _resolve_context_center(req, binmask)
                self._centroid_cache[bkey] = center
        else:
            center = _resolve_context_center(req, binmask)
        return binmask, center

    def load(self, subject, cls, req: LoadRequest) -> LoadResult:
        sp = float(req.crop_spacing_mm)
        spacing = torch.full((3,), sp, dtype=torch.float32)
        normalize_fn = lambda a: normalize_ct(a, self.ct_spec)  # noqa: E731
        if subject == "tgt":
            center = req.center if req.center is not None else self.tgt_center0
            key = ("tgt", None, tuple(float(c) for c in center), sp)
            cached = self._load_cache.get(key)
            if cached is not None:
                img_t, geom = cached
            else:
                img_t, geom = prep_target(self.tgt_ct, self.tgt_sp, center, T=self.T,
                                          crop_mm=sp, normalize_fn=normalize_fn)
                self._load_cache[key] = (img_t, geom)
            img_t = img_t.clone()
            label_t = torch.zeros((self.T, self.T, self.T), dtype=torch.long)
        else:
            k = int(subject[3:])                      # 'ctx0' -> 0
            c_ct, c_idmask, c_sp = self.contexts[k]
            binmask, center = self._resolve_ctx_binmask_center(subject, cls, c_idmask, req)
            key = (subject, cls, tuple(float(c) for c in center), sp)
            cached = self._load_cache.get(key)
            if cached is not None:
                img_t, label_t, geom = cached
            else:
                img_t, label_t, geom = prep_context(
                    c_ct, binmask, c_sp, center, T=self.T, crop_mm=sp,
                    mask_downsample=self.mask_downsample, occ_thr=self.occ_thr,
                    normalize_fn=normalize_fn)
                self._load_cache[key] = (img_t, label_t, geom)
            img_t, label_t = img_t.clone(), label_t.clone()
        # Copy geom too, cache hit or not -- torch.as_tensor on a numpy array can be a
        # zero-copy view, so a cached numpy geom reused across calls could otherwise let one
        # caller's in-place edit corrupt every other cache hit's "geom" silently.
        geom = np.array(geom, copy=True) if not torch.is_tensor(geom) else geom.clone()
        if not torch.is_tensor(geom):
            geom = torch.as_tensor(geom, dtype=torch.long)
        return LoadResult(image=img_t, label=label_t, spacing=spacing,
                          crop_geom=geom.long())

    def load_native_crop(self, subject, cls, req: LoadRequest) -> NativeCrop:
        """GPU-realize counterpart to load() -- same crop geometry (organ_crop_arrays with
        the SAME args), but returns an un-normalized, un-resampled NativeCrop payload for
        cascade._recrop_level's realize_crop=True path (src.gpu_realize_crop.
        realize_native_crops) to finish on `device` instead of here on CPU. Reuses the exact
        same center/binmask caches as load() -- both paths resolve centers identically, so
        results from the two paths should match to the <2e-2 tolerance documented in
        test_gpu_realize_crop.py (decimation + area-pool ordering differs slightly from
        direct trilinear, not a bug)."""
        sp = float(req.crop_spacing_mm)
        jitter = req.jitter or 0
        if subject == "tgt":
            center = req.center if req.center is not None else self.tgt_center0
            crop_ct, crop_lbl, out_sizes, pad_lo, geom = organ_crop_arrays(
                self.tgt_ct, self._tgt_zero_label, center, self.tgt_sp,
                image_size=(self.T, self.T, self.T), crop_mm=sp, jitter=jitter, rng=req.rng)
            class_idx = 1   # irrelevant: crop_lbl is genuinely all-zero, so has_fg is always False
        else:
            k = int(subject[3:])                      # 'ctx0' -> 0
            c_ct, c_idmask, c_sp = self.contexts[k]
            binmask, center = self._resolve_ctx_binmask_center(subject, cls, c_idmask, req)
            crop_ct, crop_lbl, out_sizes, pad_lo, geom = organ_crop_arrays(
                c_ct, binmask.astype(np.uint8), center, c_sp,
                image_size=(self.T, self.T, self.T), crop_mm=sp, jitter=jitter, rng=req.rng)
            class_idx = 1   # binmask is already {0,1} -- matches load()'s own pre-binarized mask
        return build_native_crop(crop_ct, crop_lbl, class_idx, out_sizes, pad_lo, geom,
                                 crop_spacing_mm=sp, norm=self.ct_spec)


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
                  gt_path=None, out_path=None, explicit_data_keys=None, model=None):
    """Run the v2 coarse->fine in-context cascade on nifti files (GT-free target).

    Wraps cascade.run_cascade over an in-memory NiftiProvider: level 0 crops the target on
    the volume centre; each finer level re-crops it on the previous level's predicted COM and
    (when data.cascade_query_prior is set, default 'pred') feeds the previous mask as the
    query prior. Runs patchset3d only (run_cascade needs the {'final_logit'} forward).

    cfg            : OmegaConf cfg (same surface as experiments/3d/eval.py). Uses
                     data.image_size / mask_downsample / mask_occupancy_thr /
                     cascade_spacings / cascade_query_prior[_hard] / cascade_recrop_workers /
                     cascade_center_mode / cascade_center_fg_thr and eval.model /
                     eval.checkpoint. Falls back to eval.spacing_sweep for the spacing
                     schedule when data.cascade_spacings is unset. None of these are
                     restored from the checkpoint (only `arch` is) -- re-supply the
                     training values via --override for a faithful eval; _warn_uninherited_data
                     (eval.py._FIDELITY_KEYS) flags drift on the ones it tracks.
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

    Returns {"pred", "affine", "dice", "coarse_only_dice", "pred_path", "timing"}; multi-label
    adds "labels" and "macro_dice" and makes dice/coarse_only per-label dicts. "timing" is
    {"total_s": whole-call wall time, "cascade_s": time spent in the per-chunk cascade
    forward loop, "per_label_ms": {label: per-label cascade ms} (multi-label only; each
    chunk's wall time split evenly over its batch_size labels, same convention as
    cascade.py's evaluate_cascade per_sample_ms)}.

    explicit_data_keys : dotted data.* keys (e.g. {"data.mask_occupancy_thr"}) the CALLER
                         knows were a deliberate user choice this run (infer_cli.py's own CLI
                         flags), not just a composed-config default -- see
                         eval._explicitly_overridden. Needed because this function is reached
                         via a programmatic hydra.compose(), where the Hydra-CLI-override
                         detection eval.py's own callers rely on isn't available.
    model          : pre-built model (from eval._build_model(cfg)), to reuse across many
                     calls against the same checkpoint in one process (infer_cli.py's
                     --manifest mode) instead of re-loading/rebuilding it per target. None
                     (default) builds it fresh from cfg.eval.checkpoint, same as before
                     this param existed.
    """
    if not context_pairs:
        raise ValueError("predict_nifti needs at least one context pair (in-context model)")
    t_start = time.perf_counter()

    # eval.py's own @hydra.main entrypoint sets these before building/running a model, but
    # this function is reached via infer_cli.py (and other direct callers) which bypass that
    # entrypoint entirely -- without this, cuDNN re-autotunes its conv algorithm choice on
    # every level instead of reusing one across same-shape (T,T,T) calls, and fp32 matmuls
    # skip TF32 tensor cores. Idempotent; safe to set on every call.
    if DEVICE.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    _warn_uninherited_data(cfg, explicit_keys=explicit_data_keys)
    # model=: callers running many targets against the SAME checkpoint in one process
    # (infer_cli.py's --manifest mode) build it once and pass it in here, skipping the
    # torch.load + arch-rebuild + state_dict-copy that _build_model does on every call. Single-target
    # callers (infer_cli.py, tests) are unaffected -- default None builds as before.
    if model is None:
        model = _build_model(cfg)
    if hasattr(model, "eval"):
        model.eval()
    T = int(cfg.data.image_size[0])
    crop_ds = cfg.data.get("mask_downsample", "occupancy")
    crop_thr = float(cfg.data.get("mask_occupancy_thr", 0.1))
    sched = cfg.data.get("cascade_spacings") or cfg.eval.get("spacing_sweep")
    if not sched:
        raise ValueError("predict_nifti needs data.cascade_spacings (or eval.spacing_sweep)")
    spacings = [float(s) for s in sched]
    if len(spacings) < 2:
        raise ValueError(f"cascade needs >=2 spacings, got {spacings}")
    qp = cfg.data.get("cascade_query_prior", "pred")
    qp_hard = bool(cfg.data.get("cascade_query_prior_hard", False))
    recrop_workers = int(cfg.data.get("cascade_recrop_workers", 1))
    center_mode = cfg.data.get("cascade_center_mode", "com")
    center_fg_thr = float(cfg.data.get("cascade_center_fg_thr", 0.5))
    # gpu_realize_crop: opt-in (default off) GPU-based crop/resample, same mechanism
    # training already uses (cascade._recrop_level's realize_crop branch -> NativeCrop ->
    # src.gpu_realize_crop.realize_native_crops). Off by default until proven out on real
    # inference workloads -- the CPU path (provider.load/prep_target/prep_context) remains
    # the default, unchanged behavior. ct_spec is resolved either way: the CPU path now
    # also normalizes at crop time (not whole-volume load time, see below), so both paths
    # need the same explicit spec rather than prep_target/prep_context's old implicit
    # "already normalized" input contract.
    realize_crop = bool(cfg.data.get("gpu_realize_crop", False))
    ct_spec = resolve_ct_norm(cfg.data.get("ct_norm"))
    from train import model_output_is_prob
    is_prob = bool(model_output_is_prob(cfg))
    N = len(spacings)

    # --- load target + contexts once (arrays reused across labels) -------------
    # RAW HU, not pre-normalized: NiftiProvider normalizes per-crop now (at prep_target/
    # prep_context time on the CPU path, inside NativeCrop's realize step on the GPU path)
    # instead of once over the whole volume -- normalize_ct is pointwise with fixed
    # constants (see its own docstring), so this is mathematically identical, but a
    # pre-normalized volume would otherwise get normalized a SECOND time by the GPU path
    # (which expects its NativeCrop.image input to still be raw HU).
    tgt_ct, affine = load_nifti(target_path)
    tgt_sp = voxel_spacing(affine)
    shape = tgt_ct.shape

    contexts = []  # (ct_raw, id_mask, spacing) shared across labels
    for img_p, msk_p in context_pairs:
        c_ct, c_aff = load_nifti(img_p)
        c_msk, _ = load_nifti(msk_p)
        contexts.append((c_ct, np.asarray(c_msk), voxel_spacing(c_aff)))
    K = len(contexts)

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

    provider = NiftiProvider(tgt_ct, tgt_sp, contexts, T=T,
                             mask_downsample=crop_ds, occ_thr=crop_thr, ct_spec=ct_spec)
    tasks = [{"label": lab, "passes": []} for lab in labels]

    # --- coarse->fine cascade (cascade.run_cascade), batched over labels -------
    cascade_s = 0.0
    for chunk in _iter_chunks(tasks, batch_size):
        B = len(chunk)
        meta = {
            "subjects": ["tgt"] * B,
            "context_subjects": [[f"ctx{k}" for k in range(K)] for _ in range(B)],
            "label_names": [(_FG if t["label"] is None else str(t["label"])) for t in chunk],
            # realize_crop=True's branch in _recrop_level reads T off `batch["image"].shape
            # [-1]` -- fine for its OTHER caller (run_cascade's level>0 recrops, where
            # `batch` is a previous level's real output), but level 0 here has no prior
            # batch to reuse, so this is a placeholder standing in for just that one shape
            # lookup (never read as actual data; the CPU branch doesn't touch "image" here
            # at all). Inert when realize_crop=False.
            "image": torch.empty(1, 1, 1, 1, T),
        }
        # Level-0 batch: target on the volume centre (center=None), contexts self-centred.
        # mask_downsample/occ_thr/ct_spec/device only matter when realize_crop=True (the
        # CPU path reads mask_downsample/occ_thr off the provider instance instead); passed
        # unconditionally since they're inert on the CPU branch.
        batch0 = _recrop_level(provider, meta, [None] * B, spacings[0],
                               step=0, seed=0, level=0, jitter=0,
                               recrop_workers=recrop_workers, realize_crop=realize_crop,
                               mask_downsample=crop_ds, occ_thr=crop_thr,
                               ct_spec=ct_spec, device=DEVICE)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        t0_chunk = time.perf_counter()
        with torch.no_grad():
            res = run_cascade(model, provider, batch0, augmentor=None, spacings=spacings,
                              device=DEVICE, training=False, step=0, seed=0, jitter=0,
                              is_prob=is_prob, want_hard_preds=True,
                              recrop_workers=recrop_workers, realize_crop=realize_crop,
                              mask_downsample=crop_ds, occ_thr=crop_thr, ct_spec=ct_spec,
                              query_prior=qp, query_prior_hard=qp_hard,
                              center_mode=center_mode, center_fg_thr=center_fg_thr)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        dt_chunk = time.perf_counter() - t0_chunk
        cascade_s += dt_chunk
        per_label_ms = dt_chunk * 1e3 / B     # even split across this chunk's batched labels
        for j, task in enumerate(chunk):
            task["passes"] = [(res.hard_preds[li][j].cpu().numpy().astype(bool),
                               res.geoms[li][j].cpu().numpy()) for li in range(N)]
            task["time_ms"] = round(per_label_ms, 1)

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
            # dice_binary is a CPU boolean reduction by default; on this native-grid size
            # (~10M voxels) moving to DEVICE first (even counting the transfer) measured
            # ~18x faster than computing in place on CPU. Scoped to this call site rather
            # than dice_binary itself -- that function is shared with training/eval code
            # where inputs are typically already on GPU, or (eval_pshape_families.py)
            # deliberately kept on CPU by the caller.
            gt_t = torch.from_numpy(gt).to(DEVICE)
            coarse = _native_from_passes(shape, task["passes"][:1])
            dice = float(dice_binary(torch.from_numpy(native).to(DEVICE), gt_t))
            coarse_only = float(dice_binary(torch.from_numpy(coarse).to(DEVICE), gt_t))
        pred_native, out_affine = _to_original_orientation(native, target_path)
        pred_path = None
        if out_path is not None:
            nib.save(nib.Nifti1Image(pred_native.astype(np.uint8), out_affine), str(out_path))
            pred_path = Path(out_path)
        timing = {"total_s": round(time.perf_counter() - t_start, 2),
                  "cascade_s": round(cascade_s, 2)}
        return {"pred": pred_native, "affine": out_affine, "dice": dice,
                "coarse_only_dice": coarse_only, "pred_path": pred_path, "timing": timing}

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
            # See the single-organ branch above for why this moves to DEVICE first.
            gt_t = torch.from_numpy(gt).to(DEVICE)
            coarse = _native_from_passes(shape, task["passes"][:1])
            dice[lab] = float(dice_binary(torch.from_numpy(native).to(DEVICE), gt_t))
            coarse_only[lab] = float(dice_binary(torch.from_numpy(coarse).to(DEVICE), gt_t))
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

    timing = {"total_s": round(time.perf_counter() - t_start, 2),
              "cascade_s": round(cascade_s, 2),
              "per_label_ms": {t["label"]: t["time_ms"] for t in tasks}}
    return {"pred": pred_native, "affine": out_affine, "dice": dice,
            "coarse_only_dice": coarse_only, "macro_dice": macro,
            "labels": [t["label"] for t in tasks], "label_names": label_names,
            "pred_path": pred_path, "timing": timing}
