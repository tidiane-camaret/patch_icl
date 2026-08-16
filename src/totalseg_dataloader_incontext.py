"""
TotalSegmentator In-Context 3D DataLoader.

Each item is a (target, context) pair for in-context segmentation:
  image       : (1, D, H, W) float32 — query volume
  label       : (D, H, W) int64      — binary mask for the target class
  context_in  : (K, 1, D, H, W) float32 — K context volumes
  context_out : (K, D, H, W) int64       — K context masks (same class)

On first use, scans every label.npy to build a subject→classes index and saves
it as a pickle next to the data.  All subsequent runs load the cache instantly.
The cache covers all 117 classes, so it is valid for any class subset or split.

Usage
-----
  ds = TotalSegInContextDataset(
      root="/data/totalseg",
      classes=["kidney_left"],
      image_size=(64, 64, 64),
      split="train",
      context_size=3,
  )
  loader = DataLoader(ds, batch_size=4, collate_fn=incontext_collate_fn, ...)
"""

import csv
import hashlib
import json
import math
import os
import pickle
import random
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from data.totalseg_classes import ALL_CLASSES
from src.totalseg_dataset import (
    _ALL_CLASSES_IDX,
    _load_ct,
    _build_label_volume,
    _iso_size,
    _resize_volume,
    normalize_ct,
    normalize_mri,
    CT_MEAN,
    CT_STD,
)
from src.augmentations import (
    apply_task_aug, apply_intensity_aug, apply_synth_aug, apply_per_image_aug,
)


def _to_ns(obj):
    """Recursively convert a DictConfig/dict aug config to nested SimpleNamespace.

    Per-item augmentation reads dozens of cfg fields; omegaconf's __getattr__ (validation
    + interpolation resolution) cost ~30% of __getitem__ wall time in the DataLoader hot
    path. Converting once at dataset init makes those plain-Python attribute lookups.
    Lists stay lists (indexing/unpacking preserved); leaves pass through. getattr(ns, k,
    default) still works for the optional fields the aug fns probe.
    """
    try:
        from omegaconf import OmegaConf, DictConfig, ListConfig
        if isinstance(obj, (DictConfig, ListConfig)):
            obj = OmegaConf.to_container(obj, resolve=True)
    except Exception:
        pass
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_ns(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_ns(v) for v in obj]
    return obj


def _lazy_shuffle(rng, x):
    """Yield the elements of list `x` in uniformly-random order, doing only O(k) RNG
    work to produce the first k (forward Fisher-Yates). Consuming all of it is a full
    permutation; stopping early (the common case: take context_size of ~1000 candidates)
    avoids the O(len(x)) cost of rng.shuffle. Mutates `x` in place (caller passes a
    throwaway list). `rng` is `random` or a seeded random.Random (eval determinism kept)."""
    n = len(x)
    for i in range(n):
        j = rng.randrange(i, n)
        x[i], x[j] = x[j], x[i]
        yield x[i]

def _rand_rotation(rng) -> np.ndarray:
    """3x3 rotation matrix from three uniform Euler angles (intrinsic ZYX)."""
    a, b, c = (rng.uniform(0.0, 2.0 * math.pi) for _ in range(3))
    ca, sa, cb, sb, cc, sc = (math.cos(a), math.sin(a), math.cos(b),
                              math.sin(b), math.cos(c), math.sin(c))
    rz = np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]], dtype=np.float32)
    ry = np.array([[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]], dtype=np.float32)
    rx = np.array([[1, 0, 0], [0, cc, -sc], [0, sc, cc]], dtype=np.float32)
    return rz @ ry @ rx


def make_ellipsoid_label(image, spacing, rng, min_mm=1.0, max_mm=50.0,
                         body_thresh=-0.46, max_tries=64):
    """Random rotated-ellipsoid binary label on the grid of `image` (1,D,H,W or D,H,W,
    normalised CT). Per-axis radii ~ U(min_mm, max_mm) mm are converted to voxels via
    `spacing` (mm/voxel, len 3); the centroid is rejection-sampled inside the body
    (image > body_thresh, i.e. not air). Returns (label (D,H,W) uint8 tensor, centroid
    (d,h,w) ints, radii_mm (3,) float — the sampled per-axis radii in mm, i.e. the generative
    size independent of body clipping). Never empty — the centroid voxel is always set (guards a
    sub-voxel radius). `rng` is `random` or a seeded random.Random (eval determinism preserved)."""
    img = image[0] if image.ndim == 4 else image
    D, H, W = img.shape
    body = img > body_thresh
    ctr = None
    for _ in range(max_tries):
        d, h, w = rng.randrange(D), rng.randrange(H), rng.randrange(W)
        if bool(body[d, h, w]):
            ctr = (d, h, w)
            break
    if ctr is None:
        ctr = (D // 2, H // 2, W // 2)
    cd, ch, cw = ctr
    sp = np.array([float(s) for s in spacing], dtype=np.float32)
    radii_mm = np.array([rng.uniform(min_mm, max_mm) for _ in range(3)], dtype=np.float32)
    radii = np.maximum(radii_mm / np.maximum(sp, 1e-3), 0.5)     # voxels, >= half a voxel
    rot = _rand_rotation(rng)
    rmax = int(math.ceil(float(radii.max()))) + 1
    d0, d1 = max(0, cd - rmax), min(D, cd + rmax + 1)
    h0, h1 = max(0, ch - rmax), min(H, ch + rmax + 1)
    w0, w1 = max(0, cw - rmax), min(W, cw + rmax + 1)
    zz, yy, xx = np.meshgrid(np.arange(d0, d1) - cd, np.arange(h0, h1) - ch,
                             np.arange(w0, w1) - cw, indexing="ij")
    offs = np.stack([zz.ravel(), yy.ravel(), xx.ravel()]).astype(np.float32)  # (3, N)
    local = rot.T @ offs                                        # into the ellipsoid frame
    inside = (((local / radii[:, None]) ** 2).sum(0) <= 1.0).reshape(zz.shape)
    label = np.zeros((D, H, W), dtype=np.uint8)
    label[d0:d1, h0:h1, w0:w1][inside] = 1                       # basic-slice view: writes through
    label[cd, ch, cw] = 1
    return torch.from_numpy(label), ctr, radii_mm


# Inverse map: orig label index → class name (covers all 117 classes)
_IDX_TO_CLASS: dict[int, str] = {v: k for k, v in _ALL_CLASSES_IDX.items()}


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def _adj_for_subject(root: Path, subj: str, synth_fname: str) -> tuple[str, dict | None]:
    """Vectorised face-adjacency computation for one subject (module-level for pickling)."""
    try:
        arr = np.load(root / subj / synth_fname, mmap_mode="r")
        edge_lists = []
        for axis in range(3):
            sl_a = [slice(None)] * 3; sl_a[axis] = slice(None, -1)
            sl_b = [slice(None)] * 3; sl_b[axis] = slice(1, None)
            a = arr[tuple(sl_a)].ravel()
            b = arr[tuple(sl_b)].ravel()
            mask = (a != b) & (a > 0) & (b > 0)
            if mask.any():
                edge_lists.append(np.stack([a[mask], b[mask]], axis=1))
        if not edge_lists:
            return subj, {}
        edges = np.concatenate(edge_lists)
        edges = np.sort(edges, axis=1)       # canonical direction per edge
        edges = np.unique(edges, axis=0)     # deduplicate
        us = np.concatenate([edges[:, 0], edges[:, 1]])
        vs = np.concatenate([edges[:, 1], edges[:, 0]])
        order = np.argsort(us, stable=True)
        us, vs = us[order], vs[order]
        unique_u, starts = np.unique(us, return_index=True)
        ends = np.append(starts[1:], len(us))
        return subj, {
            int(uid): frozenset(vs[starts[j]:ends[j]].tolist())
            for j, uid in enumerate(unique_u.tolist())
        }
    except Exception as e:
        return subj, None


def _bbox_for_subject(root: Path, subj: str) -> tuple[str, dict | None]:
    """Compute per-class centroids for one subject (module-level for pickling)."""
    try:
        arr = np.load(root / subj / "label.npy", mmap_mode="r")
        D, H, W = arr.shape
        d_g = np.arange(D, dtype=np.float32)[:, None, None]
        h_g = np.arange(H, dtype=np.float32)[None, :, None]
        w_g = np.arange(W, dtype=np.float32)[None, None, :]
        result: dict[str, tuple[int, int, int]] = {}
        for idx in np.unique(arr):
            if idx == 0 or idx not in _IDX_TO_CLASS:
                continue
            mask = (arr == idx)
            n = int(mask.sum())
            if n == 0:
                continue
            cd = int((d_g * mask).sum() / n)
            ch = int((h_g * mask).sum() / n)
            cw = int((w_g * mask).sum() / n)
            result[_IDX_TO_CLASS[idx]] = (cd, ch, cw)
        return subj, result
    except Exception:
        return subj, None


def organ_crop_arrays(ct_mm, label_mm, center, sp, *, image_size, crop_mm, jitter, rng):
    """Pure array-level organ crop. Returns (crop_ct, crop_lbl, out_sizes, pad_lo, crop_geom).

    Slices a centre crop of fixed physical extent (T*crop_mm/axis) from ct_mm/label_mm
    (same shape), records where it lands in the final T³ grid. Behaviour extracted verbatim
    from TotalSegInContextDataset._organ_crop_arrays; `rng` supplies the crop jitter."""
    T = image_size[0]
    cd, ch, cw = center
    D, H, W = label_mm.shape
    phys_ref = T * crop_mm
    target_sizes = [max(1, round(phys_ref / spi)) for spi in sp]
    crop_sizes = [min(dim, t) for t, dim in zip(target_sizes, (D, H, W))]
    starts = []
    for c, s, cs in zip((cd, ch, cw), (D, H, W), crop_sizes):
        smax = max(0, s - cs)
        ideal = c - cs // 2
        lo = min(max(0, ideal - jitter), smax)
        hi = min(max(0, ideal + jitter), smax)
        starts.append(rng.randint(lo, hi))
    d0, h0, w0 = starts
    crop_ct = ct_mm[d0:d0 + crop_sizes[0], h0:h0 + crop_sizes[1], w0:w0 + crop_sizes[2]]
    crop_lbl = label_mm[d0:d0 + crop_sizes[0], h0:h0 + crop_sizes[1], w0:w0 + crop_sizes[2]]
    out_sizes = [max(1, min(T, round(cs / t * T))) for cs, t in zip(crop_sizes, target_sizes)]
    pad_lo = [(T - o) // 2 for o in out_sizes]
    crop_geom = torch.tensor([starts, list(crop_sizes), out_sizes, pad_lo], dtype=torch.long)
    return crop_ct, crop_lbl, out_sizes, pad_lo, crop_geom


def place_image(crop_ct, out_sizes, pad_lo, T):
    """Resample the native CT slice to out_sizes (trilinear) and centre it in an
    air-filled T³ tensor. Returns (1, T, T, T)."""
    img_small = F.interpolate(
        torch.from_numpy(crop_ct.astype(np.float32))[None, None],
        size=tuple(out_sizes), mode="trilinear", align_corners=False)[0]
    if all(o == T for o in out_sizes):
        return img_small
    image_t = torch.full((1, T, T, T), float(crop_ct.min()), dtype=torch.float32)
    sl = (slice(None),) + tuple(slice(p, p + o) for p, o in zip(pad_lo, out_sizes))
    image_t[sl] = img_small
    return image_t


def place_label(label_small, out_sizes, pad_lo, T):
    """Centre an already-resampled label (spatial dims out_sizes, long) in a
    background-0 T³ tensor. Returns (T, T, T)."""
    if all(o == T for o in out_sizes):
        return label_small
    label_t = torch.zeros(T, T, T, dtype=torch.long)
    sl = tuple(slice(p, p + o) for p, o in zip(pad_lo, out_sizes))
    label_t[sl] = label_small
    return label_t


def resample_binary(bin_np, size, *, mode, occ_thr):
    """Resize a binary mask to `size` -> long (0/1). "occupancy" area-pools + thresholds
    (thin structures survive; non-empty input never returns empty); "nearest" point-samples."""
    t = torch.from_numpy(np.ascontiguousarray(bin_np, dtype=np.float32))[None, None]
    if mode == "occupancy":
        frac = F.interpolate(t, size=size, mode="area")[0, 0]
        out = frac >= occ_thr
        if not bool(out.any()) and bin_np.any():
            out.view(-1)[int(frac.argmax())] = True
        return out.long()
    return (F.interpolate(t, size=size, mode="nearest")[0, 0] > 0.5).long()


class TotalSegInContextDataset(Dataset):
    """
    In-context segmentation dataset over TotalSegmentator 3-D volumes.

    Args:
        root         : Dataset root (contains s0000/, s0001/, …).
        classes      : Organ names to include.  Each becomes a separate sample
                       with a binary (0/1) label volume.
        image_size   : (D, H, W) resize target.  Pass None for native size
                       (incompatible with batch_size > 1).
        split        : 'train' | 'val' | 'test' | None (all subjects).
        meta_csv     : Path to meta.csv; auto-detected when split is given.
        context_size : Number of context (image, mask) pairs per item.
        max_subjects : Limit to the first N subjects (for quick experiments).
        use_crop     : If True, load native-resolution ct.npy/label.npy and
                       extract an organ-centred random crop of image_size
                       instead of using pre-resized files.  Requires native
                       ct.npy and label.npy to exist (convert_to_npy.py always
                       writes them).  ~5-13× slower than the pre-sized fast
                       path on NFS (warm cache); useful when you want to avoid
                       the quality loss from downsampling.
        crop_jitter  : Max voxel offset from organ centroid when use_crop=True.
                       Defaults to image_size[0] // 4.
        mask_downsample : How masks are resized to image_size at every resize call
                       site (crop paths + synth slow path). "nearest" point-samples
                       (default; thin structures can vanish under heavy downsampling);
                       "occupancy" area-pools to the foreground fraction and thresholds
                       at mask_occupancy_thr, preserving sub-voxel structures.
        mask_occupancy_thr : Foreground-fraction threshold for mask_downsample="occupancy".
                       ->0 keeps every touched voxel (dilates thin parts), 0.5 = majority.

    Scan cache
    ----------
    The first init call scans every subject's label.npy to record which of the
    117 classes are present.  The result is saved as a pickle file inside the
    dataset root and reused on all subsequent runs — including runs with
    different class subsets or splits.  The cache is keyed by a hash of the
    full set of subject directories, so it is automatically invalidated if
    subjects are added or removed.

    Bbox cache
    ----------
    When use_crop=True, a second cache stores per-(subject, class) organ
    centroids (integer voxel coordinates in the native label.npy space).
    Built once on first use; keyed by the same subject-list hash.
    """

    def __init__(
        self,
        root: str | Path,
        classes: list[str],
        image_size: Optional[tuple[int, int, int]] = (64, 64, 64),
        split: Optional[str] = None,
        meta_csv: Optional[str | Path] = None,
        context_size: int = 3,
        max_subjects: Optional[int] = None,
        aug_cfg=None,
        synth_method: Optional[str] = None,
        synth_unions: bool = False,
        p_synth: float = 0.5,
        class_balanced: bool = False,
        use_crop: bool = False,
        crop_jitter: Optional[int] = None,
        crop_spacing_mm: float = 1.5,
        mask_downsample: str = "nearest",
        mask_occupancy_thr: float = 0.5,
        random_coloring: bool = False,
        num_labels_per_sample: int = 1,
        n_synth_merge_min: int = 1,
        n_synth_merge_max: int = 1,
        eval_seed: Optional[int] = None,
        raw_ct: bool = False,
        modality: str = "ct",
        self_context: float = 0.0,
        self_context_intensity: bool = False,
        self_context_per_image: bool = False,
        self_context_synth: dict | None = None,
        defer_aug_to_gpu: bool = False,
    ):
        self.root = Path(root)
        self.classes = list(classes)
        self.image_size = image_size
        # raw_ct: read native ct_raw.npy (raw intensities) and normalise the CROP on the fly
        # instead of the pre-normalised ct.npy. Affects only native-resolution reads (use_crop
        # + synth crop + slow nii path); the pre-resized fast path keeps its normalised
        # ct_{size}.npy (derived + already lossy from downsampling). For CT the normalisation is
        # a global pointwise transform, so crop==whole; for MRI it uses per-volume stats from
        # ct_stats.json (whole-volume, so crops stay consistent). See src.totalseg_dataset.
        assert modality in ("ct", "mri"), modality
        self.raw_ct = bool(raw_ct)
        self.modality = modality
        # self_context: PROBABILITY (0..1; True->1.0) that an item's K contexts are replaced by
        # clones of the (augmented) target. 1.0 = always (the decoder/matching ceiling probe:
        # every query token has an identical support twin -> trivial matching, pure reconstruction,
        # leakage by design). self_context_{intensity,per_image} independently re-augment each clone
        # (per_image = geometric pose jitter via aug_cfg.per_image; intensity = appearance jitter via
        # aug_cfg.intensity) so target != context by a controlled amount — the pose/appearance-
        # invariance training levers, each toggleable for clean A/B isolation. 0.0 = cross-subject.
        self.self_context_p = float(self_context)
        self.self_context_intensity = bool(self_context_intensity)
        self.self_context_per_image = bool(self_context_per_image)
        self.defer_aug = bool(defer_aug_to_gpu)
        # self_context_synth: when self-context fires, with probability `p` replace the real
        # target label with a freshly-generated synthetic mask (a random rotated ellipsoid placed
        # on the body of the real CT) — a purely-geometric in-context task with no real anatomy.
        # {p, ellipse_min_mm, ellipse_max_mm, body_hu}. body_hu is the CT air/body threshold in HU
        # (mapped to normalised space via the dataset CT_MEAN/CT_STD). None/{} → disabled.
        sc_synth = dict(self_context_synth or {})
        self.self_context_synth_p = float(sc_synth.get("p", 0.0))
        # sources: which synthetic-label generators the synth branch samples from (uniformly)
        # each time it fires. "ellipse" = random rotated ellipsoid (make_ellipsoid_label);
        # "supervoxel" = a supervoxel group from this subject's label_synth_{method} volume,
        # cropped onto the target grid (see _supervoxel_label_on_grid), falling back to an
        # ellipsoid when the subject has no usable supervoxel. Default [ellipse] keeps the
        # prior behaviour. Requires data.synth_method set for the supervoxel source.
        self.sc_synth_sources = list(sc_synth.get("sources", ["ellipse"])) or ["ellipse"]
        assert all(s in ("ellipse", "supervoxel") for s in self.sc_synth_sources), \
            f"self_context.synth_masks.sources must be ellipse/supervoxel, got {self.sc_synth_sources}"
        self.sc_synth_min_mm = float(sc_synth.get("ellipse_min_mm", 1.0))
        self.sc_synth_max_mm = float(sc_synth.get("ellipse_max_mm", 50.0))
        # Air/body threshold in normalised CT units (body voxels sit above it).
        self.sc_synth_body_thresh = (float(sc_synth.get("body_hu", -400.0)) - CT_MEAN) / CT_STD
        # supervoxel source: merge randint(min,max) face-adjacent supervoxels into one label.
        _sv_src = dict(sc_synth.get("supervoxel", {}) or {})
        self.sc_sv_merge_min = int(_sv_src.get("n_merge_min", 1))
        self.sc_sv_merge_max = int(_sv_src.get("n_merge_max", 1))
        self._size_str = (
            f"{image_size[0]}x{image_size[1]}x{image_size[2]}"
            if image_size is not None else None
        )
        self.context_size = context_size
        # Convert omegaconf -> plain nested namespace ONCE (per-item aug reads many fields;
        # omegaconf __getattr__ is ~30% of __getitem__ in the hot path). See _to_ns.
        if aug_cfg is not None:
            aug_cfg = _to_ns(aug_cfg)
        self.aug_cfg = aug_cfg  # None → no augmentation
        self.synth_method = synth_method
        self.p_synth = p_synth
        self.class_balanced = class_balanced
        # Deterministic eval: when set, __getitem__ draws context shuffling + crop
        # jitter from a per-item RNG seeded by (eval_seed, idx) instead of the global
        # `random`, so results are reproducible regardless of worker count, iteration
        # order, or the torch-RNG state left by model construction. None → training
        # behaviour (global `random`, freely stochastic).
        self.eval_seed = eval_seed
        self._cur_rng = random   # per-item Random in __getitem__ when eval_seed is set
        self.use_crop = use_crop
        self.random_coloring = random_coloring
        self.num_labels_per_sample = num_labels_per_sample
        self.crop_jitter = crop_jitter if crop_jitter is not None else (
            image_size[0] // 4 if image_size is not None else 0
        )
        # Output mm/voxel of use_crop=True crops: crop covers T*crop_spacing_mm and is
        # resampled to T³. Default 1.5 (native CT). Set 2.0 to match CoLiPri's 2mm training.
        self.crop_spacing_mm = crop_spacing_mm
        # Per-__getitem__ crop-spacing override (variable-spacing training). Set from a
        # (idx, spacing) index by the spacing batch sampler; None → fixed crop_spacing_mm.
        # Instance state is safe: a worker processes one item at a time (cf. _cur_rng).
        self._cur_crop_spacing = None
        # Coarse->fine cascade eval: {(subject, class): native center} overriding the
        # GT-centroid target crop with a previous-spacing predicted center. Only the TARGET
        # load consults it (contexts stay GT-centred); a value of "volume_center" means the
        # coarse prediction was empty -> centre on the volume. Empty {} disables the override.
        # See experiments/3d/evaluate.py:evaluate_spacing_sweep(cascade=True).
        self._pred_centers: dict = {}
        # Crop geometry (starts, crop_sizes, out_sizes, pad_lo) of the most recent
        # _organ_crop_arrays call — stashed so __getitem__ can attach the TARGET crop's
        # geometry to the item (inverts a grid-space prediction back to native voxels).
        self._last_crop_geom = None
        # Mask downsampling mode (all resize call sites: crop paths + synth slow path):
        #   "nearest"   — point-sample one native voxel per output voxel (default; thin
        #                 structures can vanish under heavy downsampling, e.g. 4mm crops).
        #   "occupancy" — area-pool the binary mask to the foreground FRACTION per output
        #                 voxel, then keep voxels whose fraction >= mask_occupancy_thr. thr
        #                 ->0 preserves every touched voxel (dilates thin parts), 0.5 is a
        #                 majority vote. Guarantees a non-empty mask if the input had any fg.
        assert mask_downsample in ("nearest", "occupancy"), mask_downsample
        self.mask_downsample = mask_downsample
        self.mask_occupancy_thr = float(mask_occupancy_thr)
        self.hu_jitter = (
            getattr(aug_cfg.intensity, "hu_jitter", 0)
            if aug_cfg is not None and aug_cfg.enabled
            else 0
        )

        subjects = self._get_subjects(split, meta_csv, max_subjects)

        # Load (or build) the full subject→classes cache, then filter to this split
        subject_classes = self._load_or_build_cache()

        # Build label→subjects index for the requested classes and this split's subjects
        self.label_to_subjects: dict[str, list[str]] = {cls: [] for cls in self.classes}
        cls_set = set(self.classes)
        for subj in subjects:
            for cls in subject_classes.get(subj, frozenset()):
                if cls in cls_set:
                    self.label_to_subjects[cls].append(subj)

        # Flat sample list: one entry per (subject, class) pair
        self.samples: list[tuple[str, str]] = [
            (subj, cls)
            for cls in self.classes
            for subj in self.label_to_subjects[cls]
        ]

        # Classes that actually have at least one subject (used by class-balanced sampler)
        self.active_classes = [cls for cls in self.classes if self.label_to_subjects[cls]]

        counts = {cls: len(self.label_to_subjects[cls]) for cls in self.classes}
        print(f"TotalSegInContextDataset: {len(self.samples)} samples | "
              f"context_size={context_size} | class_balanced={class_balanced} | "
              f"hu_jitter={self.hu_jitter} | use_crop={use_crop} | "
              f"mask_downsample={self.mask_downsample}"
              f"{f'(thr={self.mask_occupancy_thr})' if self.mask_downsample == 'occupancy' else ''} | "
              f"class counts: {counts}", flush=True)

        # Bbox cache: organ centroids in native-res voxel space (needed for use_crop)
        if use_crop:
            self._bbox_cache = self._load_or_build_bbox_cache()
        else:
            self._bbox_cache = {}

        self.n_synth_merge_min = n_synth_merge_min
        self.n_synth_merge_max = n_synth_merge_max

        # Spacing cache: {subject → effective spacing (mm/voxel) at image_size}.
        # Loaded once from spacings.json at the dataset root; falls back to 1mm
        # isotropic for subjects not present (pre-existing data without spacing info).
        self._spacings = self._load_spacings()

        # Per-volume MRI normalisation stats (ct_stats.json), needed only for raw_ct MRI.
        self._ct_stats = (self._load_ct_stats()
                          if self.raw_ct and self.modality == "mri" else {})
        if self.raw_ct:
            print(f"raw_ct path ON (modality={self.modality}): native reads use ct_raw.npy "
                  f"+ on-the-fly normalisation", flush=True)

        # Synth path: build SV-ID cache for fast __getitem__ sampling. Supervoxel labels
        # feed both the old p_synth path (_get_synth_item) and the self_context synth_masks
        # "supervoxel" source, so the effective merge depth is the max the two request.
        _sv_source = "supervoxel" in self.sc_synth_sources
        _eff_merge_max = max(n_synth_merge_max, self.sc_sv_merge_max if _sv_source else 1)
        if synth_method is not None:
            # merge_max > 1 → always load base labels and merge on-the-fly
            suffix = "" if _eff_merge_max > 1 else ("_union" if synth_unions else "")
            self._synth_fname = f"label_synth_{synth_method}{suffix}.npy"
            self._synth_subjects, self._synth_sv_ids = \
                self._load_or_build_synth_cache(subjects)
            print(f"Synth path: method={synth_method} "
                  f"n_synth_merge=[{n_synth_merge_min},{n_synth_merge_max}] "
                  f"p_synth={p_synth} | sc_sources={self.sc_synth_sources} "
                  f"| {len(self._synth_subjects)} subjects", flush=True)
            if _eff_merge_max > 1:
                self._adj_cache = self._load_or_build_adj_cache()
            else:
                self._adj_cache = {}
        else:
            self._synth_subjects = []
            self._synth_sv_ids   = {}
            self._adj_cache      = {}

    # ------------------------------------------------------------------
    # Init helpers
    # ------------------------------------------------------------------

    def _cache_path(self, all_subjects: list[str]) -> Path:
        """Stable cache path keyed by the full set of subject dirs in root."""
        key = hashlib.sha256("|".join(all_subjects).encode()).hexdigest()[:12]
        return self.root / f".scan_cache_{key}.pkl"

    def _load_or_build_cache(self) -> dict[str, frozenset[str]]:
        """
        Return {subject_id: frozenset[class_name]} for every subject in root.

        Covers all 117 TotalSegmentator classes so the same cache is valid for
        any class subset.  Saved as a pickle next to the data; rebuilt only when
        the set of subject directories changes.
        """
        all_subjects = sorted(p.name for p in self.root.iterdir() if p.is_dir())
        cache_path = self._cache_path(all_subjects)

        if cache_path.exists():
            with open(cache_path, "rb") as f:
                cache = pickle.load(f)
            print(f"Loaded scan cache ({len(cache)} subjects) from {cache_path.name}",
                  flush=True)
            return cache

        print(f"Building scan cache for {len(all_subjects)} subjects "
              f"(saved to {cache_path.name})...", flush=True)
        cache: dict[str, frozenset[str]] = {}
        for subj in all_subjects:
            label_npy = self.root / subj / "label.npy"
            if not label_npy.exists():
                continue
            try:
                arr = np.load(label_npy, mmap_mode="r")
                present_indices = set(np.unique(arr))
            except (EOFError, ValueError, OSError):
                print(f"  Skipping corrupt file: {label_npy}", flush=True)
                continue
            cache[subj] = frozenset(
                _IDX_TO_CLASS[i] for i in present_indices if i in _IDX_TO_CLASS
            )

        with open(cache_path, "wb") as f:
            pickle.dump(cache, f)
        print(f"Scan cache saved ({len(cache)} subjects).", flush=True)
        return cache

    def _load_or_build_synth_cache(
        self, subjects: list[str]
    ) -> tuple[list[str], dict[str, np.ndarray]]:
        """
        Return (synth_subjects, {subject: array_of_sv_ids}).

        Scans which subjects in this split have the synth label file and records
        their unique SV indices, so __getitem__ never calls np.unique at runtime.
        Persisted as a pickle next to the data; rebuilt only when the subject
        set or filename changes.
        """
        synth_subs = [s for s in subjects
                      if (self.root / s / self._synth_fname).exists()]

        key = hashlib.sha256(
            (self._synth_fname + "|".join(synth_subs)).encode()
        ).hexdigest()[:12]
        cache_path = self.root / f".synth_sv_cache_{key}.pkl"

        if cache_path.exists():
            with open(cache_path, "rb") as f:
                sv_ids = pickle.load(f)
            print(f"Loaded synth SV cache ({len(sv_ids)} subjects) from {cache_path.name}",
                  flush=True)
            return synth_subs, sv_ids

        print(f"Building synth SV cache for {len(synth_subs)} subjects "
              f"({self._synth_fname})...", flush=True)
        sv_ids: dict[str, np.ndarray] = {}
        for subj in synth_subs:
            arr = np.load(self.root / subj / self._synth_fname, mmap_mode="r")
            ids = np.unique(arr)
            sv_ids[subj] = ids[ids > 0].copy()

        with open(cache_path, "wb") as f:
            pickle.dump(sv_ids, f)
        print(f"Synth SV cache saved ({len(sv_ids)} subjects).", flush=True)
        return synth_subs, sv_ids

    def _load_or_build_adj_cache(self) -> dict[str, dict[int, frozenset[int]]]:
        """
        Return {subject: {sv_id: frozenset[face-adjacent sv_ids]}} built from
        label_synth_{method}.npy (native resolution).  Built once, keyed by
        synth filename + subject hash.  Used for on-the-fly random SV merging.
        """
        synth_subs = [s for s in self._synth_subjects
                      if (self.root / s / self._synth_fname).exists()]
        key = hashlib.sha256(
            ("adj_" + self._synth_fname + "|".join(synth_subs)).encode()
        ).hexdigest()[:12]
        cache_path = self.root / f".adj_cache_{key}.pkl"

        if cache_path.exists():
            with open(cache_path, "rb") as f:
                cache = pickle.load(f)
            print(f"Loaded adj cache ({len(cache)} subjects) from {cache_path.name}",
                  flush=True)
            return cache

        n_workers = min(16, os.cpu_count() or 1)
        print(f"Building adj cache for {len(synth_subs)} subjects "
              f"({self._synth_fname}, {n_workers} workers)...", flush=True)
        cache: dict[str, dict[int, frozenset[int]]] = {}
        done = 0
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futs = {
                ex.submit(_adj_for_subject, self.root, subj, self._synth_fname): subj
                for subj in synth_subs
            }
            for fut in as_completed(futs):
                try:
                    subj, adj = fut.result()
                    if adj is not None:
                        cache[subj] = adj
                    else:
                        print(f"  Skipping adj for {subj}", flush=True)
                except Exception as e:
                    print(f"  Error in adj worker: {e}", flush=True)
                done += 1
                if done % 100 == 0:
                    print(f"  adj cache: {done}/{len(synth_subs)}", flush=True)

        with open(cache_path, "wb") as f:
            pickle.dump(cache, f)
        print(f"Adj cache saved ({len(cache)} subjects).", flush=True)
        return cache

    @staticmethod
    def _sample_merged_svs(
        sv_ids_list: list[int],
        adj: dict[int, frozenset[int]],
        n_min: int,
        n_max: int,
    ) -> list[int]:
        """Pick a seed SV then BFS-expand to randint(n_min, n_max) face-adjacent neighbors."""
        seed = random.choice(sv_ids_list)
        n_target = random.randint(n_min, n_max)
        if n_target == 1:
            return [seed]
        merged = [seed]
        merged_set = {seed}
        frontier = list(adj.get(seed, frozenset()))
        random.shuffle(frontier)
        while len(merged) < n_target and frontier:
            nxt = frontier.pop()
            if nxt in merged_set:
                continue
            merged.append(nxt)
            merged_set.add(nxt)
            frontier += [sv for sv in adj.get(nxt, frozenset()) if sv not in merged_set]
        return merged

    def _load_or_build_bbox_cache(self) -> dict[str, dict[str, tuple[int, int, int]]]:
        """
        Return {subject: {class_name: (d, h, w)}} with integer organ centroids
        in native label.npy voxel space.  Built once, keyed by subject-list hash.
        """
        all_subjects = sorted(p.name for p in self.root.iterdir() if p.is_dir())
        key = hashlib.sha256("|".join(all_subjects).encode()).hexdigest()[:12]
        cache_path = self.root / f".bbox_cache_{key}.pkl"

        if cache_path.exists():
            with open(cache_path, "rb") as f:
                cache = pickle.load(f)
            print(f"Loaded bbox cache ({len(cache)} subjects) from {cache_path.name}",
                  flush=True)
            return cache

        valid_subjects = [s for s in all_subjects
                          if (self.root / s / "label.npy").exists()]
        n_workers = min(16, os.cpu_count() or 1)
        print(f"Building bbox cache for {len(valid_subjects)} subjects "
              f"(saved to {cache_path.name}, {n_workers} workers)...", flush=True)
        cache: dict[str, dict[str, tuple[int, int, int]]] = {}
        done = 0
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futs = {
                ex.submit(_bbox_for_subject, self.root, subj): subj
                for subj in valid_subjects
            }
            for fut in as_completed(futs):
                try:
                    subj, result = fut.result()
                    if result is not None:
                        cache[subj] = result
                    else:
                        print(f"  Skipping {subj} (error in bbox worker)", flush=True)
                except Exception as e:
                    print(f"  Error in bbox worker: {e}", flush=True)
                done += 1
                if done % 100 == 0:
                    print(f"  bbox cache: {done}/{len(valid_subjects)}", flush=True)

        with open(cache_path, "wb") as f:
            pickle.dump(cache, f)
        print(f"BBox cache saved ({len(cache)} subjects).", flush=True)
        return cache

    def _load_spacings(self) -> dict[str, torch.Tensor]:
        """Load spacings.json and convert to effective mm/voxel at self.image_size.

        For the crop path (native resolution), native spacing is returned as-is.
        For the resized path, effective spacing = native_spacing * max(native_shape) / T.
        Returns {} if the file does not exist (spacing will default to 1mm isotropic).
        """
        path = self.root / "spacings.json"
        if not path.exists():
            return {}
        with open(path) as f:
            raw = json.load(f)

        result: dict[str, torch.Tensor] = {}
        T = self.image_size[0] if self.image_size is not None else None
        for subj, meta in raw.items():
            sp = torch.tensor(meta["spacing"], dtype=torch.float32)
            if T is not None and not self.use_crop:
                # Effective spacing: _iso_resize scales longest axis → T voxels
                max_native = max(meta["shape"])
                sp = sp * (max_native / T)
            elif T is not None and self.use_crop:
                # Effective spacing after crop + resize to T³ is always 1.5mm/voxel:
                # phys_ref = T * 1.5mm is resampled into T voxels, isotropic.
                sp = torch.full((3,), 1.5, dtype=torch.float32)
            result[subj] = sp
        return result

    def _load_ct_stats(self) -> dict[str, dict]:
        """Load per-volume MRI normalisation stats (ct_stats.json) written by
        convert_to_npy --store-raw --modality mri. Returns {} if absent."""
        path = self.root / "ct_stats.json"
        if not path.exists():
            return {}
        with open(path) as f:
            return json.load(f)

    def _normalize_native(self, subj: str, arr: np.ndarray) -> np.ndarray:
        """Normalise a raw native array (or a crop of it) to model input space.

        CT: global pointwise transform (crop == whole). MRI: per-volume stats from
        ct_stats.json (whole-volume, so every crop of a subject normalises identically)."""
        if self.modality == "mri":
            st = self._ct_stats.get(subj)
            if st is None:
                raise KeyError(
                    f"raw_ct MRI: no ct_stats.json entry for {subj!r} — run "
                    f"convert_to_npy.py --store-raw --modality mri to build it")
            return normalize_mri(arr, st)
        return normalize_ct(arr)

    def _load_native_ct_mmap(self, subj_dir: Path):
        """mmap the native CT array for the crop path: ct_raw.npy (raw) when raw_ct, else the
        pre-normalised ct.npy. Guards against feeding raw HU as if it were normalised."""
        if self.raw_ct:
            arr = np.load(subj_dir / "ct_raw.npy", mmap_mode="r")
            if self.modality == "ct":
                assert arr.dtype == np.int16, (
                    f"ct_raw.npy for {subj_dir.name} is {arr.dtype}, expected int16 raw HU")
            return arr
        return np.load(subj_dir / "ct.npy", mmap_mode="r")

    def _get_spacing(self, subj: str) -> torch.Tensor:
        """Return effective spacing (3,) for subject, defaulting to 1mm isotropic."""
        return self._spacings.get(subj, torch.ones(3, dtype=torch.float32))

    @property
    def _crop_mm(self) -> float:
        """Effective crop spacing for the current item: the per-item override when the
        spacing batch sampler set one, else the fixed crop_spacing_mm."""
        return self.crop_spacing_mm if self._cur_crop_spacing is None else self._cur_crop_spacing

    def _reported_spacing(self, subj: str) -> torch.Tensor:
        """Effective mm/voxel of the returned image tensor (for item['spacing']).

        Under use_crop the crop covers T*crop_spacing_mm and is resampled to T³, so
        the output is crop_spacing_mm/voxel isotropic — regardless of the native
        spacing used internally by _load_crop to size the crop. Otherwise the resized
        effective spacing from _get_spacing already describes the output tensor.
        """
        if self.use_crop:
            return torch.full((3,), self._crop_mm, dtype=torch.float32)
        return self._get_spacing(subj)

    def _synth_native_coord(self, subj, centroid, item_shape, crop_geom):
        """Anatomical coords.npy value (3,) at a synthetic label's centroid, or None if the
        subject has no coords.npy. `centroid` is an item-grid (d,h,w); `item_shape` its (D,H,W).
        Maps the item voxel back to the native coords grid — through the crop geometry when
        use_crop (crop_geom = [starts, crop_sizes, out_sizes, pad_lo]), else by shape ratio.
        O(1): mmaps coords.npy and reads a single voxel."""
        cpath = self.root / subj / "coords.npy"
        if not cpath.exists():
            return None
        coords = np.load(cpath, mmap_mode="r")            # (Dn, Hn, Wn, 3) native
        nat = coords.shape[:3]
        if crop_geom is not None:
            starts, crop_sizes, out_sizes, pad_lo = (crop_geom[i].tolist() for i in range(4))
            idx = []
            for a in range(3):
                q = centroid[a] - pad_lo[a]               # into the resampled-crop grid
                q = min(max(q, 0), out_sizes[a] - 1)      # clamp out of the zero-pad border
                nvox = starts[a] + int(round(q * crop_sizes[a] / max(out_sizes[a], 1)))
                idx.append(min(max(nvox, 0), nat[a] - 1))
        else:
            idx = [min(int(round(centroid[a] * nat[a] / item_shape[a])), nat[a] - 1)
                   for a in range(3)]
        return torch.from_numpy(np.asarray(coords[idx[0], idx[1], idx[2]], dtype=np.float32))

    def _supervoxel_label_on_grid(self, subj, crop_geom):
        """Synthetic target label built from `subj`'s supervoxel volume, placed on the SAME
        grid as the already-loaded target crop so it pairs with the real target image in
        self-context (the self_context.synth_masks "supervoxel" source). A supervoxel group
        present in the crop is picked (merging sc_sv_merge_[min,max] face-adjacent SVs) and
        rasterised as a binary label. crop_geom (4,3 long: starts, crop_sizes, out_sizes,
        pad_lo) is the target crop geometry (use_crop); None → the full pre-resized grid.
        Returns a (T,T,T) long tensor, or None when this subject has no usable supervoxel
        (caller falls back to an ellipsoid)."""
        if not self._synth_subjects or subj not in self._synth_sv_ids:
            return None
        sv_path = self.root / subj / self._synth_fname
        if not sv_path.exists():
            return None
        adj = self._adj_cache.get(subj, {})

        if crop_geom is not None:
            starts, crop_sizes, out_sizes, pad_lo = (crop_geom[i].tolist() for i in range(4))
            d0, h0, w0 = starts
            cs0, cs1, cs2 = crop_sizes
            sv_mm = np.load(sv_path, mmap_mode="r")
            crop_sv = np.ascontiguousarray(sv_mm[d0:d0 + cs0, h0:h0 + cs1, w0:w0 + cs2])
            ids = [int(i) for i in np.unique(crop_sv) if i != 0]
            if not ids:
                return None
            # merge over the full-volume adjacency, then keep only SVs present in the crop
            group = [g for g in self._sample_merged_svs(
                ids, adj, self.sc_sv_merge_min, self.sc_sv_merge_max) if g in ids]
            if not group:
                return None
            binary = np.isin(crop_sv, group).astype(np.uint8)
            lbl_small = self._resample_multiclass(binary, tuple(out_sizes), 1)
            return place_label(lbl_small, out_sizes, pad_lo, self.image_size[0]).long()

        # fast path: the pre-resized supervoxel grid matches the pre-resized target image
        sized = (self.root / subj / self._synth_fname.replace(".npy", f"_{self._size_str}.npy")
                 if self._size_str else None)
        if sized is None or not sized.exists():
            return None
        sv_vol = np.ascontiguousarray(np.load(sized, mmap_mode="r"))
        ids = [int(i) for i in np.unique(sv_vol) if i != 0]
        if not ids:
            return None
        group = [g for g in self._sample_merged_svs(
            ids, adj, self.sc_sv_merge_min, self.sc_sv_merge_max) if g in ids]
        if not group:
            return None
        return torch.from_numpy(np.isin(sv_vol, group).astype(np.int64))

    def _get_subjects(self, split, meta_csv, max_subjects) -> list[str]:
        all_subjects = sorted(p.name for p in self.root.iterdir() if p.is_dir())
        if split is not None:
            csv_path = Path(meta_csv) if meta_csv else self.root / "meta.csv"
            valid: set[str] = set()
            with open(csv_path, encoding="utf-8-sig") as f:
                for row in csv.DictReader(f, delimiter=";"):
                    if row["split"].strip() == split:
                        valid.add(row["image_id"].strip())
            all_subjects = [s for s in all_subjects if s in valid]
        if max_subjects is not None:
            all_subjects = all_subjects[:max_subjects]
        return all_subjects

    # ------------------------------------------------------------------
    # Dataset API
    # ------------------------------------------------------------------

    @staticmethod
    def _sample_palette(
        label:      torch.Tensor,
        ctx_masks:  list[torch.Tensor],
        num_labels: int,
    ) -> torch.Tensor:
        """SegGPT-style palette: maximally separated grid colors, shuffled per sample.

        Divides the RGB cube into a base³ grid (base = ceil(num_labels^(1/3))),
        takes the first num_labels non-black grid points, then randomly permutes
        their assignment to label IDs so the model cannot memorise color↔class.

        Returns (num_labels+1, 3) float32 in [0,1]: row 0 = black (background),
        row i = a grid color if label i is shared across all masks, else (0,0,0).
        """
        shared = set(label.unique().tolist())
        for m in ctx_masks:
            shared &= set(m.unique().tolist())
        shared.discard(0)
        shared = [int(lid) for lid in shared if 1 <= int(lid) <= num_labels]

        base   = math.ceil(num_labels ** (1 / 3))
        margin = 256 // base
        grid: list[list[float]] = []
        for i in range(base):
            for j in range(base):
                for k in range(base):
                    if i == 0 and j == 0 and k == 0:
                        continue
                    grid.append([i * margin / 255.0, j * margin / 255.0, k * margin / 255.0])
        grid = grid[:num_labels]
        random.shuffle(grid)

        palette = torch.zeros(num_labels + 1, 3)
        for rank, lid in enumerate(shared):
            if rank < len(grid):
                palette[lid] = torch.tensor(grid[rank], dtype=torch.float32)
        return palette

    def __len__(self) -> int:
        return len(self.samples)

    def _get_synth_item(self) -> dict:
        """
        Build one in-context item from a single supervoxel, duplicated K+1
        times with heavy independent augmentation applied to each copy so that
        target and context diverge as much as possible.

        Crop path  (use_crop=True): loads native-res ct.npy + synth label and
        crops a T³ patch centred on the picked supervoxels (same jitter logic as
        _load_crop).
        Fast path  (default): uses pre-resized ct_{size}.npy and
        label_synth_{method}_{size}.npy when available.
        """
        subj     = self._synth_subjects[torch.randint(len(self._synth_subjects), (1,)).item()]
        sv_ids   = self._synth_sv_ids[subj]
        subj_dir = self.root / subj

        # Build SV groups: each group merges 1..n_synth_merge adjacent SVs into one label
        sv_ids_list = sv_ids.tolist()
        n_pick      = min(self.num_labels_per_sample, len(sv_ids_list))
        adj         = self._adj_cache.get(subj, {})
        sv_groups:  list[list[int]] = []
        used:       set[int] = set()
        for _ in range(n_pick):
            available = [sv for sv in sv_ids_list if sv not in used]
            if not available:
                break
            group = self._sample_merged_svs(available, adj, self.n_synth_merge_min, self.n_synth_merge_max)
            sv_groups.append(group)
            used.update(group)

        if self.use_crop:
            # Native-res crop centred on union of all picked supervoxels
            T     = self.image_size[0]
            ct_mm = self._load_native_ct_mmap(subj_dir)
            sv_mm = np.load(subj_dir / self._synth_fname, mmap_mode="r")
            D, H, W = sv_mm.shape

            sv_union = np.zeros((D, H, W), dtype=bool)
            for group in sv_groups:
                for sv_id in group:
                    sv_union |= (sv_mm == sv_id)
            n = int(sv_union.sum())
            if n > 0:
                d_g = np.arange(D, dtype=np.float32)[:, None, None]
                h_g = np.arange(H, dtype=np.float32)[None, :, None]
                w_g = np.arange(W, dtype=np.float32)[None, None, :]
                cd  = int((d_g * sv_union).sum() / n)
                ch  = int((h_g * sv_union).sum() / n)
                cw  = int((w_g * sv_union).sum() / n)
            else:
                cd, ch, cw = D // 2, H // 2, W // 2

            j = self.crop_jitter
            starts = []
            for c, s in zip((cd, ch, cw), (D, H, W)):
                ideal = c - T // 2
                lo    = max(0, ideal - j)
                hi    = max(lo, min(max(0, s - T), ideal + j))
                starts.append(random.randint(lo, hi))
            d0, h0, w0 = starts

            crop_ct = ct_mm[d0:d0+T, h0:h0+T, w0:w0+T]
            crop_sv = sv_mm[d0:d0+T, h0:h0+T, w0:w0+T]
            s       = crop_ct.shape
            if self.raw_ct:
                crop_ct = self._normalize_native(subj, np.ascontiguousarray(crop_ct))

            img_arr = np.zeros((T, T, T), dtype=np.float32)
            msk_arr = np.zeros((T, T, T), dtype=np.uint8)
            img_arr[:s[0], :s[1], :s[2]] = crop_ct.astype(np.float32)
            for label_id, group in enumerate(sv_groups, 1):
                for sv_id in group:
                    msk_arr[:s[0], :s[1], :s[2]][crop_sv[:s[0], :s[1], :s[2]] == sv_id] = label_id

            image_t = torch.from_numpy(img_arr).unsqueeze(0)  # (1, T, T, T)
            mask_t  = torch.from_numpy(msk_arr).long()        # (T, T, T)
        else:
            # CT — fast path: pre-resized; slow path: native npy/nii.gz → resize
            ct_pre = subj_dir / f"ct_{self._size_str}.npy" if self._size_str else None
            if ct_pre is not None and ct_pre.exists():
                image_t = torch.from_numpy(
                    np.load(ct_pre, mmap_mode="r").astype(np.float32)
                ).unsqueeze(0)                                          # (1, D, H, W)
            else:
                ct_npy = subj_dir / "ct.npy"
                image = (np.load(ct_npy, mmap_mode="r").astype(np.float32)
                         if ct_npy.exists() else _load_ct(subj_dir / "ct.nii.gz"))
                image_t = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)
                if self.image_size is not None:
                    image_t, _ = _resize_volume(image_t, image_t, self.image_size)
                image_t = image_t.squeeze(0)                            # (1, D, H, W)

            # Synth label: assign each picked supervoxel a unique integer ID 1..n_pick
            sized_synth = (
                subj_dir / self._synth_fname.replace(".npy", f"_{self._size_str}.npy")
                if self._size_str else None
            )
            if sized_synth is not None and sized_synth.exists():
                sv_vol = np.load(sized_synth, mmap_mode="r")
                mask   = np.zeros_like(sv_vol, dtype=np.uint8)
                for label_id, group in enumerate(sv_groups, 1):
                    for sv_id in group:
                        mask[sv_vol == sv_id] = label_id
                mask_t = torch.from_numpy(mask).long()                  # (D, H, W)
            else:
                sv_vol  = np.load(subj_dir / self._synth_fname, mmap_mode="r")
                mask    = np.zeros_like(sv_vol, dtype=np.uint8)
                for label_id, group in enumerate(sv_groups, 1):
                    for sv_id in group:
                        mask[sv_vol == sv_id] = label_id
                if self.image_size is not None:
                    T = self.image_size[0]
                    new = _iso_size(mask.shape, T)                      # aspect-preserving fit
                    mask_small = self._resample_multiclass(mask, tuple(new), len(sv_groups))
                    mask_out = torch.zeros(T, T, T, dtype=torch.long)
                    pads = [(T - s) // 2 for s in new]
                    sl = tuple(slice(p, p + s) for p, s in zip(pads, new))
                    mask_out[sl] = mask_small
                    mask_t = mask_out
                else:
                    mask_t = torch.from_numpy(mask).long()             # (D, H, W)

        # K+1 independent copies, each separately augmented
        if self.aug_cfg is not None and self.aug_cfg.enabled and not self.defer_aug:
            items = [
                apply_synth_aug(image_t.clone(), mask_t.clone(), self.aug_cfg.synth)
                for _ in range(self.context_size + 1)
            ]
        else:                                   # defer: emit K+1 RAW clones
            items = [(image_t.clone(), mask_t.clone()) for _ in range(self.context_size + 1)]

        image_out, label_out = items[0]
        ctx_masks = [it[1] for it in items[1:]]

        item = {
            "image":       image_out,
            "label":       label_out,                                  # (D, H, W) int64
            "context_in":  torch.stack([it[0] for it in items[1:]]),  # (K, 1, D, H, W)
            "context_out": torch.stack(ctx_masks),                     # (K, D, H, W) int64
            "subject":     subj,
            # synth contexts are K aug-copies of the SAME subject's supervoxel -> same case id
            "context_subjects": [subj for _ in range(self.context_size)],
            "label_name":  f"sv_{sv_groups[0][0]}",
            "spacing":     self._reported_spacing(subj),               # (3,) mm/voxel of output tensor
            "aug_mode":    torch.tensor(1, dtype=torch.long),          # synth
        }
        if self.random_coloring:
            item["label_palette"] = self._sample_palette(
                label_out, ctx_masks, self.num_labels_per_sample
            )
        return item

    def __getitem__(self, idx) -> dict:
        # The spacing batch sampler indexes with (idx, spacing) so every item in a batch
        # crops (and reports) the same physical spacing; a plain int → fixed crop_spacing_mm.
        if isinstance(idx, (tuple, list)):
            idx, self._cur_crop_spacing = int(idx[0]), float(idx[1])
        else:
            self._cur_crop_spacing = None
        # Deterministic eval draws every per-item random choice (context shuffle, crop
        # jitter) from a Random seeded by (eval_seed, idx); training keeps the global
        # `random` module. `_load`/`_load_crop` read self._cur_rng, so this covers the
        # crop jitter too. hash() over a tuple of ints is stable across processes.
        self._cur_rng = (random.Random(hash((self.eval_seed, idx)))
                         if self.eval_seed is not None else random)
        if self._synth_subjects and random.random() < self.p_synth:
            return self._get_synth_item()

        # --- Subject and class selection ------------------------------------
        tgt_crop_geom = None                # set by the single-label crop path (cascade eval)
        if self.num_labels_per_sample > 1:
            # Multi-label: pick a primary class (balanced), then add up to
            # num_labels_per_sample-1 extra classes present in the same subject.
            if self.class_balanced:
                primary_cls = random.choice(self.active_classes)
                subj = random.choice(self.label_to_subjects[primary_cls])
            else:
                subj, primary_cls = self.samples[idx]

            subj_classes = [c for c in self.active_classes
                            if subj in self.label_to_subjects[c]]
            extra = [c for c in subj_classes if c != primary_cls]
            self._cur_rng.shuffle(extra)
            selected = [primary_cls] + extra[:self.num_labels_per_sample - 1]
            label_name = primary_cls

            image_t, label_t = self._load_multi(subj, selected)
            # Context pool: subjects that share at least the primary class
            candidates = [s for s in self.label_to_subjects[primary_cls] if s != subj]
            load_ctx = lambda s: self._load_multi(s, selected)
        else:
            if self.class_balanced:
                cls  = random.choice(self.active_classes)
                subj = random.choice(self.label_to_subjects[cls])
            else:
                subj, cls = self.samples[idx]
            label_name = cls

            # Cascade eval: crop the TARGET on a previous-spacing predicted centre when one
            # was injected for this (subject, class); contexts always stay GT-centred.
            pred_center = self._pred_centers.get((subj, cls)) if self._pred_centers else None
            image_t, label_t = self._load(subj, cls, pred_center=pred_center)
            tgt_crop_geom = self._last_crop_geom            # geometry of the target crop
            candidates = [s for s in self.label_to_subjects[cls] if s != subj]
            load_ctx = lambda s: self._load(s, cls)

        # --- Context sampling ----------------------------------------------
        # Draw candidates in uniformly-random order but lazily: shuffling the whole
        # pool (often ~all subjects with the class) just to take context_size of them
        # was the top per-item RNG cost. _lazy_shuffle does O(consumed) work while still
        # able to walk every candidate if context loads fail. (Distribution identical to
        # shuffle-then-take; the exact per-seed picks differ from the old full-shuffle.)
        context_in:  list[torch.Tensor] = []
        context_out: list[torch.Tensor] = []
        ctx_subjects: list[str] = []          # case id each context came from (self-context detect)
        for ctx_subj in _lazy_shuffle(self._cur_rng, candidates):
            if len(context_in) >= self.context_size:
                break
            try:
                ctx_img, ctx_lbl = load_ctx(ctx_subj)
                context_in.append(ctx_img)
                context_out.append(ctx_lbl)
                ctx_subjects.append(ctx_subj)
            except Exception:
                continue

        # Pad by resampling if not enough candidates; fall back to target if empty
        if len(context_in) == 0:
            warnings.warn(
                "TotalSegInContextDataset: no context candidates found; "
                "falling back to target self-context (metrics for this sample "
                "are leakage-inflated).",
                stacklevel=2,
            )
            context_in.append(image_t.clone())
            context_out.append(label_t.clone())
            ctx_subjects.append(subj)
        while len(context_in) < self.context_size:
            i = self._cur_rng.randrange(len(context_in))
            context_in.append(context_in[i].clone())
            context_out.append(context_out[i].clone())
            ctx_subjects.append(ctx_subjects[i])

        # --- Augmentation + coloring (shared by both paths) ----------------
        if self.aug_cfg is not None and self.aug_cfg.enabled and len(context_in) > 0 and not self.defer_aug:
            all_images = torch.cat([image_t.unsqueeze(0), torch.stack(context_in)],  dim=0)
            all_masks  = torch.cat([label_t.unsqueeze(0), torch.stack(context_out)], dim=0)
            all_images, all_masks = apply_task_aug(all_images, all_masks, self.aug_cfg.task)
            for i in range(all_images.shape[0]):
                all_images[i] = apply_intensity_aug(all_images[i], self.aug_cfg.intensity)
            image_t     = all_images[0]
            label_t     = all_masks[0]
            context_in  = list(all_images[1:])
            context_out = list(all_masks[1:])

        # self_context: with prob self_context_p, replace the K contexts with clones of the FINAL
        # (post-aug) target. Bit-identical -> trivial matching (ceiling probe). The two toggles
        # re-augment each clone INDEPENDENTLY so the context differs from the target by a controlled
        # amount, each isolatable: self_context_per_image = geometric pose jitter (aug_cfg.per_image;
        # NB per_image not task — task shares one transform across all K+1 volumes and must NOT be
        # applied per-image); self_context_intensity = appearance jitter (aug_cfg.intensity). Both
        # off -> exact clone. No-op when aug is disabled (eval).
        self._sc_fired = (self.self_context_p > 0
                          and self._cur_rng.random() < self.self_context_p)
        synth_coord = synth_radii = None
        if self._sc_fired:
            # self_context_synth: replace the real target label with a purely synthetic mask (a
            # random rotated ellipsoid on the real CT's body) before cloning — a geometric
            # in-context task with no real anatomy. The image is untouched; only the label is
            # synthetic. coords.npy records the placed object's anatomical position (synth_coord).
            if (self.self_context_synth_p > 0
                    and self._cur_rng.random() < self.self_context_synth_p):
                # Pick a synth source (uniform); the supervoxel source falls back to an
                # ellipsoid when this subject has no usable supervoxel in the crop.
                source = self._cur_rng.choice(self.sc_synth_sources)
                sv_label = (self._supervoxel_label_on_grid(subj, tgt_crop_geom)
                            if source == "supervoxel" else None)
                label_name = "synth"                             # generic (shape logged in detail)
                if sv_label is not None:
                    label_t = sv_label                           # real image ↔ plausible blob label
                    fg = torch.nonzero(label_t, as_tuple=False)
                    if len(fg):
                        ctr = tuple(int(c) for c in fg.float().mean(0).round().tolist())
                        synth_coord = self._synth_native_coord(
                            subj, ctr, label_t.shape, tgt_crop_geom)
                else:
                    label_t, centroid, radii_mm = make_ellipsoid_label(
                        image_t, self._reported_spacing(subj), self._cur_rng,
                        min_mm=self.sc_synth_min_mm, max_mm=self.sc_synth_max_mm,
                        body_thresh=self.sc_synth_body_thresh)
                    label_t = label_t.long()
                    synth_radii = torch.from_numpy(radii_mm)      # (3,) generative size (mm)
                    synth_coord = self._synth_native_coord(
                        subj, centroid, label_t.shape, tgt_crop_geom)
            context_in  = [image_t.clone() for _ in range(self.context_size)]
            context_out = [label_t.clone() for _ in range(self.context_size)]
            ctx_subjects = [subj for _ in range(self.context_size)]   # self-context: ctx case == target
            do_augs = self.self_context_intensity or self.self_context_per_image
            if do_augs and self.aug_cfg is not None and self.aug_cfg.enabled and not self.defer_aug:
                pi_cfg = getattr(self.aug_cfg, "per_image", None)
                aug_in, aug_out = [], []
                for ci, cm in zip(context_in, context_out):
                    if self.self_context_per_image and pi_cfg is not None:
                        ci, cm = apply_per_image_aug(ci, cm, pi_cfg)
                    if self.self_context_intensity:
                        ci = apply_intensity_aug(ci, self.aug_cfg.intensity)
                    aug_in.append(ci)
                    aug_out.append(cm)
                context_in, context_out = aug_in, aug_out

        item = {
            "image":       image_t,
            "label":       label_t,                   # (D, H, W) int64 always
            "context_in":  torch.stack(context_in),   # (K, 1, D, H, W)
            "context_out": torch.stack(context_out),  # (K, D, H, W) int64 always
            "subject":     subj,
            "context_subjects": ctx_subjects,          # list[str] len K: per-context case id
            "label_name":  label_name,
            "spacing":     self._reported_spacing(subj),   # (3,) mm/voxel of output tensor
            "aug_mode":    torch.tensor(
                2 if (self.self_context_p > 0 and self._sc_fired) else 0, dtype=torch.long),
        }
        if tgt_crop_geom is not None:
            item["crop_geom"] = tgt_crop_geom              # (4,3): starts, crop_sizes, out_sizes, pad_lo
        if synth_radii is not None:
            item["synth_radii_mm"] = synth_radii           # (3,) sampled ellipsoid radii (mm)
        if synth_coord is not None:
            item["synth_coord"] = synth_coord              # (3,) anatomical coords of the synth object
        if self.random_coloring and len(context_out) > 0:
            item["label_palette"] = self._sample_palette(
                label_t, context_out, self.num_labels_per_sample
            )  # (num_labels+1, 3) float32
        return item

    # ------------------------------------------------------------------
    # Loading helpers
    # ------------------------------------------------------------------

    def _organ_crop_arrays(
        self,
        subj_dir,
        label_mm: np.ndarray,
        center: tuple[int, int, int],
        sp: list[float],
    ) -> tuple[np.ndarray, np.ndarray, list[int], list[int]]:
        """See module-level organ_crop_arrays. Adds the disk CT load, shape assert,
        raw_ct crop normalisation, and records self._last_crop_geom."""
        ct_mm = self._load_native_ct_mmap(subj_dir)
        assert ct_mm.shape == label_mm.shape, (
            f"{subj_dir.name}: ct.npy {ct_mm.shape} != label.npy {label_mm.shape} — "
            f"conversion must resample labels onto the CT grid (see convert_to_npy chemotox)")
        crop_ct, crop_lbl, out_sizes, pad_lo, geom = organ_crop_arrays(
            ct_mm, label_mm, center, sp, image_size=self.image_size,
            crop_mm=self._crop_mm, jitter=self.crop_jitter, rng=self._cur_rng)
        if self.raw_ct:
            crop_ct = self._normalize_native(subj_dir.name, np.ascontiguousarray(crop_ct))
        self._last_crop_geom = geom
        return crop_ct, crop_lbl, out_sizes, pad_lo

    def _place_image(self, crop_ct: np.ndarray, out_sizes, pad_lo) -> torch.Tensor:
        return place_image(crop_ct, out_sizes, pad_lo, self.image_size[0])

    def _place_label(self, label_small: torch.Tensor, out_sizes, pad_lo) -> torch.Tensor:
        return place_label(label_small, out_sizes, pad_lo, self.image_size[0])

    def _resample_binary(self, bin_np: np.ndarray, size) -> torch.Tensor:
        return resample_binary(bin_np, size, mode=self.mask_downsample,
                               occ_thr=self.mask_occupancy_thr)

    def _resample_multiclass(self, label_np: np.ndarray, size, n_classes: int) -> torch.Tensor:
        """Resize an integer multi-label map (values 0..n_classes) to `size` (long tensor).

        "occupancy" area-pools each foreground class to its fraction and assigns, per output
        voxel, the argmax class among those clearing mask_occupancy_thr (so small classes
        survive and don't lose ties to larger neighbours); "nearest" point-samples.
        """
        if self.mask_downsample == "occupancy" and n_classes >= 1:
            out = torch.zeros(tuple(size), dtype=torch.long)
            best = torch.zeros(tuple(size))
            for i in range(1, n_classes + 1):
                bi = torch.from_numpy(np.ascontiguousarray(label_np == i, dtype=np.float32))
                frac = F.interpolate(bi[None, None], size=size, mode="area")[0, 0]
                take = (frac >= self.mask_occupancy_thr) & (frac > best)
                out[take] = i
                best = torch.maximum(best, frac)
            return out
        t = torch.from_numpy(np.ascontiguousarray(label_np, dtype=np.float32))[None, None]
        return F.interpolate(t, size=size, mode="nearest")[0, 0].long()

    def _load_crop(self, subj: str, cls: str, pred_center=None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Load native ct.npy + label.npy and return an organ-centred crop resized to T³.

        Physical extent is fixed at T*crop_spacing_mm, so after resampling to T³ the
        output is crop_spacing_mm/voxel isotropic for every subject.  See
        _organ_crop_arrays for how thin-FOV axes are padded (not stretched).

        `pred_center` overrides the crop centre (cascade eval): a native (d,h,w) tuple,
        "volume_center" for an empty coarse prediction, or None for the GT centroid.
        """
        subj_dir = self.root / subj
        T = self.image_size[0]

        label_mm = np.load(subj_dir / "label.npy", mmap_mode="r")
        D, H, W = label_mm.shape

        if pred_center == "volume_center":
            center = (D // 2, H // 2, W // 2)
        elif pred_center is not None:
            center = tuple(int(c) for c in pred_center)
        else:
            center = self._bbox_cache.get(subj, {}).get(cls)
            center = center if center is not None else (D // 2, H // 2, W // 2)

        sp = self._get_spacing(subj).tolist()   # native mm/voxel (3,)
        crop_ct, crop_lbl, out_sizes, pad_lo = self._organ_crop_arrays(
            subj_dir, label_mm, center, sp)

        # Resample only the real slice to its extent inside T³ (trilinear image / occupancy
        # or nearest label), then centre-pad to T³ — never resample a padded T*cs/sp cube.
        image_t = self._place_image(crop_ct, out_sizes, pad_lo)

        orig_idx = _ALL_CLASSES_IDX.get(cls)
        if orig_idx is not None:
            label_t = self._place_label(
                self._resample_binary(crop_lbl == orig_idx, tuple(out_sizes)),
                out_sizes, pad_lo)
        else:
            label_t = torch.zeros(T, T, T, dtype=torch.long)

        return image_t, label_t

    def _load_crop_multi(self, subj: str, classes: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """Like _load_crop but assigns label IDs 1…L for each class in one pass."""
        subj_dir = self.root / subj

        label_mm = np.load(subj_dir / "label.npy", mmap_mode="r")
        D, H, W = label_mm.shape

        center = None
        for cls in classes:
            center = self._bbox_cache.get(subj, {}).get(cls)
            if center is not None:
                break
        center = center if center is not None else (D // 2, H // 2, W // 2)

        sp = self._get_spacing(subj).tolist()
        crop_ct, crop_lbl, out_sizes, pad_lo = self._organ_crop_arrays(
            subj_dir, label_mm, center, sp)

        image_t = self._place_image(crop_ct, out_sizes, pad_lo)

        label_np = np.zeros(crop_lbl.shape, dtype=np.uint8)
        for i, cls in enumerate(classes, 1):
            orig_idx = _ALL_CLASSES_IDX.get(cls)
            if orig_idx is not None:
                label_np[crop_lbl == orig_idx] = i
        label_t = self._place_label(
            self._resample_multiclass(label_np, tuple(out_sizes), len(classes)),
            out_sizes, pad_lo)

        return image_t, label_t

    def _load_multi(self, subj: str, classes: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """Load image + multi-class label in a single pass (IDs 1…L per class).

        Classes absent from a subject simply contribute no voxels; _apply_coloring
        will then exclude them from the shared palette automatically.
        """
        if self.use_crop:
            return self._load_crop_multi(subj, classes)

        subj_dir = self.root / subj
        if self._size_str is not None:
            ct_pre    = subj_dir / f"ct_{self._size_str}.npy"
            label_pre = subj_dir / f"label_{self._size_str}.npy"
            if ct_pre.exists() and label_pre.exists():
                image = np.load(ct_pre,    mmap_mode="r").astype(np.float32)
                full  = np.load(label_pre, mmap_mode="r")
                label = np.zeros(full.shape, dtype=np.uint8)
                for i, cls in enumerate(classes, 1):
                    orig_idx = _ALL_CLASSES_IDX.get(cls)
                    if orig_idx is not None:
                        label[full == orig_idx] = i
                return torch.from_numpy(image).unsqueeze(0), torch.from_numpy(label).long()

        # Slow path: merge single-class loads
        image_t = label_t = None
        for i, cls in enumerate(classes, 1):
            img, lbl = self._load(subj, cls)
            if image_t is None:
                image_t = img
                label_t = torch.zeros_like(lbl)
            label_t[lbl > 0] = i
        return image_t, label_t

    def _load(self, subj: str, cls: str, pred_center=None) -> tuple[torch.Tensor, torch.Tensor]:
        """Load one (image, binary_mask) pair for a single class.

        Crop path  (use_crop=True): load native ct.npy/label.npy and extract an
        organ-centred random crop — no interpolation, native resolution detail.
        Fast path  (default): use pre-resized ct_{size}.npy + label_{size}.npy.
        Slow path  (fallback): native .nii.gz → resize on the fly.

        `pred_center` (crop path only) overrides the GT-centroid crop centre for a
        cascade eval target — a native (d,h,w) tuple, or "volume_center" for an empty
        coarse prediction. None keeps the GT centroid (see _load_crop).
        """
        subj_dir = self.root / subj

        if self.use_crop:
            return self._load_crop(subj, cls, pred_center=pred_center)

        if self._size_str is not None:
            ct_pre    = subj_dir / f"ct_{self._size_str}.npy"
            label_pre = subj_dir / f"label_{self._size_str}.npy"
            if ct_pre.exists() and label_pre.exists():
                image = np.load(ct_pre,    mmap_mode="r").astype(np.float32)
                full  = np.load(label_pre, mmap_mode="r")
                label = np.zeros(full.shape, dtype=np.uint8)
                orig_idx = _ALL_CLASSES_IDX.get(cls)
                if orig_idx is not None:
                    label[full == orig_idx] = 1
                return (
                    torch.from_numpy(image).unsqueeze(0),  # (1, D, H, W)
                    torch.from_numpy(label).long(),        # (D, H, W)
                )

        # Slow path: native resolution → resize on the fly
        image   = _load_ct(subj_dir / "ct.nii.gz", jitter=self.hu_jitter)
        label   = _build_label_volume(subj_dir / "segmentations", [cls])
        image_t = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)
        label_t = torch.from_numpy(label.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        if self.image_size is not None:
            image_t, label_t = _resize_volume(image_t, label_t, self.image_size)
        return image_t.squeeze(0), label_t.squeeze(0).squeeze(0).long()


# ---------------------------------------------------------------------------
# Collate function
# ---------------------------------------------------------------------------

def incontext_collate_fn(batch: list[dict]) -> dict:
    """Stack a list of dataset items into a batch dict."""
    out = {
        "image":       torch.stack([b["image"]       for b in batch]),  # (B, 1, D, H, W)
        "label":       torch.stack([b["label"]       for b in batch]),  # (B, D, H, W)
        "context_in":  torch.stack([b["context_in"]  for b in batch]),  # (B, K, 1, D, H, W)
        "context_out": torch.stack([b["context_out"] for b in batch]),  # (B, K, D, H, W)
        "spacing":     torch.stack([b["spacing"]     for b in batch]),  # (B, 3) mm/voxel
        "subjects":    [b["subject"]    for b in batch],
        "label_names": [b["label_name"] for b in batch],
    }
    if "context_subjects" in batch[0]:
        out["context_subjects"] = [b["context_subjects"] for b in batch]  # (B) list[list[str]]
    if "label_palette" in batch[0]:
        out["label_palette"] = torch.stack([b["label_palette"] for b in batch])  # (B, L+1, 3)
    if "meta" in batch[0]:
        out["meta"] = [b["meta"] for b in batch]  # per-sample provenance (sample-table detail)
    if "crop_geom" in batch[0]:
        out["crop_geom"] = torch.stack([b["crop_geom"] for b in batch])  # (B, 4, 3) cascade inversion
    if "aug_mode" in batch[0]:
        out["aug_mode"] = torch.stack([b["aug_mode"] for b in batch])  # (B,) int64
    # Per-item NaN-pad so mixed synth+real batches still log radii/coords for their synth items
    # (an all()-gate dropped both keys whenever a single real sample shared the batch).
    if any("synth_radii_mm" in b for b in batch):
        nan3 = torch.full((3,), float("nan"))
        out["synth_radii_mm"] = torch.stack(  # (B, 3) mm; NaN rows are real (non-synth) samples
            [b["synth_radii_mm"].float() if "synth_radii_mm" in b else nan3 for b in batch])
    if any("synth_coord" in b for b in batch):
        nan3 = torch.full((3,), float("nan"))
        out["synth_coord"] = torch.stack(  # (B, 3); NaN when the synth object's subject lacks coords.npy
            [b["synth_coord"].float() if "synth_coord" in b else nan3 for b in batch])
    return out


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def get_incontext_loader(
    root: str,
    classes: list[str],
    image_size: tuple[int, int, int] = (64, 64, 64),
    split: Optional[str] = None,
    context_size: int = 3,
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 4,
    max_subjects: Optional[int] = None,
    aug_cfg=None,
    synth_method: Optional[str] = None,
    synth_unions: bool = False,
    p_synth: float = 0.5,
    class_balanced: bool = False,
    use_crop: bool = False,
    crop_jitter: Optional[int] = None,
    mask_downsample: str = "nearest",
    mask_occupancy_thr: float = 0.5,
    random_coloring: bool = False,
    num_labels_per_sample: int = 1,
    n_synth_merge_min: int = 1,
    n_synth_merge_max: int = 1,
    raw_ct: bool = False,
    modality: str = "ct",
) -> DataLoader:
    ds = TotalSegInContextDataset(
        root=root,
        classes=classes,
        image_size=image_size,
        split=split,
        context_size=context_size,
        max_subjects=max_subjects,
        aug_cfg=aug_cfg,
        synth_method=synth_method,
        synth_unions=synth_unions,
        p_synth=p_synth,
        class_balanced=class_balanced,
        use_crop=use_crop,
        crop_jitter=crop_jitter,
        mask_downsample=mask_downsample,
        mask_occupancy_thr=mask_occupancy_thr,
        random_coloring=random_coloring,
        num_labels_per_sample=num_labels_per_sample,
        n_synth_merge_min=n_synth_merge_min,
        n_synth_merge_max=n_synth_merge_max,
        raw_ct=raw_ct,
        modality=modality,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=incontext_collate_fn,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
    )
