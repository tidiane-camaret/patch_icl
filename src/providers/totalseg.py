"""TotalSegmentator volume provider for the in-context dataloader v2.

Single raw_ct organ-crop load path. `crop_and_place` is the one place crop
geometry (physical extent -> crop sizes -> resample -> centre-pad) is computed,
reusing the pure helpers extracted in the v1 module.
"""
import csv
import hashlib
import json
import os
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

from src.incontext_dataset_v2 import LoadRequest, LoadResult
from src.totalseg_dataloader_incontext import (
    organ_crop_arrays, place_image, place_label, resample_binary,
    _bbox_for_subject, _IDX_TO_CLASS,
)
from src.totalseg_dataset import (_ALL_CLASSES_IDX, normalize_ct, normalize_mri,
                                  resolve_ct_norm)


def crop_and_place(image_np, label_np, class_idx, center, T, *,
                   crop_spacing_mm, native_spacing, jitter, rng,
                   mask_downsample, occ_thr, normalize_fn=None, antialias=False):
    """Organ-centred crop of physical extent T*crop_spacing_mm around `center`,
    resampled to T^3 and centre-padded. Returns (image (1,T,T,T) f32, label
    (T,T,T) i64 binary for class_idx, crop_geom (4,3) i64).

    `normalize_fn`, when given, maps the cropped raw image slice to model input
    space BEFORE placement (so the air-pad value matches the normalized min).
    `antialias` area-prefilters downsampled image axes (see place_image) — needed
    when the native grid is finer than the crop pitch, e.g. anisotropic sources."""
    crop_ct, crop_lbl, out_sizes, pad_lo, geom = organ_crop_arrays(
        image_np, label_np, center, list(native_spacing),
        image_size=(T, T, T), crop_mm=crop_spacing_mm, jitter=jitter, rng=rng)
    crop_ct = np.ascontiguousarray(crop_ct)
    if normalize_fn is not None:
        crop_ct = normalize_fn(crop_ct)
    image_t = place_image(crop_ct, out_sizes, pad_lo, T, antialias=antialias)
    lbl_small = resample_binary(crop_lbl == class_idx, tuple(out_sizes),
                                mode=mask_downsample, occ_thr=occ_thr)
    # dtype preserved: long for occupancy/nearest, float32 (partial-volume fraction) for "soft"
    label_t = place_label(lbl_small, out_sizes, pad_lo, T)
    return image_t, label_t, geom


def crop_and_place_cached(img_cache_np, label_np, class_idx, center, T, *,
                          crop_spacing_mm, native_spacing, cache_spacing_mm, jitter, rng,
                          mask_downsample, occ_thr, normalize_fn=None):
    """Same output as `crop_and_place`, but the IMAGE is cropped from a pre-resampled
    `cache_spacing_mm`-pitch volume (`img_cache_np`) instead of the full-res native CT.

    The MASK is still cropped + occupancy-resampled from the full-res native `label_np`,
    so label fidelity (occupancy@thr, per-class) is byte-identical to the native path. Only
    the image path is accelerated: crop geometry is computed once on the native label grid,
    then the same physical box is mapped into the cache grid and resampled to out_sizes.

    Requires cache_spacing_mm <= out pitch (== crop_spacing_mm here) so the image is
    downsampled, not upsampled — guaranteed by the provider only using a cache whose pitch
    equals crop_spacing_mm. See docs/logs.md (6 mm image cache)."""
    # Geometry from the native label grid (drives out grid + mask). rng consumed once here.
    _crop_ct, crop_lbl, out_sizes, pad_lo, geom = organ_crop_arrays(
        label_np, label_np, center, list(native_spacing),
        image_size=(T, T, T), crop_mm=crop_spacing_mm, jitter=jitter, rng=rng)
    starts = geom[0].tolist()
    crop_sizes = geom[1].tolist()
    # Map the native physical crop box -> cache-grid indices (phys = idx * spacing).
    img_lo, img_sz = [], []
    for ax in range(3):
        s0 = int(round(starts[ax] * native_spacing[ax] / cache_spacing_mm))
        sz = int(round(crop_sizes[ax] * native_spacing[ax] / cache_spacing_mm))
        s0 = min(max(0, s0), max(0, img_cache_np.shape[ax] - 1))
        sz = max(1, min(sz, img_cache_np.shape[ax] - s0))
        img_lo.append(s0)
        img_sz.append(sz)
    img_crop = np.ascontiguousarray(
        img_cache_np[img_lo[0]:img_lo[0] + img_sz[0],
                     img_lo[1]:img_lo[1] + img_sz[1],
                     img_lo[2]:img_lo[2] + img_sz[2]])
    if normalize_fn is not None:
        img_crop = normalize_fn(img_crop)
    image_t = place_image(img_crop, out_sizes, pad_lo, T)  # cache->out ~identity when pitches match
    lbl_small = resample_binary(crop_lbl == class_idx, tuple(out_sizes),
                                mode=mask_downsample, occ_thr=occ_thr)
    # dtype preserved: long for occupancy/nearest, float32 (partial-volume fraction) for "soft"
    label_t = place_label(lbl_small, out_sizes, pad_lo, T)
    return image_t, label_t, geom


class TotalSegProvider:
    """Source-specific I/O for the totalseg family: scan + bbox caches and a single
    raw_ct organ-crop `load`. Missing ct_raw.npy is a hard error."""

    def __init__(self, root, classes, image_size, split=None, meta_csv=None,
                 max_subjects=None, crop_spacing_mm=1.5, crop_jitter=None,
                 mask_downsample="occupancy", mask_occupancy_thr=0.1, modality="ct",
                 ct_norm=None):
        assert modality in ("ct", "mri"), modality
        # The one CT frame the whole pipeline runs in (see src/totalseg_dataset.CtNormSpec).
        self.ct_spec = resolve_ct_norm(ct_norm)
        self.root = Path(root)
        self.classes = list(classes)
        self.image_size = tuple(image_size)
        self.T = self.image_size[0]
        self.crop_spacing_mm = float(crop_spacing_mm)
        self.crop_jitter = (crop_jitter if crop_jitter is not None else self.T // 4)
        self.mask_downsample = mask_downsample
        self.mask_occupancy_thr = float(mask_occupancy_thr)
        self.modality = modality

        subjects = self._subjects(split, meta_csv, max_subjects)
        scan = self._load_or_build_scan()
        cls_set = set(self.classes)
        self._label_to_subjects = {c: [] for c in self.classes}
        for s in subjects:
            for c in scan.get(s, frozenset()):
                if c in cls_set:
                    self._label_to_subjects[c].append(s)
        self._bbox = self._load_or_build_bbox()
        self._spacings = self._load_spacings()
        self._ct_stats = self._load_ct_stats() if modality == "mri" else {}

    # --- public API ---------------------------------------------------------
    def subjects_for(self, cls):
        return self._label_to_subjects.get(cls, [])

    def load(self, subject, cls, req: LoadRequest) -> LoadResult:
        subj_dir = self.root / subject
        label_np = np.load(subj_dir / "label.npy", mmap_mode="r")
        center = req.center
        if center is None:
            D, H, W = label_np.shape
            center = self._bbox.get(subject, {}).get(cls, (D // 2, H // 2, W // 2))
        native_sp = self._spacings.get(subject, (1.0, 1.0, 1.0))
        norm = ((lambda a: normalize_ct(a, self.ct_spec)) if self.modality == "ct"
                else (lambda a: normalize_mri(a, self._ct_stats[subject])))
        # Fast path: a pre-resampled `ct_raw_{crop_spacing:g}mm.npy` image cache (whole-body
        # native CT downsampled to the crop pitch once, offline). Used only when its pitch
        # equals the requested crop_spacing (so the image is never upsampled); the mask still
        # comes from the full-res native label so occupancy is unchanged. See docs/logs.md.
        cache_p = subj_dir / f"ct_raw_{req.crop_spacing_mm:g}mm.npy"
        if cache_p.exists():
            img_cache_np = np.load(cache_p, mmap_mode="r")
            image_t, label_t, geom = crop_and_place_cached(
                img_cache_np, label_np, _ALL_CLASSES_IDX.get(cls, -1), center, self.T,
                crop_spacing_mm=req.crop_spacing_mm, native_spacing=native_sp,
                cache_spacing_mm=float(req.crop_spacing_mm),
                jitter=self.crop_jitter, rng=req.rng,
                mask_downsample=self.mask_downsample, occ_thr=self.mask_occupancy_thr,
                normalize_fn=lambda a: norm(np.ascontiguousarray(a)))
        else:
            raw = subj_dir / "ct_raw.npy"
            if not raw.exists():
                raise FileNotFoundError(f"{raw} missing (v2 requires ct_raw.npy)")
            image_np = np.load(raw, mmap_mode="r")
            image_t, label_t, geom = crop_and_place(
                image_np, label_np, _ALL_CLASSES_IDX.get(cls, -1), center, self.T,
                crop_spacing_mm=req.crop_spacing_mm, native_spacing=native_sp,
                jitter=self.crop_jitter, rng=req.rng,
                mask_downsample=self.mask_downsample, occ_thr=self.mask_occupancy_thr,
                normalize_fn=lambda a: norm(np.ascontiguousarray(a)))
        spacing = torch.full((3,), float(req.crop_spacing_mm), dtype=torch.float32)
        return LoadResult(image=image_t, label=label_t, spacing=spacing, crop_geom=geom)

    # --- subjects + caches --------------------------------------------------
    def _subjects(self, split, meta_csv, max_subjects):
        alls = sorted(p.name for p in self.root.iterdir() if p.is_dir())
        if split is not None:
            csv_path = Path(meta_csv) if meta_csv else self.root / "meta.csv"
            valid = set()
            with open(csv_path, encoding="utf-8-sig") as f:
                for row in csv.DictReader(f, delimiter=";"):
                    if row["split"].strip() == split:
                        valid.add(row["image_id"].strip())
            alls = [s for s in alls if s in valid]
        return alls[:max_subjects] if max_subjects is not None else alls

    def _key(self):
        alls = sorted(p.name for p in self.root.iterdir() if p.is_dir())
        return hashlib.sha256("|".join(alls).encode()).hexdigest()[:12]

    def _load_or_build_scan(self):
        path = self.root / f".scan_cache_{self._key()}.pkl"
        if path.exists():
            with open(path, "rb") as f:
                return pickle.load(f)
        cache = {}
        for s in (p.name for p in self.root.iterdir() if p.is_dir()):
            lp = self.root / s / "label.npy"
            if not lp.exists():
                continue
            try:
                idxs = set(np.unique(np.load(lp, mmap_mode="r")))
            except (EOFError, ValueError, OSError):
                continue
            cache[s] = frozenset(_IDX_TO_CLASS[i] for i in idxs if i in _IDX_TO_CLASS)
        with open(path, "wb") as f:
            pickle.dump(cache, f)
        return cache

    def _load_or_build_bbox(self):
        path = self.root / f".bbox_cache_{self._key()}.pkl"
        if path.exists():
            with open(path, "rb") as f:
                return pickle.load(f)
        subs = [p.name for p in self.root.iterdir()
                if p.is_dir() and (p / "label.npy").exists()]
        cache = {}
        with ProcessPoolExecutor(max_workers=min(16, os.cpu_count() or 1)) as ex:
            futs = {ex.submit(_bbox_for_subject, self.root, s): s for s in subs}
            for fut in as_completed(futs):
                s, res = fut.result()
                if res is not None:
                    cache[s] = res
        with open(path, "wb") as f:
            pickle.dump(cache, f)
        return cache

    def _load_spacings(self):
        path = self.root / "spacings.json"
        if not path.exists():
            return {}
        with open(path) as f:
            raw = json.load(f)
        return {s: tuple(float(x) for x in m["spacing"]) for s, m in raw.items()}

    def _load_ct_stats(self):
        path = self.root / "ct_stats.json"
        if not path.exists():
            return {}
        with open(path) as f:
            return json.load(f)
