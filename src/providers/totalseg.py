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
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.incontext_dataset_v2 import LoadRequest, LoadResult
from src.providers.volume_cache import get_cache
from src.totalseg_dataloader_incontext import (
    organ_crop_arrays, place_image, place_label, resample_binary,
    _bbox_for_subject, _IDX_TO_CLASS,
)
from src.totalseg_dataset import (_ALL_CLASSES_IDX, normalize_ct, normalize_mri,
                                  resolve_ct_norm)


@dataclass
class NativeCrop:
    """Native-pitch organ crop + its geometry, integer-decimated toward out_sizes.

    NO z-score, NO resample-to-out_sizes, NO centre-placement: those happen on the
    GPU downstream. `decim` is chosen per-axis so the decimated grid stays
    >= out_sizes (the GPU step only ever downsamples). The image IS already HU-clipped
    to the ct_spec window (clip does not commute with the decimation mean, so it has to
    happen first — see `build_native_crop`), and the label is already reduced to the
    target class as a partial-volume FRACTION, never a point sample.
    """
    image: torch.Tensor           # (d,h,w) fp16, clipped HU, decimated crop
    label_frac: torch.Tensor      # (d,h,w) fp16 in [0,1], per-class partial-volume fraction
    class_idx: int                # merged-label index of `cls` (-1 if unknown)
    has_fg: bool                  # class present in the NATIVE crop (pre-decimation)
    out_sizes: list               # from organ_crop_arrays
    pad_lo: list                  # from organ_crop_arrays
    crop_geom: torch.Tensor       # (4,3) int64 — identical to crop_and_place's
    crop_spacing_mm: float
    decim: tuple                  # per-axis integer decimation factor (>=1)
    modality: str = "ct"         # "ct" | "mri" — carried for the GPU realize/aug frame


def _decim_avg_pool(arr_t, decim):
    """Integer avg-pool downsample of a (d,h,w) float tensor by `decim` (per-axis int>=1)."""
    if all(d == 1 for d in decim):
        return arr_t
    return F.avg_pool3d(arr_t[None, None], kernel_size=decim, stride=decim)[0, 0]


def build_native_crop(crop_ct, crop_lbl, class_idx, out_sizes, pad_lo, geom, *,
                      crop_spacing_mm, ct_spec=None, modality="ct"):
    """Assemble a `NativeCrop` payload from an `organ_crop_arrays` result.

    `decim[a] = crop_sizes[a] // out_sizes[a]` (>=1), so the payload grid stays >=
    out_sizes and the GPU realize only ever downsamples.

    Image: HU-clipped to `ct_spec` FIRST (clamp does not commute with the mean, and the
    reference `crop_and_place` normalizes before it resamples), then avg-pooled by `decim`.
    Label: the target-class binary mask is built at NATIVE resolution and avg-pooled by
    `decim` into a partial-volume fraction — the composition of that pool with the GPU's
    final `_area_pool_3d` to out_sizes reproduces the reference's single native->out_sizes
    area pool for integer factors. `has_fg` records class presence pre-decimation so the
    never-empty / soft peak-floor guards still fire for sub-cell structures.
    """
    crop_sizes = geom[1].tolist()
    decim = tuple(max(1, int(cs) // max(1, int(o)))
                  for cs, o in zip(crop_sizes, out_sizes))
    # np.array (not ascontiguousarray) always copies -> never aliases the read-only RAM cache
    img_t = torch.from_numpy(np.array(crop_ct)).float()
    if ct_spec is not None:
        img_t = img_t.clamp(ct_spec.clip_lo, ct_spec.clip_hi)
    binm = torch.from_numpy(np.array(crop_lbl) == class_idx)
    has_fg = bool(binm.any())
    img_t = _decim_avg_pool(img_t, decim)
    frac_t = _decim_avg_pool(binm.float(), decim)
    return NativeCrop(image=img_t.half(), label_frac=frac_t.half(),
                      class_idx=int(class_idx), has_fg=has_fg,
                      out_sizes=list(out_sizes), pad_lo=list(pad_lo),
                      crop_geom=geom, crop_spacing_mm=float(crop_spacing_mm),
                      decim=decim, modality=modality)


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


def _resolve_jitter(req: LoadRequest, default: int) -> int:
    """Per-load crop-jitter: req.jitter when set, else the provider default."""
    return int(req.jitter) if req.jitter is not None else int(default)


class TotalSegProvider:
    """Source-specific I/O for the totalseg family: scan + bbox caches and a single
    raw_ct organ-crop `load`. Missing ct_raw.npy is a hard error."""

    def __init__(self, root, classes, image_size, split=None, meta_csv=None,
                 max_subjects=None, crop_spacing_mm=1.5, crop_jitter=None,
                 mask_downsample="occupancy", mask_occupancy_thr=0.1, modality="ct",
                 ct_norm=None, ram_cache=False, ram_cache_max_subjects=None):
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

        # Optional process-lifetime RAM cache of native volumes, preloaded here in
        # the main process so DataLoader forks share the buffers copy-on-write.
        self._ram = None
        if ram_cache:
            subs = sorted({s for lst in self._label_to_subjects.values() for s in lst})
            self._ram = get_cache(self.root, subs, max_subjects=ram_cache_max_subjects)

    def __getstate__(self):
        """Never ship the RAM cache through pickle.

        DataLoader workers started with `spawn`/`forkserver` (the eval loaders, so they
        don't inherit the parent CUDA context) pickle the dataset -> the provider. The
        cache is a process-lifetime singleton holding every subject loaded so far (tens of
        GB); pickling it would blow up every worker. Workers only ever call `load()`, which
        reads npy via mmap and never consults `_ram`; `load_native_crop` (the only reader)
        runs in the main process, which keeps its `_ram`. `fork` workers are unaffected
        (no pickling, copy-on-write pages)."""
        return {**self.__dict__, "_ram": None}

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
        jitter = _resolve_jitter(req, self.crop_jitter)
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
                jitter=jitter, rng=req.rng,
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
                jitter=jitter, rng=req.rng,
                mask_downsample=self.mask_downsample, occ_thr=self.mask_occupancy_thr,
                normalize_fn=lambda a: norm(np.ascontiguousarray(a)))
        spacing = torch.full((3,), float(req.crop_spacing_mm), dtype=torch.float32)
        return LoadResult(image=image_t, label=label_t, spacing=spacing, crop_geom=geom,
                          modality=self.modality)

    def load_native_crop(self, subject, cls, req: LoadRequest) -> "NativeCrop":
        """Native-pitch organ crop + geometry, integer-decimated toward out_sizes.

        No normalize / no resample / no placement — the GPU realize step does those.
        `crop_geom` is byte-identical to `crop_and_place`'s for the same args, and
        `req.rng` is consumed exactly once (inside `organ_crop_arrays`).
        """
        if getattr(self, "_ram", None) is not None and subject in self._ram:
            image_np = self._ram[subject]["ct_raw"]
            label_np = self._ram[subject]["label"]
        else:
            subj_dir = self.root / subject
            label_np = np.load(subj_dir / "label.npy", mmap_mode="r")
            image_np = np.load(subj_dir / "ct_raw.npy", mmap_mode="r")
        center = req.center
        if center is None:
            D, H, W = label_np.shape
            center = self._bbox.get(subject, {}).get(cls, (D // 2, H // 2, W // 2))
        jitter = _resolve_jitter(req, self.crop_jitter)
        native_sp = self._spacings.get(subject, (1.0, 1.0, 1.0))
        crop_ct, crop_lbl, out_sizes, pad_lo, geom = organ_crop_arrays(
            image_np, label_np, center, list(native_sp),
            image_size=(self.T, self.T, self.T), crop_mm=req.crop_spacing_mm,
            jitter=jitter, rng=req.rng)
        # build_native_crop copies the (read-only, possibly cached) slices out.
        # ct_spec only for CT: MRI needs per-subject stats, which the GPU realize path
        # does not carry -- common._assert_cascade_supported rejects MRI + gpu_realize_crop.
        return build_native_crop(
            crop_ct, crop_lbl, _ALL_CLASSES_IDX.get(cls, -1), out_sizes, pad_lo, geom,
            crop_spacing_mm=float(req.crop_spacing_mm),
            ct_spec=(self.ct_spec if self.modality == "ct" else None),
            modality=self.modality)

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
