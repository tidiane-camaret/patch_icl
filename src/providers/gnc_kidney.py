"""GNC_705 kidney-lesion provider (`NativeGridProvider` subclass, with a per-class STORAGE
override). Local restricted-access National Cohort MRI, water-channel Dixon, 14 independent
lesion-instance classes -- see `docs/datasets/gnc_kidney_lesions.md` for the full
characterization.

`FILE_TO_CLASS` maps the cohort's raw `.nii.gz` basenames to short class names. Per
`docs/datasets/gnc_kidney_lesions.md` #5, `mask_X`/`mask_X.cyst` are NOT a subregion of `X` in
general (verified via crop-box comparison: identical box => subset of the same lesion,
different box => a second, independent lesion instance) -- every entry here is its own class,
none are merged.

**Why this subclass can't use `NativeGridProvider`'s shared single `label.npy`**: that format
stores ONE integer class value per voxel, which assumes the classes PARTITION the volume
(mutually exclusive, as every other multi-class source here — MSD Hippocampus/Prostate — is by
construction). GNC's classes do NOT partition: verified at conversion time, painting all of a
subject's classes into one shared array produces thousands of overlapping voxels (a `mask_X`
subset literally sits inside its `X` superset). Overwriting one class's voxels with another's
class index would silently shrink the superset's mask every time both are present. Instead,
`scripts/convert_gnc_kidney.py` writes one binary plane per PRESENT class,
`{subj}/label_{cls}.npy` (D,H,W) uint8 0/1 — lossless regardless of how classes overlap — and
this subclass overrides the three `NativeGridProvider` methods that touch the shared
`label.npy` path (`_load_or_build_centroids`, `load`, `load_native_crop`) to read the
per-class plane and always query it with `class_idx=1` (a binary plane needs no other value).

`hyper_mask_r` has only 1 subject total: in-context eval needs >=2 subjects per class (one
context, one target) to avoid a self-context fallback. It stays in the registry (the converter
still writes it) but `configs/experiment/3d/dataset/gnc_kidney.yaml` restricts `val_classes` to
the other 13 for eval, so it never silently produces a leakage-inflated number.
"""
import os
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

from src.incontext_dataset_v2 import LoadRequest, LoadResult
from src.providers.native_grid import NativeGridProvider, resolve_classes_for
from src.providers.totalseg import (_resolve_center, _resolve_jitter, build_native_crop,
                                    crop_and_place)
from src.totalseg_dataloader_incontext import organ_crop_arrays

FILE_TO_CLASS = {
    "Hyper_R.nii.gz": "hyper_r",
    "Hyper_L.nii.gz": "hyper_l",
    "Hypo_R.nii.gz": "hypo_r",
    "Hypo_L.nii.gz": "hypo_l",
    "Complex_R.nii.gz": "complex_r",
    "Complex_L.nii.gz": "complex_l",
    "mask_hyper.cyst.R.nii.gz": "hyper_cyst_r",
    "mask_hyper.cyst.L.nii.gz": "hyper_cyst_l",
    "mask_hyper.R.nii.gz": "hyper_mask_r",
    "mask_hyper.L.nii.gz": "hyper_mask_l",
    "mask_hypo_R.nii.gz": "hypo_mask_r",
    "mask_hypo_L.nii.gz": "hypo_mask_l",
    "mask_complex.cyst.R.nii.gz": "complex_cyst_r",
    "mask_complex.cyst.L.nii.gz": "complex_cyst_l",
}

GNC_KIDNEY_CLASSES = list(FILE_TO_CLASS.values())
CLASS_IDX = {name: i + 1 for i, name in enumerate(GNC_KIDNEY_CLASSES)}  # metadata only (see
                                                                        # module docstring) --
                                                                        # NOT used to index a
                                                                        # shared label array.

# 13 classes with >=2 subjects (mask_hyper.R excluded -- see module docstring).
GNC_KIDNEY_EVAL_CLASSES = [c for c in GNC_KIDNEY_CLASSES if c != "hyper_mask_r"]


def resolve_gnc_kidney_classes(value) -> list[str]:
    return resolve_classes_for(GNC_KIDNEY_CLASSES, value, "gnc_kidney")


def _gnc_centroid_for_subject(root, subj, classes):
    """Per-class centroid for one subject, reading each present `label_{cls}.npy` plane
    independently (module-level so it pickles for ProcessPoolExecutor)."""
    out = {}
    for cls in classes:
        p = Path(root) / subj / f"label_{cls}.npy"
        if not p.exists():
            continue
        try:
            arr = np.load(p, mmap_mode="r")
            coords = np.argwhere(np.asarray(arr) != 0)
            if coords.shape[0] > 0:
                out[cls] = tuple(int(x) for x in coords.mean(axis=0).round())
        except Exception:  # noqa: BLE001
            continue
    return subj, out


class GncKidneyProvider(NativeGridProvider):
    SOURCE = "gnc_kidney"
    ALL_CLASSES = GNC_KIDNEY_CLASSES
    CLASS_IDX = CLASS_IDX
    MODALITY = "mri"

    def _load_or_build_centroids(self, subjects):
        path = self.root / ".centroid_cache_perclass.pkl"
        if path.exists():
            with open(path, "rb") as f:
                cache = pickle.load(f)
            if all(s in cache for s in subjects):
                return cache
        cache = {}
        with ProcessPoolExecutor(max_workers=min(16, os.cpu_count() or 1)) as ex:
            futs = [ex.submit(_gnc_centroid_for_subject, str(self.root), s, self.classes)
                    for s in subjects]
            for fut in as_completed(futs):
                s, res = fut.result()
                if res:
                    cache[s] = res
        with open(path, "wb") as f:
            pickle.dump(cache, f)
        return cache

    def load(self, subject, cls, req: LoadRequest) -> LoadResult:
        subj_dir = self.root / subject
        image_np = np.load(subj_dir / "ct_raw.npy", mmap_mode="r")
        label_np = np.load(subj_dir / f"label_{cls}.npy", mmap_mode="r")
        center = req.center
        if center is None:
            D, H, W = label_np.shape
            fallback = self._centroids.get(subject, {}).get(cls, (D // 2, H // 2, W // 2))
            center = _resolve_center(req, label_np, 1, fallback)  # binary plane: class_idx=1
        image_t, label_t, geom = crop_and_place(
            image_np, label_np, 1, center, self.T,
            crop_spacing_mm=req.crop_spacing_mm,
            native_spacing=self._meta[subject]["spacing"],
            jitter=_resolve_jitter(req, self.crop_jitter), rng=req.rng,
            mask_downsample=self.mask_downsample, occ_thr=self.mask_occupancy_thr,
            normalize_fn=self._normalize_fn(subject),
            antialias=self.image_antialias)
        spacing = torch.full((3,), float(req.crop_spacing_mm), dtype=torch.float32)
        return LoadResult(image=image_t, label=label_t, spacing=spacing, crop_geom=geom,
                          modality=self.modality)

    def load_native_crop(self, subject, cls, req: LoadRequest):
        subj_dir = self.root / subject
        image_np = np.load(subj_dir / "ct_raw.npy", mmap_mode="r")
        label_np = np.load(subj_dir / f"label_{cls}.npy", mmap_mode="r")
        center = req.center
        if center is None:
            D, H, W = label_np.shape
            fallback = self._centroids.get(subject, {}).get(cls, (D // 2, H // 2, W // 2))
            center = _resolve_center(req, label_np, 1, fallback)
        crop_ct, crop_lbl, out_sizes, pad_lo, geom = organ_crop_arrays(
            image_np, label_np, center, list(self._meta[subject]["spacing"]),
            image_size=(self.T, self.T, self.T), crop_mm=req.crop_spacing_mm,
            jitter=_resolve_jitter(req, self.crop_jitter), rng=req.rng)
        return build_native_crop(
            crop_ct, crop_lbl, 1, out_sizes, pad_lo, geom,
            crop_spacing_mm=float(req.crop_spacing_mm),
            norm=self._norm_spec(subject), modality=self.modality)
