"""Shared base for volume providers whose source is stored on its NATIVE grid.

FLARE22 and NasalSeg are both converted to `{subj}/ct_raw.npy` + `{subj}/label.npy` at
native (anisotropic) spacing with a root `spacings.json`; the organ crop + resample to the
isotropic T^3 model grid happens here, at load time, so `crop_spacing_mm` stays a config
knob rather than a property of the conversion. Sources differ only in their class list,
label index map, and defaults — everything else lives in `NativeGridProvider`.

Two defaults differ from `TotalSegProvider`, both because these grids are finer and
anisotropic (on 1.5mm-isotropic totalseg the resample was an identity, which masked both):
  * `image_antialias=True`  — plain trilinear aliases under in-plane decimation.
  * `mask_occupancy_thr=0.5` — the v2 totalseg default of 0.1 dilates thin structures.

NOTE ON METRICS: the crop resample is lossy for GT, so scoring in crop space against the
RESAMPLED label overstates Dice versus native voxel space. `native_meta` exposes the
shape/spacing/affine needed to map a prediction back through `crop_geom`.
"""
import json
import os
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

from src.incontext_dataset_v2 import LoadRequest, LoadResult
from src.providers.totalseg import crop_and_place
from src.totalseg_dataset import normalize_ct


def resolve_classes_for(all_classes: list[str], value, source: str) -> list[str]:
    """Resolve a cfg.data.{train,val}_classes value against a fixed class list."""
    if isinstance(value, str):
        if value in ("all", "benchmark"):
            return list(all_classes)
        raise ValueError(f"unknown {source} class spec {value!r} (use 'all' or a list)")
    classes = [str(c) for c in value]
    unknown = [c for c in classes if c not in all_classes]
    if unknown:
        raise ValueError(f"not {source} classes: {unknown}; available: {all_classes}")
    return classes


def _centroids_for_subject(root, subj, idx_to_class):
    """Per-class label centroids for one subject (module-level so it pickles)."""
    try:
        arr = np.load(Path(root) / subj / "label.npy", mmap_mode="r")
        D, H, W = arr.shape
        d_g = np.arange(D, dtype=np.float32)[:, None, None]
        h_g = np.arange(H, dtype=np.float32)[None, :, None]
        w_g = np.arange(W, dtype=np.float32)[None, None, :]
        out = {}
        for idx in np.unique(arr):
            name = idx_to_class.get(int(idx))
            if name is None:
                continue
            mask = arr == idx
            n = int(mask.sum())
            if n == 0:
                continue
            out[name] = (int((d_g * mask).sum() / n),
                         int((h_g * mask).sum() / n),
                         int((w_g * mask).sum() / n))
        return subj, out
    except Exception:  # noqa: BLE001
        return subj, None


class NativeGridProvider:
    """Native-grid organ-crop provider (VolumeProvider protocol).

    Subclasses set `SOURCE`, `ALL_CLASSES` and `CLASS_IDX` (name -> label value)."""

    SOURCE: str = "native"
    ALL_CLASSES: list[str] = []
    CLASS_IDX: dict[str, int] = {}

    def __init__(self, root, classes=None, image_size=(128, 128, 128),
                 max_subjects=None, crop_spacing_mm=1.5, crop_jitter=0,
                 mask_downsample="occupancy", mask_occupancy_thr=0.5,
                 image_antialias=True):
        self.root = Path(root)
        self.modality = "ct"   # native-grid sources are all CT; contract for a future MultiModalProvider
        self.classes = resolve_classes_for(
            self.ALL_CLASSES, classes if classes is not None else "all", self.SOURCE)
        self.image_size = tuple(image_size)
        self.T = self.image_size[0]
        self.crop_spacing_mm = float(crop_spacing_mm)
        self.crop_jitter = int(crop_jitter or 0)
        self.mask_downsample = mask_downsample
        self.mask_occupancy_thr = float(mask_occupancy_thr)
        self.image_antialias = bool(image_antialias)

        self._meta = self._load_meta()
        subjects = sorted(p.name for p in self.root.iterdir()
                          if p.is_dir() and (p / "ct_raw.npy").exists())
        if not subjects:
            raise FileNotFoundError(
                f"no converted cases under {self.root} — run the {self.SOURCE} converter")
        if max_subjects is not None:
            subjects = subjects[:int(max_subjects)]
        self._centroids = self._load_or_build_centroids(subjects)
        self._label_to_subjects = {
            c: [s for s in subjects if c in self._centroids.get(s, {})]
            for c in self.classes}

    # --- public API ---------------------------------------------------------
    def subjects_for(self, cls):
        return self._label_to_subjects.get(cls, [])

    def native_meta(self, subject) -> dict:
        """Native-grid geometry {spacing, shape, affine} — the frame GT lives in, and what
        a native-space metric needs in order to invert `crop_geom`."""
        return self._meta[subject]

    def load(self, subject, cls, req: LoadRequest) -> LoadResult:
        subj_dir = self.root / subject
        image_np = np.load(subj_dir / "ct_raw.npy", mmap_mode="r")
        label_np = np.load(subj_dir / "label.npy", mmap_mode="r")
        center = req.center
        if center is None:
            D, H, W = label_np.shape
            center = self._centroids.get(subject, {}).get(cls, (D // 2, H // 2, W // 2))
        image_t, label_t, geom = crop_and_place(
            image_np, label_np, self.CLASS_IDX[cls], center, self.T,
            crop_spacing_mm=req.crop_spacing_mm,
            native_spacing=self._meta[subject]["spacing"],
            jitter=self.crop_jitter, rng=req.rng,
            mask_downsample=self.mask_downsample, occ_thr=self.mask_occupancy_thr,
            normalize_fn=lambda a: normalize_ct(np.ascontiguousarray(a)),
            antialias=self.image_antialias)
        spacing = torch.full((3,), float(req.crop_spacing_mm), dtype=torch.float32)
        return LoadResult(image=image_t, label=label_t, spacing=spacing, crop_geom=geom,
                          modality=self.modality)

    # --- caches -------------------------------------------------------------
    def _load_meta(self):
        path = self.root / "spacings.json"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} missing — run the {self.SOURCE} converter (spacing is not "
                "recoverable from the .npy files)")
        with open(path) as f:
            raw = json.load(f)
        return {s: {"spacing": tuple(float(x) for x in m["spacing"]),
                    "shape": tuple(int(x) for x in m["shape"]),
                    "affine": np.asarray(m["affine"], dtype=np.float64)}
                for s, m in raw.items()}

    def _load_or_build_centroids(self, subjects):
        path = self.root / ".centroid_cache.pkl"
        if path.exists():
            with open(path, "rb") as f:
                cache = pickle.load(f)
            if all(s in cache for s in subjects):
                return cache
        idx_to_class = {v: k for k, v in self.CLASS_IDX.items()}
        cache = {}
        with ProcessPoolExecutor(max_workers=min(16, os.cpu_count() or 1)) as ex:
            futs = [ex.submit(_centroids_for_subject, str(self.root), s, idx_to_class)
                    for s in subjects]
            for fut in as_completed(futs):
                s, res = fut.result()
                if res is not None:
                    cache[s] = res
        with open(path, "wb") as f:
            pickle.dump(cache, f)
        return cache
