"""In-context dataset over MAISI (NV-Generate-CTMR) synthetic CT/mask pairs.

Consumes the native `.npz` pairs written by
experiments/3d/synth_task_generation/gen_maisi_fast.py — each file holds:
    ct     float16 (D,H,W)  z-scored HU (already model-input normalised)
    label  uint8   (D,H,W)  MAISI 132-class vocabulary (NOT TotalSegmentator)
    spacing float32 (3,)

Reuses TotalSegInContextDataset for context sampling, class-balanced sampling,
augmentation, self-context, eval-seed determinism, and the collate contract.
Overrides only the storage/vocabulary seams:

  * class identity  — MAISI vocab (data/maisi_classes.py), not the 117 TotalSeg
                      classes. subject->classes comes from scanning each npz label.
  * subjects        — one "subject" per .npz file (its stem); optional hash split.
  * loading         — reads the npz and serves either
                        - resize path: the whole volume anisotropically resized to
                          image_size (reported spacing = native * shape / image_size), or
                        - use_crop path: an organ-centred crop of fixed physical extent
                          (T*crop_spacing_mm) resampled to T^3 -> isotropic crop_spacing_mm.
"""
import hashlib
import os
import pickle
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from data.maisi_classes import MAISI_IDX_TO_CLASS, MAISI_CLASS_TO_IDX, MAISI_CLASSES
from src.totalseg_dataloader_incontext import (
    TotalSegInContextDataset,
    organ_crop_arrays, place_image, place_label, resample_binary,
)


def _maisi_bbox_for_subject(args: tuple) -> tuple[str, dict | None]:
    """Per-MAISI-class centroids for one npz, in native label voxel space (module-level
    for pickling). Single D-slice loop with per-slice bincount keeps memory bounded on
    the whole-body volumes (avoids materialising 3 coordinate arrays over the full grid)."""
    npz_path = args[0]
    try:
        with np.load(npz_path) as d:
            lab = np.asarray(d["label"])
        D, H, W = lab.shape
        n = 256  # MAISI ids fit in [0, 200]
        cnt = np.zeros(n); sd = np.zeros(n); sh = np.zeros(n); sw = np.zeros(n)
        hidx = np.repeat(np.arange(H, dtype=np.float64), W)
        widx = np.tile(np.arange(W, dtype=np.float64), H)
        for z in range(D):
            ids = lab[z].ravel().astype(np.int64)
            c = np.bincount(ids, minlength=n)[:n].astype(np.float64)
            cnt += c
            sd += c * z
            sh += np.bincount(ids, weights=hidx, minlength=n)[:n]
            sw += np.bincount(ids, weights=widx, minlength=n)[:n]
        res: dict[str, tuple[int, int, int]] = {}
        for idx in range(1, n):
            if cnt[idx] == 0 or idx not in MAISI_IDX_TO_CLASS:
                continue
            res[MAISI_IDX_TO_CLASS[idx]] = (int(round(sd[idx] / cnt[idx])),
                                            int(round(sh[idx] / cnt[idx])),
                                            int(round(sw[idx] / cnt[idx])))
        return Path(npz_path).stem, res
    except Exception:
        return Path(npz_path).stem, None


class SynthGenMaisiDataset(TotalSegInContextDataset):
    def __init__(
        self,
        root: str | Path,
        classes: Optional[list[str]] = None,
        image_size: Optional[tuple[int, int, int]] = (256, 256, 256),
        split: Optional[str] = None,
        context_size: int = 3,
        max_subjects: Optional[int] = None,
        aug_cfg=None,
        class_balanced: bool = False,
        eval_seed: Optional[int] = None,
        val_frac: float = 0.1,
        defer_aug_to_gpu: bool = False,
        use_crop: bool = False,
        crop_spacing_mm: float = 1.5,
        crop_jitter: Optional[int] = None,
        mask_downsample: str = "nearest",
        mask_occupancy_thr: float = 0.5,
    ):
        self.root = Path(root)
        # subject id -> npz path (one file = one "subject"); needed by the overridden
        # _get_subjects / _load_or_build_cache which run inside super().__init__.
        self._npz = {p.stem: p for p in sorted(self.root.glob("*.npz"))}
        if not self._npz:
            raise FileNotFoundError(f"no *.npz pairs found in {self.root}")
        self._val_frac = float(val_frac)
        self._spacing_by_subj: dict[str, torch.Tensor] = {}   # native mm/voxel from npz
        self._shape_by_subj: dict[str, tuple] = {}            # native (D,H,W)

        super().__init__(
            root=root,
            classes=classes if classes is not None else list(MAISI_CLASSES),
            image_size=image_size,
            split=split,
            context_size=context_size,
            max_subjects=max_subjects,
            aug_cfg=aug_cfg,
            synth_method=None,
            p_synth=0.0,
            class_balanced=class_balanced,
            use_crop=use_crop,
            crop_spacing_mm=crop_spacing_mm,
            crop_jitter=crop_jitter,
            num_labels_per_sample=1,
            eval_seed=eval_seed,
            modality="ct",
            defer_aug_to_gpu=defer_aug_to_gpu,
            mask_downsample=mask_downsample,
            mask_occupancy_thr=mask_occupancy_thr,
        )

    # --- subjects / vocabulary ----------------------------------------------
    def _split_of(self, stem: str) -> str:
        """Deterministic per-file train/val split by stem hash (val_frac to 'val')."""
        h = int(hashlib.sha256(stem.encode()).hexdigest(), 16) % 1000
        return "val" if h < int(self._val_frac * 1000) else "train"

    def _get_subjects(self, split, meta_csv, max_subjects) -> list[str]:
        """One subject per .npz file. split=None -> all; else a stable hash split."""
        subs = sorted(self._npz)
        if split is not None:
            assert split in ("train", "val"), f"unknown split {split!r}"
            subs = [s for s in subs if self._split_of(s) == split]
        if max_subjects is not None:
            subs = subs[:max_subjects]
        return subs

    def _load_or_build_cache(self) -> dict[str, frozenset]:
        """subject -> frozenset(MAISI class names present), scanning each npz label.
        Also records native spacing + shape (for effective-spacing reporting). Cached as a
        pickle keyed by the npz-file-list hash (rebuilt when the set of files changes)."""
        stems = sorted(self._npz)
        key = hashlib.sha256(("maisi|" + "|".join(stems)).encode()).hexdigest()[:12]
        cache_path = self.root / f".maisi_scan_cache_{key}.pkl"
        if cache_path.exists():
            with open(cache_path, "rb") as f:
                blob = pickle.load(f)
            print(f"Loaded MAISI scan cache ({len(blob['classes'])} subjects) from {cache_path.name}",
                  flush=True)
        else:
            print(f"Building MAISI scan cache for {len(stems)} pairs (saved to {cache_path.name})...",
                  flush=True)
            classes: dict = {}; spac: dict = {}; shp: dict = {}
            for s in stems:
                try:
                    with np.load(self._npz[s]) as d:
                        ids = np.unique(d["label"])
                        spac[s] = np.asarray(d["spacing"], dtype=np.float32)
                        shp[s] = tuple(int(x) for x in d["label"].shape)
                except (zlib.error, ValueError, OSError):
                    print(f"  Skipping corrupt npz: {s}", flush=True)
                    continue
                classes[s] = frozenset(MAISI_IDX_TO_CLASS[int(i)] for i in ids
                                       if int(i) in MAISI_IDX_TO_CLASS)
            blob = {"classes": classes, "spacing": spac, "shape": shp}
            with open(cache_path, "wb") as f:
                pickle.dump(blob, f)
            print(f"MAISI scan cache saved ({len(classes)} subjects).", flush=True)
        self._spacing_by_subj = {s: torch.as_tensor(v, dtype=torch.float32)
                                 for s, v in blob["spacing"].items()}
        self._shape_by_subj = dict(blob["shape"])
        return blob["classes"]

    # --- spacing -------------------------------------------------------------
    def _get_spacing(self, subj: str) -> torch.Tensor:
        """Effective mm/voxel of the returned tensor.

        use_crop: native spacing (the crop path resamples to crop_spacing_mm itself, and
        _reported_spacing returns crop_mm). resize path: native * shape / image_size, per
        axis — the true spacing after the anisotropic resize-to-cube in _load."""
        native = self._spacing_by_subj.get(subj, torch.full((3,), 1.5))
        if self.use_crop or self.image_size is None:
            return native
        shape = self._shape_by_subj.get(subj)
        if shape is None:
            return native
        scale = torch.tensor([shape[i] / self.image_size[i] for i in range(3)], dtype=torch.float32)
        return native * scale

    # --- bbox cache (use_crop) ----------------------------------------------
    def _load_or_build_bbox_cache(self) -> dict[str, dict[str, tuple[int, int, int]]]:
        """{subject: {maisi_class: (d,h,w)}} organ centroids in native npz voxel space,
        for the use_crop path. Built once (parallel), pickled, keyed by the npz-list hash."""
        stems = sorted(self._npz)
        key = hashlib.sha256(("maisi_bbox|" + "|".join(stems)).encode()).hexdigest()[:12]
        cache_path = self.root / f".maisi_bbox_cache_{key}.pkl"
        if cache_path.exists():
            with open(cache_path, "rb") as f:
                cache = pickle.load(f)
            print(f"Loaded MAISI bbox cache ({len(cache)} subjects) from {cache_path.name}",
                  flush=True)
            return cache
        n_workers = min(16, os.cpu_count() or 1)
        print(f"Building MAISI bbox cache for {len(stems)} subjects ({n_workers} workers)...",
              flush=True)
        cache: dict = {}
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futs = {ex.submit(_maisi_bbox_for_subject, (str(self._npz[s]),)): s for s in stems}
            for fut in as_completed(futs):
                subj, res = fut.result()
                if res is not None:
                    cache[subj] = res
                else:
                    print(f"  Skipping bbox for {futs[fut]}", flush=True)
        with open(cache_path, "wb") as f:
            pickle.dump(cache, f)
        print(f"MAISI bbox cache saved ({len(cache)} subjects).", flush=True)
        return cache

    # --- loading -------------------------------------------------------------
    def _load(self, subj: str, cls: str, pred_center=None) -> tuple[torch.Tensor, torch.Tensor]:
        """(1,D,H,W) f32 pre-normalised CT + (D,H,W) long binary mask for `cls`."""
        if self.use_crop:
            return self._load_crop(subj, cls, pred_center=pred_center)
        with np.load(self._npz[subj]) as d:
            ct = np.asarray(d["ct"], dtype=np.float32)          # (D,H,W) normalised
            binary = np.asarray(d["label"]) == MAISI_CLASS_TO_IDX[cls]
        image_t = torch.from_numpy(ct).unsqueeze(0)             # (1,D,H,W)
        if self.image_size is not None and tuple(ct.shape) != tuple(self.image_size):
            image_t = F.interpolate(image_t.unsqueeze(0), size=tuple(self.image_size),
                                    mode="trilinear", align_corners=False)[0]
            label_t = self._resample_binary(binary, tuple(self.image_size))
        else:
            label_t = torch.from_numpy(binary.astype(np.int64))
        return image_t, label_t

    def _load_crop(self, subj: str, cls: str, pred_center=None) -> tuple[torch.Tensor, torch.Tensor]:
        """Organ-centred native crop of fixed physical extent (T*crop_spacing_mm), resampled
        to T^3 -> isotropic crop_spacing_mm/voxel. Reads ct+label from the one npz; centroids
        from the MAISI bbox cache. Reuses the module-level crop/place/resample helpers."""
        T = self.image_size[0]
        with np.load(self._npz[subj]) as d:
            ct_mm = np.asarray(d["ct"], dtype=np.float32)
            label_mm = np.asarray(d["label"])
        D, H, W = label_mm.shape
        if pred_center == "volume_center":
            center = (D // 2, H // 2, W // 2)
        elif pred_center is not None:
            center = tuple(int(c) for c in pred_center)
        else:
            center = self._bbox_cache.get(subj, {}).get(cls) or (D // 2, H // 2, W // 2)

        sp = self._get_spacing(subj).tolist()   # native mm/voxel (use_crop -> npz spacing)
        crop_ct, crop_lbl, out_sizes, pad_lo, geom = organ_crop_arrays(
            ct_mm, label_mm, center, sp, image_size=self.image_size,
            crop_mm=self._crop_mm, jitter=self.crop_jitter, rng=self._cur_rng)
        self._last_crop_geom = geom

        image_t = place_image(crop_ct, out_sizes, pad_lo, T)
        binary = crop_lbl == MAISI_CLASS_TO_IDX[cls]
        label_small = resample_binary(binary, tuple(out_sizes),
                                      mode=self.mask_downsample, occ_thr=self.mask_occupancy_thr)
        label_t = place_label(label_small, out_sizes, pad_lo, T)
        return image_t, label_t
