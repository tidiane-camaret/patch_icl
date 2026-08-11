"""In-context EVAL dataset over ChemoTox body-composition tissues (4 classes).

Reuses TotalSegInContextDataset for context sampling, eval-seed determinism, the
single-label __getitem__ path, and the collate contract. Reads the converted cache
tree (ct.npy + bc.npy + spacings.json, built at 1.5 mm iso by convert_to_npy
--source chemotox). use_crop-only, eval-only."""
import hashlib
import os
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import numpy as np

from src.totalseg_dataloader_incontext import TotalSegInContextDataset

BC_NAMES = ["muscle", "sat", "vat", "imat"]
BC_ID = {n: i + 1 for i, n in enumerate(BC_NAMES)}


def _bc_centroids_for_subject(args) -> tuple[str, dict | None]:
    """Per-class centroid (native bc.npy voxel space) for one subject."""
    root, subj = args
    try:
        arr = np.load(Path(root) / subj / "bc.npy", mmap_mode="r")
        D, H, W = arr.shape
        d_g = np.arange(D, dtype=np.float32)[:, None, None]
        h_g = np.arange(H, dtype=np.float32)[None, :, None]
        w_g = np.arange(W, dtype=np.float32)[None, None, :]
        out: dict[str, tuple[int, int, int]] = {}
        for name, lid in BC_ID.items():
            m = (arr == lid)
            n = int(m.sum())
            if n == 0:
                continue
            out[name] = (int((d_g * m).sum() / n), int((h_g * m).sum() / n),
                         int((w_g * m).sum() / n))
        return subj, out
    except Exception:
        return subj, None


class ChemoToxBCDataset(TotalSegInContextDataset):
    def __init__(self, root, classes=BC_NAMES, image_size=(128, 128, 128),
                 split: Optional[str] = "test", context_size: int = 1,
                 max_subjects: Optional[int] = None, eval_seed: int = 0,
                 use_crop: bool = True, crop_spacing_mm: float = 1.5,
                 crop_jitter: Optional[int] = None):
        assert use_crop, "ChemoToxBCDataset is use_crop-only"
        super().__init__(
            root=root, classes=list(classes), image_size=image_size, split=split,
            context_size=context_size, max_subjects=max_subjects, aug_cfg=None,
            synth_method=None, p_synth=0.0, class_balanced=False, use_crop=True,
            crop_spacing_mm=crop_spacing_mm, crop_jitter=crop_jitter,
            num_labels_per_sample=1, eval_seed=eval_seed, raw_ct=False, modality="ct")

    # --- overrides -----------------------------------------------------------
    def _get_subjects(self, split, meta_csv, max_subjects) -> list[str]:
        assert split in (None, "test"), f"eval-only (split={split!r})"
        subs = sorted(p.name for p in self.root.iterdir()
                      if p.is_dir() and (p / "bc.npy").exists())
        return subs[:max_subjects] if max_subjects is not None else subs

    def _load_or_build_cache(self) -> dict:
        """Every subject carries all 4 diffuse tissues -> trivial subject->classes."""
        return {s: frozenset(BC_NAMES) for s in
                (p.name for p in self.root.iterdir()
                 if p.is_dir() and (p / "bc.npy").exists())}

    def _load_or_build_bbox_cache(self) -> dict:
        subs = sorted(p.name for p in self.root.iterdir()
                      if p.is_dir() and (p / "bc.npy").exists())
        key = hashlib.sha256(("bc_centroid|" + "|".join(subs)).encode()).hexdigest()[:12]
        cache_path = self.root / f".bc_centroid_cache_{key}.pkl"
        if cache_path.exists():
            with open(cache_path, "rb") as f:
                return pickle.load(f)
        n_workers = min(16, os.cpu_count() or 1)
        cache: dict = {}
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futs = {ex.submit(_bc_centroids_for_subject, (str(self.root), s)): s for s in subs}
            for fut in as_completed(futs):
                subj, res = fut.result()
                if res is not None:
                    cache[subj] = res
        with open(cache_path, "wb") as f:
            pickle.dump(cache, f)
        return cache

    def _load_crop(self, subj: str, cls: str):
        """Organ-centred native crop of fixed physical extent (T*crop_spacing_mm),
        resampled to T³. Crops ct.npy + (bc.npy == BC_ID[cls])."""
        subj_dir = self.root / subj
        local_id = BC_ID[cls]
        label_mm = np.load(subj_dir / "bc.npy", mmap_mode="r")
        D, H, W = label_mm.shape
        center = self._bbox_cache.get(subj, {}).get(cls)
        center = center if center is not None else (D // 2, H // 2, W // 2)
        sp = self._get_spacing(subj).tolist()
        crop_ct, crop_lbl, out_sizes, pad_lo = self._organ_crop_arrays(
            subj_dir, label_mm, center, sp)
        image_t = self._place_image(crop_ct, out_sizes, pad_lo)
        label_t = self._place_label(
            self._resample_binary(crop_lbl == local_id, tuple(out_sizes)), out_sizes, pad_lo)
        return image_t, label_t

    def _load(self, subj: str, cls: str):
        return self._load_crop(subj, cls)
