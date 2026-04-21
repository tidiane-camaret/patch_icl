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
import pickle
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from src.totalseg_dataset import (
    ALL_CLASSES,
    _ALL_CLASSES_IDX,
    _load_ct,
    _build_label_volume,
    _resize_volume,
)
from src.augmentations import apply_task_aug, apply_intensity_aug

# Inverse map: orig label index → class name (covers all 117 classes)
_IDX_TO_CLASS: dict[int, str] = {v: k for k, v in _ALL_CLASSES_IDX.items()}


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

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

    Scan cache
    ----------
    The first init call scans every subject's label.npy to record which of the
    117 classes are present.  The result is saved as a pickle file inside the
    dataset root and reused on all subsequent runs — including runs with
    different class subsets or splits.  The cache is keyed by a hash of the
    full set of subject directories, so it is automatically invalidated if
    subjects are added or removed.
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
    ):
        self.root = Path(root)
        self.classes = list(classes)
        self.image_size = image_size
        self.context_size = context_size
        self.aug_cfg = aug_cfg  # None → no augmentation

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

        counts = {cls: len(self.label_to_subjects[cls]) for cls in self.classes}
        print(f"TotalSegInContextDataset: {len(self.samples)} samples | "
              f"context_size={context_size} | class counts: {counts}", flush=True)

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
            arr = np.load(label_npy, mmap_mode="r")
            present_indices = set(np.unique(arr))
            cache[subj] = frozenset(
                _IDX_TO_CLASS[i] for i in present_indices if i in _IDX_TO_CLASS
            )

        with open(cache_path, "wb") as f:
            pickle.dump(cache, f)
        print(f"Scan cache saved ({len(cache)} subjects).", flush=True)
        return cache

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

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        subj, cls = self.samples[idx]

        image_t, label_t = self._load(subj, cls)

        # Sample context from other subjects that have this class
        candidates = [s for s in self.label_to_subjects[cls] if s != subj]
        random.shuffle(candidates)

        context_in: list[torch.Tensor] = []
        context_out: list[torch.Tensor] = []

        for ctx_subj in candidates:
            if len(context_in) >= self.context_size:
                break
            try:
                ctx_img, ctx_lbl = self._load(ctx_subj, cls)
                context_in.append(ctx_img)
                context_out.append(ctx_lbl)
            except Exception:
                continue

        # Pad by resampling if we don't have enough candidates
        while len(context_in) < self.context_size and len(context_in) > 0:
            i = random.randrange(len(context_in))
            context_in.append(context_in[i].clone())
            context_out.append(context_out[i].clone())

        if self.aug_cfg is not None and self.aug_cfg.enabled and len(context_in) > 0:
            # Stack query + context: (K+1, 1, D, H, W) and (K+1, D, H, W)
            all_images = torch.cat([image_t.unsqueeze(0), torch.stack(context_in)],  dim=0)
            all_masks  = torch.cat([label_t.unsqueeze(0), torch.stack(context_out)], dim=0)

            # Task aug: one set of geometric params for all volumes
            all_images, all_masks = apply_task_aug(all_images, all_masks, self.aug_cfg.task)

            # Intensity aug: independent params per volume (image only)
            for i in range(all_images.shape[0]):
                all_images[i] = apply_intensity_aug(all_images[i], self.aug_cfg.intensity)

            image_t    = all_images[0]          # (1, D, H, W)
            label_t    = all_masks[0]           # (D, H, W)
            context_in  = list(all_images[1:])  # K × (1, D, H, W)
            context_out = list(all_masks[1:])   # K × (D, H, W)

        return {
            "image":       image_t,                    # (1, D, H, W)
            "label":       label_t,                    # (D, H, W)  int64
            "context_in":  torch.stack(context_in),    # (K, 1, D, H, W)
            "context_out": torch.stack(context_out),   # (K, D, H, W)  int64
            "subject":     subj,
            "label_name":  cls,
        }

    # ------------------------------------------------------------------
    # Loading helpers
    # ------------------------------------------------------------------

    def _load(self, subj: str, cls: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Load and resize one (image, binary_mask) pair for a single class."""
        image = _load_ct(self.root / subj / "ct.nii.gz")            # (D,H,W) float32
        label = _build_label_volume(
            self.root / subj / "segmentations", [cls]
        )                                                             # (D,H,W) uint8 0/1

        image_t = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)  # (1,1,D,H,W)
        label_t = torch.from_numpy(label.astype(np.float32)).unsqueeze(0).unsqueeze(0)

        if self.image_size is not None:
            image_t, label_t = _resize_volume(image_t, label_t, self.image_size)

        image_t = image_t.squeeze(0)                   # (1, D, H, W)
        label_t = label_t.squeeze(0).squeeze(0).long() # (D, H, W)
        return image_t, label_t


# ---------------------------------------------------------------------------
# Collate function
# ---------------------------------------------------------------------------

def incontext_collate_fn(batch: list[dict]) -> dict:
    """Stack a list of dataset items into a batch dict."""
    return {
        "image":       torch.stack([b["image"]       for b in batch]),  # (B, 1, D, H, W)
        "label":       torch.stack([b["label"]       for b in batch]),  # (B, D, H, W)
        "context_in":  torch.stack([b["context_in"]  for b in batch]),  # (B, K, 1, D, H, W)
        "context_out": torch.stack([b["context_out"] for b in batch]),  # (B, K, D, H, W)
        "subjects":    [b["subject"]    for b in batch],
        "label_names": [b["label_name"] for b in batch],
    }


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
) -> DataLoader:
    ds = TotalSegInContextDataset(
        root=root,
        classes=classes,
        image_size=image_size,
        split=split,
        context_size=context_size,
        max_subjects=max_subjects,
        aug_cfg=aug_cfg,
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
