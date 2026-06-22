"""
MedSegBench in-context dataset.

Loads all .npz files at a given image_size into RAM. Each __getitem__ returns
a target image/mask plus K context pairs sharing the same label.

File convention: {dataset}_{size}.npz  (e.g. abdomenus_128.npz)
Keys inside:     {split}_images, {split}_label
Special case:    {split}_label_C1, _C2, ... (idrib) → merged into single label map
"""

import glob
import os
import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from src.datasets.cow_index import SampleIndex, build_candidate_index, sample_context

DATA_ROOT = "/nfs/data/nii/data1/Analysis/camaret___in_context_segmentation/ANALYSIS_20251122/data/medsegbench"


def _load_images_and_labels(npz, split: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract images and label array from an open npz for a given split.
    Returns (images [N,H,W], labels [N,H,W]) as uint8 arrays.
    Raises KeyError if the split is not present.
    """
    keys = list(npz.keys())

    if f"{split}_images" not in keys:
        raise KeyError(f"Split '{split}' not found")

    images = npz[f"{split}_images"]

    # Grayscale: (N,H,W) → keep; RGB: (N,H,W,3) → average channels
    if images.ndim == 4:
        images = images.mean(axis=-1).astype(np.uint8)

    # Standard label key
    if f"{split}_label" in keys:
        labels = npz[f"{split}_label"]
        return images, labels

    # Per-class keys: {split}_label_C1, _C2, ...
    class_keys = sorted(k for k in keys if k.startswith(f"{split}_label_C"))
    if class_keys:
        labels = np.zeros(images.shape[:3], dtype=np.uint8)
        for ci, ck in enumerate(class_keys, start=1):
            labels[npz[ck] > 0] = ci
        return images, labels

    raise KeyError(f"No label keys found for split '{split}'")


def _to_tensor(arr: np.ndarray) -> torch.Tensor:
    """Normalize uint8 image to [0,1] float and add channel dim → [1,H,W]."""
    return torch.from_numpy(arr.astype(np.float32) / 255.0).unsqueeze(0)


def _binary_mask_tensor(labels: np.ndarray, label_value: int) -> torch.Tensor:
    """Binary mask for one label value → [1,H,W] float tensor."""
    return torch.from_numpy((labels == label_value).astype(np.float32)).unsqueeze(0)


class MedSegBenchDataset(Dataset):
    """
    In-context MedSegBench dataset.

    Args:
        split: 'train', 'val', or 'test'
        context_size: number of context (image, mask) pairs per sample
        image_size: spatial resolution to load; picks the matching {name}_{size}.npz files
        data_root: path to the medsegbench directory
        datasets: list of dataset names to load (without size suffix); None loads all
    """

    def __init__(
        self,
        split: str = "train",
        context_size: int = 3,
        image_size: int = 128,
        data_root: str = DATA_ROOT,
        datasets: Optional[List[str]] = None,
        deterministic: Optional[bool] = None,
    ):
        self.split = split
        self.context_size = context_size
        self.image_size = image_size
        # Non-train splits get reproducible context (seeded from idx) so eval Dice
        # isn't perturbed by context-sampling variance across runs/epochs.
        self.deterministic = (split != "train") if deterministic is None else deterministic

        # Discover npz files for the requested size
        if datasets is not None:
            npz_files = [
                os.path.join(data_root, f"{name}_{image_size}.npz")
                for name in datasets
            ]
        else:
            npz_files = sorted(glob.glob(os.path.join(data_root, f"*_{image_size}.npz")))

        # Load all data into RAM
        # self.images[ds] : np.ndarray [N, H, W] uint8
        # self.labels[ds] : np.ndarray [N, H, W] uint8
        self.images: Dict[str, np.ndarray] = {}
        self.labels: Dict[str, np.ndarray] = {}

        # COW-safe flat index, built from parallel int lists below (see cow_index).
        # ds_id indexes self._ds_names; one row per (image, present label_value).
        self._ds_names: List[str] = []
        _ds_ids: List[int] = []
        _img_idxs: List[int] = []
        _label_values: List[int] = []

        print(f"Loading MedSegBench (size={image_size}, split={split})...")
        for path in npz_files:
            name = os.path.basename(path).replace(f"_{image_size}.npz", "")
            if not os.path.exists(path):
                print(f"  [skip] {name}: file not found")
                continue
            try:
                npz = np.load(path)
                images, labels = _load_images_and_labels(npz, split)
            except KeyError as e:
                print(f"  [skip] {name}: {e}")
                continue
            except Exception as e:
                print(f"  [error] {name}: {e}")
                continue

            self.images[name] = images
            self.labels[name] = labels

            ds_id = len(self._ds_names)
            self._ds_names.append(name)
            for i in range(len(images)):
                for lv in np.unique(labels[i]):
                    if lv != 0:
                        _ds_ids.append(ds_id)
                        _img_idxs.append(i)
                        _label_values.append(int(lv))

            print(f"  [ok] {name}: {len(images)} samples, labels {np.unique(labels).tolist()}")

        # COW-safe index + per-(ds, label) candidate arrays for context lookup.
        self.samples = SampleIndex(_ds_ids, _img_idxs, _label_values, self._ds_names)
        self._cand = build_candidate_index(_ds_ids, _img_idxs, _label_values)
        print(f"Total: {len(self.samples)} samples from {len(self.images)} datasets")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns:
            image      [1, H, W]  – target image, normalised to [0,1]
            label      [1, H, W]  – target binary mask
            context_in [K, 1, H, W]
            context_out[K, 1, H, W]
        """
        si = self.samples
        ds = si.ds_names[int(si.ds_ids[idx])]
        sample_idx  = int(si.img_idxs[idx])
        label_value = int(si.label_values[idx])
        image = self.images[ds][sample_idx]
        mask = self.labels[ds][sample_idx]

        # Sample K context indices (same dataset, same label, different index).
        rng = random.Random(idx) if self.deterministic else None
        cand = self._cand.get((int(si.ds_ids[idx]), label_value))
        ctx = sample_context(cand, sample_idx, self.context_size, rng) if cand is not None \
            else np.empty(0, dtype=np.int32)

        context_in = torch.stack([
            _to_tensor(self.images[ds][int(i)]) for i in ctx
        ]) if len(ctx) else torch.zeros(0, 1, self.image_size, self.image_size)

        context_out = torch.stack([
            _binary_mask_tensor(self.labels[ds][int(i)], label_value) for i in ctx
        ]) if len(ctx) else torch.zeros(0, 1, self.image_size, self.image_size)

        return {
            "image": _to_tensor(image),
            "label": _binary_mask_tensor(mask, label_value),
            "context_in": context_in,    # [K, 1, H, W]
            "context_out": context_out,  # [K, 1, H, W]
        }
