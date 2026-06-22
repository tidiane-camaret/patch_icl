"""
TotalSegmentator 2D in-context dataset.

Reads the npz written by scripts/totalseg2d/to_npz.py (one axial cross-section per
subject, raw int16 HU at a fixed mm/pixel). Mirrors MedSegBenchDataset's interface so
it drops into the 2D pipeline via common.build_dataset (source="totalseg2d"):
__getitem__ returns {image, label, context_in, context_out} and the object exposes
`.samples` / `.label_index` for TaggedDataset and in-context context sampling.

Differences from MedSegBench (which stores uint8 [0,255]):
  - Images are raw HU (int16). Normalization is done here: clip to `hu_window` then
    min-max to [0,1] (deferred from export for flexibility — see logs).
  - One "dataset" (totalseg2d); label_value is the TotalSeg class index (1..117).
  - The export stores at `stored_size` px (default 256, 512mm FOV). We resize to the
    requested image_size, so model resolution is decoupled from the stored FOV (the
    128px export is a tighter 256mm FOV — avoid it; resize the 256 one instead).
  - `min_area`: a (subject, class) pair becomes a sample only if the class covers at
    least this many pixels at image_size, so tiny slivers aren't degenerate targets.
"""

import os
import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from src.datasets.cow_index import SampleIndex, build_candidate_index, sample_context

DATA_ROOT = ("/nfs/data/nii/data1/Analysis/camaret___in_context_segmentation"
             "/ANALYSIS_20251122/data/totalseg2d")


def _window_to_unit(img_hu: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Clip HU to [lo, hi] and min-max to [0, 1] float32."""
    return np.clip((img_hu.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)


def _resize(images: np.ndarray, labels: np.ndarray, size: int) -> Tuple[np.ndarray, np.ndarray]:
    """Resize (N,H,W) float images (bilinear) and uint8 labels (nearest) to size."""
    im = F.interpolate(torch.from_numpy(images)[:, None], size=(size, size),
                       mode="bilinear", align_corners=False)[:, 0].numpy()
    lb = F.interpolate(torch.from_numpy(labels.astype(np.float32))[:, None],
                       size=(size, size), mode="nearest")[:, 0].numpy().astype(np.uint8)
    return im, lb


def _to_tensor(arr: np.ndarray) -> torch.Tensor:
    """float [0,1] image (H,W) → [1,H,W] tensor (already normalized at load)."""
    return torch.from_numpy(arr).unsqueeze(0)


def _binary_mask_tensor(labels: np.ndarray, label_value: int) -> torch.Tensor:
    """Binary mask for one label value → [1,H,W] float tensor."""
    return torch.from_numpy((labels == label_value).astype(np.float32)).unsqueeze(0)


class TotalSeg2DDataset(Dataset):
    """In-context TotalSegmentator 2D dataset (see module docstring)."""

    def __init__(
        self,
        split: str = "train",
        context_size: int = 3,
        image_size: int = 128,
        data_root: str = DATA_ROOT,
        stored_size: int = 256,
        hu_window: Tuple[float, float] = (-1000.0, 1000.0),
        min_area: int = 16,
        datasets: Optional[List[str]] = None,   # accepted for API parity; ignored
        deterministic: Optional[bool] = None,
    ):
        self.split = split
        self.context_size = context_size
        self.image_size = image_size
        # Non-train splits get reproducible context (seeded from idx) so eval Dice
        # isn't perturbed by context-sampling variance across runs/epochs.
        self.deterministic = (split != "train") if deterministic is None else deterministic

        path = os.path.join(data_root, f"totalseg2d_{stored_size}.npz")
        npz = np.load(path, allow_pickle=True)
        if f"{split}_images" not in npz.files:
            raise KeyError(f"split '{split}' not in {path}")

        images = _window_to_unit(npz[f"{split}_images"], *hu_window)   # (N,S,S) [0,1]
        labels = npz[f"{split}_label"]                                  # (N,S,S) uint8
        if stored_size != image_size:
            images, labels = _resize(images, labels, image_size)

        # One dataset; arrays indexed directly (sample_idx). ds_name kept for parity
        # with MedSegBench so TaggedDataset / the eval+train scripts work unchanged.
        name = "totalseg2d"
        self.name = name
        self.images: Dict[str, np.ndarray] = {name: images}
        self.labels: Dict[str, np.ndarray] = {name: labels}
        self.subjects = npz[f"{split}_subjects"]
        self.class_names = list(npz["class_names"])

        # COW-safe index (single dataset → ds_id 0). See src/datasets/cow_index.py.
        _ds_ids: List[int] = []
        _img_idxs: List[int] = []
        _label_values: List[int] = []
        print(f"Loading TotalSeg2D (size={image_size}, split={split}, "
              f"stored={stored_size}px, window={hu_window}, min_area={min_area})...")
        for i in range(len(images)):
            vals, cnts = np.unique(labels[i], return_counts=True)
            for lv, c in zip(vals, cnts):
                if lv == 0 or c < min_area:
                    continue
                _ds_ids.append(0)
                _img_idxs.append(i)
                _label_values.append(int(lv))
        self.samples = SampleIndex(_ds_ids, _img_idxs, _label_values, [name])
        self._cand = build_candidate_index(_ds_ids, _img_idxs, _label_values)
        n_classes = len(set(_label_values))
        print(f"Total: {len(self.samples)} samples over {len(images)} slices, "
              f"{n_classes} classes")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        si = self.samples
        ds = si.ds_names[int(si.ds_ids[idx])]
        sample_idx  = int(si.img_idxs[idx])
        label_value = int(si.label_values[idx])
        image = self.images[ds][sample_idx]
        mask = self.labels[ds][sample_idx]

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
            "context_in": context_in,
            "context_out": context_out,
        }
