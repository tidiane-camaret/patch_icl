"""
TotalSegmentator Dataset
- One item per subject: full 3-D CT volume + multi-class label volume
- Returns (1, D, H, W) image  +  (D, H, W) integer label
- Main params: classes (list of organ names), image_size (D, H, W)

Speed notes
-----------
NIfTI.gz requires full decompression on every load.  For production,
pre-convert to .npy (uncompressed) for near-instant random access.
"""

import csv
import random
from pathlib import Path
from typing import Callable, Optional

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from data.totalseg_classes import ALL_CLASSES  # noqa: F401  (re-exported for back-compat)

# CT normalisation constants matching nnUNet CTNormalization on this dataset.
# Derived from the 1228-subject fingerprint in nnUNet_preprocessed/dataset_fingerprint.json.
#   Clip:   p0.5 / p99.5 of foreground HU across all subjects.
#   Z-score: global mean / std of foreground HU (after clipping).
# Stored .npy files contain float16 z-score values; the runtime range is [-1.66, +3.44].
CT_CLIP_MIN: float = -1007.0
CT_CLIP_MAX: float =  1573.0
CT_MEAN:     float =  -167.3
CT_STD:      float =   505.8
# Pre-computed z-score bounds (used by augmentations for clamping).
CT_NORM_MIN: float = (CT_CLIP_MIN - CT_MEAN) / CT_STD   # ≈ -1.661
CT_NORM_MAX: float = (CT_CLIP_MAX - CT_MEAN) / CT_STD   # ≈ +3.441


# -------------------------------------------------------------------------
# Volume helpers
# -------------------------------------------------------------------------

def normalize_ct(vol: np.ndarray) -> np.ndarray:
    """Global CT normalization: clip to the HU window, then dataset z-score.

    Pointwise with fixed dataset-fingerprint constants — so normalizing a crop is identical
    to normalizing the whole volume (no need to load the full volume for stats). This is the
    shared helper used by convert_to_npy (writing ct.npy) and the raw-CT loader path."""
    vol = np.clip(vol.astype(np.float32), CT_CLIP_MIN, CT_CLIP_MAX)
    return (vol - CT_MEAN) / CT_STD


def mri_stats(vol: np.ndarray) -> dict:
    """Per-volume MRI normalization stats from the WHOLE volume: foreground percentile clip
    bounds + foreground mean/std. Computed once (at convert time) and stored in a sidecar so
    a crop can be normalized with whole-volume stats — crop-local stats would be inconsistent
    across target/context and across crops of the same subject."""
    fg = vol[vol > 0]
    if fg.size == 0:
        return {"clip_lo": 0.0, "clip_hi": 0.0, "mean": 0.0, "std": 1.0}
    lo = float(np.percentile(fg, 0.5))
    hi = float(np.percentile(fg, 99.5))
    clipped = np.clip(vol, lo, hi)
    fg2 = clipped[clipped > 0]
    mean, std = float(fg2.mean()), float(fg2.std())
    if std < 1e-6:
        std = 1.0
    return {"clip_lo": lo, "clip_hi": hi, "mean": mean, "std": std}


def normalize_mri(vol: np.ndarray, stats: dict) -> np.ndarray:
    """Apply precomputed whole-volume MRI stats (from mri_stats) to a volume or crop."""
    vol = np.clip(vol.astype(np.float32), stats["clip_lo"], stats["clip_hi"])
    return (vol - stats["mean"]) / stats["std"]


def _load_ct(path: Path, jitter: float = 0) -> np.ndarray:
    """Load CT, clip HU and z-score normalise.  Returns float32 (D,H,W).
    Prefers a pre-converted ct.npy next to the .nii.gz for fast loading.
    When jitter > 0 the npy cache is bypassed (it stores pre-normalised values,
    not raw HU) and clip boundaries are randomly perturbed by ±jitter HU."""
    if jitter == 0:
        npy = path.with_suffix("").with_suffix(".npy")  # ct.nii.gz → ct.npy
        if npy.exists():
            return np.load(npy, mmap_mode="r").astype(np.float32)
    vol = nib.as_closest_canonical(nib.load(str(path))).get_fdata(dtype=np.float32)
    a_min, a_max = CT_CLIP_MIN, CT_CLIP_MAX
    if jitter > 0:
        a_min += random.uniform(-jitter, jitter)
        a_max += random.uniform(-jitter, jitter)
        if a_max - a_min < 200:          # guard against degenerate window
            a_max = a_min + 200
    vol = np.clip(vol, a_min, a_max)
    vol = (vol - CT_MEAN) / CT_STD
    return vol  # (D, H, W)  z-score, nominally in [CT_NORM_MIN, CT_NORM_MAX]


# Maps ALL_CLASSES name → 1-based index used in label.npy
_ALL_CLASSES_IDX: dict[str, int] = {cls: i + 1 for i, cls in enumerate(ALL_CLASSES)}


def _build_label_volume(seg_dir: Path, classes: list[str]) -> np.ndarray:
    """
    Merge per-class binary masks into one integer label volume.
    Label value = class index + 1  (0 = background).

    Fast path: if label.npy exists in the subject dir (written by convert_to_npy.py),
    remap from the all-classes encoding to the requested subset in one vectorised pass.
    Slow path: load individual .nii.gz masks.
    """
    label_npy = seg_dir.parent / "label.npy"
    if label_npy.exists():
        full = np.load(label_npy, mmap_mode="r")   # (D,H,W) uint8, ALL_CLASSES encoding
        out = np.zeros(full.shape, dtype=np.uint8)
        for new_idx, cls in enumerate(classes, start=1):
            orig_idx = _ALL_CLASSES_IDX.get(cls)
            if orig_idx is not None:
                out[full == orig_idx] = new_idx
        return out

    # Slow path — individual .nii.gz masks
    label: Optional[np.ndarray] = None
    for cls_idx, cls in enumerate(classes, start=1):
        mask_path = seg_dir / f"{cls}.nii.gz"
        if not mask_path.exists():
            continue
        mask = (nib.as_closest_canonical(nib.load(str(mask_path))).get_fdata(dtype=np.float32) > 0).astype(np.uint8)
        if label is None:
            label = np.zeros_like(mask, dtype=np.uint8)
        label[mask > 0] = cls_idx
    if label is None:
        raise FileNotFoundError(f"No matching segmentation files in {seg_dir}")
    return label  # (D, H, W)


def _iso_size(native: tuple, target: int) -> tuple:
    """Scale longest axis to target, keep proportions (result fits within target³)."""
    scale = target / max(native)
    return tuple(min(target, max(1, round(s * scale))) for s in native)


def _resize_volume(
    image: torch.Tensor,   # (1, 1, D, H, W) float
    label: torch.Tensor,   # (1, 1, D, H, W) float
    size: tuple[int, int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Isotropic resize with separable Gaussian AA, then center-pad to a cubic target.

    Preserves aspect ratio: the longest axis is scaled to size[0]; shorter axes
    are scaled proportionally and then zero-padded to size[0].  A per-axis
    Gaussian blur (σ = 0.5*(s/n − 1)) is applied before downsampling to suppress
    aliasing.  Upsampled axes skip the blur.
    """
    T = size[0]
    D, H, W = image.shape[2:]
    new = _iso_size((D, H, W), T)

    x = image
    for dim, (s, n) in enumerate(zip((D, H, W), new)):
        sigma = max(0.0, 0.5 * (s / n - 1))
        if sigma < 0.1:
            continue
        k = max(3, 2 * round(2 * sigma) + 1)
        coords = torch.arange(k, dtype=x.dtype, device=x.device) - k // 2
        kernel = torch.exp(-0.5 * (coords / sigma) ** 2)
        kernel = (kernel / kernel.sum()).view([1, 1] + [k if i == dim else 1 for i in range(3)])
        pad_amt = k // 2
        p = [0] * 6
        ax = 2 - dim  # F.pad reverses axis order
        p[ax * 2] = pad_amt; p[ax * 2 + 1] = pad_amt
        x = F.conv3d(F.pad(x, p, mode="reflect"), kernel)

    image_small = F.interpolate(x, size=new, mode="trilinear", align_corners=False)
    label_small = F.interpolate(label, size=new, mode="nearest")

    image_out = torch.zeros(1, 1, T, T, T, dtype=image.dtype, device=image.device)
    label_out = torch.zeros(1, 1, T, T, T, dtype=label.dtype, device=label.device)
    pads = [(T - s) // 2 for s in new]
    sl = (slice(None), slice(None)) + tuple(slice(p, p + s) for p, s in zip(pads, new))
    image_out[sl] = image_small
    label_out[sl] = label_small
    return image_out, label_out


# -------------------------------------------------------------------------
# Dataset
# -------------------------------------------------------------------------

class TotalSegDataset(Dataset):
    """
    Full 3-D CT volumes from TotalSegmentator.

    Each item is a (image, label) pair:
      image : float32 tensor  (1, D, H, W)  normalised to [0, 1]
      label : int64 tensor    (D, H, W)     0 = background, 1…N = classes

    Args:
        root        : Dataset root directory (contains s0000/, s0001/, …).
        classes     : Organ names to segment.  Each gets label index i+1.
                      Use ALL_CLASSES for all 117 classes.
        image_size  : Output (D, H, W) after resizing.  Pass None to keep
                      native resolution (volumes will have different sizes —
                      incompatible with batch_size > 1).
        split       : 'train' | 'val' | 'test' | None (all subjects).
        meta_csv    : Path to meta.csv.  Auto-detected when split is given.
        max_subjects: Limit to first N subjects (useful for quick demos).
        transform   : Callable(image tensor, label tensor) → (image, label),
                      applied after resizing.
    """

    def __init__(
        self,
        root: str | Path,
        classes: list[str],
        image_size: Optional[tuple[int, int, int]] = (128, 256, 256),
        split: Optional[str] = None,
        meta_csv: Optional[str | Path] = None,
        max_subjects: Optional[int] = None,
        transform: Optional[Callable] = None,
    ):
        self.root = Path(root)
        self.classes = list(classes)
        self.image_size = image_size
        self.transform = transform

        self.subjects = self._get_subjects(split, meta_csv, max_subjects)
        print(f"TotalSegDataset: {len(self.subjects)} subjects", flush=True)

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _get_subjects(self, split, meta_csv, max_subjects) -> list[str]:
        all_subjects = sorted(p.name for p in self.root.iterdir() if p.is_dir())
        if split is not None:
            csv_path = Path(meta_csv) if meta_csv else self.root / "meta.csv"
            valid = set()
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
        return len(self.subjects)

    def __getitem__(self, idx: int):
        subj = self.subjects[idx]
        subj_dir = self.root / subj

        image = _load_ct(subj_dir / "ct.nii.gz")                    # (D, H, W) float32
        label = _build_label_volume(subj_dir / "segmentations",
                                    self.classes)                    # (D, H, W) uint8

        # Convert to tensors with batch+channel dims for F.interpolate
        image_t = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)  # (1,1,D,H,W)
        label_t = torch.from_numpy(label.astype(np.float32)).unsqueeze(0).unsqueeze(0)

        if self.image_size is not None:
            image_t, label_t = _resize_volume(image_t, label_t, self.image_size)

        # Remove batch dim; keep channel dim on image only
        image_t = image_t.squeeze(0)                                  # (1, D, H, W)
        label_t = label_t.squeeze(0).squeeze(0).long()                # (D, H, W)

        if self.transform is not None:
            image_t, label_t = self.transform(image_t, label_t)

        return image_t, label_t

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def num_classes(self) -> int:
        """Total number of classes including background (= 0)."""
        return len(self.classes) + 1

    def class_name(self, label_idx: int) -> str:
        return "background" if label_idx == 0 else self.classes[label_idx - 1]
