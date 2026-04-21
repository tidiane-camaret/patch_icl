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
from pathlib import Path
from typing import Callable, Optional

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

# -------------------------------------------------------------------------
# Full list of the 117 available segmentation classes
# -------------------------------------------------------------------------
ALL_CLASSES: list[str] = [
    "adrenal_gland_left", "adrenal_gland_right", "aorta", "atrial_appendage_left",
    "autochthon_left", "autochthon_right", "brachiocephalic_trunk",
    "brachiocephalic_vein_left", "brachiocephalic_vein_right", "brain",
    "clavicula_left", "clavicula_right", "colon", "common_carotid_artery_left",
    "common_carotid_artery_right", "costal_cartilages", "duodenum", "esophagus",
    "femur_left", "femur_right", "gallbladder", "gluteus_maximus_left",
    "gluteus_maximus_right", "gluteus_medius_left", "gluteus_medius_right",
    "gluteus_minimus_left", "gluteus_minimus_right", "heart", "hip_left", "hip_right",
    "humerus_left", "humerus_right", "iliac_artery_left", "iliac_artery_right",
    "iliac_vena_left", "iliac_vena_right", "iliopsoas_left", "iliopsoas_right",
    "inferior_vena_cava", "kidney_cyst_left", "kidney_cyst_right",
    "kidney_left", "kidney_right", "liver",
    "lung_lower_lobe_left", "lung_lower_lobe_right", "lung_middle_lobe_right",
    "lung_upper_lobe_left", "lung_upper_lobe_right",
    "pancreas", "portal_vein_and_splenic_vein", "prostate", "pulmonary_vein",
    "rib_left_1", "rib_left_2", "rib_left_3", "rib_left_4", "rib_left_5",
    "rib_left_6", "rib_left_7", "rib_left_8", "rib_left_9", "rib_left_10",
    "rib_left_11", "rib_left_12",
    "rib_right_1", "rib_right_2", "rib_right_3", "rib_right_4", "rib_right_5",
    "rib_right_6", "rib_right_7", "rib_right_8", "rib_right_9", "rib_right_10",
    "rib_right_11", "rib_right_12",
    "sacrum", "scapula_left", "scapula_right", "skull", "small_bowel",
    "spinal_cord", "spleen", "sternum", "stomach",
    "subclavian_artery_left", "subclavian_artery_right", "superior_vena_cava",
    "thyroid_gland", "trachea", "urinary_bladder",
    "vertebrae_C1", "vertebrae_C2", "vertebrae_C3", "vertebrae_C4",
    "vertebrae_C5", "vertebrae_C6", "vertebrae_C7",
    "vertebrae_L1", "vertebrae_L2", "vertebrae_L3", "vertebrae_L4", "vertebrae_L5",
    "vertebrae_S1",
    "vertebrae_T1", "vertebrae_T2", "vertebrae_T3", "vertebrae_T4", "vertebrae_T5",
    "vertebrae_T6", "vertebrae_T7", "vertebrae_T8", "vertebrae_T9",
    "vertebrae_T10", "vertebrae_T11", "vertebrae_T12",
]

# Soft-tissue Hounsfield-unit window → normalised to [0, 1]
HU_MIN, HU_MAX = -150, 250


# -------------------------------------------------------------------------
# Volume helpers
# -------------------------------------------------------------------------

def _load_ct(path: Path) -> np.ndarray:
    """Load CT, clip HU window, normalise to [0, 1].  Returns float32 (D,H,W).
    Prefers a pre-converted ct.npy next to the .nii.gz for fast loading."""
    npy = path.with_suffix("").with_suffix(".npy")  # ct.nii.gz → ct.npy
    if npy.exists():
        return np.load(npy, mmap_mode="r").astype(np.float32)
    vol = nib.load(str(path)).get_fdata(dtype=np.float32)
    vol = np.clip(vol, HU_MIN, HU_MAX)
    vol = (vol - HU_MIN) / (HU_MAX - HU_MIN)
    return vol  # (D, H, W)


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
        mask = (nib.load(str(mask_path)).get_fdata(dtype=np.float32) > 0).astype(np.uint8)
        if label is None:
            label = np.zeros_like(mask, dtype=np.uint8)
        label[mask > 0] = cls_idx
    if label is None:
        raise FileNotFoundError(f"No matching segmentation files in {seg_dir}")
    return label  # (D, H, W)


def _resize_volume(
    image: torch.Tensor,   # (1, 1, D, H, W) float
    label: torch.Tensor,   # (1, 1, D, H, W) float
    size: tuple[int, int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Resize both volumes to (D, H, W) using trilinear / nearest."""
    image = F.interpolate(image, size=size, mode="trilinear", align_corners=False)
    label = F.interpolate(label, size=size, mode="nearest")
    return image, label


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
