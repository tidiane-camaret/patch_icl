"""ISLES 2022 volume provider for the in-context dataloader v2.

Multi-center MRI acute/subacute stroke lesion segmentation, stored at NATIVE (mildly
anisotropic) spacing by scripts/convert_isles22.py. Single class: `stroke_lesion`. Eval-only —
zero overlap with TotalSegmentator's vocabulary (checked against data/totalseg_classes.py
directly: no lesion class of any kind exists there).

Two things to know before reading numbers off this source (docs/datasets/isles22.md):
  * The converted image is the DWI sequence only (b=1000, the acute-stroke-sensitive one of
    the three co-registered sequences shipped — DWI/ADC/FLAIR); ADC/FLAIR are not converted.
  * 3/250 subjects have an entirely empty lesion mask in the source data (not a conversion
    bug — verified in the raw BIDS release). They convert fine (an all-zero label.npy) but
    the class-centroid cache naturally excludes them from `subjects_for("stroke_lesion")`,
    same mechanism NativeGridProvider already uses for any class absent in a given subject.
"""
from src.providers.native_grid import NativeGridProvider, resolve_classes_for

ISLES22_CLASSES = ["stroke_lesion"]
ISLES22_IDX = {name: i + 1 for i, name in enumerate(ISLES22_CLASSES)}


def resolve_isles22_classes(value) -> list[str]:
    """Resolve cfg.data.val_classes for the isles22 source ("all" or an explicit list)."""
    return resolve_classes_for(ISLES22_CLASSES, value, "isles22")


class Isles22Provider(NativeGridProvider):
    SOURCE = "isles22"
    ALL_CLASSES = ISLES22_CLASSES
    CLASS_IDX = ISLES22_IDX
    MODALITY = "mri"   # per-subject mri_stats normalization (native_grid.py), not the CT frame
