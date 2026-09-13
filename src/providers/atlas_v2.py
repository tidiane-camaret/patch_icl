"""ATLAS v2.0 volume provider for the in-context dataloader v2.

Chronic stroke lesion segmentation, T1 MRI, stored at NATIVE (uniform 197x233x189 @ 1mm/RAS)
spacing by scripts/convert_atlas_v2.py. Single class: `stroke_lesion`. Eval-only -- zero
overlap with TotalSegmentator's vocabulary (checked against data/totalseg_classes.py directly:
no lesion class of any kind exists there).

PROVENANCE CAVEAT (docs/datasets/atlas_v2.md #1): the official ATLAS v2.0 release is
DUA-gated; this source was converted from an UNOFFICIAL HuggingFace re-upload with no
README/license metadata, very likely an unauthorized redistribution of consent-controlled
patient data. Integrated on explicit user direction after this was flagged. Unlike every other
NativeGridProvider source in this repo, this one does not have a genuinely open provenance.
"""
from src.providers.native_grid import NativeGridProvider, resolve_classes_for

ATLAS_V2_CLASSES = ["stroke_lesion"]
ATLAS_V2_IDX = {name: i + 1 for i, name in enumerate(ATLAS_V2_CLASSES)}


def resolve_atlas_v2_classes(value) -> list[str]:
    """Resolve cfg.data.val_classes for the atlas_v2 source ("all" or an explicit list)."""
    return resolve_classes_for(ATLAS_V2_CLASSES, value, "atlas_v2")


class AtlasV2Provider(NativeGridProvider):
    SOURCE = "atlas_v2"
    ALL_CLASSES = ATLAS_V2_CLASSES
    CLASS_IDX = ATLAS_V2_IDX
    MODALITY = "mri"   # per-subject mri_stats normalization (native_grid.py), not the CT frame
