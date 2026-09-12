"""MSD Task04_Hippocampus volume provider for the in-context dataloader v2.

T1 MRI hippocampus anterior/posterior segmentation, stored at NATIVE (isotropic 1mm) spacing
by scripts/convert_msd_hippocampus.py. Two classes: `hippocampus_anterior`,
`hippocampus_posterior`. Eval-only -- zero overlap with TotalSegmentator's vocabulary (checked
against data/totalseg_classes.py directly: no hippocampus or any brain substructure exists
there, only undifferentiated `brain`).

One thing that's different from every other NativeGridProvider source
(docs/datasets/msd_hippocampus.md): the native volumes are ALREADY ROI-cropped to a tiny box
around the hippocampus (median ~35x50x36 vox at 1mm, i.e. a ~35-50mm physical FOV) -- not a
whole organ/head/torso. A round-trip grid-occupancy sweep (T=128, all 260 training shapes)
picked `crop_spacing_mm=0.5`: the smallest pitch with ZERO subjects clipped on ANY axis
(0.45mm already clips 1/260 on one axis), giving 55-78% mean grid fill. Every other source in
the harness uses `crop_spacing_mm >= 0.6` -- this is the harness's first sub-mm crop pitch.
"""
from src.providers.native_grid import NativeGridProvider, resolve_classes_for

MSD_HIPPOCAMPUS_CLASSES = ["hippocampus_anterior", "hippocampus_posterior"]
MSD_HIPPOCAMPUS_IDX = {name: i + 1 for i, name in enumerate(MSD_HIPPOCAMPUS_CLASSES)}


def resolve_msd_hippocampus_classes(value) -> list[str]:
    """Resolve cfg.data.val_classes for the msd_hippocampus source ("all" or an explicit list)."""
    return resolve_classes_for(MSD_HIPPOCAMPUS_CLASSES, value, "msd_hippocampus")


class MsdHippocampusProvider(NativeGridProvider):
    SOURCE = "msd_hippocampus"
    ALL_CLASSES = MSD_HIPPOCAMPUS_CLASSES
    CLASS_IDX = MSD_HIPPOCAMPUS_IDX
    MODALITY = "mri"   # per-subject mri_stats normalization (native_grid.py), not the CT frame
