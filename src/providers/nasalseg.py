"""NasalSeg volume provider for the in-context dataloader v2.

Head/sinus CT, 5 air-filled cavity classes, stored at NATIVE spacing
(0.586 x 0.586 x 1.5 mm, RAS) by scripts/convert_nasalseg.py — which also de-duplicates
130 files down to 107 unique cases. Eval-only.

Two things to know before reading numbers off this source (docs/datasets/nasalseg.md):
  * Targets are AIR (~ -840 HU) bounded by bone — inverted contrast polarity vs every
    soft-tissue organ in totalseg/flare22.
  * The volumes are small (FOV ~89 x 110 x 76 mm), so a 128^3 crop at 1.5 mm (192 mm)
    is larger than the head and air-pads heavily. Use a finer crop_spacing_mm.
"""
from src.providers.native_grid import NativeGridProvider, resolve_classes_for

# Label index -> name. Verified geometrically (the files carry no label metadata): 1/2 stand
# 12-15 mm clear of the mid-sagittal plane, 3/4 touch it (septum-adjacent), 5 is midline +
# posterior + inferior. No TotalSegmentator class corresponds to any of these.
NASALSEG_CLASSES = [
    "maxillary_sinus_right", "maxillary_sinus_left",
    "nasal_cavity_right", "nasal_cavity_left", "nasopharynx",
]
NASALSEG_IDX = {name: i + 1 for i, name in enumerate(NASALSEG_CLASSES)}


def resolve_nasalseg_classes(value) -> list[str]:
    """Resolve cfg.data.val_classes for the nasalseg source ("all" or an explicit list)."""
    return resolve_classes_for(NASALSEG_CLASSES, value, "nasalseg")


class NasalSegProvider(NativeGridProvider):
    SOURCE = "nasalseg"
    ALL_CLASSES = NASALSEG_CLASSES
    CLASS_IDX = NASALSEG_IDX
