"""zanderch HU_Messung provider (`NativeGridProvider` subclass) -- local polytrauma whole-body
CT cohort, 36 cases with a single class: `l1_center`, a small (~15mm) HU-measurement ROI at
the L1 vertebral body centrum (not a vertebra segmentation). See
`docs/datasets/hu_lwk1.md`/`scripts/convert_hu_lwk1.py` for the full characterization.

CT modality (real HU units) -- the first CT (not MRI) native-grid eval source added to this
family; unlike ISLES22/Shifts-MS/MSD/ATLAS/GNC it uses the default `normalize_ct` global
fingerprint, no per-subject `ct_stats.json`. A single partitioning class needs none of
GncKidneyProvider's per-class-plane overrides -- this is a plain `NativeGridProvider` subclass.
"""
from src.providers.native_grid import NativeGridProvider, resolve_classes_for

HU_LWK1_CLASSES = ["l1_center"]
HU_LWK1_IDX = {"l1_center": 1}


def resolve_hu_lwk1_classes(value) -> list[str]:
    """Resolve cfg.data.val_classes for the hu_lwk1 source ("all" or an explicit list)."""
    return resolve_classes_for(HU_LWK1_CLASSES, value, "hu_lwk1")


class HuLwk1Provider(NativeGridProvider):
    SOURCE = "hu_lwk1"
    ALL_CLASSES = HU_LWK1_CLASSES
    CLASS_IDX = HU_LWK1_IDX
