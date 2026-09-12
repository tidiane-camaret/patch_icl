"""MSD Task05_Prostate volume provider for the in-context dataloader v2.

Multi-parametric MRI (T2 + ADC) prostate zonal segmentation, stored at NATIVE spacing by
scripts/convert_msd_prostate.py. Two classes: `prostate_pz` (peripheral zone), `prostate_tz`
(transitional zone). Eval-only -- zero overlap with TotalSegmentator's vocabulary (checked
against data/totalseg_classes.py directly: only an undifferentiated `prostate` region exists
there, no zonal anatomy).

Design decision (docs/datasets/msd_prostate.md §7): the source is genuinely 4D (T2+ADC stacked
per case) and every provider in this harness is single-channel. Rather than add multi-channel
input support, **the converter splits each case's two channels into two independent
single-channel "subjects"** sharing the same label mask (`prostate_00_t2`, `prostate_00_adc`)
-- so this provider is structurally identical to every other `NativeGridProvider` subclass; the
channel split is entirely a converter-time decision, invisible here. 32 cases -> 64 converted
channel-subjects.

`crop_spacing_mm=0.75` was picked via a per-class label-extent clip-avoidance sweep (not a
whole-native-volume sweep like ISLES22/Shifts-MS/Hippocampus): the prostate is small relative
to the native in-plane FOV (median 200x200mm vs. a ~53x45x56mm median label extent), a
"find-in-scene" geometry closer to FLARE22/NasalSeg than to the other MRI sources. 0.75mm is
the smallest pitch with zero of the 62 (case, class) label instances clipped from their own
centroid.
"""
from src.providers.native_grid import NativeGridProvider, resolve_classes_for

MSD_PROSTATE_CLASSES = ["prostate_pz", "prostate_tz"]
MSD_PROSTATE_IDX = {name: i + 1 for i, name in enumerate(MSD_PROSTATE_CLASSES)}


def resolve_msd_prostate_classes(value) -> list[str]:
    """Resolve cfg.data.val_classes for the msd_prostate source ("all" or an explicit list)."""
    return resolve_classes_for(MSD_PROSTATE_CLASSES, value, "msd_prostate")


class MsdProstateProvider(NativeGridProvider):
    SOURCE = "msd_prostate"
    ALL_CLASSES = MSD_PROSTATE_CLASSES
    CLASS_IDX = MSD_PROSTATE_IDX
    MODALITY = "mri"   # per-subject (per-channel) mri_stats normalization (native_grid.py)
