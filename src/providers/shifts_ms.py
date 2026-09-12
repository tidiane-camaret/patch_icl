"""Shifts-MS (Part 2, openly-licensed cohorts) volume provider for the in-context dataloader v2.

Multi-center MRI multiple-sclerosis lesion segmentation, stored at NATIVE (near-isotropic 1mm)
spacing by scripts/convert_shifts_ms.py. Single class: `ms_lesion`. Eval-only — zero overlap
with TotalSegmentator's vocabulary (checked against data/totalseg_classes.py directly: no
lesion class of any kind exists there).

Two things to know before reading numbers off this source (docs/datasets/shifts_ms.md):
  * The converted image is the FLAIR sequence only (t1/t2/pd/t1ce also ship in the source but
    aren't converted — the usual MS-lesion-conspicuous sequence).
  * Subjects span TWO cohorts with a genuine built-in domain-shift split: `best_train_*` /
    `best_dev_in_*` / `best_eval_in_*` (ISBI 2015) vs. `ljubljana_dev_out_*` (PubMRI,
    deliberately out-of-distribution relative to `best`). Class name is shared across both —
    filter subjects_for("ms_lesion") by the `best_`/`ljubljana_` prefix if a cohort-specific
    read is wanted.
"""
from src.providers.native_grid import NativeGridProvider, resolve_classes_for

SHIFTS_MS_CLASSES = ["ms_lesion"]
SHIFTS_MS_IDX = {name: i + 1 for i, name in enumerate(SHIFTS_MS_CLASSES)}


def resolve_shifts_ms_classes(value) -> list[str]:
    """Resolve cfg.data.val_classes for the shifts_ms source ("all" or an explicit list)."""
    return resolve_classes_for(SHIFTS_MS_CLASSES, value, "shifts_ms")


class ShiftsMsProvider(NativeGridProvider):
    SOURCE = "shifts_ms"
    ALL_CLASSES = SHIFTS_MS_CLASSES
    CLASS_IDX = SHIFTS_MS_IDX
    MODALITY = "mri"   # per-subject mri_stats normalization (native_grid.py), not the CT frame
