"""
Full list of the 117 TotalSegmentator segmentation classes, in the canonical order
used by label.npy (index = position + 1, background = 0).

Also provides resolve_classes(), which lets config values be either an explicit list
or a split name ("train" / "val") resolved against label_stats.csv.
"""

import csv
from pathlib import Path
from typing import Union

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


BENCHMARK_CLASSES: list[str] = [
    # Large solid organs
    "liver", "spleen", "kidney_left", "kidney_right", "heart", "pancreas", "gallbladder",
    # GI tract
    "stomach", "colon", "small_bowel", "esophagus",
    # Lungs
    "lung_lower_lobe_left", "lung_upper_lobe_right", "lung_middle_lobe_right",
    # Muscles
    "gluteus_maximus_left", "gluteus_medius_right", "autochthon_left", "iliopsoas_right",
    # Appendicular bones
    "femur_left", "hip_right", "humerus_left", "clavicula_right", "scapula_left",
    # Axial bones
    "sacrum", "sternum", "skull", "costal_cartilages",
    # Vertebrae (one per region)
    "vertebrae_C3", "vertebrae_T6", "vertebrae_L3", "vertebrae_S1",
    # Ribs
    "rib_left_6", "rib_right_9",
    # Vasculature
    "aorta", "inferior_vena_cava", "portal_vein_and_splenic_vein", "iliac_artery_left",
    # Neuro / airway / pelvis
    "brain", "spinal_cord", "trachea", "urinary_bladder", "thyroid_gland",
    # Near-zero sentinels
    "adrenal_gland_left", "atrial_appendage_left", "common_carotid_artery_left",
    "pulmonary_vein", "brachiocephalic_trunk",
]


def resolve_classes(
    value: Union[str, list],
    totalseg_root: Union[str, Path, None] = None,
) -> list[str]:
    """Resolve a class list from a Hydra config value.

    If *value* is already a list (or OmegaConf ListConfig), return it as a plain list.
    If *value* is ``"benchmark"``, return BENCHMARK_CLASSES.
    If *value* is a string such as ``"train"`` or ``"val"``, read
    ``{totalseg_root}/label_stats.csv`` and return the classes whose ``split``
    column matches that string.

    Args:
        value:          cfg.data.train_classes or cfg.data.val_classes.
        totalseg_root:  Path to the TotalSegmentator data root (cfg.paths.totalseg).
                        Required when value is a string.
    """
    if not isinstance(value, str):
        return list(value)

    if value == "benchmark":
        return list(BENCHMARK_CLASSES)

    if totalseg_root is None:
        raise ValueError("totalseg_root must be provided when train/val_classes is a split name")

    csv_path = Path(totalseg_root) / "label_stats.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"label_stats.csv not found at {csv_path}")

    classes = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if row["split"].strip() == value:
                classes.append(row["label_id"].strip())
    if not classes:
        raise ValueError(f"No classes found for split '{value}' in {csv_path}")
    return classes
