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
    # TotalSegMRI-only (indices 118–121): no direct CT equivalent
    "lung_left", "lung_right",     # whole-lung; CT has 5 lobe-level labels instead
    "intervertebral_discs",        # MRI-only merged disc label
    "vertebrae",                   # merged vertebrae; CT has per-level labels
]


MRI_ALL_CLASSES: list[str] = [
    # Solid organs
    "adrenal_gland_left", "adrenal_gland_right", "aorta", "brain",
    "duodenum", "esophagus", "gallbladder", "heart",
    "inferior_vena_cava", "kidney_left", "kidney_right", "liver",
    "pancreas", "portal_vein_and_splenic_vein", "prostate", "spleen",
    "stomach", "urinary_bladder",
    # Lungs (whole — no lobe-level labels in TotalSegMRI)
    "lung_left", "lung_right",
    # GI
    "colon", "small_bowel",
    # Vasculature
    "iliac_artery_left", "iliac_artery_right", "iliac_vena_left", "iliac_vena_right",
    # Bones
    "clavicula_left", "clavicula_right", "femur_left", "femur_right",
    "hip_left", "hip_right", "humerus_left", "humerus_right",
    "sacrum", "scapula_left", "scapula_right",
    # Muscles
    "autochthon_left", "autochthon_right",
    "gluteus_maximus_left", "gluteus_maximus_right",
    "gluteus_medius_left", "gluteus_medius_right",
    "gluteus_minimus_left", "gluteus_minimus_right",
    "iliopsoas_left", "iliopsoas_right",
    # Spine
    "spinal_cord", "intervertebral_discs", "vertebrae",
]


MRI_BENCHMARK_CLASSES: list[str] = [
    # Large solid organs
    "liver", "spleen", "kidney_left", "kidney_right", "heart", "pancreas",
    # GI
    "stomach", "colon", "small_bowel", "esophagus",
    # Whole lungs (MRI-specific — no lobe-level labels in TotalSegMRI)
    "lung_left", "lung_right",
    # Muscles
    "gluteus_maximus_left", "gluteus_medius_right", "autochthon_left", "iliopsoas_right",
    # Bones
    "femur_left", "hip_right", "humerus_left", "sacrum",
    # Vasculature
    "aorta", "inferior_vena_cava", "portal_vein_and_splenic_vein", "iliac_artery_left",
    # Spine
    "spinal_cord", "intervertebral_discs", "vertebrae",
    # Near-zero sentinels
    "adrenal_gland_left", "prostate", "urinary_bladder",
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
    is_mri: bool = False,
) -> list[str]:
    """Resolve a class list from a Hydra config value.

    If *value* is already a list (or OmegaConf ListConfig), return it as a plain list.
    Special string values (CT by default; pass ``is_mri=True`` for MRI variants):
      ``"benchmark"``     → BENCHMARK_CLASSES / MRI_BENCHMARK_CLASSES
      ``"not_benchmark"`` → ALL_CLASSES[:117] / MRI_ALL_CLASSES minus the benchmark set
    Otherwise read ``{totalseg_root}/label_stats.csv`` and return classes whose
    ``split`` column matches *value* (e.g. ``"train"`` / ``"val"``).

    Args:
        value:          cfg.data.train_classes or cfg.data.val_classes.
        totalseg_root:  Path to the dataset root. Required for split-name strings.
        is_mri:         Set True when cfg.data.dataset == "totalsegmri".
    """
    if not isinstance(value, str):
        return list(value)

    if value == "benchmark":
        return list(MRI_BENCHMARK_CLASSES if is_mri else BENCHMARK_CLASSES)

    if value == "not_benchmark":
        if is_mri:
            bench_set = set(MRI_BENCHMARK_CLASSES)
            return [c for c in MRI_ALL_CLASSES if c not in bench_set]
        bench_set = set(BENCHMARK_CLASSES)
        return [c for c in ALL_CLASSES[:117] if c not in bench_set]

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


# Category mappings (only for TotalSeg datasets)
category_map_ct = {
    # --- ORGANS (ABDOMINAL & PELVIC) ---
    "esophagus": "Organs (Abd/Pelvis)",
    "stomach": "Organs (Abd/Pelvis)",
    "duodenum": "Organs (Abd/Pelvis)",
    "small_bowel": "Organs (Abd/Pelvis)",
    "colon": "Organs (Abd/Pelvis)",
    "liver": "Organs (Abd/Pelvis)",
    "gallbladder": "Organs (Abd/Pelvis)",
    "pancreas": "Organs (Abd/Pelvis)",
    "spleen": "Organs (Abd/Pelvis)",
    "kidney_left": "Organs (Abd/Pelvis)",
    "kidney_right": "Organs (Abd/Pelvis)",
    "urinary_bladder": "Organs (Abd/Pelvis)",
    "prostate": "Organs (Abd/Pelvis)",
    "adrenal_gland_right": "Organs (Abd/Pelvis)",
    "adrenal_gland_left": "Organs (Abd/Pelvis)",
    "kidney_cyst_left": "Organs (Abd/Pelvis)",
    "kidney_cyst_right": "Organs (Abd/Pelvis)",
    # --- ORGANS (THORAX, HEAD & SPINE) ---
    "heart": "Organs (Thorax/Head/Spine)",
    "lung_upper_lobe_left": "Organs (Thorax/Head/Spine)",
    "lung_lower_lobe_left": "Organs (Thorax/Head/Spine)",
    "lung_upper_lobe_right": "Organs (Thorax/Head/Spine)",
    "lung_middle_lobe_right": "Organs (Thorax/Head/Spine)",
    "lung_lower_lobe_right": "Organs (Thorax/Head/Spine)",
    "trachea": "Organs (Thorax/Head/Spine)",
    "thyroid_gland": "Organs (Thorax/Head/Spine)",
    "brain": "Organs (Thorax/Head/Spine)",
    "spinal_cord": "Organs (Thorax/Head/Spine)",
    "atrial_appendage_left": "Organs (Thorax/Head/Spine)",
    # --- BONES (SPINE) ---
    "vertebrae_C7": "Bones (Spine)",
    "vertebrae_T1": "Bones (Spine)",
    "vertebrae_T2": "Bones (Spine)",
    "vertebrae_T3": "Bones (Spine)",
    "vertebrae_T4": "Bones (Spine)",
    "vertebrae_T5": "Bones (Spine)",
    "vertebrae_T6": "Bones (Spine)",
    "vertebrae_T7": "Bones (Spine)",
    "vertebrae_T8": "Bones (Spine)",
    "vertebrae_T9": "Bones (Spine)",
    "vertebrae_T10": "Bones (Spine)",
    "vertebrae_T11": "Bones (Spine)",
    "vertebrae_T12": "Bones (Spine)",
    "vertebrae_L1": "Bones (Spine)",
    "vertebrae_L2": "Bones (Spine)",
    "vertebrae_L3": "Bones (Spine)",
    "vertebrae_L4": "Bones (Spine)",
    "vertebrae_L5": "Bones (Spine)",
    "vertebrae_S1": "Bones (Spine)",
    "sacrum": "Bones (Spine)",
    "vertebrae_C1": "Bones (Spine)",
    "vertebrae_C2": "Bones (Spine)",
    "vertebrae_C3": "Bones (Spine)",
    "vertebrae_C4": "Bones (Spine)",
    "vertebrae_C5": "Bones (Spine)",
    "vertebrae_C6": "Bones (Spine)",
    # --- BONES (RIBS & STERNUM) ---
    "sternum": "Bones (Ribs/Sternum)",
    "costal_cartilages": "Bones (Ribs/Sternum)",
    "rib_left_1": "Bones (Ribs/Sternum)",
    "rib_left_2": "Bones (Ribs/Sternum)",
    "rib_left_3": "Bones (Ribs/Sternum)",
    "rib_left_4": "Bones (Ribs/Sternum)",
    "rib_left_5": "Bones (Ribs/Sternum)",
    "rib_left_6": "Bones (Ribs/Sternum)",
    "rib_left_7": "Bones (Ribs/Sternum)",
    "rib_left_8": "Bones (Ribs/Sternum)",
    "rib_left_9": "Bones (Ribs/Sternum)",
    "rib_left_10": "Bones (Ribs/Sternum)",
    "rib_left_11": "Bones (Ribs/Sternum)",
    "rib_right_1": "Bones (Ribs/Sternum)",
    "rib_right_2": "Bones (Ribs/Sternum)",
    "rib_right_3": "Bones (Ribs/Sternum)",
    "rib_right_4": "Bones (Ribs/Sternum)",
    "rib_right_5": "Bones (Ribs/Sternum)",
    "rib_right_6": "Bones (Ribs/Sternum)",
    "rib_right_7": "Bones (Ribs/Sternum)",
    "rib_right_8": "Bones (Ribs/Sternum)",
    "rib_right_9": "Bones (Ribs/Sternum)",
    "rib_right_10": "Bones (Ribs/Sternum)",
    "rib_right_11": "Bones (Ribs/Sternum)",
    "rib_left_12": "Bones (Ribs/Sternum)",
    "rib_right_12": "Bones (Ribs/Sternum)",
    # --- BONES (LIMBS, SHOULDER & PELVIS) ---
    "skull": "Bones (Limbs/Shoulder/Pelvis)",
    "clavicula_left": "Bones (Limbs/Shoulder/Pelvis)",
    "clavicula_right": "Bones (Limbs/Shoulder/Pelvis)",
    "scapula_left": "Bones (Limbs/Shoulder/Pelvis)",
    "scapula_right": "Bones (Limbs/Shoulder/Pelvis)",
    "humerus_left": "Bones (Limbs/Shoulder/Pelvis)",
    "humerus_right": "Bones (Limbs/Shoulder/Pelvis)",
    "hip_left": "Bones (Limbs/Shoulder/Pelvis)",
    "hip_right": "Bones (Limbs/Shoulder/Pelvis)",
    "femur_left": "Bones (Limbs/Shoulder/Pelvis)",
    "femur_right": "Bones (Limbs/Shoulder/Pelvis)",
    # --- MUSCLES ---
    "autochthon_left": "Muscles",
    "autochthon_right": "Muscles",
    "iliopsoas_left": "Muscles",
    "iliopsoas_right": "Muscles",
    "gluteus_maximus_left": "Muscles",
    "gluteus_maximus_right": "Muscles",
    "gluteus_medius_left": "Muscles",
    "gluteus_medius_right": "Muscles",
    "gluteus_minimus_left": "Muscles",
    "gluteus_minimus_right": "Muscles",
    # --- VESSELS ---
    "aorta": "Vessels",
    "iliac_artery_left": "Vessels",
    "iliac_artery_right": "Vessels",
    "subclavian_artery_left": "Vessels",
    "subclavian_artery_right": "Vessels",
    "superior_vena_cava": "Vessels",
    "inferior_vena_cava": "Vessels",
    "brachiocephalic_vein_left": "Vessels",
    "iliac_vena_left": "Vessels",
    "iliac_vena_right": "Vessels",
    "pulmonary_vein": "Vessels",
    "portal_vein_and_splenic_vein": "Vessels",
    "common_carotid_artery_left": "Vessels",
    "common_carotid_artery_right": "Vessels",
    "brachiocephalic_vein_right": "Vessels",
    "brachiocephalic_trunk": "Vessels",
}

category_map_mri = {
    # --- ORGANS (ABDOMINAL & PELVIC) ---
    "esophagus": "Organs (Abd/Pelvis)",
    "stomach": "Organs (Abd/Pelvis)",
    "duodenum": "Organs (Abd/Pelvis)",
    "small_bowel": "Organs (Abd/Pelvis)",
    "colon": "Organs (Abd/Pelvis)",
    "liver": "Organs (Abd/Pelvis)",
    "gallbladder": "Organs (Abd/Pelvis)",
    "pancreas": "Organs (Abd/Pelvis)",
    "spleen": "Organs (Abd/Pelvis)",
    "kidney_left": "Organs (Abd/Pelvis)",
    "kidney_right": "Organs (Abd/Pelvis)",
    "urinary_bladder": "Organs (Abd/Pelvis)",
    "prostate": "Organs (Abd/Pelvis)",
    "adrenal_gland_left": "Organs (Abd/Pelvis)",
    "adrenal_gland_right": "Organs (Abd/Pelvis)",
    # --- ORGANS (THORAX & HEAD/SPINE) ---
    "heart": "Organs (Thorax/Head/Spine)",
    "lung_left": "Organs (Thorax/Head/Spine)",
    "lung_right": "Organs (Thorax/Head/Spine)",
    "brain": "Organs (Thorax/Head/Spine)",
    "spinal_cord": "Organs (Thorax/Head/Spine)",
    # --- BONES (SPINE) ---
    "vertebrae": "Bones (Spine)",
    "intervertebral_discs": "Bones (Spine)",
    "sacrum": "Bones (Spine)",
    # --- BONES (LIMBS & PELVIS) ---
    "hip_left": "Bones (Limbs/Pelvis)",
    "hip_right": "Bones (Limbs/Pelvis)",
    "femur_left": "Bones (Limbs/Pelvis)",
    "femur_right": "Bones (Limbs/Pelvis)",
    "humerus_left": "Bones (Limbs/Pelvis)",
    "humerus_right": "Bones (Limbs/Pelvis)",
    "tibia": "Bones (Limbs/Pelvis)",
    "fibula": "Bones (Limbs/Pelvis)",
    # --- MUSCLES (TRUNK) ---
    "autochthon_left": "Muscles (Trunk)",
    "autochthon_right": "Muscles (Trunk)",
    "iliopsoas_left": "Muscles (Trunk)",
    "iliopsoas_right": "Muscles (Trunk)",
    "gluteus_maximus_left": "Muscles (Trunk)",
    "gluteus_maximus_right": "Muscles (Trunk)",
    "gluteus_medius_left": "Muscles (Trunk)",
    "gluteus_medius_right": "Muscles (Trunk)",
    "gluteus_minimus_left": "Muscles (Trunk)",
    "gluteus_minimus_right": "Muscles (Trunk)",
    # --- MUSCLES (THIGH) ---
    "quadriceps_femoris_left": "Muscles (Thigh)",
    "quadriceps_femoris_right": "Muscles (Thigh)",
    "sartorius_left": "Muscles (Thigh)",
    "sartorius_right": "Muscles (Thigh)",
    "thigh_medial_compartment_left": "Muscles (Thigh)",
    "thigh_medial_compartment_right": "Muscles (Thigh)",
    "thigh_posterior_compartment_left": "Muscles (Thigh)",
    "thigh_posterior_compartment_right": "Muscles (Thigh)",
    # --- VESSELS ---
    "aorta": "Vessels",
    "iliac_artery_left": "Vessels",
    "iliac_artery_right": "Vessels",
    "inferior_vena_cava": "Vessels",
    "iliac_vena_left": "Vessels",
    "iliac_vena_right": "Vessels",
    "portal_vein_and_splenic_vein": "Vessels",
}

