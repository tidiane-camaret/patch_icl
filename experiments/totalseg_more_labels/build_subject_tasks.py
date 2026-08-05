"""Build a per-subject list of relevant TotalSegmentator `more_labels` tasks.

A task is region-specific: running e.g. `brain_structures` on an abdomen scan just
yields empty masks. We gate each task on the subject's coarse `total` footprint
(the 122 classes recorded in label.npy, read exactly like the dataloader's scan
cache: np.unique(label.npy) -> _IDX_TO_CLASS). If none of a task's anchor classes
are present, the task is dropped for that subject.

Anchors are deliberately coarse (a task's body REGION, not its fine classes, which
mostly aren't in ALL_CLASSES). Output: totalseg_test_subject_tasks.json.
"""

import json
from pathlib import Path

import numpy as np

from src.totalseg_dataset import _ALL_CLASSES_IDX

IDX_TO_CLASS = {v: k for k, v in _ALL_CLASSES_IDX.items()}

DATA = Path("/nfs/data/nii/data1/Analysis/camaret___in_context_segmentation/ANALYSIS_20251122/data")
ROOT = DATA / "totalseg"
META = DATA / "meta.csv"
TASKS_JSON = Path(__file__).parent / "totalseg_ct_tasks.json"
OUT = Path(__file__).parent / "totalseg_test_subject_tasks.json"

# ---------------------------------------------------------------------------
# Coarse anatomical region groups, expressed in `total` (122-class) names.
# ---------------------------------------------------------------------------
HEAD    = {"brain", "skull"}
CERV    = {f"vertebrae_C{i}" for i in range(1, 8)}          # cervical spine ~ neck
NECK    = CERV | {"thyroid_gland", "trachea"}
LUNG    = {"lung_upper_lobe_left", "lung_upper_lobe_right", "lung_middle_lobe_right",
           "lung_lower_lobe_left", "lung_lower_lobe_right", "lung_left", "lung_right"}
HEART   = {"heart"}
THORAX  = LUNG | HEART | {"aorta", "pulmonary_vein", "costal_cartilages", "sternum"}
ABDOMEN = {"liver", "spleen", "kidney_left", "kidney_right", "pancreas", "stomach",
           "colon", "small_bowel", "duodenum", "gallbladder",
           "adrenal_gland_left", "adrenal_gland_right"}
PELVIS  = {"hip_left", "hip_right", "sacrum", "urinary_bladder", "prostate",
           "femur_left", "femur_right"}
LEGS    = {"femur_left", "femur_right"}
ARMS    = {"humerus_left", "humerus_right", "scapula_left", "scapula_right"}
TORSO   = {"autochthon_left", "autochthon_right"}   # paraspinals: present in any torso scan

def has_spine(f):     # any vertebra at all
    return any(c.startswith("vertebrae_") for c in f)

# ---------------------------------------------------------------------------
# Task -> gate predicate over the footprint set `f`.
#   True  = keep (region plausibly in FOV)
#   ALWAYS = body-composition / whole-body tasks that apply to any scan with a body
#   None   = drop the task entirely (debug task / needs distal extremities we never scan)
# ---------------------------------------------------------------------------
def _any(group):
    return lambda f: bool(f & group)

ALWAYS = lambda f: True

GATES = {
    # --- torso organs (re-seg of total) ---
    "total_highres_test":        _any(ABDOMEN | LUNG),
    # --- lungs ---
    "lung_vessels":              _any(LUNG),
    "covid":                     _any(LUNG),
    "lung_nodules":              _any(LUNG),
    "pleural_pericard_effusion": _any(LUNG | HEART),
    # --- heart / great vessels ---
    "heartchambers_highres":     _any(HEART),
    "coronary_arteries":         _any(HEART),
    "aortic_sinuses":            _any(HEART),
    "aorta_annulus":             _any(HEART),
    "aortic_dissection":         _any({"aorta"}),
    "pulmonary_artery_landmarks":_any(HEART | {"pulmonary_vein"}),
    # --- liver ---
    "liver_vessels":             _any({"liver"}),
    "liver_segments":            _any({"liver"}),
    "liver_lesions":             _any({"liver"}),
    # --- kidney ---
    "kidney_cysts":              _any({"kidney_left", "kidney_right"}),
    "kidney_cysts_auxiliary":    _any({"kidney_left", "kidney_right"}),
    "renal_arteries":            _any({"kidney_left", "kidney_right", "aorta"}),
    "renal_arteries_auxiliary":  _any({"aorta"}),
    # --- spine ---
    "vertebrae_body":            has_spine,
    "vertebrae_pp":              has_spine,
    "vertebrae_pp_refined":      has_spine,
    # --- head / brain ---
    "oculomotor_muscles":        _any(HEAD),
    "cerebral_bleed":            _any({"brain"}),
    "brain_structures":          _any({"brain"}),
    "ventricle_parts":           _any({"brain"}),
    "brain_aneurysm":            _any({"brain"}),
    "head_glands_cavities":      _any(HEAD),
    "head_muscles":              _any(HEAD),
    "craniofacial_structures":   _any({"skull"}),
    "face":                      _any(HEAD),
    "face_mr_auxiliary":         _any({"brain", "liver"}),
    "teeth":                     _any({"skull"}),
    # --- head/neck junction ---
    "headneck_bones_vessels":    _any(HEAD | NECK),
    "headneck_muscles":          _any(HEAD | NECK),
    # --- extremities ---
    "appendicular_bones":            _any(LEGS | ARMS),
    "appendicular_bones_auxiliary":  _any(LEGS | ARMS | {"liver", "spleen"}),
    "thigh_shoulder_muscles":        _any(LEGS | ARMS),
    "hip_implant":               _any({"hip_implant"}),
    # --- breast (anterior chest wall): gate on true chest markers, not aorta which
    #     also traverses the abdomen (would falsely keep breasts on pelvis scans) ---
    "breasts":                   _any(LUNG | HEART | {"sternum", "costal_cartilages"}),
    # --- body composition / trunk (any body in FOV) ---
    "body":                      ALWAYS,
    "tissue_types":              ALWAYS,
    "tissue_4_types":            ALWAYS,
    "abdominal_muscles":         _any(TORSO | ABDOMEN | THORAX),
    "trunk_cavities":            _any(TORSO | ABDOMEN | THORAX),
    # --- debug / never ---
    "test":                      None,   # {carpal} debug task
}


def footprint(subj: str) -> set[str] | None:
    lp = ROOT / subj / "label.npy"
    if not lp.exists():
        return None
    arr = np.load(lp, mmap_mode="r")
    return {IDX_TO_CLASS[i] for i in np.unique(arr) if i in IDX_TO_CLASS}


def test_subjects() -> list[str]:
    import csv
    out = []
    with open(META, encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh, delimiter=";"):
            if row["split"].strip() == "test":
                out.append(row["image_id"].strip())
    return sorted(out)


def main():
    tasks = json.load(open(TASKS_JSON))
    # sanity: every task in the json has a gate
    missing = [t for t in tasks if t not in GATES]
    assert not missing, f"tasks without a gate: {missing}"

    result = {}
    for subj in test_subjects():
        f = footprint(subj)
        if f is None:
            result[subj] = {"footprint_classes": None, "tasks": None,
                            "note": "no label.npy (subject dir absent)"}
            continue
        keep = [t for t in tasks
                if GATES[t] is not None and GATES[t](f)]
        result[subj] = {
            "n_footprint": len(f),
            "n_tasks": len(keep),
            "tasks": keep,
        }
        print(f"{subj}: {len(keep):2d}/{len(tasks)} tasks  (footprint {len(f)})")

    json.dump(result, open(OUT, "w"), indent=2)
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
