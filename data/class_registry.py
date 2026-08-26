"""Unified class registry for MAISI and TotalSegmentator vocabularies.

Provides a single source of truth for class names across both vocabularies,
with bidirectional lookup and automatic normalization.

Canonical format: TotalSeg underscore style (e.g., "kidney_left", "rib_left_6")

Usage:
    from data.class_registry import normalize, to_maisi_idx, to_totalseg_idx

    # Normalize any format to canonical
    normalize("left kidney")      # → "kidney_left"
    normalize("kidney_left")      # → "kidney_left" (passthrough)
    normalize("left rib 6")       # → "rib_left_6"

    # Get indices
    to_maisi_idx("kidney_left")   # → 14
    to_totalseg_idx("kidney_left") # → 42
"""

from dataclasses import dataclass
from typing import Optional
import re


@dataclass(frozen=True)
class ClassDef:
    """Metadata for a single anatomical class."""
    canonical: str                    # Canonical name (TotalSeg format)
    maisi_name: Optional[str] = None  # MAISI name if different
    maisi_idx: Optional[int] = None   # MAISI label index
    totalseg_idx: Optional[int] = None  # TotalSeg label index (1-based)
    category: Optional[str] = None    # Anatomical category


def _build_registry() -> dict[str, ClassDef]:
    """Build the complete class registry from both vocabularies."""
    registry = {}

    # === ORGANS (Abd/Pelvis) ===
    cat = "Organs (Abd/Pelvis)"
    registry["liver"] = ClassDef("liver", "liver", 1, 44, cat)
    registry["spleen"] = ClassDef("spleen", "spleen", 3, 84, cat)
    registry["pancreas"] = ClassDef("pancreas", "pancreas", 4, 50, cat)
    registry["kidney_right"] = ClassDef("kidney_right", "right kidney", 5, 43, cat)
    registry["kidney_left"] = ClassDef("kidney_left", "left kidney", 14, 42, cat)
    registry["gallbladder"] = ClassDef("gallbladder", "gallbladder", 10, 21, cat)
    registry["stomach"] = ClassDef("stomach", "stomach", 12, 86, cat)
    registry["duodenum"] = ClassDef("duodenum", "duodenum", 13, 17, cat)
    registry["small_bowel"] = ClassDef("small_bowel", "small bowel", 19, 82, cat)
    registry["colon"] = ClassDef("colon", "colon", 62, 13, cat)
    registry["urinary_bladder"] = ClassDef("urinary_bladder", "bladder", 15, 92, cat)
    registry["prostate"] = ClassDef("prostate", "prostate", 118, 52, cat)
    registry["adrenal_gland_left"] = ClassDef("adrenal_gland_left", "left adrenal gland", 9, 1, cat)
    registry["adrenal_gland_right"] = ClassDef("adrenal_gland_right", "right adrenal gland", 8, 2, cat)
    registry["kidney_cyst_left"] = ClassDef("kidney_cyst_left", "left kidney cyst", 116, 40, cat)
    registry["kidney_cyst_right"] = ClassDef("kidney_cyst_right", "right kidney cyst", 117, 41, cat)
    registry["esophagus"] = ClassDef("esophagus", "esophagus", 11, 18, cat)

    # === ORGANS (Thorax/Head/Spine) ===
    cat = "Organs (Thorax/Head/Spine)"
    registry["heart"] = ClassDef("heart", "heart", 115, 28, cat)
    registry["lung_upper_lobe_left"] = ClassDef("lung_upper_lobe_left", "left lung upper lobe", 28, 48, cat)
    registry["lung_lower_lobe_left"] = ClassDef("lung_lower_lobe_left", "left lung lower lobe", 29, 45, cat)
    registry["lung_upper_lobe_right"] = ClassDef("lung_upper_lobe_right", "right lung upper lobe", 30, 49, cat)
    registry["lung_middle_lobe_right"] = ClassDef("lung_middle_lobe_right", "right lung middle lobe", 31, 47, cat)
    registry["lung_lower_lobe_right"] = ClassDef("lung_lower_lobe_right", "right lung lower lobe", 32, 46, cat)
    registry["trachea"] = ClassDef("trachea", "trachea", 57, 91, cat)
    registry["thyroid_gland"] = ClassDef("thyroid_gland", "thyroid gland", 126, 90, cat)
    registry["brain"] = ClassDef("brain", "brain", 22, 10, cat)
    registry["spinal_cord"] = ClassDef("spinal_cord", "spinal cord", 121, 83, cat)
    registry["atrial_appendage_left"] = ClassDef("atrial_appendage_left", "left atrial appendage", 108, 4, cat)

    # === VESSELS ===
    cat = "Vessels"
    registry["aorta"] = ClassDef("aorta", "aorta", 6, 3, cat)
    registry["inferior_vena_cava"] = ClassDef("inferior_vena_cava", "inferior vena cava", 7, 39, cat)
    registry["portal_vein_and_splenic_vein"] = ClassDef("portal_vein_and_splenic_vein", "portal vein and splenic vein", 17, 51, cat)
    registry["superior_vena_cava"] = ClassDef("superior_vena_cava", "superior vena cava", 125, 89, cat)
    registry["pulmonary_vein"] = ClassDef("pulmonary_vein", "pulmonary vein", 119, 53, cat)
    registry["brachiocephalic_trunk"] = ClassDef("brachiocephalic_trunk", "brachiocephalic trunk", 109, 7, cat)
    registry["brachiocephalic_vein_left"] = ClassDef("brachiocephalic_vein_left", "left brachiocephalic vein", 110, 8, cat)
    registry["brachiocephalic_vein_right"] = ClassDef("brachiocephalic_vein_right", "right brachiocephalic vein", 111, 9, cat)
    registry["common_carotid_artery_left"] = ClassDef("common_carotid_artery_left", "left common carotid artery", 112, 14, cat)
    registry["common_carotid_artery_right"] = ClassDef("common_carotid_artery_right", "right common carotid artery", 113, 15, cat)
    registry["subclavian_artery_left"] = ClassDef("subclavian_artery_left", "left subclavian artery", 123, 87, cat)
    registry["subclavian_artery_right"] = ClassDef("subclavian_artery_right", "right subclavian artery", 124, 88, cat)
    registry["iliac_artery_left"] = ClassDef("iliac_artery_left", "left iliac artery", 58, 33, cat)
    registry["iliac_artery_right"] = ClassDef("iliac_artery_right", "right iliac artery", 59, 34, cat)
    registry["iliac_vena_left"] = ClassDef("iliac_vena_left", "left iliac vena", 60, 35, cat)
    registry["iliac_vena_right"] = ClassDef("iliac_vena_right", "right iliac vena", 61, 36, cat)

    # === MUSCLES ===
    cat = "Muscles"
    registry["autochthon_left"] = ClassDef("autochthon_left", "left autochthon", 104, 5, cat)
    registry["autochthon_right"] = ClassDef("autochthon_right", "right autochthon", 105, 6, cat)
    registry["iliopsoas_left"] = ClassDef("iliopsoas_left", "left iliopsoas", 106, 37, cat)
    registry["iliopsoas_right"] = ClassDef("iliopsoas_right", "right iliopsoas", 107, 38, cat)
    registry["gluteus_maximus_left"] = ClassDef("gluteus_maximus_left", "left gluteus maximus", 98, 22, cat)
    registry["gluteus_maximus_right"] = ClassDef("gluteus_maximus_right", "right gluteus maximus", 99, 23, cat)
    registry["gluteus_medius_left"] = ClassDef("gluteus_medius_left", "left gluteus medius", 100, 24, cat)
    registry["gluteus_medius_right"] = ClassDef("gluteus_medius_right", "right gluteus medius", 101, 25, cat)
    registry["gluteus_minimus_left"] = ClassDef("gluteus_minimus_left", "left gluteus minimus", 102, 26, cat)
    registry["gluteus_minimus_right"] = ClassDef("gluteus_minimus_right", "right gluteus minimus", 103, 27, cat)

    # === BONES (Limbs/Shoulder/Pelvis) ===
    cat = "Bones (Limbs/Shoulder/Pelvis)"
    registry["skull"] = ClassDef("skull", "skull", 120, 81, cat)
    registry["humerus_left"] = ClassDef("humerus_left", "left humerus", 87, 31, cat)
    registry["humerus_right"] = ClassDef("humerus_right", "right humerus", 88, 32, cat)
    registry["scapula_left"] = ClassDef("scapula_left", "left scapula", 89, 79, cat)
    registry["scapula_right"] = ClassDef("scapula_right", "right scapula", 90, 80, cat)
    registry["clavicula_left"] = ClassDef("clavicula_left", "left clavicula", 91, 11, cat)
    registry["clavicula_right"] = ClassDef("clavicula_right", "right clavicula", 92, 12, cat)
    registry["femur_left"] = ClassDef("femur_left", "left femur", 93, 19, cat)
    registry["femur_right"] = ClassDef("femur_right", "right femur", 94, 20, cat)
    registry["hip_left"] = ClassDef("hip_left", "left hip", 95, 29, cat)
    registry["hip_right"] = ClassDef("hip_right", "right hip", 96, 30, cat)

    # === BONES (Spine) ===
    cat = "Bones (Spine)"
    registry["sacrum"] = ClassDef("sacrum", "sacrum", 97, 78, cat)
    # Vertebrae: MAISI ids 33-56 (L5→C1), 127 (S1)
    vert_maisi = {
        "L5": 33, "L4": 34, "L3": 35, "L2": 36, "L1": 37,
        "T12": 38, "T11": 39, "T10": 40, "T9": 41, "T8": 42, "T7": 43,
        "T6": 44, "T5": 45, "T4": 46, "T3": 47, "T2": 48, "T1": 49,
        "C7": 50, "C6": 51, "C5": 52, "C4": 53, "C3": 54, "C2": 55, "C1": 56,
        "S1": 127,
    }
    # TotalSeg indices: C1=93, C2=94, ..., C7=99, L1=100, ..., L5=104, S1=105, T1=106, ..., T12=117
    vert_ts = {
        "C1": 93, "C2": 94, "C3": 95, "C4": 96, "C5": 97, "C6": 98, "C7": 99,
        "L1": 100, "L2": 101, "L3": 102, "L4": 103, "L5": 104, "S1": 105,
        "T1": 106, "T2": 107, "T3": 108, "T4": 109, "T5": 110, "T6": 111,
        "T7": 112, "T8": 113, "T9": 114, "T10": 115, "T11": 116, "T12": 117,
    }
    for level in vert_maisi:
        canon = f"vertebrae_{level}"
        maisi_name = f"vertebrae {level}"
        registry[canon] = ClassDef(canon, maisi_name, vert_maisi[level], vert_ts[level], cat)

    # === BONES (Ribs/Sternum) ===
    cat = "Bones (Ribs/Sternum)"
    registry["sternum"] = ClassDef("sternum", "sternum", 122, 85, cat)
    registry["costal_cartilages"] = ClassDef("costal_cartilages", "costal cartilages", 114, 16, cat)
    # Ribs: MAISI 63-74 (left 1-12), 75-86 (right 1-12)
    # TotalSeg: rib_left_1=54, ..., rib_left_12=65, rib_right_1=66, ..., rib_right_12=77
    for i in range(1, 13):
        registry[f"rib_left_{i}"] = ClassDef(
            f"rib_left_{i}", f"left rib {i}", 62 + i, 53 + i, cat)
        registry[f"rib_right_{i}"] = ClassDef(
            f"rib_right_{i}", f"right rib {i}", 74 + i, 65 + i, cat)

    # === MAISI-ONLY CLASSES (no TotalSeg equivalent) ===
    cat = "MAISI-only"
    registry["lung_tumor"] = ClassDef("lung_tumor", "lung tumor", 23, None, cat)
    registry["pancreatic_tumor"] = ClassDef("pancreatic_tumor", "pancreatic tumor", 24, None, cat)
    registry["hepatic_vessel"] = ClassDef("hepatic_vessel", "hepatic vessel", 25, None, cat)
    registry["hepatic_tumor"] = ClassDef("hepatic_tumor", "hepatic tumor", 26, None, cat)
    registry["colon_cancer_primaries"] = ClassDef("colon_cancer_primaries", "colon cancer primaries", 27, None, cat)
    registry["bone_lesion"] = ClassDef("bone_lesion", "bone lesion", 128, None, cat)
    registry["airway"] = ClassDef("airway", "airway", 132, None, cat)
    registry["body"] = ClassDef("body", "body", 200, None, cat)

    # === TOTALSEG-ONLY CLASSES (MRI, subtasks) ===
    cat = "TotalSeg-only"
    registry["lung_left"] = ClassDef("lung_left", None, None, 118, cat)
    registry["lung_right"] = ClassDef("lung_right", None, None, 119, cat)
    registry["intervertebral_discs"] = ClassDef("intervertebral_discs", None, None, 120, cat)
    registry["vertebrae"] = ClassDef("vertebrae", None, None, 121, cat)
    registry["hip_implant"] = ClassDef("hip_implant", None, None, 122, cat)

    return registry


# Build registry once at import time
CLASS_REGISTRY: dict[str, ClassDef] = _build_registry()

# Build reverse lookup maps
_MAISI_NAME_TO_CANONICAL: dict[str, str] = {}
_MAISI_IDX_TO_CANONICAL: dict[int, str] = {}
_TS_IDX_TO_CANONICAL: dict[int, str] = {}

for canon, cdef in CLASS_REGISTRY.items():
    if cdef.maisi_name:
        _MAISI_NAME_TO_CANONICAL[cdef.maisi_name.lower()] = canon
    if cdef.maisi_idx is not None:
        _MAISI_IDX_TO_CANONICAL[cdef.maisi_idx] = canon
    if cdef.totalseg_idx is not None:
        _TS_IDX_TO_CANONICAL[cdef.totalseg_idx] = canon


def _algorithmic_normalize(name: str) -> str:
    """Algorithmic fallback for names not in registry.

    Handles the L/R flip pattern and rib format conversion.
    """
    n = name.lower().strip()

    # Handle rib patterns: "left rib 6" → "rib_left_6"
    m = re.match(r"(left|right)\s+rib\s+(\d+)", n)
    if m:
        return f"rib_{m.group(1)}_{m.group(2)}"

    # Handle L/R prefix flip: "left kidney" → "kidney_left"
    for d in ("left", "right"):
        if n.startswith(f"{d} "):
            rest = n[len(d) + 1:]
            return rest.replace(" ", "_") + f"_{d}"

    # Default: just replace spaces with underscores
    return n.replace(" ", "_")


def normalize(name: str) -> str:
    """Normalize any class name format to canonical (TotalSeg) format.

    Handles:
    - TotalSeg format passthrough: "kidney_left" → "kidney_left"
    - MAISI L/R format: "left kidney" → "kidney_left"
    - MAISI rib format: "left rib 6" → "rib_left_6"
    - Synonyms: "bladder" → "urinary_bladder"

    Raises:
        KeyError: If the name cannot be normalized to a known class.
    """
    # Fast path: already canonical
    if name in CLASS_REGISTRY:
        return name

    # Try MAISI name lookup
    name_lower = name.lower().strip()
    if name_lower in _MAISI_NAME_TO_CANONICAL:
        return _MAISI_NAME_TO_CANONICAL[name_lower]

    # Try with underscores replaced by spaces (TotalSeg → MAISI lookup)
    name_spaces = name_lower.replace("_", " ")
    if name_spaces in _MAISI_NAME_TO_CANONICAL:
        return _MAISI_NAME_TO_CANONICAL[name_spaces]

    # Algorithmic fallback
    canonical = _algorithmic_normalize(name)
    if canonical in CLASS_REGISTRY:
        return canonical

    raise KeyError(
        f"Unknown class name: {name!r}. "
        f"Use canonical format (e.g., 'kidney_left') or MAISI format (e.g., 'left kidney')."
    )


def normalize_lenient(name: str) -> str:
    """Like normalize(), but returns the algorithmic result for unknown classes."""
    try:
        return normalize(name)
    except KeyError:
        return _algorithmic_normalize(name)


def get(name: str) -> ClassDef:
    """Get ClassDef for a class name (any format)."""
    return CLASS_REGISTRY[normalize(name)]


def to_maisi_idx(name: str) -> Optional[int]:
    """Get MAISI label index for a class name (any format)."""
    return get(name).maisi_idx


def to_totalseg_idx(name: str) -> Optional[int]:
    """Get TotalSeg label index for a class name (any format)."""
    return get(name).totalseg_idx


def from_maisi_idx(idx: int) -> str:
    """Get canonical name from MAISI label index."""
    if idx not in _MAISI_IDX_TO_CANONICAL:
        raise KeyError(f"Unknown MAISI index: {idx}")
    return _MAISI_IDX_TO_CANONICAL[idx]


def from_totalseg_idx(idx: int) -> str:
    """Get canonical name from TotalSeg label index."""
    if idx not in _TS_IDX_TO_CANONICAL:
        raise KeyError(f"Unknown TotalSeg index: {idx}")
    return _TS_IDX_TO_CANONICAL[idx]


def all_canonical() -> list[str]:
    """Return all canonical class names."""
    return list(CLASS_REGISTRY.keys())


def all_with_maisi() -> list[str]:
    """Return canonical names for classes with MAISI indices."""
    return [c for c, d in CLASS_REGISTRY.items() if d.maisi_idx is not None]


def all_with_totalseg() -> list[str]:
    """Return canonical names for classes with TotalSeg indices."""
    return [c for c, d in CLASS_REGISTRY.items() if d.totalseg_idx is not None]


def maisi_to_totalseg_idx(maisi_idx: int) -> Optional[int]:
    """Convert MAISI index to TotalSeg index (None if no mapping)."""
    if maisi_idx not in _MAISI_IDX_TO_CANONICAL:
        return None
    return CLASS_REGISTRY[_MAISI_IDX_TO_CANONICAL[maisi_idx]].totalseg_idx


def totalseg_to_maisi_idx(ts_idx: int) -> Optional[int]:
    """Convert TotalSeg index to MAISI index (None if no mapping)."""
    if ts_idx not in _TS_IDX_TO_CANONICAL:
        return None
    return CLASS_REGISTRY[_TS_IDX_TO_CANONICAL[ts_idx]].maisi_idx


# Convenience: MAISI_IDX → canonical name (for drop-in replacement)
MAISI_IDX_TO_CANONICAL = _MAISI_IDX_TO_CANONICAL
TS_IDX_TO_CANONICAL = _TS_IDX_TO_CANONICAL
