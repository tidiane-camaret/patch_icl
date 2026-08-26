"""FLARE22 volume provider for the in-context dataloader v2.

Stored at NATIVE anisotropic spacing (~0.8 x 0.8 x 2.5 mm) by scripts/convert_flare22.py;
the crop + isotropic resample happens at load time in NativeGridProvider, so
`crop_spacing_mm` is a config knob and not a property of the conversion.

See docs/datasets/flare22.md for the GT-fidelity table (crop-space Dice overstates
native-space Dice by 2-21 points, organ-dependent) and why 2.5 mm is the fidelity optimum.
"""
from src.providers.native_grid import NativeGridProvider, resolve_classes_for

# FLARE22 label index -> TotalSegmentator class name (index order verified on disk;
# laterality confirmed 50/50 on the RAS x-axis for both the kidney and adrenal pairs).
FLARE22_CLASSES = [
    "liver", "kidney_right", "spleen", "pancreas", "aorta",
    "inferior_vena_cava", "adrenal_gland_right", "adrenal_gland_left",
    "gallbladder", "esophagus", "stomach", "duodenum", "kidney_left",
]
FLARE22_IDX = {name: i + 1 for i, name in enumerate(FLARE22_CLASSES)}


def resolve_flare22_classes(value) -> list[str]:
    """Resolve cfg.data.val_classes for the flare22 source ("all" or an explicit list)."""
    return resolve_classes_for(FLARE22_CLASSES, value, "flare22")


class Flare22Provider(NativeGridProvider):
    SOURCE = "flare22"
    ALL_CLASSES = FLARE22_CLASSES
    CLASS_IDX = FLARE22_IDX
