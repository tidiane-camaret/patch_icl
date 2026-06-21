"""
Foreground morphology generators (build-time).

`make_foreground` dispatches a morphology name to its generator and returns a
binary uint8 mask plus a realized-meta dict (actual area, caliber, etc.). The
caller (task.make_base_geometry) then enforces area, roughens the boundary, and
composites background + distractors.
"""

import numpy as np

from . import blob, vessel, scattered

_DISPATCH = {
    "blob":      blob.make_blob,
    "elongated": blob.make_elongated,
    "annular":   blob.make_annular,
    "tubular":   vessel.make_vessel_tree,
    "scattered": scattered.make_scattered,
}

MORPHOLOGIES = tuple(_DISPATCH)


def make_foreground(morphology: str, image_size: int, geo_params: dict,
                    rng: np.random.Generator):
    """Return (mask uint8 [H,W], realized_meta dict) for one foreground shape."""
    if morphology not in _DISPATCH:
        raise ValueError(f"unknown morphology {morphology!r}; "
                         f"choices: {sorted(_DISPATCH)}")
    return _DISPATCH[morphology](image_size, geo_params, rng)
