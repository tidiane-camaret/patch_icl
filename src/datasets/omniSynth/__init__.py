"""omniSynth: in-context 2D segmentation from Omniglot characters on a grid.

See docs/superpowers/specs/2026-06-29-omnisynth-design.md. A task = one target
character class; each item is a 4x4 grid of characters where k cells hold the
target (mask = ink pixels of the target characters within those cells) and the rest are distractors. Plugs into the 2D
pipeline via data.source=omnisynth.
"""
from .config import (OmniDiversityConfig, OmniMedSegConfig, OmniSceneConfig,
                     OmniSamplingConfig)
from .dataset import OmniSynthICLDataset

__all__ = [
    "OmniDiversityConfig",
    "OmniMedSegConfig",
    "OmniSceneConfig",
    "OmniSamplingConfig",
    "OmniSynthICLDataset",
]
