"""
controlSynth: difficulty-controlled synthetic data for in-context 2D segmentation.

V1 (this package) is the *minimal generator*: on-the-fly procedural tasks with
calibrated difficulty knobs, plugged into experiments/2d/pfn_seg.py via the
`data.source=synthetic` switch. Base geometry is precomputed in RAM at dataset
init (the in-memory analog of the spec's LMDB store); only the cheap per-subject
path (deform -> intensity -> noise) runs per __getitem__.

Deferred to later sub-projects (see docs/datasets/controlSynth.md): LMDB store,
precompute CLI, eval harness + oracle UNets, clDice/NSD metrics, MixedDataLoader.
"""

from .config import (
    DiversityConfig,
    DifficultyBuildSpec,
    DifficultyLiveConfig,
    SamplingConfig,
)
from .dataset import SynthICLDataset

__all__ = [
    "DiversityConfig",
    "DifficultyBuildSpec",
    "DifficultyLiveConfig",
    "SamplingConfig",
    "SynthICLDataset",
]
