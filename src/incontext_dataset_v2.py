"""Generic in-context segmentation dataset engine (v2).

A source-agnostic `InContextDataset` assembles items from a `VolumeProvider`.
Per-item state flows through `LoadRequest`/`LoadResult`, so there is no mutable
instance side-channel (contrast the v1 `_cur_rng`/`_last_crop_geom`).
"""
import random
from dataclasses import dataclass
from typing import Optional, Protocol

import torch


@dataclass
class LoadRequest:
    rng: random.Random                 # per-item RNG (eval determinism or global)
    crop_spacing_mm: float             # physical crop pitch for THIS item
    center: Optional[tuple] = None     # native-voxel crop center; None -> provider default
                                       # (cascade fine-crop seam; v2 always passes None)


@dataclass
class LoadResult:
    image: torch.Tensor                # (1, T, T, T) f32, normalized
    label: torch.Tensor               # (T, T, T) i64, binary {0,1}
    spacing: torch.Tensor              # (3,) mm/voxel of the output
    crop_geom: torch.Tensor            # (4, 3) i64: starts, crop_sizes, out_sizes, pad_lo


class VolumeProvider(Protocol):
    classes: list
    def subjects_for(self, cls: str) -> list: ...
    def load(self, subject: str, cls: str, req: LoadRequest) -> LoadResult: ...
