"""Encoder registry: name -> build recipe + uniform call convention for the bench."""
from dataclasses import dataclass, field
from typing import Callable

import torch
import torch.nn as nn

from src.models.patchset3d import ConvEncoder3D
from src.models.encoders import ResEncEncoder


@dataclass
class EncoderSpec:
    name: str
    family: str                       # "cnn" | "transformer" | "mamba"
    factory: Callable[..., nn.Module]
    call: str = "single"              # "single" -> module(x); "img_mask" -> module(x, zeros)
    in_ch: int = 1
    size_multiple: int = 1            # input D=H=W must be divisible by this
    requires_ckpt: bool = False
    opt_profile: dict = field(default_factory=dict)


REGISTRY: dict[str, EncoderSpec] = {}


def register(spec: EncoderSpec) -> None:
    REGISTRY[spec.name] = spec


def list_encoders() -> list[str]:
    return sorted(REGISTRY)


def make_inputs(spec: EncoderSpec, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
    if spec.call == "img_mask":
        return (x, torch.zeros_like(x))
    return (x,)


# --- trivial, weights-free encoders ---------------------------------------
register(EncoderSpec(
    name="conv_encoder3d", family="cnn", call="single",
    factory=lambda: ConvEncoder3D(in_ch=1, dims=(32, 32, 32, 32), resolution=16),
    opt_profile={"autocast": "bf16", "channels_last": True, "compile": "reduce-overhead"},
))
register(EncoderSpec(
    name="resenc", family="cnn", call="img_mask",
    factory=lambda: ResEncEncoder(in_channels=1, features_per_stage=(32, 64, 128, 256)),
    opt_profile={"autocast": "bf16", "channels_last": True, "compile": "reduce-overhead"},
))
