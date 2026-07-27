"""Encoder registry: name -> build recipe + uniform call convention for the bench."""
import os
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn

from src.models.patchset3d import ConvEncoder3D
from src.models.encoders import ResEncEncoder
from encoder_bench.encoders_standin import PrimusStandin, SegMambaStandin


@dataclass
class EncoderSpec:
    name: str
    family: str                       # "cnn" | "transformer" | "mamba"
    factory: Callable[..., nn.Module]
    call: str = "single"              # "single" -> module(x); "img_mask" -> module(x, zeros)
    in_ch: int = 1
    size_multiple: int = 1            # input D=H=W must be divisible by this
    requires_ckpt: bool = False
    opt_profile: dict | None = None


REGISTRY: dict[str, EncoderSpec] = {}


def register(spec: EncoderSpec) -> None:
    REGISTRY[spec.name] = spec


def list_encoders() -> list[str]:
    return sorted(REGISTRY)


def make_inputs(spec: EncoderSpec, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
    if spec.call == "img_mask":
        return (x, torch.zeros_like(x))
    if spec.call == "single":
        return (x,)
    raise ValueError(f"unknown call convention: {spec.call!r}")


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
register(EncoderSpec(
    name="primus", family="transformer", call="single", size_multiple=8,
    factory=lambda: PrimusStandin(img_size=64, patch=8, embed_dim=384, depth=12, heads=6),
    opt_profile={"autocast": "bf16", "compile": "max-autotune"},
))
register(EncoderSpec(
    name="segmamba", family="mamba", call="single", size_multiple=8,
    factory=lambda: SegMambaStandin(dims=(32, 64, 128, 256)),
    opt_profile={"autocast": "bf16", "channels_last": True},
))

# --- pretrained zoo encoders (weights-off + gated) ---------------------------

_OPT_CNN = {"autocast": "bf16", "channels_last": True, "compile": "reduce-overhead"}


def _stunet():
    """Build STU-Net-Base encoder (full depth) with no pretrained weights.

    Full-depth STU-Net-B has stride-32 total; inputs must be divisible by 32.
    Inputs too small (e.g. 32³ → 1³ bottleneck) will hit InstanceNorm errors
    and are recorded as honest error rows by the profiler.
    """
    from src.models.encoders.stunet import STUNetEncoder
    return STUNetEncoder(in_channels=1, variant="base", pretrained=None)


def _vocomni_swin():
    """Build VoComni SwinUNETR encoder with no checkpoint (random weights).

    compile_model=False: the wrapper self-compiles by default, but that internal
    torch.compile bypasses the harness's set_compiler_env() (broken bare-g++ -> can't
    find <algorithm>). Let apply_optimization own optimization, like every other encoder.
    """
    from src.models.encoders.vocomni import VoComniEncoder
    return VoComniEncoder(ckpt_path=None, feature_size=48, compile_model=False,
                          freeze_encoder=False)


def _vocomni_nnunet():
    """Build VoComni NNUNet encoder with no checkpoint (random weights)."""
    from src.models.encoders.vocomni_nnunet import VoComniNNUNetEncoder
    return VoComniNNUNetEncoder(ckpt_path=None, freeze_encoder=False, compile_model=False)


# Default checkpoint locations on thor (override via NNINT_CKPT / THREEDINO_CKPT env vars).
_NNINT_CKPT_DEFAULT = "/home/dpxuser/model_checkpoints/nnint/nnInteractive_v1.0"
_THREEDINO_CKPT_DEFAULT = "/home/dpxuser/model_checkpoints/3DINO/3dino_vit_weights.pth"


def _nninteractive():
    """Build NNInteractive encoder (default ckpt on thor; override via NNINT_CKPT)."""
    from src.models.encoders.nninteractive import NNInteractiveEncoder
    ckpt = os.environ.get("NNINT_CKPT", _NNINT_CKPT_DEFAULT)
    if not ckpt:
        raise FileNotFoundError("NNINT_CKPT not set")
    # freeze_encoder=False so the fwd+bwd timing has a real backward graph (matches
    # the other trainable zoo factories); the bench measures training compute, not a
    # frozen-feature use case.
    return NNInteractiveEncoder(ckpt_dir=ckpt, num_stages=6,
                                freeze_encoder=False, device="cpu")


def _threedino():
    """Build ThreeDINO encoder (default ckpt on thor; override via THREEDINO_CKPT)."""
    from src.models.encoders.threedino import ThreeDINOEncoder
    ckpt = os.environ.get("THREEDINO_CKPT", _THREEDINO_CKPT_DEFAULT)
    if not ckpt:
        raise FileNotFoundError("THREEDINO_CKPT not set")
    # freeze_encoder=False so fwd+bwd timing has a real backward graph (same reason as
    # nninteractive): the bench measures training compute, not frozen-feature extraction.
    return ThreeDINOEncoder(ckpt_path=ckpt, freeze_encoder=False)


register(EncoderSpec("stunet", "cnn", _stunet, call="img_mask",
                     size_multiple=32, opt_profile=_OPT_CNN))
register(EncoderSpec("vocomni_swin", "transformer", _vocomni_swin, call="single",
                     size_multiple=32, opt_profile={"autocast": "bf16"}))
register(EncoderSpec("vocomni_nnunet", "cnn", _vocomni_nnunet, call="img_mask",
                     size_multiple=32, opt_profile=_OPT_CNN))
register(EncoderSpec("nninteractive", "cnn", _nninteractive, call="img_mask",
                     requires_ckpt=True, opt_profile=_OPT_CNN))
register(EncoderSpec("threedino", "transformer", _threedino, call="single",
                     size_multiple=16, requires_ckpt=True, opt_profile={"autocast": "bf16"}))
