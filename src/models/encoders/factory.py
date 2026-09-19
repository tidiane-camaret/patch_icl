"""Shared arch.encoder dispatch — used by both PatchSet3D and PatchSetV2 so the ~70-line
if/elif selecting an encoder implementation isn't duplicated between them. Each branch's
body is copied verbatim from PatchSet3D.__init__'s inline dispatch (see git history on
src/models/patchset3d.py predating this file for the version this replaces) — behavior is
unchanged, only the location moved."""

import torch.nn as nn


def build_encoder(
    name: str,
    resolution: int,
    *,
    in_ch: int = 1,
    enc_dims=None,
    encoder_frozen: bool = True,
    primus_sidecar: str | None = None,
    nnunet_ts_weights: str | None = None,
    nnunet_ts_stages=(2, 3, 4),
    nnunet_ts_random_init: bool = False,
    resenc_n_stages: int = 5,
    plainconv_ts_n_stages: int = 5,
    plainconv_ts_features_per_stage=None,
    encoder_input_norm: str | None = None,
    image_size=None,
    encoder_stage: int | None = None,
    encoder_native_grid: bool = False,
    encoder_spacing_aware: bool = False,
    encoder_precision: str = "bf16",
) -> nn.Module:
    if name == "primus":
        if not primus_sidecar:
            raise ValueError("encoder='primus' requires arch.primus_sidecar")
        from src.models.primus_encoder import PrimusEncoder
        return PrimusEncoder(primus_sidecar, resolution, frozen=encoder_frozen, device="cpu",
                             encoder_stage=encoder_stage, native_grid=encoder_native_grid,
                             spacing_aware=encoder_spacing_aware, precision=encoder_precision)
    elif name == "tap_ct":
        from src.models.tapct_encoder import TapCTEncoder
        if not image_size:
            raise ValueError("encoder='tap_ct' requires arch.image_size (from data.image_size)")
        return TapCTEncoder(resolution, image_size, frozen=encoder_frozen, device="cpu",
                            encoder_stage=encoder_stage, precision=encoder_precision)
    elif name == "nnunet_ts":
        from src.models.encoders.nnunet_ts import NnUNetTSEncoder
        if not nnunet_ts_weights:
            raise ValueError("encoder='nnunet_ts' requires arch.nnunet_ts_weights")
        _in_norm = {"input_norm": encoder_input_norm} if encoder_input_norm else {}
        return NnUNetTSEncoder(nnunet_ts_weights, resolution, stages=tuple(nnunet_ts_stages),
                               frozen=encoder_frozen, device="cpu", precision=encoder_precision,
                               random_init=nnunet_ts_random_init, **_in_norm)
    elif name == "resenc_ts":
        from src.models.encoders.resenc_ts import ResEncTSEncoder
        _in_norm = {"input_norm": encoder_input_norm} if encoder_input_norm else {}
        return ResEncTSEncoder(resolution, n_stages=resenc_n_stages, stages=tuple(nnunet_ts_stages),
                               frozen=encoder_frozen, device="cpu", precision=encoder_precision,
                               **_in_norm)
    elif name == "plainconv_ts":
        from src.models.encoders.plainconv_ts import PlainConvTSEncoder
        _in_norm = {"input_norm": encoder_input_norm} if encoder_input_norm else {}
        return PlainConvTSEncoder(resolution, n_stages=plainconv_ts_n_stages,
                                  stages=tuple(nnunet_ts_stages),
                                  features_per_stage=plainconv_ts_features_per_stage,
                                  frozen=encoder_frozen, device="cpu",
                                  precision=encoder_precision, **_in_norm)
    elif name == "conv":
        from src.models.patchset3d import ConvEncoder3D   # lazy: avoids import cycle
        return ConvEncoder3D(in_ch, tuple(enc_dims), resolution)
    raise ValueError(f"unknown arch.encoder {name!r} "
                     f"(conv | primus | tap_ct | nnunet_ts | resenc_ts | plainconv_ts)")
