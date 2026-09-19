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
        # Frozen fomofo/tap-ct-b-3d ViT. Weights fixed on HF (no sidecar); it always
        # tokenizes at the native anisotropic grid (image_size drives the token count)
        # and is not spacing-aware — the physical cell size is set by data.crop_spacing_mm.
        # encoder_stage early-exits the transformer blocks (like Primus). Needs image_size
        # divisible by 8. Ignores encoder_native_grid/encoder_spacing_aware (always native).
        from src.models.tapct_encoder import TapCTEncoder
        if not image_size:
            raise ValueError("encoder='tap_ct' requires arch.image_size (from data.image_size)")
        return TapCTEncoder(resolution, image_size, frozen=encoder_frozen, device="cpu",
                            encoder_stage=encoder_stage, precision=encoder_precision)
    elif name == "nnunet_ts":
        # Frozen TotalSegmentator nnU-Net PlainConvUNet encoder (default: Dataset297,
        # total 3 mm). Multi-scale concat of nnunet_ts_stages resampled to R^3; input is
        # 1-channel (image only), spacing arg ignored (conv net). nnunet_ts_weights points
        # at the weights folder (plans.json + fold_0/checkpoint_final.pth) and is required
        # even when nnunet_ts_random_init=True — plans.json defines the architecture and the
        # CTNormalization stats; only the trained weights are dropped (He init instead).
        # encoder_input_norm: None keeps each encoder's own default (nnunet_ts=reframe,
        # so a frozen pretrained encoder still converts loader-frame -> its plans frame).
        from src.models.encoders.nnunet_ts import NnUNetTSEncoder
        if not nnunet_ts_weights:
            raise ValueError("encoder='nnunet_ts' requires arch.nnunet_ts_weights")
        _in_norm = {"input_norm": encoder_input_norm} if encoder_input_norm else {}
        return NnUNetTSEncoder(nnunet_ts_weights, resolution, stages=tuple(nnunet_ts_stages),
                               frozen=encoder_frozen, device="cpu", precision=encoder_precision,
                               random_init=nnunet_ts_random_init, **_in_norm)
    elif name == "resenc_ts":
        # From-scratch nnU-Net ResidualEncoderUNet (the ResEnc twin of nnunet_ts). No
        # plans.json / checkpoint: the architecture is the ResEnc M/L/XL recipe with
        # resenc_n_stages stages (base 32, x2, cap 320; blocks 1/3/4/6/6/...), He init.
        # Multi-scale concat of nnunet_ts_stages resampled to R^3; 1-channel image input,
        # spacing arg ignored. encoder_input_norm defaults to passthrough (the image is
        # already in the pipeline CT frame — see src/totalseg_dataset.CtNormSpec).
        from src.models.encoders.resenc_ts import ResEncTSEncoder
        _in_norm = {"input_norm": encoder_input_norm} if encoder_input_norm else {}
        return ResEncTSEncoder(resolution, n_stages=resenc_n_stages, stages=tuple(nnunet_ts_stages),
                               frozen=encoder_frozen, device="cpu", precision=encoder_precision,
                               **_in_norm)
    elif name == "plainconv_ts":
        # From-scratch nnU-Net PlainConvUNet (the PlainConv twin of resenc_ts). No
        # plans.json / checkpoint: width is plainconv_ts_features_per_stage if given,
        # else the same base=32/x2/cap=320 formula resenc_ts uses; n_conv_per_stage=2
        # throughout (nnU-Net's standard plain-conv schedule), He init. Multi-scale
        # concat of nnunet_ts_stages resampled to R^3; 1-channel image input, spacing
        # arg ignored. encoder_input_norm defaults to zscore (per-volume HU) here — this
        # encoder carries no plans-file CTNormalization stats to reframe into.
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
