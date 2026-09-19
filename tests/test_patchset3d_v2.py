import torch
from src.models.encoders.factory import build_encoder


def test_build_encoder_conv_matches_direct_construction():
    from src.models.patchset3d import ConvEncoder3D
    enc = build_encoder("conv", resolution=4, enc_dims=(8, 8, 8))
    assert isinstance(enc, ConvEncoder3D)
    assert enc.out_ch == 24
    out = enc(torch.randn(2, 1, 16, 16, 16))
    assert out.shape == (2, 24, 4, 4, 4)
