import torch
from src.models.encoders.factory import build_encoder
from src.models.patchset3d_v2 import PatchSetV2


def test_build_encoder_conv_matches_direct_construction():
    from src.models.patchset3d import ConvEncoder3D
    enc = build_encoder("conv", resolution=4, enc_dims=(8, 8, 8))
    assert isinstance(enc, ConvEncoder3D)
    assert enc.out_ch == 24
    out = enc(torch.randn(2, 1, 16, 16, 16))
    assert out.shape == (2, 24, 4, 4, 4)


def _dummy_batch(B=2, K=2, S=16):
    image = torch.randn(B, 1, S, S, S)
    context_in = torch.randn(B, K, 1, S, S, S)
    context_out = (torch.rand(B, K, S, S, S) > 0.5).float()
    return image, context_in, context_out


def test_tokens_all_shape():
    m = PatchSetV2(resolution=4, enc_dims=(8, 8, 8), e=32, h=64, l=2, a=2, thinking_rows=2,
                   fourier_bands=4, compress_m=3, fine_stage=[0], image_size=[16, 16, 16])
    B, K, T = 2, 2, 3
    feat = torch.randn(B, T, m.N, m.encoder.out_ch)
    occ = torch.randn(B, T, m.N, 1)
    tok = m._tokens_all(feat, occ, B, T)
    assert tok.shape == (B, T, m.N, 2, 32)


def test_occupancy_shapes():
    m = PatchSetV2(resolution=4, enc_dims=(8, 8, 8), e=32, h=64, l=2, a=2, thinking_rows=2,
                   fourier_bands=4, compress_m=3, fine_stage=[0], image_size=[16, 16, 16])
    _, cin, cout = _dummy_batch(S=16, K=2)
    occ = m._occupancy(cout)
    assert occ.shape == (2, 2, m.N, 1)
    prior = torch.rand(2, 1, 16, 16, 16)
    pocc = m._prior_occupancy(prior)
    assert pocc.shape == (2, 1, m.N, 1)
