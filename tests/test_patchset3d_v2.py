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


def test_pool_all_native_resolution_and_value():
    m = PatchSetV2(resolution=4, enc_dims=(8, 8, 8), e=32, h=64, l=2, a=2, thinking_rows=2,
                   fourier_bands=4, compress_m=3, fine_stage=[0], image_size=[16, 16, 16])
    B, K, T = 1, 1, 2
    S = m.encoder.fine_stage_size(16, 0)          # stage-0 native side for a 16^3 input
    Cf = m.encoder.fine_stage_channels(0)
    fine_finest = torch.full((B * T, Cf, S, S, S), 3.0)   # constant feature map
    context_out = torch.zeros(B, K, 16, 16, 16)
    context_out[:, :, :2, :2, :2] = 1.0            # a small 2^3 foreground corner
    pool = m._pool_all(fine_finest, context_out, None, B, K, T)
    assert pool.shape == (B, T, 32)
    # constant input -> after per-volume z-score the whole map is 0 everywhere (std=0 branch
    # uses the 1e-8 floor), so the pooled *projection* is deterministic and identical for
    # every volume regardless of mask shape/size -- this is what we can assert without
    # depending on pool_proj's random init producing any particular non-zero value.
    assert torch.allclose(pool[:, 0], pool[:, 1])
