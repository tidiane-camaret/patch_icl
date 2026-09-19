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


def test_pool_all_native_resolution_matters():
    m = PatchSetV2(resolution=4, enc_dims=(8, 8, 8), e=32, h=64, l=2, a=2, thinking_rows=2,
                   fourier_bands=4, compress_m=3, fine_stage=[1], image_size=[16, 16, 16])
    B, K, T = 1, 1, 2
    S = m.encoder.fine_stage_size(16, 1)          # stage-1 side: coarser than native 16
    assert S < 16
    Cf = m.encoder.fine_stage_channels(1)

    # Feature map with real spatial structure: two halves along the D axis at different values.
    fine_finest = torch.ones(B * T, Cf, S, S, S)
    fine_finest[:, :, S // 2:] = 5.0

    # A tiny, native-resolution foreground region sitting entirely in the LOW-value half, near
    # the region boundary -- small enough that coarse (S-resolution) downsampling of the mask
    # would blur it across the boundary into the HIGH-value half, while upsampling the FEATURE
    # to native first (this task's implementation) keeps the mask exact.
    context_out = torch.zeros(B, K, 16, 16, 16)
    context_out[:, :, 6:7, :2, :2] = 1.0           # native slice just before the D-axis midpoint

    pool = m._pool_all(fine_finest, context_out, None, B, K, T)
    assert pool.shape == (B, T, 32)

    # Replicate the WRONG (coarse-mask-first) ordering this task must NOT match: downsample the
    # mask to S first, then mask the still-coarse feature map, then reduce.
    mask_coarse = torch.nn.functional.interpolate(
        context_out.reshape(B * K, 1, 16, 16, 16).float(), size=(S, S, S),
        mode="trilinear", align_corners=False).reshape(B, K, 1, S, S, S)
    feat_for_wrong = fine_finest.reshape(B, T, Cf, S, S, S)[:, :K]
    mu = feat_for_wrong.mean(dim=(-3, -2, -1), keepdim=True)
    sig = feat_for_wrong.std(dim=(-3, -2, -1), keepdim=True) + 1e-8
    feat_z = ((feat_for_wrong - mu) / sig).clamp(-10, 10)
    num_wrong = (feat_z * mask_coarse).sum(dim=(-3, -2, -1))
    den_wrong = mask_coarse.sum(dim=(-3, -2, -1)).clamp_min(1e-6)
    pooled_wrong_raw = (num_wrong / den_wrong).squeeze(1)          # (B,Cf), pre-projection

    # The CORRECT (native-upsample-first) raw pooled vector, computed the same way _pool_all
    # does internally, for a like-for-like comparison before pool_proj's learned weights:
    feat_native = torch.nn.functional.interpolate(
        fine_finest.float(), size=(16, 16, 16), mode="trilinear", align_corners=False
        ).reshape(B, T, Cf, 16, 16, 16)[:, :K]
    mu2 = feat_native.mean(dim=(-3, -2, -1), keepdim=True)
    sig2 = feat_native.std(dim=(-3, -2, -1), keepdim=True) + 1e-8
    feat_z2 = ((feat_native - mu2) / sig2).clamp(-10, 10)
    mask_native = context_out.reshape(B, K, 1, 16, 16, 16).float()
    num_right = (feat_z2 * mask_native).sum(dim=(-3, -2, -1))
    den_right = mask_native.sum(dim=(-3, -2, -1)).clamp_min(1e-6)
    pooled_right_raw = (num_right / den_right).squeeze(1)

    assert not torch.allclose(pooled_right_raw, pooled_wrong_raw, atol=1e-3), (
        "native-resolution and coarse-resolution masking orders produced the same result -- "
        "this test setup doesn't actually distinguish upsample-before-mask from mask-before-upsample")

    # Now tie the REAL _pool_all output to the correct reference, and confirm it does NOT
    # match the wrong one -- this is what actually proves _pool_all follows the correct
    # (upsample-to-native-first) ordering, not just that the two references differ from
    # each other.
    right_via_proj = m.pool_proj(pooled_right_raw)
    wrong_via_proj = m.pool_proj(pooled_wrong_raw)
    assert torch.allclose(pool[:, 0], right_via_proj, atol=1e-4), (
        "_pool_all's real output for the context volume does not match the correct "
        "(upsample-to-native-first) reference computation")
    assert not torch.allclose(pool[:, 0], wrong_via_proj, atol=1e-3), (
        "_pool_all's real output matches the WRONG (mask-at-coarse-first) ordering")
