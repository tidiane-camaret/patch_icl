import warnings

import torch
from src.models.pfn_seg_2d import (DecodeCrossBlock, LowerPrecisionRMSNorm, MaskConvEmbedV2,
                                   RowCrossAttention)


def test_lower_precision_rmsnorm_bf16_matches_fp32_reference_and_returns_bf16():
    """x.float() must genuinely upcast (not just disable autocast on an already-bf16
    tensor) -- verified by comparing against RMSNorm computed directly in fp32, and the
    output must be cast back to bf16 for the caller."""
    torch.manual_seed(0)
    m = LowerPrecisionRMSNorm(8)
    x_fp32 = torch.randn(4, 8)
    x_bf16 = x_fp32.to(torch.bfloat16)

    out = m(x_bf16)
    assert out.dtype == torch.bfloat16

    expected_fp32 = torch.nn.functional.rms_norm(x_fp32, (8,), m.weight, m.eps)
    # bf16 has ~3 decimal digits of precision -- the input itself already lost precision
    # converting to bf16, so compare in bf16-appropriate tolerance, not fp32-tight
    assert torch.allclose(out.float(), expected_fp32, atol=2e-2, rtol=2e-2)


def test_lower_precision_rmsnorm_no_fused_dispatch_warning():
    m = LowerPrecisionRMSNorm(8)
    x_bf16 = torch.randn(4, 8).to(torch.bfloat16)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        m(x_bf16)   # raises if the "Cannot dispatch to fused implementation" warning fires


def test_row_cross_attention_shape_with_mismatched_row_counts():
    """r_q != r_kv must work -- this is the whole point (self-attention can't do this)."""
    m = RowCrossAttention(a=2, e=16, h=32)
    q_in = torch.randn(3, 5, 2, 16)     # B=3, r_q=5
    kv_in = torch.randn(3, 40, 2, 16)   # r_kv=40
    out = m(q_in, kv_in)
    assert out.shape == (3, 5, 2, 16)


def test_row_cross_attention_backward_reaches_all_params():
    m = RowCrossAttention(a=2, e=16, h=32)
    q_in = torch.randn(2, 4, 2, 16, requires_grad=True)
    kv_in = torch.randn(2, 10, 2, 16, requires_grad=True)
    out = m(q_in, kv_in)
    out.mean().backward()
    missing = [n for n, p in m.named_parameters() if p.requires_grad and p.grad is None]
    assert not missing, f"no grad reached: {missing}"
    assert q_in.grad is not None and kv_in.grad is not None


def test_row_cross_attention_output_depends_on_kv_content():
    """Sanity: changing kv_in must change the output (proves cross-attention actually reads
    kv, not just passing q_in through via the residual)."""
    torch.manual_seed(0)
    m = RowCrossAttention(a=2, e=16, h=32)
    m.eval()
    q_in = torch.randn(1, 3, 2, 16)
    kv_a = torch.randn(1, 8, 2, 16)
    kv_b = torch.randn(1, 8, 2, 16)
    out_a = m(q_in, kv_a)
    out_b = m(q_in, kv_b)
    assert not torch.allclose(out_a, out_b)


def test_decode_cross_block_shape_with_mismatched_row_counts():
    """T (m-scale) and F (N-scale) may have very different row counts -- the whole point."""
    m = DecodeCrossBlock(e=16, a=2, h=32)
    T = torch.randn(3, 5, 16)     # B=3, r_t=5
    F = torch.randn(3, 40, 16)    # r_f=40
    T2, F2 = m(T, F)
    assert T2.shape == (3, 5, 16)
    assert F2.shape == (3, 40, 16)


def test_decode_cross_block_backward_reaches_all_params():
    m = DecodeCrossBlock(e=16, a=2, h=32)
    T = torch.randn(2, 4, 16, requires_grad=True)
    F = torch.randn(2, 10, 16, requires_grad=True)
    T2, F2 = m(T, F)
    (T2.mean() + F2.mean()).backward()
    missing = [n for n, p in m.named_parameters() if p.requires_grad and p.grad is None]
    assert not missing, f"no grad reached: {missing}"
    assert T.grad is not None and F.grad is not None


def test_decode_cross_block_f_output_depends_on_t_input():
    """F's output must depend on T -- proves f2t actually reads the (T-derived) T2, i.e. T's
    influence reaches F within a single block, not just T's own output."""
    torch.manual_seed(0)
    m = DecodeCrossBlock(e=16, a=2, h=32)
    m.eval()
    F = torch.randn(1, 10, 16)
    T_a = torch.randn(1, 4, 16)
    T_b = torch.randn(1, 4, 16)
    _, F_out_a = m(T_a, F)
    _, F_out_b = m(T_b, F)
    assert not torch.allclose(F_out_a, F_out_b)


def test_mask_conv_embed_v2_shape():
    m = MaskConvEmbedV2(p=8, e=16)
    occ = torch.rand(2, 3, 8 ** 3)
    out = m(occ)
    assert out.shape == (2, 3, 16)


def test_mask_conv_embed_v2_p1_reduces_to_linear_shape():
    """Matches MaskConvEmbed's own p=1 edge case: no conv layers, proj is effectively
    Linear(1,e)."""
    m = MaskConvEmbedV2(p=1, e=16)
    assert len(m.convs) == 0
    assert m.final_ch == 1 and m.final_s == 1
    occ = torch.rand(2, 1)
    out = m(occ)
    assert out.shape == (2, 16)


def test_mask_conv_embed_v2_distinguishes_permuted_feature_positions():
    """The whole point: unlike MaskConvEmbed's pooled read-out (provably permutation
    -invariant over the final s^3 cells, docs/logs.md 2026-09-20), flatten+Linear must
    distinguish two feature maps that differ only in WHICH final cell holds a given value."""
    torch.manual_seed(0)
    m = MaskConvEmbedV2(p=8, e=16)
    m.eval()
    occ = torch.rand(1, 8 ** 3)
    with torch.no_grad():
        x = m.convs(occ.reshape(-1, 1, 8, 8, 8))
        x_perm = x.clone()
        x_perm[:, :, 0, 0, 0], x_perm[:, :, 1, 1, 1] = (
            x[:, :, 1, 1, 1].clone(), x[:, :, 0, 0, 0].clone())
        out = m.proj(x.flatten(1))
        out_perm = m.proj(x_perm.flatten(1))
    assert not torch.allclose(out, out_perm), (
        "MaskConvEmbedV2's output did not change under a pure positional permutation of the "
        "post-conv feature map -- the flatten fix may not actually be wired to the projection")


def test_mask_conv_embed_v2_backward_reaches_all_params():
    m = MaskConvEmbedV2(p=8, e=16)
    occ = torch.rand(2, 8 ** 3, requires_grad=True)
    out = m(occ)
    out.mean().backward()
    missing = [n for n, p in m.named_parameters() if p.requires_grad and p.grad is None]
    assert not missing, f"no grad reached: {missing}"
    assert occ.grad is not None
