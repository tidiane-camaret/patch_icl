import warnings

import torch
from src.models.pfn_seg_2d import LowerPrecisionRMSNorm, RowCrossAttention


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
