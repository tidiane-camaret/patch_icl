import torch
from src.models.pfn_seg_2d import RowCrossAttention


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
