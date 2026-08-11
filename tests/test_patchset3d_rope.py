import torch

from src.models.patchset3d import PatchSet3D


def _inputs(B=1, K=2, S=16):
    image = torch.randn(B, 1, S, S, S)
    context_in = torch.randn(B, K, 1, S, S, S)
    context_out = (torch.rand(B, K, S, S, S) > 0.5).float()
    return image, context_in, context_out


def _model(transformer_rope, **kw):
    torch.manual_seed(0)
    return PatchSet3D(resolution=4, enc_dims=(8, 8), e=32, h=64, l=2, a=4,
                      thinking_rows=2, encoder="conv", full_attn=True,
                      transformer_rope=transformer_rope, **kw).eval()


def test_rope_forward_shape_finite():
    """transformer_rope=True forward returns finite logits at the R^3 grid."""
    m = _model(True)
    out = m(*_inputs(), mode="train")["final_logit"]
    assert out.shape == (1, 1, 4, 4, 4)
    assert torch.isfinite(out).all()


def test_rope_changes_output():
    """RoPE-on differs from RoPE-off (position actually enters attention)."""
    image, cin, cout = _inputs()
    off = _model(False)(image, cin, cout, mode="train")["final_logit"]
    on = _model(True)(image, cin, cout, mode="train")["final_logit"]
    assert not torch.allclose(off, on, atol=1e-4)


def test_rope_only_drops_additive_pe():
    """RoPE-only mode has no additive Fourier positional module."""
    assert not hasattr(_model(True), "pos") or _model(True).pos is None
    assert hasattr(_model(False), "pos")


def test_spacing_scales_positions():
    """A passed spacing scales the transformer RoPE positions -> different logits.
    (conv encoder ignores spacing, so this isolates the transformer's scaling.)"""
    image, cin, cout = _inputs()
    m = _model(True)
    a = m(image, cin, cout, mode="train", spacing=2.0)["final_logit"]
    b = m(image, cin, cout, mode="train", spacing=4.0)["final_logit"]
    assert not torch.allclose(a, b, atol=1e-4)


def test_spacing_two_equals_no_spacing():
    """spacing == rope_train_mm (2mm) -> identity integer positions == no-spacing RoPE."""
    image, cin, cout = _inputs()
    m = _model(True)
    none = m(image, cin, cout, mode="train", spacing=None)["final_logit"]
    two = m(image, cin, cout, mode="train", spacing=2.0)["final_logit"]
    assert torch.allclose(none, two, atol=1e-5)
