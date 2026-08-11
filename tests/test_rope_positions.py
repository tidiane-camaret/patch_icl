import torch

from src.rope import build_3d_rope_freqs, build_3d_rope_freqs_from_positions


def _grid_positions(g):
    """Row-major (d,h,w) positions matching build_3d_rope_freqs' flatten order."""
    d = torch.arange(g).repeat_interleave(g * g)
    h = torch.arange(g).repeat_interleave(g).repeat(g)
    w = torch.arange(g).repeat(g * g)
    return torch.stack([d, h, w], dim=-1).float()


def test_positions_match_grid_builder():
    """Explicit integer grid positions reproduce the grid-based RoPE table exactly."""
    head_dim, g = 64, 8
    cos_g, sin_g = build_3d_rope_freqs(head_dim, (g, g, g), theta=100.0)
    cos_p, sin_p = build_3d_rope_freqs_from_positions(head_dim, _grid_positions(g), theta=100.0)
    assert cos_p.shape == cos_g.shape == (g ** 3, head_dim)
    assert torch.allclose(cos_p, cos_g, atol=1e-6)
    assert torch.allclose(sin_p, sin_g, atol=1e-6)


def test_zero_position_is_identity():
    """A token at (0,0,0) gets cos=1, sin=0 (no rotation) — used for thinking rows."""
    head_dim = 64
    cos, sin = build_3d_rope_freqs_from_positions(head_dim, torch.zeros(1, 3), theta=100.0)
    assert torch.allclose(cos, torch.ones_like(cos))
    assert torch.allclose(sin, torch.zeros_like(sin))
