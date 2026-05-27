"""3D Rotary Position Embedding utilities.

Splits the attention dimension into 3 axis blocks aligned to head_dim boundaries,
one per spatial axis (d, h, w).  Any leftover features are left unrotated.

Works with arbitrary token subsets — RoPE is applied via per-token integer
coordinates, not by assuming tokens fill a regular grid.

Analogous to the 2D RoPE in the ic_segmentation backbone (rope.py).
"""

from __future__ import annotations

import torch


def build_rope_cache_3d(
    max_pos:   int,
    dim:       int,
    num_heads: int   = 1,
    base:      float = 10000.0,
) -> torch.Tensor:
    """Precompute 3D RoPE sin/cos cache.

    Args:
        max_pos   : positions 0 … max_pos-1 are covered
        dim       : full attention dimension (applied before head split)
        num_heads : number of attention heads; per_axis is rounded down to the
                    nearest multiple of head_dim so head boundaries never fall
                    inside a rotated axis block (fixes mixed-axis heads)
        base      : RoPE base frequency

    Returns:
        (max_pos, n_pairs, 2)  where n_pairs = per_axis // 2.
        The two trailing values are [cos, sin] for each (position, pair) entry.
    """
    head_dim = dim // max(num_heads, 1)
    # Align per_axis to head_dim so no head straddles two axis blocks.
    # Example: dim=512, num_heads=8, head_dim=64 → per_axis=128 (2 heads/axis).
    per_axis = (dim // 3 // head_dim) * head_dim  # multiple of head_dim
    if per_axis == 0:
        per_axis = (dim // 3 // 2) * 2            # fallback: just round to even
    n_pairs  = per_axis // 2              # complex pairs per axis
    if n_pairs == 0:
        return torch.zeros(max_pos, 1, 2)

    theta = 1.0 / (base ** (
        torch.arange(0, n_pairs * 2, 2, dtype=torch.float32) / (n_pairs * 2)
    ))
    positions = torch.arange(max_pos, dtype=torch.float32)
    freqs     = torch.einsum("i,j->ij", positions, theta)  # (max_pos, n_pairs)
    return torch.stack([freqs.cos(), freqs.sin()], dim=-1)  # (max_pos, n_pairs, 2)


def apply_rope_3d(
    x:          torch.Tensor,  # (B, K, dim)
    coords:     torch.Tensor,  # (B, K, 3) integer coords (d, h, w)
    rope_cache: torch.Tensor,  # (max_pos, n_pairs, 2)
) -> torch.Tensor:             # (B, K, dim)
    """Apply 3D RoPE to Q or K tokens via their explicit spatial coordinates.

    The first 3 * per_axis features are rotated (per_axis = n_pairs * 2);
    any remainder is returned unchanged.
    """
    B, K, D = x.shape
    n_pairs  = rope_cache.shape[1]
    per_axis = n_pairs * 2
    used     = per_axis * 3

    if used == 0 or used > D:
        return x

    orig_dtype = x.dtype
    x         = x.float()
    max_pos   = rope_cache.shape[0]

    d_idx = coords[..., 0].long().clamp(0, max_pos - 1)
    h_idx = coords[..., 1].long().clamp(0, max_pos - 1)
    w_idx = coords[..., 2].long().clamp(0, max_pos - 1)

    d_rope = rope_cache[d_idx]  # (B, K, n_pairs, 2)
    h_rope = rope_cache[h_idx]
    w_rope = rope_cache[w_idx]

    def _rotate(part: torch.Tensor, rope: torch.Tensor) -> torch.Tensor:
        """Real-space rotation — avoids view_as_complex for inductor compatibility.

        Equivalent to complex multiply (x0 + x1·i)(cos + sin·i):
          real = x0·cos − x1·sin
          imag = x0·sin + x1·cos
        part: (B, K, n_pairs, 2)  rope: (B, K, n_pairs, 2) = [cos, sin]
        """
        x0, x1   = part[..., 0], part[..., 1]
        cos, sin = rope[..., 0], rope[..., 1]
        return torch.stack([x0 * cos - x1 * sin,
                            x0 * sin + x1 * cos], dim=-1).reshape(B, K, per_axis)

    d_part = x[:, :, 0           : per_axis    ].reshape(B, K, n_pairs, 2)
    h_part = x[:, :, per_axis    : per_axis * 2].reshape(B, K, n_pairs, 2)
    w_part = x[:, :, per_axis * 2: per_axis * 3].reshape(B, K, n_pairs, 2)

    out = torch.cat([
        _rotate(d_part, d_rope),
        _rotate(h_part, h_rope),
        _rotate(w_part, w_rope),
        x[:, :, used:],           # unrotated remainder
    ], dim=-1)

    return out.to(orig_dtype)
