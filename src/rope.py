"""
3D axial Rotary Position Embeddings (RoPE) for volumetric transformers.

Tokens come from a 3D spatial grid (Gd, Gh, Gw).  head_dim is split into
three even chunks — one per axis — so the dot product decomposes as:

    q_a · k_b  =  f_d(d_a − d_b) + f_h(h_a − h_b) + f_w(w_a − w_b)

This gives the model a strong inductive bias for spatial correspondence:
target token at (d, h, w) attends most to context tokens at the same location.

head_dim does not need to be divisible by 6; the remainder is distributed
evenly (2 extra dims) to the first axes.  All chunks are guaranteed even.

For small grids (≤ 32 tokens per axis) use theta=100 rather than the NLP
default of 10000 — the lower base spreads frequency variation across the
narrow position range.

References
----------
    Su et al. (2021) "RoFormer: Enhanced Transformer with Rotary Position Embedding"
    Heo et al. (2024) "Rotary Position Embedding for Vision Transformer" (RoPE-ViT)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.vit_seg import FFN


# ---------------------------------------------------------------------------
# Frequency precomputation
# ---------------------------------------------------------------------------

def _axis_splits(head_dim: int) -> tuple[int, int, int]:
    """Split head_dim into 3 even chunks, as equal as possible."""
    base = (head_dim // 6) * 2        # largest even number ≤ head_dim / 3
    rem  = head_dim - 3 * base        # always 0, 2, or 4
    splits = [base, base, base]
    for i in range(rem // 2):
        splits[i] += 2
    return tuple(splits)


def build_3d_rope_freqs(
    head_dim:  int,
    grid_size: tuple[int, int, int],
    theta:     float = 100.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Precompute cos/sin tables for 3D axial RoPE.

    Returns cos, sin each of shape (N, head_dim) where N = Gd * Gh * Gw.
    Register as non-trainable buffers; no gradient flows through these.
    """
    Gd, Gh, Gw = grid_size
    d_dim, h_dim, w_dim = _axis_splits(head_dim)

    def _cossin(n_pos: int, dim: int):
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / head_dim))
        freqs = torch.outer(torch.arange(n_pos).float(), inv_freq)  # (n_pos, dim/2)
        freqs = torch.cat([freqs, freqs], dim=-1)                   # (n_pos, dim) — rotate_half
        return freqs.cos(), freqs.sin()

    cd, sd = _cossin(Gd, d_dim)   # (Gd, d_dim)
    ch, sh = _cossin(Gh, h_dim)   # (Gh, h_dim)
    cw, sw = _cossin(Gw, w_dim)   # (Gw, w_dim)

    # Broadcast each 1-D table over the 3-D grid, then flatten → (N, dim_i)
    cd = cd[:, None, None, :].expand(Gd, Gh, Gw, d_dim).reshape(-1, d_dim)
    ch = ch[None, :, None, :].expand(Gd, Gh, Gw, h_dim).reshape(-1, h_dim)
    cw = cw[None, None, :, :].expand(Gd, Gh, Gw, w_dim).reshape(-1, w_dim)
    sd = sd[:, None, None, :].expand(Gd, Gh, Gw, d_dim).reshape(-1, d_dim)
    sh = sh[None, :, None, :].expand(Gd, Gh, Gw, h_dim).reshape(-1, h_dim)
    sw = sw[None, None, :, :].expand(Gd, Gh, Gw, w_dim).reshape(-1, w_dim)

    cos = torch.cat([cd, ch, cw], dim=-1)   # (N, head_dim)
    sin = torch.cat([sd, sh, sw], dim=-1)
    return cos, sin


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    return torch.cat([-x[..., d:], x[..., :d]], dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: (B, heads, N, head_dim);  cos/sin: (1, 1, N, head_dim)."""
    return x * cos + _rotate_half(x) * sin


# ---------------------------------------------------------------------------
# Attention modules
# ---------------------------------------------------------------------------

class RoPESelfAttn(nn.Module):
    """
    Multi-head self-attention with 3D axial RoPE.
    Includes residual + LayerNorm (mirrors MHA from vit_seg).
    Uses F.scaled_dot_product_attention (Flash Attention when available).
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout:   float = 0.0,
        grid_size: tuple[int, int, int] = (8, 8, 8),
        theta:     float = 100.0,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads
        self.dropout   = dropout

        self.qkv  = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

        cos, sin = build_3d_rope_freqs(self.head_dim, grid_size, theta)
        self.register_buffer("rope_cos", cos)   # (N, head_dim)
        self.register_buffer("rope_sin", sin)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)                        # each (B, N, heads, head_dim)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)  # (B, heads, N, d)

        cos = self.rope_cos[None, None]                # (1, 1, N, head_dim)
        sin = self.rope_sin[None, None]
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0
        )
        out = out.transpose(1, 2).reshape(B, N, C)
        return self.norm(x + self.proj(out))


class RoPECrossAttn(nn.Module):
    """
    Multi-head cross-attention with 3D axial RoPE.
    Q comes from target tokens (N positions); K/V from K*N context tokens.

    The same per-token frequency table is applied to both Q and K: for K*N
    context keys, the table is tiled K times.  The attention score therefore
    encodes the relative spatial offset between target and context patches —
    the model can learn to match anatomically corresponding locations.

    Includes residual + LayerNorm.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout:   float = 0.0,
        grid_size: tuple[int, int, int] = (8, 8, 8),
        theta:     float = 100.0,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads
        self.dropout   = dropout

        self.q_proj  = nn.Linear(embed_dim, embed_dim, bias=False)
        self.kv_proj = nn.Linear(embed_dim, 2 * embed_dim, bias=False)
        self.proj    = nn.Linear(embed_dim, embed_dim)
        self.norm    = nn.LayerNorm(embed_dim)

        cos, sin = build_3d_rope_freqs(self.head_dim, grid_size, theta)
        self.register_buffer("rope_cos", cos)   # (N, head_dim)
        self.register_buffer("rope_sin", sin)

    def forward(self, tgt: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        # tgt: (B, N, C),  ctx: (B, K*N, C)
        B, N, C = tgt.shape
        K = ctx.shape[1] // N

        q  = self.q_proj(tgt).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        kv = self.kv_proj(ctx).reshape(B, K * N, 2, self.num_heads, self.head_dim)
        k  = kv[..., 0, :, :].transpose(1, 2)   # (B, heads, K*N, head_dim)
        v  = kv[..., 1, :, :].transpose(1, 2)

        cos_q = self.rope_cos[None, None]                         # (1, 1,   N, head_dim)
        sin_q = self.rope_sin[None, None]
        cos_k = self.rope_cos.repeat(K, 1)[None, None]           # (1, 1, K*N, head_dim)
        sin_k = self.rope_sin.repeat(K, 1)[None, None]

        q = apply_rope(q, cos_q, sin_q)
        k = apply_rope(k, cos_k, sin_k)

        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0
        )
        out = out.transpose(1, 2).reshape(B, N, C)
        return self.norm(tgt + self.proj(out))


# ---------------------------------------------------------------------------
# Transformer blocks
# ---------------------------------------------------------------------------

class RoPETransformerBlock(nn.Module):
    """
    Stage-1 block: within-volume self-attention + FFN with 3D axial RoPE.
    Drop-in replacement for TransformerBlock (vit_seg).
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: int = 4,
        dropout:   float = 0.0,
        grid_size: tuple[int, int, int] = (8, 8, 8),
        theta:     float = 100.0,
    ):
        super().__init__()
        self.attn = RoPESelfAttn(embed_dim, num_heads, dropout, grid_size, theta)
        self.ffn  = FFN(embed_dim, mlp_ratio, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(self.attn(x))


class RoPECrossAttentionBlock(nn.Module):
    """
    Stage-2 block: target self-attention + cross-attention to context + FFN.
    Drop-in replacement for CrossAttentionBlock (vit_in_context).

    Forward:
        tgt : (B, N, C)   — target tokens
        ctx : (B, K*N, C) — all context tokens concatenated
    Returns updated target tokens (B, N, C).
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: int = 4,
        dropout:   float = 0.0,
        grid_size: tuple[int, int, int] = (8, 8, 8),
        theta:     float = 100.0,
    ):
        super().__init__()
        self.self_attn  = RoPESelfAttn(embed_dim, num_heads, dropout, grid_size, theta)
        self.cross_attn = RoPECrossAttn(embed_dim, num_heads, dropout, grid_size, theta)
        self.ffn        = FFN(embed_dim, mlp_ratio, dropout)

    def forward(self, tgt: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        tgt = self.self_attn(tgt)
        tgt = self.cross_attn(tgt, ctx)
        return self.ffn(tgt)
