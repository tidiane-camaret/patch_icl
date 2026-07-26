"""Compute-only, architecturally-faithful encoder stand-ins (no pretrained weights).

Primus: high-res-token pure-ViT (arxiv 2503.01835). SegMamba: CNN-stem + SSM blocks
(arxiv 2401.13560). Faithful block structure/dims for FLOPs/latency/VRAM; NOT for Dice.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class _Block(nn.Module):
    def __init__(self, dim, heads, mlp_ratio):
        super().__init__()
        self.n1 = nn.LayerNorm(dim); self.n2 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3); self.proj = nn.Linear(dim, dim)
        self.heads = heads
        h = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, h), nn.GELU(), nn.Linear(h, dim))

    def forward(self, x):
        B, N, C = x.shape
        q, k, v = self.qkv(self.n1(x)).reshape(B, N, 3, self.heads, C // self.heads).unbind(2)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))          # (B,heads,N,d)
        a = F.scaled_dot_product_attention(q, k, v)               # -> FlashAttention-2
        x = x + self.proj(a.transpose(1, 2).reshape(B, N, C))
        return x + self.mlp(self.n2(x))


class PrimusStandin(nn.Module):
    def __init__(self, in_ch=1, img_size=64, patch=8, embed_dim=384, depth=12,
                 heads=6, mlp_ratio=4.0):
        super().__init__()
        self.patch = patch
        self.embed = nn.Conv3d(in_ch, embed_dim, patch, stride=patch)
        g = img_size // patch
        self.pos = nn.Parameter(torch.zeros(1, embed_dim, g, g, g))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList([_Block(embed_dim, heads, mlp_ratio)
                                     for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.embed(x)                                          # (B,C,g,g,g)
        pos = F.interpolate(self.pos, size=x.shape[-3:], mode="trilinear",
                            align_corners=False)
        x = (x + pos).flatten(2).transpose(1, 2)                   # (B,N,C)
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)
