"""
Minimal 3-D Vision Transformer for volumetric segmentation.

Pipeline
--------
1. Split (1, D, H, W) volume into non-overlapping 3-D patches
2. Linear patch embedding → transformer encoder
3. Reshape token grid → trilinear upsample → 1×1×1 conv head

With the default patch_size=(16,32,32) a (128,256,256) volume gives
8×8×8 = 512 tokens, which fits comfortably in memory.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class PatchEmbed3D(nn.Module):
    """Split volume into non-overlapping patches and project to embed_dim."""

    def __init__(self, patch_size: tuple[int,int,int], in_channels: int, embed_dim: int):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int,int,int]]:
        # x: (B, C, D, H, W)
        x = self.proj(x)                          # (B, embed_dim, Gd, Gh, Gw)
        grid = x.shape[2:]                        # (Gd, Gh, Gw)
        x = x.flatten(2).transpose(1, 2)          # (B, N, embed_dim)
        return x, grid


class MHA(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.attn(x, x, x)
        return self.norm(x + out)


class FFN(nn.Module):
    def __init__(self, embed_dim: int, mlp_ratio: int = 4, dropout: float = 0.0):
        super().__init__()
        hidden = embed_dim * mlp_ratio
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim), nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.net(x))


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: int = 4, dropout: float = 0.0):
        super().__init__()
        self.mha = MHA(embed_dim, num_heads, dropout)
        self.ffn = FFN(embed_dim, mlp_ratio, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(self.mha(x))


# ---------------------------------------------------------------------------
# ViT segmentation model
# ---------------------------------------------------------------------------

class ViTSeg3D(nn.Module):
    """
    Args:
        image_size   : (D, H, W) of the input volume
        patch_size   : (pd, ph, pw) — must divide image_size evenly
        in_channels  : 1 for CT
        num_classes  : number of output classes (including background)
        embed_dim    : token embedding dimension
        depth        : number of transformer blocks
        num_heads    : attention heads (embed_dim % num_heads == 0)
        mlp_ratio    : FFN hidden dimension multiplier
        dropout      : attention + FFN dropout
    """

    def __init__(
        self,
        image_size: tuple[int,int,int] = (128, 256, 256),
        patch_size: tuple[int,int,int] = (16, 32, 32),
        in_channels: int = 1,
        num_classes: int = 2,
        embed_dim: int = 256,
        depth: int = 6,
        num_heads: int = 8,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_classes = num_classes

        grid = tuple(i // p for i, p in zip(image_size, patch_size))
        n_tokens = math.prod(grid)

        self.patch_embed = PatchEmbed3D(patch_size, in_channels, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, n_tokens, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.blocks = nn.Sequential(*[
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # Decoder: reshape → upsample → conv head
        self.decoder = nn.Sequential(
            nn.Conv3d(embed_dim, embed_dim // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(embed_dim // 2, num_classes, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, D, H, W)
        tokens, grid = self.patch_embed(x)         # (B, N, embed_dim)
        tokens = tokens + self.pos_embed
        tokens = self.blocks(tokens)
        tokens = self.norm(tokens)                 # (B, N, embed_dim)

        B, N, C = tokens.shape
        Gd, Gh, Gw = grid
        feat = tokens.transpose(1, 2).reshape(B, C, Gd, Gh, Gw)

        feat = F.interpolate(feat, size=self.image_size, mode="trilinear", align_corners=False)
        return self.decoder(feat)                  # (B, num_classes, D, H, W)
