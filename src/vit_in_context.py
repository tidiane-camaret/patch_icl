"""
In-context 3-D segmentation ViT.

Each forward pass receives one target volume and K context (image, mask) pairs
for the same class.  Attention is split into two stages:

  Stage 1  — within-image self-attention.
             All volumes (target + K context pairs) share the same transformer
             blocks and are processed in one batched forward pass.
             Context tokens are formed by adding image tokens and mask tokens,
             so each context token encodes both appearance and supervision.

  Stage 2  — cross-context attention.
             Only target tokens are updated.  Each block does:
               (a) self-attention within the target sequence, then
               (b) cross-attention: Q = target tokens, K = V = all K*N context tokens.
             Context tokens are never modified in stage 2.

Decoder — identical to ViTSeg3D: reshape token grid → trilinear upsample → 1×1×1 conv head.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.vit_seg import PatchEmbed3D, MHA, FFN, TransformerBlock


# ---------------------------------------------------------------------------
# Stage-2 block: target self-attn + cross-attn to context
# ---------------------------------------------------------------------------

class CrossAttentionBlock(nn.Module):
    """
    One stage-2 block.

    Forward:
      tgt : (B, N, C)   — target tokens
      ctx : (B, K*N, C) — all context tokens concatenated
    Returns updated target tokens (B, N, C).
    """

    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: int = 4, dropout: float = 0.0):
        super().__init__()
        self.self_attn  = MHA(embed_dim, num_heads, dropout)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_norm = nn.LayerNorm(embed_dim)
        self.ffn        = FFN(embed_dim, mlp_ratio, dropout)

    def forward(self, tgt: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        tgt = self.self_attn(tgt)                        # within-target self-attn
        out, _ = self.cross_attn(tgt, ctx, ctx)          # target ← context
        tgt = self.cross_norm(tgt + out)
        return self.ffn(tgt)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class ViTInContext3D(nn.Module):
    """
    In-context 3-D segmentation ViT.

    Args:
        image_size    : (D, H, W) of all input volumes.
        patch_size    : (pd, ph, pw) — must divide image_size evenly.
        in_channels   : image channels (1 for CT).
        num_classes   : segmentation output classes (including background).
        embed_dim     : token embedding dimension.
        depth_stage1  : number of within-image transformer blocks.
        depth_stage2  : number of cross-context transformer blocks.
        num_heads     : attention heads.
        mlp_ratio     : FFN hidden-dim multiplier.
        dropout       : attention + FFN dropout.

    Forward signature
    -----------------
        target_img    : (B, 1, D, H, W)      query CT volumes
        context_imgs  : (B, K, 1, D, H, W)   context CT volumes
        context_masks : (B, K, D, H, W)       context binary masks (int / float)

    Returns
        logits : (B, num_classes, D, H, W)
    """

    def __init__(
        self,
        image_size: tuple[int, int, int]  = (64, 64, 64),
        patch_size: tuple[int, int, int]  = (8, 8, 8),
        in_channels: int   = 1,
        num_classes: int   = 2,
        embed_dim: int     = 256,
        depth_stage1: int  = 3,
        depth_stage2: int  = 3,
        num_heads: int     = 8,
        mlp_ratio: int     = 4,
        dropout: float     = 0.1,
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_classes = num_classes

        grid = tuple(i // p for i, p in zip(image_size, patch_size))
        n_tokens = math.prod(grid)

        # Separate embedders for CT images and binary masks; same architecture
        self.img_embed  = PatchEmbed3D(patch_size, in_channels, embed_dim)
        self.mask_embed = PatchEmbed3D(patch_size, 1, embed_dim)

        # Shared positional embedding (same grid for all volumes)
        self.pos_embed = nn.Parameter(torch.zeros(1, n_tokens, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Stage 1: within-image self-attention (weights shared across all volumes)
        self.stage1 = nn.Sequential(*[
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth_stage1)
        ])

        # Stage 2: cross-context attention (target only)
        self.stage2 = nn.ModuleList([
            CrossAttentionBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth_stage2)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        self.decoder = nn.Sequential(
            nn.Conv3d(embed_dim, embed_dim // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(embed_dim // 2, num_classes, 1),
        )

    # ------------------------------------------------------------------

    def forward(
        self,
        target_img:    torch.Tensor,  # (B, 1, D, H, W)
        context_imgs:  torch.Tensor,  # (B, K, 1, D, H, W)
        context_masks: torch.Tensor,  # (B, K, D, H, W)
    ) -> torch.Tensor:

        B, K = context_imgs.shape[:2]

        # ---- Embed target image ----------------------------------------
        tgt_tokens, grid = self.img_embed(target_img)   # (B, N, C)
        tgt_tokens = tgt_tokens + self.pos_embed

        # ---- Embed context pairs ---------------------------------------
        # Flatten batch × context dims for a single batched embed call
        ctx_imgs_flat  = context_imgs.flatten(0, 1)                       # (B*K, 1, D, H, W)
        ctx_masks_flat = context_masks.float().unsqueeze(2).flatten(0, 1) # (B*K, 1, D, H, W)

        ctx_img_tok,  _ = self.img_embed(ctx_imgs_flat)   # (B*K, N, C)
        ctx_mask_tok, _ = self.mask_embed(ctx_masks_flat) # (B*K, N, C)

        # Fuse: appearance + supervision signal + position
        ctx_tokens = ctx_img_tok + ctx_mask_tok + self.pos_embed  # (B*K, N, C)

        # ---- Stage 1: within-image self-attention ----------------------
        # Stack all volumes into one batch dimension: (B + B*K, N, C)
        all_tokens = torch.cat([tgt_tokens, ctx_tokens], dim=0)
        all_tokens = self.stage1(all_tokens)

        # Split back
        tgt_tokens = all_tokens[:B]          # (B, N, C)
        ctx_tokens = all_tokens[B:]          # (B*K, N, C)

        # ---- Stage 2: cross-context attention --------------------------
        # Reshape context: (B*K, N, C) → (B, K*N, C) so target can attend all K contexts
        N, C = ctx_tokens.shape[1], ctx_tokens.shape[2]
        ctx_tokens = ctx_tokens.view(B, K * N, C)

        for block in self.stage2:
            tgt_tokens = block(tgt_tokens, ctx_tokens)

        tgt_tokens = self.norm(tgt_tokens)

        # ---- Decode ----------------------------------------------------
        Gd, Gh, Gw = grid
        feat = tgt_tokens.transpose(1, 2).reshape(B, C, Gd, Gh, Gw)
        feat = F.interpolate(feat, size=self.image_size, mode="trilinear", align_corners=False)
        return self.decoder(feat)             # (B, num_classes, D, H, W)
