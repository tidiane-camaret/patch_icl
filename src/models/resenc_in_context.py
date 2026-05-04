"""
In-context 3-D segmentation with a residual encoder backbone.

Architecture overview
---------------------
  Encoder   — nnUNet ResidualEncoderUNet.encoder (dynamic_network_architectures),
              4 stages, nnUNetResEncM block counts [1,3,4,6], uniform features.
              Input is 2-channel: (image, mask).  For the target the mask channel
              is zeros; for context volumes it carries the binary label.
              Spatial resolutions: [N, N/2, N/4, N/8].

  Bottleneck transformer
              Stage-1: within-volume self-attention (shared weights, all volumes).
              Stage-2: cross-context attention — target attends to all K context
                       bottlenecks; context tokens are read-only.

  Decoder   — U-Net skip-connection decoder (trilinear upsample + concat + conv).
              Skip connections from encoder stages 2 → 1 → 0.
              Final 1×1×1 conv → (B, num_classes, D, H, W).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet

from src.rope import RoPETransformerBlock, RoPECrossAttentionBlock


# ---------------------------------------------------------------------------
# Decoder block
# ---------------------------------------------------------------------------

class DecoderBlock3D(nn.Module):
    """Trilinear upsample × 2 → concat skip → Conv → Norm → Act."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=True),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
        return self.block(torch.cat([x, skip], dim=1))


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class ResEncInContext3D(nn.Module):
    """
    In-context 3-D segmentation: nnUNet residual encoder + transformer bottleneck
    + U-Net decoder.

    The encoder follows the nnUNetResEncM preset:
        stages       : 4
        strides      : (1, 2, 2, 2)
        n_blocks     : (1, 3, 4, 6)
        features     : (features,) * 4   — uniform width, default 256

    Args
    ----
        image_size    : (D, H, W) of input volumes (must be divisible by 8).
        in_channels   : image channels (1 for CT).
        num_classes   : segmentation output classes.
        features_per_stage : feature widths at each encoder stage, e.g. (32, 64, 128, 256).
                             len must equal n_stages (4). Last entry is the bottleneck width.
        depth_stage1  : within-volume self-attention blocks at the bottleneck.
        depth_stage2  : cross-context attention blocks at the bottleneck.
        num_heads     : attention heads.
        mlp_ratio     : FFN hidden-dim multiplier.
        dropout       : attention + FFN dropout.

    Forward
    -------
        target_img    : (B, 1, D, H, W)
        context_imgs  : (B, K, 1, D, H, W)
        context_masks : (B, K, D, H, W)   — binary, int or float

    Returns
    -------
        logits        : (B, num_classes, D, H, W)
    """

    def __init__(
        self,
        image_size:   tuple[int, int, int] = (64, 64, 64),
        in_channels:  int   = 1,
        num_classes:  int   = 2,
        features_per_stage: tuple[int, ...] = (32, 64, 128, 256),
        depth_stage1: int   = 3,
        depth_stage2: int   = 3,
        num_heads:    int   = 8,
        mlp_ratio:    int   = 4,
        dropout:      float = 0.1,
        rope_theta:   float = 100.0,
    ):
        super().__init__()
        assert len(features_per_stage) == 4, "expected 4 encoder stages"
        self.image_size        = image_size
        self.num_classes       = num_classes
        self.features_per_stage = features_per_stage
        bot_features = features_per_stage[-1]

        # Shared encoder: (image + mask) → list of 4 feature maps
        _unet = ResidualEncoderUNet(
            input_channels=in_channels + 1,
            n_stages=4,
            features_per_stage=features_per_stage,
            conv_op=nn.Conv3d,
            kernel_sizes=3,
            strides=(1, 2, 2, 2),
            n_blocks_per_stage=(1, 3, 4, 6),
            num_classes=num_classes,
            n_conv_per_stage_decoder=(1, 1, 1),
            conv_bias=True,
            norm_op=nn.InstanceNorm3d,
            norm_op_kwargs={"eps": 1e-5, "affine": True},
            dropout_op=None,
            nonlin=nn.LeakyReLU,
            nonlin_kwargs={"inplace": True},
        )
        self.encoder = _unet.encoder  # forward: x → [s0, s1, s2, s3]

        # Bottleneck grid: image_size / 8  (strides 1,2,2,2 → 3 halvings)
        grid_size = tuple(s // 8 for s in image_size)

        # Stage 1: within-volume self-attention (shared, target + all contexts)
        self.stage1 = nn.Sequential(*[
            RoPETransformerBlock(bot_features, num_heads, mlp_ratio, dropout, grid_size, rope_theta)
            for _ in range(depth_stage1)
        ])

        # Stage 2: cross-context attention (target only)
        self.stage2 = nn.ModuleList([
            RoPECrossAttentionBlock(bot_features, num_heads, mlp_ratio, dropout, grid_size, rope_theta)
            for _ in range(depth_stage2)
        ])

        self.bottleneck_norm = nn.LayerNorm(bot_features)

        # Decoder: N/8 → N/4 → N/2 → N
        # Each block: upsample(x) + concat(skip) → out = skip channels
        self.decoder_stages = nn.ModuleList()
        x_ch = bot_features
        for i in range(len(features_per_stage) - 1, 0, -1):
            skip_ch = features_per_stage[i - 1]
            self.decoder_stages.append(DecoderBlock3D(x_ch + skip_ch, skip_ch))
            x_ch = skip_ch

        self.head = nn.Conv3d(features_per_stage[0], num_classes, 1)

    # ------------------------------------------------------------------

    def _encode(self, imgs: torch.Tensor, masks: torch.Tensor) -> list[torch.Tensor]:
        return self.encoder(torch.cat([imgs, masks], dim=1))

    def forward(
        self,
        target_img:    torch.Tensor,  # (B, 1, D, H, W)
        context_imgs:  torch.Tensor,  # (B, K, 1, D, H, W)
        context_masks: torch.Tensor,  # (B, K, D, H, W)
    ) -> torch.Tensor:

        B, K = context_imgs.shape[:2]

        # ---- Encode target (empty mask) --------------------------------
        tgt_feats = self._encode(target_img, torch.zeros_like(target_img))  # [s0..s3]

        # ---- Encode context pairs (flattened B×K batch) ----------------
        ctx_imgs_flat  = context_imgs.flatten(0, 1)                        # (B*K, 1, D, H, W)
        ctx_masks_flat = context_masks.float().unsqueeze(2).flatten(0, 1)  # (B*K, 1, D, H, W)
        ctx_feats = self._encode(ctx_imgs_flat, ctx_masks_flat)             # [s0..s3]

        # ---- Transformer bottleneck ------------------------------------
        def to_tokens(f: torch.Tensor) -> torch.Tensor:
            return f.flatten(2).transpose(1, 2)   # (B', C, d, h, w) → (B', N, C)

        tgt_tok = to_tokens(tgt_feats[3])   # (B, N, C)
        ctx_tok = to_tokens(ctx_feats[3])   # (B*K, N, C)

        # Stage 1: shared within-volume self-attention
        all_tok = torch.cat([tgt_tok, ctx_tok], dim=0)       # (B + B*K, N, C)
        all_tok = self.stage1(all_tok)
        tgt_tok = all_tok[:B]
        ctx_tok = all_tok[B:]

        # Stage 2: target cross-attends to all K context bottlenecks
        N, C = ctx_tok.shape[1], ctx_tok.shape[2]
        ctx_tok = ctx_tok.view(B, K * N, C)                  # (B, K*N, C)
        for blk in self.stage2:
            tgt_tok = blk(tgt_tok, ctx_tok)

        tgt_tok = self.bottleneck_norm(tgt_tok)

        # Reshape to spatial: (B, C, D/8, H/8, W/8)
        d, h, w = tgt_feats[3].shape[2:]
        x = tgt_tok.transpose(1, 2).reshape(B, C, d, h, w)

        # ---- U-Net decoder (skips from encoder stages 2, 1, 0) --------
        for i, dec in enumerate(self.decoder_stages):
            x = dec(x, tgt_feats[2 - i])

        return self.head(x)                                  # (B, num_classes, D, H, W)
