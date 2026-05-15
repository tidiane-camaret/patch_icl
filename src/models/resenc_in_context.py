"""
In-context 3-D segmentation with a pluggable encoder backbone.

Architecture overview
---------------------
  Encoder   — either ResEncEncoder (nnUNet ResidualEncoderUNet, 4 stages,
              8× downsample, 2-channel image+mask input) or STUNetEncoder
              (STU-Net 6-stage image encoder + separate 3-D mask encoder,
              32× downsample, pretrained on TotalSegmentator).

              Both encoders expose the same interface:
                  forward(imgs, masks) → list[feature_maps]   (high-res → bottleneck)
                  skip_channels : list[int]
                  bot_features  : int
                  total_stride  : int

  Bottleneck transformer
              Stage-1: within-volume self-attention (shared weights, all volumes).
              Stage-2: cross-context attention — target attends to all K context
                       bottlenecks; context tokens are read-only.

  Decoder   — U-Net skip-connection decoder (trilinear upsample + concat + conv).
              Uses all skip connections produced by the encoder.
              Final 1×1×1 conv → (B, num_classes, D, H, W).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.encoders import ResEncEncoder, STUNetEncoder
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
    In-context 3-D segmentation: pluggable encoder + transformer bottleneck
    + U-Net decoder.

    Args
    ----
        image_size         : (D, H, W) of input volumes.
        in_channels        : image channels (1 for CT).
        num_classes        : segmentation output classes.
        encoder_name       : "resenc" (default) | "stunet"

        --- ResEncEncoder options ---
        features_per_stage : feature widths for the 4 nnUNet stages, e.g. (32,64,128,256).

        --- STUNetEncoder options ---
        stunet_variant     : "small" | "base" | "large" | "huge"
        stunet_pretrained  : path to a STU-Net checkpoint (optional)
        stunet_freeze      : freeze the STU-Net image encoder after loading weights
        mask_fusion        : "additive" | "concat"  (how mask features join image features)

        --- Transformer ---
        depth_stage1       : within-volume self-attention blocks.
        depth_stage2       : cross-context attention blocks.
        num_heads          : attention heads.
        mlp_ratio          : FFN hidden-dim multiplier.
        dropout            : attention + FFN dropout.
        rope_theta         : RoPE base frequency (use ~100 for small grids).

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
        image_size:          tuple[int, int, int] = (64, 64, 64),
        in_channels:         int   = 1,
        num_classes:         int   = 2,
        encoder_name:        str   = "resenc",
        # resenc options
        features_per_stage:  tuple[int, ...] = (32, 64, 128, 256),
        # stunet options
        stunet_variant:      str   = "base",
        stunet_pretrained:   str | None = None,
        stunet_freeze:       bool  = False,
        stunet_num_stages:   int | None = None,
        mask_fusion:         str   = "additive",
        # transformer
        depth_stage1:        int   = 3,
        depth_stage2:        int   = 3,
        num_heads:           int   = 8,
        mlp_ratio:           int   = 4,
        dropout:             float = 0.1,
        rope_theta:          float = 100.0,
        # token conditioning
        num_registers:       int   = 0,
        num_context_layers:  int   = 0,
    ):
        super().__init__()
        self.image_size  = image_size
        self.num_classes = num_classes

        # ---- Encoder -------------------------------------------------------
        if encoder_name == "resenc":
            self.encoder = ResEncEncoder(
                in_channels=in_channels,
                features_per_stage=features_per_stage,
                num_classes=num_classes,
            )
        elif encoder_name == "stunet":
            self.encoder = STUNetEncoder(
                in_channels=in_channels,
                variant=stunet_variant,
                mask_fusion=mask_fusion,
                pretrained=stunet_pretrained,
                freeze_encoder=stunet_freeze,
                num_stages=stunet_num_stages,
            )
        else:
            raise ValueError(f"Unknown encoder_name: {encoder_name!r}. "
                             "Choose 'resenc' or 'stunet'.")

        bot_features  = self.encoder.bot_features
        skip_channels = self.encoder.skip_channels
        total_stride  = self.encoder.total_stride

        # ---- Bottleneck grid for RoPE -------------------------------------
        grid_size = tuple(s // total_stride for s in image_size)

        # ---- Token-role conditioning (target vs context) ------------------
        # Learnable per-role offsets added to bottleneck tokens before Stage 1.
        # Init to zero → no effect at initialisation, learned during training.
        self.target_type_embed  = nn.Parameter(torch.zeros(1, 1, bot_features))
        self.context_type_embed = nn.Parameter(torch.zeros(1, 1, bot_features))

        # ---- Register tokens (global memory for Stage 2) ------------------
        # Appended to the context KV sequence in Stage 2, giving the target a
        # set of global "scratchpad" tokens to attend to alongside spatial ctx.
        if num_registers > 0:
            self.register_tokens = nn.Parameter(
                torch.randn(1, num_registers, bot_features) * 0.02
            )
        self.num_registers = num_registers

        # ---- Context-first layers (optional pre-enrichment) ---------------
        # Applied only to ctx_tok before Stage 1 (target sees context that has
        # already done self-attention — mirrors PatchICL v3's context_layers).
        if num_context_layers > 0:
            self.context_layers: nn.Module = nn.Sequential(*[
                RoPETransformerBlock(
                    bot_features, num_heads, mlp_ratio, dropout, grid_size, rope_theta
                )
                for _ in range(num_context_layers)
            ])
        else:
            self.context_layers = None

        # ---- Stage 1: within-volume self-attention ------------------------
        self.stage1 = nn.Sequential(*[
            RoPETransformerBlock(
                bot_features, num_heads, mlp_ratio, dropout, grid_size, rope_theta
            )
            for _ in range(depth_stage1)
        ])

        # ---- Stage 2: cross-context attention (target only) ---------------
        self.stage2 = nn.ModuleList([
            RoPECrossAttentionBlock(
                bot_features, num_heads, mlp_ratio, dropout, grid_size, rope_theta
            )
            for _ in range(depth_stage2)
        ])

        self.bottleneck_norm = nn.LayerNorm(bot_features)

        # ---- Decoder: one block per skip connection -----------------------
        # Iterates skips from closest-to-bottleneck outward (reversed order).
        self.decoder_stages = nn.ModuleList()
        x_ch = bot_features
        for skip_ch in reversed(skip_channels):
            self.decoder_stages.append(DecoderBlock3D(x_ch + skip_ch, skip_ch))
            x_ch = skip_ch

        self.head = nn.Conv3d(skip_channels[0], num_classes, 1)

    # ------------------------------------------------------------------

    def _encode(self, imgs: torch.Tensor, masks: torch.Tensor) -> list[torch.Tensor]:
        return self.encoder(imgs, masks)

    def forward(
        self,
        target_img:    torch.Tensor,   # (B, 1, D, H, W)
        context_imgs:  torch.Tensor,   # (B, K, 1, D, H, W)
        context_masks: torch.Tensor,   # (B, K, D, H, W)
    ) -> torch.Tensor:

        B, K = context_imgs.shape[:2]

        # ---- Encode target (zero mask) ------------------------------------
        zero_mask = torch.zeros_like(target_img)
        tgt_feats = self._encode(target_img, zero_mask)                       # [s0 … sN]

        # ---- Encode context pairs (flattened B×K batch) ------------------
        ctx_imgs_flat  = context_imgs.flatten(0, 1)                           # (B*K, 1, D, H, W)
        ctx_masks_flat = context_masks.float().unsqueeze(2).flatten(0, 1)     # (B*K, 1, D, H, W)
        ctx_feats = self._encode(ctx_imgs_flat, ctx_masks_flat)               # [s0 … sN]

        # ---- Transformer bottleneck ---------------------------------------
        def to_tokens(f: torch.Tensor) -> torch.Tensor:
            return f.flatten(2).transpose(1, 2)   # (B', C, …) → (B', N, C)

        tgt_tok = to_tokens(tgt_feats[-1])   # (B, N, C)
        ctx_tok = to_tokens(ctx_feats[-1])   # (B*K, N, C)

        # Token-role conditioning: teach the model which volumes have a mask.
        tgt_tok = tgt_tok + self.target_type_embed
        ctx_tok = ctx_tok + self.context_type_embed

        # Context-first layers: enrich context tokens before joint attention.
        if self.context_layers is not None:
            ctx_tok = self.context_layers(ctx_tok)

        # Stage 1: shared within-volume self-attention
        all_tok = torch.cat([tgt_tok, ctx_tok], dim=0)   # (B + B*K, N, C)
        all_tok = self.stage1(all_tok)
        tgt_tok = all_tok[:B]
        ctx_tok = all_tok[B:]

        # Stage 2: target cross-attends to all K context bottlenecks.
        # Register tokens (global memory) are appended to the KV sequence so
        # the target can also attend to a set of learned global scratchpad slots.
        N, C = ctx_tok.shape[1], ctx_tok.shape[2]
        ctx_for_attn = ctx_tok.view(B, K * N, C)         # (B, K*N, C)
        num_spatial_ctx = K * N
        if self.num_registers > 0:
            regs = self.register_tokens.expand(B, -1, -1)
            ctx_for_attn = torch.cat([ctx_for_attn, regs], dim=1)  # (B, K*N+R, C)
        for blk in self.stage2:
            tgt_tok = blk(tgt_tok, ctx_for_attn, num_spatial_ctx=num_spatial_ctx)

        tgt_tok = self.bottleneck_norm(tgt_tok)

        # Reshape to spatial
        d, h, w = tgt_feats[-1].shape[2:]
        x = tgt_tok.transpose(1, 2).reshape(B, C, d, h, w)

        # ---- U-Net decoder (skips from tgt_feats, closest-to-bot first) --
        for i, dec in enumerate(self.decoder_stages):
            x = dec(x, tgt_feats[-(i + 2)])

        return self.head(x)   # (B, num_classes, D, H, W)
