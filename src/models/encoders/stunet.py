"""STU-Net encoder with a separate 3-D mask encoder, fused at the bottleneck.

Reference architecture: https://github.com/openmedlab/STU-Net
Pretrained on TotalSegmentator (104 classes, 1204 volumes, 4000 epochs).

The image encoder is reproduced exactly from STU-Net so pretrained weights
can be loaded directly (conv_blocks_context.* keys).  The mask encoder is a
new SAM-style 3-D CNN; its weights are always randomly initialised.

Fusion is additive by default: bottleneck = img_feat + mask_feat.
A zero mask (target) leaves img_feat unchanged; context masks inject information.

Variants
--------
    small : dims=[16,32,64,128,256,256],  depth=[1]*6  — 14.6 M params
    base  : dims=[32,64,128,256,512,512], depth=[1]*6  — 58.3 M params
    large : dims=[64,128,256,512,1024,1024], depth=[2]*6  — 440 M params
    huge  : dims=[96,192,384,768,1536,1536], depth=[3]*6  — 1457 M params
"""

from __future__ import annotations

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Variant presets
# ---------------------------------------------------------------------------

_VARIANTS: dict[str, dict] = {
    "small": {"dims": [16,  32,   64,  128,  256,  256], "depth": [1]*6},
    "base":  {"dims": [32,  64,  128,  256,  512,  512], "depth": [1]*6},
    "large": {"dims": [64, 128,  256,  512, 1024, 1024], "depth": [2]*6},
    "huge":  {"dims": [96, 192,  384,  768, 1536, 1536], "depth": [3]*6},
}

# 5 stride-2 pooling ops → 32× total downsampling
_DEFAULT_STRIDES = [[2, 2, 2]] * 5


# ---------------------------------------------------------------------------
# Building blocks  (names match STU-Net exactly for weight compatibility)
# ---------------------------------------------------------------------------

class _BasicResBlock(nn.Module):
    """Residual block from STU-Net.  Attribute names are preserved verbatim
    so that ``conv_blocks_context.*`` weights load without key remapping."""

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        stride: int = 1,
        use_1x1conv: bool = False,
    ):
        super().__init__()
        self.conv1 = nn.Conv3d(input_channels, output_channels, kernel_size,
                               stride=stride, padding=padding)
        self.norm1 = nn.InstanceNorm3d(output_channels, affine=True)
        self.act1  = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv3d(output_channels, output_channels, kernel_size,
                               padding=padding)
        self.norm2 = nn.InstanceNorm3d(output_channels, affine=True)
        self.act2  = nn.LeakyReLU(inplace=True)
        self.conv3 = (nn.Conv3d(input_channels, output_channels, 1, stride=stride)
                      if use_1x1conv else None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.act1(self.norm1(self.conv1(x)))
        y = self.norm2(self.conv2(y))
        if self.conv3 is not None:
            x = self.conv3(x)
        return self.act2(y + x)


# ---------------------------------------------------------------------------
# Image-only encoder  (conv_blocks_context naming = STU-Net checkpoint compat)
# ---------------------------------------------------------------------------

class _ImageEncoder(nn.Module):
    """6-stage STU-Net encoder operating on raw CT images (1 channel)."""

    def __init__(
        self,
        in_channels: int,
        dims: list[int],
        depth: list[int],
        strides: list[list[int]],
        kernel_size: int = 3,
    ):
        super().__init__()
        num_pool = len(strides)               # 5
        assert len(dims) == num_pool + 1      # 6
        pad = kernel_size // 2

        self.conv_blocks_context = nn.ModuleList()

        # Stage 0: no spatial downsampling
        self.conv_blocks_context.append(nn.Sequential(
            _BasicResBlock(in_channels, dims[0], kernel_size, pad, use_1x1conv=True),
            *[_BasicResBlock(dims[0], dims[0], kernel_size, pad)
              for _ in range(depth[0] - 1)],
        ))

        # Stages 1 … num_pool: stride comes from pool_op_kernel_sizes
        for d in range(1, num_pool + 1):
            s = strides[d - 1]
            stride = s[0] if isinstance(s, (list, tuple)) else s
            self.conv_blocks_context.append(nn.Sequential(
                _BasicResBlock(dims[d-1], dims[d], kernel_size, pad,
                               stride=stride, use_1x1conv=True),
                *[_BasicResBlock(dims[d], dims[d], kernel_size, pad)
                  for _ in range(depth[d] - 1)],
            ))

    def forward(
        self, x: torch.Tensor, num_stages: int | None = None
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Returns (bottleneck, [s0, …, s_{num_stages-2}]).

        num_stages: how many stages to run (default: all).  Must be ≥ 2.
        """
        n = num_stages if num_stages is not None else len(self.conv_blocks_context)
        skips: list[torch.Tensor] = []
        for stage in self.conv_blocks_context[:n - 1]:
            x = stage(x)
            skips.append(x)
        x = self.conv_blocks_context[n - 1](x)
        return x, skips

    def load_pretrained_weights(self, state: dict[str, torch.Tensor]) -> None:
        """Load conv_blocks_context weights from a STU-Net state dict."""
        prefix = "conv_blocks_context."
        enc_state = {k[len(prefix):]: v
                     for k, v in state.items()
                     if k.startswith(prefix)}
        missing, unexpected = self.conv_blocks_context.load_state_dict(
            enc_state, strict=False
        )
        if missing:
            print(f"[STUNet] missing encoder keys ({len(missing)}): {missing[:3]} …")
        if unexpected:
            print(f"[STUNet] unexpected encoder keys ({len(unexpected)}): {unexpected[:3]} …")


# ---------------------------------------------------------------------------
# 3-D mask encoder  (SAM-style, stride-2 CNN)
# ---------------------------------------------------------------------------

class _Mask3DEncoder(nn.Module):
    """Encodes a binary mask to spatial features at the same resolution as
    the STU-Net bottleneck (1/total_stride of input).

    Architecture mirrors SAM's mask_downscaling but in 3-D:
        num_pools × [Conv3d stride-2 → InstanceNorm3d → GELU]
        1×1×1 projection to embed_dim
    """

    def __init__(self, embed_dim: int, num_pools: int):
        super().__init__()
        channels = [1, 16, 32, 64, 128, 256]
        # Extend channel list if num_pools > len(channels)-1
        while len(channels) <= num_pools:
            channels.append(channels[-1] * 2)

        layers: list[nn.Module] = []
        for i in range(num_pools):
            layers += [
                nn.Conv3d(channels[i], channels[i + 1], kernel_size=2, stride=2),
                nn.InstanceNorm3d(channels[i + 1], affine=True),
                nn.GELU(),
            ]
        layers.append(nn.Conv3d(channels[num_pools], embed_dim, kernel_size=1))
        self.net = nn.Sequential(*layers)

    def forward(self, mask: torch.Tensor) -> torch.Tensor:
        """mask: [B, 1, D, H, W]  →  [B, embed_dim, D/2^n, H/2^n, W/2^n]"""
        return self.net(mask)


# ---------------------------------------------------------------------------
# Public encoder
# ---------------------------------------------------------------------------

class STUNetEncoder(nn.Module):
    """STU-Net image encoder + separate 3-D mask encoder, fused at the bottleneck.

    Image path : STU-Net conv_blocks_context (pretrained weights compatible).
    Mask path  : SAM-style 3-D CNN (always randomly initialised).
    Fusion     : additive (default) or concat+proj.

    Args
    ----
        in_channels    : image channels (1 for CT).
        variant        : "small" | "base" | "large" | "huge".
        mask_fusion    : "additive" | "concat".
        pretrained     : path to a STU-Net checkpoint (.model or state-dict .pt).
                         Only encoder weights (conv_blocks_context.*) are loaded.
        freeze_encoder : if True, freeze image encoder after loading pretrained.
        strides        : pool_op_kernel_sizes (default: [[2,2,2]]*5 → 32× stride).

    Properties (required by ResEncInContext3D)
    ------------------------------------------
        skip_channels : list[int]  — channel widths of skip outputs, high-res first.
        bot_features  : int        — bottleneck channel width.
        total_stride  : int        — spatial downsampling factor.
    """

    def __init__(
        self,
        in_channels:    int = 1,
        variant:        str = "base",
        mask_fusion:    str = "additive",
        pretrained:     str | None = None,
        freeze_encoder: bool = False,
        strides:        list | None = None,
        num_stages:     int | None = None,
    ):
        """
        num_stages : how many encoder stages to run (2 … total_stages).
                     Controls the spatial downsampling factor (2^(num_stages-1))
                     and the bottleneck channel width.
                     Default: all stages (6 for standard STU-Net → 32× stride).

                     Examples for STU-Net-B (dims=[32,64,128,256,512,512]):
                         num_stages=4 → 8×  stride, bottleneck=256 ch  (matches ResEnc at 64³)
                         num_stages=5 → 16× stride, bottleneck=512 ch
                         num_stages=6 → 32× stride, bottleneck=512 ch  (default)
        """
        super().__init__()
        assert variant in _VARIANTS, f"variant must be one of {list(_VARIANTS)}"
        assert mask_fusion in ("additive", "concat")

        cfg = _VARIANTS[variant]
        dims: list[int]  = cfg["dims"]
        depth: list[int] = cfg["depth"]
        strides = strides or _DEFAULT_STRIDES
        total_stages = len(strides) + 1          # 6

        if num_stages is None:
            num_stages = total_stages
        assert 2 <= num_stages <= total_stages, (
            f"num_stages must be between 2 and {total_stages}, got {num_stages}"
        )
        self._num_stages = num_stages
        num_pools = num_stages - 1               # stride-2 ops actually executed

        self.image_encoder = _ImageEncoder(in_channels, dims, depth, strides)
        self.mask_encoder  = _Mask3DEncoder(dims[num_stages - 1], num_pools)

        self.mask_fusion   = mask_fusion
        self.skip_channels = dims[:num_stages - 1]   # [d0 … d_{n-2}]
        self.bot_features  = dims[num_stages - 1]    # d_{n-1}
        self.total_stride  = 2 ** num_pools

        if mask_fusion == "concat":
            self.fusion_proj = nn.Conv3d(self.bot_features * 2, self.bot_features, kernel_size=1)

        if pretrained is not None:
            self._load_pretrained(pretrained)

        if freeze_encoder:
            for p in self.image_encoder.parameters():
                p.requires_grad_(False)

    # ------------------------------------------------------------------

    def _load_pretrained(self, path: str) -> None:
        """Load STU-Net encoder weights.  Accepts either a pickled nnUNet v1
        model object or a plain state dict."""
        import os
        print(f"[STUNet] loading pretrained encoder from {os.path.basename(path)} …")
        obj = torch.load(path, map_location="cpu", weights_only=False)
        state: dict[str, torch.Tensor] = (
            obj if isinstance(obj, dict) else obj.state_dict()
        )
        self.image_encoder.load_pretrained_weights(state)
        print("[STUNet] encoder weights loaded.")

    # ------------------------------------------------------------------

    def forward(self, imgs: torch.Tensor, masks: torch.Tensor) -> list[torch.Tensor]:
        """
        Args
        ----
            imgs  : (B, 1, D, H, W) — CT image
            masks : (B, 1, D, H, W) — binary mask; zeros for the target volume

        Returns
        -------
            [s0, s1, s2, s3, s4, s5_fused]   — ordered high-res → bottleneck
        """
        bottleneck, skips = self.image_encoder(imgs, num_stages=self._num_stages)

        mask_feat = self.mask_encoder(masks)

        if self.mask_fusion == "additive":
            bottleneck = bottleneck + mask_feat
        else:  # concat
            bottleneck = self.fusion_proj(torch.cat([bottleneck, mask_feat], dim=1))

        return skips + [bottleneck]
