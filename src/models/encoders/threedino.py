"""3DINO ViT-Large-3D pretrained encoder for patch_icl.

Wraps DinoVisionTransformer3d from the 3DINO checkpoint (ViT-Large, patch_size=16,
trained with DINO self-supervised learning on 3-D medical images).

Architecture summary
--------------------
    embed_dim  : 1024
    depth      : 24 transformer blocks in 4 chunks of 6
    patch_size : 16 × 16 × 16
    stride     : 16× (single spatial scale)

Outputs
-------
    get_intermediate_layers(n=n_last_blocks, reshape=True) returns n tensors,
    one from each of the last n block-chunk boundaries, all at the same
    resolution (B, 1024, D//16, H//16, W//16).

Interface (same as STUNetEncoder / NNInteractiveEncoder)
---------------------------------------------------------
    skip_channels : [1024] * (n_last_blocks - 1)
    bot_features  : 1024
    total_stride  : 16
    forward(imgs, masks) -> list[torch.Tensor]   (masks are ignored)

Input normalisation
-------------------
3DINO trains with percentile clip to [-1, 1]:
    lo, hi = quantile(x, 0.0005), quantile(x, 0.9995)
    x = clip((x - lo) / (hi - lo) * 2 - 1, -1, 1)
Applied per volume inside forward(); the caller does NOT need to pre-normalise.

Requirements
------------
    /home/dpxuser/repos/3DINO  must be on sys.path (added automatically).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

_DINO_ROOT = Path("/home/dpxuser/repos/3DINO")
_EMBED_DIM = 1024   # vit_large_3d


# ---------------------------------------------------------------------------
# Thin wrapper that routes get_intermediate_layers through nn.Module.forward
# so that torch.compile covers the full ViT forward pass.
# ---------------------------------------------------------------------------

class _IntermediateLayerWrapper(nn.Module):
    def __init__(self, model: nn.Module, n: int):
        super().__init__()
        self.model = model
        self.n = n

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return self.model.get_intermediate_layers(
            x, n=self.n, reshape=True, norm=True
        )


# ---------------------------------------------------------------------------
# Public encoder
# ---------------------------------------------------------------------------

class ThreeDINOEncoder(nn.Module):
    """3DINO ViT-Large-3D encoder for patch_icl.

    Args
    ----
        ckpt_path       : path to the 3DINO weights file (*.pth).
                          Expected to contain a 'teacher' key with 'backbone.*' sub-keys.
        n_last_blocks   : how many block-group outputs to return (1 … 4).
                          All outputs are at the same D//16 resolution; choosing
                          more gives shallower features in the earlier list entries.
        freeze_encoder  : freeze all ViT weights (default True).
        compile_model   : wrap with torch.compile for kernel-fusion speedup.
        device          : device string for initial weight loading.
    """

    def __init__(
        self,
        ckpt_path:      str | Path,
        n_last_blocks:  int  = 4,
        freeze_encoder: bool = True,
        compile_model:  bool = False,
        device:         str  = "cpu",
    ):
        super().__init__()
        assert 1 <= n_last_blocks <= 4, "n_last_blocks must be 1 … 4"

        dino_root = str(_DINO_ROOT)
        if dino_root not in sys.path:
            sys.path.insert(0, dino_root)

        from dinov2.models.vision_transformer import vit_large_3d
        from dinov2.utils.utils import load_pretrained_weights

        # Instantiate with the same kwargs as ssl3d_default_config
        vit = vit_large_3d(
            patch_size=16,
            img_size=112,         # training crop size; pos-encoding interpolates to others
            init_values=1e-5,
            ffn_layer="mlp",
            block_chunks=4,
            qkv_bias=True,
            proj_bias=True,
            ffn_bias=True,
        )
        load_pretrained_weights(vit, str(ckpt_path), "teacher")
        vit.eval()
        print(f"[3DINO] ViT-Large-3D loaded from {Path(ckpt_path).name}")

        if freeze_encoder:
            for p in vit.parameters():
                p.requires_grad_(False)

        wrapper = _IntermediateLayerWrapper(vit, n_last_blocks)
        self._extractor = torch.compile(wrapper) if compile_model else wrapper

        # Public interface
        self.skip_channels = [_EMBED_DIM] * (n_last_blocks - 1)
        self.bot_features  = _EMBED_DIM
        self.total_stride  = 16

    # ------------------------------------------------------------------

    @staticmethod
    def _normalise(imgs: torch.Tensor) -> torch.Tensor:
        """Per-volume percentile clip to [-1, 1] (3DINO training convention)."""
        out = imgs.float().clone()
        for b in range(out.shape[0]):
            v = out[b]
            lo = torch.quantile(v, 0.0005)
            hi = torch.quantile(v, 0.9995)
            out[b] = ((v - lo) / (hi - lo + 1e-8) * 2 - 1).clamp(-1, 1)
        return out

    # ------------------------------------------------------------------

    def forward(
        self, imgs: torch.Tensor, masks: torch.Tensor | None = None
    ) -> list[torch.Tensor]:
        """
        Args
        ----
            imgs  : (B, 1, D, H, W) — CT image (any normalisation; re-normed internally)
            masks : ignored — 3DINO is a pure image encoder

        Returns
        -------
            list of n_last_blocks tensors, each (B, 1024, D//16, H//16, W//16),
            ordered shallow (earliest block group) → deep (last block group).
        """
        x = self._normalise(imgs)
        # _extractor.forward returns a tuple; convert to list for interface compat
        feats = self._extractor(x)
        return list(feats)
