"""VoComni nnUNet PlainConvUNet encoder for patch_icl.

Wraps the PlainConvUNet CNN backbone trained with VoComni supervised labels
(20K CT volumes, 20 organ/tumor classes + background = 21 classes).

Architecture
------------
    Type        : PlainConvUNet (nnUNet-style CNN, no residual connections)
    Input       : (B, 1, D, H, W)
    Conv        : 3×3×3, InstanceNorm3d (affine), LeakyReLU
    n_stages    : 6
    n_conv/stage: 2
    Channels    : 1 → 32 → 64 → 128 → 256 → 320 → 320
    Strides     : 1, 2, 2, 2, 2, 2  (stage-0 full-res, stages 1-5 stride-2)
    Total stride: 32×

Outputs (6 levels, shallow → deep)
------------------------------------
    level 0 : (B,  32, D,    H,    W   )   stride  1
    level 1 : (B,  64, D/2,  H/2,  W/2)   stride  2
    level 2 : (B, 128, D/4,  H/4,  W/4)   stride  4
    level 3 : (B, 256, D/8,  H/8,  W/8)   stride  8
    level 4 : (B, 320, D/16, H/16, W/16)  stride 16
    level 5 : (B, 320, D/32, H/32, W/32)  stride 32  ← bottleneck

Interface
---------
    skip_channels : [32, 64, 128, 256, 320]
    bot_features  : 320
    total_stride  : 32
    forward(imgs, masks) -> list[torch.Tensor]   (masks are ignored)

Pretrained checkpoint
---------------------
    VoComni_nnunet.pt  (31M encoder params / 88M full model)
    Download from https://huggingface.co/Luffy503/VoCo

Input normalisation
-------------------
No internal renormalisation.  Expects nnUNet z-score values (~[-1.7, +3.5]),
consistent with what totalseg_dataloader_incontext.py produces.
InstanceNorm at each stage provides robustness to minor scale drift.

Checkpoint key structure
------------------------
    The checkpoint stores the full PlainConvUNet (encoder + decoder).
    Encoder weights are under the 'encoder.*' prefix.
    The decoder also stores a reference copy at 'decoder.encoder.*' — ignored.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def _load_nnunet_encoder(encoder: nn.Module, ckpt: dict) -> None:
    """Load the encoder portion of a VoComni_nnunet checkpoint.

    The checkpoint stores the full PlainConvUNet state dict.  Encoder weights
    are prefixed with 'encoder.'; we strip that prefix before loading.
    """
    prefix = "encoder."
    enc_state = {k[len(prefix):]: v for k, v in ckpt.items() if k.startswith(prefix)}
    current = encoder.state_dict()
    merged = {
        k: enc_state[k]
        if k in enc_state and enc_state[k].size() == current[k].size()
        else current[k]
        for k in current
    }
    encoder.load_state_dict(merged, strict=True)


# ---------------------------------------------------------------------------
# Public encoder
# ---------------------------------------------------------------------------

class VoComniNNUNetEncoder(nn.Module):
    """PlainConvUNet CNN encoder with VoComni supervised pretrained weights.

    Args
    ----
        ckpt_path       : path to VoComni_nnunet.pt, or None for random weights.
        freeze_encoder  : freeze all weights after loading (default True).
        compile_model   : wrap with torch.compile (default True; ~2× speedup).
    """

    def __init__(
        self,
        ckpt_path:      str | Path | None = None,
        freeze_encoder: bool = True,
        compile_model:  bool = True,
    ):
        super().__init__()
        from dynamic_network_architectures.architectures.unet import PlainConvUNet

        full_model = PlainConvUNet(
            input_channels=1,
            n_stages=6,
            features_per_stage=[32, 64, 128, 256, 320, 320],
            conv_op=nn.Conv3d,
            kernel_sizes=3,
            strides=[1, 2, 2, 2, 2, 2],
            n_conv_per_stage=2,
            num_classes=21,
            n_conv_per_stage_decoder=2,
            norm_op=nn.InstanceNorm3d,
            norm_op_kwargs={"affine": True},
            nonlin=nn.LeakyReLU,
            nonlin_kwargs={"inplace": True},
        )

        if ckpt_path is not None:
            ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
            _load_nnunet_encoder(full_model.encoder, ckpt)
            print(f"[VoComniNNUNet] PlainConvEncoder loaded from {Path(ckpt_path).name}")
        else:
            print("[VoComniNNUNet] PlainConvEncoder — no checkpoint, random weights")

        enc = full_model.encoder
        enc.eval()
        if freeze_encoder:
            for p in enc.parameters():
                p.requires_grad_(False)

        self._encoder = torch.compile(enc) if compile_model else enc

        self.skip_channels = [32, 64, 128, 256, 320]
        self.bot_features  = 320
        self.total_stride  = 32

    # ------------------------------------------------------------------

    def forward(
        self, imgs: torch.Tensor, masks: torch.Tensor | None = None
    ) -> list[torch.Tensor]:
        """
        Args
        ----
            imgs  : (B, 1, D, H, W) — CT image; expects nnUNet z-score values (~[-2, +4]),
                    consistent with what totalseg_dataloader_incontext.py provides.
            masks : ignored

        Returns
        -------
            list of 6 tensors ordered shallow → deep:
              [  (B,  32, D,    H,    W   ),
                 (B,  64, D/2,  H/2,  W/2),
                 (B, 128, D/4,  H/4,  W/4),
                 (B, 256, D/8,  H/8,  W/8),
                 (B, 320, D/16, H/16, W/16),
                 (B, 320, D/32, H/32, W/32)  ]
        """
        return self._encoder(imgs)
