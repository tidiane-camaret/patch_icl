"""NNInteractive pretrained encoder for patch_icl.

Wraps the ResidualEncoder from the nnInteractive v1.0 checkpoint
(8-channel input: 1 image + 7 interaction slots, trained on interactive
segmentation across CT, MRI, PET, and microscopy).

Two mask injection modes
------------------------
    ch1       — pack (image, mask, 0, 0, 0, 0, 0, 0) as the 8-channel input.
                Uses the model's native "current segmentation" channel (ch1),
                which is exactly how nnInteractive feeds a prior prediction.
    separate  — image in ch0 only (ch1-7 = 0); mask is encoded by a lightweight
                SAM-style 3-D CNN and fused at the bottleneck — identical to
                STUNetEncoder's mask path.

Interface (same as STUNetEncoder)
-----------------------------------
    skip_channels : list[int]  — feature widths, high-res first
    bot_features  : int        — bottleneck channel width
    total_stride  : int        — spatial downsampling factor
    forward(imgs, masks) -> list[torch.Tensor]

Encoder stages (v1.0, all 6 stages)
--------------------------------------
    stage  features  stride  output at 192³
    0      32        1×      192³
    1      64        2×       96³
    2      128       2×       48³
    3      256       2×       24³
    4      320       2×       12³
    5      320       2×        6³   ← bottleneck

Input normalisation
--------------------
nnInteractive trains with per-volume z-score normalisation (clip to ±3σ then
z-score). Apply the same before passing imgs:
    imgs = (imgs - imgs.mean()) / (imgs.std() + 1e-8)
    imgs = imgs.clamp(-3, 3)

Requirements
------------
    pip install -e /home/dpxuser/repos/nnInteractive   (or equivalent)
    nnunetv2, dynamic_network_architectures (already in pyproject.toml)
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn

from src.models.encoders.stunet import _Mask3DEncoder

# Feature dimensions per stage for nnInteractive v1.0
_DIMS = [32, 64, 128, 256, 320, 320]


# ---------------------------------------------------------------------------
# Checkpoint loader
# ---------------------------------------------------------------------------

def _load_encoder(ckpt_dir: str | Path, device: str = "cpu") -> nn.Module:
    """Load the pretrained ResidualEncoder from an nnInteractive checkpoint folder.

    Returns the encoder (ResidualEncoder) only — the decoder is discarded.
    Requires the nninteractive package to be installed.
    """
    ckpt_dir = Path(ckpt_dir)

    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    from nnInteractive.trainer.nnInteractiveTrainer import nnInteractiveTrainer_stub

    plans   = json.loads((ckpt_dir / "plans.json").read_text())
    dataset = json.loads((ckpt_dir / "dataset.json").read_text())
    ckpt    = torch.load(
        ckpt_dir / "fold_0" / "checkpoint_final.pth",
        map_location=device, weights_only=False,
    )

    plans_mgr  = PlansManager(plans)
    config_mgr = plans_mgr.get_configuration("3d_fullres_ps192")

    # num_input_channels=1 (CT image); the stub adds +7 interaction channels internally.
    # num_output_channels=2 (BG + FG).
    network = nnInteractiveTrainer_stub.build_network_architecture(
        plans_mgr, config_mgr,
        num_input_channels=1, num_output_channels=2,
        enable_deep_supervision=False,
    )
    network.load_state_dict(ckpt["network_weights"])
    print(f"[NNInteractive] encoder loaded from {ckpt_dir.name}")
    return network.encoder   # ResidualEncoder


# ---------------------------------------------------------------------------
# Public encoder
# ---------------------------------------------------------------------------

class NNInteractiveEncoder(nn.Module):
    """NNInteractive pretrained 3-D encoder for patch_icl.

    Args
    ----
        ckpt_dir        : path to the nnInteractive_v1.0 checkpoint folder.
        mask_injection  : "ch1" | "separate"  (see module docstring).
        mask_fusion     : "additive" | "concat"  (only used for mask_injection="separate").
        freeze_encoder  : freeze all pretrained encoder weights (default True).
        num_stages      : how many encoder stages to run (2 … 6, default 6).
                          Controls downsampling depth:
                              num_stages=4 → 8×  stride, bottleneck 256-ch
                              num_stages=5 → 16× stride, bottleneck 320-ch
                              num_stages=6 → 32× stride, bottleneck 320-ch
        device          : device string for loading the checkpoint.
    """

    def __init__(
        self,
        ckpt_dir:       str | Path,
        mask_injection: str  = "ch1",
        mask_fusion:    str  = "additive",
        freeze_encoder: bool = True,
        num_stages:     int | None = None,
        device:         str  = "cpu",
    ):
        super().__init__()
        assert mask_injection in ("ch1", "separate"), (
            "mask_injection must be 'ch1' or 'separate'"
        )
        assert mask_fusion in ("additive", "concat")

        enc = _load_encoder(ckpt_dir, device=device)

        total_stages = len(enc.stages)   # 6
        if num_stages is None:
            num_stages = total_stages
        assert 2 <= num_stages <= total_stages, (
            f"num_stages must be 2..{total_stages}, got {num_stages}"
        )
        self._num_stages = num_stages

        # Keep only the parts we need
        self.stem   = enc.stem    # Conv3d(8 → 32) or None
        self.stages = enc.stages  # full ModuleList; we index [:num_stages]

        # Public interface properties
        self.skip_channels = _DIMS[:num_stages - 1]    # high-res skips
        self.bot_features  = _DIMS[num_stages - 1]     # bottleneck width
        # stage-0 is stride-1; stages 1..n-1 are stride-2 each
        self.total_stride  = 2 ** (num_stages - 1)

        self.mask_injection = mask_injection
        self.mask_fusion    = mask_fusion

        if mask_injection == "separate":
            num_pools = num_stages - 1
            self.mask_encoder = _Mask3DEncoder(self.bot_features, num_pools)
            if mask_fusion == "concat":
                self.fusion_proj = nn.Conv3d(
                    self.bot_features * 2, self.bot_features, kernel_size=1
                )

        if freeze_encoder:
            if self.stem is not None:
                for p in self.stem.parameters():
                    p.requires_grad_(False)
            for p in self.stages.parameters():
                p.requires_grad_(False)

    # ------------------------------------------------------------------

    def _encode(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Run stem + stages[:num_stages]. Returns (bottleneck, skips)."""
        if self.stem is not None:
            x = self.stem(x)
        skips: list[torch.Tensor] = []
        for stage in self.stages[:self._num_stages - 1]:
            x = stage(x)
            skips.append(x)
        bottleneck = self.stages[self._num_stages - 1](x)
        return bottleneck, skips

    # ------------------------------------------------------------------

    def forward(
        self, imgs: torch.Tensor, masks: torch.Tensor
    ) -> list[torch.Tensor]:
        """
        Args
        ----
            imgs  : (B, 1, D, H, W) — z-score normalised CT image
            masks : (B, 1, D, H, W) — binary mask; zeros for the target volume

        Returns
        -------
            [s0, s1, …, s_{n-2}, bottleneck]  ordered high-res → low-res
        """
        B, _, D, H, W = imgs.shape
        x = torch.zeros(B, 8, D, H, W, device=imgs.device, dtype=imgs.dtype)
        x[:, 0] = imgs[:, 0]
        if self.mask_injection == "ch1":
            x[:, 1] = masks[:, 0]
        # For "separate": ch1-7 stay zero; mask goes through self.mask_encoder

        bottleneck, skips = self._encode(x)

        if self.mask_injection == "separate":
            mask_feat = self.mask_encoder(masks)
            if self.mask_fusion == "additive":
                bottleneck = bottleneck + mask_feat
            else:
                bottleneck = self.fusion_proj(
                    torch.cat([bottleneck, mask_feat], dim=1)
                )

        return skips + [bottleneck]
