"""VoComni SwinUNETR encoder for patch_icl.

Wraps the MONAI SwinUNETR backbone (VoCo/VoComni pretrained weights) as a
patch_icl encoder.  Only the swinViT is used; the decoder head is discarded.

Architecture summary (feature_size=48, Base)
--------------------------------------------
    feature_size : 48 / 96 / 192  (Base / Large / Huge)
    patch_size   : 2 × 2 × 2
    stages       : 4 Swin Transformer stages with PatchMerge
    total_stride : 32× (5 output levels at strides 2 4 8 16 32)
    use_v2       : True  (matches VoCo/VoComni training)

Outputs (feature_size=48)
-------------------------
    level 0 : (B,  48, D/ 2, H/ 2, W/ 2)
    level 1 : (B,  96, D/ 4, H/ 4, W/ 4)
    level 2 : (B, 192, D/ 8, H/ 8, W/ 8)
    level 3 : (B, 384, D/16, H/16, W/16)
    level 4 : (B, 768, D/32, H/32, W/32)   ← bottleneck

Interface (same as ThreeDINOEncoder / NNInteractiveEncoder)
-----------------------------------------------------------
    skip_channels : [fs, 2*fs, 4*fs, 8*fs]
    bot_features  : 16 * fs
    total_stride  : 32
    forward(imgs, masks) -> list[torch.Tensor]  (masks are ignored)

Pretrained checkpoints
----------------------
    VoComni_B.pt  (feature_size=48,  72M params)
    VoComni_L.pt  (feature_size=96, 290M params)
    VoComni_H.pt  (feature_size=192, 1.2B params)
    Download from https://huggingface.co/Luffy503/VoCo

Input normalisation
-------------------
VoCo/VoComni were trained with MONAI ScaleIntensityRanged(-175, 250, 0.0, 1.0).
forward() undoes the totalseg nnUNet z-score (mean=-167.3, std=505.8) then applies
that fixed HU window:
    HU  = z_score * 505.8 - 167.3
    out = (clip(HU, -175, 250) + 175) / 425   →  [0, 1]
Applied in float32 before autocast; the transformer forward runs in fp16.

torch.compile
-------------
    compile_model=True wraps swinViT with torch.compile (inductor backend).
    Requires a compatible triton version; if compilation fails at warmup,
    fall back to compile_model=False.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Thin wrapper so torch.compile covers the full swinViT forward
# ---------------------------------------------------------------------------

class _SwinViTWrapper(nn.Module):
    def __init__(self, swin: nn.Module) -> None:
        super().__init__()
        self._swin = swin

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return tuple(self._swin.swinViT(x, normalize=True))


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def _load_swin_checkpoint(model: nn.Module, ckpt: dict) -> nn.Module:
    """Load a VoCo/VoComni checkpoint into SwinUNETR, handling common key prefixes."""
    if "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    elif "network_weights" in ckpt:
        state_dict = ckpt["network_weights"]
    elif "net" in ckpt:
        state_dict = ckpt["net"]
    elif "student" in ckpt:
        state_dict = ckpt["student"]
    else:
        state_dict = ckpt

    keys = list(state_dict.keys())
    if keys and keys[0].startswith("module."):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    if keys and keys[0].startswith("backbone."):
        state_dict = {k.replace("backbone.", "", 1): v for k, v in state_dict.items()}
    if keys and "swin_vit" in keys[0]:
        state_dict = {k.replace("swin_vit", "swinViT"): v for k, v in state_dict.items()}

    current = model.state_dict()
    merged = {
        k: state_dict[k]
        if k in state_dict and state_dict[k].size() == current[k].size()
        else current[k]
        for k in current
    }
    model.load_state_dict(merged, strict=True)
    return model


# ---------------------------------------------------------------------------
# Public encoder
# ---------------------------------------------------------------------------

class VoComniEncoder(nn.Module):
    """SwinUNETR encoder with optional VoCo/VoComni pretrained weights.

    Args
    ----
        ckpt_path       : path to VoCo/VoComni .pt checkpoint, or None for
                          random weights (useful for ablation / interface test).
        feature_size    : base embedding dim — 48 (Base), 96 (Large), 192 (Huge).
        freeze_encoder  : freeze all weights after loading (default True).
        compile_model   : wrap with torch.compile for kernel-fusion speedup.
                          Requires a compatible triton installation; default False.
    """

    def __init__(
        self,
        ckpt_path:      str | Path | None = None,
        feature_size:   int  = 48,
        freeze_encoder: bool = True,
        compile_model:  bool = True,
    ):
        super().__init__()
        from monai.networks.nets import SwinUNETR

        swin = SwinUNETR(
            in_channels=1,
            out_channels=2,      # decoder head is never called; 2 is the minimum
            feature_size=feature_size,
            use_v2=True,
        )

        if ckpt_path is not None:
            ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
            swin = _load_swin_checkpoint(swin, ckpt)
            print(f"[VoComni] SwinUNETR (fs={feature_size}) loaded from {Path(ckpt_path).name}")
        else:
            print(f"[VoComni] SwinUNETR (fs={feature_size}) — no checkpoint, random weights")

        swin.eval()
        if freeze_encoder:
            for p in swin.parameters():
                p.requires_grad_(False)

        wrapper = _SwinViTWrapper(swin)
        self._extractor = torch.compile(wrapper) if compile_model else wrapper

        fs = feature_size
        self.skip_channels = [fs, fs * 2, fs * 4, fs * 8]
        self.bot_features  = fs * 16
        self.total_stride  = 32

    # ------------------------------------------------------------------

    @staticmethod
    def _normalise(imgs: torch.Tensor) -> torch.Tensor:
        """Undo nnUNet z-score then apply VoCo ScaleIntensityRanged(-175, 250, 0, 1).

        totalseg z-score: z = (HU + 167.3) / 505.8  →  HU = z * 505.8 - 167.3
        VoCo training:    clip(HU, -175, 250) → (HU + 175) / 425  →  [0, 1]
        """
        hu = imgs.float() * 505.8 - 167.3
        return ((hu.clamp(-175.0, 250.0) + 175.0) / 425.0)

    # ------------------------------------------------------------------

    def forward(
        self, imgs: torch.Tensor, masks: torch.Tensor | None = None
    ) -> list[torch.Tensor]:
        """
        Args
        ----
            imgs  : (B, 1, D, H, W) — CT image; expects totalseg nnUNet z-score (~[-1.7, +3.5]); re-normed to [0,1] internally
            masks : ignored

        Returns
        -------
            list of 5 tensors ordered shallow → deep:
              [  (B, fs,   D/2,  H/2,  W/2),
                 (B, 2fs,  D/4,  H/4,  W/4),
                 (B, 4fs,  D/8,  H/8,  W/8),
                 (B, 8fs,  D/16, H/16, W/16),
                 (B, 16fs, D/32, H/32, W/32)  ]
        """
        x = self._normalise(imgs)      # float32, outside autocast
        return list(self._extractor(x))
