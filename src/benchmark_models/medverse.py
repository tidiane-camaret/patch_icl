"""
Adapter for Medverse (https://github.com/jiesihu/Medverse).

Medverse is a native 3D in-context segmentation model that takes K reference
(image, mask) pairs and predicts a continuous map for the target volume.

Input convention (same axis order as this project: D, H, W):
  target_in   : [B, 1, D, H, W]
  context_in  : [B, K, 1, D, H, W]
  context_out : [B, K, 1, D, H, W]  ← add channel dim to our [B, K, D, H, W]

Normalization: Medverse expects per-volume min-max [0, 1].  Our dataloader
returns z-scored values, so we apply model.normalize_3d_volume() on images
before inference.  Mask inputs ({0, 1}) are unaffected.
"""

import math
import sys
import torch
import torch.utils.checkpoint as _cp

from src.benchmark_models.base import InContextModel

MEDVERSE_REPO = "/nfs/norasys/notebooks/camaret/repos/Medverse"
MEDVERSE_CKPT = "/nfs/norasys/notebooks/camaret/repos/Medverse/Medverse.ckpt"


def _checkpoint_forward(module):
    """Monkeypatch a module's forward to be gradient-checkpointed (activations dropped,
    recomputed in backward). Non-reentrant so it supports kwargs / None args / tuple
    returns and works when only parameters (not inputs) require grad. Patched in place —
    no wrapper submodule — so parameter names (hence saved checkpoints) are unchanged.
    A no-op passthrough when grad is disabled (eval), so predict() is unaffected."""
    if getattr(module, "_ckpt_wrapped", False):
        return
    orig_forward = module.forward

    def forward(*args, **kwargs):
        if not torch.is_grad_enabled():
            return orig_forward(*args, **kwargs)
        return _cp.checkpoint(orig_forward, *args, use_reentrant=False, **kwargs)

    module.forward = forward
    module._ckpt_wrapped = True


class MedverseModel(InContextModel):
    """Wraps the Medverse LightningModel for in-context 3D segmentation."""

    def __init__(
        self,
        ckpt_path: str = MEDVERSE_CKPT,
        device: torch.device = None,
        forward_l_arg: int = 1,
        sw_roi_size: tuple = (128, 128, 128),
        random_init: bool = False,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.forward_l_arg = forward_l_arg
        self.sw_roi_size = sw_roi_size

        if MEDVERSE_REPO not in sys.path:
            sys.path.insert(0, MEDVERSE_REPO)

        from medverse.lightning_model import LightningModel  # noqa: PLC0415
        if random_init:
            # Build the same architecture (from the checkpoint's hparams) but with
            # freshly initialized weights — the pretrained state_dict is NOT loaded.
            ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            self.model = LightningModel(ckpt["hyper_parameters"]).to(self.device).eval()
            print("MedverseModel: RANDOM weight init "
                  "(architecture from hparams; pretrained weights not loaded)", flush=True)
        else:
            self.model = LightningModel.load_from_checkpoint(
                ckpt_path, map_location=self.device
            ).to(self.device).eval()

    @torch.no_grad()
    def predict(self, target_img, context_imgs, context_masks):
        """
        Args:
            target_img    : (B, 1, D, H, W)
            context_imgs  : (B, K, 1, D, H, W)
            context_masks : (B, K, D, H, W) binary int64

        Returns:
            (B, D, H, W) binary int64
        """
        target_img    = target_img.to(self.device)
        context_imgs  = context_imgs.to(self.device)
        context_masks = context_masks.to(self.device).float()

        # Medverse expects per-volume min-max normalized images in [0, 1]
        target_norm = self.model.normalize_3d_volume(target_img)
        # context_imgs is [B, K, 1, D, H, W] — normalize each volume independently
        B, K, C, D, H, W = context_imgs.shape
        ctx_flat = context_imgs.view(B * K, C, D, H, W)
        ctx_flat = self.model.normalize_3d_volume(ctx_flat)
        context_norm = ctx_flat.view(B, K, C, D, H, W)

        # context_out: add channel dim → [B, K, 1, D, H, W]
        context_out = context_masks.unsqueeze(2)  # (B, K, 1, D, H, W)

        # Medverse auto-computes level as ceil(log2(max_axis/128))+1, which yields
        # 0 for inputs smaller than 128³ (loop never runs → returns None). Force
        # level=1 (direct single-scale inference) whenever the volume fits in one ROI.
        D, H, W = target_norm.shape[2:]
        auto_level = max(1, int(math.ceil(math.log2(max(D, H, W) / 128))) + 1)

        pred = self.model.autoregressive_inference(
            target_norm,
            context_norm,
            context_out,
            level=auto_level,
            forward_l_arg=self.forward_l_arg,
            sw_roi_size=self.sw_roi_size,
        )
        # pred: [B, 1, D, H, W] continuous map → threshold at 0.5
        return (pred.squeeze(1) > 0.5).long()

    def train_forward(self, target_img, context_imgs, context_masks, l: int = None):
        """Grad-enabled single-ROI forward for fine-tuning — returns raw logits.

        Mirrors predict()'s preprocessing (per-volume min-max norm, mask channel)
        but keeps gradients and skips the sliding-window autoregressive path: the
        input is assumed to fit one ROI (image_size == sw_roi_size at train time).

        Args:
            target_img    : (B, 1, D, H, W)
            context_imgs  : (B, K, 1, D, H, W)
            context_masks : (B, K, D, H, W)
        Returns:
            (B, 1, D, H, W) raw logits (no sigmoid).
        """
        target_img    = target_img.to(self.device)
        context_imgs  = context_imgs.to(self.device)
        context_masks = context_masks.to(self.device).float()

        target_norm = self.model.normalize_3d_volume(target_img)
        B, K, C, D, H, W = context_imgs.shape
        ctx_flat = self.model.normalize_3d_volume(context_imgs.view(B * K, C, D, H, W))
        context_norm = ctx_flat.view(B, K, C, D, H, W)
        context_out = context_masks.unsqueeze(2)  # (B, K, 1, D, H, W)

        return self.model.forward(
            target_norm, context_in=context_norm, context_out=context_out,
            l=(l if l is not None else self.forward_l_arg),
        )

    def enable_gradient_checkpointing(self) -> int:
        """Gradient-checkpoint every conv block of the three U-Nets (context_unet /
        target_encoder / target_decoder) — the ModuleLists holding the full-res
        activations that dominate training memory. ~-50% activation memory for ~+25%
        step time (exact: verified bit-identical loss + grads within conv nondeterminism).
        Returns the number of blocks wrapped. Call AFTER loading weights (names unchanged,
        but keep the order consistent with compile)."""
        net = self.model.net
        n = 0
        for unet in (net.context_unet, net.target_encoder, net.target_decoder):
            for attr in ("enc_blocks", "dec_blocks", "downsample_blocks", "upsample_blocks"):
                ml = getattr(unet, attr, None)
                if ml is None:
                    continue
                for blk in ml:
                    _checkpoint_forward(blk)
                    n += 1
        return n

    def compile_net(self) -> None:
        """torch.compile the whole Medverse net (fuses norm/elementwise/copy tail). Compiles
        cleanly (no fatal graph break); at B=4 combined with checkpointing it beats the eager
        baseline on both time and memory. Costs a slow first batch. Call AFTER weight load;
        train.py strips the `_orig_mod.` prefix compile adds when saving."""
        self.model.net = torch.compile(self.model.net)

    def load_finetuned(self, state_dict: dict) -> None:
        """Load a fine-tuned LightningModel state_dict (as saved by experiments/3d/train.py)."""
        state = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
        self.model.load_state_dict(state)
