"""
Patch-level binary segmentation losses.

All functions expect:
  pred : (B, N) float32  — sigmoid probabilities in [0, 1]
  gt   : (B, N) float32  — soft labels (avg-pooled binary mask), in [0, 1]

Available loss names (for train.loss config key):
  bce          — binary cross-entropy only (default, original)
  dice_bce     — soft Dice + BCE; recommended for class-imbalanced organs
  dice_focal   — soft Dice + focal; focal alpha=0.5 (unlike RetinaNet's 0.25,
                 which over-penalises background when class balance is already
                 addressed by class-balanced sampling + synth augmentation)
"""

import torch
import torch.nn.functional as F
from torch import Tensor


def soft_dice_loss(pred: Tensor, gt: Tensor, smooth: float = 1e-6) -> Tensor:
    """Soft Dice averaged over batch items."""
    num = 2 * (pred * gt).sum(dim=1)
    den = pred.sum(dim=1) + gt.sum(dim=1) + smooth
    return (1 - num / den).mean()


def focal_loss(
    pred: Tensor,
    gt: Tensor,
    alpha: float = 0.5,
    gamma: float = 2.0,
) -> Tensor:
    """Focal loss for probability outputs (not logits).

    alpha=0.5 treats fore/background symmetrically; tune toward 0.25 if false
    negatives dominate, toward 0.75 if false positives dominate.
    """
    bce   = F.binary_cross_entropy(pred, gt, reduction="none")
    p_t   = pred * gt + (1 - pred) * (1 - gt)
    alpha_t = alpha * gt + (1 - alpha) * (1 - gt)
    return (alpha_t * (1 - p_t) ** gamma * bce).mean()


def dice_bce_loss(pred: Tensor, gt: Tensor, smooth: float = 1e-6) -> Tensor:
    return F.binary_cross_entropy(pred, gt) + soft_dice_loss(pred, gt, smooth)


def dice_focal_loss(
    pred: Tensor,
    gt: Tensor,
    alpha: float = 0.5,
    gamma: float = 2.0,
    smooth: float = 1e-6,
) -> Tensor:
    return focal_loss(pred, gt, alpha, gamma) + soft_dice_loss(pred, gt, smooth)


_LOSSES = {
    "bce":        F.binary_cross_entropy,
    "dice_bce":   dice_bce_loss,
    "dice_focal": dice_focal_loss,
}


def get_loss_fn(name: str):
    """Return a loss function by name.  Callable signature: (pred, gt) -> scalar."""
    if name not in _LOSSES:
        raise ValueError(f"Unknown loss '{name}'. Choose from: {list(_LOSSES)}")
    return _LOSSES[name]
