"""Grid-resolution monitoring metrics for the low-res PatchSet3D (dice_ds / dice_ds_soft
/ cossim). Flat-tensor reductions ported from experiments/2d/train.py — dimension-agnostic
(flatten(1)); shared by train.py's train loop and evaluate.py's val step."""

import torch
import torch.nn.functional as F


def target_like(lbl: torch.Tensor, logit: torch.Tensor) -> torch.Tensor:
    """Pool GT (B,1,D,H,W) DOWN to the logit's spatial size (soft occupancy target)."""
    return F.adaptive_avg_pool3d(lbl, logit.shape[-3:])


def soft_sum(prob, target, eps: float = 1e-6):
    """Threshold-free Dice SUM + valid-row count."""
    p = prob.detach().flatten(1).float(); g = target.detach().flatten(1).float()
    den = p.sum(1) + g.sum(1); ok = den > eps
    s = torch.where(ok, 2 * (p * g).sum(1) / den.clamp_min(eps), torch.zeros_like(den))
    return s.sum(), ok.sum()


def hard_sum(prob, gt, eps: float = 1e-6):
    """Hard Dice SUM + valid-row count: pred>=0.5 vs gt>=0.5 (>=0.5 not >0 so a soft
    partial-volume target counts a boundary cell as foreground only past the half mark)."""
    p = (prob.detach().flatten(1).float() >= 0.5).float()
    g = (gt.detach().flatten(1).float() >= 0.5).float()
    den = p.sum(1) + g.sum(1); ok = den > eps
    h = torch.where(ok, 2 * (p * g).sum(1) / den.clamp_min(eps), torch.zeros_like(den))
    return h.sum(), ok.sum()


def cos_sum(prob, target, eps: float = 1e-6):
    """Scale-invariant cosine similarity SUM + valid-row count."""
    p = prob.detach().flatten(1).float(); g = target.detach().flatten(1).float()
    den = p.norm(dim=1) * g.norm(dim=1); ok = den > eps
    c = torch.where(ok, (p * g).sum(1) / den.clamp_min(eps), torch.zeros_like(den))
    return c.sum(), ok.sum()
