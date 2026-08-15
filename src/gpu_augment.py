"""Batched GPU augmentation for 3D in-context segmentation.

Runs in the training loop after batch.to(device), replacing the per-item CPU
augmentation in totalseg_dataloader_incontext. All ops are device/dtype-agnostic
and run under torch.no_grad(). See docs/superpowers/specs/2026-08-15-*.
"""
import math
from typing import Tuple

import torch
import torch.nn.functional as F

from src.augmentations import _make_affine_theta
from src.totalseg_dataset import CT_NORM_MIN, CT_NORM_MAX

REAL, SYNTH, SELF_CONTEXT = 0, 1, 2


def _stack_task(batch: dict) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
    """batch tensors -> (vols (B*T,1,D,H,W), masks (B*T,D,H,W), B, T). vol b*T+0 = target b."""
    img, ctx = batch["image"], batch["context_in"]          # (B,1,D,H,W),(B,K,1,D,H,W)
    lbl, cout = batch["label"], batch["context_out"]         # (B,D,H,W),(B,K,D,H,W)
    B, K = ctx.shape[0], ctx.shape[1]
    T = K + 1
    D, H, W = img.shape[-3:]
    vols = torch.cat([img.unsqueeze(1), ctx], dim=1).reshape(B * T, 1, D, H, W)
    masks = torch.cat([lbl.unsqueeze(1), cout], dim=1).reshape(B * T, D, H, W).long()
    return vols, masks, B, T


def _unstack_task(vols: torch.Tensor, masks: torch.Tensor, B: int, T: int, batch: dict) -> None:
    D, H, W = vols.shape[-3:]
    v = vols.reshape(B, T, 1, D, H, W)
    m = masks.reshape(B, T, D, H, W)
    batch["image"] = v[:, 0]
    batch["context_in"] = v[:, 1:]
    batch["label"] = m[:, 0]
    batch["context_out"] = m[:, 1:]


def _rand(gen, device, *shape):
    return torch.rand(*shape, generator=gen, device=device)


def _uniform(gen, device, lo, hi):
    return (lo + (hi - lo) * torch.rand((), generator=gen, device=device)).item()


def _geometric(vols, masks, group_size, cfg, gen):
    """Shared (group_size=T) or independent (group_size=1) flip/affine/elastic."""
    N = vols.shape[0]
    device = vols.device
    G = N // group_size                              # number of groups
    D, H, W = vols.shape[-3:]

    # --- flips: one decision per group, per axis ---
    fp = cfg.flip
    for g in range(G):
        sl = slice(g * group_size, (g + 1) * group_size)
        for vol_dim, mask_dim, p in [(2, 1, fp.p_d), (3, 2, fp.p_h), (4, 3, fp.p_w)]:
            if _rand(gen, device, 1).item() < p:
                vols[sl] = vols[sl].flip(vol_dim)
                masks[sl] = masks[sl].flip(mask_dim)

    # --- affine: one theta per group (built with the existing helper) ---
    ac = cfg.affine
    thetas = []
    for g in range(G):
        if _rand(gen, device, 1).item() < ac.p:
            mr = ac.max_angle_deg * math.pi / 180.0
            rx, ry, rz = (_uniform(gen, device, -mr, mr) for _ in range(3))
            scale = _uniform(gen, device, ac.scale_min, ac.scale_max)
            tx, ty, tz = (_uniform(gen, device, -ac.max_translate, ac.max_translate)
                          for _ in range(3))
            thetas.append(_make_affine_theta(rx, ry, rz, scale, tx, ty, tz)[0])
        else:
            thetas.append(torch.eye(3, 4))
    theta = torch.stack(thetas).to(device)                       # (G,3,4)
    theta = theta.repeat_interleave(group_size, dim=0)           # (N,3,4)
    grid = F.affine_grid(theta, vols.shape, align_corners=False)  # (N,D,H,W,3)

    # --- elastic: one coarse displacement field per group, added to grid ---
    ec = getattr(cfg, "elastic", None)
    if ec is not None and ec.p > 0:
        gs = max(int(getattr(ec, "grid_scale", 8)), 2)
        sd, sh, sw = max(D // gs, 2), max(H // gs, 2), max(W // gs, 2)
        for g in range(G):
            if _rand(gen, device, 1).item() < ec.p:
                disp = torch.randn(1, 3, sd, sh, sw, generator=gen, device=device) * ec.alpha
                disp = F.interpolate(disp, size=(D, H, W), mode="trilinear",
                                     align_corners=False).permute(0, 2, 3, 4, 1)  # (1,D,H,W,3)
                sl = slice(g * group_size, (g + 1) * group_size)
                grid[sl] = (grid[sl] + disp).clamp(-1.0, 1.0)

    vols = F.grid_sample(vols, grid, mode="bilinear", padding_mode="border",
                         align_corners=False)
    m = F.grid_sample(masks.unsqueeze(1).float(), grid, mode="nearest",
                      padding_mode="zeros", align_corners=False)
    return vols, m.squeeze(1).long()
