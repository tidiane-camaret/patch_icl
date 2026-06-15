"""
Shared training utilities for the 2D ImagePFN / PatchSetPFN scripts.

Factored out of pfn_seg.py so pfn_seg.py and experiments/2d/multilevel/train.py
share one copy: Muon optimizer, batched GPU augmentation, LAWA averaging, soft-Dice.
"""

import collections

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Muon optimizer ────────────────────────────────────────────────────────────

def _newtonschulz5_batched(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Batched approximate matrix orthogonalization via Newton-Schulz iteration."""
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16()
    X = X / (X.norm(dim=(1, 2), keepdim=True) + eps)
    if X.size(1) > X.size(2):
        X = X.transpose(1, 2)
    for _ in range(steps):
        A = X @ X.transpose(1, 2)
        B = A @ X
        X = a * X + b * B + c * A @ B
    if G.size(1) > G.size(2):
        X = X.transpose(1, 2)
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    """Muon optimizer for hidden-layer 2D weight matrices (Newton-Schulz orthogonalized grads)."""
    def __init__(self, params, lr: float = 3e-4, momentum: float = 0.95,
                 weight_decay: float = 0.0, steps: int = 5):
        super().__init__(params, dict(lr=lr, momentum=momentum,
                                      weight_decay=weight_decay, steps=steps))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr, mu, wd, ns = group['lr'], group['momentum'], group['weight_decay'], group['steps']
            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if 'buf' not in state:
                    state['buf'] = torch.zeros_like(g)
                buf = state['buf']
                buf.mul_(mu).add_(g)
                g = g.add(buf, alpha=mu)  # Nesterov
                if g.ndim == 2 and g.size(0) == 3 * g.size(1):
                    g_batch = g.view(3, g.size(1), g.size(1))
                    g_orth  = _newtonschulz5_batched(g_batch, steps=ns).view_as(g)
                    scale   = g.size(1) ** 0.5
                else:
                    g_orth = _newtonschulz5_batched(g.unsqueeze(0), steps=ns).squeeze(0)
                    scale  = max(g.size(0), g.size(1)) ** 0.5
                p.data.add_(g_orth, alpha=-lr * scale)
                if wd > 0:
                    p.data.mul_(1 - lr * wd)


# ── Augmentation ─────────────────────────────────────────────────────────────

def augment(images: torch.Tensor, masks: torch.Tensor, K: int, cfg):
    """Batched GPU augmentation. Geometric on context pairs; intensity on all images.

    images, masks: (B, T, 1, H, W) float32 on device; query is at index K. cfg = cfg.aug.
    """
    B, T, _, H, W = images.shape
    dev = images.device
    BK  = B * K

    c_imgs = images[:, :K].reshape(BK, 1, H, W)
    c_msks = masks[:, :K].reshape(BK, 1, H, W)

    g = cfg.geometric
    if g.hflip_p > 0:
        m = torch.rand(BK, 1, 1, 1, device=dev) < g.hflip_p
        c_imgs = torch.where(m, c_imgs.flip(-1), c_imgs)
        c_msks = torch.where(m, c_msks.flip(-1), c_msks)
    if g.vflip_p > 0:
        m = torch.rand(BK, 1, 1, 1, device=dev) < g.vflip_p
        c_imgs = torch.where(m, c_imgs.flip(-2), c_imgs)
        c_msks = torch.where(m, c_msks.flip(-2), c_msks)
    if g.rotate.p > 0:
        active = torch.rand(BK, device=dev) < g.rotate.p
        angles = (torch.rand(BK, device=dev) * 2 - 1) * g.rotate.max_angle_deg * active.float()
        rad    = torch.deg2rad(angles)
        cos_t, sin_t = torch.cos(rad), torch.sin(rad)
        z     = torch.zeros_like(cos_t)
        theta = torch.stack([cos_t, -sin_t, z, sin_t, cos_t, z], dim=1).reshape(BK, 2, 3)
        grid  = F.affine_grid(theta, (BK, 1, H, W), align_corners=False)
        c_imgs = F.grid_sample(c_imgs, grid, mode="bilinear", align_corners=False, padding_mode="zeros")
        c_msks = F.grid_sample(c_msks, grid, mode="nearest",  align_corners=False, padding_mode="zeros")

    images = torch.cat([c_imgs.reshape(B, K, 1, H, W), images[:, K:]], dim=1)
    masks  = torch.cat([c_msks.reshape(B, K, 1, H, W), masks[:, K:]],  dim=1)

    BT   = B * T
    imgs = images.reshape(BT, 1, H, W)
    ic   = cfg.intensity
    if ic.brightness.p > 0:
        m = torch.rand(BT, 1, 1, 1, device=dev) < ic.brightness.p
        d = (torch.rand(BT, 1, 1, 1, device=dev) * 2 - 1) * ic.brightness.max_delta
        imgs = torch.where(m, (imgs + d).clamp(0, 1), imgs)
    if ic.contrast.p > 0:
        lo, hi = ic.contrast.range
        m  = torch.rand(BT, 1, 1, 1, device=dev) < ic.contrast.p
        s  = torch.rand(BT, 1, 1, 1, device=dev) * (hi - lo) + lo
        mu = imgs.mean(dim=(-2, -1), keepdim=True)
        imgs = torch.where(m, ((imgs - mu) * s + mu).clamp(0, 1), imgs)
    if ic.gamma.p > 0:
        lo, hi = ic.gamma.range
        m = torch.rand(BT, 1, 1, 1, device=dev) < ic.gamma.p
        gm = torch.rand(BT, 1, 1, 1, device=dev) * (hi - lo) + lo
        imgs = torch.where(m, imgs.clamp(1e-6).pow(gm), imgs)
    if ic.noise.p > 0:
        m = torch.rand(BT, 1, 1, 1, device=dev) < ic.noise.p
        n = torch.randn_like(imgs) * ic.noise.std
        imgs = torch.where(m, (imgs + n).clamp(0, 1), imgs)

    return imgs.reshape(B, T, 1, H, W), masks


# ── LAWA ─────────────────────────────────────────────────────────────────────

def lawa_average(queue: collections.deque, model: nn.Module, device: torch.device):
    """Average checkpoint queue into model weights; return original state for restore."""
    if len(queue) <= 1:
        return None
    avg = {k: sum(s[k].float() for s in queue) / len(queue) for k in queue[0]}
    avg = {k: v.to(dtype=queue[0][k].dtype, device=device) for k, v in avg.items()}
    saved = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(avg)
    return saved


# ── Loss ──────────────────────────────────────────────────────────────────────

def soft_dice_loss(p: torch.Tensor, t: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    """Per-sample soft Dice loss between probability map p and soft target t (both flattened per row)."""
    p = p.flatten(1).float()
    t = t.flatten(1).float()
    num = 2 * (p * t).sum(1) + eps
    den = p.sum(1) + t.sum(1) + eps
    return (1 - num / den).mean()
