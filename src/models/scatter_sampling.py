"""Unconstrained scatter patch sampling for PatchSetCNN refinement.

Productionized copy of experiments/2d/multilevel/sampling.py (the capped variant from
plot_sampling.py). Selects a budget of individual grid cells via three priority tiers —
boundary core, a fixed foreground-core quota, and a blurred-proximity neighbor fill — then
gathers features/coords for them and scatters refined predictions back. Pure tensor ops.
"""

import numpy as np
import torch
import torch.nn.functional as F


def gaussian_blur(x_flat: torch.Tensor, grid_res: int, sigma: float) -> torch.Tensor:
    """(B, N) -> (B, N) separable Gaussian blur on the grid_res x grid_res grid."""
    B, N = x_flat.shape
    x = x_flat.reshape(B, 1, grid_res, grid_res)
    k = int(2 * np.ceil(2 * sigma) + 1)
    coords = torch.arange(k, dtype=torch.float32, device=x.device) - (k - 1) / 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = (g / g.sum()).to(x.dtype)
    pad = k // 2
    x = F.conv2d(F.pad(x, (pad, pad, 0, 0), mode="reflect"), g.view(1, 1, 1, k))
    x = F.conv2d(F.pad(x, (0, 0, pad, pad), mode="reflect"), g.view(1, 1, k, 1))
    return x.reshape(B, N)


def sample_patches(values: torch.Tensor, n_total: int, tau: float, blur_sigma: float,
                   floor: float, grid_res: int, temperature: float = 1.0,
                   stochastic: bool = True, n_fg_core: int = 0, boundary_tier: bool = True,
                   n_boundary_core: int = 0):
    """values: (B, N) in [0,1]. Returns (idx, is_core, is_fg_core), each (B, n_total).

    Three priority tiers combined into one score + a single top-k:
      1. boundary core: |value-0.5| < tau (ranked by closeness to 0.5); optionally capped to
         the n_boundary_core cells closest to 0.5. Disabled when boundary_tier=False.
      2. fg core: a fixed n_fg_core quota of value>=0.5 cells chosen uniformly at random.
    The remaining budget is a blurred proximity field over (core u fg_core) + uniform floor +
    Gumbel-top-k neighbor fill.
    """
    assert n_total <= values.shape[1], \
        f"n_total={n_total} exceeds grid cells {values.shape[1]} (grid_res={grid_res})"
    d = (values - 0.5).abs()
    core_b = (d < tau) if boundary_tier else torch.zeros_like(values, dtype=torch.bool)
    if boundary_tier and n_boundary_core > 0:
        masked_d = torch.where(core_b, d, torch.full_like(d, 2.0))          # non-core -> large
        keep = masked_d.topk(min(n_boundary_core, d.shape[1]), dim=1, largest=False).indices
        core_b = torch.zeros_like(core_b).scatter_(1, keep, True) & core_b   # real core only

    fg_core = torch.zeros_like(core_b)
    if n_fg_core > 0:
        fg_pool = (values >= 0.5) & ~core_b
        key = torch.where(fg_pool, torch.rand_like(values), values.new_full((), -1.0))
        take = key.topk(n_fg_core, dim=1).indices
        fg_core = torch.zeros_like(core_b).scatter_(1, take, True) & fg_pool  # guard: <n_fg_core fg

    g = gaussian_blur((core_b | fg_core).float(), grid_res, blur_sigma)
    w = g + floor
    if stochastic:
        u = torch.rand_like(w).clamp(1e-6, 1 - 1e-6)
        gumbel = -torch.log(-torch.log(u))
        neigh_score = (w + 1e-12).log() + temperature * gumbel
    else:
        neigh_score = (w + 1e-12).log()

    BIG_B, BIG_F = 2e4, 1e4                                                   # boundary > fg > neighbor
    score = torch.where(core_b, BIG_B - d, torch.where(fg_core, BIG_F, neigh_score))
    idx = score.topk(n_total, dim=1).indices
    is_fg_core = fg_core.gather(1, idx)
    is_core = (core_b | fg_core).gather(1, idx)
    return idx, is_core, is_fg_core


def idx_to_ij(idx: torch.Tensor, grid_res: int) -> torch.Tensor:
    """Flat cell index (B, M) -> (B, M, 2) row/col on a grid_res x grid_res grid (row-major)."""
    return torch.stack([torch.div(idx, grid_res, rounding_mode="floor"),
                        idx % grid_res], dim=-1)


def gather_grid(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Gather along the cell axis. x: (B, N, C) -> (B, M, C), or x: (B, N) -> (B, M)."""
    if x.dim() == 3:
        C = x.shape[-1]
        return torch.gather(x, 1, idx.unsqueeze(-1).expand(-1, -1, C))
    return torch.gather(x, 1, idx)


def composite_predictions(coarse_flat: torch.Tensor, idx: torch.Tensor,
                          vals: torch.Tensor) -> torch.Tensor:
    """(B,N) dense map + (B,M) indices + (B,M) values -> (B,N) NEW tensor with vals scattered in."""
    refined = coarse_flat.clone()
    refined.scatter_(1, idx, vals)
    return refined
