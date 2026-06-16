"""
Pure tensor ops for multilevel patch sampling.

Given per-cell values on a flattened grid, select a budget of cells via three
priority tiers — threshold boundary core, a fixed random-foreground-core quota,
and a blurred-proximity neighbor fill — then gather features/coords for them.
See docs/superpowers/specs/2026-06-16-multilevel-patch-sampling-design.md.
"""

import numpy as np
import torch
import torch.nn.functional as F


def gaussian_blur(x_flat: torch.Tensor, grid_res: int, sigma: float) -> torch.Tensor:
    """(B, N) → (B, N) separable Gaussian blur on the grid_res×grid_res grid."""
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


def sample_patches(values: torch.Tensor, n_total: int, tau: float, n_fg_core: int,
                   blur_sigma: float, floor: float, grid_res: int,
                   temperature: float = 1.0, stochastic: bool = True):
    """values: (B, N) in [0,1]. Returns (idx, is_core, is_fg_core), each (B, n_total).

    A single combined score + one top-k handles the variable per-row core count
    without ragged tensors. Three priority tiers (all above the neighbor tier):
      1. boundary core : cells with |value-0.5| < tau (ranked by closeness to 0.5)
      2. fg core       : a fixed quota of n_fg_core foreground cells (value>=0.5),
                         chosen uniformly at random, to cover object interiors the
                         boundary misses — independent of the neighbor field.
    The remainder is filled by cells sampled near the boundary core via a blurred
    proximity field + Gumbel-top-k (uniform `floor` keeps far cells in play).

    is_core    = boundary core ∪ fg core; is_fg_core = fg core only.
    The boundary core (is_core & ~is_fg_core) is the genuinely-uncertain set.
    """
    d = (values - 0.5).abs()
    core_b = d < tau                                   # boundary core (variable count)

    # ── Tier 2: forced foreground quota (random fg, excluding boundary core) ──
    fg_core = torch.zeros_like(core_b)
    if n_fg_core > 0:
        fg_pool = (values >= 0.5) & ~core_b
        key = torch.where(fg_pool, torch.rand_like(values), values.new_full((), -1.0))
        take = key.topk(n_fg_core, dim=1).indices
        fg_core = torch.zeros_like(core_b).scatter_(1, take, True) & fg_pool  # guard: <n_fg_core fg

    # ── Neighbor proximity field (around the boundary core only) ──
    g = gaussian_blur(core_b.float(), grid_res, blur_sigma)
    w = g + floor
    if stochastic:
        u = torch.rand_like(w).clamp(1e-6, 1 - 1e-6)
        gumbel = -torch.log(-torch.log(u))
        neigh_score = (w + 1e-12).log() + temperature * gumbel
    else:
        neigh_score = (w + 1e-12).log()

    BIG_B, BIG_F = 2e4, 1e4                             # boundary > fg core > neighbors
    score = torch.where(core_b, BIG_B - d,
            torch.where(fg_core, BIG_F, neigh_score))
    idx = score.topk(n_total, dim=1).indices
    is_fg_core = fg_core.gather(1, idx)
    is_core    = (core_b | fg_core).gather(1, idx)
    return idx, is_core, is_fg_core


def idx_to_ij(idx: torch.Tensor, grid_res: int) -> torch.Tensor:
    """Flat cell index (B, M) → (B, M, 2) row/col coords on a grid_res×grid_res grid (row-major)."""
    return torch.stack([torch.div(idx, grid_res, rounding_mode="floor"),
                        idx % grid_res], dim=-1)


def gather_grid(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Gather along the cell axis. x: (B, N, C) → (B, M, C), or x: (B, N) → (B, M)."""
    if x.dim() == 3:
        C = x.shape[-1]
        return torch.gather(x, 1, idx.unsqueeze(-1).expand(-1, -1, C))
    return torch.gather(x, 1, idx)
