"""
Pure tensor ops for multilevel patch sampling.

Given per-cell values on a flattened grid, select cells nearest to 0.5 (uncertain)
and farthest from 0.5 (certain), and gather features/coords for the selected cells.
"""

import torch


def sample_patch_indices(values: torch.Tensor, n_uncertain: int, n_certain: int) -> torch.Tensor:
    """values: (B, N) in [0,1]. Returns (B, n_uncertain + n_certain) long indices:
    the n_uncertain cells closest to 0.5 followed by the n_certain cells farthest
    from 0.5. Disjoint as long as n_uncertain + n_certain <= N."""
    d = (values - 0.5).abs()                 # (B, N): 0 == on the 0.5 boundary
    order = d.argsort(dim=1)                  # ascending: closest-to-0.5 first
    unc = order[:, :n_uncertain]
    cer = order[:, order.shape[1] - n_certain:]   # farthest from 0.5 (largest d)
    return torch.cat([unc, cer], dim=1)


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
