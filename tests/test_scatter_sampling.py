import sys; sys.path.insert(0, ".")
import torch
from src.models.scatter_sampling import (
    sample_patches, idx_to_ij, gather_grid, composite_predictions)


def test_sample_patches_shapes_and_budget():
    torch.manual_seed(0)
    B, R, M = 3, 16, 64
    values = torch.rand(B, R * R)
    idx, is_core, is_fg = sample_patches(values, M, tau=0.30, blur_sigma=1.0,
                                         floor=0.005, grid_res=R, n_fg_core=8)
    assert idx.shape == (B, M) and is_core.shape == (B, M) and is_fg.shape == (B, M)
    assert idx.min() >= 0 and idx.max() < R * R
    # indices unique per row (top-k over distinct cells)
    for b in range(B):
        assert idx[b].unique().numel() == M


def test_sample_patches_deterministic_when_seeded():
    R, M = 16, 32
    values = torch.rand(1, R * R)
    torch.manual_seed(7); a, _, _ = sample_patches(values, M, 0.3, 1.0, 0.005, R)
    torch.manual_seed(7); b, _, _ = sample_patches(values, M, 0.3, 1.0, 0.005, R)
    assert torch.equal(a, b)


def test_boundary_core_cap_limits_core_count():
    R, M = 16, 128
    # a smooth ramp so many cells fall in the tau band
    values = torch.linspace(0, 1, R * R).reshape(1, R * R)
    _, core_uncapped, _ = sample_patches(values, M, 0.45, 1.0, 0.005, R, stochastic=False)
    _, core_capped, _ = sample_patches(values, M, 0.45, 1.0, 0.005, R, stochastic=False,
                                       n_boundary_core=10)
    assert int(core_capped.sum()) <= int(core_uncapped.sum())
    assert int(core_capped.sum()) <= 10


def test_compact_blob_boundary_in_core():
    # a solid square blob → its fractional-boundary cells should land in the core tier
    R, M = 16, 96
    g = torch.zeros(R, R); g[4:12, 4:12] = 1.0
    values = g.reshape(1, R * R)
    idx, is_core, _ = sample_patches(values, M, 0.30, 1.0, 0.005, R, stochastic=False, n_fg_core=16)
    # every selected core cell that is fractional (0<v<1) — here the blob is binary so use fg cells
    sel_fg = gather_grid(values, idx)[is_core]
    assert (sel_fg >= 0.5).float().mean() > 0.5   # core is dominated by foreground


def test_gather_and_composite_roundtrip():
    B, R, M = 2, 8, 10
    coarse = torch.rand(B, R * R)
    idx = torch.stack([torch.randperm(R * R)[:M] for _ in range(B)])
    vals = torch.ones(B, M)
    out = composite_predictions(coarse, idx, vals)
    assert out.shape == coarse.shape
    assert torch.allclose(gather_grid(out, idx), vals)          # scattered cells overwritten
    assert not out.data_ptr() == coarse.data_ptr()              # new tensor


def test_idx_to_ij():
    idx = torch.tensor([[0, 1, 8, 9]])
    ij = idx_to_ij(idx, 8)
    assert torch.equal(ij, torch.tensor([[[0, 0], [0, 1], [1, 0], [1, 1]]]))
