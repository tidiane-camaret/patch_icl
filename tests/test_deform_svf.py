"""Diffeomorphic SVF deform op: no folding (positive Jacobian) + determinism.

The payoff vs legacy `elastic`: scaling-and-squaring integration of a smooth
velocity field is guaranteed invertible, so the deformation never folds and
masks stay valid. Legacy elastic (random field added straight to the grid) folds
at comparable magnitude.
"""
import sys; sys.path.insert(0, ".")
import math
import torch
import torch.nn.functional as F
from types import SimpleNamespace

from src.augmentations import _svf_displacement, apply_task_aug
from src.gpu_augment import _geometric


def _jac_det(grid):
    """Jacobian determinant of a sampling grid (1,D,H,W,3), grid comps order (W,H,D).
    Reorder comps to (D,H,W) so the identity grid gives det=+1. det>0 ⇒ no folding."""
    f = grid[0][..., [2, 1, 0]]                  # (D,H,W,3) comps in (D,H,W) order
    d0 = torch.gradient(f, dim=0)[0]
    d1 = torch.gradient(f, dim=1)[0]
    d2 = torch.gradient(f, dim=2)[0]
    J = torch.stack([d0, d1, d2], dim=-2)        # J[...,axis,comp]
    return torch.linalg.det(J)                   # (D,H,W)


def _base_grid(D, H, W):
    return F.affine_grid(torch.eye(3, 4).unsqueeze(0), (1, 1, D, H, W), align_corners=False)


def test_svf_is_diffeomorphic():
    """Deform grid has strictly positive Jacobian everywhere, across seeds/magnitudes."""
    D = H = W = 16
    base = _base_grid(D, H, W)
    for seed in range(10):
        for max_disp in (0.1, 0.2, 0.3):
            g = torch.Generator().manual_seed(seed)
            phi = _svf_displacement((D, H, W), control_points=4, max_disp=max_disp,
                                    num_steps=6, generator=g)
            # interior only: torch.gradient uses one-sided diffs at the boundary,
            # which can spuriously show ~0 det at edge voxels (not a real fold).
            det = _jac_det(base + phi)[1:-1, 1:-1, 1:-1]
            assert det.min() > 0, f"folding at seed={seed} max_disp={max_disp}: min det={det.min():.4f}"


def test_legacy_elastic_can_fold():
    """Contrast: legacy elastic (unintegrated field) folds at comparable magnitude."""
    D = H = W = 16
    base = _base_grid(D, H, W)
    folded = 0
    for seed in range(10):
        g = torch.Generator().manual_seed(seed)
        disp = F.interpolate(torch.randn(1, 3, 4, 4, 4, generator=g) * 0.35,
                             size=(D, H, W), mode="trilinear", align_corners=False)
        disp = disp.permute(0, 2, 3, 4, 1)
        det = _jac_det((base + disp).clamp(-1, 1))
        folded += int(det.min() <= 0)
    assert folded > 0, "expected legacy elastic to fold at least once at this magnitude"


def test_svf_deterministic():
    """Same seed → identical field (CPU/GPU paths call this same helper)."""
    a = _svf_displacement((12, 12, 12), 4, 0.2, 6, generator=torch.Generator().manual_seed(3))
    b = _svf_displacement((12, 12, 12), 4, 0.2, 6, generator=torch.Generator().manual_seed(3))
    assert torch.equal(a, b)


def test_deform_shared_across_task_and_valid_mask():
    """apply_task_aug shares one field across all volumes; mask stays binary."""
    N, D = 3, 16
    torch.manual_seed(0)
    img = torch.randn(N, 1, D, D, D)
    # a solid ball mask, identical in every volume
    zz, yy, xx = torch.meshgrid(*[torch.arange(D)] * 3, indexing="ij")
    ball = (((zz - 8.) ** 2 + (yy - 8.) ** 2 + (xx - 8.) ** 2) < 25).long()
    msk = ball.unsqueeze(0).repeat(N, 1, 1, 1).clone()
    cfg = SimpleNamespace(
        flip=SimpleNamespace(p_d=0., p_h=0., p_w=0.),
        affine=SimpleNamespace(p=0., max_angle_deg=0., scale_min=1., scale_max=1., max_translate=0.),
        elastic=SimpleNamespace(p=0., alpha=0.1, grid_scale=4),
        deform=SimpleNamespace(p=1.0, control_points=4, max_disp=0.2, num_steps=6),
    )
    import random; random.seed(0)
    out_img, out_msk = apply_task_aug(img.clone(), msk.clone(), cfg)
    assert set(out_msk.unique().tolist()) <= {0, 1}          # still binary, no tearing
    # shared field ⇒ identical warp on identical inputs
    assert torch.equal(out_msk[0], out_msk[1]) and torch.equal(out_msk[1], out_msk[2])
    # ball roughly preserved in size (diffeomorphic, small warp)
    assert 0.6 < out_msk[0].sum().item() / ball.sum().item() < 1.6


def test_gpu_geometric_deform_branch_runs():
    """_geometric deform branch executes on CPU and stays diffeomorphic-valid (binary mask)."""
    N = 2
    vols = torch.randn(N, 1, 12, 12, 12)
    masks = torch.randint(0, 2, (N, 12, 12, 12))
    cfg = SimpleNamespace(
        flip=SimpleNamespace(p_d=0., p_h=0., p_w=0.),
        affine=SimpleNamespace(p=0., max_angle_deg=0., scale_min=1., scale_max=1., max_translate=0.),
        elastic=SimpleNamespace(p=0., alpha=0.1, grid_scale=4),
        deform=SimpleNamespace(p=1.0, control_points=4, max_disp=0.2, num_steps=6),
    )
    gen = torch.Generator().manual_seed(0)
    v, m = _geometric(vols, masks, group_size=N, cfg=cfg, gen=gen)
    assert v.shape == vols.shape
    assert set(m.unique().tolist()) <= {0, 1}
