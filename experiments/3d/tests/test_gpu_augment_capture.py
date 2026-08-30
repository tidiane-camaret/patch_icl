"""Tasks 2-3: GpuAugmentor grid/flip capture + injected-generator replay."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import torch
from omegaconf import OmegaConf

from src.gpu_augment import _geometric, GpuAugmentor, GeoState


def _task_cfg(flip_p=0.5, affine_p=1.0, deform_p=0.0, elastic_p=0.0):
    return OmegaConf.create({
        "flip": {"p_d": flip_p, "p_h": flip_p, "p_w": flip_p},
        "affine": {"p": affine_p, "max_angle_deg": 20.0, "scale_min": 0.9,
                   "scale_max": 1.1, "max_translate": 0.1},
        "elastic": {"p": elastic_p, "alpha": 0.1, "grid_scale": 8},
        "deform": {"p": deform_p, "control_points": 4, "max_disp": 0.1, "num_steps": 4},
        "mask_interp": "bilinear",
    })


def test_geometric_capture_shapes():
    N, T = 4, 8
    vols = torch.randn(N, 1, T, T, T)
    masks = torch.zeros(N, T, T, T)
    g = torch.Generator().manual_seed(0)
    out = _geometric(vols, masks, group_size=N, cfg=_task_cfg(deform_p=1.0), gen=g, capture=True)
    assert len(out) == 4
    _v, _m, grid, flips = out
    assert grid.shape == (N, T, T, T, 3) and grid.dtype == torch.float32
    assert flips.shape == (N, 3) and flips.dtype == torch.bool


def test_geometric_no_capture_is_two_tuple():
    N, T = 2, 8
    vols = torch.randn(N, 1, T, T, T)
    masks = torch.zeros(N, T, T, T)
    g = torch.Generator().manual_seed(0)
    out = _geometric(vols, masks, group_size=N, cfg=_task_cfg(), gen=g)
    assert len(out) == 2


def test_geometric_same_seed_same_transform():
    # Same generator seed + same shapes -> identical grid and flips, on different content.
    N, T = 3, 8
    cfg = _task_cfg(deform_p=1.0)
    a = torch.randn(N, 1, T, T, T)
    b = torch.randn(N, 1, T, T, T)
    m = torch.zeros(N, T, T, T)
    g1 = torch.Generator().manual_seed(42)
    g2 = torch.Generator().manual_seed(42)
    _, _, grid_a, flip_a = _geometric(a, m.clone(), N, cfg, g1, capture=True)
    _, _, grid_b, flip_b = _geometric(b, m.clone(), N, cfg, g2, capture=True)
    assert torch.equal(flip_a, flip_b)
    assert torch.allclose(grid_a, grid_b, atol=1e-6)
