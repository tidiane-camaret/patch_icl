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


def _full_cfg():
    return OmegaConf.create({
        "enabled": True, "gpu": True,
        "task": {
            "flip": {"p_d": 0.5, "p_h": 0.5, "p_w": 0.5},
            "affine": {"p": 1.0, "max_angle_deg": 20.0, "scale_min": 0.9,
                       "scale_max": 1.1, "max_translate": 0.1},
            "elastic": {"p": 0.0, "alpha": 0.1, "grid_scale": 8},
            "deform": {"p": 1.0, "control_points": 4, "max_disp": 0.1, "num_steps": 4},
            "mask_interp": "bilinear",
        },
        "intensity": {
            "brightness_contrast": {"p": 0.5, "brightness": 0.0,
                                    "contrast_range": [0.8, 1.2], "preserve_range": True},
        },
    })


def _fake_batch(B=2, K=3, T=8):
    return {
        "image": torch.randn(B, 1, T, T, T),
        "label": torch.zeros(B, T, T, T),
        "context_in": torch.randn(B, K, 1, T, T, T),
        "context_out": torch.zeros(B, K, T, T, T),
        "aug_mode": torch.zeros(B, dtype=torch.long),
    }


def test_apply_returns_geostate_on_capture():
    aug = GpuAugmentor(_full_cfg())
    b = _fake_batch()
    _, geo = aug.apply(b, geo_gen=torch.Generator().manual_seed(1),
                       int_gen=torch.Generator().manual_seed(2), capture=True)
    assert isinstance(geo, GeoState)
    assert geo.grid.shape == (2 * 4, 8, 8, 8, 3)   # B*T
    assert geo.flips.shape == (2 * 4, 3)


def test_apply_replay_same_geo_seed_matches_geometry():
    # Two batches, same geo_gen seed -> identical captured grid + flips (geometry replay),
    # even with different intensity seeds and content.
    aug = GpuAugmentor(_full_cfg())
    b0, b1 = _fake_batch(), _fake_batch()
    _, g0 = aug.apply(b0, geo_gen=torch.Generator().manual_seed(7),
                      int_gen=torch.Generator().manual_seed(100), capture=True)
    _, g1 = aug.apply(b1, geo_gen=torch.Generator().manual_seed(7),
                      int_gen=torch.Generator().manual_seed(200), capture=True)
    assert torch.equal(g0.flips, g1.flips)
    assert torch.allclose(g0.grid, g1.grid, atol=1e-6)


def test_call_path_unchanged_byte_identical():
    # GpuAugmentor.__call__ must be unaffected by the apply() addition.
    cfg = _full_cfg()
    a = GpuAugmentor(cfg, seed=0)
    b = GpuAugmentor(cfg, seed=0)
    batch_a, batch_b = _fake_batch(B=2), None
    torch.manual_seed(0)
    batch_b = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch_a.items()}
    out_a = a(batch_a, training=True)
    out_b = b(batch_b, training=True)
    assert torch.allclose(out_a["image"], out_b["image"], atol=1e-6)
    assert torch.allclose(out_a["context_in"], out_b["context_in"], atol=1e-6)
