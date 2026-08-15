import sys; sys.path.insert(0, ".")
import torch
from types import SimpleNamespace
from src.gpu_augment import _stack_task, _unstack_task, _geometric


def _fake_batch(B=2, K=3, D=6, H=6, W=6):
    return {
        "image":       torch.randn(B, 1, D, H, W),
        "label":       torch.randint(0, 2, (B, D, H, W)),
        "context_in":  torch.randn(B, K, 1, D, H, W),
        "context_out": torch.randint(0, 2, (B, K, D, H, W)),
        "aug_mode":    torch.zeros(B, dtype=torch.long),
    }


def test_stack_unstack_roundtrip():
    b = _fake_batch()
    ref = {k: v.clone() for k, v in b.items()}
    vols, masks, B, T = _stack_task(b)
    assert vols.shape == (B * T, 1, 6, 6, 6)
    assert masks.shape == (B * T, 6, 6, 6)
    assert masks.dtype == torch.long
    # target of task 0 is vols[0]; first context of task 0 is vols[1]
    assert torch.equal(vols[0, 0], ref["image"][0, 0])
    assert torch.equal(vols[1, 0], ref["context_in"][0, 0, 0])
    _unstack_task(vols, masks, B, T, b)
    for k in ("image", "label", "context_in", "context_out"):
        assert torch.equal(b[k], ref[k])


def _geo_cfg(affine_p=1.0, flip_p=0.0, elastic_p=0.0):
    return SimpleNamespace(
        flip=SimpleNamespace(p_d=flip_p, p_h=flip_p, p_w=flip_p),
        affine=SimpleNamespace(p=affine_p, max_angle_deg=30.0, scale_min=0.9,
                               scale_max=1.1, max_translate=0.1),
        elastic=SimpleNamespace(p=elastic_p, alpha=0.1, grid_scale=4),
    )


def test_geometric_shared_within_group():
    # 1 task, T=3 identical volumes -> shared transform keeps them identical
    D = 8
    vol = torch.randn(1, 1, D, D, D)
    vols = vol.repeat(3, 1, 1, 1, 1)                 # 3 identical volumes, one group
    masks = torch.randint(0, 2, (1, D, D, D)).repeat(3, 1, 1, 1)
    gen = torch.Generator().manual_seed(0)
    out, om = _geometric(vols.clone(), masks.clone(), group_size=3, cfg=_geo_cfg(), gen=gen)
    assert out.shape == vols.shape
    assert torch.allclose(out[0], out[1]) and torch.allclose(out[1], out[2])   # shared
    assert not torch.allclose(out[0], vols[0])       # actually transformed


def test_geometric_independent_diverges():
    D = 8
    vols = torch.randn(1, 1, D, D, D).repeat(4, 1, 1, 1, 1)
    masks = torch.zeros(4, D, D, D, dtype=torch.long)
    gen = torch.Generator().manual_seed(1)
    out, _ = _geometric(vols.clone(), masks.clone(), group_size=1, cfg=_geo_cfg(), gen=gen)
    assert not torch.allclose(out[0], out[1])        # independent per volume


def test_geometric_mask_follows_image():
    # a mask blob and an image blob at the same voxels move together
    D = 10
    vols = torch.zeros(2, 1, D, D, D); vols[:, 0, 2:5, 2:5, 2:5] = 1.0
    masks = torch.zeros(2, D, D, D, dtype=torch.long); masks[:, 2:5, 2:5, 2:5] = 1
    gen = torch.Generator().manual_seed(2)
    out, om = _geometric(vols, masks, group_size=2, cfg=_geo_cfg(), gen=gen)
    assert om.dtype == torch.long
    # where the mask is 1, the image is high (they co-moved)
    m = om[0] == 1
    assert m.sum() > 0 and out[0, 0][m].mean() > 0.3
