import sys; sys.path.insert(0, ".")
import torch
from types import SimpleNamespace
from src.gpu_augment import _stack_task, _unstack_task, _geometric, _batched_intensity, _batched_gin_ipa
from src.totalseg_dataset import CT_NORM_MIN, CT_NORM_MAX


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


def _int_cfg():
    return SimpleNamespace(
        brightness_contrast=SimpleNamespace(p=1.0, brightness=0.1, contrast_range=[0.8, 1.2]),
        gamma=SimpleNamespace(p=1.0, range=[0.8, 1.3]),
        gaussian_noise=SimpleNamespace(p=1.0, max_std=0.1),
        gaussian_blur=SimpleNamespace(p=1.0, sigma_range=[0.5, 1.0]),
    )


def _gin_cfg(mode="ipa"):
    return SimpleNamespace(p=1.0, mode=mode, n_layer=4, interm_channel=2,
                           scale_pool=[1, 3], out_norm="frob",
                           ipa_copies=2, ipa_control_points=3)


def test_intensity_shape_range_and_changes():
    span = CT_NORM_MAX - CT_NORM_MIN
    vols = torch.rand(5, 1, 8, 8, 8) * span + CT_NORM_MIN
    gen = torch.Generator().manual_seed(3)
    out = _batched_intensity(vols.clone(), _int_cfg(), gen)
    assert out.shape == vols.shape
    assert out.min() >= CT_NORM_MIN - 1e-4 and out.max() <= CT_NORM_MAX + 1e-4
    assert not torch.allclose(out, vols)


def test_intensity_p_zero_is_noop():
    cfg = _int_cfg()
    for k in ("brightness_contrast", "gamma", "gaussian_noise", "gaussian_blur"):
        getattr(cfg, k).p = 0.0
    vols = torch.rand(3, 1, 8, 8, 8)
    gen = torch.Generator().manual_seed(4)
    out = _batched_intensity(vols.clone(), cfg, gen)
    assert torch.allclose(out, vols)


def test_gin_ipa_shape_range_changes():
    span = CT_NORM_MAX - CT_NORM_MIN
    vols = torch.rand(4, 1, 8, 8, 8) * span + CT_NORM_MIN
    gen = torch.Generator().manual_seed(5)
    for mode in ("gin", "ipa"):
        out = _batched_gin_ipa(vols.clone(), _gin_cfg(mode), gen)
        assert out.shape == vols.shape
        assert out.min() >= CT_NORM_MIN - 1e-4 and out.max() <= CT_NORM_MAX + 1e-4
        assert not torch.allclose(out, vols)


def test_intensity_invokes_gin_when_configured():
    span = CT_NORM_MAX - CT_NORM_MIN
    vols = torch.rand(3, 1, 8, 8, 8) * span + CT_NORM_MIN
    cfg = _int_cfg()
    for k in ("brightness_contrast", "gamma", "gaussian_noise", "gaussian_blur"):
        getattr(cfg, k).p = 0.0
    cfg.gin = _gin_cfg("gin")
    gen = torch.Generator().manual_seed(6)
    out = _batched_intensity(vols.clone(), cfg, gen)
    assert not torch.allclose(out, vols)             # gin fired even with others off


def _full_cfg():
    return SimpleNamespace(
        enabled=True,
        task=_geo_cfg(affine_p=1.0),
        per_image=_geo_cfg(affine_p=1.0),
        synth=SimpleNamespace(**vars(_geo_cfg(affine_p=1.0)), **{
            "brightness_contrast": SimpleNamespace(p=1.0, brightness=0.1, contrast_range=[0.8, 1.2]),
            "gamma": SimpleNamespace(p=1.0, range=[0.8, 1.2]),
            "gaussian_noise": SimpleNamespace(p=1.0, mean_range=[0.0, 0.05], std_range=[0.0, 0.05]),
            "gaussian_blur": SimpleNamespace(p=0.0, sigma_range=[0.5, 1.0]),
        }),
        intensity=_int_cfg(),
    )


def test_eval_is_identity():
    from src.gpu_augment import GpuAugmentor
    b = _fake_batch()
    ref = {k: v.clone() for k, v in b.items()}
    aug = GpuAugmentor(_full_cfg())
    out = aug(b, training=False)
    for k in ("image", "context_in"):
        assert torch.allclose(out[k], ref[k])


def test_real_mode_shares_geometry_across_task():
    from src.gpu_augment import GpuAugmentor
    # target and its contexts identical -> shared geo keeps their geometry aligned
    B, K, D = 1, 3, 8
    base = torch.randn(1, 1, D, D, D)
    b = {
        "image": base.clone(),
        "label": torch.zeros(B, D, D, D, dtype=torch.long),
        "context_in": base.view(1, 1, 1, D, D, D).repeat(1, K, 1, 1, 1, 1),
        "context_out": torch.zeros(B, K, D, D, D, dtype=torch.long),
        "aug_mode": torch.tensor([0]),
    }
    cfg = _full_cfg()
    for kk in ("brightness_contrast", "gamma", "gaussian_noise", "gaussian_blur"):
        getattr(cfg.intensity, kk).p = 0.0            # isolate geometry
    aug = GpuAugmentor(cfg)
    out = aug(b, training=True)
    # target and each context underwent the SAME geometric transform
    for k in range(K):
        assert torch.allclose(out["image"][0, 0], out["context_in"][0, k, 0], atol=1e-5)


def test_mixed_modes_route_and_preserve_shape():
    from src.gpu_augment import GpuAugmentor
    B, K, D = 3, 2, 8
    b = {
        "image": torch.rand(B, 1, D, D, D),
        "label": torch.zeros(B, D, D, D, dtype=torch.long),
        "context_in": torch.rand(B, K, 1, D, D, D),
        "context_out": torch.zeros(B, K, D, D, D, dtype=torch.long),
        "aug_mode": torch.tensor([0, 1, 2]),          # real, synth, self_context
    }
    aug = GpuAugmentor(_full_cfg(), self_context_per_image=True)
    out = aug(b, training=True)
    assert out["image"].shape == (B, 1, D, D, D)
    assert out["context_in"].shape == (B, K, 1, D, D, D)
    assert out["context_out"].dtype == torch.long
