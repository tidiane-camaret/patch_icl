import sys; sys.path.insert(0, ".")
import torch
from types import SimpleNamespace
from omegaconf import OmegaConf
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


# ---------------------------------------------------------------------------
# Finding 1: self_context_intensity gate
# ---------------------------------------------------------------------------
def test_self_context_per_image_does_not_touch_target():
    """per_image jitter must not alter the target (t=0); only context clones (t>=1) change."""
    from src.gpu_augment import GpuAugmentor
    B, K, D = 2, 3, 8
    base = {
        "image": torch.randn(B, 1, D, D, D),
        "label": torch.zeros(B, D, D, D, dtype=torch.long),
        "context_in": torch.randn(B, K, 1, D, D, D),
        "context_out": torch.zeros(B, K, D, D, D, dtype=torch.long),
        "aug_mode": torch.tensor([2, 2]),
    }
    clone = {k: v.clone() for k, v in base.items()}

    # intensity off so only geometry distinguishes the two runs
    cfg = _full_cfg()
    for kk in ("brightness_contrast", "gamma", "gaussian_noise", "gaussian_blur"):
        getattr(cfg.intensity, kk).p = 0.0

    aug_true  = GpuAugmentor(cfg, self_context_per_image=True,  self_context_intensity=False, seed=0)
    aug_false = GpuAugmentor(cfg, self_context_per_image=False, self_context_intensity=False, seed=0)

    out_true  = aug_true(base,  training=True)
    out_false = aug_false(clone, training=True)

    # Both runs see the same shared geometric for the whole task (same seed, same REAL branch),
    # so the target (t=0) must come out identical.
    assert torch.allclose(out_true["image"], out_false["image"]), \
        "per_image flag must not touch the target volume (t=0)"


def test_self_context_intensity_gate():
    """With self_context_intensity=False identical clones must stay identical after augmentation.
    With self_context_intensity=True they must diverge (intensity ops make them differ)."""
    from src.gpu_augment import GpuAugmentor
    B, K, D = 1, 3, 8

    def _make_batch():
        img = torch.randn(B, 1, D, D, D)
        # context clones are identical copies of the target image
        ctx = img.unsqueeze(1).expand(B, K, 1, D, D, D).clone()
        return {
            "image": img.clone(),
            "label": torch.zeros(B, D, D, D, dtype=torch.long),
            "context_in": ctx,
            "context_out": torch.zeros(B, K, D, D, D, dtype=torch.long),
            "aug_mode": torch.tensor([2]),
        }

    cfg_off = _full_cfg()
    # disable per_image geo jitter so only intensity (or lack thereof) distinguishes volumes
    cfg_off.per_image = _geo_cfg(affine_p=0.0, flip_p=0.0, elastic_p=0.0)
    # confirm _full_cfg intensity has at least one op with p>0 (brightness_contrast.p=1.0)
    assert cfg_off.intensity.brightness_contrast.p > 0, \
        "_full_cfg must have at least one intensity op with p>0 for this test to be meaningful"

    # Case A: intensity gated OFF — all K+1 volumes share the same geometric, no intensity →
    # every volume should remain identical to the target after augmentation.
    aug_off = GpuAugmentor(cfg_off, self_context_per_image=False, self_context_intensity=False, seed=0)
    out_off = aug_off(_make_batch(), training=True)
    for k in range(K):
        assert torch.allclose(out_off["context_in"][:, k, 0], out_off["image"][:, 0], atol=1e-5), \
            f"context {k} diverged from target despite self_context_intensity=False"

    # Case B: intensity gated ON — per-volume intensity randomness should make at least one
    # context differ from the target.
    cfg_on = _full_cfg()
    cfg_on.per_image = _geo_cfg(affine_p=0.0, flip_p=0.0, elastic_p=0.0)
    aug_on = GpuAugmentor(cfg_on, self_context_per_image=False, self_context_intensity=True, seed=0)
    out_on = aug_on(_make_batch(), training=True)
    any_differ = any(
        not torch.allclose(out_on["context_in"][:, k, 0], out_on["image"][:, 0], atol=1e-5)
        for k in range(K)
    )
    assert any_differ, \
        "with self_context_intensity=True at least one context should differ from the target"


# ---------------------------------------------------------------------------
# Task 6: collate aug_mode
# ---------------------------------------------------------------------------
from src.totalseg_dataloader_incontext import incontext_collate_fn

def _item(mode, K=2, D=6):
    return {
        "image": torch.randn(1, D, D, D),
        "label": torch.zeros(D, D, D, dtype=torch.long),
        "context_in": torch.randn(K, 1, D, D, D),
        "context_out": torch.zeros(K, D, D, D, dtype=torch.long),
        "subject": "s0", "label_name": "x",
        "spacing": torch.ones(3),
        "context_subjects": ["s1", "s2"],
        "aug_mode": torch.tensor(mode, dtype=torch.long),
    }

def test_collate_stacks_aug_mode():
    out = incontext_collate_fn([_item(0), _item(2)])
    assert "aug_mode" in out
    assert out["aug_mode"].tolist() == [0, 2]
    assert out["aug_mode"].dtype == torch.long


# ---------------------------------------------------------------------------
# Task 7: config flag
# ---------------------------------------------------------------------------
def test_nnunet_config_has_gpu_flag():
    cfg = OmegaConf.load("configs/augmentations/nnunet.yaml")
    assert cfg.augmentations.gpu is False


# ---------------------------------------------------------------------------
# Task 8: end-to-end train-loop smoke
# ---------------------------------------------------------------------------
def test_augmentor_end_to_end_batch_smoke():
    # emulate the train-loop call: raw batch -> to(device) -> augmentor -> shapes intact
    from src.gpu_augment import GpuAugmentor
    B, K, D = 2, 3, 8
    b = {
        "image": torch.rand(B, 1, D, D, D),
        "label": torch.zeros(B, D, D, D, dtype=torch.long),
        "context_in": torch.rand(B, K, 1, D, D, D),
        "context_out": torch.zeros(B, K, D, D, D, dtype=torch.long),
        "aug_mode": torch.tensor([2, 2]),
        "spacing": torch.ones(B, 3),
    }
    aug = GpuAugmentor(_full_cfg(), self_context_per_image=True)
    out = aug(b, training=True)
    assert out["image"].shape == (B, 1, D, D, D)
    assert out["context_in"].shape == (B, K, 1, D, D, D)
    assert torch.isfinite(out["image"]).all()
    assert torch.isfinite(out["context_in"]).all()


# ---------------------------------------------------------------------------
# GPU-device regression: GIN/IPA on CUDA. The CPU tests never exercise the
# cuda generator path; _gin_once's torch.randint must pass device= or it
# builds a CPU op against a cuda Generator and raises. Skipped without CUDA.
# ---------------------------------------------------------------------------
def test_gin_ipa_run_on_cuda():
    if not torch.cuda.is_available():
        import pytest
        pytest.skip("no CUDA device")
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(0)
    span = CT_NORM_MAX - CT_NORM_MIN
    vols = (torch.rand(4, 1, 16, 16, 16, device=dev) * span + CT_NORM_MIN)
    for mode in ("gin", "ipa"):
        out = _batched_gin_ipa(vols.clone(), _gin_cfg(mode=mode), gen)
        assert out.shape == vols.shape
        assert out.device.type == "cuda"
        assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# Task 4: de-pinnable clamp frame
# ---------------------------------------------------------------------------
def test_batched_intensity_clamp_default_is_ct_frame():
    import torch
    from src.gpu_augment import _batched_intensity, CT_NORM_MIN, CT_NORM_MAX

    class _NC:  # gaussian-noise-only cfg forcing a large clamp excursion
        class gaussian_noise:
            p = 1.0
            max_std = 50.0
    g = torch.Generator().manual_seed(0)
    vols = torch.zeros(2, 1, 8, 8, 8)
    out = _batched_intensity(vols, _NC, g)
    assert out.max() <= CT_NORM_MAX + 1e-4
    assert out.min() >= CT_NORM_MIN - 1e-4


def test_batched_intensity_clamp_override():
    import torch
    from src.gpu_augment import _batched_intensity

    class _NC:
        class gaussian_noise:
            p = 1.0
            max_std = 50.0
    g = torch.Generator().manual_seed(0)
    vols = torch.zeros(2, 1, 8, 8, 8)
    out = _batched_intensity(vols, _NC, g, clamp=(-4.0, 4.0))
    assert out.max() <= 4.0 + 1e-4
    assert out.min() >= -4.0 - 1e-4


def test_gpu_augmentor_clamp_frame_skips_ct_guard():
    from src.gpu_augment import GpuAugmentor
    # A non-default ct_norm normally raises; clamp_frame set -> allowed.
    aug = GpuAugmentor(aug_cfg=None,
                       ct_norm={"clip_lo": -500.0, "clip_hi": 500.0, "mean": 0.0, "std": 100.0},
                       clamp_frame=(-3.0, 3.0))
    assert aug._clamp == (-3.0, 3.0)


def test_gpu_augmentor_default_still_guards():
    import pytest
    from src.gpu_augment import GpuAugmentor
    with pytest.raises(NotImplementedError):
        GpuAugmentor(aug_cfg=None,
                     ct_norm={"clip_lo": -500.0, "clip_hi": 500.0, "mean": 0.0, "std": 100.0})
