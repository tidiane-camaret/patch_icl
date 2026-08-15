import random
import sys; sys.path.insert(0, ".")
from types import SimpleNamespace

import torch

from src.augmentations import _gin_transform_3d, _ipa_blend_3d, apply_intensity_aug
from src.totalseg_dataset import CT_NORM_MIN, CT_NORM_MAX


def _dummy_image(seed=0):
    g = torch.Generator().manual_seed(seed)
    # z-score-like values within the CT normalisation range
    img = torch.rand(1, 8, 12, 10, generator=g) * (CT_NORM_MAX - CT_NORM_MIN) + CT_NORM_MIN
    return img


def test_gin_preserves_shape_and_changes_image():
    img = _dummy_image()
    torch.manual_seed(1)
    out = _gin_transform_3d(img)
    assert out.shape == img.shape
    assert not torch.allclose(out, img)          # GIN actually warps intensities


def test_ipa_preserves_shape():
    img = _dummy_image()
    torch.manual_seed(2)
    out = _ipa_blend_3d(img, n_copies=3, control_points=3)
    assert out.shape == img.shape


def test_gin_deterministic_under_seed():
    img = _dummy_image()
    random.seed(42); torch.manual_seed(42)
    a = _gin_transform_3d(img)
    random.seed(42); torch.manual_seed(42)
    b = _gin_transform_3d(img)
    assert torch.allclose(a, b)


def test_apply_intensity_aug_gin_stays_in_range():
    # only GIN active; all other intensity ops disabled
    cfg = SimpleNamespace(
        gin=SimpleNamespace(p=1.0, mode="ipa", n_layer=4, interm_channel=2,
                            scale_pool=[1, 3], out_norm="frob",
                            ipa_copies=2, ipa_control_points=4),
        brightness_contrast=SimpleNamespace(p=0.0, brightness=0.0, contrast_range=[1.0, 1.0]),
        gamma=SimpleNamespace(p=0.0, range=[1.0, 1.0]),
        gaussian_noise=SimpleNamespace(p=0.0, max_std=0.0),
        gaussian_blur=SimpleNamespace(p=0.0, sigma_range=[0.5, 1.0]),
    )
    img = _dummy_image(seed=7)
    torch.manual_seed(3)
    out = apply_intensity_aug(img.clone(), cfg)
    assert out.shape == img.shape
    assert out.min() >= CT_NORM_MIN - 1e-4 and out.max() <= CT_NORM_MAX + 1e-4


def test_gin_disabled_is_noop_passthrough():
    # p=0 → GIN block skipped; with all others off, image unchanged
    cfg = SimpleNamespace(
        gin=SimpleNamespace(p=0.0, mode="gin"),
        brightness_contrast=SimpleNamespace(p=0.0, brightness=0.0, contrast_range=[1.0, 1.0]),
        gamma=SimpleNamespace(p=0.0, range=[1.0, 1.0]),
        gaussian_noise=SimpleNamespace(p=0.0, max_std=0.0),
        gaussian_blur=SimpleNamespace(p=0.0, sigma_range=[0.5, 1.0]),
    )
    img = _dummy_image(seed=9)
    out = apply_intensity_aug(img.clone(), cfg)
    assert torch.allclose(out, img)
