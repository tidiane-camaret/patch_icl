"""Task 6: _assert_cascade_supported guard."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from omegaconf import OmegaConf

from common import _assert_cascade_supported


def _cfg(**over):
    base = {
        "model": "patchset3d",
        "data": {"loader_v2": True, "source": "totalseg", "crop_spacing_mm": 3,
                 "cascade_spacings": [3, 1.5], "train_spacing_range": None},
        "train": {"cascade_loss_weights": [1.0, 1.0]},
    }
    cfg = OmegaConf.create(base)
    cfg.merge_with(OmegaConf.create(over))
    return cfg


def test_ok():
    _assert_cascade_supported(_cfg())            # no raise


def test_off_is_noop():
    _assert_cascade_supported(_cfg(data={"cascade_spacings": None}))


def test_rejects_non_patchset():
    with pytest.raises(ValueError, match="patchset3d"):
        _assert_cascade_supported(_cfg(model="medverse"))


def test_rejects_loader_v1():
    with pytest.raises(ValueError, match="loader_v2"):
        _assert_cascade_supported(_cfg(data={"loader_v2": False}))


def test_rejects_spacing_mismatch():
    with pytest.raises(ValueError, match="crop_spacing_mm"):
        _assert_cascade_supported(_cfg(data={"crop_spacing_mm": 2}))


def test_rejects_train_spacing_range_combo():
    with pytest.raises(ValueError, match="train_spacing_range"):
        _assert_cascade_supported(_cfg(data={"train_spacing_range": [1.5, 3.0]}))


def test_rejects_short_list():
    with pytest.raises(ValueError, match="at least 2"):
        _assert_cascade_supported(_cfg(data={"cascade_spacings": [3], "crop_spacing_mm": 3}))


def test_rejects_weight_length_mismatch():
    with pytest.raises(ValueError, match="cascade_loss_weights"):
        _assert_cascade_supported(_cfg(train={"cascade_loss_weights": [1.0]}))


def test_rejects_enabled_cpu_aug():
    with pytest.raises(ValueError, match="augmentations.gpu"):
        _assert_cascade_supported(_cfg(augmentations={"enabled": True, "gpu": False}))


def test_allows_enabled_gpu_aug():
    _assert_cascade_supported(_cfg(augmentations={"enabled": True, "gpu": True}))


def test_warns_non_descending_spacings():
    # Spec: non-descending is a warning, not an error (valid for ablations).
    with pytest.warns(UserWarning, match="coarse->fine"):
        _assert_cascade_supported(_cfg(data={"cascade_spacings": [1.5, 3], "crop_spacing_mm": 1.5}))


def test_warns_equal_adjacent_spacings():
    with pytest.warns(UserWarning, match="coarse->fine"):
        _assert_cascade_supported(_cfg(data={"cascade_spacings": [3, 3], "crop_spacing_mm": 3}))


def test_accepts_query_prior_mixture():
    _assert_cascade_supported(_cfg(data={"cascade_query_prior":
                                         {"modes": ["pred", "none", "gt"], "p": [0.4, 0.4, 0.2]}}))


def test_rejects_query_prior_bad_mixture():
    with pytest.raises(ValueError, match="cascade_query_prior"):
        _assert_cascade_supported(_cfg(data={"cascade_query_prior":
                                             {"modes": ["pred", "oracle"]}}))


def test_cascade_realize_requires_ram_cache():
    # Explicit ram_cache: false on a gpu_realize_crop cascade run -> hard error.
    with pytest.raises(ValueError, match="ram_cache"):
        _assert_cascade_supported(_cfg(data={"gpu_realize_crop": True, "ram_cache": False}))


def test_cascade_realize_default_ok():
    # Neither key set under a cascade config -> both default true -> no raise.
    _assert_cascade_supported(_cfg())
