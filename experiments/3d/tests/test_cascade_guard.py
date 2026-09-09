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


def test_accepts_medverse():
    _assert_cascade_supported(_cfg(model="medverse"))          # v2 cascade eval via train_forward


def test_rejects_other_model():
    with pytest.raises(ValueError, match="patchset3d or model=medverse"):
        _assert_cascade_supported(_cfg(model="native_resenc"))


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


def test_allows_mri_source_with_gpu_realize():
    # NativeCrop now carries a per-subject CtNormSpec, so MRI GPU-realize normalizes
    # correctly (2026-09-07 multisource-cascade spec).
    _assert_cascade_supported(_cfg(data={"source": "totalsegmri"}))


def test_allows_mri_source_without_gpu_realize():
    _assert_cascade_supported(_cfg(data={"source": "totalsegmri",
                                         "gpu_realize_crop": False}))


def test_allows_multisource_source():
    _assert_cascade_supported(_cfg(data={"source": "multisource"}))


def test_allows_multisource_with_gpu_realize_and_ram_cache():
    _assert_cascade_supported(_cfg(data={"source": "multisource",
                                         "gpu_realize_crop": True, "ram_cache": True}))


def test_rejects_gpu_realize_without_cascade_spacings():
    # native-crop payloads only have a consumer under the cascade train loop; without
    # cascade_spacings the loader would push NativeCrop dataclasses into default_collate.
    with pytest.raises(ValueError, match="cascade_spacings"):
        _assert_cascade_supported(_cfg(data={"cascade_spacings": None,
                                             "gpu_realize_crop": True}))


# --- data.cascade_train: random per-batch N-level training ladder ---------------

def test_allows_cascade_train_within_eval_range():
    _assert_cascade_supported(_cfg(data={"cascade_train":
                                         {"levels": 2, "spacing_range": [1.5, 3]}}))


def test_cascade_train_warns_out_of_eval_range():
    with pytest.warns(UserWarning, match="cascade_train"):
        _assert_cascade_supported(_cfg(data={"cascade_train":
                                             {"levels": 2, "spacing_range": [1.5, 6]}}))


def test_cascade_train_weight_accepts_train_levels_len():
    # base cascade_spacings has len 2; cascade_train.levels=3 -> weights of len 3 accepted.
    _assert_cascade_supported(_cfg(data={"cascade_train":
                                         {"levels": 3, "spacing_range": [1.5, 3]}},
                                   train={"cascade_loss_weights": [1.0, 1.0, 1.0]}))


def test_cascade_train_weight_accepts_eval_ladder_len():
    # levels=2 but weights match the 3-entry eval ladder -> accepted (train loop slices
    # to the coarsest 2). This is the exp-80 command shape ([1,1,1] + levels=2).
    _assert_cascade_supported(_cfg(
        data={"cascade_spacings": [6, 3, 1.5], "crop_spacing_mm": 6,
              "cascade_train": {"levels": 2, "spacing_range": [1.5, 6]}},
        train={"cascade_loss_weights": [1.0, 1.0, 1.0]}))


def test_cascade_train_weight_rejects_other_len():
    with pytest.raises(ValueError, match="cascade_loss_weights"):
        _assert_cascade_supported(_cfg(
            data={"cascade_train": {"levels": 2, "spacing_range": [1.5, 3]}},
            train={"cascade_loss_weights": [1.0]}))


def test_cascade_train_rejects_bad_range_order():
    with pytest.raises(ValueError, match="lo < hi"):
        _assert_cascade_supported(_cfg(data={"cascade_train":
                                             {"levels": 2, "spacing_range": [3, 1.5]}}))


def test_cascade_train_rejects_bad_range_len():
    with pytest.raises(ValueError, match="spacing_range"):
        _assert_cascade_supported(_cfg(data={"cascade_train":
                                             {"levels": 2, "spacing_range": [1.5]}}))


def test_cascade_train_rejects_levels_lt_2():
    with pytest.raises(ValueError, match="levels"):
        _assert_cascade_supported(_cfg(data={"cascade_train":
                                             {"levels": 1, "spacing_range": [1.5, 3]}}))
