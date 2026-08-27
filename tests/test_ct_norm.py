"""CtNormSpec / resolve_ct_norm / normalize_ct — the one CT normalization frame."""
import numpy as np
import pytest

from src import totalseg_dataset as td
from src.totalseg_dataset import (CT_CLIP_MIN, CT_CLIP_MAX, CT_MEAN, CT_STD,
                                  CT_NORM_MIN, CT_NORM_MAX, CtNormSpec,
                                  DEFAULT_CT_NORM, normalize_ct, resolve_ct_norm)


def test_back_compat_constants_track_default_frame():
    s = DEFAULT_CT_NORM
    assert (CT_CLIP_MIN, CT_CLIP_MAX, CT_MEAN, CT_STD) == (s.clip_lo, s.clip_hi, s.mean, s.std)
    assert CT_NORM_MIN == pytest.approx(s.norm_min)
    assert CT_NORM_MAX == pytest.approx(s.norm_max)
    # values the rest of the codebase (convert_to_npy assert) depends on
    assert (CT_CLIP_MIN, CT_CLIP_MAX) == (-1007.0, 1573.0)


def test_resolve_forms():
    assert resolve_ct_norm(None) is DEFAULT_CT_NORM
    assert resolve_ct_norm("fingerprint_1228") is DEFAULT_CT_NORM
    spec = CtNormSpec(-1004.0, 1588.0, -50.0, 503.0)
    assert resolve_ct_norm(spec) is spec
    assert resolve_ct_norm({"clip_lo": -1004.0, "clip_hi": 1588.0,
                            "mean": -50.0, "std": 503.0}) == spec
    with pytest.raises(KeyError):
        resolve_ct_norm("nope")


def test_normalize_ct_default_unchanged():
    rng = np.random.default_rng(0)
    hu = rng.uniform(-2000, 3000, size=(4, 5, 6)).astype(np.float32)
    want = (np.clip(hu, CT_CLIP_MIN, CT_CLIP_MAX) - CT_MEAN) / CT_STD
    assert np.allclose(normalize_ct(hu), want)
    assert np.allclose(normalize_ct(hu, None), want)
    assert np.allclose(normalize_ct(hu, "fingerprint_1228"), want)


def test_normalize_ct_alt_frame():
    d297 = td.CT_NORM_PRESETS["d297"]
    hu = np.array([-3000.0, 0.0, 500.0, 5000.0], np.float32)
    want = (np.clip(hu, d297.clip_lo, d297.clip_hi) - d297.mean) / d297.std
    assert np.allclose(normalize_ct(hu, "d297"), want)
