"""Guard for the eval spacing sweep: totalseg + use_crop only, else a clear ValueError."""
import sys
from pathlib import Path

import pytest
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # sibling eval.py

from eval import _assert_sweep_supported  # noqa: E402


def _cfg(source="totalseg", use_crop=True, **eval_kw):
    return OmegaConf.create(
        {"data": {"source": source, "use_crop": use_crop}, "eval": dict(eval_kw)})


def test_totalseg_crop_ok():
    _assert_sweep_supported(_cfg())  # no raise


def test_use_crop_false_rejected():
    with pytest.raises(ValueError, match="use_crop"):
        _assert_sweep_supported(_cfg(use_crop=False))


@pytest.mark.parametrize("src", ["omnisynth3d", "anchor_synth3d"])
def test_unsupported_source_rejected(src):
    with pytest.raises(ValueError, match="spacing_sweep"):
        _assert_sweep_supported(_cfg(source=src))


def test_totalseg_more_labels_crop_ok():
    # more_labels subclasses TotalSegInContextDataset, so it honours the per-item spacing
    # override (its _load_crop sizes the FOV as T*self._crop_mm) — supported like totalseg.
    _assert_sweep_supported(_cfg(source="totalseg_more_labels"))  # no raise


def test_locator_with_descending_sweep_ok():
    _assert_sweep_supported(_cfg(spacing_locator=True, spacing_sweep=[4, 2]))  # no raise


def test_locator_without_descending_step_rejected():
    with pytest.raises(ValueError, match="spacing_locator"):
        _assert_sweep_supported(_cfg(spacing_locator=True, spacing_sweep=[2, 4]))


def test_locator_single_spacing_rejected():
    with pytest.raises(ValueError, match="spacing_locator"):
        _assert_sweep_supported(_cfg(spacing_locator=True, spacing_sweep=[2]))
