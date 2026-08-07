"""Guard for the eval spacing sweep: totalseg + use_crop only, else a clear ValueError."""
import sys
from pathlib import Path

import pytest
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # sibling eval.py

from eval import _assert_sweep_supported  # noqa: E402


def _cfg(source="totalseg", use_crop=True):
    return OmegaConf.create({"data": {"source": source, "use_crop": use_crop}})


def test_totalseg_crop_ok():
    _assert_sweep_supported(_cfg())  # no raise


def test_use_crop_false_rejected():
    with pytest.raises(ValueError, match="use_crop"):
        _assert_sweep_supported(_cfg(use_crop=False))


@pytest.mark.parametrize("src", ["omnisynth3d", "anchor_synth3d", "totalseg_more_labels"])
def test_unsupported_source_rejected(src):
    with pytest.raises(ValueError, match="spacing_sweep"):
        _assert_sweep_supported(_cfg(source=src))
