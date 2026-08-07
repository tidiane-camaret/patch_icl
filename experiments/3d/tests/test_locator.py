"""Geometry unit tests for the coarse->fine locator containment helper."""
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # sibling evaluate.py

from evaluate import _locator_containment  # noqa: E402


def _cube(T, sl):
    v = torch.zeros((T, T, T))
    v[sl, sl, sl] = 1.0
    return v


def test_perfect_locator_object_fits():
    # T=8, ratio 0.5 -> box side 4 centered; object 2^3 at center fits entirely.
    obj = _cube(8, slice(3, 5))
    cont, orc, empty, err = _locator_containment(obj, obj, 0.5)
    assert cont == 1.0 and orc == 1.0
    assert empty is False
    assert err < 1e-6


def test_object_larger_than_box():
    # GT fills all 8^3=512; centered box 4^3=64 -> containment 64/512 = 0.125.
    gt = torch.ones((8, 8, 8))
    cont, orc, empty, err = _locator_containment(gt, gt, 0.5)
    assert abs(cont - 0.125) < 1e-6
    assert abs(orc - 0.125) < 1e-6
    assert empty is False


def test_empty_prob_falls_back_to_center():
    prob = torch.zeros((8, 8, 8))
    gt = _cube(8, slice(3, 5))
    cont, orc, empty, err = _locator_containment(prob, gt, 0.5)
    assert empty is True
    # Center fallback = crop center, which coincides with the centered object -> full.
    assert cont == 1.0


def test_offset_prediction_low_containment_high_oracle():
    # GT in one corner, prediction blob in the opposite corner.
    gt = _cube(8, slice(0, 2))
    prob = _cube(8, slice(6, 8))
    cont, orc, empty, err = _locator_containment(prob, gt, 0.5)
    assert cont == 0.0            # pred box misses the GT corner
    assert orc == 1.0            # oracle box centered on GT captures it
    assert err > 5.0             # centroids ~6 voxels apart per axis


def test_gt_empty_returns_nan():
    prob = _cube(8, slice(3, 5))
    gt = torch.zeros((8, 8, 8))
    cont, orc, empty, err = _locator_containment(prob, gt, 0.5)
    assert math.isnan(cont) and math.isnan(orc) and math.isnan(err)
