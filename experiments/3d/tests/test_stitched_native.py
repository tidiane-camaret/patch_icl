"""evaluate.py — _stitched_native_metrics_and_levels vs. the two calls it replaces in
cascade.py's evaluate_cascade (_stitched_native_metrics_multi + per-level
_stitched_native_dice_multi)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # experiments/3d siblings

import numpy as np
import pytest

from evaluate import (_stitched_native_dice_multi, _stitched_native_metrics_multi,
                      _stitched_native_metrics_and_levels)


def _entry(pred):
    """pred: (T,T,T) bool ndarray -> a pg_levels[i][key] entry (packbits, shape, geom).
    Identity geom (starts=pad=0, crop=out=T) so _place_patch writes `pred` unchanged at
    the native origin -- keeps the fixtures simple; the placement math itself is exercised
    by test_cascade.py's invert_geo_center / _place_patch call sites."""
    T = pred.shape[0]
    geom = np.array([[0, 0, 0], [T, T, T], [T, T, T], [0, 0, 0]], dtype=np.int64)
    return np.packbits(pred), tuple(pred.shape), geom


def _rand_bool(rng, T=6):
    return rng.random((T, T, T)) > 0.5


@pytest.fixture
def three_level_fixture():
    rng = np.random.default_rng(0)
    T = 6
    gt = {"s0": _rand_bool(rng, T), "s1": _rand_bool(rng, T)}
    keys = [("s0", "organ"), ("s1", "organ")]
    pg_levels = [dict() for _ in range(3)]
    for li in range(3):
        for k in keys:
            pg_levels[li][k] = _entry(_rand_bool(rng, T))
    gt_loader = lambda subj, cls: gt[subj]
    return pg_levels, gt_loader, keys


def test_full_matches_metrics_multi(three_level_fixture):
    pg_levels, gt_loader, keys = three_level_fixture
    full, _ = _stitched_native_metrics_and_levels(pg_levels, "/dummy", gt_loader=gt_loader)
    want = _stitched_native_metrics_multi(pg_levels, "/dummy", gt_loader=gt_loader)
    assert set(full) == set(want) == set(keys)
    for k in keys:
        assert full[k] == want[k]


def test_per_level_matches_dice_multi(three_level_fixture):
    pg_levels, gt_loader, keys = three_level_fixture
    _, per_level = _stitched_native_metrics_and_levels(pg_levels, "/dummy", gt_loader=gt_loader)
    for li in range(3):
        want = _stitched_native_dice_multi([pg_levels[li]], "/dummy", gt_loader=gt_loader)
        assert per_level[li] == want


def test_nsd_matches_metrics_multi(three_level_fixture):
    pg_levels, gt_loader, keys = three_level_fixture
    full, _ = _stitched_native_metrics_and_levels(
        pg_levels, "/dummy", tol_mm=2.0, gt_loader=gt_loader)
    want = _stitched_native_metrics_multi(pg_levels, "/dummy", tol_mm=2.0, gt_loader=gt_loader)
    for k in keys:
        d, nsd = full[k]
        wd, wnsd = want[k]
        assert d == wd
        assert nsd == pytest.approx(wnsd)


def test_key_missing_from_one_level_drops_full_but_keeps_solo_for_present_levels():
    T = 6
    rng = np.random.default_rng(1)
    key = ("s0", "organ")
    pg_levels = [dict(), dict()]
    pg_levels[0][key] = _entry(_rand_bool(rng, T))
    # level 1 lacks `key` entirely -> full stitch must skip it (matches
    # _stitched_native_metrics_multi's "present in every level" requirement), but level 0's
    # own solo dice is still computable and must match _stitched_native_dice_multi([lvl0], ..).
    gt_loader = lambda subj, cls: _rand_bool(rng, T)
    full, per_level = _stitched_native_metrics_and_levels(pg_levels, "/dummy", gt_loader=gt_loader)
    assert key not in full
    assert key in per_level[0]
    assert key not in per_level[1]


def test_gt_loader_called_once_per_key(three_level_fixture):
    pg_levels, gt_loader, keys = three_level_fixture
    calls = []
    def counting_loader(subj, cls):
        calls.append((subj, cls))
        return gt_loader(subj, cls)
    _stitched_native_metrics_and_levels(pg_levels, "/dummy", gt_loader=counting_loader)
    assert sorted(calls) == sorted(keys)  # exactly one load per key, not N+1


def test_empty_pg_levels_returns_empty():
    full, per_level = _stitched_native_metrics_and_levels([], "/dummy")
    assert full == {}
    assert per_level == []
