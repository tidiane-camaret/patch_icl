"""RAM volume cache: fork-COW read-only preload (plan Task 1)."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.providers.volume_cache import get_cache, clear_cache


@pytest.fixture(autouse=True)
def _clean():
    clear_cache()
    yield
    clear_cache()


def _fake_root(tmp_path, ids=("s0", "s1", "s2")):
    for i, s in enumerate(ids):
        d = tmp_path / s
        d.mkdir()
        np.save(d / "ct_raw.npy", np.full((4, 5, 6), i, dtype=np.float16))
        np.save(d / "label.npy", np.full((4, 5, 6), i, dtype=np.uint8))
    return tmp_path


def test_loads_arrays_readonly(tmp_path):
    root = _fake_root(tmp_path)
    c = get_cache(root, ["s0", "s1", "s2"])
    assert set(c) == {"s0", "s1", "s2"}
    assert c["s1"]["ct_raw"].shape == (4, 5, 6)
    assert c["s1"]["ct_raw"].dtype == np.float16
    assert c["s1"]["label"].dtype == np.uint8
    assert c["s0"]["ct_raw"].flags.writeable is False
    assert c["s0"]["label"].flags.writeable is False


def test_idempotent_same_object_and_topup(tmp_path):
    root = _fake_root(tmp_path)
    c1 = get_cache(root, ["s0"])
    c2 = get_cache(root, ["s0", "s1"])
    assert c1 is c2                      # same singleton dict
    assert set(c2) == {"s0", "s1"}       # s1 topped up
    assert c1["s0"]["ct_raw"] is c2["s0"]["ct_raw"]   # s0 not reloaded


def test_max_subjects_caps_total(tmp_path):
    root = _fake_root(tmp_path)
    c = get_cache(root, ["s0", "s1", "s2"], max_subjects=2)
    assert len(c) == 2


def test_missing_files_skipped(tmp_path):
    root = _fake_root(tmp_path, ids=("s0",))
    (tmp_path / "s9").mkdir()            # dir exists, no npy
    c = get_cache(root, ["s0", "s9"])
    assert set(c) == {"s0"}
