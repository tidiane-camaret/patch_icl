import sys; sys.path.insert(0, ".")
import pickle
from pathlib import Path

import numpy as np
import pytest

from src.datasets.omniSynth.bank_totalseg import TotalSegObjectBank


def _fixture_cache(tmp_path, size=16):
    """Write a minimal T{size} train cache with 2 classes, 3 tiles each."""
    split_dir = tmp_path / f"T{size}" / "train"
    split_dir.mkdir(parents=True)
    index = {1: "adrenal_gland_left", 3: "aorta"}
    for lv, name in index.items():
        tiles = []
        for _ in range(3):
            t = np.zeros((2, 8, 8, 8), dtype=np.float16)
            t[0, 2:6, 2:6, 2:6] = 0.5
            t[1, 2:6, 2:6, 2:6] = 1.0
            tiles.append(t)
        (split_dir / f"class_{lv}.pkl").write_bytes(
            pickle.dumps({"name": name, "tiles": tiles}))
    (split_dir / "index.pkl").write_bytes(pickle.dumps(index))
    return tmp_path


def test_interface_parity(tmp_path):
    root = _fixture_cache(tmp_path)
    bank = TotalSegObjectBank(root, (16, 16, 16), "train")
    ids = bank.task_ids()
    assert set(ids) == {1, 3}
    assert bank.alphabet(1) == "adrenal_gland_left"
    r = bank.get(3)
    assert len(r) == 3 and r[0].shape == (2, 8, 8, 8) and r[0].dtype == np.float16


def test_class_subset_filter(tmp_path):
    root = _fixture_cache(tmp_path)
    bank = TotalSegObjectBank(root, (16, 16, 16), "train", classes=("aorta",))
    assert bank.task_ids() == [3]


def test_get_is_cached(tmp_path):
    root = _fixture_cache(tmp_path)
    bank = TotalSegObjectBank(root, (16, 16, 16), "train")
    assert bank.get(1) is bank.get(1)                    # same list object (LRU hit)


def test_missing_cache_raises(tmp_path):
    with pytest.raises((FileNotFoundError, ValueError)):
        TotalSegObjectBank(tmp_path, (16, 16, 16), "train")


def test_lru_evicts_when_over_capacity(tmp_path):
    root = _fixture_cache(tmp_path)
    bank = TotalSegObjectBank(root, (16, 16, 16), "train", lru_classes=1)
    first = bank.get(1)
    bank.get(3)                                          # evicts class 1 (lru_classes=1)
    assert bank.get(1) is not first                      # re-loaded from disk, new object


def test_get_unknown_class_id_raises(tmp_path):
    root = _fixture_cache(tmp_path)
    bank = TotalSegObjectBank(root, (16, 16, 16), "train")
    with pytest.raises(KeyError):
        bank.get(999)
    with pytest.raises(KeyError):
        bank.alphabet(999)


if __name__ == "__main__":
    import tempfile
    for fn in (test_interface_parity, test_class_subset_filter,
               test_get_is_cached, test_missing_cache_raises):
        with tempfile.TemporaryDirectory() as d:
            fn(Path(d))
    print("ALL TOTALSEG BANK TESTS PASSED")
