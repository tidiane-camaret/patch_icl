import sys; sys.path.insert(0, ".")
import numpy as np
from src.datasets.omniSynth.config import OmniDiversityConfig
from src.datasets.omniSynth.bank import get_or_build_bank, OmniglotBank

DIV = OmniDiversityConfig()
CELL = 16


def test_pools_nonempty_and_disjoint():
    bank = get_or_build_bank(DIV, CELL)
    train, val, test = bank.task_ids("train"), bank.task_ids("val"), bank.task_ids("test")
    assert len(train) > 100          # ~964 background classes
    assert len(val) > 0 and len(test) > 0
    assert set(train).isdisjoint(val) and set(train).isdisjoint(test)
    assert set(val).isdisjoint(test)


def test_renditions_are_cell_sized_binary_foreground():
    bank = get_or_build_bank(DIV, CELL)
    cid = bank.task_ids("val")[0]
    rends = bank.get(cid)
    assert len(rends) >= 1
    r = rends[0]
    assert r.shape == (CELL, CELL) and r.dtype == np.uint8
    assert set(np.unique(r)).issubset({0, 1})
    assert r.sum() > 0               # inverted: foreground (ink) is 1, not all-zero
    assert r.sum() < r.size          # not all-foreground (background present)


def test_alphabet_lookup():
    bank = get_or_build_bank(DIV, CELL)
    cid = bank.task_ids("train")[0]
    assert isinstance(bank.alphabet(cid), str) and len(bank.alphabet(cid)) > 0


def test_val_test_split_deterministic():
    # Build two independent instances (bypassing cache) to truly exercise the seeded split.
    b1 = OmniglotBank(OmniDiversityConfig(), CELL)
    b2 = OmniglotBank(OmniDiversityConfig(), CELL)
    val1 = b1.task_ids("val")
    val2 = b2.task_ids("val")
    assert len(val1) > 0, "val pool must be non-empty"
    assert val1 == val2, "seeded split must be deterministic"


if __name__ == "__main__":
    test_pools_nonempty_and_disjoint()
    test_renditions_are_cell_sized_binary_foreground()
    test_alphabet_lookup()
    test_val_test_split_deterministic()
    print("ALL BANK TESTS PASSED")
