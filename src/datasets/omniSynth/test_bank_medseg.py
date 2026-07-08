import os

import numpy as np
import pytest

import sys; sys.path.insert(0, ".")
from src.datasets.omniSynth.config import OmniMedSegConfig
from src.datasets.omniSynth.bank_medseg import MedSegObjectBank

DSETS = ("busi", "drive", "cellnuclei", "isic2018", "monusac", "kvasir", "wbc", "pandental")
CFG = OmniMedSegConfig(train_datasets=DSETS, val_datasets=DSETS,
                       max_renditions_per_class=30)
CELL = 32

# Skip cleanly when the MedSegBench npz store isn't mounted (mirrors the omniglot
# bank test's implicit dependence on the zips being present).
_have_data = os.path.exists(os.path.join(CFG.data_root, f"busi_{CFG.source_size}.npz"))
pytestmark = pytest.mark.skipif(not _have_data, reason="MedSegBench data not available")


def test_pools_nonempty_and_named():
    train = MedSegObjectBank(CFG, CELL, cell_margin=-0.15, split="train").task_ids()
    val = MedSegObjectBank(CFG, CELL, cell_margin=-0.15, split="val").task_ids()
    assert len(train) > 0 and len(val) > 0
    name = MedSegObjectBank(CFG, CELL, split="train").alphabet(train[0])
    assert "/label_" in name          # "<dataset>/label_<lv>"


def test_train_val_read_different_images():
    # Same datasets feed both pools (default), but each reads its own MedSegBench split,
    # so a class's rendition pools differ between train and val.
    bt = MedSegObjectBank(CFG, CELL, split="train")
    bv = MedSegObjectBank(CFG, CELL, split="val")
    names_t = {bt.alphabet(c): c for c in bt.task_ids()}
    names_v = {bv.alphabet(c): c for c in bv.task_ids()}
    shared = set(names_t) & set(names_v)
    assert shared                                     # same classes present in both splits
    n = next(iter(shared))
    rt, rv = bt.get(names_t[n]), bv.get(names_v[n])
    assert not np.array_equal(rt[0], rv[0])           # different underlying images


def test_renditions_are_two_channel_masked_intensity():
    bank = MedSegObjectBank(CFG, CELL, cell_margin=-0.15, split="train")
    r = bank.get(bank.task_ids()[0])[0]
    assert r.ndim == 3 and r.shape[0] == 2 and r.dtype == np.float32
    intensity, mask = r[0], r[1]
    assert set(np.unique(mask)).issubset({0.0, 1.0})
    assert mask.sum() > 0
    # intensity is masked: no texture bleeds outside the object
    assert float(intensity[mask == 0].max(initial=0.0)) == 0.0
    assert 0.0 <= intensity.min() and intensity.max() <= 1.0


def test_dataset_subset_selection():
    cfg = OmniMedSegConfig(train_datasets=("busi",), val_datasets=("kvasir",))
    train = MedSegObjectBank(cfg, CELL, split="train")
    val = MedSegObjectBank(cfg, CELL, split="val")
    assert train.task_ids() and val.task_ids()
    assert all(train.alphabet(c).startswith("busi/") for c in train.task_ids())
    assert all(val.alphabet(c).startswith("kvasir/") for c in val.task_ids())


if __name__ == "__main__":
    test_pools_nonempty_and_named()
    test_train_val_read_different_images()
    test_renditions_are_two_channel_masked_intensity()
    test_dataset_subset_selection()
    print("ALL MEDSEG BANK TESTS PASSED")
