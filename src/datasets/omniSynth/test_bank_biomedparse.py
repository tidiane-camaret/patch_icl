import os

import numpy as np
import pytest

import sys; sys.path.insert(0, ".")
from src.datasets.omniSynth.config import OmniMedSegConfig
from src.datasets.omniSynth.bank_biomedparse import BiomedParseObjectBank

DSETS = ("ACDC", "ISIC", "LIDC-IDRI", "amos22")
CFG = OmniMedSegConfig(train_datasets=DSETS, val_datasets=DSETS,
                       max_renditions_per_class=25)
CELL = 32

# Skip cleanly when the BiomedParse store isn't mounted.
_have_data = os.path.isdir(os.path.join(CFG.biomedparse_root, "train", "ACDC"))
pytestmark = pytest.mark.skipif(not _have_data, reason="BiomedParse store not available")


def test_classes_are_dataset_slash_target():
    bank = BiomedParseObjectBank(CFG, CELL, cell_margin=-0.15, split="train")
    ids = bank.task_ids()
    assert len(ids) > 0
    names = [bank.alphabet(c) for c in ids]
    assert any(n.startswith("ACDC/") for n in names)          # "<dataset>/<target>"
    assert all("/" in n for n in names)


def test_renditions_two_channel_masked_intensity():
    bank = BiomedParseObjectBank(CFG, CELL, cell_margin=-0.15, split="train", image_size=128)
    r = bank.get(bank.task_ids()[0])[0]
    assert r.ndim == 3 and r.shape[0] == 2 and r.dtype == np.float32
    intensity, mask = r[0], r[1]
    assert set(np.unique(mask)).issubset({0.0, 1.0}) and mask.sum() > 0
    assert float(intensity[mask == 0].max(initial=0.0)) == 0.0   # texture only under mask
    assert 0.0 <= intensity.min() and intensity.max() <= 1.0


def test_train_uses_train_store_val_uses_test_store():
    # Same datasets, but train reads the train store and val the test store, so a shared
    # class's renditions differ between splits.
    bt = BiomedParseObjectBank(CFG, CELL, split="train")
    bv = BiomedParseObjectBank(CFG, CELL, split="val")
    nt = {bt.alphabet(c): c for c in bt.task_ids()}
    nv = {bv.alphabet(c): c for c in bv.task_ids()}
    shared = set(nt) & set(nv)
    assert shared
    n = sorted(shared)[0]
    assert not np.array_equal(bt.get(nt[n])[0], bv.get(nv[n])[0])


if __name__ == "__main__":
    test_classes_are_dataset_slash_target()
    test_renditions_two_channel_masked_intensity()
    test_train_uses_train_store_val_uses_test_store()
    print("ALL BIOMEDPARSE BANK TESTS PASSED")
