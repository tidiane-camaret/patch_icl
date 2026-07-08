import os

import numpy as np
import pytest

import sys; sys.path.insert(0, ".")
from src.datasets.omniSynth.config import OmniMedSegConfig, OmniSceneConfig
from src.datasets.omniSynth.bank_background import BackgroundBank

MED = OmniMedSegConfig()
_have_med = os.path.exists(os.path.join(MED.data_root, f"busi_{MED.source_size}.npz"))
pytestmark = pytest.mark.skipif(not _have_med, reason="MedSegBench data not available")


def test_background_pool_samples_canvas_sized_images():
    scene = OmniSceneConfig(background="image", bg_source="medseg",
                            bg_datasets=("busi", "cellnuclei", "dca1"), bg_max_images=120)
    bank = BackgroundBank(MED, scene, image_size=64, split="train")
    assert 0 < len(bank._items) <= 120
    rng = np.random.default_rng(0)
    im = bank.sample(rng)
    assert im.shape == (64, 64) and im.dtype == np.float32
    assert 0.0 <= im.min() and im.max() <= 1.0
    # different draws give different backgrounds (pool has variety)
    assert not np.array_equal(im, bank.sample(np.random.default_rng(1)))


def test_budget_spread_across_datasets():
    # bg_max_images is spread per-dataset, so no single dataset fills the whole pool.
    scene = OmniSceneConfig(background="image", bg_source="medseg",
                            bg_datasets=("busi", "cellnuclei", "dca1"), bg_max_images=30)
    bank = BackgroundBank(MED, scene, image_size=64, split="train")
    assert len(bank._items) <= 30
    # per-dataset cap = 30 // 3 = 10, so at most ~10 rows reference any one array object
    from collections import Counter
    c = Counter(id(arr) for arr, _ in bank._items)
    assert max(c.values()) <= 10


if __name__ == "__main__":
    test_background_pool_samples_canvas_sized_images()
    test_budget_spread_across_datasets()
    print("ALL BACKGROUND BANK TESTS PASSED")
