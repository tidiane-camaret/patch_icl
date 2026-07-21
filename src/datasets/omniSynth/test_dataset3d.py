import sys; sys.path.insert(0, ".")
import pickle
from pathlib import Path

import numpy as np
import torch

from src.datasets.omniSynth.config import OmniTotalSegConfig
from src.datasets.omniSynth.dataset3d import OmniSynth3DICLDataset


def _fixture_cache(tmp_path, size=16, splits=("train", "val")):
    for split in splits:
        split_dir = tmp_path / f"T{size}" / split
        split_dir.mkdir(parents=True)
        index = {1: "adrenal_gland_left", 3: "aorta", 5: "autochthon_left"}
        for lv, name in index.items():
            tiles = []
            for j in range(3):
                t = np.zeros((2, 6, 6, 6), dtype=np.float16)
                t[0] = 0.3 + 0.1 * j
                t[1, 1:5, 1:5, 1:5] = 1.0
                tiles.append(t)
            (split_dir / f"class_{lv}.pkl").write_bytes(
                pickle.dumps({"name": name, "tiles": tiles}))
        (split_dir / "index.pkl").write_bytes(pickle.dumps(index))
    return tmp_path


def _cfg(root):
    return OmniTotalSegConfig(tiles_root=str(root), size=(16, 16, 16),
                              n_objects=4, k_min=1, k_max=2)


def test_contract_shapes_and_dtypes(tmp_path):
    root = _fixture_cache(tmp_path)
    ds = OmniSynth3DICLDataset(split="train", context_size=3, cfg=_cfg(root))
    item = ds[0]
    assert item["image"].shape == (1, 16, 16, 16) and item["image"].dtype == torch.float32
    assert item["label"].shape == (16, 16, 16) and item["label"].dtype == torch.int64
    assert item["context_in"].shape == (3, 1, 16, 16, 16)
    assert item["context_out"].shape == (3, 16, 16, 16) and item["context_out"].dtype == torch.int64
    assert item["spacing"].shape == (3,)
    assert isinstance(item["subject"], str) and isinstance(item["label_name"], str)
    assert item["label"].max() <= 1                       # binary target


def test_eval_is_deterministic(tmp_path):
    root = _fixture_cache(tmp_path)
    ds = OmniSynth3DICLDataset(split="val", context_size=2, cfg=_cfg(root))
    a, b = ds[0], ds[0]
    assert torch.equal(a["image"], b["image"]) and torch.equal(a["label"], b["label"])
    assert torch.equal(a["context_in"], b["context_in"])


def test_collate_compatible(tmp_path):
    from src.totalseg_dataloader_incontext import incontext_collate_fn
    root = _fixture_cache(tmp_path)
    ds = OmniSynth3DICLDataset(split="train", context_size=2, cfg=_cfg(root))
    batch = incontext_collate_fn([ds[0], ds[1]])
    assert batch["image"].shape == (2, 1, 16, 16, 16)
    assert batch["context_in"].shape == (2, 2, 1, 16, 16, 16)


if __name__ == "__main__":
    import tempfile
    for fn in (test_contract_shapes_and_dtypes, test_eval_is_deterministic,
               test_collate_compatible):
        with tempfile.TemporaryDirectory() as d:
            fn(Path(d))
    print("ALL DATASET3D TESTS PASSED")
