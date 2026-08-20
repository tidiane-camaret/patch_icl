# tests/test_incontext_v2_wiring.py
import json
import numpy as np
from omegaconf import OmegaConf

from src.incontext_dataset_v2 import InContextDataset
from src.totalseg_dataloader_incontext import TotalSegInContextDataset
from src.totalseg_dataset import _ALL_CLASSES_IDX

import sys, pathlib
sys.path.insert(0, str(pathlib.Path("experiments/3d").resolve()))
from common import build_dataset  # noqa: E402

_CLS, _IDX = next((c, i) for c, i in _ALL_CLASSES_IDX.items() if i > 0)


def _make_tree(root, n=3, D=48):
    root.mkdir(parents=True, exist_ok=True)
    spac = {}
    for i in range(n):
        s = root / f"s{i:04d}"; s.mkdir()
        np.save(s / "ct_raw.npy", (np.random.rand(D, D, D) * 200 - 100).astype(np.int16))
        lbl = np.zeros((D, D, D), dtype=np.uint8); lbl[10:30, 10:30, 10:30] = _IDX
        np.save(s / "label.npy", lbl)
        spac[f"s{i:04d}"] = {"spacing": [1.5, 1.5, 1.5], "shape": [D, D, D]}
    json.dump(spac, open(root / "spacings.json", "w"))
    with open(root / "meta.csv", "w") as f:
        f.write("image_id;split\n")
        for i in range(n):
            f.write(f"s{i:04d};val\n")


def _cfg(root, loader_v2):
    return OmegaConf.create({
        "data": {"source": "totalseg", "image_size": [32, 32, 32], "context_size": 2,
                 "val_classes": [_CLS], "train_classes": [_CLS], "use_crop": True,
                 "crop_spacing_mm": 1.5, "class_balanced": False,
                 "max_val_subjects": None, "max_train_subjects": None,
                 "loader_v2": loader_v2,
                 "synth_unions": False, "synth_method": None, "p_synth": 0.0},
        "paths": {"totalseg": str(root)},
        "eval": {"seed": 0},
    })


def test_build_dataset_v2_flag(tmp_path):
    root = tmp_path / "ts"; _make_tree(root)
    ds = build_dataset(_cfg(root, True), "val")
    assert isinstance(ds, InContextDataset)
    it = ds[0]
    assert it["image"].shape == (1, 32, 32, 32)


def test_build_dataset_default_is_v1(tmp_path):
    root = tmp_path / "ts"; _make_tree(root)
    # v1 needs pre-resized files; just assert the TYPE routing, not a full load
    ds = build_dataset(_cfg(root, False), "val")
    assert isinstance(ds, TotalSegInContextDataset)
