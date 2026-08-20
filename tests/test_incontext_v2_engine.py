# tests/test_incontext_v2_engine.py
import json
import numpy as np
import torch

from src.incontext_dataset_v2 import InContextDataset
from src.providers.totalseg import TotalSegProvider
from src.totalseg_dataloader_incontext import incontext_collate_fn
from src.totalseg_dataset import _ALL_CLASSES_IDX

_CLS, _IDX = next((c, i) for c, i in _ALL_CLASSES_IDX.items() if i > 0)


def _make_tree(root, n_subjects=4, D=48):
    root.mkdir(parents=True, exist_ok=True)
    spac = {}
    for i in range(n_subjects):
        s = root / f"s{i:04d}"; s.mkdir()
        np.save(s / "ct_raw.npy", (np.random.rand(D, D, D) * 200 - 100).astype(np.int16))
        lbl = np.zeros((D, D, D), dtype=np.uint8); lbl[10:30, 10:30, 10:30] = _IDX
        np.save(s / "label.npy", lbl)
        spac[f"s{i:04d}"] = {"spacing": [1.5, 1.5, 1.5], "shape": [D, D, D]}
    json.dump(spac, open(root / "spacings.json", "w"))


def _ds(root, **kw):
    prov = TotalSegProvider(root=root, classes=[_CLS], image_size=(32, 32, 32),
                            crop_jitter=0)
    return InContextDataset(prov, context_size=2, **kw)


def test_engine_item_schema(tmp_path):
    root = tmp_path / "ts"; _make_tree(root)
    ds = _ds(root, eval_seed=0)
    assert len(ds.samples) == 4                       # one (subject, _CLS) each
    it = ds[0]
    assert it["image"].shape == (1, 32, 32, 32)
    assert it["label"].shape == (32, 32, 32)
    assert it["context_in"].shape == (2, 1, 32, 32, 32)
    assert it["context_out"].shape == (2, 32, 32, 32)
    assert len(it["context_subjects"]) == 2
    assert it["label_name"] == _CLS
    assert it["spacing"].shape == (3,)
    assert it["crop_geom"].shape == (4, 3)
    assert int(it["aug_mode"]) == 0
    b = incontext_collate_fn([ds[0], ds[1]])
    assert b["image"].shape == (2, 1, 32, 32, 32)
    assert b["context_in"].shape == (2, 2, 1, 32, 32, 32)


def test_engine_eval_seed_reproducible(tmp_path):
    root = tmp_path / "ts"; _make_tree(root)
    ds = _ds(root, eval_seed=0)
    a, b = ds[1], ds[1]
    assert torch.equal(a["image"], b["image"])
    assert a["context_subjects"] == b["context_subjects"]


def test_engine_spacing_tuple_index(tmp_path):
    root = tmp_path / "ts"; _make_tree(root)
    ds = _ds(root, eval_seed=0)
    it = ds[(0, 3.0)]                                  # (idx, spacing) from SpacingBatchSampler
    assert torch.allclose(it["spacing"], torch.full((3,), 3.0))
