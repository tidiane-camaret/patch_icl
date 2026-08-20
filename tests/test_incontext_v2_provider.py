# tests/test_incontext_v2_provider.py
import json
import random
import numpy as np
import pytest
import torch

from src.incontext_dataset_v2 import LoadRequest, LoadResult
from src.providers.totalseg import TotalSegProvider
from src.totalseg_dataset import _ALL_CLASSES_IDX

_CLS, _IDX = next((c, i) for c, i in _ALL_CLASSES_IDX.items() if i > 0)


def _make_tree(root, n_subjects=3, D=48, with_raw=True):
    root.mkdir(parents=True, exist_ok=True)
    spac = {}
    for i in range(n_subjects):
        s = root / f"s{i:04d}"; s.mkdir()
        if with_raw:
            np.save(s / "ct_raw.npy", (np.random.rand(D, D, D) * 200 - 100).astype(np.int16))
        lbl = np.zeros((D, D, D), dtype=np.uint8)
        lbl[10:30, 10:30, 10:30] = _IDX                     # a blob of the target class
        np.save(s / "label.npy", lbl)
        spac[f"s{i:04d}"] = {"spacing": [1.5, 1.5, 1.5], "shape": [D, D, D]}
    json.dump(spac, open(root / "spacings.json", "w"))


def test_provider_load_returns_valid_result(tmp_path):
    root = tmp_path / "ts"; _make_tree(root)
    prov = TotalSegProvider(root=root, classes=[_CLS], image_size=(32, 32, 32),
                            crop_spacing_mm=1.5)
    assert prov.classes == [_CLS]
    assert len(prov.subjects_for(_CLS)) == 3
    req = LoadRequest(rng=random.Random(0), crop_spacing_mm=1.5)
    res = prov.load("s0000", _CLS, req)
    assert isinstance(res, LoadResult)
    assert res.image.shape == (1, 32, 32, 32) and res.image.dtype == torch.float32
    assert res.label.shape == (32, 32, 32) and res.label.dtype == torch.int64
    assert set(torch.unique(res.label).tolist()) <= {0, 1}
    assert res.label.sum() > 0
    assert res.spacing.shape == (3,)
    assert torch.allclose(res.spacing, torch.full((3,), 1.5))
    assert res.crop_geom.shape == (4, 3)


def test_provider_hard_fails_without_ct_raw(tmp_path):
    root = tmp_path / "ts"; _make_tree(root, with_raw=False)
    prov = TotalSegProvider(root=root, classes=[_CLS], image_size=(32, 32, 32))
    req = LoadRequest(rng=random.Random(0), crop_spacing_mm=1.5)
    with pytest.raises((FileNotFoundError, AssertionError)):
        prov.load("s0000", _CLS, req)


def test_provider_request_center_overrides_centroid(tmp_path):
    root = tmp_path / "ts"; _make_tree(root)
    prov = TotalSegProvider(root=root, classes=[_CLS], image_size=(32, 32, 32),
                            crop_jitter=0)
    req_c = LoadRequest(rng=random.Random(0), crop_spacing_mm=1.5, center=(0, 0, 0))
    req_d = LoadRequest(rng=random.Random(0), crop_spacing_mm=1.5)   # centroid
    g_corner = prov.load("s0000", _CLS, req_c).crop_geom
    g_default = prov.load("s0000", _CLS, req_d).crop_geom
    assert not torch.equal(g_corner, g_default)                     # center changed the crop
