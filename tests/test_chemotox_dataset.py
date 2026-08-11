import json
import numpy as np
import torch
from src.chemotox_dataset import ChemoToxBCDataset, BC_NAMES


def _make_tree(root, n_subjects=2, D=48):
    root.mkdir(parents=True, exist_ok=True)
    spac = {}
    for i in range(n_subjects):
        s = root / f"subj_{i}"; s.mkdir()
        ct = (np.random.rand(D, D, D).astype(np.float16))
        bc = np.zeros((D, D, D), dtype=np.uint8)
        bc[5:20, 5:20, 5:20] = 1     # muscle
        bc[25:40, 5:20, 5:20] = 2    # sat
        bc[5:20, 25:40, 5:20] = 3    # vat
        bc[25:40, 25:40, 5:20] = 4   # imat
        np.save(s / "ct.npy", ct); np.save(s / "bc.npy", bc)
        spac[f"subj_{i}"] = {"spacing": [1.5, 1.5, 1.5], "shape": [D, D, D]}
    json.dump(spac, open(root / "spacings.json", "w"))


def test_bc_dataset_items(tmp_path):
    root = tmp_path / "chemo"
    _make_tree(root)
    ds = ChemoToxBCDataset(root=root, classes=BC_NAMES, image_size=(32, 32, 32),
                           split="test", context_size=1, use_crop=True,
                           crop_spacing_mm=1.5, eval_seed=0)
    assert len(ds) == 2 * len(BC_NAMES)   # (subject, class) pairs
    item = ds[0]
    assert item["image"].shape == (1, 32, 32, 32)
    assert item["label"].shape == (32, 32, 32)
    assert item["context_in"].shape == (1, 1, 32, 32, 32)
    assert set(torch.unique(item["label"]).tolist()) <= {0, 1}
    assert item["label"].sum() > 0        # foreground present for the cropped tissue
    assert item["label_name"] in BC_NAMES
