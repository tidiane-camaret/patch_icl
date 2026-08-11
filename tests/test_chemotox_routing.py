import json
import numpy as np
from omegaconf import OmegaConf
from experiments_common_shim import build_dataset  # see Step 3 note
from src.chemotox_dataset import ChemoToxBCDataset, BC_NAMES
from src.totalseg_dataloader_incontext import TotalSegInContextDataset


def _make_bc_tree(root, D=48):
    root.mkdir(parents=True, exist_ok=True); spac = {}
    for i in range(2):
        s = root / f"subj_{i}"; s.mkdir()
        np.save(s / "ct.npy", np.random.rand(D, D, D).astype(np.float16))
        bc = np.zeros((D, D, D), np.uint8)
        bc[5:20, 5:20, 5:20] = 1; bc[25:40, 25:40, 5:20] = 2
        bc[5:20, 25:40, 5:20] = 3; bc[25:40, 5:20, 5:20] = 4
        np.save(s / "bc.npy", bc)
        spac[f"subj_{i}"] = {"spacing": [1.5, 1.5, 1.5], "shape": [D, D, D]}
    json.dump(spac, open(root / "spacings.json", "w"))


def _cfg(source, root):
    return OmegaConf.create({
        "data": {"source": source, "image_size": [32, 32, 32], "context_size": 1,
                 "use_crop": True, "crop_spacing_mm": 1.5, "val_classes": "benchmark",
                 "train_classes": "benchmark", "max_val_subjects": None,
                 "max_train_subjects": None},
        "paths": {"chemotox": str(root)},
        "eval": {"seed": 0, "crop_jitter": 0},
    })


def test_build_dataset_chemotox_bc(tmp_path):
    root = tmp_path / "chemo"; _make_bc_tree(root)
    ds = build_dataset(_cfg("chemotox_bc", root), "test")
    assert isinstance(ds, ChemoToxBCDataset)
    assert len(ds) == 2 * len(BC_NAMES)
