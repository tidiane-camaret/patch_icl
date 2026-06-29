import sys; sys.path.insert(0, ".")
import torch
from omegaconf import OmegaConf
import importlib.util, pathlib
spec = importlib.util.spec_from_file_location(
    "omnisynth_common", str(pathlib.Path("experiments/2d/common.py")))
common = importlib.util.module_from_spec(spec); spec.loader.exec_module(common)
build_dataset = common.build_dataset

CFG = OmegaConf.create({
    "data": {"source": "omnisynth", "context_size": 3, "image_size": 64},
    "paths": {"omniglot": "/home/dpxuser/repos/omniglot/python"},
    "synth": {
        "diversity": {"master_seed": 42, "train_zip": "images_background.zip",
                      "eval_zip": "images_evaluation.zip", "val_test_split": 0.5},
        "scene": {"grid": 4, "k_min": 1, "k_max": 6, "cell_margin": 0.1,
                  "target_mode": "class", "aug_rotate": 15.0, "aug_scale": 0.1,
                  "aug_translate": 0.1},
        "sampling": {"epoch_length": 100, "eval_subjects_per_task": 2,
                     "eval_seed_namespace": 0},
    },
})


def test_build_dataset_omnisynth():
    ds = build_dataset(CFG, "train")
    item = ds[0]
    assert item["image"].shape == (1, 64, 64)
    assert item["context_in"].shape == (3, 1, 64, 64)


if __name__ == "__main__":
    test_build_dataset_omnisynth()
    print("ALL INTEGRATION TESTS PASSED")
