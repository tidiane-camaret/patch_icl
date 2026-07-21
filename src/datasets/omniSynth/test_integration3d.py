import sys; sys.path.insert(0, ".")
import pickle
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from experiments.threed_common_shim import build_dataset  # see Step 3 note


def _fixture_cache(tmp_path, size=16):
    split_dir = tmp_path / f"T{size}" / "train"
    split_dir.mkdir(parents=True)
    index = {1: "adrenal_gland_left", 3: "aorta"}
    for lv, name in index.items():
        tiles = [np.pad(np.ones((2, 4, 4, 4), dtype=np.float16), 0) for _ in range(2)]
        (split_dir / f"class_{lv}.pkl").write_bytes(
            pickle.dumps({"name": name, "tiles": tiles}))
    (split_dir / "index.pkl").write_bytes(pickle.dumps(index))
    return tmp_path


def test_build_dataset_dispatches_omnisynth3d(tmp_path):
    root = _fixture_cache(tmp_path)
    cfg = OmegaConf.create({
        "data": {"source": "omnisynth3d", "context_size": 2, "image_size": [16, 16, 16]},
        "paths": {"totalseg": str(root)},
        "synth3d": {"tiles_root": str(root), "size": [16, 16, 16], "n_objects": 3,
                    "k_min": 1, "k_max": 1, "target_mode": "class"},
    })
    ds = build_dataset(cfg, "train")
    item = ds[0]
    assert item["image"].shape == (1, 16, 16, 16)
    assert item["label"].shape == (16, 16, 16)


if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        test_build_dataset_dispatches_omnisynth3d(Path(d))
    print("INTEGRATION3D TEST PASSED")
