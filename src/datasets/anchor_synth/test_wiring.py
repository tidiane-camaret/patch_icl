import sys; sys.path.insert(0, ".")
sys.path.insert(0, "experiments/3d")
import numpy as np
from omegaconf import OmegaConf

from src.totalseg_dataset import _ALL_CLASSES_IDX

SIZE = 16


def _make_root(tmp_path, n=4):
    # place 4 non-coplanar anchor blocks
    rows = ["image_id;split"]
    for i in range(n):
        subj = f"s{i:04d}"
        d = tmp_path / subj
        d.mkdir()
        label = np.zeros((SIZE, SIZE, SIZE), dtype=np.uint8)
        for cls, (z, y, x) in zip(["aorta", "liver", "spleen", "kidney_left"],
                                   [(2, 2, 2), (2, 10, 10), (10, 2, 10), (10, 10, 2)]):
            label[z:z + 3, y:y + 3, x:x + 3] = _ALL_CLASSES_IDX[cls]
        ct = 0.3 * np.ones((SIZE, SIZE, SIZE), dtype=np.float16)
        np.save(d / "label.npy", label)
        np.save(d / f"label_{SIZE}x{SIZE}x{SIZE}.npy", label)
        np.save(d / f"ct_{SIZE}x{SIZE}x{SIZE}.npy", ct)
        rows.append(f"{subj};train")
    (tmp_path / "meta.csv").write_text("\n".join(rows) + "\n")
    return tmp_path


def test_build_dataset_dispatches_anchor_synth3d(tmp_path):
    from common import build_dataset
    from src.datasets.anchor_synth.dataset3d import AnchorSynth3DICLDataset

    root = str(_make_root(tmp_path))
    cfg = OmegaConf.create({
        "paths": {"totalseg": root},
        "data": {"source": "anchor_synth3d", "image_size": [SIZE, SIZE, SIZE],
                 "context_size": 1, "max_train_subjects": None},
        "anchor_synth": {"object_source": "blob", "shape": "blob", "n_objects": 1,
                         "anchor_classes": ["aorta", "liver", "spleen", "kidney_left"],
                         "n_anchors": 4, "extrapolation": 0.3,
                         "weight_concentration": 1.0, "max_select_tries": 20,
                         "object_size_frac_min": 0.6, "object_size_frac_max": 1.2,
                         "object_size_min_vox": 3, "scale_jitter": 0.15,
                         "rotate_jitter": 12.0, "contrast_delta": 0.3,
                         "edge_blur": 0.08, "boundary_complexity": 0.0,
                         "eval_subjects_per_task": 2, "eval_seed_namespace": 0,
                         "epoch_length": 5},
    })
    ds = build_dataset(cfg, "train")
    assert isinstance(ds, AnchorSynth3DICLDataset)
    assert len(ds) == 5                              # epoch_length (train)
    item = ds[0]
    assert item["image"].shape == (1, SIZE, SIZE, SIZE)
    assert item["label"].sum() > 0
