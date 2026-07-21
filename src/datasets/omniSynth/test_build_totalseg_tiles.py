import sys; sys.path.insert(0, ".")
import pickle
from pathlib import Path

import numpy as np
import pytest

from scripts.synth3d.build_totalseg_tiles import (
    subjects_for_split, build_tiles_for_split)
from data.totalseg_classes import ALL_CLASSES


def _make_fake_root(tmp_path, size=(16, 16, 16)):
    """Two subjects, each a pre-resized ct/label cube with two organs (label values
    1 and 3). meta.csv puts both in the train split."""
    D, H, W = size
    lv_a = 1                                 # ALL_CLASSES[0]
    lv_b = 3                                 # ALL_CLASSES[2]
    for i, subj in enumerate(("s0000", "s0001")):
        d = tmp_path / subj
        d.mkdir()
        lab = np.zeros(size, dtype=np.uint8)
        lab[2:7, 2:7, 2:7] = lv_a
        lab[9:13, 9:13, 9:13] = lv_b
        ct = (np.random.default_rng(i).random(size) * 255).astype(np.float16)
        np.save(d / f"label_{D}x{H}x{W}.npy", lab)
        np.save(d / f"ct_{D}x{H}x{W}.npy", ct)
    (tmp_path / "meta.csv").write_text(
        "image_id;split\ns0000;train\ns0001;train\n", encoding="utf-8")
    return lv_a, lv_b


def test_subjects_for_split(tmp_path):
    _make_fake_root(tmp_path)
    assert subjects_for_split(tmp_path, "train") == ["s0000", "s0001"]
    assert subjects_for_split(tmp_path, "val") == []


def test_build_writes_index_and_class_files(tmp_path):
    lv_a, lv_b = _make_fake_root(tmp_path)
    out = tmp_path / "tiles"
    split_dir = build_tiles_for_split(tmp_path, out, (16, 16, 16), "train",
                                      max_renditions=10, min_vox=4)
    index = pickle.loads((split_dir / "index.pkl").read_bytes())
    assert set(index) == {lv_a, lv_b}
    assert index[lv_a] == ALL_CLASSES[lv_a - 1]
    data = pickle.loads((split_dir / f"class_{lv_a}.pkl").read_bytes())
    assert data["name"] == ALL_CLASSES[lv_a - 1]
    assert len(data["tiles"]) == 2                       # one rendition per subject
    t = data["tiles"][0]
    assert t.shape[0] == 2 and t.dtype == np.float16
    assert set(np.unique(t[1].astype(np.float32))).issubset({0.0, 1.0})


def test_non_cubic_size_raises(tmp_path):
    (tmp_path / "meta.csv").write_text("image_id;split\n", encoding="utf-8")
    with pytest.raises(ValueError):
        build_tiles_for_split(tmp_path, tmp_path / "o", (32, 64, 64), "train")


if __name__ == "__main__":
    import tempfile
    for fn in (test_subjects_for_split, test_build_writes_index_and_class_files):
        with tempfile.TemporaryDirectory() as d:
            fn(Path(d))
    print("ALL BUILD TESTS PASSED")
