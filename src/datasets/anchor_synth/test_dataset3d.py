import sys; sys.path.insert(0, ".")
import numpy as np
import torch

from src.datasets.anchor_synth.dataset3d import AnchorSynth3DICLDataset
from src.totalseg_dataset import _ALL_CLASSES_IDX

SIZE = 16
ANCHOR = "aorta"


def _make_root(tmp_path, n=5):
    """Fake TotalSeg root: each subject has native label.npy + pre-resized
    ct_/label_ npy (native == resized at SIZE), all containing the anchor organ."""
    idx = _ALL_CLASSES_IDX[ANCHOR]
    rows = ["image_id;split"]
    for i in range(n):
        subj = f"s{i:04d}"
        d = tmp_path / subj
        d.mkdir()
        label = np.zeros((SIZE, SIZE, SIZE), dtype=np.uint8)
        # anchor block, position jittered per subject so it is a real landmark
        z0 = 3 + (i % 3)
        label[z0:z0 + 6, 5:11, 6:10] = idx
        ct = (0.3 + 0.01 * i) * np.ones((SIZE, SIZE, SIZE), dtype=np.float16)
        np.save(d / "label.npy", label)
        np.save(d / f"label_{SIZE}x{SIZE}x{SIZE}.npy", label)
        np.save(d / f"ct_{SIZE}x{SIZE}x{SIZE}.npy", ct)
        rows.append(f"{subj};val")
    (tmp_path / "meta.csv").write_text("\n".join(rows) + "\n")
    return tmp_path


def _ds(root, **kw):
    return AnchorSynth3DICLDataset(
        root=root, classes=[ANCHOR], image_size=(SIZE, SIZE, SIZE),
        split="val", context_size=2, eval_subjects_per_task=2,
        offset_range=0.2, scale_frac=0.4, contrast_delta=0.3, **kw)


def test_contract_shapes_and_object_drawn(tmp_path):
    ds = _ds(_make_root(tmp_path))
    assert len(ds) == 2                              # 1 anchor class * 2 subjects/task
    item = ds[0]
    assert item["image"].shape == (1, SIZE, SIZE, SIZE)
    assert item["label"].shape == (SIZE, SIZE, SIZE)
    assert item["label"].dtype == torch.int64
    assert item["context_in"].shape == (2, 1, SIZE, SIZE, SIZE)
    assert item["context_out"].shape == (2, SIZE, SIZE, SIZE)
    assert item["label"].sum() > 0                   # a blob was drawn
    assert item["label_name"] == ANCHOR


def test_anchor_not_emitted_as_label(tmp_path):
    ds = _ds(_make_root(tmp_path))
    item = ds[0]
    idx = _ALL_CLASSES_IDX[ANCHOR]
    anchor_mask = (np.load(tmp_path / f"{item['subject']}/label_{SIZE}x{SIZE}x{SIZE}.npy") == idx)
    # the label is the drawn object, not the anchor organ
    assert not np.array_equal(item["label"].numpy() > 0, anchor_mask)
    assert int(item["label"].max()) <= 1        # n_objects=1 here: only bg(0) or object(1)


def test_deterministic_across_instances(tmp_path):
    root = _make_root(tmp_path)
    a = _ds(root)[0]
    b = _ds(root)[0]
    assert torch.equal(a["label"], b["label"])
    assert torch.equal(a["image"], b["image"])


def test_organ_source_not_implemented(tmp_path):
    import pytest
    with pytest.raises(NotImplementedError):
        _ds(_make_root(tmp_path), object_source="organ")
