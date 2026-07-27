import sys; sys.path.insert(0, ".")
import numpy as np
import torch

from src.datasets.anchor_synth.dataset3d import AnchorSynth3DICLDataset
from src.totalseg_dataset import _ALL_CLASSES_IDX

SIZE = 16
ANCHORS = ["aorta", "liver", "spleen", "kidney_left"]   # 4 co-occurring landmarks
# non-coplanar blocks (tetrahedron corners) so the frame is well-conditioned
BLOCKS = [(2, 2, 2), (2, 10, 10), (10, 2, 10), (10, 10, 2)]


def _make_root(tmp_path, n=5):
    rows = ["image_id;split"]
    for i in range(n):
        subj = f"s{i:04d}"
        d = tmp_path / subj
        d.mkdir()
        label = np.zeros((SIZE, SIZE, SIZE), dtype=np.uint8)
        for cls, (z, y, x) in zip(ANCHORS, BLOCKS):
            label[z:z + 3, y:y + 3, x:x + 3] = _ALL_CLASSES_IDX[cls]
        ct = (0.3 + 0.01 * i) * np.ones((SIZE, SIZE, SIZE), dtype=np.float16)
        np.save(d / "label.npy", label)
        np.save(d / f"label_{SIZE}x{SIZE}x{SIZE}.npy", label)
        np.save(d / f"ct_{SIZE}x{SIZE}x{SIZE}.npy", ct)
        rows.append(f"{subj};val")
    (tmp_path / "meta.csv").write_text("\n".join(rows) + "\n")
    return tmp_path


def _ds(root, **kw):
    return AnchorSynth3DICLDataset(
        root=root, classes=list(ANCHORS), image_size=(SIZE, SIZE, SIZE),
        split="val", context_size=2, eval_subjects_per_task=2,
        n_anchors=4, object_size_frac_min=0.6, object_size_frac_max=1.2,
        object_size_min_vox=3, contrast_delta=0.3, **kw)


def test_contract_shapes_and_object_drawn(tmp_path):
    ds = _ds(_make_root(tmp_path))
    assert len(ds) == 2 * 5                       # eligible subjects (5) * subjects/task (2)
    item = ds[0]
    assert item["image"].shape == (1, SIZE, SIZE, SIZE)
    assert item["label"].shape == (SIZE, SIZE, SIZE)
    assert item["label"].dtype == torch.int64
    assert item["context_in"].shape == (2, 1, SIZE, SIZE, SIZE)
    assert item["context_out"].shape == (2, SIZE, SIZE, SIZE)
    assert item["label"].sum() > 0                # a blob was drawn
    assert item["label_name"] in ("blob", "elongated", "tubular")
    assert len(item["meta"]["anchors"]) == 4
    assert len(item["meta"]["weights"][0]) == 4


def test_anchor_not_emitted_as_label(tmp_path):
    ds = _ds(_make_root(tmp_path))
    item = ds[0]
    full = np.load(tmp_path / f"{item['subject']}/label_{SIZE}x{SIZE}x{SIZE}.npy")
    anchor_union = np.isin(full, [_ALL_CLASSES_IDX[c] for c in ANCHORS])
    assert not np.array_equal(item["label"].numpy() > 0, anchor_union)
    assert int(item["label"].max()) <= 1          # n_objects=1: bg(0) or object(1)


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
