"""CohortSampler.exclude_src: drop masks from named source datasets (e.g. to avoid training-
set leakage when data.source=multisource also uses that same real dataset as a CT/MRI source
alongside synth_gmm). No real gmm_bank needed -- tiny synthetic bank fixture."""
import pickle

import numpy as np

from src.gmm_cohort_sampler import CohortSampler

CLS = 5
DIM = 16


def _make_bank(tmp_path, srcs):
    """One mask per entry in `srcs` (all containing class CLS), tagged with that src."""
    masks_dir = tmp_path / "masks"
    masks_dir.mkdir()
    entries, size_vecs = [], []
    for i, src in enumerate(srcs):
        arr = np.zeros((DIM, DIM, DIM), dtype=np.uint8)
        arr[2:4, 2:4, 2:4] = CLS
        fname = f"m{i:05d}.npy"
        np.save(masks_dir / fname, arr)
        counts = np.bincount(arr.ravel(), minlength=256).astype(np.float64)
        size_vecs.append((counts / counts[1:].sum()).astype(np.float32))
        entries.append({"file": fname, "spacing": [3.0, 3.0, 3.0], "dim": [DIM, DIM, DIM],
                         "label_list": [CLS], "span": (1, 1), "cents": {CLS: [3, 3, 3]},
                         "src": src})
    index = {"maxid": 256, "spacing": 3.0, "entries": entries, "size_mat": np.stack(size_vecs)}
    with open(tmp_path / "index.pkl", "wb") as f:
        pickle.dump(index, f)
    return tmp_path


def test_no_exclude_src_keeps_every_mask(tmp_path):
    bank = _make_bank(tmp_path, ["TotalSegmentatorV2", "TotalSegmentatorV2", "Other", "Other"])
    cs = CohortSampler(bank, k=1)
    assert len(cs.entries) == 4
    assert len(cs.cls2masks[CLS]) == 4


def test_exclude_src_drops_matching_masks_only(tmp_path, capsys):
    bank = _make_bank(tmp_path, ["TotalSegmentatorV2", "TotalSegmentatorV2", "Other", "Other"])
    cs = CohortSampler(bank, k=1, exclude_src=["TotalSegmentatorV2"])
    assert len(cs.entries) == 2
    assert all(e["src"] == "Other" for e in cs.entries)
    assert len(cs.cls2masks[CLS]) == 2
    out = capsys.readouterr().out
    assert "TotalSegmentatorV2" in out and "excluding 2" in out


def test_exclude_src_of_everything_still_needs_min_masks_per_class(tmp_path):
    """Excluding down to fewer than k+1 masks for a class correctly drops that class
    (mirrors min_masks_per_class's existing behavior), rather than crashing."""
    bank = _make_bank(tmp_path, ["TotalSegmentatorV2", "Other"])
    cs = CohortSampler(bank, k=1, exclude_src=["TotalSegmentatorV2"])
    assert CLS not in cs.classes  # only 1 mask left for CLS, need k+1=2
