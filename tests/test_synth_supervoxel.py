"""Unit tests for the self_context.synth_masks "supervoxel" source: a supervoxel group
from the target subject's label_synth_{method} volume, placed on the target grid, with an
ellipsoid fallback when the subject has no usable supervoxel. See
src/totalseg_dataloader_incontext._supervoxel_label_on_grid + the __getitem__ synth branch."""

import numpy as np
import torch

from src.totalseg_dataloader_incontext import (
    TotalSegInContextDataset, _ALL_CLASSES_IDX,
)

T = 16
CLS = next(iter(_ALL_CLASSES_IDX))            # any real TotalSegmentator class name
CLS_IDX = _ALL_CLASSES_IDX[CLS]
SV_ID = 5                                      # single supervoxel id used in the fixtures


def _write_subject(root, subj, *, with_sv):
    """A minimal fast-path subject: native label.npy (scan cache), pre-resized ct/label,
    and (optionally) native + pre-resized supervoxel volumes with one SV block."""
    d = root / subj
    d.mkdir(parents=True)
    size = f"{T}x{T}x{T}"

    lbl = np.zeros((T, T, T), dtype=np.uint8)
    lbl[4:12, 4:12, 4:12] = CLS_IDX           # organ present -> scan cache picks up CLS
    np.save(d / "label.npy", lbl)
    np.save(d / f"label_{size}.npy", lbl)
    np.save(d / f"ct_{size}.npy", np.zeros((T, T, T), dtype=np.float32))  # normalised: all body

    if with_sv:
        sv = np.zeros((T, T, T), dtype=np.uint8)
        sv[6:12, 6:12, 6:12] = SV_ID          # one supervoxel region
        np.save(d / "label_synth_seeds3d.npy", sv)
        np.save(d / f"label_synth_seeds3d_{size}.npy", sv)


def _make_ds(root):
    return TotalSegInContextDataset(
        root=str(root),
        classes=[CLS],
        image_size=(T, T, T),
        split=None,
        context_size=1,
        aug_cfg=None,
        synth_method="seeds3d",
        use_crop=False,
        self_context=1.0,
        self_context_synth={
            "p": 1.0,
            "sources": ["supervoxel"],
            "supervoxel": {"n_merge_min": 1, "n_merge_max": 1},
        },
    )


def _ds(tmp_path):
    _write_subject(tmp_path, "s0000", with_sv=True)
    _write_subject(tmp_path, "s0001", with_sv=False)
    return _make_ds(tmp_path)


def test_supervoxel_label_matches_sv_region(tmp_path):
    """Fast path (crop_geom=None): the label equals the supervoxel occupancy on the grid."""
    ds = _ds(tmp_path)
    label = ds._supervoxel_label_on_grid("s0000", None)
    assert label is not None
    assert label.shape == (T, T, T)
    assert set(label.unique().tolist()) <= {0, 1}
    expected = torch.zeros(T, T, T, dtype=torch.long)
    expected[6:12, 6:12, 6:12] = 1
    assert torch.equal(label, expected)


def test_supervoxel_crop_geom_branch(tmp_path):
    """Crop path: a full-volume crop_geom reproduces the supervoxel region."""
    ds = _ds(tmp_path)
    crop_geom = torch.tensor(
        [[0, 0, 0], [T, T, T], [T, T, T], [0, 0, 0]], dtype=torch.long
    )                                          # starts, crop_sizes, out_sizes, pad_lo
    label = ds._supervoxel_label_on_grid("s0000", crop_geom)
    assert label is not None and label.sum() == 6 ** 3


def test_no_supervoxel_returns_none(tmp_path):
    """A subject without a supervoxel file yields None (caller falls back to an ellipse)."""
    ds = _ds(tmp_path)
    assert ds._supervoxel_label_on_grid("s0001", None) is None


def test_getitem_fallback_never_empty(tmp_path):
    """With sources=[supervoxel], both a SV subject and a no-SV subject (ellipse fallback)
    produce a non-empty synthetic target label, and self-context clones it to the context."""
    ds = _ds(tmp_path)
    seen = set()
    for idx in range(len(ds)):
        item = ds[idx]
        seen.add(item["subject"])
        assert item["label_name"] == "synth"
        assert item["label"].sum() > 0                       # never empty (SV or ellipse)
        assert torch.equal(item["label"], item["context_out"][0])  # self-context clone
    assert {"s0000", "s0001"} <= seen                        # both target subjects exercised
