# tests/test_convert_chemotox.py
import json
import numpy as np
import nibabel as nib
import pytest
from scripts.convert_to_npy import load_raw
from data.totalseg_classes import ALL_CLASSES

_CLASS_TO_IDX = {c: i + 1 for i, c in enumerate(ALL_CLASSES)}


def _write_nii(path, arr, spacing):
    aff = np.diag(list(spacing) + [1.0])
    nib.save(nib.Nifti1Image(arr, aff), str(path))


def test_load_raw_chemotox_remaps_and_takes_bc_channel0(tmp_path):
    D = (6, 6, 4)
    img = (np.random.rand(*D) * 100).astype(np.float32)
    ts = np.zeros(D, dtype=np.int16); ts[0, 0, 0] = 5   # TS liver
    bc = np.zeros(D + (2,), dtype=np.int16)
    bc[..., 0][1, 1, 1] = 1                             # muscle (channel 0)
    bc[..., 1][1, 1, 1] = 9999                          # instance id (channel 1, ignored)
    _write_nii(tmp_path / "img.nii", img, (1.5, 1.5, 3.0))
    _write_nii(tmp_path / "ts.nii", ts, (1.5, 1.5, 3.0))
    _write_nii(tmp_path / "bc.nii", bc, (1.5, 1.5, 3.0))

    task = {"source": "chemotox",
            "inputs": {"img": str(tmp_path / "img.nii"),
                       "totalseg": str(tmp_path / "ts.nii"),
                       "bclabels": str(tmp_path / "bc.nii")}}
    img_nib, sp, labels = load_raw(task)

    # load_raw now returns nibabel images (each label carries its own affine) so the
    # converter can resample labels onto the CT grid via world coordinates.
    assert img_nib.shape == D
    assert sp == pytest.approx([1.5, 1.5, 3.0])
    assert set(labels) == {"label", "bc"}
    lab = np.asanyarray(labels["label"].dataobj)
    bc = np.asanyarray(labels["bc"].dataobj)
    assert lab[0, 0, 0] == _CLASS_TO_IDX["liver"]
    assert bc[1, 1, 1] == 1
    assert bc.ndim == 3 and bc.max() == 1   # channel 0 only; channel 1's 9999 must not leak in
