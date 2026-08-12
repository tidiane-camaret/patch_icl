"""Tests for the nifti in-context cascade inference module."""
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # sibling infer_nifti / evaluate

from infer_nifti import (  # noqa: E402
    load_nifti, voxel_spacing, mask_centroid, prep_context, prep_target,
)


def _write_nifti(tmp_path, name, arr, spacing=(1.0, 1.0, 1.0)):
    aff = np.diag([*spacing, 1.0])
    p = tmp_path / name
    nib.save(nib.Nifti1Image(arr, aff), str(p))
    return p


def test_load_and_spacing(tmp_path):
    arr = np.arange(4 * 5 * 6, dtype=np.int16).reshape(4, 5, 6)
    p = _write_nifti(tmp_path, "ct.nii.gz", arr, spacing=(2.0, 1.5, 1.0))
    got, aff = load_nifti(p)
    assert got.shape == (4, 5, 6)
    assert voxel_spacing(aff) == [2.0, 1.5, 1.0]


def test_mask_centroid_and_empty():
    m = np.zeros((10, 10, 10), dtype=bool)
    m[4:6, 4:6, 4:6] = True
    assert mask_centroid(m) == (4, 4, 4)      # COM of the cube (floored)
    # empty -> volume centre (with a warning)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert mask_centroid(np.zeros((8, 8, 8), bool)) == (4, 4, 4)


def test_prep_target_shapes():
    ct = np.zeros((40, 40, 40), dtype=np.float32)
    img_t, geom = prep_target(ct, [1.0, 1.0, 1.0], (20, 20, 20),
                              T=8, crop_mm=1.0)
    assert img_t.shape == (1, 8, 8, 8)
    assert geom.shape == (4, 3)


def test_prep_context_shapes():
    ct = np.zeros((40, 40, 40), dtype=np.float32)
    mask = np.zeros((40, 40, 40), dtype=bool)
    mask[18:22, 18:22, 18:22] = True
    img_t, mask_t = prep_context(ct, mask, [1.0, 1.0, 1.0], (20, 20, 20),
                                 T=8, crop_mm=1.0, mask_downsample="occupancy", occ_thr=0.1)
    assert img_t.shape == (1, 8, 8, 8)
    assert mask_t.shape == (8, 8, 8)
    assert mask_t.sum() > 0                    # organ survives into the crop
