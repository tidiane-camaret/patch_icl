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


from omegaconf import OmegaConf  # noqa: E402


class _StubModel:
    """Minimal model with .predict: emits a centred cube in the T³ grid (independent of
    input), so the cascade wiring/stitch/metrics can be exercised without a checkpoint."""
    spacing_aware = False

    def predict(self, target_img, context_imgs, context_masks, **kw):
        B, _, T, _, _ = target_img.shape
        out = torch.zeros(B, T, T, T)
        q = T // 4
        out[:, q:T - q, q:T - q, q:T - q] = 1.0
        return out


def _cfg():
    return OmegaConf.create({
        "data": {"image_size": [16, 16, 16], "crop_spacing_mm": 1.5,
                 "use_crop": True, "mask_downsample": "occupancy",
                 "mask_occupancy_thr": 0.1, "source": "totalseg"},
        "eval": {"model": "stub", "checkpoint": None, "spacing_sweep": [4, 1.5]},
    })


def test_predict_nifti_end_to_end(tmp_path, monkeypatch):
    import infer_nifti
    # Bypass the real model builder + drift warning (no checkpoint in the test).
    monkeypatch.setattr(infer_nifti, "_build_model", lambda cfg: _StubModel())
    monkeypatch.setattr(infer_nifti, "_warn_uninherited_data", lambda cfg: None)

    shape = (32, 32, 32)
    ct = np.zeros(shape, dtype=np.int16)
    organ = np.zeros(shape, dtype=np.uint8)
    organ[12:20, 12:20, 12:20] = 1
    aff = np.diag([1.5, 1.5, 1.5, 1.0])
    tgt = tmp_path / "tgt.nii.gz"; nib.save(nib.Nifti1Image(ct, aff), str(tgt))
    cimg = tmp_path / "cimg.nii.gz"; nib.save(nib.Nifti1Image(ct, aff), str(cimg))
    cmsk = tmp_path / "cmsk.nii.gz"; nib.save(nib.Nifti1Image(organ, aff), str(cmsk))
    gt = tmp_path / "gt.nii.gz"; nib.save(nib.Nifti1Image(organ, aff), str(gt))
    out = tmp_path / "pred.nii.gz"

    res = infer_nifti.predict_nifti(
        _cfg(), tgt, [(cimg, cmsk)], gt_path=gt, out_path=out)

    assert res["pred"].shape == shape
    assert res["pred"].dtype == bool
    assert res["pred"].any()                    # stub emits a non-empty cube
    assert 0.0 <= res["dice"] <= 1.0
    assert 0.0 <= res["coarse_only_dice"] <= 1.0
    assert out.exists()
    loaded, _ = load_nifti(out)
    assert loaded.shape == shape


def test_predict_nifti_requires_context(tmp_path, monkeypatch):
    import infer_nifti
    monkeypatch.setattr(infer_nifti, "_build_model", lambda cfg: _StubModel())
    monkeypatch.setattr(infer_nifti, "_warn_uninherited_data", lambda cfg: None)
    ct = np.zeros((8, 8, 8), dtype=np.int16)
    aff = np.eye(4)
    tgt = tmp_path / "t.nii.gz"; nib.save(nib.Nifti1Image(ct, aff), str(tgt))
    import pytest
    with pytest.raises(ValueError, match="context"):
        infer_nifti.predict_nifti(_cfg(), tgt, [])
