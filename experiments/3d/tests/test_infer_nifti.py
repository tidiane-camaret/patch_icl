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


def test_load_nifti_canonicalizes():
    """load_nifti must reorient to closest-canonical (RAS) regardless of stored orientation."""
    import tempfile, os
    # Build a non-canonical affine: flip the first axis (L instead of R).
    non_canon_aff = np.diag([-1.0, 1.0, 1.0, 1.0])
    arr = np.arange(4 * 5 * 6, dtype=np.int16).reshape(4, 5, 6)
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "noncanon.nii.gz"
        nib.save(nib.Nifti1Image(arr, non_canon_aff), str(p))
        got_arr, got_aff = load_nifti(p)
    # The returned affine must be closest-canonical.
    canon_img = nib.as_closest_canonical(nib.Nifti1Image(arr, non_canon_aff))
    assert np.allclose(got_aff, canon_img.affine), (
        f"affine not canonical: axcodes={nib.aff2axcodes(got_aff)}")
    # Axis codes must be ('R','A','S') or equivalent positive-canonical.
    axcodes = nib.aff2axcodes(got_aff)
    assert axcodes == ('R', 'A', 'S'), f"expected RAS axcodes, got {axcodes}"
    assert got_arr.shape == canon_img.shape


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


def test_predict_nifti_output_matches_target_orientation(tmp_path, monkeypatch):
    """Output mask must be saved on the TARGET's on-disk grid, not the RAS-canonical grid.

    Target is stored LAS (first axis flipped) while the context is RAS; the model runs in
    canonical space but the written/returned mask must be reoriented back to the target's
    orientation + affine so it overlays the input CT voxel-for-voxel."""
    import infer_nifti
    monkeypatch.setattr(infer_nifti, "_build_model", lambda cfg: _StubModel())
    monkeypatch.setattr(infer_nifti, "_warn_uninherited_data", lambda cfg: None)

    shape = (32, 32, 32)
    ct = np.zeros(shape, dtype=np.int16)
    organ = np.zeros(shape, dtype=np.uint8); organ[12:20, 12:20, 12:20] = 1
    las_aff = np.diag([-1.5, 1.5, 1.5, 1.0])   # non-canonical target (L instead of R)
    ras_aff = np.diag([1.5, 1.5, 1.5, 1.0])
    tgt = tmp_path / "tgt.nii.gz"; nib.save(nib.Nifti1Image(ct, las_aff), str(tgt))
    cimg = tmp_path / "cimg.nii.gz"; nib.save(nib.Nifti1Image(ct, ras_aff), str(cimg))
    cmsk = tmp_path / "cmsk.nii.gz"; nib.save(nib.Nifti1Image(organ, ras_aff), str(cmsk))
    out = tmp_path / "pred.nii.gz"

    res = infer_nifti.predict_nifti(_cfg(), tgt, [(cimg, cmsk)], out_path=out)

    # Returned affine + saved affine must be the target's original (LAS), not canonical RAS.
    assert np.allclose(res["affine"], las_aff)
    saved = nib.load(str(out))
    assert np.allclose(saved.affine, las_aff)
    assert nib.aff2axcodes(saved.affine) == ('L', 'A', 'S')
    # Re-canonicalising the saved mask reproduces the canonical prediction the model made:
    # the round-trip preserved content, only the grid orientation changed.
    recanon = np.asanyarray(nib.as_closest_canonical(saved).dataobj) > 0
    assert recanon.shape == shape and recanon.any()


class _EchoStubModel:
    """Model whose prediction echoes the (single) context mask in the grid, so different
    labels yield different-shaped/placed predictions — lets us check the batch dim pairs
    each target with the RIGHT context (no cross-task contamination) and exercises the
    small-organ-wins overlap combine."""
    spacing_aware = False

    def predict(self, target_img, context_imgs, context_masks, **kw):
        return (context_masks[:, 0] > 0.5).float()          # (B,T,T,T)


def _combine_small_wins(natives_by_label, shape):
    """Reference stitch: id-valued volume, larger organs written first so smaller win."""
    out = np.zeros(shape, np.uint8)
    for lab in sorted(natives_by_label, key=lambda L: int(natives_by_label[L].sum()),
                      reverse=True):
        out[natives_by_label[lab]] = lab
    return out


def test_predict_nifti_multilabel_matches_per_label_loop(tmp_path, monkeypatch):
    """Multi-label batched inference == looping single-label predict_nifti per id, then
    combining with the documented small-organ-wins overlap rule."""
    import infer_nifti
    monkeypatch.setattr(infer_nifti, "_build_model", lambda cfg: _EchoStubModel())
    monkeypatch.setattr(infer_nifti, "_warn_uninherited_data", lambda cfg: None)

    shape = (32, 32, 32)
    ct = np.zeros(shape, dtype=np.int16)
    ml = np.zeros(shape, dtype=np.uint8)
    ml[6:22, 6:22, 6:22] = 1                    # large organ
    ml[14:18, 14:18, 14:18] = 2                 # small organ (carved inside label 1)
    aff = np.diag([1.5, 1.5, 1.5, 1.0])
    tgt = tmp_path / "tgt.nii.gz"; nib.save(nib.Nifti1Image(ct, aff), str(tgt))
    cimg = tmp_path / "cimg.nii.gz"; nib.save(nib.Nifti1Image(ct, aff), str(cimg))
    cmsk = tmp_path / "cmsk.nii.gz"; nib.save(nib.Nifti1Image(ml, aff), str(cmsk))

    # per-label single-organ runs (binary context == id) as the reference
    ref = {}
    for lab in (1, 2):
        one = tmp_path / f"one_{lab}.nii.gz"
        nib.save(nib.Nifti1Image((ml == lab).astype(np.uint8), aff), str(one))
        r = infer_nifti.predict_nifti(_cfg(), tgt, [(cimg, one)])
        ref[lab] = r["pred"] > 0

    out = tmp_path / "ml_pred.nii.gz"
    res = infer_nifti.predict_nifti(_cfg(), tgt, [(cimg, cmsk)],
                                    label_ids=[1, 2], batch_size=8, out_path=out)

    assert res["labels"] == [1, 2]
    assert res["pred"].dtype == np.uint8
    assert set(np.unique(res["pred"]).tolist()) <= {0, 1, 2}
    assert (res["pred"] == 1).any() and (res["pred"] == 2).any()
    # batched multi-label output must equal the per-label loop combined small-wins.
    assert np.array_equal(res["pred"], _combine_small_wins(ref, shape))
    assert out.exists()


def test_predict_nifti_multilabel_all_and_dice(tmp_path, monkeypatch):
    """--labels 'all' resolves to the context's non-zero ids; gt gives per-label + macro Dice."""
    import infer_nifti
    monkeypatch.setattr(infer_nifti, "_build_model", lambda cfg: _EchoStubModel())
    monkeypatch.setattr(infer_nifti, "_warn_uninherited_data", lambda cfg: None)

    shape = (32, 32, 32)
    ct = np.zeros(shape, dtype=np.int16)
    ml = np.zeros(shape, dtype=np.uint8)
    ml[6:14, 6:14, 6:14] = 1
    ml[20:26, 20:26, 20:26] = 3                 # note: non-contiguous ids
    aff = np.diag([1.5, 1.5, 1.5, 1.0])
    tgt = tmp_path / "tgt.nii.gz"; nib.save(nib.Nifti1Image(ct, aff), str(tgt))
    cimg = tmp_path / "cimg.nii.gz"; nib.save(nib.Nifti1Image(ct, aff), str(cimg))
    cmsk = tmp_path / "cmsk.nii.gz"; nib.save(nib.Nifti1Image(ml, aff), str(cmsk))
    gt = tmp_path / "gt.nii.gz"; nib.save(nib.Nifti1Image(ml, aff), str(gt))

    res = infer_nifti.predict_nifti(_cfg(), tgt, [(cimg, cmsk)],
                                    label_ids="all", gt_path=gt)
    assert res["labels"] == [1, 3]
    assert isinstance(res["dice"], dict) and set(res["dice"]) == {1, 3}
    for lab in (1, 3):
        assert 0.0 <= res["dice"][lab] <= 1.0
        assert 0.0 <= res["coarse_only_dice"][lab] <= 1.0
    assert 0.0 <= res["macro_dice"] <= 1.0


def _caret_ext(names):
    """Nifti1Extension (code 0) holding a Caret LabelTable for {id: name}, mirroring the
    real TotalSegmentator format (xml declaration, CDATA, VolumeType=Label atlas marker)."""
    body = "".join(
        f'\n<Label Key="{k}" Red="1.0" Green="0.0" Blue="0.0" Alpha="1">'
        f'<![CDATA[{nm}]]></Label>' for k, nm in names.items())
    xml = ('<?xml version="1.0" encoding="UTF-8"?> <CaretExtension>  '
           '<Date><![CDATA[2013]]></Date>   <VolumeInformation Index="0">   '
           f'<LabelTable>{body}\n  </LabelTable>  <VolumeType><![CDATA[Label]]></VolumeType>'
           '   </VolumeInformation></CaretExtension>')
    return nib.nifti1.Nifti1Extension(0, xml.encode("utf-8"))


def _write_labeled_nifti(path, arr, aff, names):
    img = nib.Nifti1Image(arr, aff)
    img.header.extensions.append(_caret_ext(names))
    nib.save(img, str(path))


def _out_label_names(path):
    """{id: name} parsed back from a saved prediction's Caret LabelTable extension."""
    import re
    exts = nib.load(str(path)).header.extensions
    out = {}
    for e in exts:
        c = e.get_content()
        if isinstance(c, (bytes, bytearray)) and b"<CaretExtension" in c:
            for k, nm in re.findall(rb'<Label Key="(\d+)"[^>]*>\s*(?:<!\[CDATA\[)?'
                                    rb'([^<]*?)(?:\]\]>)?</Label>', c):
                out[int(k)] = nm.decode().strip()
    return out


def _ml_case(tmp_path, monkeypatch):
    """Shared multi-label fixture: returns (cfg-runner args) tgt, cimg, ml paths."""
    import infer_nifti
    monkeypatch.setattr(infer_nifti, "_build_model", lambda cfg: _EchoStubModel())
    monkeypatch.setattr(infer_nifti, "_warn_uninherited_data", lambda cfg: None)
    shape = (32, 32, 32)
    ct = np.zeros(shape, dtype=np.int16)
    ml = np.zeros(shape, dtype=np.uint8)
    ml[6:14, 6:14, 6:14] = 1
    ml[20:26, 20:26, 20:26] = 2
    aff = np.diag([1.5, 1.5, 1.5, 1.0])
    tgt = tmp_path / "tgt.nii.gz"; nib.save(nib.Nifti1Image(ct, aff), str(tgt))
    cimg = tmp_path / "cimg.nii.gz"; nib.save(nib.Nifti1Image(ct, aff), str(cimg))
    return infer_nifti, tgt, cimg, ml, aff


def test_multilabel_output_carries_subset_label_table(tmp_path, monkeypatch):
    """Segmenting a subset of ids -> output LabelTable holds ONLY those ids, with names."""
    infer_nifti, tgt, cimg, ml, aff = _ml_case(tmp_path, monkeypatch)
    cmsk = tmp_path / "cmsk.nii.gz"
    _write_labeled_nifti(cmsk, ml, aff, {1: "spleen", 2: "kidney_right", 3: "liver"})
    out = tmp_path / "pred.nii.gz"
    res = infer_nifti.predict_nifti(_cfg(), tgt, [(cimg, cmsk)],
                                    label_ids=[1], out_path=out)      # only id 1
    assert res["label_names"] == {1: "spleen"}
    got = _out_label_names(out)
    assert got.get(1) == "spleen" and 2 not in got and 3 not in got   # subset to segmented ids


def test_multilabel_label_table_preserves_atlas_structure(tmp_path, monkeypatch):
    """Subsetting must keep the atlas markers a viewer needs (xml declaration, CDATA,
    VolumeType=Label) byte-intact — only the dropped label lines are removed."""
    infer_nifti, tgt, cimg, ml, aff = _ml_case(tmp_path, monkeypatch)
    cmsk = tmp_path / "cmsk.nii.gz"
    _write_labeled_nifti(cmsk, ml, aff, {1: "spleen", 2: "kidney_right", 3: "liver"})
    out = tmp_path / "pred.nii.gz"
    infer_nifti.predict_nifti(_cfg(), tgt, [(cimg, cmsk)], label_ids=[1], out_path=out)
    content = infer_nifti._read_caret_label_table(out)
    assert content.lstrip().startswith(b"<?xml")               # declaration preserved
    assert b"<VolumeType><![CDATA[Label]]></VolumeType>" in content  # atlas marker intact
    assert b"<![CDATA[spleen]]>" in content                    # kept label keeps its CDATA
    assert b'Key="2"' not in content and b'Key="3"' not in content   # dropped labels gone


def test_multilabel_label_table_prefers_gt(tmp_path, monkeypatch):
    """When --gt has its own LabelTable, its names/colors win over the context's."""
    infer_nifti, tgt, cimg, ml, aff = _ml_case(tmp_path, monkeypatch)
    cmsk = tmp_path / "cmsk.nii.gz"
    _write_labeled_nifti(cmsk, ml, aff, {1: "ctx_spleen", 2: "ctx_kidney"})
    gt = tmp_path / "gt.nii.gz"
    _write_labeled_nifti(gt, ml, aff, {1: "gt_spleen", 2: "gt_kidney"})
    out = tmp_path / "pred.nii.gz"
    res = infer_nifti.predict_nifti(_cfg(), tgt, [(cimg, cmsk)],
                                    label_ids=[1, 2], gt_path=gt, out_path=out)
    assert res["label_names"] == {1: "gt_spleen", 2: "gt_kidney"}


def test_multilabel_label_table_falls_back_to_context(tmp_path, monkeypatch):
    """No GT table (gt lacks one) -> context mask's names are used."""
    infer_nifti, tgt, cimg, ml, aff = _ml_case(tmp_path, monkeypatch)
    cmsk = tmp_path / "cmsk.nii.gz"
    _write_labeled_nifti(cmsk, ml, aff, {1: "ctx_spleen", 2: "ctx_kidney"})
    gt = tmp_path / "gt.nii.gz"; nib.save(nib.Nifti1Image(ml, aff), str(gt))  # no LabelTable
    out = tmp_path / "pred.nii.gz"
    res = infer_nifti.predict_nifti(_cfg(), tgt, [(cimg, cmsk)],
                                    label_ids=[1, 2], gt_path=gt, out_path=out)
    assert res["label_names"] == {1: "ctx_spleen", 2: "ctx_kidney"}


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
