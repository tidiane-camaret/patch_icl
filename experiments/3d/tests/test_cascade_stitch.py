"""Task 8: _stitched_native_dice_multi — coarse->fine composite, each overwriting the previous."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

evaluate = pytest.importorskip("evaluate")


def _pg_entry(pred_native_bool, geom):
    return {("s0", "liver"): (np.packbits(pred_native_bool.astype(bool)),
                              tuple(pred_native_bool.shape), np.asarray(geom))}


def test_finer_level_overwrites_coarser(tmp_path, monkeypatch):
    # Native volume 1 subject, 1 class. Coarse pred is an offset (wrong) box that lies inside
    # the fine level's crop; the fine level re-writes that whole crop region with the correct
    # GT box -> stitched Dice must beat coarse-only and hit 1.0.
    from src.totalseg_dataloader_incontext import _ALL_CLASSES_IDX
    idx = _ALL_CLASSES_IDX["liver"]
    D = H = W = 16
    lbl = np.zeros((D, H, W), dtype=np.uint8)
    lbl[6:10, 6:10, 6:10] = idx            # GT box
    subj = tmp_path / "s0"
    subj.mkdir()
    np.save(subj / "label.npy", lbl)

    # Patch class-index lookup + source root used by _stitched_native_dice_multi.
    monkeypatch.setattr(evaluate, "_source_root", lambda cfg: (None, str(tmp_path), False),
                        raising=False)

    # coarse: whole-volume crop, prediction = an offset/wrong box (inside the fine crop)
    coarse_pred = np.zeros((D, H, W), bool); coarse_pred[4:8, 4:8, 4:8] = True
    coarse_geom = [[0, 0, 0], [D, H, W], [D, H, W], [0, 0, 0]]
    # fine: crop = native[4:12, 4:12, 4:12]; prediction = exactly the GT box within that crop
    fine_pred = np.zeros((8, 8, 8), bool); fine_pred[2:6, 2:6, 2:6] = True
    fine_geom = [[4, 4, 4], [8, 8, 8], [8, 8, 8], [0, 0, 0]]

    base = _pg_entry(coarse_pred, coarse_geom)
    fine = _pg_entry(fine_pred, fine_geom)

    d_coarse = evaluate._stitched_native_dice_multi([base], str(tmp_path))
    d_casc = evaluate._stitched_native_dice_multi([base, fine], str(tmp_path))
    assert d_casc[("s0", "liver")] > d_coarse[("s0", "liver")]
    assert d_casc[("s0", "liver")] == pytest.approx(1.0, abs=1e-6)


def test_two_arg_wrapper_matches_multi(tmp_path):
    # _stitched_native_dice(base, over, root) == _stitched_native_dice_multi([base, over], root),
    # and _stitched_native_dice(base, {}, root) == _stitched_native_dice_multi([base], root).
    from src.totalseg_dataloader_incontext import _ALL_CLASSES_IDX
    idx = _ALL_CLASSES_IDX["liver"]
    D = H = W = 16
    lbl = np.zeros((D, H, W), dtype=np.uint8)
    lbl[4:8, 4:8, 4:8] = idx
    subj = tmp_path / "s0"
    subj.mkdir()
    np.save(subj / "label.npy", lbl)

    coarse_pred = np.zeros((D, H, W), bool); coarse_pred[0:8, :, :] = True
    coarse_geom = [[0, 0, 0], [D, H, W], [D, H, W], [0, 0, 0]]
    fine_pred = np.ones((4, 4, 4), bool)
    fine_geom = [[4, 4, 4], [4, 4, 4], [4, 4, 4], [0, 0, 0]]
    base = _pg_entry(coarse_pred, coarse_geom)
    fine = _pg_entry(fine_pred, fine_geom)

    assert evaluate._stitched_native_dice(base, fine, str(tmp_path)) == \
        evaluate._stitched_native_dice_multi([base, fine], str(tmp_path))
    assert evaluate._stitched_native_dice(base, {}, str(tmp_path)) == \
        evaluate._stitched_native_dice_multi([base], str(tmp_path))
