import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "experiments" / "3d"))
from evaluate import _summarize


def test_summarize_includes_grid_means_when_present():
    cases = [{"class": "c", "subject": "s", "dice": 0.8, "time_ms": 1.0,
              "dice_ds": 0.6, "dice_ds_soft": 0.7, "cossim": 0.9},
             {"class": "c", "subject": "s2", "dice": 0.6, "time_ms": 1.0,
              "dice_ds": 0.4, "dice_ds_soft": 0.5, "cossim": 0.7}]
    row = _summarize("c", cases)
    assert row["mean_dice_ds"] == 0.5
    assert row["mean_dice_ds_soft"] == 0.6
    assert row["mean_cossim"] == 0.8


def test_summarize_omits_grid_means_when_absent():
    cases = [{"class": "c", "subject": "s", "dice": 0.8, "time_ms": 1.0}]
    assert "mean_dice_ds" not in _summarize("c", cases)
