import sys; sys.path.insert(0, "."); sys.path.insert(0, "experiments/2d")
from train import _select_metric


def test_prefers_fused_hard_not_soft():
    s = {"dice_fused@64/mean": 0.7, "dice_fused_soft@64/mean": 0.6,
         "cossim/mean": 0.5, "dice/mean": 0.4}
    assert _select_metric(s) == ("dice_fused@64", 0.7)


def test_falls_back_to_cossim():
    assert _select_metric({"cossim/mean": 0.5, "dice/mean": 0.4}) == ("cossim", 0.5)


def test_falls_back_to_dice():
    assert _select_metric({"dice/mean": 0.4}) == ("dice", 0.4)
