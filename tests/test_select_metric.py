import sys; sys.path.insert(0, "."); sys.path.insert(0, "experiments/2d")
from train import _select_metric


def test_refine_selects_native_dice():
    # A refine model (dice_fused@R present) selects on native `dice` — for refine that is
    # the fused prediction scored at full resolution — not dice_fused@R and not cossim.
    s = {"dice_fused@64/mean": 0.7, "dice_fused_soft@64/mean": 0.6,
         "cossim/mean": 0.5, "dice/mean": 0.4}
    assert _select_metric(s) == ("dice", 0.4)


def test_falls_back_to_cossim():
    # cossim now carries the coarse token grid in its name (cossim@{T}); selection finds it.
    assert _select_metric({"cossim@32/mean": 0.5, "dice/mean": 0.4}) == ("cossim@32", 0.5)


def test_falls_back_to_dice():
    assert _select_metric({"dice/mean": 0.4}) == ("dice", 0.4)
