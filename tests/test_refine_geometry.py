import sys; sys.path.insert(0, "."); sys.path.insert(0, "experiments/2d")
import torch
from evaluate import refine_geometry


def _out(B=2, T=4):
    torch.manual_seed(0)
    return {"final_logit": torch.randn(B, 1, T, T),
            "refine_logit": torch.randn(B, 1, T, T),
            "refine_origin": torch.tensor([[0, 0], [8, 8]]),
            "refine_crop": 8, "resolutions": [4, 8]}


def test_none_for_single_level():
    assert refine_geometry({"final_logit": torch.randn(2, 1, 4, 4)},
                           torch.rand(2, 1, 16, 16)) is None


def test_shapes_and_ranges():
    out = _out()
    lbl = (torch.rand(2, 1, 16, 16) > 0.5).float()
    rg = refine_geometry(out, lbl)
    assert rg["Rf"] == 8
    assert rg["refine_prob"].shape == (2, 1, 4, 4)
    assert rg["refine_target"].shape == (2, 1, 4, 4)
    assert rg["fused_R"].shape == (2, 1, 8, 8)
    assert rg["gt_R"].shape == (2, 1, 8, 8)
    assert (rg["fused_R"] >= 0).all() and (rg["fused_R"] <= 1).all()   # probabilities


def test_fused_takes_refine_inside_window():
    # coarse all -inf (prob 0), refine all +inf (prob 1) → fused prob is 1 inside the crop
    # window and 0 outside. With origin (0,0) crop 8 on a 16 image, pooled-to-8 fused should be
    # 1 in the top-left 4x4 (the crop) and 0 elsewhere.
    B, T = 1, 4
    out = {"final_logit": torch.full((B, 1, T, T), -30.0),
           "refine_logit": torch.full((B, 1, T, T), 30.0),
           "refine_origin": torch.tensor([[0, 0]]), "refine_crop": 8, "resolutions": [4, 8]}
    rg = refine_geometry(out, torch.zeros(B, 1, 16, 16))
    f = rg["fused_R"][0, 0]
    assert f[:4, :4].mean() > 0.99            # crop region takes refine (prob 1)
    assert f[4:, 4:].mean() < 0.01            # outside stays coarse (prob 0)
