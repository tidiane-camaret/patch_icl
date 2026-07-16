import sys; sys.path.insert(0, "."); sys.path.insert(0, "experiments/2d")
import torch
from src.models.patchset_cnn import PatchSetCNN


def _scatter_out(B=2, K=2, H=32, res=(8, 16), M=20):
    torch.manual_seed(0)
    m = PatchSetCNN(image_size=H, resolution=res[0], enc_dims=[16], e=32, h=64, l=1, a=2,
                    thinking_rows=1, resolutions=list(res), refine_mode="scatter",
                    sample={"n_total": M, "n_fg_core": 4, "n_fg_core_ctx": 4}).eval()
    img = torch.rand(B, 1, H, H)
    cin = torch.rand(B, K, 1, H, H)
    cout = (torch.rand(B, K, 1, H, H) > 0.5).float()
    lbl = (torch.rand(B, 1, H, H) > 0.5).float()
    with torch.no_grad():
        out = m(img, context_in=cin, context_out=cout)
    return out, lbl


def test_refine_geometry_scatter_keys_and_shapes():
    import importlib
    ev = importlib.import_module("evaluate")   # experiments/2d on sys.path at runtime
    out, lbl = _scatter_out()
    rg = ev.refine_geometry(out, lbl)
    B, H, Rf, M = 2, 32, 16, 20
    assert rg["Rf"] == Rf
    assert rg["refine_prob"].shape == (B, 1, M)
    assert rg["refine_target"].shape == (B, 1, M)
    assert rg["fused"].shape == (B, 1, H, H)
    assert rg["fused_R"].shape == (B, 1, Rf, Rf)
    assert rg["gt_R"].shape == (B, 1, Rf, Rf)
    assert rg["coarse_nat"].shape == (B, 1, H, H)
    assert rg["coarse_R"].shape == (B, 1, Rf, Rf)
    # fused is a valid probability map everywhere
    assert torch.isfinite(rg["fused"]).all()
    # dice helpers work on scatter's (M,) per-sample slices
    import numpy as np
    from common import hard_dice, soft_dice   # experiments/2d already on sys.path in this test
    rdh = hard_dice(rg["refine_prob"][0, 0], (rg["refine_target"][0, 0] >= 0.5).float())
    assert (0.0 <= rdh <= 1.0) or np.isnan(rdh)
    rds = soft_dice(rg["refine_prob"][0, 0], rg["refine_target"][0, 0])
    assert (0.0 <= rds <= 1.0) or np.isnan(rds)
