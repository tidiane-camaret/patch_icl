import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "experiments" / "3d"))
import torch
from grid_metrics import target_like, soft_sum, hard_sum, cos_sum


def test_target_like_pools_to_logit_res():
    lbl = torch.rand(2, 1, 16, 16, 16)
    logit = torch.zeros(2, 1, 4, 4, 4)
    assert target_like(lbl, logit).shape == (2, 1, 4, 4, 4)


def test_perfect_overlap_scores_one():
    g = (torch.rand(3, 1, 4, 4, 4) > 0.5).float()
    s, c = soft_sum(g, g); h, hc = hard_sum(g, g); k, kc = cos_sum(g, g)
    assert torch.isclose(s / c, torch.tensor(1.0), atol=1e-4)
    assert torch.isclose(h / hc, torch.tensor(1.0), atol=1e-4)
    assert torch.isclose(k / kc, torch.tensor(1.0), atol=1e-4)
