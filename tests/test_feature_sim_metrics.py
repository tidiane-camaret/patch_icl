# tests/test_feature_sim_metrics.py
import sys; sys.path.insert(0, "."); sys.path.insert(0, "experiments/3d")
import torch
from feature_sim.metrics import (
    l2norm, auroc, average_precision,
    prototype_cosine, fg_match_margin, retrieval_at1)


def _separable(n=64, c=8, sep=6.0, seed=0):
    """FG clustered near +e0, BG near -e0; both target and context share the geometry."""
    g = torch.Generator().manual_seed(seed)
    def make(nfg, nbg):
        f = 0.3 * torch.randn(nfg + nbg, c, generator=g)
        f[:nfg, 0] += sep; f[nfg:, 0] -= sep
        lab = torch.cat([torch.ones(nfg), torch.zeros(nbg)])
        return f, lab
    tf, tl = make(n, n)
    cf, cl = make(n, n)
    return tf, tl, cf, cl


def test_auroc_perfect_and_chance():
    s = torch.tensor([0.9, 0.8, 0.2, 0.1]); y = torch.tensor([1., 1., 0., 0.])
    assert abs(auroc(s, y) - 1.0) < 1e-6
    # reversed labels -> 0.0
    assert abs(auroc(s, 1 - y) - 0.0) < 1e-6


def test_average_precision_perfect():
    s = torch.tensor([0.9, 0.8, 0.2, 0.1]); y = torch.tensor([1., 1., 0., 0.])
    assert abs(average_precision(s, y) - 1.0) < 1e-6


def test_l2norm_unit():
    x = torch.randn(5, 8)
    assert torch.allclose(l2norm(x).norm(dim=-1), torch.ones(5), atol=1e-5)


def test_prototype_cosine_separable_dense():
    tf, tl, cf, cl = _separable()
    out = prototype_cosine(tf, tl, cf, cl, mode="dense")
    assert out["auroc"] > 0.95 and out["soft_dice"] > 0.9


def test_prototype_cosine_separable_point():
    tf, tl, cf, cl = _separable()
    out = prototype_cosine(tf, tl, cf, cl, mode="point")
    assert out["auroc"] > 0.95 and out["ap"] > 0.9
    assert "soft_dice" not in out


def test_margin_and_retrieval_separable():
    tf, tl, cf, cl = _separable()
    assert fg_match_margin(tf, tl, cf, cl) > 0.3
    assert retrieval_at1(tf, tl, cf, cl) > 0.95


def test_random_features_are_chance():
    g = torch.Generator().manual_seed(1)
    tf = torch.randn(200, 8, generator=g); cf = torch.randn(200, 8, generator=g)
    tl = (torch.arange(200) % 2).float(); cl = (torch.arange(200) % 2).float()
    out = prototype_cosine(tf, tl, cf, cl, mode="dense")
    assert 0.35 < out["auroc"] < 0.65
    assert abs(fg_match_margin(tf, tl, cf, cl)) < 0.1
