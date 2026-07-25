# tests/test_feature_sim_metrics.py
import sys; sys.path.insert(0, "."); sys.path.insert(0, "experiments/3d")
import torch
from feature_sim.metrics import (
    l2norm, auroc, average_precision, soft_auroc, soft_dice,
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


def test_auroc_ties():
    s = torch.tensor([1.0, 1.0, 0.0, 0.0]); y = torch.tensor([1., 0., 1., 0.])
    assert abs(auroc(s, y) - 0.5) < 1e-6  # all ties -> chance


def test_metrics_run_on_cuda_and_match_cpu():
    # metrics run on the features' device (GPU in production); guard that no helper tensor
    # is hard-coded to CPU and that GPU results match CPU. No-op on CPU-only nodes.
    if not torch.cuda.is_available():
        return
    tf, tl, cf, cl = _separable()
    occ = tl.clone()                                    # binary here, but exercise soft path too
    cpu = (soft_auroc(_p(tf, cf, cl), occ), soft_dice(_p(tf, cf, cl), occ),
           fg_match_margin(tf, tl, cf, cl), retrieval_at1(tf, tl, cf, cl))
    d = "cuda"
    tf, tl, cf, cl, occ = tf.to(d), tl.to(d), cf.to(d), cl.to(d), occ.to(d)
    gpu = (soft_auroc(_p(tf, cf, cl), occ), soft_dice(_p(tf, cf, cl), occ),
           fg_match_margin(tf, tl, cf, cl), retrieval_at1(tf, tl, cf, cl))
    for a, b in zip(cpu, gpu):
        assert abs(a - b) < 1e-4


def _p(tf, cf, cl):
    from feature_sim.metrics import _prototype_scores
    return _prototype_scores(tf, cf, cl)


def test_soft_auroc_reduces_to_binary():
    # with binary weights, soft_auroc must equal the hard rank-based auroc
    g = torch.Generator().manual_seed(3)
    s = torch.randn(200, generator=g)
    y = (torch.rand(200, generator=g) > 0.5).float()
    assert abs(soft_auroc(s, y) - auroc(s, y)) < 1e-5
    # perfect and tie cases too
    sp = torch.tensor([0.9, 0.8, 0.2, 0.1]); yp = torch.tensor([1., 1., 0., 0.])
    assert abs(soft_auroc(sp, yp) - 1.0) < 1e-6
    st = torch.tensor([1.0, 1.0, 0.0, 0.0]); yt = torch.tensor([1., 0., 1., 0.])
    assert abs(soft_auroc(st, yt) - 0.5) < 1e-6


def test_soft_auroc_fractional_ranks_by_occupancy():
    # higher scores carry higher occupancy weight -> soft_auroc > 0.5
    s = torch.tensor([0.9, 0.7, 0.3, 0.1]); occ = torch.tensor([0.9, 0.6, 0.2, 0.05])
    assert soft_auroc(s, occ) > 0.5
    assert soft_auroc(-s, occ) < 0.5          # reversed scores -> below chance
    # all-foreground (P>0,N=0) is undefined
    import math
    assert math.isnan(soft_auroc(s, torch.ones(4)))


def test_soft_dice_matches_occupancy():
    # a score map aligned with occupancy beats a reversed one; soft-Dice of a graded map
    # against itself is Σg²/Σg (~0.74 here), not 1 — that is inherent to soft-Dice.
    occ = torch.tensor([0.9, 0.6, 0.1, 0.0])
    scores = occ * 2 - 1                       # invert the [-1,1]->[0,1] mapping
    matched = soft_dice(scores, occ)
    assert matched > soft_dice(-scores, occ)
    assert matched > 0.7
    # never nan when the object is present, even if scores are flat
    flat = soft_dice(torch.zeros(4), occ)
    assert flat == flat                        # not nan


def test_prototype_cosine_soft_labels_dense():
    # graded target occupancy that tracks the FG direction -> sensible soft metrics
    g = torch.Generator().manual_seed(7)
    c = 8
    tf = 0.2 * torch.randn(80, c, generator=g); tf[:40, 0] += 5.0     # first 40 near +e0
    occ = torch.cat([torch.linspace(0.4, 1.0, 40), torch.zeros(40)])  # soft FG, hard BG
    cf = 0.2 * torch.randn(80, c, generator=g); cf[:40, 0] += 5.0
    cocc = torch.cat([torch.ones(40), torch.zeros(40)])
    out = prototype_cosine(tf, occ, cf, cocc, mode="dense")
    assert set(out) == {"auroc", "soft_dice"}
    assert out["auroc"] > 0.85 and out["soft_dice"] > 0.6


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
