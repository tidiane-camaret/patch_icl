# experiments/3d/feature_sim/metrics.py
"""Transformer-free target<->context matching metrics on encoder feature rows.

Rows are either dense grid cells or sampled points — the functions don't care.
Pure torch (no sklearn); AUROC is the rank-based Mann-Whitney U statistic."""
import torch


def l2norm(x, dim=-1, eps=1e-8):
    return x / (x.norm(dim=dim, keepdim=True) + eps)


def auroc(scores, labels):
    """P(score[pos] > score[neg]) via mean rank of positives (ties -> average rank)."""
    labels = labels.float()
    n_pos = labels.sum().item(); n_neg = labels.numel() - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = scores.argsort()
    ranks = torch.empty_like(scores, dtype=torch.float)
    ranks[order] = torch.arange(1, scores.numel() + 1, dtype=torch.float, device=scores.device)
    # average tied ranks so exact ties score 0.5
    uniq, inv = torch.unique(scores, return_inverse=True)
    mean_rank = torch.zeros_like(uniq, dtype=torch.float).scatter_reduce(
        0, inv, ranks, reduce="mean", include_self=False)
    ranks = mean_rank[inv]
    sum_pos = ranks[labels == 1].sum().item()
    return (sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def average_precision(scores, labels):
    """Area under precision-recall via the step sum sum_k P(k) * dRecall(k)."""
    labels = labels.float()
    n_pos = labels.sum().item()
    if n_pos == 0:
        return float("nan")
    order = scores.argsort(descending=True)
    y = labels[order]
    tp = torch.cumsum(y, 0)
    precision = tp / torch.arange(1, y.numel() + 1, dtype=torch.float, device=y.device)
    return (precision * y).sum().item() / n_pos


def soft_auroc(scores, pos_w):
    """Probabilistic AUC with soft positive weights `pos_w` in [0,1] (neg weight = 1-pos_w):
    P(score of a pos-weighted draw > score of a neg-weighted draw), ties counted 0.5.
    Reduces exactly to `auroc` when pos_w is binary. O(n log n) via score-grouped cumsum.

    Matches the model's soft-Dice training, where the target is an occupancy fraction per
    cell rather than a hard 0/1 label."""
    pw = pos_w.float().clamp(0, 1)
    nw = 1.0 - pw
    P, N = pw.sum(), nw.sum()
    if P <= 0 or N <= 0:
        return float("nan")
    uniq, inv = torch.unique(scores, return_inverse=True)          # uniq ascending
    grp_nw = torch.zeros(len(uniq), device=scores.device).scatter_add(0, inv, nw)  # neg weight per group
    below = torch.cumsum(grp_nw, 0) - grp_nw                        # neg weight of strictly lower scores
    contrib = pw * (below[inv] + 0.5 * grp_nw[inv])                # + half the ties at the same score
    return (contrib.sum() / (P * N)).item()


def soft_dice(scores, soft_labels, eps=1e-8):
    """Soft-Dice between the cosine score map and the occupancy fraction `soft_labels`.

    Scores are per-map min-max normalized to [0,1] first: raw cosine sits near a ~0.5
    baseline for every cell, which swamps the sparse occupancy of thin structures, so a
    fixed [-1,1]->[0,1] map yields a near-zero, uninformative Dice. Min-max makes it a
    relative overlap comparable across tiers/resolutions (not an absolute probability, and
    not directly comparable to the model's real Dice). Never nan when the object is present.
    soft_auroc is the scale-free separability headline; this is the segmentation-quality proxy."""
    lo, hi = scores.min(), scores.max()
    s = (scores - lo) / (hi - lo + eps)
    g = soft_labels.float().clamp(0, 1)
    inter = (s * g).sum()
    return (2 * inter / (s.sum() + g.sum() + eps)).item()


def _prototype_scores(target_feats, ctx_feats, ctx_w):
    """Occupancy-weighted context prototype (weights = ctx occupancy fraction, clamped >=0);
    reduces to the FG-mean prototype when ctx_w is binary. Returns per-target-cell cosine."""
    n = l2norm(ctx_feats)
    w = ctx_w.float().clamp(min=0)
    proto = l2norm((n * w.unsqueeze(1)).sum(0) / (w.sum() + 1e-8), dim=0)
    return l2norm(target_feats) @ proto


def prototype_cosine(target_feats, target_labels, ctx_feats, ctx_labels, mode="dense"):
    """Dense: soft labels (occupancy fraction) -> {soft_auroc, soft_dice}. Point (native
    res, exact 0/1 voxels): {binary auroc, average precision}. `auroc` key holds the soft
    variant in dense mode, the hard one in point mode."""
    scores = _prototype_scores(target_feats, ctx_feats, ctx_labels)
    if mode == "dense":
        return {"auroc": soft_auroc(scores, target_labels),
                "soft_dice": soft_dice(scores, target_labels)}
    if mode == "point":
        return {"auroc": auroc(scores, target_labels),
                "ap": average_precision(scores, target_labels)}
    raise ValueError(f"mode must be 'dense' or 'point', got {mode!r}")


def fg_match_margin(target_feats, target_labels, ctx_feats, ctx_labels):
    """Mean cosine of target-FG cells to context-FG minus to context-BG. FG membership is
    `label > 0` (any occupancy), so it is non-degenerate for soft (fractional) labels and
    identical to `== 1` for binary labels. nan if the target has no FG cell."""
    tf = l2norm(target_feats)[target_labels > 0]
    cf = l2norm(ctx_feats)
    sims = tf @ cf.T                                   # (n_tfg, M)
    fg = sims[:, ctx_labels > 0].mean(1)
    bg = sims[:, ctx_labels == 0].mean(1)
    return (fg - bg).mean().item()


def retrieval_at1(target_feats, target_labels, ctx_feats, ctx_labels):
    """Fraction of target-FG cells whose nearest context cell is also FG (`label > 0`)."""
    tf = l2norm(target_feats)[target_labels > 0]
    cf = l2norm(ctx_feats)
    nn = (tf @ cf.T).argmax(1)
    return (ctx_labels[nn] > 0).float().mean().item()
