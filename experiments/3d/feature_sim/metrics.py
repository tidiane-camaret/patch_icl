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
    ranks[order] = torch.arange(1, scores.numel() + 1, dtype=torch.float)
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
    precision = tp / torch.arange(1, y.numel() + 1, dtype=torch.float)
    return (precision * y).sum().item() / n_pos


def _best_soft_dice(scores, labels):
    """Max Dice over thresholds at each unique score (scores expected in a bounded range)."""
    labels = labels.float()
    thr = torch.unique(scores)
    best = 0.0
    for t in thr:
        pred = (scores >= t).float()
        inter = (pred * labels).sum().item()
        den = pred.sum().item() + labels.sum().item()
        d = (2 * inter) / den if den > 0 else 0.0
        best = max(best, d)
    return best


def _prototype_scores(target_feats, ctx_feats, ctx_labels):
    proto = l2norm(l2norm(ctx_feats)[ctx_labels == 1].mean(0), dim=0)
    return l2norm(target_feats) @ proto


def prototype_cosine(target_feats, target_labels, ctx_feats, ctx_labels, mode="dense"):
    scores = _prototype_scores(target_feats, ctx_feats, ctx_labels)
    out = {"auroc": auroc(scores, target_labels)}
    if mode == "dense":
        out["soft_dice"] = _best_soft_dice(scores, target_labels)
    elif mode == "point":
        out["ap"] = average_precision(scores, target_labels)
    else:
        raise ValueError(f"mode must be 'dense' or 'point', got {mode!r}")
    return out


def fg_match_margin(target_feats, target_labels, ctx_feats, ctx_labels):
    tf = l2norm(target_feats)[target_labels == 1]
    cf = l2norm(ctx_feats)
    sims = tf @ cf.T                                   # (n_tfg, M)
    fg = sims[:, ctx_labels == 1].mean(1)
    bg = sims[:, ctx_labels == 0].mean(1)
    return (fg - bg).mean().item()


def retrieval_at1(target_feats, target_labels, ctx_feats, ctx_labels):
    tf = l2norm(target_feats)[target_labels == 1]
    cf = l2norm(ctx_feats)
    nn = (tf @ cf.T).argmax(1)
    return (ctx_labels[nn] == 1).float().mean().item()
