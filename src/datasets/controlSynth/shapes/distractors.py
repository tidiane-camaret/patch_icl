"""
Distractor injection: the GEOMETRY side of task_ambiguity (spec ss10.5).

Adds regions that share the foreground's SHAPE (translated/flipped copies of the
fg mask). Their count scales with task_ambiguity. Each gets its own label so the
live intensity step (appearance.gmm_fill) can decide how many additionally share
the fg's intensity mean -- that is the live side of ambiguity. At ambiguity 0 the
foreground is uniquely identifiable; as it rises the rule must be read off context.
"""

import numpy as np

from ..config import map_ambiguity_n_distractors


def _random_place(fg_mask, rng):
    """A translated (and maybe flipped) copy of fg_mask, kept inside the frame."""
    m = fg_mask
    if rng.random() < 0.5:
        m = m[:, ::-1]
    if rng.random() < 0.5:
        m = m[::-1, :]
    ys, xs = np.nonzero(m)
    if len(ys) == 0:
        return np.zeros_like(fg_mask)
    H, W = m.shape
    # feasible shift range that keeps the shape on-canvas
    dy = rng.integers(-ys.min(), H - ys.max())
    dx = rng.integers(-xs.min(), W - xs.max())
    out = np.zeros_like(fg_mask)
    out[ys + dy, xs + dx] = 1
    return out


def inject_distractors(label_map, fg_mask, ambiguity, num_labels,
                       label_start, rng, max_tries=6):
    """Paint shape-distractors into label_map. Returns list of distractor labels.

    `label_start` is the first free label id; distractors take consecutive ids.
    Distractors are painted over background filler but will be overwritten by the
    foreground (caller paints fg last), so they never occlude the true region.
    """
    n = map_ambiguity_n_distractors(ambiguity, num_labels)
    labels = []
    for i in range(n):
        placed = None
        for _ in range(max_tries):
            cand = _random_place(fg_mask, rng)
            # avoid stacking distractors directly on top of each other
            if placed is None or (cand & np.isin(label_map, labels)).sum() < cand.sum() * 0.3:
                placed = cand
                break
            placed = cand
        lab = label_start + i
        label_map[placed > 0] = lab
        labels.append(lab)
    return labels
