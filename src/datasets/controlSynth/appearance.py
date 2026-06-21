"""
Appearance: per-region GMM intensity fill + noise (the cheap live path).

`gmm_fill` gives every region a Gaussian mean. `foreground_contrast` sets the
MIN separation of the fg mean from its non-distractor neighbours (low contrast =
hard, fg nearly intensity-invisible). `task_ambiguity_intensity` decides how many
distractor regions are additionally given the fg mean -- the live side of
ambiguity, so those distractors become separable from fg only by shape/context.

`add_noise` blends precomputed Perlin (cropped from the store's noise bank) with
white Gaussian noise -- no per-call Perlin synthesis (spec ss9 hard rule).
"""

import numpy as np

from .config import map_contrast_gap, map_noise_sigma, map_texture_std


def gmm_fill(label_map, fg_label, distractor_labels, contrast, texture,
             ambiguity_intensity, rng, fg_sign=None):
    """Render a float32 image in ~[0,1] from a region label map.

    `foreground_contrast` sets how far the FOREGROUND mean sits from the background
    intensity cluster: low contrast -> fg inside the cluster (intensity-invisible,
    found only via shape/context -> hard); high contrast -> fg pushed to the
    intensity extremes (salient -> easy). Background region means stay in a fixed
    central band [0.2, 0.8], so the extremes belong to a high-contrast foreground
    rather than the background. (Fixes the original inversion where raising contrast
    saturated the *background* and buried the fg as a bland mid-grey blob.)

    `fg_sign` (+1/-1) is the side of the cluster the fg sits on. It must be a
    TASK-level constant (passed from the base-geometry meta) so the foreground has a
    consistent intensity across a task's context+target subjects -- otherwise a high
    contrast pushes each subject's fg to an independent extreme and the context no
    longer matches the target (which inverts the difficulty). None -> drawn here
    (standalone use only).
    """
    labels = np.unique(label_map)
    gap = map_contrast_gap(contrast)        # fg distance from the background centre
    tex_std = map_texture_std(texture)

    # Background: a stable moderate-contrast cluster, kept off the [0,1] extremes so a
    # high-contrast foreground can own them.
    bg_center, bg_spread = 0.5, 0.13
    # Size the LUT to every label we *write* (fg + distractors), not just those
    # present in `label_map`: the elastic warp can fold a thin foreground (or the
    # high-numbered distractors) out of frame, so fg_label may exceed labels.max().
    # Slots for absent labels stay 0 but are never read (img = lut[label_map]).
    max_label = int(labels.max())
    max_label = max(max_label, int(fg_label))
    if distractor_labels:
        max_label = max(max_label, int(max(distractor_labels)))
    lut = np.zeros(max_label + 1, dtype=np.float32)
    for lab in labels:
        if lab != fg_label:
            lut[lab] = float(np.clip(rng.normal(bg_center, bg_spread), 0.25, 0.75))

    # Foreground: distance `gap` from the cluster centre, toward the task's fixed side.
    sign = (1.0 if rng.random() < 0.5 else -1.0) if fg_sign is None else float(fg_sign)
    fg_mean = float(np.clip(bg_center + sign * gap, 0.0, 1.0))
    lut[fg_label] = fg_mean

    # Live ambiguity: a fraction of distractors collide with the fg mean.
    if distractor_labels:
        n_share = int(round(float(ambiguity_intensity) * len(distractor_labels)))
        for lab in rng.permutation(distractor_labels)[:n_share]:
            lut[lab] = float(np.clip(fg_mean + rng.normal(0, 0.02), 0.0, 1.0))

    img = lut[label_map].astype(np.float32)
    if tex_std > 0:
        img = img + rng.normal(0.0, tex_std, size=img.shape).astype(np.float32)
    return np.clip(img, 0.0, 1.0)


def add_noise(img, level, noise_bank, rng):
    """Add Perlin (from bank) + white Gaussian noise scaled by `level`."""
    sigma = map_noise_sigma(level)
    if sigma <= 0.0:
        return np.clip(img, 0.0, 1.0)
    H, W = img.shape
    out = img.astype(np.float32)
    if noise_bank is not None and len(noise_bank) > 0:
        field = noise_bank[rng.integers(len(noise_bank))]
        if field.shape != img.shape:                 # crop a random window
            fy = rng.integers(0, field.shape[0] - H + 1)
            fx = rng.integers(0, field.shape[1] - W + 1)
            field = field[fy:fy + H, fx:fx + W]
        out = out + 0.6 * sigma * field.astype(np.float32)
    out = out + 0.6 * sigma * rng.standard_normal((H, W)).astype(np.float32)
    return np.clip(out, 0.0, 1.0)
