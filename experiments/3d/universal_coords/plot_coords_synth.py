"""Visual QA for coords-function synthetic labels BEFORE wiring into training.

For each field family we sample one field f(coords) (params anchored at a random
reference subject), then find K+1 subjects whose in-FOV label mass passes the
guard, evaluate f on each subject's coords, and render target + K contexts on
axial CT slices. If the shared frame works, the SAME anatomy is highlighted in
every panel (contexts + target).

Rows = field instances; cols = subjects (col 0 = target, rest = contexts).
Run (loki): .venv_thor_fresh/bin/python experiments/3d/universal_coords/plot_coords_synth.py
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from coords_synth_consistency import (TS, DS, MIN_MASS, FIELDS, sample_params,
                                      subject_ids, soft_hist, coords_aabb)
from data.totalseg_classes import ALL_CLASSES

MIN_HI = 0.15         # cross-subject anatomy-consistency backstop for a task

FIGS = os.path.join(os.path.dirname(__file__), "figs")
CTW = (-160, 240)
IDX2NAME = {i + 1: c for i, c in enumerate(ALL_CLASSES)}


def top_anatomy(lab, w, n=2):
    """Top-n anatomy names by soft field-weighted mass (bg-excluded)."""
    h = np.bincount(lab.reshape(-1), weights=w.reshape(-1), minlength=256)[:256]
    h[0] = 0.0
    if h.sum() <= 0:
        return "(none)"
    order = np.argsort(h)[::-1]
    return ", ".join(f"{IDX2NAME.get(int(i), i)[:14]} {h[i]/h.sum():.0%}"
                     for i in order[:n] if h[i] > 0)


def load_full(sid):
    co = np.load(os.path.join(TS, sid, "coords.npy"))[::DS, ::DS, ::DS].astype(np.float32)
    lab = np.load(os.path.join(TS, sid, "label.npy"))[::DS, ::DS, ::DS]
    ct = np.load(os.path.join(TS, sid, "ct.npy"))[::DS, ::DS, ::DS].astype(np.float32)
    return co, lab, ct


def best_k(vol):
    return int(vol.sum((0, 1)).argmax())


def panel(ax, ct, field3d, title, hard):
    k = best_k(field3d)
    ax.imshow(np.rot90(np.clip(ct[:, :, k], *CTW)), cmap="gray")
    ov = field3d[:, :, k]
    if hard:
        ax.imshow(np.rot90(np.ma.masked_less(ov, 0.5)), cmap="autumn", alpha=0.55,
                  interpolation="nearest", vmin=0, vmax=1)
    else:
        ax.imshow(np.rot90(np.ma.masked_less(ov, 0.05)), cmap="jet", alpha=0.55,
                  interpolation="nearest", vmin=0, vmax=1)
    ax.set_title(title, fontsize=8); ax.axis("off")


def build_task(family, scale, cache, aabb, pool, K, rng):
    """FOV-aware task: anchor field at a random ref location, keep only subjects
    whose coords-AABB contains the anchor, apply the mass guard, then require the
    K+1 picked subjects to agree on anatomy (mean pairwise HI >= MIN_HI)."""
    fld = FIELDS[family]
    for _ in range(120):
        ref = cache[pool[rng.integers(len(pool))]][0].reshape(-1, 3)
        p = sample_params(family, scale, ref, rng)
        mu = p["mu"]
        cand = [s for s in pool if (aabb[s][0] <= mu).all() and (mu <= aabb[s][1]).all()]
        picked, hists = [], []
        for s in rng.permutation(cand):
            co, lab, ct = cache[s]
            w = fld(co.reshape(-1, 3), p)
            if w.sum() < MIN_MASS:
                continue
            h = soft_hist(lab.reshape(-1), w)
            if h is None:
                continue
            picked.append((s, w.reshape(lab.shape), ct)); hists.append(h)
            if len(picked) == K + 1:
                break
        if len(picked) < K + 1:
            continue
        pair = [float(np.minimum(hists[i], hists[j]).sum())
                for i in range(len(hists)) for j in range(i + 1, len(hists))]
        if np.mean(pair) >= MIN_HI:
            return picked, float(np.mean(pair))
    return picked, (np.mean(pair) if picked else 0.0)


def main(K=3, seed=3):
    os.makedirs(FIGS, exist_ok=True)
    rng = np.random.default_rng(seed)
    ids = subject_ids(); rng.shuffle(ids); pool = ids[:30]
    cache = {s: load_full(s) for s in pool}
    aabb = {s: coords_aabb(cache[s][0].reshape(-1, 3), cache[s][1].reshape(-1)) for s in pool}

    # localized/bounded families only — (family, scale, hard?)
    configs = [("ellipsoid", 60, True), ("ellipsoid", 90, True), ("cyl_capped", 45, True),
               ("gaussian", 60, False), ("gaussian", 40, False)]

    fig, axs = plt.subplots(len(configs), K + 1, figsize=(3.2 * (K + 1), 3.0 * len(configs)))
    for row, (family, scale, hard) in zip(axs, configs):
        picked, hi = build_task(family, scale, cache, aabb, pool, K, rng)
        tag = f"{family}" + (f" s={scale}" if scale else "") + ("[hard]" if hard else "[soft]")
        for j, (ax, (s, field3d, ct)) in enumerate(zip(row, picked)):
            role = "TARGET" if j == 0 else f"ctx{j}"
            anat = top_anatomy(cache[s][1], field3d)
            panel(ax, ct, field3d, f"{tag} {role} {s} (HI={hi:.2f})\n{anat}", hard)

    fig.suptitle(f"Coords-function synth labels (FOV-aware, localized): same field on {K+1} "
                 f"subjects (col0=target). Same anatomy => frame works.", fontsize=12)
    fig.tight_layout()
    out = os.path.join(FIGS, "coords_synth_examples.png")
    fig.savefig(out, dpi=95, bbox_inches="tight"); print("saved", out)


if __name__ == "__main__":
    main()
