"""Dummy driver for the cascade figure path — builds synthetic coarse/fine figure caches
(exactly what evaluate_classes stashes) and runs the real _save_cascade_pair, so the box
placement + fine-pred refit are exercised end to end before wiring to real eval data.

    python experiments/3d/cascade_fig_demo.py   # -> results/figures/cascade/<cls>_4to1.5mm.png
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import _save_cascade_pair


def ball(T, center, r):
    zz, yy, xx = np.indices((T, T, T))
    return ((zz - center[0]) ** 2 + (yy - center[1]) ** 2 + (xx - center[2]) ** 2 <= r * r).astype(np.float32)


def ct_like(mask, seed=0):
    rng = np.random.default_rng(seed)
    img = 0.25 * rng.standard_normal(mask.shape).astype(np.float32) + 0.9 * mask
    return img + 0.4 * ball(mask.shape[0], (mask.shape[0] // 2,) * 3, mask.shape[0] // 2 - 2)


def main():
    T, s0, s1 = 64, 4.0, 1.5
    zoom = T / round(T * s1 / s0)
    gc, pc = np.array([30, 34, 33]), np.array([33, 30, 36])   # GT / coarse-pred centroids

    coarse = {                                                # what evaluate_classes stores
        "img":     ct_like(ball(T, gc, 9), 1),
        "gt":      ball(T, gc, 9),
        "pred":    ball(T, pc, 8),
        "ctx_img": ct_like(ball(T, (34, 30, 30), 10), 2),
        "ctx_gt":  ball(T, (34, 30, 30), 10),
        "prob":    ball(T, pc, 8),                            # soft prob ~ hard here
        "spacing": s0,
    }
    gc_f = (gc - pc) * zoom + T / 2                           # GT centroid inside the fine crop
    fine = {
        "img":     ct_like(ball(T, gc_f, 9 * zoom), 3),
        "gt":      ball(T, gc_f, 9 * zoom),
        "pred":    ball(T, np.array([26, 40, 27]), 8 * zoom),
        "ctx_img": ct_like(ball(T, (30, 30, 34), 20), 4),
        "ctx_gt":  ball(T, (30, 30, 34), 20),
        "prob":    None,
        "spacing": s1,
    }
    coarse_cache = {("s0000", "dummy_organ"): coarse}
    fine_cache   = {("s0000", "dummy_organ"): fine}

    out_dir = Path("results/figures/cascade")
    _save_cascade_pair(coarse_cache, fine_cache, s0, s1, out_dir)
    print(f"wrote {out_dir}/dummy_organ_4to1.5mm.png")


if __name__ == "__main__":
    main()
