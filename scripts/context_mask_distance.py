"""
Within-set mask variability for an in-context (target + K context) set.

Each dataset item is a set of masks sharing the same (dataset, target) cell:
the target label + its K context masks. This measures how much those masks differ
from each other -- the real anatomical/positional spread that controlSynth's
`support_query_shift` (per-subject deformation of one base shape) is meant to mimic.

Per set, averages over all mask pairs:
  overlay_dice  — Dice of the two binary masks laid on the same grid (1 = identical
                  position+shape; low = masks land in different places / sizes).
  centroid_dist — normalized distance between mask centroids (pure position spread).
  area_logratio — |log2(area_i / area_j)| (size spread, scale-free).

Usage:
    python scripts/context_mask_distance.py --source biomedparse --n 1500
    python scripts/context_mask_distance.py --source synthetic --synth hard_diverse
"""

import argparse
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from hydra import compose, initialize_config_dir

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "experiments" / "2d"))
from src.datasets.medsegbench import MedSegBenchDataset  # noqa: F401  (cache patch_icl src)
from common import build_dataset  # noqa: E402


def _centroid(m):
    ys, xs = np.nonzero(m)
    H, W = m.shape
    return ys.mean() / (H - 1), xs.mean() / (W - 1)


def set_distances(masks):
    """Mean pairwise (overlay_dice, centroid_dist, area_logratio) over a mask set."""
    masks = [m for m in masks if m.any()]
    if len(masks) < 2:
        return None
    od, cd, ar = [], [], []
    cents = [_centroid(m) for m in masks]
    areas = [m.sum() for m in masks]
    for i, j in combinations(range(len(masks)), 2):
        inter = np.logical_and(masks[i], masks[j]).sum()
        denom = areas[i] + areas[j]
        od.append(2 * inter / denom if denom else np.nan)
        cd.append(float(np.hypot(cents[i][0] - cents[j][0], cents[i][1] - cents[j][1])))
        ar.append(abs(np.log2(areas[i] / areas[j])))
    return float(np.nanmean(od)), float(np.mean(cd)), float(np.mean(ar))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="biomedparse", choices=["biomedparse", "synthetic", "medsegbench"])
    p.add_argument("--split", default="train")
    p.add_argument("--synth", default="hard_diverse")
    p.add_argument("--context-size", type=int, default=3)
    p.add_argument("--res", type=int, default=128)
    p.add_argument("--n", type=int, default=1500, help="number of sets sampled")
    p.add_argument("--num-tasks", type=int, default=600, help="synth bank size")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    overrides = [f"data.source={args.source}", f"data.context_size={args.context_size}",
                 f"data.image_size={args.res}"]
    if args.source == "synthetic":
        overrides += [f"synth={args.synth}", f"synth.diversity.num_tasks={args.num_tasks}"]
    with initialize_config_dir(version_base=None, config_dir=str(_ROOT / "configs/experiment/2d")):
        cfg = compose(config_name="pfn_seg", overrides=overrides)
    ds = build_dataset(cfg, args.split)

    rng = np.random.default_rng(args.seed)
    idxs = rng.choice(len(ds), size=min(args.n, len(ds)), replace=False)
    rows = []
    per_ds = defaultdict(list)
    for idx in idxs:
        item = ds[int(idx)]
        masks = [item["label"][0].numpy() > 0.5]
        co = item["context_out"]
        for k in range(co.shape[0]):
            masks.append(co[k, 0].numpy() > 0.5)
        d = set_distances(masks)
        if d is None:
            continue
        rows.append(d)
        if args.source == "biomedparse":
            per_ds[ds.dataset_of(int(idx))].append(d)

    if not rows:
        print("no valid sets"); return
    A = np.array(rows)                                   # (N, 3)
    names = ["overlay_dice", "centroid_dist", "area_logratio"]
    pct = lambda a, q: float(np.percentile(a, q))
    label = f"{args.source}" + (f"/{args.synth}" if args.source == "synthetic" else "")
    print(f"\nWithin-(target+{args.context_size} context) mask distance — {label} "
          f"(split={args.split}, {len(rows)} sets, res={args.res})")
    for c, nm in enumerate(names):
        a = A[:, c]
        print(f"  {nm:<14} mean={a.mean():.3f}  p25={pct(a,25):.3f} p50={pct(a,50):.3f} "
              f"p75={pct(a,75):.3f} p95={pct(a,95):.3f}")

    if per_ds:
        # macro over datasets (each cell weighted equally) for the headline number
        macro = np.array([np.mean(v, axis=0) for v in per_ds.values()])
        print(f"\n  macro-avg over {len(per_ds)} datasets:  "
              + "  ".join(f"{nm}={macro[:, c].mean():.3f}" for c, nm in enumerate(names)))
        worst = sorted(per_ds.items(), key=lambda kv: np.mean([r[0] for r in kv[1]]))[:6]
        print("  lowest overlay_dice (most within-set spread):")
        for ds_name, v in worst:
            m = np.mean(v, axis=0)
            print(f"    {ds_name:<28} overlay_dice={m[0]:.3f} centroid_dist={m[1]:.3f} area_logratio={m[2]:.2f}")


if __name__ == "__main__":
    main()
