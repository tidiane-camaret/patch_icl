"""
Quantify how well anchor_synth3d target objects blend into the CT background.

Draws N random items and, for each target object, compares the object's interior
intensity to a thin shell of surrounding background voxels. Reports separability
metrics per object and their distribution:

  delta      = mean(object) - mean(shell)                (signed local contrast)
  cohen_d    = |delta| / pooled_std                      (effect size)
  local_auc  = direction-agnostic AUC of intensity separating object from shell,
               in [0.5, 1.0].  ~0.5 => object blends into its surroundings (the
               model must use the anchor-relative in-context cue); ~1.0 => the
               object pops out and the task is trivially solvable by thresholding.

Usage
-----
  python experiments/3d/analyze_object_blend.py dataset=anchor_synth3d
  python experiments/3d/analyze_object_blend.py dataset=anchor_synth3d \
      anchor_synth.contrast_delta=0.25 --n_items 64 --shell 3
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from hydra import compose, initialize_config_dir
from scipy.ndimage import binary_dilation

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling `common`

from common import build_dataset  # noqa: E402


def _local_auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Direction-agnostic Mann-Whitney AUC of intensity separating `pos` (object)
    from `neg` (shell), folded to [0.5, 1.0] so darker- and brighter-than-background
    objects are equally 'separable'."""
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    ranks = allv.argsort().argsort().astype(np.float64) + 1.0  # average-ish ranks
    r_pos = ranks[: pos.size].sum()
    auc = (r_pos - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size)
    return float(max(auc, 1.0 - auc))


def _stats(item, shell_iter: int):
    """Per-object blend stats for the target volume of one item. Returns None if
    the object is empty (anchor absent)."""
    img = item["image"][0].numpy().astype(np.float32)   # (D, H, W)
    obj = item["label"].numpy() > 0
    if not obj.any():
        return None
    shell = binary_dilation(obj, iterations=shell_iter) & ~obj
    obj_v, shell_v = img[obj], img[shell]
    if shell_v.size == 0:
        return None
    pooled = np.sqrt((obj_v.var() + shell_v.var()) / 2.0) + 1e-8
    return {
        "obj_mean": float(obj_v.mean()), "obj_std": float(obj_v.std()),
        "shell_mean": float(shell_v.mean()), "shell_std": float(shell_v.std()),
        "delta": float(obj_v.mean() - shell_v.mean()),
        "cohen_d": float(abs(obj_v.mean() - shell_v.mean()) / pooled),
        "local_auc": _local_auc(obj_v, shell_v),
        "n_obj": int(obj.sum()),
    }


def _pct(x, p):
    return float(np.percentile(x, p)) if len(x) else float("nan")


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--n_items", type=int, default=48)
    parser.add_argument("--shell", type=int, default=3, help="shell thickness (voxels)")
    parser.add_argument("-h", "--help", action="store_true")
    args, hydra_overrides = parser.parse_known_args()
    if args.help:
        parser.print_help(); return

    with initialize_config_dir(config_dir=str(ROOT / "configs" / "experiment" / "3d"),
                               version_base="1.3"):
        cfg = compose(config_name="train", overrides=hydra_overrides)

    ds = build_dataset(cfg, args.split)
    n = min(args.n_items, len(ds))
    rng = np.random.default_rng(0)
    idxs = rng.integers(0, len(ds), size=n)

    rows, img_lo, img_hi = [], [], []
    for i in idxs:
        item = ds[int(i)]
        img_lo.append(float(item["image"].min())); img_hi.append(float(item["image"].max()))
        s = _stats(item, args.shell)
        if s is not None:
            rows.append(s)

    if not rows:
        print("No non-empty objects drawn — check anchor pool / config."); return

    auc = np.array([r["local_auc"] for r in rows])
    d = np.array([r["cohen_d"] for r in rows])
    delta = np.array([r["delta"] for r in rows])
    nobj = np.array([r["n_obj"] for r in rows])

    a = cfg.anchor_synth
    print(f"\nanchor_synth3d blend analysis  (split={args.split}, n={len(rows)}/{n} "
          f"non-empty, shell={args.shell}vx)")
    print(f"config: contrast_delta={a.contrast_delta}  "
          f"object_size={a.object_size_min}-{a.object_size_max_frac}·img  "
          f"offset_range={a.offset_range}  edge_blur={a.edge_blur}")
    print(f"image intensity range over items: [{min(img_lo):.3f}, {max(img_hi):.3f}]")
    print("-" * 68)

    def line(name, x, fmt="{:.3f}"):
        print(f"  {name:<12s} med={fmt.format(_pct(x,50))}  "
              f"p10={fmt.format(_pct(x,10))}  p90={fmt.format(_pct(x,90))}  "
              f"mean={fmt.format(float(np.mean(x)))}")

    line("local_auc", auc)
    line("cohen_d", d)
    line("|delta|", np.abs(delta))
    line("obj_voxels", nobj, "{:.0f}")

    frac_easy = float((auc > 0.90).mean())
    frac_blend = float((auc < 0.70).mean())
    print("-" * 68)
    print(f"  fraction 'too easy' (local_auc>0.90): {frac_easy:.0%}")
    print(f"  fraction well-blended (local_auc<0.70): {frac_blend:.0%}")
    verdict = ("TOO EASY — objects pop out; lower contrast_delta" if _pct(auc, 50) > 0.90
               else "OK — objects blend; task needs the in-context cue"
               if _pct(auc, 50) < 0.75 else "BORDERLINE — consider lowering contrast_delta")
    print(f"  verdict (median local_auc={_pct(auc,50):.3f}): {verdict}\n")


if __name__ == "__main__":
    main()
