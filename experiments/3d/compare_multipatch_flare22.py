"""Compare cascade_n_patches=1 vs >1 on FLARE22, same checkpoint/eval seed.

Zero-shot probe (docs/superpowers/specs/2026-10-13-multipatch-cascade-design.md):
no training change, just whether sampling N windows per role from a continuous
probability field helps or hurts FLARE22's 13 abdominal organs under the existing
best cascade checkpoint. Forces data.cascade_query_prior="pred" for n_patches>1 (the
only prior mode _build_query_prior_multi supports -- it asserts on anything else) and
ALSO for the n_patches=1 baseline run, matching the checkpoint's own trained recipe
(`cascade_query_prior: pred` in configs/experiment/3d/experiment/
168_cascade_real_host_all_shapes.yaml), so the comparison isolates the n-patch effect
alone under the SAME prior regime the checkpoint was trained with -- not a comparison
against some other unrelated baseline.

Usage:
    cd experiments/3d
    python compare_multipatch_flare22.py \
        eval.checkpoint=/path/to/best.pt \
        eval.model=patchset3d \
        dataset=flare22 \
        data.cascade_spacings=[4,1.5] \
        data.cascade_n_patches_sweep=[1,2,4]
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import hydra
from omegaconf import OmegaConf

from cascade import evaluate_cascade
from eval import _build_model
from common import make_eval_loader
from src.providers.flare22 import resolve_flare22_classes


@hydra.main(config_path="../../configs/experiment/3d", config_name="eval", version_base="1.3")
def main(cfg):
    OmegaConf.set_struct(cfg, False)
    cfg.data.source = "flare22"
    cfg.data.cascade_query_prior = "pred"
    # gpu_realize_crop=True is unsupported for n_patches>1; force it off
    # to keep the script self-contained and runnable without extra CLI overrides.
    # Like the forced prior mode, this means the numbers printed here are NOT directly
    # comparable to any externally-published number produced with gpu_realize_crop=true.
    cfg.data.gpu_realize_crop = False
    sweep = [int(n) for n in cfg.data.get("cascade_n_patches_sweep", [1, 2, 4])]
    assert sweep[0] == 1, "the first sweep value must be 1 (the baseline)"

    model = _build_model(cfg)
    classes = resolve_flare22_classes(cfg.data.get("val_classes", "all"))
    loader = make_eval_loader(cfg, classes, split=cfg.eval.get("split", "val"))

    baseline = None
    for n in sweep:
        cfg.data.cascade_n_patches = n
        rows, _cases = evaluate_cascade(model, cfg, classes, loader=loader,
                                        seed=int(cfg.eval.get("seed", 0)), is_prob=False)
        by_cls = {r["class"]: r for r in rows if "error" not in r}
        if n == 1:
            baseline = by_cls
            print(f"\n=== n_patches=1 (baseline) ===")
            for cls, r in by_cls.items():
                print(f"  {cls:24s} dice={r['mean_dice']:.4f}"
                     + (f" nsd={r.get('mean_nsd', float('nan')):.4f}" if "mean_nsd" in r else ""))
            continue
        print(f"\n=== n_patches={n} (delta vs baseline) ===")
        for cls, r in by_cls.items():
            base = baseline.get(cls)
            if base is None:
                continue
            ddice = r["mean_dice"] - base["mean_dice"]
            line = f"  {cls:24s} dice={r['mean_dice']:.4f} (Δ{ddice:+.4f})"
            if "mean_nsd" in r and "mean_nsd" in base:
                dnsd = r["mean_nsd"] - base["mean_nsd"]
                line += f" nsd={r['mean_nsd']:.4f} (Δ{dnsd:+.4f})"
            print(line)


if __name__ == "__main__":
    main()
