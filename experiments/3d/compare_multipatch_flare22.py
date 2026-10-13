"""Compare cascade_n_patches=1 vs >1 on FLARE22, same checkpoint/eval seed.

Zero-shot probe (docs/superpowers/specs/2026-10-13-multipatch-cascade-design.md):
no training change, just whether sampling N windows per role from a continuous
probability field helps or hurts FLARE22's 13 abdominal organs under the existing
best cascade checkpoint. Forces data.cascade_query_prior="none" for n_patches>1
(the only prior mode Task 5's _build_query_prior_multi supports today) and ALSO for
the n_patches=1 baseline run, so the comparison isolates the n-patch effect alone --
NOT a comparison against the checkpoint's own best-published query_prior="pred" n=1
number (see memory project_gnc_kidney_lesions / docs/logs.md for that reference point).

Usage:
    cd experiments/3d
    python compare_multipatch_flare22.py \
        eval.checkpoint=/path/to/best.pt \
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


@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def main(cfg):
    OmegaConf.set_struct(cfg, False)
    cfg.data.source = "flare22"
    cfg.data.cascade_query_prior = "none"
    # gpu_realize_crop=True is unsupported for n_patches>1; force it off
    # to keep the script self-contained and runnable without extra CLI overrides
    cfg.data.gpu_realize_crop = False
    sweep = list(cfg.data.get("cascade_n_patches_sweep", [1, 2, 4]))
    assert sweep[0] == 1, "the first sweep value must be 1 (the baseline)"

    model = _build_model(cfg)
    loader = make_eval_loader(cfg, split=cfg.eval.get("split", "val"))
    classes = list(loader.dataset.provider.ALL_CLASSES)

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
