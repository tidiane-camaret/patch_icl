"""Compare cascade_n_patches=1 vs >1 (and sample_mode=prob vs uncertainty) on a real
eval source (FLARE22 or totalseg-multisource CT+MRI), same checkpoint/eval seed.

Zero-shot probe (docs/superpowers/specs/2026-10-13-multipatch-cascade-design.md), now
extended with two follow-up directions run as an autonomous overnight sweep:
  - data.cascade_sample_mode: "prob" (default, peaked at confident foreground) vs
    "uncertainty" (peaked at the model's own decision boundary, p~0.5).
  - a smaller fine-level crop_spacing_mm (proxy for "smaller input size" -- the frozen
    checkpoint's decode head bakes in a fixed native image_size, so data.image_size
    itself can't be changed at eval time without rebuilding the model; crop_spacing_mm
    instead holds T=128 fixed and makes each voxel cover more physical distance,
    which is the cascade's own native "resolution vs FOV" lever and the one actually
    safe to sweep against a frozen checkpoint -- see docs/logs.md for the full
    reasoning and what was tried first).

Despite the filename (kept for continuity with earlier runs/docs), this script now
works on ANY cascade-capable dataset via the usual `dataset=<group>` override -- class
resolution is picked by cfg.data.source after composition, not hardcoded.

Forces data.cascade_query_prior="pred" (the only prior mode _build_query_prior_multi
supports -- it asserts on anything else) for every sweep point including the n_patches=1
baseline, matching the checkpoint's own trained recipe (`cascade_query_prior: pred` in
configs/experiment/3d/experiment/168_cascade_real_host_all_shapes.yaml), so the
comparison isolates the swept variable alone under the SAME prior regime the checkpoint
was trained with -- not a comparison against some other unrelated baseline.

Usage (run from the REPO ROOT, not experiments/3d -- the eval config's own hydra
searchpath resolves relative to $PWD):
    python experiments/3d/compare_multipatch_flare22.py \
        eval.model=patchset3d \
        eval.checkpoint=/path/to/best.pt \
        eval.split=val \
        eval.batch_size=1 \
        dataset=flare22 \
        augmentations.enabled=false \
        data.crop_spacing_mm=4 \
        +data.cascade_spacings=[4,1.5] \
        +data.cascade_n_patches_sweep=[1,2,4] \
        +data.cascade_sample_mode_sweep=[prob,uncertainty]

    # totalseg CT+MRI (in-distribution; the checkpoint's own training source):
    python experiments/3d/compare_multipatch_flare22.py \
        eval.model=patchset3d eval.checkpoint=/path/to/best.pt eval.batch_size=1 \
        dataset=multisource_ct_mri augmentations.enabled=false \
        data.crop_spacing_mm=4 +data.cascade_spacings=[4,1.5] \
        +data.cascade_n_patches_sweep=[1,2,4]

Results are both printed AND written as one JSON file per run under
results/experiments/3d/multipatch_cascade_probe/ (timestamped, dataset+source-tagged) --
the printed table is for a human watching; the JSON is the thing later analysis reads.
`eval.batch_size=1` and `data.gpu_realize_crop=false` (forced below) are both required
for n_patches>1 on a 16GB GPU / the current run_cascade implementation respectively --
see docs/logs.md. Like the forced prior mode, both mean the numbers here are NOT
directly comparable to any externally-published number produced under different
settings.
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import hydra
from omegaconf import OmegaConf

from cascade import evaluate_cascade
from eval import _build_model
from common import make_eval_loader


def _resolve_classes(cfg):
    """Class list for evaluate_cascade's reporting, by cfg.data.source (set by whichever
    `dataset=` group was composed -- NOT hardcoded, so this script works for any
    cascade-capable source, not just FLARE22)."""
    src = cfg.data.get("source", "totalseg")
    if src == "flare22":
        from src.providers.flare22 import resolve_flare22_classes
        return resolve_flare22_classes(cfg.data.get("val_classes", "all"))
    if src == "multisource":
        from common import resolve_multisource_classes
        return resolve_multisource_classes(cfg, "val")
    from data.totalseg_classes import resolve_classes
    from common import _source_root
    _, root, is_mri = _source_root(cfg)
    return resolve_classes(cfg.data.get("val_classes"), root, is_mri=is_mri)


@hydra.main(config_path="../../configs/experiment/3d", config_name="eval", version_base="1.3")
def main(cfg):
    OmegaConf.set_struct(cfg, False)
    cfg.data.cascade_query_prior = "pred"
    # gpu_realize_crop=True is unsupported for n_patches>1; force it off to keep the
    # script self-contained. Like the forced prior mode, this means the numbers printed
    # here are NOT directly comparable to any externally-published number produced with
    # gpu_realize_crop=true.
    cfg.data.gpu_realize_crop = False

    n_sweep = [int(n) for n in cfg.data.get("cascade_n_patches_sweep", [1, 2, 4])]
    assert n_sweep[0] == 1, "the first cascade_n_patches_sweep value must be 1 (the baseline)"
    mode_sweep = list(cfg.data.get("cascade_sample_mode_sweep", ["prob"]))
    for m in mode_sweep:
        assert m in ("prob", "uncertainty"), f"cascade_sample_mode_sweep entry {m!r} invalid"

    model = _build_model(cfg)
    classes = _resolve_classes(cfg)
    loader = make_eval_loader(cfg, classes, split=cfg.eval.get("split", "val"))

    src = cfg.data.get("source", "totalseg")
    spacings = list(cfg.data.get("cascade_spacings", []))
    run_id = time.strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "results" / "experiments" / "3d" / "multipatch_cascade_probe"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_id}_{src}.json"
    record = {
        "run_id": run_id, "source": src, "spacings": spacings,
        "checkpoint": str(cfg.eval.get("checkpoint")), "n_sweep": n_sweep,
        "mode_sweep": mode_sweep, "points": [],
    }

    baseline_by_mode = {}
    for sample_mode in mode_sweep:
        cfg.data.cascade_sample_mode = sample_mode
        for n in n_sweep:
            # n_patches=1 is mode-independent (sample_mode only affects levels>=1's
            # re-crop, which n_patches<=1 never reaches via the multi-window path) --
            # skip re-running an identical point for every mode after the first.
            if n == 1 and sample_mode != mode_sweep[0]:
                by_cls = baseline_by_mode[mode_sweep[0]]
            else:
                cfg.data.cascade_n_patches = n
                rows, _cases = evaluate_cascade(model, cfg, classes, loader=loader,
                                                seed=int(cfg.eval.get("seed", 0)), is_prob=False)
                by_cls = {r["class"]: r for r in rows if "error" not in r}
            if n == 1:
                baseline_by_mode[sample_mode] = by_cls
            tag = f"n={n} mode={sample_mode}"
            print(f"\n=== {tag} ===")
            for cls, r in sorted(by_cls.items()):
                dice, nsd = r["mean_dice"], r.get("mean_nsd")
                line = f"  {cls:24s} dice={dice:.4f}"
                if nsd is not None:
                    line += f" nsd={nsd:.4f}"
                print(line)
                record["points"].append({
                    "n_patches": n, "sample_mode": sample_mode, "class": cls,
                    "mean_dice": dice, "mean_nsd": nsd, "n_samples": r.get("n_samples"),
                })
            out_path.write_text(json.dumps(record, indent=2))   # incremental: never lose partial progress

    print(f"\nWrote {len(record['points'])} result rows to {out_path}")


if __name__ == "__main__":
    main()
