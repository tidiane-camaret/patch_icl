"""Route A: faithful TotalSegmentator baseline on NATIVE volumes.

Runs the official `totalsegmentator()` pipeline once per test subject — native ct.nii.gz
through its OWN rough-crop + 1.5 mm resample + sliding-window + CTNormalization — extracts
all requested val classes from the single multilabel output, and scores each against our
native GT (label.npy, RAS) with the SAME metrics (evaluate.dice_batch / nsd_batch) and
wandb/CSV logging as experiments/3d/eval.py, for comparability.

Contrast with Route B (`eval.py eval.model=totalsegmentator`): B runs the TS organ net on
our 128^3 crops inside the in-context eval loop; A runs the full published system on whole
volumes. A is the faithfulness reference — the A-B gap isolates TS's rough-crop / full-FOV /
own-resampling. Route A scores the WHOLE organ in native RAS; Route B is crop-limited, so the
two Dice numbers are not on the same geometry (documented, intended).

    python experiments/3d/eval_totalseg_native.py experiment=52_organs_real_nnunet_ts
    python experiments/3d/eval_totalseg_native.py experiment=52_organs_real_nnunet_ts \
        eval.n_subjects=20 wandb.project=null           # quick offline run

Env: needs `totalsegmentator` importable and its weights cached (~/.totalsegmentator). Use
.venv_blackwell on the Blackwell node (has TS + GPU).
"""
import csv
import datetime
import json
import sys
import time
from pathlib import Path

import hydra
import nibabel as nib
import numpy as np
import torch
import wandb
from omegaconf import DictConfig, ListConfig, OmegaConf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling common/evaluate

from data.totalseg_classes import resolve_classes
from src.totalseg_dataset import _ALL_CLASSES_IDX
from common import DEVICE
from evaluate import dice_batch, nsd_batch, _summarize


def _test_subjects(root: Path, split: str) -> list[str]:
    """Subject ids for `split` from meta.csv (BOM + semicolon-delimited)."""
    subs = []
    with open(root / "meta.csv", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f, delimiter=";"):
            if (row.get("split") or "").strip() == split:
                subs.append(row["image_id"].strip())
    return subs


@hydra.main(config_path="../../configs/experiment/3d", config_name="eval", version_base="1.3")
def main(cfg: DictConfig) -> None:
    root = Path(cfg.paths.totalseg)
    split = cfg.eval.split
    classes = resolve_classes(cfg.data.val_classes, root)
    tol = cfg.eval.get("nsd_tolerance_mm")

    # class name -> label id in each space: TS multilabel uses class_map["total"] (merged 117),
    # our GT label.npy uses ALL_CLASSES ordering. Extract each class BY NAME on both sides.
    from totalsegmentator.map_to_binary import class_map
    ts_id = {name: idx for idx, name in class_map["total"].items()}
    classes = [c for c in classes if c in ts_id and c in _ALL_CLASSES_IDX]

    spacings = json.loads((root / "spacings.json").read_text())
    subjects = _test_subjects(root, split)[: cfg.eval.n_subjects]
    ts_device = "gpu" if DEVICE.type == "cuda" else "cpu"

    print(f"Device       : {DEVICE}  (TS device={ts_device})")
    print(f"Data root    : {root}  (split={split})")
    print(f"Classes ({len(classes)}): {', '.join(classes)}")
    print(f"Subjects     : {len(subjects)}  |  NSD tol: {tol} mm\n")

    from totalsegmentator.python_api import totalsegmentator

    # ── wandb / output ───────────────────────────────────────────────────────
    wb_on = bool(cfg.wandb.get("project"))
    val_classes = cfg.data.get("val_classes")
    if isinstance(val_classes, (DictConfig, ListConfig)):
        val_classes = OmegaConf.to_container(val_classes, resolve=True)
    run = wandb.init(
        project=cfg.wandb.project, name=cfg.wandb.name or "totalsegmentator_native",
        mode="online" if wb_on else "disabled",
        config={"model": "totalsegmentator_native", "route": "A", "split": split,
                "classes": list(classes), "val_classes": val_classes,
                "n_subjects": len(subjects), "nsd_tolerance_mm": tol},
    )
    run_name = (wandb.run.name if wandb.run is not None else None) or "totalsegmentator_native"
    out_dir = Path(cfg.eval.out_dir) / f"{datetime.date.today():%Y-%m-%d}_{run_name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    cases: list[dict] = []
    subj_times, n_skipped = [], 0
    from tqdm import tqdm
    for sid in tqdm(subjects, desc="TS native"):
        subj_dir = root / sid
        ct_path, gt_path = subj_dir / "ct.nii.gz", subj_dir / "label.npy"
        if not ct_path.exists() or not gt_path.exists():
            n_skipped += 1
            continue
        gt = np.asanyarray(np.load(gt_path, mmap_mode="r")).astype(np.uint8)
        present = [c for c in classes if (gt == _ALL_CLASSES_IDX[c]).any()]
        if not present:
            continue

        # ONE faithful pass -> multilabel in native space. Canonicalize to RAS so it lands on
        # label.npy's grid (both = as_closest_canonical of the same acquisition).
        t0 = time.perf_counter()
        ts_img = totalsegmentator(str(ct_path), None, ml=True, task="total",
                                  roi_subset=list(classes), device=ts_device, quiet=True,
                                  nr_thr_resamp=1, nr_thr_saving=1)
        pass_ms = (time.perf_counter() - t0) * 1000
        subj_times.append(pass_ms)
        ts = np.asanyarray(nib.as_closest_canonical(ts_img).dataobj).astype(np.uint8)
        if ts.shape != gt.shape:
            print(f"  [warn] {sid}: TS shape {ts.shape} != GT {gt.shape}; skipping subject.")
            n_skipped += 1
            continue

        sp = [float(x) for x in spacings[sid]["spacing"]]
        gt_t = torch.from_numpy(gt).to(DEVICE)
        ts_t = torch.from_numpy(ts).to(DEVICE)
        for c in present:
            ts_bin = (ts_t == ts_id[c]).float()[None]
            gt_bin = (gt_t == _ALL_CLASSES_IDX[c]).float()[None]
            case = {"class": c, "subject": sid,
                    "dice": round(float(dice_batch(ts_bin, gt_bin)[0]), 4),
                    "time_ms": round(pass_ms, 1)}   # per-VOLUME pass (all classes at once)
            if tol is not None:
                case["nsd"] = round(float(nsd_batch(ts_bin, gt_bin, sp, float(tol))[0]), 4)
            cases.append(case)

    # ── aggregate (reuse evaluate._summarize per class) ──────────────────────
    from collections import defaultdict
    by_class: dict[str, list[dict]] = defaultdict(list)
    for c in cases:
        by_class[c["class"]].append(c)
    rows = [_summarize(cls, by_class[cls]) for cls in classes if by_class.get(cls)]

    for r in rows:
        nsd_str = f"  nsd={r['mean_nsd']:.3f}" if "mean_nsd" in r else ""
        print(f"  {r['class']:<24s} dice={r['mean_dice']:.3f} ± {r['std_dice']:.3f}{nsd_str}"
              f"  n={r['n_samples']}")
        if wb_on:
            wandb.log({f"class/{r['class']}/mean_dice": r["mean_dice"]})
            if "mean_nsd" in r:
                wandb.log({f"class/{r['class']}/mean_nsd": r["mean_nsd"]})

    if rows:
        mean_dice = sum(r["mean_dice"] for r in rows) / len(rows)
        nsd_vals = [r["mean_nsd"] for r in rows if "mean_nsd" in r]
        mean_nsd = sum(nsd_vals) / len(nsd_vals) if nsd_vals else None
        mean_vol_s = (sum(subj_times) / len(subj_times) / 1000) if subj_times else 0.0
        nsd_line = f"  |  Mean NSD: {mean_nsd:.4f}" if mean_nsd is not None else ""
        print(f"\n  Mean Dice: {mean_dice:.4f}{nsd_line}  |  {mean_vol_s:.1f} s/volume  "
              f"|  skipped {n_skipped}")
        if wb_on:
            log = {"mean_dice": round(mean_dice, 4), "sec_per_volume": round(mean_vol_s, 2)}
            if mean_nsd is not None:
                log["mean_nsd"] = round(mean_nsd, 4)
            wandb.log(log)

    # ── save (mirror eval.py's eval.json / eval.csv) ─────────────────────────
    (out_dir / "eval.json").write_text(json.dumps(
        {"model": "totalsegmentator_native", "route": "A",
         "config": OmegaConf.to_container(cfg.eval, resolve=True),
         "rows": rows, "cases": cases}, indent=2))
    nsd_col = ",mean_nsd,std_nsd" if any("mean_nsd" in r for r in rows) else ""
    csv_lines = [f"model,class,mean_dice,std_dice,mean_time_ms,n_samples{nsd_col}"]
    csv_lines += [f"totalsegmentator_native,{r['class']},{r['mean_dice']},{r['std_dice']},"
                  f"{r.get('mean_time_ms','')},{r['n_samples']}"
                  + (f",{r.get('mean_nsd','')},{r.get('std_nsd','')}" if nsd_col else "")
                  for r in rows]
    (out_dir / "eval.csv").write_text("\n".join(csv_lines) + "\n")
    print(f"  Saved -> {out_dir}")
    run.finish()


if __name__ == "__main__":
    main()
