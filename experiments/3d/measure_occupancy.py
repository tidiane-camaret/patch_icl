"""
Measure per-class target-mask occupancy over one full train epoch, at two resolutions:

  - native   : foreground fraction of the 128³ target crop (object-size proxy).
  - grid @R  : the mask pooled DOWN to the model's R³ prediction grid exactly like
               train's `target_like` (adaptive_avg_pool3d), then the count / fraction of
               cells that survive the `data.mask_occupancy_thr` gate. This is the
               granularity the loss/dice actually see — the suspected driver of the
               "small object -> low dice" stall.

Iterates train_loader(cfg) (same sampler / aug / class-balance the trainer uses) for one
epoch, so the numbers reflect exactly what the model is trained on. Writes a per-item CSV
(for later correlation with Dice) and a per-class summary CSV, and prints the summary
sorted by native occupancy.

    python experiments/3d/measure_occupancy.py experiment=35_colipri_enc_8_i_128 \
      data.train_classes=all data.crop_spacing_mm=1.5 data.mask_occupancy_thr=0.1 \
      data.class_balanced=true data.raw_ct=true
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path

import hydra
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import DEVICE, train_loader


@hydra.main(config_path="../../configs/experiment/3d", config_name="train", version_base="1.3")
def main(cfg: DictConfig) -> None:
    R = int(cfg.arch.resolution)
    thr = float(cfg.data.get("mask_occupancy_thr", 0.5))
    ncells = R ** 3
    loader = train_loader(cfg)
    print(f"Measuring occupancy over one epoch: {len(loader)} batches | R={R} "
          f"(cells={ncells}) | occupancy_thr={thr} | crop_spacing={cfg.data.get('crop_spacing_mm')}mm")

    # Per-class running accumulators.
    agg = defaultdict(lambda: {"n": 0, "native": 0.0, "gcells": 0.0,
                               "gocc": 0.0, "maxcell": 0.0, "empty": 0})
    rows = []  # per-item, for later Dice correlation

    for batch in tqdm(loader, desc="epoch"):
        lbl = batch["label"].to(DEVICE).float()          # (B,D,H,W)
        names = batch["label_names"]
        subs = batch.get("subjects", [None] * lbl.shape[0])
        B = lbl.shape[0]
        native = lbl.flatten(1).mean(1)                  # (B,) native fg fraction
        cells = F.adaptive_avg_pool3d(lbl.unsqueeze(1), (R, R, R)).flatten(1)  # (B,R³) soft occ
        on = (cells >= thr).float()
        n_on = on.sum(1)                                 # (B,) positive cells @thr
        gocc = n_on / ncells                             # (B,) grid occupancy fraction
        maxcell = cells.max(1).values                    # (B,) strongest cell occupancy
        for b in range(B):
            c = names[b]
            a = agg[c]
            a["n"] += 1
            a["native"] += float(native[b])
            a["gcells"] += float(n_on[b])
            a["gocc"] += float(gocc[b])
            a["maxcell"] += float(maxcell[b])
            a["empty"] += int(n_on[b] == 0)
            rows.append({"class": c, "subject": subs[b],
                         "native_occ": round(float(native[b]), 6),
                         "grid_cells_on": int(n_on[b]),
                         "grid_occ": round(float(gocc[b]), 6),
                         "max_cell_occ": round(float(maxcell[b]), 4)})

    # Per-class summary.
    summary = []
    for c, a in agg.items():
        n = a["n"]
        summary.append({
            "class": c, "n": n,
            "native_occ": a["native"] / n,
            "grid_cells_on": a["gcells"] / n,
            "grid_occ": a["gocc"] / n,
            "max_cell_occ": a["maxcell"] / n,
            "pct_empty_grid": 100.0 * a["empty"] / n,
        })
    summary.sort(key=lambda r: r["native_occ"])

    out_dir = Path("results/3d"); out_dir.mkdir(parents=True, exist_ok=True)
    sp_tag = f"sp{float(cfg.data.get('crop_spacing_mm', 1.5)):g}".replace(".", "p")
    item_csv = out_dir / f"occupancy_items_{sp_tag}.csv"
    sum_csv = out_dir / f"occupancy_per_class_{sp_tag}.csv"
    with open(item_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    with open(sum_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys())); w.writeheader(); w.writerows(summary)

    print(f"\n{'class':<28}{'n':>4}{'native_occ':>12}{'grid_cells':>12}"
          f"{'grid_occ':>10}{'maxcell':>9}{'%empty':>8}")
    print("-" * 91)
    def fmt(r):
        return (f"{r['class']:<28}{r['n']:>4}{r['native_occ']:>12.5f}{r['grid_cells_on']:>12.2f}"
                f"{r['grid_occ']:>10.4f}{r['max_cell_occ']:>9.3f}{r['pct_empty_grid']:>7.1f}%")
    for r in summary[:20]:
        print(fmt(r))
    print(f"  ... ({len(summary)} classes total) ...")
    for r in summary[-8:]:
        print(fmt(r))

    tot = sum(a["n"] for a in agg.values())
    print(f"\nItems: {tot} over {len(agg)} classes "
          f"({tot/max(len(agg),1):.1f}/class avg)")
    print(f"Grid cells total = {ncells}. Classes with mean grid_cells_on < 5: "
          f"{sum(1 for r in summary if r['grid_cells_on'] < 5)}")
    print(f"Mean %empty-grid (mask vanishes at R={R} under thr={thr}): "
          f"{sum(r['pct_empty_grid'] for r in summary)/len(summary):.1f}%")
    print(f"Wrote {item_csv}  and  {sum_csv}")


if __name__ == "__main__":
    main()
