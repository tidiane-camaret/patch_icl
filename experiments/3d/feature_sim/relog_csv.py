"""Re-log an already-written feature_sim.csv to wandb as the aggregated per-class table
(+ per-(tier,res) scalar summary) without recomputing the sweep. Use when run.py finished
the CSV but the wandb.Table step failed, or to (re)build the aggregate from an old CSV.

The full per-task table (130k rows) is intentionally NOT logged — it is too large for
wandb; the CSV on disk stays the source of truth. See run.py:aggregate_by_class.

    python experiments/3d/feature_sim/relog_csv.py \
        /path/to/3d_feature_sim/feature_sim.csv --project patch_icl_feature_similarity
"""
import argparse
import collections
import csv
import sys
from pathlib import Path

import wandb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # experiments/3d
from feature_sim.run import aggregate_by_class, summarize_by_config, _num   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path")
    ap.add_argument("--project", default="patch_icl_feature_similarity")
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    with open(args.csv_path, newline="") as fh:
        rows = list(csv.DictReader(fh))    # cells stay strings; aggregate_by_class coerces
    print(f"Loaded {len(rows)} per-task rows from {args.csv_path}")

    afields, arows = aggregate_by_class(rows)
    sfields, srows = summarize_by_config(rows)
    print(f"Aggregated to {len(arows)} per-class rows + {len(srows)} config-summary rows")

    wandb.init(project=args.project, name=args.name)
    wandb.log({"feature_sim/by_class":
               wandb.Table(columns=afields, data=[[r[c] for c in afields] for r in arows])})
    wandb.log({"feature_sim/by_config":
               wandb.Table(columns=sfields, data=[[r[c] for c in sfields] for r in srows])})
    agg = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        key = f"{r['tier']}@{r['res']}"
        for m in ("auroc", "margin", "retrieval_at1"):
            v = _num(r[m])
            if v is not None:
                agg[key][m].append(v)
    wandb.log({f"feature_sim/{m}/{key}": sum(v) / len(v)
               for key, ms in agg.items() for m, v in ms.items()})
    wandb.finish()
    print(f"Re-logged aggregated table ({len(arows)} rows) + summary.")


if __name__ == "__main__":
    main()
