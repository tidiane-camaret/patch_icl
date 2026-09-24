#!/usr/bin/env python3
"""Generic per-sample eval-table extractor for thesis experiment writeups.

Reads a `runs.json` (wandb run pointers + names/descriptions, one per thesis
experiment folder — see `2a_cascade/runs.json` for the format) and
concatenates each run's per-task eval table (`evaluate.py`'s `val/samples_*`
wandb Table, one row per eval task) into one unified CSV, tagged with
`run_id`/`run_name`/`description` so every row is traceable back to its
source run.

Reads tables from the LOCAL `wandb/run-*-<id>/files/media/table/<split>/`
directory (same source `results/experiments/82_multisource.py` uses) — no
wandb API call, so it only works for runs whose local wandb dir still exists
on this machine.

Usage:
    python extract_run_results.py 2a_cascade/runs.json
    python extract_run_results.py 2a_cascade/runs.json -o 2a_cascade/samples.csv
"""

import argparse
import json
import re
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[4]
WANDB_DIR = REPO_ROOT / "wandb"


def _find_run_dir(run_id: str) -> Path:
    matches = sorted(WANDB_DIR.glob(f"run-*-{run_id}"))
    if not matches:
        raise FileNotFoundError(
            f"no local wandb run dir found for run_id={run_id!r} under {WANDB_DIR}"
        )
    return matches[-1]


def _load_samples_table(run_dir: Path, split: str) -> pd.DataFrame:
    table_dir = run_dir / "files" / "media" / "table" / split
    paths = sorted(
        table_dir.glob("samples_*.table.json"),
        key=lambda p: int(re.search(r"samples_(\d+)_", p.name).group(1)),
    )
    if not paths:
        raise FileNotFoundError(f"no samples_*.table.json found under {table_dir}")
    parts = []
    for p in paths:
        d = json.loads(p.read_text())
        parts.append(pd.DataFrame(d["data"], columns=d["columns"]))
    return pd.concat(parts, ignore_index=True)


def extract(runs_json: Path, split: str = "val") -> pd.DataFrame:
    spec = json.loads(runs_json.read_text())
    frames = []
    for run in spec["runs"]:
        run_dir = _find_run_dir(run["id"])
        df = _load_samples_table(run_dir, split=split)
        df.insert(0, "description", run.get("description", ""))
        df.insert(0, "run_name", run["name"])
        df.insert(0, "run_id", run["id"])
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs_json", type=Path, help="path to a runs.json (wandb run pointers)")
    ap.add_argument(
        "-o", "--output", type=Path, default=None,
        help="output CSV path (default: <runs_json dir>/samples.csv)",
    )
    ap.add_argument("--split", default="val", help="eval split subfolder to read (default: val)")
    args = ap.parse_args()

    out = args.output or args.runs_json.parent / "samples.csv"
    df = extract(args.runs_json, split=args.split)
    df.to_csv(out, index=False)
    print(f"wrote {len(df)} rows ({df['run_id'].nunique()} runs) -> {out}")


if __name__ == "__main__":
    main()
