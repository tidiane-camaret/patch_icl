"""Shared helpers for the results/experiments analysis notebooks.

Keeps the wandb-table plumbing and the common size bucketing in one place so the
marimo notebooks (11, 12, …) don't each redefine it.
"""

import re
from pathlib import Path

import pandas as pd

ARTIFACTS = Path(__file__).parent / "artifacts"        # absolute — notebooks run from elsewhere
PROJECT = "tidiane/patch_icl_2d_exps_train"

# Shared native-foreground-px (@128²) size buckets used across notebooks.
SZ_EDGES = [0, 32, 128, 512, 2048, 1e9]
SZ_LABELS = ["≤32", "33-128", "129-512", "513-2048", ">2048"]


def get_latest_table(run, table_key="val/samples.table.json"):
    """Download the *latest* version of a run's logged val sample table as a DataFrame."""
    def vnum(a):
        m = re.search(r"v(\d+)$", str(a.version))
        return int(m.group(1)) if m else 0

    arts = [a for a in run.logged_artifacts() if a.type == "run_table"]
    if not arts:
        return None
    latest = max(arts, key=vnum)
    return latest.get(table_key).get_dataframe()


def add_szbin(df):
    """Add the shared `szbin` (and `src` from the dataset path) columns in place; return df."""
    df["szbin"] = pd.cut(df.tgt_size, SZ_EDGES, labels=SZ_LABELS)
    if "dataset" in df.columns:
        df["src"] = df.dataset.str.split("/").str[1]
    return df
