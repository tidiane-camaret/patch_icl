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


def _table_vnum(a):
    m = re.search(r"v(\d+)$", str(a.version))
    return int(m.group(1)) if m else 0


def latest_table_version(run):
    """Latest run_table artifact version as an int, WITHOUT downloading the table.

    A new version is logged each time the run logs the table (i.e. per eval epoch),
    so this is a cheap staleness signal for artifact caches. Returns -1 if none.
    """
    vs = [_table_vnum(a) for a in run.logged_artifacts() if a.type == "run_table"]
    return max(vs) if vs else -1


def get_latest_table(run, table_key="val/samples.table.json", return_version=False):
    """Download the *latest* version of a run's logged val sample table as a DataFrame.

    With return_version=True, returns (df, version_int) so callers can cache the
    version and later detect a newer table via latest_table_version (no download).
    """
    arts = [a for a in run.logged_artifacts() if a.type == "run_table"]
    if not arts:
        return (None, -1) if return_version else None
    latest = max(arts, key=_table_vnum)
    df = latest.get(table_key).get_dataframe()
    return (df, _table_vnum(latest)) if return_version else df


def add_szbin(df):
    """Add the shared `szbin` (and `src` from the dataset path) columns in place; return df."""
    df["szbin"] = pd.cut(df.tgt_size, SZ_EDGES, labels=SZ_LABELS)
    if "dataset" in df.columns:
        df["src"] = df.dataset.str.split("/").str[1]
    return df
