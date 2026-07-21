import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():

    # rebuild it by deleting results/experiments/artifacts/13_occ_runs.csv.
    import sys
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    sys.path.insert(0, str(Path(__file__).parent))
    from nb_common import ARTIFACTS, PROJECT, SZ_LABELS, add_szbin, get_latest_table

    pd.set_option("display.width", 240)
    pd.set_option("display.float_format", lambda v: f"{v:.3f}")

    # Two full-res-decode runs to compare. Both are SINGLE-LEVEL (mask_patch_decode_size=4, no
    # coarse grid / refine level), so only the final `dice` (+ GT size/occupancy) is logged; the
    # coarse-head / @32 / @64 diagnostic columns below are fetched for forward-compat but are NaN
    # for these runs.
    RUNS = {"patchset": "45me4tdi", "universeg": "08zmho80"}
    _cache = ARTIFACTS / "13_occ_runs.csv"
    if _cache.exists():
        D = pd.read_csv(_cache)
        print(f"loaded cached tables {D.shape}")
    else:
        import wandb
        _api = wandb.Api()
        _cols = ["dataset", "sample_idx", "tgt_size", "tgt_occ", "ctx_size", "ctx_occ", "dice"]
        _rows = []
        for _tag, _rid in RUNS.items():
            _r = _api.run(f"{PROJECT}/{_rid}")
            _d = get_latest_table(_r)
            _d = _d[_d.epoch == _d.epoch.max()].copy()
            for _c in _cols:
                if _c not in _d:
                    _d[_c] = pd.NA
            _d = _d[_cols].copy()
            _d.insert(0, "run", _tag); _d.insert(1, "run_id", _rid); _d.insert(2, "epoch", int(_r.summary.get("epoch", -1)))
            _rows.append(_d)
        D = pd.concat(_rows, ignore_index=True)
        D.to_csv(_cache, index=False)
        print(f"rebuilt tables {D.shape}")

    # universeg's val table predates the GT size/occupancy columns (all-NaN there). GT size is
    # model-independent (same deterministic val samples), so backfill it from whichever run logged
    # it, keyed on (dataset, sample_idx).
    _szc = ["tgt_size", "tgt_occ", "ctx_size", "ctx_occ"]
    _ref = (D[D.tgt_size.notna()].drop_duplicates(["dataset", "sample_idx"])
            [["dataset", "sample_idx"] + _szc])
    D = D.drop(columns=_szc).merge(_ref, on=["dataset", "sample_idx"], how="left")

    add_szbin(D)
    print("epochs:", D.groupby("run", observed=True).epoch.max().to_dict())
    print("runs:", list(RUNS))
    return D, SZ_LABELS, np, pd, plt


@app.cell
def _(D, np, pd, plt):
    # ── 1. FINAL DICE vs OBJECT SIZE — patchset vs universeg ───────────────────────────────────
    # The whole story in one plot. universeg is size-robust (native res). patchset's full-res
    # decode falls off on small objects where its 32² token grid loses them.
    _piv = D.pivot_table(index=["dataset", "sample_idx", "tgt_size"], columns="run",
                         values="dice").reset_index()

    # Guard the log-bins: drop NaN/<1 sizes and require a real spread, else logspace(0, log10(max))
    # is non-monotonic (max NaN → NaN bins; max<=1 → flat/descending) and pd.cut raises.
    _piv = _piv[_piv.tgt_size.notna() & (_piv.tgt_size >= 1)].copy()
    assert not _piv.empty and _piv.tgt_size.max() > 1, (
        "tgt_size not populated — GT-size backfill failed (both runs missing it?); "
        "delete artifacts/13_occ_runs.csv and rebuild")
    _bins = np.logspace(0, np.log10(_piv.tgt_size.max()), 24)
    _piv["b"] = pd.cut(_piv.tgt_size, _bins, include_lowest=True)
    _g = _piv.groupby("b", observed=True).agg(
        x=("tgt_size", "median"),
        ps=("patchset", "mean"), uv=("universeg", "mean"))

    _fig, _ax = plt.subplots(figsize=(10, 6))
    _ax.plot(_g.x, _g.ps, "-o", ms=4, color="tab:red", label="patchset (final)")
    _ax.plot(_g.x, _g.uv, "-o", ms=4, color="tab:blue", label="universeg (final)")
    _ax.axhline(0.5, color="k", lw=0.6, ls="--", alpha=0.4)
    _ax.set_xscale("log"); _ax.grid(alpha=0.3); _ax.legend(fontsize=9)
    _ax.set(xlabel="target object size (fg px @128², log)", ylabel="mean hard dice",
            title="Final dice vs object size — patchset vs universeg")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(D, SZ_LABELS, pd):
    # ── 2. DETAILED DICE TABLE — mean final dice & complete-miss rate by size ───────────────────
    # Occupancy binned as native fg px (@128²). uv−ps = how far patchset trails the native baseline.
    # patchset trails most on the smallest objects; above ~130px the two converge.
    _t = D.pivot_table(index="szbin", columns="run", values="dice", observed=True, aggfunc="mean")
    _miss = D.assign(miss=(D.dice == 0)).pivot_table(index="szbin", columns="run",
                                                    values="miss", observed=True, aggfunc="mean")
    _n = D[D.run == "universeg"].groupby("szbin", observed=True).size().rename("n")
    _tab = pd.concat([_n, _t[["universeg", "patchset"]]], axis=1)
    _tab["uv−ps"] = _tab.universeg - _tab.patchset
    print("MEAN FINAL DICE by object size:\n" + _tab.reindex(SZ_LABELS).to_string())

    _mt = _miss[["universeg", "patchset"]].reindex(SZ_LABELS)
    print("\nCOMPLETE-MISS RATE (dice==0) by object size:\n" + _mt.to_string())

    _micro = D.groupby("run", observed=True).dice.mean()
    print("\nMICRO mean final dice:  "
          + "  ".join(f"{k} {_micro[k]:.3f}" for k in ["universeg", "patchset"]))
    return


@app.cell
def _(D, pd):
    # ── 3. PER-DATASET BREAKDOWN — where each model wins ────────────────────────────────────────
    # Mean final dice per source dataset, patchset vs universeg (uses only `dice`, the one metric
    # both single-level runs log). Sorted by the gap so the datasets driving the overall difference
    # are on top.
    _t = D.pivot_table(index="dataset", columns="run", values="dice", observed=True, aggfunc="mean")
    _n = D[D.run == "universeg"].groupby("dataset", observed=True).size().rename("n")
    _tab = pd.concat([_n, _t[["universeg", "patchset"]]], axis=1)
    _tab["ps−uv"] = _tab.patchset - _tab.universeg
    print("MEAN FINAL DICE by dataset (sorted by patchset−universeg):\n"
          + _tab.sort_values("ps−uv", ascending=False).to_string())
    return


@app.cell
def _(D):
    # ── 4. SMALL-OBJECT DEEP DIVE (≤32px) — the historically weak bucket ────────────────────────
    # For the ≤32px bucket: final dice and complete-miss rate. This is where patchset's coarse token
    # grid used to lose small objects; the full-res decode (mask_patch_decode_size=4) is meant to
    # recover them. Read the two rows to see whether it closes the gap to universeg's native res.
    _s = D[D.tgt_size <= 32]
    print(f"≤32px bucket: n={ (_s.run=='universeg').sum() } samples per run\n")
    _agg = _s.groupby("run", observed=True).agg(
        final_dice=("dice", "mean"),
        miss_rate=("dice", lambda s: (s == 0).mean()))
    print(_agg.reindex(["universeg", "patchset"]).to_string())
    return


@app.cell
def _():
    # ── Takeaways — patchset vs universeg ───────────────────────────────────────────────────────
    # Comparison of the two full-res-decode runs on final dice (the one metric both single-level runs
    # log): overall (cell 2 micro), by object size (cells 1-2), by dataset (cell 3), and on the
    # historically weak ≤32px bucket (cell 4). On this cache patchset leads on micro mean and holds
    # up on small objects — the full-res decode (mask_patch_decode_size=4) recovers the small-object
    # regime the coarse token grid used to lose. Read the printed tables for the current numbers.
    print("figures: artifacts/ (cells 1-4 tables)")
    return


if __name__ == "__main__":
    app.run()
