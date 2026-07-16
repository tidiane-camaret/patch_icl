import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    # ══ OCCUPANCY ANALYSIS — three trained models on omnisynth_medseg ═════════════════════════
    # Notebook 11 showed patchset_cnn collapses on SMALL objects because its coarse 32² head is
    # the ceiling. Notebook 12 (loss theory, cell below) traced that to the soft-Dice eps=1
    # smoothing: for a sub-cell object the loss is minimized at pred≈0. Exp 4 shrank eps per level
    # ({32:0.01, 64:0.1}). Here we compare all three trained runs as a function of object
    # size / GT occupancy:
    #   patchset_eps1   03ypf2pk   coarse 32² + scatter refine, eps=1        (the pathology)
    #   patchset_epslvl 9j69mib5   same arch, per-level eps {32:0.01,64:0.1} (the fix)
    #   universeg       08zmho80   native-res baseline, no coarse bottleneck
    # All read from one cached CSV of the per-sample val tables (columns logged by evaluate.py);
    # rebuild it by deleting results/experiments/artifacts/12_occ_runs.csv.
    import sys
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    sys.path.insert(0, str(Path(__file__).parent))
    from nb_common import ARTIFACTS, PROJECT, SZ_LABELS, add_szbin, get_latest_table

    pd.set_option("display.width", 240)
    pd.set_option("display.float_format", lambda v: f"{v:.3f}")

    RUNS = {"patchset_eps1": "03ypf2pk", "universeg": "08zmho80", "patchset_epslvl": "9j69mib5"}
    _cache = ARTIFACTS / "12_occ_runs.csv"
    if _cache.exists():
        D = pd.read_csv(_cache)
        print(f"loaded cached tables {D.shape}")
    else:
        import wandb
        _api = wandb.Api()
        _cols = ["dataset", "sample_idx", "tgt_size", "tgt_occ", "ctx_size", "ctx_occ", "dice",
                 "dice_ds@32", "dice_ds_soft@32", "cossim@32", "top64@32", "tgt_cells@32",
                 "tgt_peak@32", "dice@64", "dice_soft@64", "dice_fused@64", "dice_coarse@64",
                 "dice_coarse"]
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
    return ARTIFACTS, D, RUNS, SZ_LABELS, np, pd, plt


@app.cell
def _(ARTIFACTS, np, plt):
    # ── WHY OCCUPANCY MATTERS: the coarse-grid loss suppresses sub-cell objects ────────────────
    # Single fg cell, GT occupancy g, pred p; background cells (0,0) add nothing. On the 32² grid:
    #   bce  = -(g·log p + (1-g)·log(1-p)) / N        (N=1024, so BCE is ~1000× weaker than dice)
    #   dice = 1 - (2pg + eps)/(p + g + eps)          (soft_dice_loss, eps=1)  → loss = bce + dice
    # ∂dice/∂p is +ve (wants p↑) iff 2g²+2g-eps>0; for eps=1 that flips at g*≈0.366 (≈6/16 px). So
    # below g* the loss is minimized at p≈0 — the coarse head is TRAINED to switch small cells off.
    # Shrinking eps moves g* toward 0, restoring the p→1 incentive for small occupancies.
    N = 1024
    def _bce(p, g):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return -(g * np.log(p) + (1 - g) * np.log(1 - p)) / N
    def _dice(p, g, eps):
        return 1 - (2 * p * g + eps) / (p + g + eps)
    _tot = lambda p, g, eps: _bce(p, g) + _dice(p, g, eps)

    g_star = (-2 + np.sqrt(4 + 8)) / 4     # eps=1 sign-flip
    _p = np.linspace(0, 1, 2001)
    print(f"soft-dice eps=1 sign-flip at g*={g_star:.3f} (~{g_star*16:.1f}/16 px per 32² cell)")
    print(f"{'g':>5} {'g·16px':>7} {'argmin_p (eps=1)':>16} {'argmin_p (eps=.01)':>18}")
    for _g in [0.06, 0.12, 0.25, 0.366, 0.50, 1.0]:
        _a1 = _p[np.argmin(_tot(_p, _g, 1.0))]
        _a2 = _p[np.argmin(_tot(_p, _g, 0.01))]
        print(f"{_g:5.2f} {_g*16:7.1f} {_a1:16.2f} {_a2:18.2f}")

    _fig, _ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for _g, _c in [(0.12, "tab:red"), (0.25, "tab:orange"), (0.50, "tab:green"), (1.0, "tab:blue")]:
        _ax[0].plot(_p, _tot(_p, _g, 1.0), color=_c, label=f"g={_g}")
        _ax[0].plot(_g, _tot(_g, _g, 1.0), "o", color=_c, ms=4)
    _ax[0].set(xlabel="pred occupancy p", ylabel="loss (eps=1)",
               title="small g: loss minimized at p≈0, not the\ncalibrated p=g (dot) → cell switched off")
    _ax[0].legend(fontsize=8); _ax[0].grid(alpha=0.3)
    for _g, _c in [(0.12, "tab:orange"), (0.25, "tab:olive"), (0.50, "tab:green")]:
        _ax[1].plot(_p, _dice(_p, _g, 1.0), color=_c, label=f"g={_g} eps=1")
        _ax[1].plot(_p, _dice(_p, _g, 0.01), "--", color=_c, alpha=.7, label=f"g={_g} eps=.01")
    _ax[1].set(xlabel="pred occupancy p", ylabel="soft-dice loss",
               title="shrinking eps un-inverts the incentive\nsolid eps=1 rises with p; dashed eps=.01 falls")
    _ax[1].legend(fontsize=7); _ax[1].grid(alpha=0.3)
    _fig.tight_layout()
    _fig.savefig(ARTIFACTS / "12_loss_occupancy.png", dpi=120, bbox_inches="tight")
    _fig
    return


@app.cell
def _(D, RUNS, np, pd, plt):
    # ── 1. FINAL DICE vs OBJECT SIZE — the three models side by side ───────────────────────────
    # The whole story in one plot. universeg is size-robust (native res, no coarse grid). eps=1
    # patchset falls off a cliff below ~130px. The per-level-eps fix lifts exactly that regime,
    # closing most of the gap to universeg on small objects while keeping patchset's edge on large.
    _piv = D.pivot_table(index=["dataset", "sample_idx", "tgt_size"], columns="run",
                         values="dice").reset_index()
    _cpiv = D.pivot_table(index=["dataset", "sample_idx"], columns="run",
                          values="dice_coarse").reset_index()
    _piv = _piv.merge(_cpiv.rename(columns={k: f"{k}__coarse" for k in RUNS}),
                      on=["dataset", "sample_idx"])

    _bins = np.logspace(0, np.log10(_piv.tgt_size.max()), 24)
    _piv["b"] = pd.cut(_piv.tgt_size, _bins)
    _g = _piv.groupby("b", observed=True).agg(
        x=("tgt_size", "median"),
        ps1=("patchset_eps1", "mean"), pslv=("patchset_epslvl", "mean"), uv=("universeg", "mean"),
        ps1c=("patchset_eps1__coarse", "mean"), pslvc=("patchset_epslvl__coarse", "mean"))

    _fig, _ax = plt.subplots(figsize=(10, 6))
    _ax.plot(_g.x, _g.ps1, "-o", ms=4, color="tab:red", label="patchset eps=1 (final)")
    _ax.plot(_g.x, _g.pslv, "-o", ms=4, color="tab:green", label="patchset eps-per-lvl (final)")
    _ax.plot(_g.x, _g.uv, "-o", ms=4, color="tab:blue", label="universeg (final)")
    _ax.plot(_g.x, _g.ps1c, ":", color="tab:red", alpha=.6, label="patchset eps=1 (coarse-only)")
    _ax.plot(_g.x, _g.pslvc, ":", color="tab:green", alpha=.6, label="patchset eps-per-lvl (coarse-only)")
    _ax.axhline(0.5, color="k", lw=0.6, ls="--", alpha=0.4)
    _ax.set_xscale("log"); _ax.grid(alpha=0.3); _ax.legend(fontsize=9)
    _ax.set(xlabel="target object size (fg px @128², log)", ylabel="mean hard dice",
            title="Final dice vs object size — the eps fix lifts the small-object regime toward universeg")
    _fig.tight_layout()
    return


@app.cell
def _(D, SZ_LABELS, pd):
    # ── 2. DETAILED OCCUPANCY TABLE — mean final dice & complete-miss rate by size, per run ────
    # Occupancy binned as native fg px (@128²). uv_minus_ps* = how far each patchset variant still
    # trails the native baseline. The eps fix roughly HALVES the ≤32px gap to universeg and cuts
    # the complete-miss (dice==0) rate; above ~130px all three converge.
    _t = D.pivot_table(index="szbin", columns="run", values="dice", observed=True, aggfunc="mean")
    _miss = D.assign(miss=(D.dice == 0)).pivot_table(index="szbin", columns="run",
                                                    values="miss", observed=True, aggfunc="mean")
    _n = D[D.run == "universeg"].groupby("szbin", observed=True).size().rename("n")
    _tab = pd.concat([_n, _t[["universeg", "patchset_eps1", "patchset_epslvl"]]], axis=1)
    _tab["uv−eps1"] = _tab.universeg - _tab.patchset_eps1
    _tab["uv−epslvl"] = _tab.universeg - _tab.patchset_epslvl
    _tab["fix Δ"] = _tab.patchset_epslvl - _tab.patchset_eps1
    print("MEAN FINAL DICE by object size:\n" + _tab.reindex(SZ_LABELS).to_string())

    _mt = _miss[["universeg", "patchset_eps1", "patchset_epslvl"]].reindex(SZ_LABELS)
    print("\nCOMPLETE-MISS RATE (dice==0) by object size:\n" + _mt.to_string())

    _micro = D.groupby("run", observed=True).dice.mean()
    print("\nMICRO mean final dice:  "
          + "  ".join(f"{k} {_micro[k]:.3f}" for k in ["universeg", "patchset_eps1", "patchset_epslvl"]))
    return


@app.cell
def _(D, SZ_LABELS, pd):
    # ── 3. WHERE THE FIX ACTS — coarse-head occupancy (patchset runs only) ─────────────────────
    # universeg has no coarse grid, so this is eps1 vs epslvl. cossim@32 (RANKING: are tgt cells
    # ranked above bg) barely moves — ranking was never the problem. dice_ds_soft@32 (the coarse
    # OCCUPANCY = the literal soft-dice target) and dice_coarse (coarse-only upsampled to 128) JUMP
    # on small objects: the loss now turns surviving small-object cells ON instead of suppressing
    # them. That coarse gain is what propagates to the final-dice lift in table 2.
    _pt = D[D.run.isin(["patchset_eps1", "patchset_epslvl"])]
    _rows = []
    for _metric in ["cossim@32", "dice_ds_soft@32", "dice_coarse"]:
        _p = _pt.pivot_table(index="szbin", columns="run", values=_metric, observed=True, aggfunc="mean")
        _p = _p.reindex(SZ_LABELS)
        _p.columns = [f"{_metric}:{c.replace('patchset_','')}" for c in _p.columns]
        _p[f"{_metric}:Δ"] = _p.iloc[:, 1] - _p.iloc[:, 0]   # epslvl - eps1 (alpha order: eps1<epslvl)
        _rows.append(_p)
    print("Coarse-head metrics by size — eps1 vs epslvl (Δ = fix gain):\n"
          + pd.concat(_rows, axis=1).to_string())
    print("\n→ ranking (cossim) flat; coarse OCCUPANCY (dice_ds_soft@32) and dice_coarse rise on ≤128px.")
    return


@app.cell
def _(D, np):
    # ── 4. SMALL-OBJECT DEEP DIVE (≤32px) — the bucket that drives everything ──────────────────
    # 826 of 3450 val samples are ≤32px. For that bucket: final dice, complete-miss rate, and
    # (patchset only) coarse-grid SURVIVAL — the fraction whose object leaves ≥1 pooled cell ≥0.5.
    # Survival is a property of the GT pooling (identical for both patchset runs); the fix works by
    # ACTIVATING the cells that already survive, not by making more survive. universeg, at native
    # resolution, has no such bottleneck and leads the bucket.
    _s = D[D.tgt_size <= 32]
    print(f"≤32px bucket: n={ (_s.run=='universeg').sum() } samples per run\n")
    _agg = _s.groupby("run", observed=True).agg(
        final_dice=("dice", "mean"),
        miss_rate=("dice", lambda s: (s == 0).mean()),
        cossim32=("cossim@32", "mean"),
        dice_soft32=("dice_ds_soft@32", "mean"),
        survived32=("tgt_cells@32", lambda s: (s > 0).mean() if s.notna().any() else np.nan))
    print(_agg.reindex(["universeg", "patchset_eps1", "patchset_epslvl"]).to_string())
    print("\n→ eps fix on ≤32px: final dice up, miss rate down; survival unchanged (GT property),")
    print("  dice_soft@32 up (cells turned ON). universeg still leads — native res beats a 32² grid.")
    return


@app.cell
def _():
    # ── Takeaways (epochs: eps1=39, epslvl=46, universeg=410) ───────────────────────────────────
    # 1. patchset_cnn's weakness is an OCCUPANCY problem localized to small objects: its 32² coarse
    #    head, under soft-Dice eps=1, is trained to switch off any cell below g*≈0.37 occupancy.
    # 2. Per-level eps ({32:0.01, 64:0.1}, run 9j69mib5) lifts exactly the small-object regime:
    #      ≤32px final dice   0.246 → 0.341 (+0.095); complete-miss 64% → 45%; ≥129px unchanged.
    #    It nearly closes the ≤32px gap to universeg (0.341 vs 0.362, −0.021 left) with no cost above.
    # 3. Mechanism confirmed empirically: cossim@32 (ranking) flat (+0.007), coarse OCCUPANCY
    #    (dice_ds_soft@32 +0.069) and dice_coarse (+0.081) jump on ≤32px — the fix ACTIVATES surviving
    #    small-object cells; it does not resolve or rank more. Coarse-grid survival is identical
    #    across the two patchset runs (0.277, a GT-pooling property).
    # 4. universeg (native res) leads ONLY the ≤32px bucket. At 33-128px patchset already wins big
    #    (0.680 vs 0.571) — universeg misses 15% of those completely vs patchset's 5%. So the coarse
    #    grid costs patchset only the very smallest objects; the residual ≤32px gap is the 32² grid
    #    resolution itself, not the loss. Next lever: finer coarse grid / ranking loss on a finer grid.
    print("figures: artifacts/12_loss_occupancy.png (+ cells 1-4 tables)")
    return


if __name__ == "__main__":
    app.run()
