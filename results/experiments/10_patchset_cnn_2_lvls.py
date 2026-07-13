import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import re
    import wandb
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr

    pd.set_option("display.width", 240)
    pd.set_option("display.float_format", lambda v: f"{v:.3f}")
    wandb.login()
    api = wandb.Api()

    # === CONFIGURATION ===
    PROJECT_NAME = "patch_icl_2d_exps_train"
    DATASET = "omnisynth_medseg"  # or "medsegbench"

    model_configs = {
        "medsegbench": {"patchset_cnn": "ttt6kmnk"},
        "omnisynth_medseg": {"patchset_cnn": "mp31d05x", "universeg": "08zmho80"},
    }

    runs = {k: {"wandb_name": v} for k, v in model_configs[DATASET].items()}
    print(f"Dataset: {DATASET}")
    return PROJECT_NAME, api, np, pd, plt, runs


@app.function
def get_latest_table(run, table_key_substring="val/samples.table.json"):
    """Find and download the *latest* version of a logged table artifact for a run.

    NOTE: the previous version selected `max(..., key=lambda a: a.version)`, but a.version
    is the string "v9"/"v67", so it did a LEXICAL compare where "v9" > "v67" — silently
    returning an early-epoch table. We parse the integer suffix instead.

    `vnum` is a nested local (not a separate cell/@app.function) so this stays a
    self-contained @app.function — a marimo @app.function may only reference imports and
    other top-level defs, never a name defined inside a cell."""
    import re

    def vnum(a):
        m = re.search(r"v(\d+)$", a.version if isinstance(a.version, str) else str(a.version))
        return int(m.group(1)) if m else int(a.version)

    artifacts = [a for a in run.logged_artifacts() if a.type == "run_table"]
    if not artifacts:
        return None
    latest = max(artifacts, key=vnum)   # true newest by integer version
    print(f"  latest artifact {latest.name} (v{vnum(latest)})")
    table = latest.get(table_key_substring)
    return table.get_dataframe()


@app.cell
def _(PROJECT_NAME, api, runs):
    # Load the per-sample table (patchset_cnn packs metadata into a `detail` string; empty
    # for medsegbench sources). The run crashed at epoch 67/500, so this is a partial model.
    for run_name, run_data in runs.items():
        r = api.run(f"tidiane/{PROJECT_NAME}/{run_data['wandb_name']}")
        run_data["run"] = r
        run_data["df"] = get_latest_table(r)
    dp = runs["patchset_cnn"]["df"]
    print("patchset:", dp.shape, "epoch(s):", sorted(dp["epoch"].unique()))
    print("cols:", list(dp.columns))
    return (dp,)


@app.cell
def _(dp):
    # ── Metric columns (resolution ladder 32 coarse -> 64 refine crop -> 128 native fused) ──
    #   dice_ds@32        coarse level-1 hard dice at token grid 32
    #   dice@64           refine level hard dice on its bbox crop, at Rf=64
    #   dice_fused@64     fused prob (coarse + refine-in-crop) pooled to 64, hard dice
    #   dice             FINAL native (128) hard dice on the fused stitch
    #   dice_coarse@64 / dice_coarse   coarse-only counterfactual (present only for runs eval'd
    #                                  after the evaluate.py refine-delta patch; absent here)
    C, R, F64, N = "dice_ds@32", "dice@64", "dice_fused@64", "dice"
    has_ctf = "dice_coarse" in dp.columns          # direct refine delta available?
    print("N =", len(dp), " datasets =", dp["dataset"].nunique(),
          " coarse-only counterfactual logged:", has_ctf)
    return C, F64, N, R, has_ctf


@app.cell
def _(C, F64, N, R, dp, pd):
    # ── 1. Overall means: micro (per-sample) vs macro (per-dataset, unweighted) ──
    # Micro is dominated by m2caiseg (42% of samples); macro strips that weighting out.
    cols = [C, R, F64, N]
    per = dp.groupby("dataset")[cols].mean()
    n_per = dp.groupby("dataset").size()
    micro = dp[cols].mean()
    macro = per.mean()
    summary = pd.DataFrame({"micro (n-wt)": micro, "macro (unwt)": macro})
    summary.index = ["coarse@32", "refine_crop@64", "fused@64", "native@128 (final)"]
    print(summary.to_string())
    print(f"\nm2caiseg = {n_per.get('m2caiseg', 0) / n_per.sum() * 100:.0f}% of samples")
    return (per,)


@app.cell
def _(C, F64, N, R, dp, pd):
    # ── 2. Refine can't rescue coarse misses: final outcome vs coarse@32 quality ──
    # Local copy under a new name — marimo requires `dp` to be assigned in only one cell.
    dpb = dp.copy()
    dpb["cbin"] = pd.cut(dpb[C], [-.01, 1e-6, .25, .5, .75, 1.001],
                         labels=["=0", "0-.25", ".25-.5", ".5-.75", ".75-1"])
    tbl = dpb.groupby("cbin", observed=True).agg(
        n=(N, "size"), coarse32=(C, "mean"), refine_crop64=(R, "mean"),
        fused64=(F64, "mean"), native128=(N, "mean"),
        native_zero=(N, lambda s: (s == 0).mean()))
    print("Final-prediction outcome conditioned on coarse@32 quality bucket:")
    print(tbl.to_string())
    good = dpb[dpb[C] > 0.5]
    print(f"\nWhere coarse@32>0.5 ({len(good)} samples): native==0 in "
          f"{(good[N] == 0).mean() * 100:.1f}%  (refine rarely destroys a localized object)")
    print(f"Where coarse@32==0: native==0 in "
          f"{(dpb[dpb[C] == 0][N] == 0).mean() * 100:.1f}%  (single crop can't recover a coarse miss)")
    return


@app.cell
def _(per):
    # ── 3. Stage decomposition per dataset — WHERE the refine gain is made and lost ──
    #   refine_crop64 - coarse32 : refinement encoder's local gain (mixes 32->64 res, optimistic)
    #   fused64 - refine_crop64  : STITCH COST — same 64 grid, confound-free. Single bbox is
    #                              refined; the rest of the image keeps coarse -> gain evaporates
    #                              on multi-region / multi-instance targets.
    #   native - fused64         : hi-res thresholding cost (64->128), ~free.
    d = per.copy()
    d["stitch_cost"] = d["dice_fused@64"] - d["dice@64"]
    d["hires_cost"] = d["dice"] - d["dice_fused@64"]
    big = d.sort_values("stitch_cost")
    show = big[["dice_ds@32", "dice@64", "dice_fused@64", "dice", "stitch_cost", "hires_cost"]]
    show.columns = ["coarse32", "refine64", "fused64", "native", "stitch_cost", "hires_cost"]
    print(show.to_string())
    print(f"\nMACRO stitch cost {show['stitch_cost'].mean():+.3f}   "
          f"hires cost {show['hires_cost'].mean():+.3f}")
    print(f"Net native vs coarse@32: {(per['dice'] - per['dice_ds@32']).mean():+.3f} "
          f"(NB: coarse@32 hard-dice is inflated by the coarser grid)")
    return


@app.cell
def _(dp, has_ctf):
    # ── 4. DIRECT refine delta (only when the coarse-only counterfactual was logged) ──
    # After the evaluate.py patch, dice_coarse (native) and dice_coarse@64 are logged, so the
    # exact refine contribution is  fused - coarse  at a matched resolution, per sample.
    if has_ctf:
        dd = dp.copy()
        dd["refine_delta_native"] = dd["dice"] - dd["dice_coarse"]
        dd["refine_delta@64"] = dd["dice_fused@64"] - dd["dice_coarse@64"]
        agg = dd.groupby("dataset").agg(
            n=("dice", "size"),
            coarse_nat=("dice_coarse", "mean"), fused_nat=("dice", "mean"),
            delta_nat=("refine_delta_native", "mean"),
            help=("refine_delta_native", lambda s: (s > 1e-3).mean()),
            hurt=("refine_delta_native", lambda s: (s < -1e-3).mean()))
        print(agg.sort_values("delta_nat").to_string())
        print(f"\nMACRO native refine delta: {agg['delta_nat'].mean():+.3f}")
    else:
        print("dice_coarse column absent — this run predates the evaluate.py refine-delta patch.")
        print("Re-run eval to get per-sample  fused - coarse  at matched resolution.")
    return


@app.cell
def _(dp, runs):
    # ── 5. Head-to-head vs universeg on the same val samples (native `dice`) ──
    # Needs a `universeg` run in the DATASET config (omnisynth_medseg has one). patchset_cnn
    # here crashed @174 while universeg is still training @308 — a fairness caveat, yet
    # patchset still edges ahead on the mean.
    _du = runs.get("universeg", {}).get("df")
    if _du is None:
        print("no universeg run in this DATASET config — head-to-head skipped.")
        paired = None
    else:
        _key = ["dataset", "sample_idx", "label"]
        paired = dp[_key + ["dice"]].merge(_du[_key + ["dice"]], on=_key,
                                           suffixes=("_ps", "_uv"))
        paired["src"] = paired["dataset"].str.split("/").str[1]   # omniglot/<src>/label_N -> src
        _sm = paired.groupby("src")[["dice_ps", "dice_uv"]].mean()
        _w = (paired.dice_ps > paired.dice_uv + 1e-3).mean()
        _l = (paired.dice_uv > paired.dice_ps + 1e-3).mean()
        print(f"paired {len(paired)} samples over {_sm.shape[0]} sources")
        print(f"MICRO      ps {paired.dice_ps.mean():.3f}  uv {paired.dice_uv.mean():.3f}  "
              f"Δ {paired.dice_ps.mean() - paired.dice_uv.mean():+.3f}")
        print(f"MACRO-src  ps {_sm.dice_ps.mean():.3f}  uv {_sm.dice_uv.mean():.3f}  "
              f"Δ {_sm.dice_ps.mean() - _sm.dice_uv.mean():+.3f}")
        print(f"win rate   ps {_w * 100:.0f}%   uv {_l * 100:.0f}%   tie {(1 - _w - _l) * 100:.0f}%")
    return (paired,)


@app.cell
def _(paired, plt):
    # ── plot A: per-source scatter (patchset vs universeg) + sorted Δ bars ──
    if paired is None:
        _fig = None
    else:
        _sm = paired.groupby("src")[["dice_ps", "dice_uv"]].mean()
        _sm["n"] = paired.groupby("src").size()
        _sm["delta"] = _sm.dice_ps - _sm.dice_uv
        _sm = _sm.sort_values("delta")
        _clr = ["tab:green" if d > 0 else "tab:red" for d in _sm.delta]
        _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(13, 6))
        _ax1.scatter(_sm.dice_uv, _sm.dice_ps, s=_sm.n * 0.7, alpha=0.55, c=_clr)
        for _s, _r in _sm.iterrows():
            _ax1.annotate(_s, (_r.dice_uv, _r.dice_ps), fontsize=6, alpha=0.75)
        _ax1.plot([0, 1], [0, 1], "k--", lw=0.8)
        _ax1.set(xlim=(0, 1), ylim=(0, 1), xlabel="universeg native dice",
                 ylabel="patchset_cnn native dice",
                 title="per-source (above diagonal = patchset wins; size ∝ n)")
        _ax2.barh(range(len(_sm)), _sm.delta, color=_clr)
        _ax2.set_yticks(range(len(_sm)))
        _ax2.set_yticklabels(_sm.index, fontsize=6)
        _ax2.axvline(0, color="k", lw=0.8)
        _ax2.set(xlabel="Δ dice (patchset − universeg)",
                 title="green = patchset better · red = universeg better")
        _fig.tight_layout()
    _fig


@app.cell
def _(C, F64, N, R, dp, np, plt):
    # ── plot B: patchset_cnn refine ladder per source — where the refine gain is lost ──
    # coarse@32 -> refine_crop@64 (encoder gain) -> fused@64 (single-bbox STITCH cost, shaded)
    # -> native@128 (final). Net native ~= coarse: the stitch eats the refinement.
    _d = dp.copy()
    _d["src"] = _d["dataset"].str.split("/").str[1]
    _per = _d.groupby("src")[[C, R, F64, N]].mean().sort_values(N)
    _x = np.arange(len(_per))
    _fig, _ax = plt.subplots(figsize=(12, 5.5))
    _ax.fill_between(_x, _per[R], _per[F64], color="red", alpha=0.12,
                     label="stitch cost (refine → fused)")
    for _col, _lab in [(C, "coarse@32"), (R, "refine_crop@64"),
                       (F64, "fused@64"), (N, "native@128 (final)")]:
        _ax.plot(_x, _per[_col], "-o", ms=3, label=_lab)
    _ax.set_xticks(_x)
    _ax.set_xticklabels(_per.index, rotation=90, fontsize=6)
    _ax.set(ylabel="hard dice",
            title="patchset_cnn refine ladder per source — refine gain lost at the stitch")
    _ax.legend(fontsize=8)
    _fig.tight_layout()
    _fig


if __name__ == "__main__":
    app.run()
