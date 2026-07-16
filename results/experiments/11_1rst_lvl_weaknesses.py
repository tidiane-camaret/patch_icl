import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import re
    import wandb
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr

    pd.set_option("display.width", 240)
    pd.set_option("display.float_format", lambda v: f"{v:.3f}")
    wandb.login()
    api = wandb.Api()

    # === CONFIGURATION ===
    PROJECT_NAME = "patch_icl_2d_exps_train"
    DATASET = "omnisynth_medseg"

    # patchset_cnn = SCATTER-refine (T=32 coarse, Rf=64 fine), re-eval'd on the trained checkpoint
    # so its per-sample table carries the NEW columns logged by evaluate.py:
    #   ranking:   cossim@32, top64@32        (the coarse pred's real job — rank tgt cells high)
    #   survival:  tgt_cells@32, tgt_peak@32  (does the object survive avg-pool to the 32² grid)
    #   GT size:   tgt_size/tgt_occ, ctx_size/ctx_occ   (native fg px / fraction, model-independent)
    #   ladder:    dice_ds@32, dice_coarse(@64), dice_fused@64, dice (native@128 = final)
    # universeg is native-res (no coarse grid); its table predates the new columns, so we borrow
    # tgt_size from the patchset table (GT size is model-independent, same val samples).
    model_configs = {
        "omnisynth_medseg": {"patchset_cnn": "03ypf2pk", "universeg": "08zmho80"},
    }

    runs = {k: {"wandb_name": v} for k, v in model_configs[DATASET].items()}
    print(f"Dataset: {DATASET}")
    return PROJECT_NAME, api, np, pd, plt, runs, spearmanr


@app.cell
def _():
    # get_latest_table now lives in the shared results/experiments/nb_common.py.
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).parent))
    from nb_common import get_latest_table
    return (get_latest_table,)


@app.cell
def _(PROJECT_NAME, api, get_latest_table, pd, runs):
    # Load both per-sample tables. All size/ranking/survival columns are now IN the patchset
    # table — no deterministic-val regeneration needed anymore.
    for run_name, run_data in runs.items():
        r = api.run(f"tidiane/{PROJECT_NAME}/{run_data['wandb_name']}")
        run_data["run"] = r
        run_data["df"] = get_latest_table(r)

    dp = runs["patchset_cnn"]["df"].copy()
    du = runs["universeg"]["df"].copy()
    # Shared size bucketing on native foreground px (@128²).
    SZ_EDGES = [0, 32, 128, 512, 2048, 1e9]
    SZ_LABELS = ["≤32", "33-128", "129-512", "513-2048", ">2048"]
    dp["szbin"] = pd.cut(dp.tgt_size, SZ_EDGES, labels=SZ_LABELS)
    dp["src"] = dp.dataset.str.split("/").str[1]
    print("patchset:", dp.shape, "| epochs", sorted(dp.epoch.unique()))
    print("cols:", [c for c in dp.columns if c != "detail"])
    print("universeg:", du.shape)
    return SZ_EDGES, SZ_LABELS, dp, du


@app.cell
def _(dp, plt):
    # ── 1. THE COARSE LADDER BY OBJECT SIZE ────────────────────────────────────────────────
    # The coarse pred's job is RANKING (put target cells above background), not exact occupancy.
    #   ranking:   cossim@32 (scale-invariant, whole-distribution) ; top64@32 (recall of GT cells
    #              into the top-64 pred cells — a SPARSE-target metric, degenerate for big objects)
    #   occupancy: dice_ds@32 (hard) / dice_ds_soft@32 (soft, = what the BCE+soft-dice loss fits)
    #   survival:  tgt_peak@32 (best cell's pooled GT) ; empty32 = object left NO cell ≥0.5
    #   outcome:   dice_coarse (coarse-only @128) -> dice (final native@128)
    # Every coarse signal degrades together toward small objects; the object is under-resolved at
    # the 32² grid, so even ranking (cossim) falls off — not just the thresholded occupancy.
    lad = dp.groupby("szbin", observed=True).agg(
        n=("dice", "size"), tgt_peak32=("tgt_peak@32", "median"),
        empty32=("tgt_cells@32", lambda s: (s == 0).mean()),
        cossim=("cossim@32", "mean"), top64=("top64@32", "mean"),
        dice_soft=("dice_ds_soft@32", "mean"),
        coarse_nat=("dice_coarse", "mean"), final=("dice", "mean"))
    print("Coarse-level quality by target size (fg px @128²):\n" + lad.to_string())

    _fig, _ax = plt.subplots(figsize=(9.5, 5.5))
    _x = range(len(lad))
    for _c, _lab, _st in [("cossim", "cossim@32 (ranking)", "-o"),
                          ("dice_soft", "dice_soft@32 (occupancy, = loss target)", "-s"),
                          ("coarse_nat", "dice_coarse (coarse-only @128)", "-^"),
                          ("final", "dice (final @128)", "-D")]:
        _ax.plot(_x, lad[_c], _st, ms=5, label=_lab)
    _ax.plot(_x, lad.tgt_peak32, ":x", color="gray", label="tgt_peak@32 (best-cell occupancy)")
    _ax.axhline(0.5, color="k", lw=0.6, ls="--", alpha=0.5)
    _ax.set_xticks(list(_x)); _ax.set_xticklabels(lad.index)
    _ax.set(xlabel="target object size (fg px @128²)", ylabel="metric",
            title="Coarse ranking AND occupancy both collapse on small objects")
    _ax.legend(fontsize=8); _ax.grid(alpha=0.3)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(dp, spearmanr):
    # ── 2. WHAT DETERMINES THE FINAL DICE? (Spearman ρ vs native `dice`) ────────────────────
    # Coarse quality all but determines the outcome. cossim@32 (ranking) and dice_ds_soft@32 /
    # dice_coarse (occupancy) are the top predictors; top64@32 is near-useless as a scalar because
    # it saturates/degenerates on large objects. Object size and context size track each other.
    cols = ["dice_coarse", "dice_ds_soft@32", "cossim@32", "tgt_peak@32", "dice_ds@32",
            "tgt_occ", "tgt_size", "ctx_occ", "ctx_size", "top64@32"]
    print("Spearman ρ  (feature vs final native dice):")
    for _c in cols:
        print(f"  {_c:18s} {spearmanr(dp[_c], dp['dice'], nan_policy='omit').correlation:+.3f}")
    print("\n→ final dice is set by COARSE quality (ρ≈0.93-0.98); cossim is the best-behaved coarse")
    print("  ranking metric; top64 (ρ≈0.11) is degenerate on large objects — don't read it alone.")
    return


@app.cell
def _(dp, plt):
    # ── 3. DOES THE COARSE PRED STILL *RANK* SMALL OBJECTS IT CAN'T SEGMENT? ─────────────────
    # Reframe on ≤32px objects, split by whether they leave ANY coarse cell (tgt_cells@32>0).
    # If they SURVIVE pooling, the model ranks them (cossim 0.74) and refine recovers ~0.58 dice.
    # If they're LOST at pooling (72% of ≤32px), the soft training target is ~flat, so there is
    # nothing to push those cells up: cossim falls to 0.37 and 81% end at dice==0. The ranking
    # objective can't help where the pooled target carries no gradient — the ceiling is the 32²
    # target grid, not the loss form.
    _sub = dp[dp.tgt_size <= 32].copy()
    _sub["survived"] = _sub["tgt_cells@32"] > 0
    _t = _sub.groupby("survived").agg(
        n=("dice", "size"), tgt_peak32=("tgt_peak@32", "median"),
        cossim=("cossim@32", "mean"), top64=("top64@32", "mean"),
        dice_soft=("dice_ds_soft@32", "mean"),
        coarse_nat=("dice_coarse", "mean"), final=("dice", "mean"),
        final_is0=("dice", lambda s: (s == 0).mean()))
    _t.index = ["LOST at pooling (0 cells)", "survived (≥1 cell)"]
    print("≤32px objects, split by coarse-grid survival:\n" + _t.to_string())

    _fig, _ax = plt.subplots(figsize=(8, 5))
    _c = _sub["survived"].map({True: "tab:green", False: "tab:red"})
    _ax.scatter(_sub["tgt_peak@32"], _sub["cossim@32"], s=14, alpha=0.5, c=_c)
    _ax.axvline(0.5, color="k", lw=0.7, ls="--", alpha=0.6)
    _ax.set(xlabel="tgt_peak@32 (best coarse cell occupancy)", ylabel="cossim@32 (coarse ranking)",
            xlim=(0, 1.02), ylim=(-0.02, 1.02),
            title="≤32px: ranking works only once a cell crosses ~0.5 occupancy\n"
                  "red = lost at pooling (cossim collapses) · green = survived")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(dp, du, np, pd, plt, spearmanr):
    # ── 4. vs UNIVERSEG (native-res, no coarse grid) BY OBJECT SIZE ──────────────────────────
    # Borrow tgt_size from the patchset table (model-independent GT). Clean CROSSOVER: universeg
    # wins ONLY on the smallest objects (≤32px, where the 32² coarse grid under-resolves them);
    # patchset wins everywhere else. The overall ~tie is entirely the ≤32 bucket.
    _k = ["dataset", "sample_idx"]
    mu = (dp[_k + ["dice", "dice_coarse", "tgt_size", "szbin"]]
          .rename(columns={"dice": "ps", "dice_coarse": "ps_coarse"})
          .merge(du[_k + ["dice"]].rename(columns={"dice": "uv"}), on=_k))
    _t = mu.groupby("szbin", observed=True).agg(
        n=("uv", "size"), ps_coarse=("ps_coarse", "mean"), ps=("ps", "mean"), uv=("uv", "mean"),
        uv_minus_ps=("uv", lambda s: (s - mu.loc[s.index, "ps"]).mean()),
        uv_is0=("uv", lambda s: (s == 0).mean()), ps_is0=("ps", lambda s: (s == 0).mean()))
    print(f"MICRO  ps {mu.ps.mean():.3f}  uv {mu.uv.mean():.3f}   "
          f"ρ(size,uv)={spearmanr(mu.tgt_size, mu.uv).correlation:+.3f} "
          f"vs ρ(size,ps)={spearmanr(mu.tgt_size, mu.ps).correlation:+.3f} "
          f"(universeg less size-sensitive — no coarse bottleneck)")
    print("\nBy object size — patchset (ps) vs universeg (uv):\n" + _t.to_string())

    _fig, _ax = plt.subplots(figsize=(9, 6))
    _bins = np.logspace(0, np.log10(mu.tgt_size.max()), 24)
    mu["b"] = pd.cut(mu.tgt_size, _bins)
    _g = mu.groupby("b", observed=True).agg(x=("tgt_size", "median"), psc=("ps_coarse", "mean"),
                                            psn=("ps", "mean"), uvn=("uv", "mean"))
    _ax.plot(_g.x, _g.psc, "-o", ms=4, color="tab:blue", alpha=.5, label="patchset coarse-only")
    _ax.plot(_g.x, _g.psn, "-o", ms=4, color="tab:orange", label="patchset final @128")
    _ax.plot(_g.x, _g.uvn, "-o", ms=4, color="tab:green", label="universeg @128")
    _ax.set_xscale("log"); _ax.grid(alpha=.3); _ax.legend()
    _ax.set(xlabel="object size (fg px @128², log)", ylabel="mean hard dice",
            title="patchset vs universeg by object size — crossover at ≈40 px")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(dp, spearmanr):
    # ── 5. CONTEXT SIZE + THE TWO FAILURE REGIMES ───────────────────────────────────────────
    # (a) Context is the SAME size as the target (task-aug), so a tiny target comes with an equally
    #     tiny context mask — the support occupancy tokens are equally under-resolved. ctx_occ
    #     "helps" only because it co-varies with tgt_size; it is not an independent lever.
    print(f"corr(tgt_size, ctx_size) Spearman = {spearmanr(dp.tgt_size, dp.ctx_size).correlation:.3f}"
          f"   median ctx/tgt size ratio = {(dp.ctx_size / dp.tgt_size.clip(lower=1)).median():.2f}")

    # (b) Two distinct small-object failure regimes, separated by top64@32 (is the object ranked
    #     into the coarse top-64 = the scatter sampler's fg-core budget?):
    #       NOT ranked (top64<0.5): sampler never selects the cell -> refine can't fire -> ~always 0.
    #       ranked (top64≥0.5) but still lost: the cell IS found, yet the fused prob stays <0.5
    #                                          (sub-cell even at Rf=64) -> half still end at 0.
    _tiny = dp[dp.tgt_size <= 32]
    for _lab, _m in [("ranked  (top64≥0.5)", _tiny["top64@32"] >= 0.5),
                     ("NOT ranked (top64<0.5)", _tiny["top64@32"] < 0.5)]:
        _g = _tiny[_m]
        print(f"  {_lab:22s} n={len(_g):4d} ({len(_g)/len(_tiny)*100:2.0f}%)  "
              f"cossim={_g['cossim@32'].mean():.3f}  final={_g.dice.mean():.3f}  "
              f"final==0 {(_g.dice == 0).mean()*100:.0f}%")
    print("\n→ even when the coarse pass ranks the tiny object, ~half still vanish: the fix must lift")
    print("  the coarse target resolution (finer grid / ranking loss on a finer grid), not just the")
    print("  loss weighting — a sub-0.5-occupancy cell has no positive target to aim at.")
    return


@app.cell
def _(dp):
    # ── 6. FINER-GRID PROJECTION (optional enrichment) ──────────────────────────────────────
    # tgt_peak@32 median is ~0.25 for ≤32px objects: their best cell is a quarter full, below the
    # 0.5 hard cut. A finer grid concentrates that mass. The 64² survival numbers aren't in the
    # table (only @32), so read them from the cached CSV if present (else skip). Earlier result:
    # ≤32px empty-hard-GT drops 72%→16% at 64², i.e. ~77% of lost objects gain a surviving cell.
    from pathlib import Path as _Path

    _cache = _Path(__file__).parent / "artifacts" / "11_pool_survival.csv"
    print(f"≤32px median tgt_peak@32 = {dp[dp.tgt_size <= 32]['tgt_peak@32'].median():.2f}  "
          f"(best coarse cell only ~1/4 full → below the 0.5 threshold)")
    if _cache.exists():
        import pandas as _pd
        _pv = _pd.read_csv(_cache)
        _m = dp[["dataset", "sample_idx", "tgt_size"]].merge(_pv, on=["dataset", "sample_idx"])
        _s = _m[_m.tgt_size <= 32]
        _e32 = (_s.cells32 == 0).mean(); _e64 = (_s.cells64 == 0).mean()
        _lost = _s[_s.cells32 == 0]
        print(f"cached @64 projection: ≤32px empty hard GT  {_e32*100:.0f}% (@32) -> {_e64*100:.0f}% (@64); "
              f"of the {len(_lost)} lost @32, {(_lost.cells64 > 0).mean()*100:.0f}% survive @64.")
    else:
        print("(no cached 11_pool_survival.csv — skip the @64 projection)")
    return


if __name__ == "__main__":
    app.run()
