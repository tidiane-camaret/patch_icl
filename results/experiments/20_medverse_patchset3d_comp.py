import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
    # ── medverse vs patchset3d on TotalSegmentator val ──────────────────────────────────────────
    # Paired per-class/-shape/-size comparison of 3D EVAL runs (project patch_icl_3d_eval). Add runs
    # by extending RUNS below (tag -> run_id); the cells generalise to any set of runs. Cases pair on
    # (class, subject, tgt_size); with the per-item eval RNG fix (TotalSegInContextDataset.eval_seed)
    # runs over the same split share identical targets + context. Rebuild the cache by deleting
    # artifacts/20_crop_false_runs.csv.
    import sys
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    sys.path.insert(0, str(Path(__file__).parent))
    from nb_common import ARTIFACTS, get_latest_table

    pd.set_option("display.width", 240)
    pd.set_option("display.float_format", lambda v: f"{v:.3f}")

    # 3D eval runs live in a DIFFERENT project than the 2D nb_common.PROJECT, and log a
    # `cases.table.json` run-table (one row per eval case) instead of `val/samples`.
    PROJECT = "tidiane/patch_icl_3d_eval"
    RUNS = {"patchset": "vc7kfdto", "medverse": "94nlx7yw"}

    _cache = ARTIFACTS / "20_crop_false_runs.csv"
    if _cache.exists():
        D = pd.read_csv(_cache)
        print(f"loaded cached tables {D.shape}")
    else:
        import wandb
        _api = wandb.Api()
        _rows = []
        for _tag, _rid in RUNS.items():
            _r = _api.run(f"{PROJECT}/{_rid}")
            _d = get_latest_table(_r, table_key="cases.table.json")
            _d.insert(0, "run", _tag)
            _d.insert(1, "run_id", _rid)
            _d.insert(2, "gflops", float(_r.config.get("gflops", float("nan"))))
            _rows.append(_d)
        D = pd.concat(_rows, ignore_index=True)
        D.to_csv(_cache, index=False)
        print(f"rebuilt tables {D.shape}")

    # ── shape taxonomy (data-driven from real-mask geometry) ─────────────────────────────────────
    # Morphology families are CLUSTERED (Ward, k=N_SHAPE) from per-class median geometry at 128³
    # (totalseg_geometry_extract.shape_families), not hand-mapped. Auto-labelled
    # {thick|mid|thin}_{blob|tube|sheet} (+ 'frag'); thickness (surf/vol) is the primary axis — the
    # same axis that drives the medverse↔patchset gap. Set N_SHAPE to taste. GEOM (built here from
    # artifacts/20_geometry.csv; first build touches label_128³.npy on NFS) is shared with cell 4.
    from totalseg_geometry_extract import load_or_build_geometry, shape_families
    N_SHAPE = 10
    GEOM = load_or_build_geometry(D[["subject", "class"]], ARTIFACTS / "20_geometry.csv")
    SHAPE, SHAPE_ORDER = shape_families(GEOM, k=N_SHAPE)
    D["shape"] = D["class"].map(SHAPE)
    # classes with no non-empty mask never get clustered → catch-all 'other'
    _unmapped = sorted(D[D["shape"].isna()]["class"].unique())
    if _unmapped:
        print(f"unclustered classes → 'other' ({len(_unmapped)}): {_unmapped}")
        SHAPE_ORDER = SHAPE_ORDER + ["other"]
    D["shape"] = D["shape"].fillna("other")
    print(f"shape families (k={N_SHAPE}, thick→thin): {SHAPE_ORDER}")

    # native fg-voxel (@128 whole-body) size buckets
    SZ_EDGES = [0, 2048, 8192, 32768, 131072, 1e12]
    SZ_LABELS = ["≤2K", "2K-8K", "8K-32K", "32K-131K", ">131K"]
    D["szbin"] = pd.cut(D.tgt_size, SZ_EDGES, labels=SZ_LABELS)
    print("runs:", RUNS, "| rows/run:", D.groupby("run").size().to_dict())
    return D, GEOM, SHAPE_ORDER, SZ_LABELS, np, pd, plt


@app.cell
def _(D):
    # ── 0. PAIRING SANITY + OVERALL HEAD-TO-HEAD ────────────────────────────────────────────────
    # Pair on (class, subject, tgt_size). Under the per-item eval RNG fix, runs over the same split
    # share identical targets + context, so pairing should be CLEAN (no tgt_size split) — the print
    # flags it if not. PAIR (no leading underscore) is shared downstream; marimo scopes _-prefixed
    # names per cell. NB: this cell assumes exactly two runs tagged `medverse`/`patchset`.
    PAIR = D.pivot_table(index=["class", "subject", "tgt_size"], columns="run",
                         values="dice").reset_index().dropna(subset=["medverse", "patchset"])
    _n_union = D.groupby(["class", "subject"]).ngroups
    print(f"paired samples: {len(PAIR)} / {_n_union} (class,subject) groups  "
          f"→ pairing {'CLEAN' if len(PAIR) == _n_union else 'SPLIT (repro broken!)'}")

    _cls = PAIR.groupby("class").agg(mv=("medverse", "mean"), ps=("patchset", "mean"))
    _gf = D.groupby("run").gflops.first()
    _tm = D.groupby("run").time_ms.median()
    print("\n=== OVERALL (use_crop=false, whole-body fast path) ===")
    print(f"  MACRO dice   medverse={_cls.mv.mean():.4f}   patchset={_cls.ps.mean():.4f}   "
          f"gap(mv-ps)={_cls.mv.mean() - _cls.ps.mean():+.4f}")
    print(f"  MICRO dice   medverse={PAIR.medverse.mean():.4f}   patchset={PAIR.patchset.mean():.4f}")
    print(f"  complete-miss (dice==0)  medverse={(PAIR.medverse == 0).mean():.1%}   "
          f"patchset={(PAIR.patchset == 0).mean():.1%}")
    print(f"  per-sample   medverse wins {(PAIR.medverse > PAIR.patchset).mean():.1%}   "
          f"patchset wins {(PAIR.patchset > PAIR.medverse).mean():.1%}   "
          f"ties {(PAIR.medverse == PAIR.patchset).mean():.1%}")
    _ncls = len(_cls)
    print(f"  classes won  medverse {int((_cls.mv > _cls.ps).sum())}/{_ncls}   "
          f"patchset {int((_cls.ps >= _cls.mv).sum())}/{_ncls}")
    print(f"  compute      medverse {_gf['medverse']:.0f} GFLOPs / {_tm['medverse']:.0f}ms   "
          f"patchset {_gf['patchset']:.0f} GFLOPs / {_tm['patchset']:.0f}ms "
          f"({_gf['patchset'] / _gf['medverse']:.0%} FLOPs)")
    return (PAIR,)


@app.cell
def _(D, pd):
    # ── 1. PER-CLASS WIN/LOSS ───────────────────────────────────────────────────────────────────
    # Mean dice per class with shape + median tgt_size, sorted by patchset−medverse.
    _t = D.pivot_table(index="class", columns="run", values="dice", aggfunc="mean")
    _sz = D.groupby("class").tgt_size.median().rename("tgt_size")
    _shp = D.groupby("class").shape.first().rename("shape")
    _tab = pd.concat([_shp, _sz, _t[["medverse", "patchset"]]], axis=1)
    _tab["ps-mv"] = _tab.patchset - _tab.medverse
    _tab = _tab.sort_values("ps-mv", ascending=False)
    print(f"PATCHSET wins {int((_tab['ps-mv'] > 0).sum())}/{len(_tab)} classes.\n")
    print("── patchset wins (ps>mv) ──\n" + _tab[_tab["ps-mv"] > 0].to_string())
    print("\n── medverse wins (mv>ps), top 12 ──\n"
          + _tab[_tab["ps-mv"] < 0].tail(12).iloc[::-1].to_string())
    return


@app.cell
def _(D, SHAPE_ORDER, np, plt):
    # ── 1b. PER-CLASS DICE ANALYSIS + PLOT ──────────────────────────────────────────────────────
    # Per-class mean dice for every run (generalises to any number of runs). When exactly two runs
    # are present: two run-vs-run scatters per class (coloured by shape family | by target size,
    # symlog axes, y=x reference) plus a full-width diverging bar of every class's gap. With >2
    # runs: per-class dice sorted, one line per run. Complements cell 1's table with the visual read.
    _runs = list(D.run.unique())
    _pc = D.pivot_table(index="class", columns="run", values="dice", aggfunc="mean")[_runs]
    _shp = D.groupby("class").shape.first().reindex(_pc.index)
    _n = D[D.run == _runs[0]].groupby("class").size().reindex(_pc.index)  # eval cases / class
    print(f"PER-CLASS mean dice — {len(_pc)} classes × {len(_runs)} runs:")
    print(_pc.describe().loc[["mean", "std", "min", "max"]].to_string())
    print("  classes-led: " + "  ".join(
        f"{r}={int((_pc[r] >= _pc.drop(columns=r).max(axis=1)).sum())}" for r in _runs))

    if len(_runs) == 2:
        from matplotlib.colors import LogNorm as _LogNorm
        _a, _b = _runs  # x-axis, y-axis
        _delta = (_pc[_b] - _pc[_a]).sort_values()
        _sz = D.groupby("class").tgt_size.median().reindex(_pc.index)  # per-class median size
        _lt, _hi = 0.02, float(_pc.max().max())
        # top row: two scatters (by shape family | by target size); bottom row: full-width delta bar
        _fig, _axd = plt.subplot_mosaic([["shape", "size"], ["bar", "bar"]],
                                        figsize=(max(13, 0.24 * len(_delta)), 12),
                                        gridspec_kw={"height_ratios": [1.2, 1]})
        # scatter 1 — coloured by shape family
        _axS = _axd["shape"]
        for _fam in SHAPE_ORDER:
            _m = _shp == _fam
            if _m.any():
                _axS.scatter(_pc.loc[_m, _a], _pc.loc[_m, _b], s=28, alpha=0.8, label=_fam)
        _axS.legend(fontsize=8, title="shape")
        _axS.set_title(f"Per-class dice: {_b} vs {_a}  (by shape)")
        # scatter 2 — coloured by median target size (log colour scale)
        _axZ = _axd["size"]
        _scz = _axZ.scatter(_pc[_a], _pc[_b], c=_sz.values, s=28, alpha=0.85,
                            cmap="viridis", norm=_LogNorm())
        _fig.colorbar(_scz, ax=_axZ, label="median tgt_size (fg vox @128³)")
        _axZ.set_title(f"Per-class dice: {_b} vs {_a}  (by target size)")
        # shared styling: y=x + symlog axes (linear below _lt, log above) spread the near-0 cluster
        # while still showing complete-miss classes at dice==0 (true log would drop them).
        for _ax in (_axS, _axZ):
            _ax.plot([0, _hi], [0, _hi], "k--", lw=0.8, alpha=0.6)
            _ax.set_xscale("symlog", linthresh=_lt); _ax.set_yscale("symlog", linthresh=_lt)
            _ax.set_xlim(0, 1); _ax.set_ylim(0, 1); _ax.set_aspect("equal", adjustable="box")
            _ax.set(xlabel=f"{_a} mean dice (symlog)", ylabel=f"{_b} mean dice (symlog)")
            _ax.grid(alpha=0.3)
        # bottom: diverging bar of per-class gap for ALL classes in one horizontal row (sorted by Δ)
        _axB = _axd["bar"]
        _col = ["tab:blue" if _v < 0 else "tab:red" for _v in _delta]
        _axB.bar(np.arange(len(_delta)), _delta.values, color=_col)
        _axB.axhline(0, color="k", lw=0.6)
        _axB.set_xticks(np.arange(len(_delta)))
        _axB.set_xticklabels([f"{c} (n={int(_n[c])})" for c in _delta.index],
                             rotation=90, fontsize=6)
        _axB.set_xlim(-0.5, len(_delta) - 0.5)
        _axB.set(ylabel=f"Δ dice ({_b} − {_a})",
                 title=f"Per-class gap, all classes  (blue={_a} better, red={_b} better)")
        _axB.grid(alpha=0.3, axis="y")
    else:
        _order = _pc.mean(axis=1).sort_values().index
        _fig, _ax = plt.subplots(figsize=(12, 6))
        for _r in _runs:
            _ax.plot(np.arange(len(_order)), _pc.loc[_order, _r], "-o", ms=3, alpha=0.8, label=_r)
        _ax.set(xlabel="class (sorted by mean dice over runs)", ylabel="per-class mean dice",
                title="Per-class dice by run")
        _ax.legend(); _ax.grid(alpha=0.3)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(PAIR, SZ_LABELS, np, pd, plt):
    # ── 2. DICE vs TARGET SIZE ──────────────────────────────────────────────────────────────────
    # Mean dice per model in native fg-voxel (@128³) size buckets, with the mv-ps gap per bucket.
    _p = PAIR.copy()
    _p["szbin"] = pd.cut(_p.tgt_size, [0, 2048, 8192, 32768, 131072, 1e12], labels=SZ_LABELS)
    _g = _p.groupby("szbin", observed=True).agg(
        n=("medverse", "size"), mv=("medverse", "mean"), ps=("patchset", "mean"))
    _g["mv-ps"] = _g.mv - _g.ps
    print("MEAN DICE by target size (fg vox @128³ whole-body):\n" + _g.to_string())

    _x = np.arange(len(_g))
    _fig, _ax = plt.subplots(figsize=(9, 5))
    _ax.bar(_x - 0.2, _g.mv, 0.4, color="tab:blue", label="medverse")
    _ax.bar(_x + 0.2, _g.ps, 0.4, color="tab:red", label="patchset3d")
    for _i, (_m, _s) in enumerate(zip(_g.mv, _g.ps)):
        _ax.text(_i - 0.2, _m + .003, f"{_m:.3f}", ha="center", va="bottom", fontsize=7)
        _ax.text(_i + 0.2, _s + .003, f"{_s:.3f}", ha="center", va="bottom", fontsize=7)
    _ax.set_xticks(_x); _ax.set_xticklabels([f"{l}\n(n={int(n)})" for l, n in zip(_g.index, _g.n)])
    _ax.set(xlabel="target size (fg vox)", ylabel="mean hard dice",
            title="Mean dice vs target size")
    _ax.legend(); _ax.grid(alpha=0.3, axis="y")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(D, PAIR, SHAPE_ORDER, np, plt):
    # ── 3. PER-SHAPE ───────────────────────────────────────────────────────────────────────────
    # Per shape family (SHAPE map, cell 0): macro dice (mean over classes so big-n classes don't
    # dominate), per-sample patchset win rate, and each model's complete-miss (dice==0) rate.
    _shp = D.groupby("class").shape.first()
    _p = PAIR.assign(shape=PAIR["class"].map(_shp))

    # macro (per-class then per-shape) so big-n classes don't dominate a family
    _cls = _p.groupby(["shape", "class"]).agg(mv=("medverse", "mean"), ps=("patchset", "mean"))
    _fam = _cls.groupby("shape").agg(
        n_cls=("mv", "size"), mv=("mv", "mean"), ps=("ps", "mean")).reindex(SHAPE_ORDER)
    _fam["n_samp"] = _p.groupby("shape").size().reindex(SHAPE_ORDER)
    # per-sample win rate + miss rate within family
    _fam["ps_win"] = _p.groupby("shape").apply(
        lambda g: (g.patchset > g.medverse).mean(), include_groups=False).reindex(SHAPE_ORDER)
    _fam["ps_miss"] = _p.groupby("shape").apply(
        lambda g: (g.patchset == 0).mean(), include_groups=False).reindex(SHAPE_ORDER)
    _fam["mv_miss"] = _p.groupby("shape").apply(
        lambda g: (g.medverse == 0).mean(), include_groups=False).reindex(SHAPE_ORDER)
    _fam["ps-mv"] = _fam.ps - _fam.mv
    print("MEAN DICE by shape family (macro over classes):\n"
          + _fam[["n_cls", "n_samp", "mv", "ps", "ps-mv", "ps_win", "ps_miss", "mv_miss"]].to_string())

    _x = np.arange(len(_fam))
    _fig, _ax = plt.subplots(figsize=(9, 5))
    _ax.bar(_x - 0.2, _fam.mv, 0.4, color="tab:blue", label="medverse")
    _ax.bar(_x + 0.2, _fam.ps, 0.4, color="tab:red", label="patchset3d")
    for _i, (_m, _s) in enumerate(zip(_fam.mv, _fam.ps)):
        _ax.text(_i - 0.2, _m + .003, f"{_m:.3f}", ha="center", va="bottom", fontsize=8)
        _ax.text(_i + 0.2, _s + .003, f"{_s:.3f}", ha="center", va="bottom", fontsize=8)
    _ax.set_xticks(_x)
    _ax.set_xticklabels([f"{s}\n({int(c)} cls)" for s, c in zip(_fam.index, _fam.n_cls)])
    _ax.set(xlabel="shape family", ylabel="macro mean hard dice",
            title="Macro dice by shape family")
    _ax.legend(); _ax.grid(alpha=0.3, axis="y")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(D, GEOM):
    # ── 4. GEOMETRY → DICE JOIN ─────────────────────────────────────────────────────────────────
    # GEOM (per-(subject,class) real-mask descriptors at the 128³ resolution the models saw) is
    # built once in cell 0 and shared here — no re-import/re-load (marimo: each name is defined by
    # exactly one cell). GEO = per-sample geometry joined to each run's dice; feeds the driver cells.
    import sys as _sys
    from pathlib import Path as _P
    _sys.path.insert(0, str(_P(__file__).parent))
    from totalseg_geometry_extract import FEATURES as GEO_FEATURES, LOG_FEATURES as GEO_LOGF

    _dice = D.pivot_table(index=["subject", "class"], columns="run", values="dice").reset_index()
    GEO = GEOM.merge(_dice, on=["subject", "class"], how="inner")
    print("GEO:", GEO.shape, "| runs:", [c for c in GEO.columns if c in D.run.unique()])
    return GEO, GEO_FEATURES, GEO_LOGF


@app.cell
def _(D, GEO, GEO_FEATURES, GEO_LOGF, np, pd, plt):
    # ── 5. GEOMETRIC DRIVERS OF DICE ────────────────────────────────────────────────────────────
    # Per-class Pearson: each geometry feature vs each run's mean dice (heavy features log1p'd).
    # Per-class (not per-sample) avoids Dice==0 zero-inflation distorting the correlation.
    _runs = [r for r in D.run.unique() if r in GEO.columns]
    _aggs = {f: (f, "median") for f in GEO_FEATURES}
    _aggs.update({r: (r, "mean") for r in _runs})
    GC = GEO.groupby("class").agg(**_aggs)
    _cl = GC.copy()
    for _c in GEO_LOGF:
        _cl[_c] = np.log1p(_cl[_c])
    _corr = pd.DataFrame({r: [_cl[f].corr(_cl[r]) for f in GEO_FEATURES] for r in _runs},
                         index=GEO_FEATURES)
    if {"medverse", "patchset"} <= set(_runs):
        _corr["delta(ps-mv)"] = [_cl[f].corr(_cl["patchset"] - _cl["medverse"]) for f in GEO_FEATURES]
    _corr = _corr.reindex(_corr[_runs[0]].abs().sort_values(ascending=False).index)
    print(f"PER-CLASS Pearson — geometry vs mean dice (n={len(GC)} classes):\n" + _corr.to_string())

    _y = np.arange(len(_corr))
    _w = 0.8 / max(len(_runs), 1)
    _fig, _ax = plt.subplots(figsize=(8, 7))
    for _j, _r in enumerate(_runs):
        _ax.barh(_y + (_j - (len(_runs) - 1) / 2) * _w, _corr[_r], _w, label=_r)
    _ax.axvline(0, color="k", lw=0.6)
    _ax.set_yticks(_y); _ax.set_yticklabels(_corr.index); _ax.invert_yaxis()
    _ax.set(xlabel="Pearson r (per-class, vs mean dice)", title="Geometric drivers of dice")
    _ax.legend(); _ax.grid(alpha=0.3, axis="x")
    _fig.tight_layout()
    _fig
    return (GC,)


@app.cell
def _(D, GC, GEO, pd, plt):
    # ── 6. THICKNESS CROSSOVER ──────────────────────────────────────────────────────────────────
    # Dice vs object thickness (interior-EDT p90, voxels @128³). Per-sample binned table + per-class
    # scatter. When both medverse/patchset present, also the thin/thick-half mean delta(ps-mv).
    _runs = [r for r in D.run.unique() if r in GEO.columns]
    _g = GEO.copy()
    _g["tbin"] = pd.cut(_g.thick_p90, [0, 1.5, 2.5, 3.5, 5.0, 1e9],
                        labels=["1", "2", "3", "4-5", ">5"])
    _t = _g.groupby("tbin", observed=True).agg(n=("volume", "size"),
                                               **{r: (r, "mean") for r in _runs})
    print("mean dice by thickness (interior-EDT p90, voxels):\n" + _t.to_string())
    if {"medverse", "patchset"} <= set(_runs):
        _d = GC.assign(delta=GC.patchset - GC.medverse)
        _med = _d.thick_p90.median()
        print(f"\nper-class crossover at thick_p90 median {_med:.1f} vox:")
        print(f"  thin  half (<= {_med:.1f}): mean delta(ps-mv) = {_d[_d.thick_p90 <= _med].delta.mean():+.3f}")
        print(f"  thick half ( > {_med:.1f}): mean delta(ps-mv) = {_d[_d.thick_p90 >  _med].delta.mean():+.3f}")

    _fig, _ax = plt.subplots(figsize=(9, 5))
    for _r in _runs:
        _ax.scatter(GC.thick_p90, GC[_r], s=28, alpha=0.8, label=_r)
    _ax.set_xscale("log")
    _ax.set(xlabel="object thickness — interior-EDT p90 (vox @128³, log)",
            ylabel="per-class mean hard dice", title="Dice vs thickness (per class)")
    _ax.legend(); _ax.grid(alpha=0.3)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(D, GEO, GEO_FEATURES, GEO_LOGF, np, pd, plt):
    # ── 7. PER-FEATURE DICE RESPONSE ────────────────────────────────────────────────────────────
    # One panel per geometric feature: per-sample dice binned into quantiles of that feature, mean
    # (line) ± std (band) for each run. Shows each feature's dice-response shape + monotonicity and
    # where the two models diverge — the direct read of "what drives final dice".
    _runs = [r for r in D.run.unique() if r in GEO.columns]
    _nbins = 6
    _ncol = 4
    _nrow = int(np.ceil(len(GEO_FEATURES) / _ncol))
    _fig, _axes = plt.subplots(_nrow, _ncol, figsize=(4 * _ncol, 3 * _nrow))
    _axes = _axes.ravel()
    for _k, _f in enumerate(GEO_FEATURES):
        _ax = _axes[_k]
        _d = GEO[[_f, *_runs]].dropna(subset=[_f]).copy()
        try:  # quantile bins; duplicates='drop' handles ties (e.g. thickness=1, n_components)
            _d["b"] = pd.qcut(_d[_f], _nbins, duplicates="drop")
        except (ValueError, IndexError):
            _ax.set_visible(False); continue
        _grp = _d.groupby("b", observed=True)
        _x = _grp[_f].median()
        for _r in _runs:
            _m, _s = _grp[_r].mean(), _grp[_r].std()
            _ax.plot(_x, _m, "-o", ms=3, label=_r)
            _ax.fill_between(_x, (_m - _s).clip(lower=0), _m + _s, alpha=0.15)
        if _f in GEO_LOGF:
            _ax.set_xscale("log")
        _ax.set_title(_f, fontsize=9)
        _ax.grid(alpha=0.3)
        if _k == 0:
            _ax.legend(fontsize=7)
    for _k in range(len(GEO_FEATURES), len(_axes)):
        _axes[_k].set_visible(False)
    _fig.suptitle("Per-feature dice response (per-sample quantile bins, mean ± std)", y=1.002)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _():
    # ── Contents ─────────────────────────────────────────────────────────────────────────────────
    # cell 0: fetch + cache + shape/size maps.   cell 0b: pairing sanity + overall head-to-head.
    # cell 1: per-class win/loss.   cell 1b: per-class dice analysis + plot.
    # cell 2: dice vs target size.   cell 3: per-shape.
    # cell 4: real-mask geometry extraction.   cell 5: geometric drivers (per-class corr).
    # cell 6: thickness crossover.   cell 7: per-feature dice response (mean ± std).
    # Geometry via results/experiments/totalseg_geometry_extract.py (reusable; --pairs/--classes CLI).
    # (Interpretation of specific runs is kept out of this notebook so it stays valid as runs change.)
    print("tables: cells 0-6;  figures: cells 2,3,5,6,7")
    return


if __name__ == "__main__":
    app.run()
