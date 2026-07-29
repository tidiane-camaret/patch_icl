import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
    # ── two-run comparison: detailed val breakdown (patch_icl_3d_exps) ───────────────────────────
    # Set RUNS = {display_name: wandb_run_id}. Every table/figure below reports the SAME per-run
    # metrics side by side (delta = <run2> − <run1>). The class list, sample set and shape families
    # are DERIVED from the logged data and SHARED across both runs, so this is reusable for any pair.
    # Caches each run under artifacts/21_<id>_*.{csv,json} (+ a combined geometry); delete to refresh.
    import json
    import sys
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    sys.path.insert(0, str(Path(__file__).parent))
    from nb_common import ARTIFACTS as _ARTIFACTS, get_latest_table
    from totalseg_geometry_extract import load_or_build_geometry, shape_families

    pd.set_option("display.width", 240)
    pd.set_option("display.float_format", lambda v: f"{v:.3f}")

    PROJECT = "tidiane/patch_icl_3d_exps"
    RUNS = {"run_name_1": "93cz8fba", "run_name_2": "mj19797e"}
    N_SHAPE = 10                     # morphology cluster count (thick→thin families)

    def _load(_id):
        """Load one run's cached samples table + meta, fetching from W&B on cache miss."""
        _samp_c = _ARTIFACTS / f"21_{_id}_samples.csv"
        _meta_c = _ARTIFACTS / f"21_{_id}_meta.json"
        if _samp_c.exists() and _meta_c.exists():
            return pd.read_csv(_samp_c), json.loads(_meta_c.read_text())
        import wandb
        _run = wandb.Api().run(f"{PROJECT}/{_id}")
        _s = get_latest_table(_run, table_key="val/samples.table.json")
        _m = {"name": _run.name, "config": dict(_run.config),
              "summary": {k: _run.summary.get(k) for k in ("epoch", "val/dice", "val/best_dice")}}
        _ARTIFACTS.mkdir(parents=True, exist_ok=True)
        _s.to_csv(_samp_c, index=False)
        _meta_c.write_text(json.dumps(_m, default=float))
        return _s, _m

    RUN_NAMES = list(RUNS)
    _frames, META = [], {}
    for _name, _id in RUNS.items():
        _s, _m = _load(_id)
        _s = _s.copy(); _s["run"] = _name
        _frames.append(_s); META[_name] = _m
        print(f"loaded {_name} ({_id}): samples {_s.shape}  | {_m['name']}")
    S = pd.concat(_frames, ignore_index=True)

    # shared morphology families + per-sample geometry from ALL evaluated (subject,class) real masks
    _pairs = S[["subject", "class"]].drop_duplicates()
    GEOM = load_or_build_geometry(_pairs, _ARTIFACTS / f"21_cmp_{'_'.join(RUNS.values())}_geometry.csv")
    SHAPE, SHAPE_ORDER = shape_families(GEOM, k=N_SHAPE)
    S = S.merge(GEOM, on=["subject", "class"], how="left")
    S["shape"] = S["class"].map(SHAPE).fillna("other")
    if (S["shape"] == "other").any():
        SHAPE_ORDER = SHAPE_ORDER + ["other"]

    def _cfg(_name, *path):
        d = META[_name]["config"]
        for k in path:
            d = d.get(k) if isinstance(d, dict) else None
        return d
    for _name in RUN_NAMES:
        _sm = META[_name]["summary"]
        _kv = {"model": _cfg(_name, "model"), "synth": _cfg(_name, "data", "synth_method"),
               "ctx": _cfg(_name, "data", "context_size"), "size": _cfg(_name, "data", "image_size")}
        print(f"  [{_name}] " + "  ".join(f"{k}={v}" for k, v in _kv.items() if v is not None)
              + f" | epoch {int(_sm['epoch'])} val/dice={_sm['val/dice']:.4f} best={_sm['val/best_dice']:.4f}")
    print(f"  {S['class'].nunique()} classes / {len(S)} samples across {len(RUN_NAMES)} runs")
    print(f"  shape families (k={N_SHAPE}, thick→thin): {SHAPE_ORDER}")
    return RUN_NAMES, S, SHAPE_ORDER, np, pd, plt


@app.cell
def _(RUN_NAMES, S, SHAPE_ORDER, pd, plt):
    # ── 1. PER-CLASS VAL DICE — run comparison ───────────────────────────────────────────────────
    # Per-class mean dice for each run, pivoted side by side with delta = <run2> − <run1>. The scatter
    # plots run1 (x) vs run2 (y) with a parity line; points ABOVE the diagonal improved in run2. Colour
    # = shape family; the largest movers (either direction) are annotated.
    _r0, _r1 = RUN_NAMES[0], RUN_NAMES[1]
    _pc = S.groupby(["class", "run"]).dice.mean().unstack("run").reindex(columns=RUN_NAMES)
    _meta = S.groupby("class").agg(shape=("shape", "first"), in_train=("in_train", "first"),
                                   n=("dice", "size"), tgt_size=("tgt_size", "median"))
    _tab = _pc.join(_meta)
    _tab["delta"] = _tab[_r1] - _tab[_r0]
    _tab = _tab.sort_values("delta")

    _macro = _pc.mean()
    _micro = S.groupby("run").dice.mean().reindex(RUN_NAMES)
    print("macro dice: " + "  ".join(f"{r}={_macro[r]:.4f}" for r in RUN_NAMES)
          + f"  Δ={_macro[_r1] - _macro[_r0]:+.4f}")
    print("micro dice: " + "  ".join(f"{r}={_micro[r]:.4f}" for r in RUN_NAMES)
          + f"  Δ={_micro[_r1] - _micro[_r0]:+.4f}")
    print(f"biggest movers (Δ dice = {_r1} − {_r0}):")
    print(pd.concat([_tab.head(8), _tab.tail(8)]).to_string())

    _pal = plt.cm.tab20.colors + plt.cm.tab20b.colors
    _cmap = {f: _pal[i % len(_pal)] for i, f in enumerate(SHAPE_ORDER)}
    _d = _tab.dropna(subset=[_r0, _r1])
    _fig, _ax = plt.subplots(figsize=(9, 9))
    _ax.scatter(_d[_r0], _d[_r1], s=36, c=[_cmap[s] for s in _d["shape"]],
                alpha=0.85, edgecolor="k", linewidth=0.3, zorder=3)
    _ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="parity")
    _mv = pd.concat([_d.sort_values("delta").head(6), _d.sort_values("delta").tail(6)])
    for _c, _row in _mv.iterrows():
        _ax.annotate(_c, (_row[_r0], _row[_r1]), fontsize=6, xytext=(3, 3), textcoords="offset points")
    _ax.set(xlim=(0, 1), ylim=(0, 1), xlabel=f"{_r0} dice", ylabel=f"{_r1} dice",
            title=f"Per-class val dice: {_r1} vs {_r0}  (Δmacro={_macro[_r1] - _macro[_r0]:+.3f})")
    from matplotlib.patches import Patch
    _seen = [f for f in SHAPE_ORDER if f in set(_d["shape"])]
    _ax.legend(handles=[Patch(color=_cmap[f], label=f) for f in _seen]
               + _ax.get_legend_handles_labels()[0], fontsize=7, ncol=2, loc="lower right")
    _ax.grid(alpha=0.3)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(RUN_NAMES, S, SHAPE_ORDER, np, pd, plt):
    # ── 1b. TRAIN vs HELD-OUT GENERALIZATION — run comparison ────────────────────────────────────
    # Held-out (unseen) classes are the point of in-context seg. For EACH run: macro/micro dice and
    # miss-rate split by membership, plus the train−heldout gap. Held-out dice by shape family and
    # matched lateral-mirror pairs (morphology-fair) are compared per run so the runs are judged on the
    # same anatomy, not confounded by which classes each held out.
    def _macro(g):
        return g.groupby("class").dice.mean().mean()

    _ov = S.groupby(["run", "in_train"]).apply(lambda g: pd.Series({
        "n_cls": g["class"].nunique(), "n_samp": len(g),
        "macro_dice": _macro(g), "micro_dice": g.dice.mean(),
        "miss_rate": (g.dice < 0.01).mean(), "med_tgt_size": g.tgt_size.median(),
    }), include_groups=False).rename(index={True: "train", False: "held-out"})
    print("per-run overall (membership: train=seen, held-out=unseen):")
    print(_ov.to_string())
    for _r in RUN_NAMES:
        if (_r, "train") in _ov.index and (_r, "held-out") in _ov.index:
            _gap = _ov.loc[(_r, "train"), "macro_dice"] - _ov.loc[(_r, "held-out"), "macro_dice"]
            print(f"  [{_r}] macro gap train−heldout = {_gap:+.3f}  (heldout med tgt "
                  f"{_ov.loc[(_r, 'held-out'), 'med_tgt_size']:.0f} vs {_ov.loc[(_r, 'train'), 'med_tgt_size']:.0f})")
        else:
            print(f"  [{_r}] no held-out classes; skipping gap")

    # matched lateral-mirror pairs per run (same organ, trained side vs held-out side)
    def _flip(c):
        return (c.replace("_left", "_TMP").replace("_right", "_left").replace("_TMP", "_right")
                if ("_left" in c or "_right" in c) else None)
    _mrows = []
    for _r in RUN_NAMES:
        _cls = S[S.run == _r].groupby("class").agg(dice=("dice", "mean"),
                                                   in_train=("in_train", "first")).to_dict("index")
        for _c, _rr in _cls.items():
            _o = _flip(_c)
            if _o in _cls and _rr["in_train"] and not _cls[_o]["in_train"]:
                _mrows.append((_r, _c.replace("_left", "").replace("_right", ""),
                               _rr["dice"], _cls[_o]["dice"]))
    _M = pd.DataFrame(_mrows, columns=["run", "organ", "trained_side", "heldout_side"])
    if len(_M):
        _M["delta"] = _M.trained_side - _M.heldout_side
        print("\nmatched lateral-mirror pairs (mean over pairs, per run → morphology controlled):")
        print(_M.groupby("run").agg(n=("organ", "size"), trained=("trained_side", "mean"),
                                    heldout=("heldout_side", "mean"), delta=("delta", "mean")).to_string())
    else:
        print("\nno matched train/held-out mirror pairs")

    _w = 0.8 / len(RUN_NAMES)
    _fig, (_a0, _a1) = plt.subplots(1, 2, figsize=(15, 5))
    # (a) macro dice by membership × run
    _mem = ["train", "held-out"]
    _x = np.arange(len(_mem))
    for _i, _r in enumerate(RUN_NAMES):
        _vals = [_ov.loc[(_r, _m), "macro_dice"] if (_r, _m) in _ov.index else np.nan for _m in _mem]
        _a0.bar(_x + (_i - (len(RUN_NAMES) - 1) / 2) * _w, _vals, _w, label=_r)
    _a0.set_xticks(_x); _a0.set_xticklabels(_mem)
    _a0.set(ylabel="macro val dice", title="(a) macro dice by membership × run")
    _a0.legend(fontsize=8); _a0.grid(alpha=0.3, axis="y")
    # (b) held-out macro dice by shape family × run
    _hd = (S[~S.in_train].groupby(["shape", "run", "class"]).dice.mean()
           .groupby(["shape", "run"]).mean().unstack("run").reindex(SHAPE_ORDER).dropna(how="all"))
    if len(_hd):
        _xf = np.arange(len(_hd))
        for _i, _r in enumerate(RUN_NAMES):
            if _r in _hd.columns:
                _a1.bar(_xf + (_i - (len(RUN_NAMES) - 1) / 2) * _w, _hd[_r].values, _w, label=_r)
        _a1.set_xticks(_xf); _a1.set_xticklabels(_hd.index, rotation=45, ha="right", fontsize=7)
        _a1.legend(fontsize=8)
    _a1.set(ylabel="held-out macro dice", title="(b) held-out dice by shape family × run (thick→thin)")
    _a1.grid(alpha=0.3, axis="y")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(RUN_NAMES, S, SHAPE_ORDER, np, plt):
    # ── 1c. PER-CLASS DICE vs VOLUME — run comparison ────────────────────────────────────────────
    # One labelled panel per shape family; both runs plotted (colour = run) with a faint vertical
    # connector per class (volume is fixed per class, so the segment length = dice change between
    # runs). Below, a membership scatter PER RUN (no labels) shows train vs held-out at a glance.
    import marimo as mo
    import matplotlib.ticker as mticker
    from matplotlib.lines import Line2D
    import textalloc as ta

    _pc = (S.groupby(["class", "run"])
             .agg(dice=("dice", "mean"), volume=("volume", "mean"),
                  in_train=("in_train", "first"), shape=("shape", "first"))
             .dropna(subset=["volume"]).reset_index())
    _pc = _pc[_pc.volume > 0]
    _pc["logvol"] = np.log10(_pc["volume"].values)

    # shared bounds across all panels
    _xmin, _xmax = float(_pc.logvol.min()), float(_pc.logvol.max())
    _ymin, _ymax = float(_pc.dice.min()), float(_pc.dice.max())
    _xpad = 0.05 * (_xmax - _xmin + 1e-9); _ypad = max(0.03, 0.08 * (_ymax - _ymin + 1e-9))
    _xlim = (_xmin - _xpad, _xmax + _xpad); _ylim = (max(0.0, _ymin - _ypad), min(1.0, _ymax + _ypad))

    _run_pal = plt.cm.tab10.colors
    _run_cmap = {r: _run_pal[i % len(_run_pal)] for i, r in enumerate(RUN_NAMES)}
    _mem_cmap = {True: "tab:blue", False: "tab:orange"}

    def _log_ticks(ax):
        lo, hi = int(np.floor(_xmin)), int(np.ceil(_xmax))
        majors = list(range(lo, hi + 1))
        ax.xaxis.set_major_locator(mticker.FixedLocator(majors))
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: rf"$10^{{{int(round(v))}}}$"))
        ax.xaxis.set_minor_locator(
            mticker.FixedLocator([d + np.log10(m) for d in majors for m in range(2, 10)]))

    def _labels(ax, lx, dy, names):
        if len(names):
            ta.allocate(ax, lx, dy, names, x_scatter=lx, y_scatter=dy, textsize=6,
                        margin=0.01, min_distance=0.015, max_distance=0.5, linewidth=0.4,
                        linecolor="0.6", draw_lines=True, nbr_candidates=400, draw_all=True)

    # ===== FIG 1: per-shape small multiples, both runs + connectors =====
    _shapes = [f for f in SHAPE_ORDER if f in set(_pc["shape"])]
    _ncol = 3; _nrow = int(np.ceil(len(_shapes) / _ncol))
    _fig1, _axes = plt.subplots(_nrow, _ncol, figsize=(6.5 * _ncol, 4.8 * _nrow),
                                sharex=True, sharey=True, squeeze=False)
    _axes = _axes.ravel()
    for _i, _sh in enumerate(_shapes):
        _ax = _axes[_i]
        _sub = _pc[_pc["shape"] == _sh]
        for _, _cc in _sub.groupby("class"):      # connector per class across runs
            if len(_cc) >= 2:
                _ax.plot(_cc.logvol, _cc.dice, "-", color="0.7", lw=0.6, zorder=1)
        for _r in RUN_NAMES:
            _rr = _sub[_sub.run == _r]
            _ax.scatter(_rr.logvol, _rr.dice, s=34, color=_run_cmap[_r], alpha=0.9,
                        zorder=3, edgecolor="k", linewidth=0.3, label=_r)
        _ax.set_xlim(_xlim); _ax.set_ylim(_ylim); _log_ticks(_ax)
        _last = _sub.groupby("class").tail(1)     # label each class once
        _labels(_ax, _last.logvol.values, _last.dice.values, _last["class"].tolist())
        _ax.set_title(f"{_sh} ({_sub['class'].nunique()} cls)", fontsize=9)
        _ax.grid(alpha=0.3, which="both")
    for _j in range(len(_shapes), len(_axes)):
        _axes[_j].set_visible(False)
    _axes[0].legend(fontsize=8, loc="lower right")
    _fig1.supxlabel("mean object volume (voxels, log)")
    _fig1.supylabel("mean per-class dice")
    _fig1.suptitle("Per-class dice vs volume by shape family — run comparison (grey line = same class)")
    _fig1.tight_layout()

    # ===== FIG 2: membership scatter, one panel per run (no labels) =====
    _fig2, _ax2 = plt.subplots(1, len(RUN_NAMES), figsize=(8 * len(RUN_NAMES), 6.5),
                               sharex=True, sharey=True, squeeze=False)
    _ax2 = _ax2.ravel()
    for _i, _r in enumerate(RUN_NAMES):
        _rr = _pc[_pc.run == _r]
        _ax2[_i].scatter(_rr.logvol, _rr.dice, s=36, c=_rr.in_train.map(_mem_cmap).tolist(),
                         alpha=0.85, zorder=3, edgecolor="k", linewidth=0.3)
        _ax2[_i].set_xlim(_xlim); _ax2[_i].set_ylim(_ylim); _log_ticks(_ax2[_i])
        _ax2[_i].set(xlabel="mean object volume (voxels, log)", title=f"{_r} — membership")
        _ax2[_i].grid(alpha=0.3, which="both")
    _ax2[0].set_ylabel("mean per-class dice")
    _ax2[0].legend(handles=[Line2D([], [], marker="o", ls="", color=_mem_cmap[True], label="train (seen)"),
                            Line2D([], [], marker="o", ls="", color=_mem_cmap[False], label="held-out")],
                   fontsize=9, loc="upper left")
    _fig2.tight_layout()

    mo.vstack([_fig1, _fig2])
    return


@app.cell
def _(RUN_NAMES, S, SHAPE_ORDER, np, plt):
    # ── 2. PER-SHAPE FAMILY BREAKDOWN — run comparison ───────────────────────────────────────────
    # Macro dice (mean over classes) per morphology family for each run, side by side. SHAPE_ORDER is
    # thick→thin, so this reads as a dice-vs-thickness profile compared across runs; delta = <run2>−<run1>.
    _cls = S.groupby(["run", "shape", "class"]).dice.mean()
    _fam = _cls.groupby(["run", "shape"]).mean().unstack("run").reindex(SHAPE_ORDER).dropna(how="all")
    _ncls = S.groupby("shape")["class"].nunique()
    _show = _fam.copy()
    if len(RUN_NAMES) == 2:
        _show["delta"] = _fam[RUN_NAMES[1]] - _fam[RUN_NAMES[0]]
    print("macro dice by shape family × run (thick→thin):\n" + _show.to_string())

    _x = np.arange(len(_fam)); _w = 0.8 / len(RUN_NAMES)
    _fig, _ax = plt.subplots(figsize=(max(9, 1.4 * len(_fam)), 5))
    for _i, _r in enumerate(RUN_NAMES):
        if _r in _fam.columns:
            _b = _ax.bar(_x + (_i - (len(RUN_NAMES) - 1) / 2) * _w, _fam[_r].values, _w, label=_r)
            _ax.bar_label(_b, fmt="%.2f", fontsize=6, padding=1)
    _ax.set_xticks(_x)
    _ax.set_xticklabels([f"{s}\n({int(_ncls.get(s, 0))} cls)" for s in _fam.index], fontsize=7)
    _ax.set(ylabel="macro val dice", title="Macro dice by morphology family × run (thick→thin)")
    _ax.legend(fontsize=8); _ax.grid(alpha=0.3, axis="y")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(RUN_NAMES, S, pd, plt):
    # ── 3. PER-SAMPLE DICE vs GEOMETRY DRIVERS — run comparison ───────────────────────────────────
    # Binned-mean dice vs four geometry drivers (object thickness, target volume, target/context
    # occupancy), ONE line per run over the same quantile bins. Faint points show the raw per-sample
    # spread. Diverging lines mean the runs respond differently to that driver.
    _drivers = [("thick_p90", True), ("volume", True), ("tgt_occ", True), ("ctx_occ", True)]
    _run_pal = plt.cm.tab10.colors
    _run_cmap = {r: _run_pal[i % len(_run_pal)] for i, r in enumerate(RUN_NAMES)}
    _fig, _axes = plt.subplots(2, 2, figsize=(13, 9)); _axes = _axes.ravel()
    for _k, (_f, _logx) in enumerate(_drivers):
        _ax = _axes[_k]
        if _f not in S.columns:
            _ax.set_visible(False); continue
        for _r in RUN_NAMES:
            _d = S[S.run == _r][[_f, "dice"]].dropna()
            _d = _d[_d[_f] > 0] if _logx else _d
            _ax.scatter(_d[_f], _d.dice, s=6, alpha=0.12, color=_run_cmap[_r])
            try:
                _grp = _d.groupby(pd.qcut(_d[_f], 8, duplicates="drop"), observed=True)
                _ax.plot(_grp[_f].median(), _grp.dice.mean(), "-o", color=_run_cmap[_r], ms=4, label=_r)
            except (ValueError, IndexError):
                pass
        if _logx:
            _ax.set_xscale("log")
        _ax.set(xlabel=_f, ylabel="dice", title=f"dice vs {_f}")
        _ax.legend(fontsize=8); _ax.grid(alpha=0.3)
    _fig.suptitle("Per-sample dice vs geometry drivers — run comparison", y=1.002)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _():
    # ── Contents ─────────────────────────────────────────────────────────────────────────────────
    # Compares TWO runs (RUNS dict in cell 0) on the SAME per-run metrics; delta = <run2> − <run1>.
    # cell 0:  fetch + cache both runs, tag with `run`, concat, shared morphology clustering + header.
    # cell 1:  per-class dice pivoted per run (+ movers table); scatter run1 vs run2 with parity line.
    # cell 1b: train vs held-out per run — membership × run bars (a), held-out dice by shape family (b),
    #          matched lateral-mirror pairs per run (morphology-controlled).
    # cell 1c: per-class dice vs volume — per-shape panels with both runs + connectors (labelled);
    #          membership scatter per run (no labels).
    # cell 2:  per-shape family macro dice, grouped bars per run (+ delta table).
    # cell 3:  per-sample dice vs geometry drivers, one binned-mean line per run.
    # Edit RUNS / N_SHAPE in cell 0. Shape taxonomy + geometry come from totalseg_geometry_extract.
    print("tables: cells 0-2;  figures: cells 1-3")
    return


if __name__ == "__main__":
    app.run()
