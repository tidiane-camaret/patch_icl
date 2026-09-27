import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
    # ── 150 / 151 / 152 (cold-transformer chain + medverse arm): feature-fusion + architecture ──
    # 150 = arch.dual_axis=True,l=4 (bi-axial: row cross-context + column img<->mask attention).
    # 151 = arch.dual_axis=False,l=9 (early img-mask PixelShuffle fusion, row-axis-only transformer,
    # l=9 chosen to match 150's gflops_transformer within 3% — see 151's own config header; params
    # are NOT matched, 151 is +52% over 150). Both warm-start ONLY encoder+decoder off a stripped 108
    # checkpoint; the entire in-context reasoning core is randomly initialized for both arms, so this
    # isolates the feature-fusion design axis cleanly (no asymmetric-warm-start confound).
    # 152 = model=medverse (released architecture), started from the released Medverse.ckpt
    # (checkpoint=orig_weights) — NOT a matched cold-init, the fairest available starting point for
    # that architecture instead. Same data/augmentation/schedule as 150/151 except data.query_prior
    # disabled (a torch.compile/AOTAutograd backward crash specific to Medverse's image_context
    # NA-ICL channel — see 152's own config header). Treat 152 as the architecture-axis outlier, not
    # a clean ablation arm like 150/151.
    # Same per-class/-shape/-size breakdown as nb 30 (colipri two-run comparison), generalised to N
    # runs, reading each run's LOCAL wandb sample table (no network) at a matched training epoch.
    # Rebuild by bumping EPOCH or deleting the artifacts/41_* cache.
    import json
    import sys
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    sys.path.insert(0, str(Path(__file__).parent))
    from nb_common import ARTIFACTS
    from totalseg_geometry_extract import load_or_build_geometry, shape_families

    pd.set_option("display.width", 240)
    pd.set_option("display.float_format", lambda v: f"{v:.3f}")

    REPO_ROOT = Path(__file__).resolve().parents[2]
    WANDB_DIR = REPO_ROOT / "wandb"

    RUNS = {"150_dualaxis": "502z85nl", "151_singleaxis_l9": "xs5933wt",
            "152_medverse": "9s7d69ca"}
    # matched training budget: both runs have a logged eval at this epoch (see docs/logs.md
    # 2026-09-2x cold-transformer chain notes) — bump once both runs progress further.
    EPOCH = 250
    N_SHAPE = 10                     # morphology cluster count (thick→thin families)

    def _find_run_dir(run_id):
        matches = sorted(WANDB_DIR.glob(f"run-*-{run_id}"))
        if not matches:
            raise FileNotFoundError(f"no local wandb run dir for {run_id!r} under {WANDB_DIR}")
        return matches[-1]

    def _load(_name, _id):
        """Load one run's cached samples table at EPOCH, reading the local wandb run dir on miss."""
        _cache = ARTIFACTS / f"41_{_id}_e{EPOCH}_samples.csv"
        if _cache.exists():
            return pd.read_csv(_cache)
        _run_dir = _find_run_dir(_id)
        _table_dir = _run_dir / "files" / "media" / "table" / "val"
        _paths = sorted(_table_dir.glob(f"samples_{EPOCH}_*.table.json"))
        if not _paths:
            raise FileNotFoundError(
                f"no samples_{EPOCH}_*.table.json under {_table_dir} for {_name} ({_id}) — "
                f"epoch {EPOCH} not logged yet (train.eval_every)?")
        _parts = [pd.DataFrame((_d := json.loads(_p.read_text()))["data"], columns=_d["columns"])
                  for _p in _paths]
        _s = pd.concat(_parts, ignore_index=True)
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        _s.to_csv(_cache, index=False)
        return _s

    RUN_NAMES = list(RUNS)
    _frames = []
    for _name, _id in RUNS.items():
        _s = _load(_name, _id)
        _s = _s.copy(); _s["run"] = _name
        _frames.append(_s)
        print(f"loaded {_name} ({_id}): samples {_s.shape}")
    S = pd.concat(_frames, ignore_index=True)

    # shared morphology families + per-sample geometry from ALL evaluated (subject,class) real masks
    _pairs = S[["subject", "class"]].drop_duplicates()
    GEOM = load_or_build_geometry(_pairs, ARTIFACTS / f"41_e{EPOCH}_geometry.csv")
    SHAPE, SHAPE_ORDER = shape_families(GEOM, k=N_SHAPE)
    S = S.merge(GEOM, on=["subject", "class"], how="left")
    S["shape"] = S["class"].map(SHAPE).fillna("other")
    if (S["shape"] == "other").any():
        SHAPE_ORDER = SHAPE_ORDER + ["other"]

    print(f"epoch {EPOCH} (matched budget)  |  {S['class'].nunique()} classes / {len(S)} samples "
          f"across {len(RUN_NAMES)} runs")
    print(f"shape families (k={N_SHAPE}, thick→thin): {SHAPE_ORDER}")
    return RUN_NAMES, S, SHAPE_ORDER, np, pd, plt


@app.cell
def _(RUN_NAMES, S, SHAPE_ORDER, pd, plt):
    # ── 1. PER-CLASS VAL DICE — run comparison ───────────────────────────────────────────────────
    # Per-class mean dice for each run, pivoted side by side. "range" = max-min across runs (movers
    # metric, generalises to N runs). Pairwise scatter grid (one panel per run pair) with a parity
    # line; points ABOVE the diagonal improved under the y-axis run. Colour = shape family; the
    # largest movers per pair are annotated. With 152 (medverse) in the mix, its panels vs 150/151
    # read as an ARCHITECTURE gap, not a controlled ablation — see cell 0 header.
    import itertools
    _pc = S.groupby(["class", "run"]).dice.mean().unstack("run").reindex(columns=RUN_NAMES)
    _meta = S.groupby("class").agg(shape=("shape", "first"), in_train=("in_train", "first"),
                                   n=("dice", "size"), tgt_size=("tgt_size", "median"))
    _tab = _pc.join(_meta)
    _tab["range"] = _pc.max(axis=1) - _pc.min(axis=1)
    _tab = _tab.sort_values("range", ascending=False)

    _macro = _pc.mean()
    _micro = S.groupby("run").dice.mean().reindex(RUN_NAMES)
    print("macro dice: " + "  ".join(f"{r}={_macro[r]:.4f}" for r in RUN_NAMES))
    print("micro dice: " + "  ".join(f"{r}={_micro[r]:.4f}" for r in RUN_NAMES))
    print("biggest movers (by max-min range across runs):")
    print(_tab.head(16).to_string())

    _pal = plt.cm.tab20.colors + plt.cm.tab20b.colors
    _cmap = {f: _pal[i % len(_pal)] for i, f in enumerate(SHAPE_ORDER)}
    from matplotlib.patches import Patch
    _pairs = list(itertools.combinations(RUN_NAMES, 2))
    _fig, _axes = plt.subplots(1, len(_pairs), figsize=(8 * len(_pairs), 8), squeeze=False)
    _axes = _axes.ravel()
    for _k, (_r0, _r1) in enumerate(_pairs):
        _ax = _axes[_k]
        _d = _tab.dropna(subset=[_r0, _r1]).copy()
        _d["delta"] = _d[_r1] - _d[_r0]
        _ax.scatter(_d[_r0], _d[_r1], s=36, c=[_cmap[s] for s in _d["shape"]],
                    alpha=0.85, edgecolor="k", linewidth=0.3, zorder=3)
        _ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="parity")
        _mv = pd.concat([_d.sort_values("delta").head(4), _d.sort_values("delta").tail(4)])
        for _c, _row in _mv.iterrows():
            _ax.annotate(_c, (_row[_r0], _row[_r1]), fontsize=6, xytext=(3, 3), textcoords="offset points")
        _ax.set(xlim=(0, 1), ylim=(0, 1), xlabel=f"{_r0} dice", ylabel=f"{_r1} dice",
                title=f"{_r1} vs {_r0}  (Δmacro={_macro[_r1] - _macro[_r0]:+.3f})")
        _ax.grid(alpha=0.3)
    _seen = [f for f in SHAPE_ORDER if f in set(_tab["shape"])]
    _axes[0].legend(handles=[Patch(color=_cmap[f], label=f) for f in _seen]
                    + _axes[0].get_legend_handles_labels()[0], fontsize=7, ncol=2, loc="lower right")
    _fig.suptitle("Per-class val dice — pairwise run comparison", y=1.01)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(RUN_NAMES, S, SHAPE_ORDER, np, pd, plt):
    # ── 1b. TRAIN vs HELD-OUT GENERALIZATION — run comparison ────────────────────────────────────
    # Held-out (unseen) classes are the point of in-context seg. For EACH run: macro/micro dice and
    # miss-rate split by membership, plus the train−heldout gap. Held-out dice by shape family and
    # matched lateral-mirror pairs (morphology-fair) compare the runs on the same anatomy.
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
    # runs). Only the largest per-shape movers are text-labelled (no auto text-placement dependency).
    # Below, a membership scatter PER RUN (no labels) shows train vs held-out at a glance.
    import marimo as mo

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
        import matplotlib.ticker as mticker
        lo, hi = int(np.floor(_xmin)), int(np.ceil(_xmax))
        majors = list(range(lo, hi + 1))
        ax.xaxis.set_major_locator(mticker.FixedLocator(majors))
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: rf"$10^{{{int(round(v))}}}$"))
        ax.xaxis.set_minor_locator(
            mticker.FixedLocator([d + np.log10(m) for d in majors for m in range(2, 10)]))

    # ===== FIG 1: per-shape small multiples, both runs + connectors =====
    _shapes = [f for f in SHAPE_ORDER if f in set(_pc["shape"])]
    _ncol = 3; _nrow = int(np.ceil(len(_shapes) / _ncol))
    _fig1, _axes = plt.subplots(_nrow, _ncol, figsize=(6.5 * _ncol, 4.8 * _nrow),
                                sharex=True, sharey=True, squeeze=False)
    _axes = _axes.ravel()
    for _i, _sh in enumerate(_shapes):
        _ax = _axes[_i]
        _sub = _pc[_pc["shape"] == _sh]
        _conn = []
        for _cn, _cc in _sub.groupby("class"):      # connector per class across runs
            if len(_cc) >= 2:
                _ax.plot(_cc.logvol, _cc.dice, "-", color="0.7", lw=0.6, zorder=1)
                _conn.append((_cn, abs(_cc.dice.max() - _cc.dice.min()),
                              _cc.logvol.mean(), _cc.dice.mean()))
        for _r in RUN_NAMES:
            _rr = _sub[_sub.run == _r]
            _ax.scatter(_rr.logvol, _rr.dice, s=34, color=_run_cmap[_r], alpha=0.9,
                        zorder=3, edgecolor="k", linewidth=0.3, label=_r)
        # label only the biggest movers per panel (top 4) to keep it legible w/o textalloc
        for _cn, _mv, _lx, _ly in sorted(_conn, key=lambda t: -t[1])[:4]:
            _ax.annotate(_cn, (_lx, _ly), fontsize=6, xytext=(4, 4), textcoords="offset points")
        _ax.set_xlim(_xlim); _ax.set_ylim(_ylim); _log_ticks(_ax)
        _ax.set_title(f"{_sh} ({_sub['class'].nunique()} cls)", fontsize=9)
        _ax.grid(alpha=0.3, which="both")
    for _j in range(len(_shapes), len(_axes)):
        _axes[_j].set_visible(False)
    _axes[0].legend(fontsize=8, loc="lower right")
    _fig1.supxlabel("mean object volume (voxels, log)")
    _fig1.supylabel("mean per-class dice")
    _fig1.suptitle("Per-class dice vs volume by shape family — run comparison "
                   "(grey line = same class; labels = biggest movers per panel)")
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
    from matplotlib.lines import Line2D
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
    # thick→thin, so this reads as a dice-vs-thickness profile compared across runs; delta columns
    # are vs the FIRST run (150_dualaxis) as reference, one per remaining run.
    _cls = S.groupby(["run", "shape", "class"]).dice.mean()
    _fam = _cls.groupby(["run", "shape"]).mean().unstack("run").reindex(SHAPE_ORDER).dropna(how="all")
    _ncls = S.groupby("shape")["class"].nunique()
    _show = _fam.copy()
    _ref = RUN_NAMES[0]
    for _r in RUN_NAMES[1:]:
        _show[f"Δ_{_r}"] = _fam[_r] - _fam[_ref]
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
    # 150 (dual_axis=True, bi-axial) vs 151 (dual_axis=False, l=9, early PixelShuffle fusion) —
    # compute-matched cold-transformer feature-fusion ablation — plus 152 (medverse, released
    # architecture) as a third, non-ablation architecture-axis arm (see cell 0 header for caveats:
    # different init/recipe deviation). All cells generalise to N runs.
    # cell 0:  load all runs' LOCAL wandb sample tables at the matched EPOCH; shared morphology
    #          clustering + geometry join; header.
    # cell 1:  per-class dice pivoted per run (+ movers-by-range table); pairwise scatter grid (one
    #          panel per run pair) with parity line.
    # cell 1b: train vs held-out per run — membership × run bars (a), held-out dice by shape family (b),
    #          matched lateral-mirror pairs per run (morphology-controlled).
    # cell 1c: per-class dice vs volume — per-shape panels with all runs + connectors (biggest movers
    #          labelled); membership scatter per run (no labels).
    # cell 2:  per-shape family macro dice, grouped bars per run (+ delta-vs-150 columns).
    # cell 3:  per-sample dice vs geometry drivers, one binned-mean line per run.
    # Edit RUNS / EPOCH / N_SHAPE in cell 0. Shape taxonomy + geometry come from
    # totalseg_geometry_extract. Reads LOCAL wandb/run-*-<id> dirs only — no network/API needed.
    print("tables: cells 0-2;  figures: cells 1-3")
    return


if __name__ == "__main__":
    app.run()
