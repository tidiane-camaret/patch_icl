import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
    # ── single 3D training-run: detailed val breakdown (patch_icl_3d_exps) ───────────────────────
    # Point RUN at any training run of the project; the class list, sample set and shape families
    # are all DERIVED from the logged data — no hardcoded values — so this is reusable across runs.
    # Focus is the FINAL-epoch val/samples table, broken down by class, morphology family and
    # per-sample geometry. (Aggregate learning curves live in W&B; not repeated here.)
    #
    # Morphology families are clustered from real-mask geometry via the shared
    # totalseg_geometry_extract helpers (same taxonomy as nb 20). Caches the samples table, the
    # config+summary, and the geometry under artifacts/21_<id>_*.{csv,json}; delete to refresh.
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

    RUN = "tidiane/patch_icl_3d_exps/93cz8fba"
    N_SHAPE = 10                     # morphology cluster count (thick→thin families)
    _id = RUN.split("/")[-1]
    _samp_c = _ARTIFACTS / f"21_{_id}_samples.csv"
    _meta_c = _ARTIFACTS / f"21_{_id}_meta.json"
    if _samp_c.exists() and _meta_c.exists():
        S = pd.read_csv(_samp_c)
        META = json.loads(_meta_c.read_text())
        print(f"loaded cached samples {S.shape}")
    else:
        import wandb
        _run = wandb.Api().run(RUN)
        S = get_latest_table(_run, table_key="val/samples.table.json")
        META = {"name": _run.name, "config": dict(_run.config),
                "summary": {k: _run.summary.get(k) for k in ("epoch", "val/dice", "val/best_dice")}}
        _ARTIFACTS.mkdir(parents=True, exist_ok=True)
        S.to_csv(_samp_c, index=False)
        _meta_c.write_text(json.dumps(META, default=float))
        print(f"fetched {_run.name}: samples {S.shape}")

    # morphology families + per-sample geometry from the evaluated (subject,class) real masks
    GEOM = load_or_build_geometry(S[["subject", "class"]], _ARTIFACTS / f"21_{_id}_geometry.csv")
    SHAPE, SHAPE_ORDER = shape_families(GEOM, k=N_SHAPE)
    S = S.merge(GEOM, on=["subject", "class"], how="left")
    S["shape"] = S["class"].map(SHAPE).fillna("other")
    if (S["shape"] == "other").any():
        SHAPE_ORDER = SHAPE_ORDER + ["other"]

    def _cfg(*path, d=META["config"]):
        for k in path:
            d = d.get(k) if isinstance(d, dict) else None
        return d
    _sm = META["summary"]
    print("  " + "  ".join(f"{k}={v}" for k, v in {
        "model": _cfg("model"), "synth": _cfg("data", "synth_method"),
        "ctx": _cfg("data", "context_size"), "size": _cfg("data", "image_size"),
        "val_classes": _cfg("data", "val_classes")}.items() if v is not None))
    print(f"  epoch {int(_sm['epoch'])} | {S['class'].nunique()} classes / {len(S)} samples | "
          f"val/dice={_sm['val/dice']:.4f}  best={_sm['val/best_dice']:.4f}")
    print(f"  shape families (k={N_SHAPE}, thick→thin): {SHAPE_ORDER}")
    return S, SHAPE_ORDER, np, pd, plt


@app.cell
def _(S, SHAPE_ORDER, np, plt):
    # ── 1. PER-CLASS VAL DICE (coloured by morphology family) ────────────────────────────────────
    # Every class ranked by mean dice, bar colour = its shape family; macro mean (mean over classes)
    # is the reference line. Table adds sample count, median target size and family.
    _g = S.groupby("class").agg(dice=("dice", "mean"), soft=("soft_dice", "mean"),
                                n=("dice", "size"), tgt_size=("tgt_size", "median")).sort_values("dice")
    _g["shape"] = S.groupby("class").shape.first()
    _g["in_train"] = S.groupby("class").in_train.first()  # class seen during training?
    _macro = _g.dice.mean()
    print(f"{len(_g)} classes | macro dice={_macro:.4f} | micro dice={S.dice.mean():.4f} | "
          f"complete-miss (mean<0.01): {int((_g.dice < 0.01).sum())}")
    print(_g.to_string())

    _pal = (plt.cm.tab20.colors + plt.cm.tab20b.colors)
    _cmap = {f: _pal[i % len(_pal)] for i, f in enumerate(SHAPE_ORDER)}
    _y = np.arange(len(_g))
    _fig, _ax = plt.subplots(figsize=(9, max(6, 0.24 * len(_g))))
    # bar fill = shape family; hatch marks held-out (unseen) classes so train/held-out is legible
    _ax.barh(_y, _g.dice.values, color=[_cmap[s] for s in _g["shape"]],
             hatch=["" if t else "///" for t in _g["in_train"]], edgecolor="k", linewidth=0.3)
    _ax.axvline(_macro, color="k", lw=0.8, ls="--", label=f"macro {_macro:.3f}")
    _ax.set_yticks(_y)
    # trailing "*" flags held-out classes not seen in training
    _ax.set_yticklabels([f"{c}{'' if t else ' *'} (n={int(n)})"
                         for c, t, n in zip(_g.index, _g["in_train"], _g.n)], fontsize=7)
    _ax.set_ylim(-0.5, len(_g) - 0.5)
    _ax.set(xlabel="mean val dice (final epoch)", title="Per-class val dice, coloured by shape family")
    # legend: macro line + one swatch per family that actually appears
    from matplotlib.patches import Patch
    _seen = [f for f in SHAPE_ORDER if f in set(_g["shape"])]
    _ax.legend(handles=[Patch(color=_cmap[f], label=f) for f in _seen]
               + [Patch(facecolor="white", hatch="///", edgecolor="k", label="held-out (unseen)")]
               + _ax.get_legend_handles_labels()[0], fontsize=7, ncol=2)
    _ax.grid(alpha=0.3, axis="x")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(S, SHAPE_ORDER, np, pd, plt):
    # ── 1b. TRAIN vs HELD-OUT CLASS GENERALIZATION ──────────────────────────────────────────────
    # The point of in-context segmentation is generalising to classes UNSEEN in training. `in_train`
    # splits the eval set into seen vs held-out when both are present. Some runs contain only seen
    # classes, so the comparison below degrades gracefully to train-only summaries instead of failing.
    def _macro(g):
        return g.groupby("class").dice.mean().mean()

    _ov = S.groupby("in_train").apply(lambda g: pd.Series({
        "n_cls": g["class"].nunique(), "n_samp": len(g),
        "macro_dice": _macro(g), "micro_dice": g.dice.mean(),
        "miss_rate": (g.dice < 0.01).mean(), "med_tgt_size": g.tgt_size.median(),
    }), include_groups=False).rename(index={True: "train", False: "held-out"})
    print("overall (in_train=True → seen, False → held-out when present):")
    print(_ov.T.to_string())

    if {"train", "held-out"}.issubset(_ov.index):
        _raw = _ov.loc["train", "macro_dice"] - _ov.loc["held-out", "macro_dice"]
        print(f"  RAW macro gap = {_raw:+.3f}  ← confounded: held-out med tgt_size "
              f"{_ov.loc['held-out','med_tgt_size']:.0f} vs {_ov.loc['train','med_tgt_size']:.0f} (smaller)")
    else:
        print("  no held-out classes in this run; skipping train-vs-held-out gap")

    # (a) within shape family — same morphology bucket, split by membership
    _sf = (S.groupby(["shape", "in_train", "class"]).dice.mean()
           .groupby(["shape", "in_train"]).mean().unstack().reindex(SHAPE_ORDER))
    _sf = _sf.rename(columns={True: "train", False: "heldout"})
    if "heldout" in _sf.columns:
        _sf["delta"] = _sf["train"] - _sf["heldout"]
    else:
        _sf["delta"] = np.nan
    print("\n(a) macro dice by shape family × membership (thick→thin):")
    print(_sf.to_string())

    # (c) matched lateral-mirror pairs: same organ, trained side vs held-out side (morphology-fair)
    _cls = S.groupby("class").agg(dice=("dice", "mean"), in_train=("in_train", "first")).to_dict("index")
    def _flip(c):
        return (c.replace("_left", "_TMP").replace("_right", "_left").replace("_TMP", "_right")
                if ("_left" in c or "_right" in c) else None)
    _rows = []
    for _c, _r in _cls.items():
        _o = _flip(_c)
        if _o in _cls and _r["in_train"] and not _cls[_o]["in_train"]:
            _organ = _c.replace("_left", "").replace("_right", "")
            _rows.append((_organ, _r["dice"], _cls[_o]["dice"]))
    _M = pd.DataFrame(_rows, columns=["organ", "trained_side", "heldout_side"])
    _M["delta"] = _M.trained_side - _M.heldout_side
    print(f"\n(c) matched lateral-mirror pairs (n={len(_M)}, same organ → morphology controlled):")
    if len(_M):
        print(_M.sort_values("delta").to_string(index=False))
        print(f"  mean trained={_M.trained_side.mean():.3f} held-out={_M.heldout_side.mean():.3f} "
              f"delta={_M.delta.mean():+.3f}  → near-parity when anatomy is matched")
    else:
        print("  no matched train/held-out mirror pairs in this run")

    _fig, (_a0, _a1) = plt.subplots(1, 2, figsize=(14, 5))
    # left: macro dice per shape family, train vs held-out side by side when available
    _sd = _sf.dropna(how="all", subset=[c for c in ("train", "heldout") if c in _sf.columns])
    _x = np.arange(len(_sd))
    if "train" in _sd.columns:
        _a0.bar(_x - (0.2 if "heldout" in _sd.columns else 0.0), _sd["train"], 0.4 if "heldout" in _sd.columns else 0.6,
                color="tab:blue", label="train (seen)")
    if "heldout" in _sd.columns:
        _a0.bar(_x + 0.2, _sd["heldout"], 0.4, color="tab:orange", label="held-out (unseen)")
    _a0.set_xticks(_x); _a0.set_xticklabels(_sd.index, rotation=45, ha="right", fontsize=7)
    _a0.set(ylabel="macro val dice", title="(a) dice by shape family × membership")
    if len(_sd.columns):
        _a0.legend(fontsize=8)
    _a0.grid(alpha=0.3, axis="y")
    # right: matched-mirror scatter, points below diagonal = held-out side better
    if len(_M):
        _a1.scatter(_M.trained_side, _M.heldout_side, s=40, color="tab:purple")
        for _, _p in _M.iterrows():
            _a1.annotate(_p.organ, (_p.trained_side, _p.heldout_side), fontsize=6,
                         xytext=(3, 3), textcoords="offset points")
    _lim = [0, 1]
    _a1.plot(_lim, _lim, "k--", lw=0.8, label="parity")
    _delta_txt = f"{_M.delta.mean():+.3f}" if len(_M) else "n/a"
    _a1.set(xlim=_lim, ylim=_lim, xlabel="trained side dice", ylabel="held-out side dice",
            title=f"(c) matched mirror pairs (Δ={_delta_txt})")
    _a1.legend(fontsize=8); _a1.grid(alpha=0.3)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(S, SHAPE_ORDER):
    # ── 1c. PER-CLASS DICE vs VOLUME (labels via textalloc) ──────────────────────────────────────
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    import textalloc as ta

    # ---- aggregate per class ----
    _pc = (S.groupby("class")
             .agg(dice=("dice", "mean"),
                  volume=("volume", "mean"),
                  in_train=("in_train", "first"),
                  shape=("shape", "first"))
             .dropna(subset=["volume"]))
    _pc = _pc[_pc.volume > 0].sort_index()

    # Work in log10(volume) on a LINEAR axis, then relabel ticks as 10^n.
    # This makes textalloc's placement correct and evenly spread.
    _pc["logvol"] = np.log10(_pc["volume"].values)
    _names = _pc.index.tolist()
    _lx = _pc["logvol"].values
    _dy = _pc["dice"].values

    # ---- fixed bounds ----
    _xmin, _xmax = float(_lx.min()), float(_lx.max())
    _ymin, _ymax = float(_dy.min()), float(_dy.max())
    _xpad = 0.05 * (_xmax - _xmin + 1e-9)
    _ypad = max(0.03, 0.08 * (_ymax - _ymin + 1e-9))
    _xlim = (_xmin - _xpad, _xmax + _xpad)
    _ylim = (max(0.0, _ymin - _ypad), min(1.0, _ymax + _ypad))

    # ---- colour maps ----
    _shape_pal = plt.cm.tab20.colors + plt.cm.tab20b.colors
    _shape_cmap = {f: _shape_pal[i % len(_shape_pal)] for i, f in enumerate(SHAPE_ORDER)}
    _mem_cmap = {True: "tab:blue", False: "tab:orange"}

    def _log_ticks(ax):
        """Give the linear (log10) axis proper 10^n majors and log-style minors."""
        lo, hi = int(np.floor(_xmin)), int(np.ceil(_xmax))
        majors = list(range(lo, hi + 1))
        ax.xaxis.set_major_locator(mticker.FixedLocator(majors))
        ax.xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda v, _: rf"$10^{{{int(round(v))}}}$"))
        minors = [d + np.log10(m) for d in majors for m in range(2, 10)]
        ax.xaxis.set_minor_locator(mticker.FixedLocator(minors))

    def _allocate_labels(ax):
        """Run textalloc on an axis whose limits/scale are already set."""
        ta.allocate(
            ax, _lx, _dy, _names,
            x_scatter=_lx, y_scatter=_dy,   # avoid the data points
            textsize=7,
            margin=0.008,                   # padding around each label (axis fraction)
            min_distance=0.012,             # min gap label→point
            max_distance=0.30,              # how far a label may travel into empty space
            linewidth=0.5,
            linecolor="0.45",
            draw_lines=True,                # connector lines
            nbr_candidates=500,             # more candidate slots → tighter packing
            draw_all=True,                  # force all 115 to appear
        )

    # ---- figure (bigger canvas = more room for 115 labels) ----
    _fig, (_a0, _a1) = plt.subplots(1, 2, figsize=(24, 13), sharey=True)

    # ================= LEFT: shape-coloured =================
    _shape_colors = _pc["shape"].map(lambda s: _shape_cmap.get(s, "tab:gray")).tolist()
    _a0.scatter(_lx, _dy, s=32, c=_shape_colors, alpha=0.85, zorder=3)
    _a0.set_xlim(_xlim); _a0.set_ylim(_ylim)
    _log_ticks(_a0)
    _allocate_labels(_a0)

    _seen = [f for f in SHAPE_ORDER if f in set(_pc["shape"])]
    if "other" in set(_pc["shape"]) and "other" not in _seen:
        _seen.append("other")
    _a0.legend(handles=[Patch(color=_shape_cmap.get(f, "tab:gray"), label=f) for f in _seen],
               title="shape", fontsize=7, title_fontsize=8, loc="upper left", frameon=True)
    _a0.set(xlabel="mean object volume (voxels, log)", ylabel="mean per-class dice",
            title=f"per-class dice vs volume — shape coloured ({len(_pc)} labels)")
    _a0.grid(alpha=0.3, which="both")

    # ================= RIGHT: membership-coloured =================
    _mem_colors = _pc["in_train"].map(_mem_cmap).tolist()
    _a1.scatter(_lx, _dy, s=32, c=_mem_colors, alpha=0.85, zorder=3)
    _a1.set_xlim(_xlim); _a1.set_ylim(_ylim)
    _log_ticks(_a1)
    _allocate_labels(_a1)

    _a1.legend(handles=[Line2D([], [], marker="o", ls="", color=_mem_cmap[True], label="train (seen)"),
                        Line2D([], [], marker="o", ls="", color=_mem_cmap[False], label="held-out")],
               fontsize=9, loc="upper left")
    _a1.set(xlabel="mean object volume (voxels, log)",
            title="per-class dice vs volume — membership")
    _a1.grid(alpha=0.3, which="both")

    _fig.subplots_adjust(left=0.045, right=0.985, bottom=0.06, top=0.95, wspace=0.10)
    _fig
    return np, plt


@app.cell
def _(S, SHAPE_ORDER, np, plt):
    # ── 2. PER-SHAPE FAMILY BREAKDOWN ───────────────────────────────────────────────────────────
    # Dice aggregated by morphology family (SHAPE_ORDER is thick→thin, so this reads as a
    # dice-vs-thickness profile). MACRO = mean over classes (so many-class families don't dominate);
    # micro = per-sample mean; miss = per-sample complete-miss rate; thick_p90 = family median.
    _cls = S.groupby(["shape", "class"]).dice.mean()  # Series (shape,class) -> mean dice
    _fam = _cls.groupby("shape").agg(n_cls="size", macro="mean").reindex(SHAPE_ORDER)
    _fam["n_samp"] = S.groupby("shape").size().reindex(SHAPE_ORDER)
    _fam["micro"] = S.groupby("shape").dice.mean().reindex(SHAPE_ORDER)
    _fam["miss"] = S.groupby("shape").apply(lambda g: (g.dice < 0.01).mean(),
                                            include_groups=False).reindex(SHAPE_ORDER)
    _fam["thick_p90"] = S.groupby("shape").thick_p90.median().reindex(SHAPE_ORDER)
    _fam = _fam.dropna(subset=["n_cls"])
    print("dice by shape family (macro over classes):\n" + _fam.to_string())

    _x = np.arange(len(_fam))
    _fig, _ax = plt.subplots(figsize=(max(9, 1.1 * len(_fam)), 5))
    _ax.bar(_x - 0.2, _fam.macro, 0.4, color="tab:blue", label="macro dice")
    _ax.bar(_x + 0.2, _fam.micro, 0.4, color="tab:cyan", label="micro dice")
    for _i, (_m, _mi, _ms) in enumerate(zip(_fam.macro, _fam.micro, _fam.miss)):
        _ax.text(_i - 0.2, _m + .002, f"{_m:.3f}", ha="center", va="bottom", fontsize=7)
        _ax.text(_i, -0.006, f"miss {_ms:.0%}", ha="center", va="top", fontsize=6, color="gray")
    _ax.set_xticks(_x)
    _ax.set_xticklabels([f"{s}\n({int(c)} cls, n={int(n)})" for s, c, n in
                         zip(_fam.index, _fam.n_cls, _fam.n_samp)], fontsize=7)
    _ax.set(ylabel="val dice", title="Val dice by morphology family (thick→thin)")
    _ax.legend(fontsize=8); _ax.grid(alpha=0.3, axis="y")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(S, pd, plt):
    # ── 3. PER-SAMPLE DICE vs GEOMETRY DRIVERS ──────────────────────────────────────────────────
    # Where does the run succeed/fail at the sample level? Dice vs four drivers: object thickness
    # (interior-EDT p90), target volume, target occupancy, and CONTEXT occupancy (does a fuller
    # in-context example help?). Scatter (per sample) + mean over quantile bins (line).
    _drivers = [("thick_p90", True), ("volume", True), ("tgt_occ", True), ("ctx_occ", True)]
    _fig, _axes = plt.subplots(2, 2, figsize=(13, 9))
    _axes = _axes.ravel()
    for _k, (_f, _logx) in enumerate(_drivers):
        _ax = _axes[_k]
        if _f not in S.columns:
            _ax.set_visible(False); continue
        _d = S[[_f, "dice"]].dropna()
        _d = _d[_d[_f] > 0] if _logx else _d
        _ax.scatter(_d[_f], _d.dice, s=10, alpha=0.3, color="tab:blue")
        try:
            _b = pd.qcut(_d[_f], 8, duplicates="drop")
            _grp = _d.groupby(_b, observed=True)
            _ax.plot(_grp[_f].median(), _grp.dice.mean(), "-o", color="tab:red", ms=4,
                     label="binned mean")
            _ax.legend(fontsize=8)
        except (ValueError, IndexError):
            pass
        if _logx:
            _ax.set_xscale("log")
        _ax.set(xlabel=_f, ylabel="dice", title=f"dice vs {_f}")
        _ax.grid(alpha=0.3)
    _fig.suptitle("Per-sample dice vs geometry drivers", y=1.002)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _():
    # ── Contents ─────────────────────────────────────────────────────────────────────────────────
    # cell 0:  fetch + cache (samples / config+summary / geometry) + morphology clustering + header.
    # cell 1:  per-class val dice ranked, coloured by shape family, held-out classes hatched (+ table).
    # cell 1b: train vs held-out generalisation — raw gap is confounded by size; controlled within
    #          shape family (a), and via matched lateral-mirror pairs (c) → near-parity when matched.
    # cell 1c: per-class dice vs volume, labelled scatter ×2 (by class / by train membership).
    # cell 2:  per-shape family breakdown (macro/micro dice, miss rate, thickness) — dice-vs-shape.
    # cell 3:  per-sample dice vs geometry drivers (thickness, volume, target/context occupancy).
    # Set RUN (cell 0) to any patch_icl_3d_exps run; N_SHAPE controls the family granularity.
    # Shape taxonomy + geometry come from totalseg_geometry_extract (shared with nb 20).
    print("tables: cells 0-2;  figures: cells 1-3")
    return


if __name__ == "__main__":
    app.run()
