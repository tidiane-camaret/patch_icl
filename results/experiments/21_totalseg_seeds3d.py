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
    from nb_common import ARTIFACTS, get_latest_table
    from totalseg_geometry_extract import load_or_build_geometry, shape_families

    pd.set_option("display.width", 240)
    pd.set_option("display.float_format", lambda v: f"{v:.3f}")

    RUN = "tidiane/patch_icl_3d_exps/d7fk2k9h"
    N_SHAPE = 10                     # morphology cluster count (thick→thin families)
    _id = RUN.split("/")[-1]
    _samp_c = ARTIFACTS / f"21_{_id}_samples.csv"
    _meta_c = ARTIFACTS / f"21_{_id}_meta.json"
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
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        S.to_csv(_samp_c, index=False)
        _meta_c.write_text(json.dumps(META, default=float))
        print(f"fetched {_run.name}: samples {S.shape}")

    # morphology families + per-sample geometry from the evaluated (subject,class) real masks
    GEOM = load_or_build_geometry(S[["subject", "class"]], ARTIFACTS / f"21_{_id}_geometry.csv")
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
def _(S, SHAPE_ORDER, np, pd, plt):
    # ── 1. PER-CLASS VAL DICE (coloured by morphology family) ────────────────────────────────────
    # Every class ranked by mean dice, bar colour = its shape family; macro mean (mean over classes)
    # is the reference line. Table adds sample count, median target size and family.
    _g = S.groupby("class").agg(dice=("dice", "mean"), soft=("soft_dice", "mean"),
                                n=("dice", "size"), tgt_size=("tgt_size", "median")).sort_values("dice")
    _g["shape"] = S.groupby("class").shape.first()
    _macro = _g.dice.mean()
    print(f"{len(_g)} classes | macro dice={_macro:.4f} | micro dice={S.dice.mean():.4f} | "
          f"complete-miss (mean<0.01): {int((_g.dice < 0.01).sum())}")
    print(_g.to_string())

    _pal = (plt.cm.tab20.colors + plt.cm.tab20b.colors)
    _cmap = {f: _pal[i % len(_pal)] for i, f in enumerate(SHAPE_ORDER)}
    _y = np.arange(len(_g))
    _fig, _ax = plt.subplots(figsize=(9, max(6, 0.24 * len(_g))))
    _ax.barh(_y, _g.dice.values, color=[_cmap[s] for s in _g["shape"]])
    _ax.axvline(_macro, color="k", lw=0.8, ls="--", label=f"macro {_macro:.3f}")
    _ax.set_yticks(_y)
    _ax.set_yticklabels([f"{c} (n={int(n)})" for c, n in zip(_g.index, _g.n)], fontsize=7)
    _ax.set_ylim(-0.5, len(_g) - 0.5)
    _ax.set(xlabel="mean val dice (final epoch)", title="Per-class val dice, coloured by shape family")
    # legend: macro line + one swatch per family that actually appears
    from matplotlib.patches import Patch
    _seen = [f for f in SHAPE_ORDER if f in set(_g["shape"])]
    _ax.legend(handles=[Patch(color=_cmap[f], label=f) for f in _seen]
               + _ax.get_legend_handles_labels()[0], fontsize=7, ncol=2)
    _ax.grid(alpha=0.3, axis="x")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(S, SHAPE_ORDER, np, pd, plt):
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
def _(S, np, pd, plt):
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
    # cell 0: fetch + cache (samples / config+summary / geometry) + morphology clustering + header.
    # cell 1: per-class val dice ranked, coloured by shape family (+ table).
    # cell 2: per-shape family breakdown (macro/micro dice, miss rate, thickness) — dice-vs-shape.
    # cell 3: per-sample dice vs geometry drivers (thickness, volume, target/context occupancy).
    # Set RUN (cell 0) to any patch_icl_3d_exps run; N_SHAPE controls the family granularity.
    # Shape taxonomy + geometry come from totalseg_geometry_extract (shared with nb 20).
    print("tables: cells 0-2;  figures: cells 1-3")
    return


if __name__ == "__main__":
    app.run()
