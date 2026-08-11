# /// script
# requires-python = ">=3.8"
# dependencies = [
#     "marimo>=0.8.22",
#     "matplotlib>=3.7.5",
#     "numpy>=1.24.4",
#     "pandas>=2.0.3",
# ]
# ///

import marimo

__generated_with = "0.23.16"
app = marimo.App(width="medium")


@app.cell
def _():
    # ── single-run coarse→fine spacing sweep + locator (patch_icl_3d_eval) ────────────────────────
    # One eval run logged eval.spacing_sweep=[4,1.5] with eval.spacing_locator=true. Its cases.table
    # carries a per-sample `spacing` column, so 4mm (coarse) and 1.5mm (fine) are two conditions in ONE
    # table — the comparison axis here (delta = fine − coarse = the refinement gain), mirroring nb 36's
    # two-run axis. Per-class LOCATOR containment (coarse crop vs oracle) is NOT per-sample; it lives in
    # run.summary as class/<c>/containment@4 + containment_oracle@4, pulled alongside. Geometry + shape
    # families are derived from the evaluated real masks (as in nb 36). Cache under artifacts/37_<id>_*.
    import json
    import re
    import sys
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    sys.path.insert(0, str(Path(__file__).parent))
    from nb_common import ARTIFACTS as _ARTIFACTS, get_latest_table, latest_table_version
    from totalseg_geometry_extract import load_or_build_geometry, shape_families

    pd.set_option("display.width", 240)
    pd.set_option("display.float_format", lambda v: f"{v:.3f}")

    PROJECT = "tidiane/patch_icl_3d_eval"
    RUN_ID = "05kb6kcc"              # patchset3d, occupancy masks, spacing_sweep=[4,1.5], locator
    N_SHAPE = 10                     # morphology cluster count (thick→thin families)
    CHECK_FRESH = True               # ping W&B on cache hit; refetch if a newer table version exists

    def _parse_locator(summary):
        """Pull per-class locator containment (+ per-spacing dice) from a run summary dict.

        The sweep logs class/<c>/{mean_dice@4,mean_dice@1.5} and, for the locator, class/<c>/
        {containment@4,containment_oracle@4}. Returns a tidy DataFrame keyed by class."""
        rows = {}
        pat = re.compile(r"class/(.+?)/(containment@4|containment_oracle@4)$")
        for k, v in summary.items():
            m = pat.match(k)
            if m:
                rows.setdefault(m.group(1), {})[m.group(2)] = v
        df = pd.DataFrame(
            [{"class": c, "containment": d.get("containment@4"),
              "oracle": d.get("containment_oracle@4")} for c, d in rows.items()])
        return df

    def _load():
        """Load the run's cached cases table + meta (per-class locator + aggregates), refetching from
        W&B on cache miss or when CHECK_FRESH and a newer cases-table version exists."""
        _cases_c = _ARTIFACTS / f"37_{RUN_ID}_cases.csv"
        _meta_c = _ARTIFACTS / f"37_{RUN_ID}_meta.json"
        _run = None
        if _cases_c.exists() and _meta_c.exists():
            _m = json.loads(_meta_c.read_text())
            if not CHECK_FRESH:
                return pd.read_csv(_cases_c), _m
            import wandb
            _run = wandb.Api().run(f"{PROJECT}/{RUN_ID}")
            if latest_table_version(_run) <= _m.get("table_version", -1):
                return pd.read_csv(_cases_c), _m           # cache matches latest → no download
            print(f"  [{RUN_ID}] newer table on W&B (cached v{_m.get('table_version', -1)}) → refetching")
        if _run is None:
            import wandb
            _run = wandb.Api().run(f"{PROJECT}/{RUN_ID}")
        _s, _ver = get_latest_table(_run, table_key="cases.table.json", return_version=True)
        _summary = dict(_run.summary)
        _agg = {k: _summary.get(k) for k in ("mean_dice", "mean_dice@4", "mean_dice@1.5",
                                             "mean_time_ms", "gflops")}
        _m = {"name": _run.name, "config": dict(_run.config), "table_version": _ver,
              "agg": _agg, "locator": _parse_locator(_summary).to_dict("records")}
        _ARTIFACTS.mkdir(parents=True, exist_ok=True)
        _s.to_csv(_cases_c, index=False)
        _meta_c.write_text(json.dumps(_m, default=float))
        return _s, _m

    S, META = _load()
    AGG = META["agg"]
    LOC = pd.DataFrame(META["locator"])
    print(f"loaded {RUN_ID}: cases {S.shape}  | {META['name']}")

    # spacings live in the per-sample table; coarse = max, fine = min (the sweep was [4, 1.5]).
    SPACINGS = sorted(S["spacing"].dropna().unique(), reverse=True)
    COARSE, FINE = float(SPACINGS[0]), float(SPACINGS[-1])

    # shape families + per-sample geometry from ALL evaluated (subject,class) real masks (as in nb 36).
    _pairs = S[["subject", "class"]].drop_duplicates()
    GEOM = load_or_build_geometry(_pairs, _ARTIFACTS / f"37_{RUN_ID}_geometry.csv")
    SHAPE, SHAPE_ORDER = shape_families(GEOM, k=N_SHAPE)
    S = S.merge(GEOM, on=["subject", "class"], how="left")
    S["shape"] = S["class"].map(SHAPE).fillna("other")
    if (S["shape"] == "other").any():
        SHAPE_ORDER = SHAPE_ORDER + ["other"]
    LOC = LOC.merge(S.groupby("class").agg(shape=("shape", "first"),
                                           thick_p90=("thick_p90", "median")).reset_index(),
                    on="class", how="left")
    # Carry the per-class locator metrics onto every sample so the fine-accuracy cells (which work on
    # the per-sample fine dice) can group/colour by coarse containment + oracle without a re-merge.
    S = S.merge(LOC[["class", "containment", "oracle"]], on="class", how="left")

    def _macro(g):
        return g.groupby("class").dice.mean().mean()
    print(f"  spacings (coarse→fine): {COARSE:g} → {FINE:g} mm  "
          f"| {S['class'].nunique()} classes / {len(S)} samples")
    for _sp in SPACINGS:
        _g = S[S.spacing == _sp]
        print(f"    @{_sp:g}mm  macro={_macro(_g):.4f}  micro={_g.dice.mean():.4f}  n={len(_g)}")
    print(f"  aggregate (from summary): mean_dice@{COARSE:g}={AGG.get('mean_dice@4')}  "
          f"@{FINE:g}={AGG.get('mean_dice@1.5')}  gflops={AGG.get('gflops')}")
    if len(LOC):
        print(f"  locator@{COARSE:g}mm: mean containment={LOC.containment.mean():.4f}  "
              f"oracle={LOC.oracle.mean():.4f}  gap={LOC.oracle.mean() - LOC.containment.mean():.4f}"
              f"  | oracle<1 (crop-size ceiling): {(LOC.oracle < 0.999).sum()}/{len(LOC)} classes")
    print(f"  in_train: {S[S.in_train].groupby('class').ngroups} trained / "
          f"{S[~S.in_train].groupby('class').ngroups} held-out classes")
    print(f"  shape families (k={N_SHAPE}, thick→thin): {SHAPE_ORDER}")
    return COARSE, FINE, LOC, S, SHAPE_ORDER, np, pd, plt


@app.cell
def _(FINE, S, SHAPE_ORDER, np, pd, plt):
    # ── 1. FINE ACCURACY: TRAINED vs HELD-OUT ────────────────────────────────────────────────────
    # Distribution of the FINE (@{FINE}mm) per-sample dice split by membership (in_train). NOTE the raw
    # split is confounded: the held-out classes happen to be easier anatomy, so held-out dice reads
    # HIGHER. Two controls make the comparison morphology-fair: (a) macro dice by shape family × member-
    # ship (same anatomy family, both memberships), and (b) matched lateral-mirror pairs (same organ,
    # trained side vs held-out side). Panels: violin/box + strip, ECDF, and shape-family bars.
    _SF = S[S.spacing == FINE].copy()
    _SF["mem"] = _SF.in_train.map({True: "trained", False: "held-out"})

    def _macro(g):
        return g.groupby("class").dice.mean().mean()
    _ov = _SF.groupby("mem").apply(lambda g: pd.Series({
        "n_cls": g["class"].nunique(), "n_samp": len(g),
        "macro": _macro(g), "micro": g.dice.mean(), "median": g.dice.median(),
        "miss_rate": (g.dice < 0.01).mean(), "med_thick": g.thick_p90.median(),
    }), include_groups=False)
    print(f"fine (@{FINE:g}mm) dice by membership (RAW — confounded by anatomy):\n{_ov.to_string()}")
    if {"trained", "held-out"} <= set(_ov.index):
        print(f"  raw macro gap trained−heldout = {_ov.loc['trained','macro'] - _ov.loc['held-out','macro']:+.3f}"
              f"  (heldout med thick {_ov.loc['held-out','med_thick']:.1f} vs trained {_ov.loc['trained','med_thick']:.1f})")

    # control (a): held-out advantage per shape family (fair anatomy)
    _fam = (_SF.groupby(["shape", "mem", "class"]).dice.mean()
               .groupby(["shape", "mem"]).mean().unstack("mem").reindex(SHAPE_ORDER).dropna(how="all"))
    # control (b): matched lateral-mirror pairs (same organ, trained side vs held-out side)
    def _flip(c):
        return (c.replace("_left", "_TMP").replace("_right", "_left").replace("_TMP", "_right")
                if ("_left" in c or "_right" in c) else None)
    _cls = _SF.groupby("class").agg(dice=("dice", "mean"), in_train=("in_train", "first")).to_dict("index")
    _mrows = [(c.replace("_left", "").replace("_right", ""), r["dice"], _cls[_flip(c)]["dice"])
              for c, r in _cls.items()
              if _flip(c) in _cls and r["in_train"] and not _cls[_flip(c)]["in_train"]]
    _M = pd.DataFrame(_mrows, columns=["organ", "trained_side", "heldout_side"])
    if len(_M):
        _M["delta"] = _M.trained_side - _M.heldout_side
        print(f"\nmatched lateral-mirror pairs (morphology-controlled, n={len(_M)}):"
              f"  trained={_M.trained_side.mean():.3f}  heldout={_M.heldout_side.mean():.3f}"
              f"  Δ={_M.delta.mean():+.3f}")

    _mems = [m for m in ["trained", "held-out"] if m in _SF.mem.values]
    _mc = {"trained": "tab:blue", "held-out": "tab:orange"}
    _fig, (_a0, _a1, _a2) = plt.subplots(1, 3, figsize=(17, 5.5))
    # (a) box + jittered strip of per-sample fine dice
    _bx = _a0.boxplot([_SF[_SF.mem == m].dice.values for m in _mems], tick_labels=_mems,
                      showmeans=True, widths=0.5)
    for _i, _m in enumerate(_mems):
        _y = _SF[_SF.mem == _m].dice.values
        _a0.scatter(np.random.normal(_i + 1, 0.06, len(_y)), _y, s=5, alpha=0.15, color=_mc[_m], zorder=1)
    _a0.set(ylabel=f"fine dice @{FINE:g}mm", title="(a) per-sample fine dice by membership", ylim=(-0.02, 1.02))
    _a0.grid(alpha=0.3, axis="y")
    # (b) ECDF
    for _m in _mems:
        _y = np.sort(_SF[_SF.mem == _m].dice.values)
        _a1.plot(_y, np.arange(1, len(_y) + 1) / len(_y), color=_mc[_m], label=_m, lw=1.8)
    _a1.set(xlabel=f"fine dice @{FINE:g}mm", ylabel="ECDF", title="(b) fine dice ECDF by membership")
    _a1.legend(fontsize=8); _a1.grid(alpha=0.3)
    # (c) macro dice by shape family × membership (thick→thin) — the anatomy-fair control
    _x = np.arange(len(_fam)); _w = 0.8 / max(len(_mems), 1)
    for _i, _m in enumerate(_mems):
        if _m in _fam.columns:
            _a2.bar(_x + (_i - (len(_mems) - 1) / 2) * _w, _fam[_m].values, _w, label=_m, color=_mc[_m])
    _a2.set_xticks(_x); _a2.set_xticklabels(_fam.index, rotation=45, ha="right", fontsize=7)
    _a2.set(ylabel=f"macro fine dice @{FINE:g}mm", title="(c) fine dice by shape family × membership (thick→thin)")
    _a2.legend(fontsize=8); _a2.grid(alpha=0.3, axis="y")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(COARSE, FINE, S, np, pd, plt):
    # ── 2. FINE ACCURACY vs COARSE CONTAINMENT ───────────────────────────────────────────────────
    # Does a well-framed coarse crop (high containment@{COARSE}mm) buy a better fine result? Per class,
    # containment (x) vs fine dice (y), coloured by object volume. The marginal correlation is WEAK/
    # slightly negative: the low-containment classes are large/elongated organs (colon, autochthon,
    # aorta) that segment fine anyway (containment↔volume is negative), so containment does NOT predict
    # accuracy. Panel (b) bins samples by containment and shows fine dice + median volume per bin.
    _cls = S[S.spacing == FINE].groupby("class").agg(dice_fine=("dice", "mean")).reset_index()
    _cls = _cls.merge(S.groupby("class").agg(containment=("containment", "first"),
                                             oracle=("oracle", "first"),
                                             volume=("volume", "median"),
                                             in_train=("in_train", "first")).reset_index(), on="class")
    _cls = _cls.dropna(subset=["containment", "dice_fine"])
    _rho = _cls.containment.corr(_cls.dice_fine, method="spearman")
    _rho_v = _cls.containment.corr(np.log10(_cls.volume.clip(lower=1)), method="spearman")
    print(f"per-class Spearman ρ:  containment↔fine_dice = {_rho:+.2f}   "
          f"containment↔log_volume = {_rho_v:+.2f}  (low containment = large objects, not low accuracy)")
    print("\nlowest-containment classes (coarse crop drops most GT):")
    print(_cls.sort_values("containment").head(10)
             [["class", "containment", "oracle", "dice_fine", "volume"]].to_string(index=False))

    _fig, (_a0, _a1) = plt.subplots(1, 2, figsize=(15, 6.5))
    # (a) scatter containment vs fine dice, colour = log volume
    _sc = _a0.scatter(_cls.containment, _cls.dice_fine, s=44,
                      c=np.log10(_cls.volume.clip(lower=1)), cmap="viridis",
                      alpha=0.9, edgecolor="k", linewidth=0.3, zorder=3)
    for _, _r in _cls.sort_values("containment").head(8).iterrows():
        _a0.annotate(_r["class"], (_r.containment, _r.dice_fine), fontsize=6,
                     xytext=(3, 3), textcoords="offset points")
    _fig.colorbar(_sc, ax=_a0, label="log10 volume")
    _a0.set(xlabel=f"coarse containment @{COARSE:g}mm", ylabel=f"fine dice @{FINE:g}mm",
            title=f"(a) fine dice vs coarse containment  (ρ={_rho:+.2f}; colour=volume)")
    _a0.grid(alpha=0.3)
    # (b) per-sample fine dice by containment bin + median volume annotation
    _SF = S[S.spacing == FINE].dropna(subset=["containment"]).copy()
    _SF["cbin"] = pd.qcut(_SF.containment, 4, duplicates="drop")
    _bins = list(_SF["cbin"].cat.categories)
    _a1.boxplot([_SF[_SF.cbin == b].dice.values for b in _bins],
                tick_labels=[f"{b.left:.2f}–{b.right:.2f}" for b in _bins], showmeans=True)
    for _i, _b in enumerate(_bins):
        _mv = _SF[_SF.cbin == _b].volume.median()
        _a1.annotate(f"vol≈{_mv:.0f}", (_i + 1, 0.02), fontsize=7, ha="center", color="0.4")
    _a1.set(xlabel=f"coarse containment @{COARSE:g}mm bin", ylabel=f"fine dice @{FINE:g}mm",
            title="(b) fine dice by containment quartile (median volume annotated)", ylim=(-0.02, 1.02))
    _a1.grid(alpha=0.3, axis="y")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(COARSE, FINE, S, np, plt):
    # ── 3. WHEN THE ORACLE FAILS TO CONTAIN (crop-size ceiling) ──────────────────────────────────
    # oracle@{COARSE}mm < 1 means even a GT-centred fine crop cannot hold the whole object — the fine
    # window is too small for it (elongated/large anatomy). Does that ceiling cap fine accuracy? Split
    # classes into fully-containable (oracle≈1) vs ceiling (oracle<1) and compare fine-dice
    # distributions; then relate ceiling severity (1−oracle) to fine dice, coloured by object volume.
    # The catch: oracle<1 marks LARGE objects, which are easier, so the ceiling group scores HIGHER —
    # i.e. the sweep window is not the accuracy bottleneck for the classes that actually fail.
    _cls = S[S.spacing == FINE].groupby("class").agg(dice_fine=("dice", "mean")).reset_index()
    _cls = _cls.merge(S.groupby("class").agg(oracle=("oracle", "first"),
                                             volume=("volume", "median"),
                                             thick=("thick_p90", "median")).reset_index(), on="class")
    _cls = _cls.dropna(subset=["oracle", "dice_fine"])
    _cls["group"] = np.where(_cls.oracle >= 0.999, "fully-containable\n(oracle≈1)", "ceiling\n(oracle<1)")
    _cls["ceiling"] = 1.0 - _cls.oracle
    _rho = _cls.ceiling.corr(_cls.dice_fine, method="spearman")
    _rho_v = _cls.ceiling.corr(np.log10(_cls.volume.clip(lower=1)), method="spearman")
    print(_cls.groupby("group").dice_fine.agg(["mean", "median", "count"]).to_string())
    print(f"\nSpearman ρ:  ceiling(1−oracle)↔fine_dice = {_rho:+.2f}   "
          f"ceiling↔log_volume = {_rho_v:+.2f}  (oracle fails on LARGE objects)")
    print("\nworst crop-size ceiling (oracle drops most GT even when perfectly centred) — top 10:")
    print(_cls.sort_values("oracle").head(10)
             [["class", "oracle", "dice_fine", "volume", "thick"]].to_string(index=False))

    _grp_c = {"fully-containable\n(oracle≈1)": "tab:green", "ceiling\n(oracle<1)": "tab:purple"}
    _groups = [g for g in _grp_c if g in _cls.group.values]
    _fig, (_a0, _a1) = plt.subplots(1, 2, figsize=(15, 6.5))
    # (a) fine-dice distribution: fully-containable vs ceiling (box + strip)
    _bx = _a0.boxplot([_cls[_cls.group == g].dice_fine.values for g in _groups],
                      tick_labels=[g for g in _groups], showmeans=True, widths=0.5)
    for _i, _g in enumerate(_groups):
        _y = _cls[_cls.group == _g].dice_fine.values
        _a0.scatter(np.random.normal(_i + 1, 0.06, len(_y)), _y, s=18, alpha=0.5,
                    color=_grp_c[_g], zorder=3, edgecolor="k", linewidth=0.2)
    _a0.set(ylabel=f"per-class fine dice @{FINE:g}mm",
            title="(a) fine dice: fully-containable vs crop-size ceiling", ylim=(-0.02, 1.02))
    _a0.grid(alpha=0.3, axis="y")
    # (b) ceiling severity vs fine dice, colour = log volume
    _sc = _a1.scatter(_cls.ceiling, _cls.dice_fine, s=44, c=np.log10(_cls.volume.clip(lower=1)),
                      cmap="magma", alpha=0.9, edgecolor="k", linewidth=0.3, zorder=3)
    for _, _r in _cls.sort_values("oracle").head(8).iterrows():
        _a1.annotate(_r["class"], (_r.ceiling, _r.dice_fine), fontsize=6,
                     xytext=(3, 3), textcoords="offset points")
    _fig.colorbar(_sc, ax=_a1, label="log10 volume")
    _a1.set(xlabel=f"crop-size ceiling severity  1 − oracle@{COARSE:g}mm", ylabel=f"fine dice @{FINE:g}mm",
            title=f"(b) ceiling severity vs fine dice  (ρ={_rho:+.2f}; colour=volume)")
    _a1.grid(alpha=0.3)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _():
    # ── Contents ─────────────────────────────────────────────────────────────────────────────────
    # ONE eval run (RUN_ID in cell 0), eval.spacing_sweep=[4,1.5] + eval.spacing_locator=true. Focus =
    # the FINE (@1.5mm) accuracy distribution under three cuts. Coarse containment + oracle are per-class
    # (from run.summary), merged onto every sample in cell 0.
    # cell 0:  fetch + cache cases table + summary locator/aggregates; geometry + shape families; header.
    # cell 1:  fine dice — TRAINED vs HELD-OUT. Raw split (confounded: held-out is easier anatomy) +
    #          two morphology controls (shape family, matched lateral-mirror pairs). box/strip, ECDF, bars.
    # cell 2:  fine dice vs COARSE CONTAINMENT — scatter (colour=thickness) + containment-quartile boxes;
    #          containment tracks thickness, not accuracy, so the marginal ρ is weak/negative.
    # cell 3:  WHEN THE ORACLE FAILS (oracle<1, crop-size ceiling) — containable vs ceiling fine-dice
    #          distributions + ceiling-severity vs fine dice (colour=volume); ceiling marks large easy objects.
    # Edit RUN_ID / N_SHAPE in cell 0. Shape taxonomy + geometry come from totalseg_geometry_extract.
    print("tables + figures: cells 1-3")
    return


if __name__ == "__main__":
    app.run()
