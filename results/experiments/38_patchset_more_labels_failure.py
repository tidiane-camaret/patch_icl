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
    # ── patchset3d on totalseg_more_labels: failure analysis (all held-out, 285 novel classes) ────
    # Eval run gcoroxrx evaluated patchset3d on data.source=totalseg_more_labels — 285 novel hierarchical
    # classes ("task/structure"), ALL held-out (in_train is null), spacing_sweep=[4,1.5] + locator. This
    # notebook dissects the FINE (@1.5mm) accuracy distribution by task, class and object size, then
    # separates localization from segmentation failure. Unlike nb 37 there is NO totalseg_geometry_extract
    # (different dataset root + hierarchical names): the size drivers come from the cases table itself
    # (tgt_size / tgt_occ = target GT foreground voxels & fraction in the crop; ctx_occ = the K-shot
    # prompt occupancy). Per-class locator containment comes from run.summary. Cache: artifacts/38_<id>_*.
    import json
    import re
    import sys
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    sys.path.insert(0, str(Path(__file__).parent))
    from nb_common import ARTIFACTS as _ARTIFACTS, get_latest_table, latest_table_version

    pd.set_option("display.width", 240)
    pd.set_option("display.float_format", lambda v: f"{v:.3f}")

    PROJECT = "tidiane/patch_icl_3d_eval"
    RUN_ID = "gcoroxrx"              # patchset3d, totalseg_more_labels, spacing_sweep=[4,1.5], locator
    CHECK_FRESH = True               # ping W&B on cache hit; refetch if a newer table version exists

    def _parse_locator(summary):
        """Per-class locator containment (+ oracle) from a run summary dict. Class names contain '/'
        (task/structure); the non-greedy group + fixed suffix token handles that. NaN containment is
        logged as a string sentinel → coerced to NaN (marks classes whose coarse pass was empty)."""
        pat = re.compile(r"class/(.+?)/(containment@4|containment_oracle@4)$")
        rows = {}
        for k, v in summary.items():
            m = pat.match(k)
            if m:
                rows.setdefault(m.group(1), {})[m.group(2)] = v
        df = pd.DataFrame([{"class": c, "containment": d.get("containment@4"),
                            "oracle": d.get("containment_oracle@4")} for c, d in rows.items()])
        for _c in ("containment", "oracle"):
            df[_c] = pd.to_numeric(df[_c], errors="coerce")
        return df

    def _load():
        """Load the run's cached cases table + meta (locator + aggregates), refetching from W&B on cache
        miss or when CHECK_FRESH and a newer cases-table version exists."""
        _cases_c = _ARTIFACTS / f"38_{RUN_ID}_cases.csv"
        _meta_c = _ARTIFACTS / f"38_{RUN_ID}_meta.json"
        _run = None
        if _cases_c.exists() and _meta_c.exists():
            _m = json.loads(_meta_c.read_text())
            if not CHECK_FRESH:
                return pd.read_csv(_cases_c), _m
            import wandb
            _run = wandb.Api().run(f"{PROJECT}/{RUN_ID}")
            if latest_table_version(_run) <= _m.get("table_version", -1):
                return pd.read_csv(_cases_c), _m
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
    for _c in ("containment", "oracle"):          # re-coerce after JSON round-trip
        LOC[_c] = pd.to_numeric(LOC[_c], errors="coerce")
    print(f"loaded {RUN_ID}: cases {S.shape}  | {META['name']}  (source={META['config'].get('source')})")

    SPACINGS = sorted(S["spacing"].dropna().unique(), reverse=True)
    COARSE, FINE = float(SPACINGS[0]), float(SPACINGS[-1])
    S["task"] = S["class"].str.split("/").str[0]          # dataset group
    S["structure"] = S["class"].str.split("/").str[1]
    # per-class fine dice + coarse locator onto every sample (for the size/containment cells)
    _dice_fine = S[S.spacing == FINE].groupby("class").dice.mean()
    LOC = LOC.merge(S.groupby("class").agg(task=("task", "first"),
                                           tgt_size=("tgt_size", "median"),
                                           ctx_occ=("ctx_occ", "median")).reset_index(),
                    on="class", how="left")
    LOC["dice_fine"] = LOC["class"].map(_dice_fine)
    S = S.merge(LOC[["class", "containment", "oracle"]], on="class", how="left")

    def _macro(g):
        return g.groupby("class").dice.mean().mean()
    print(f"  spacings (coarse→fine): {COARSE:g} → {FINE:g} mm  "
          f"| {S['task'].nunique()} tasks / {S['class'].nunique()} classes / {len(S)} samples")
    for _sp in SPACINGS:
        _g = S[S.spacing == _sp]
        print(f"    @{_sp:g}mm  macro={_macro(_g):.4f}  micro={_g.dice.mean():.4f}  "
              f"miss(<0.01)={(_g.dice < 0.01).mean():.3f}  n={len(_g)}")
    _lv = LOC.dropna(subset=["containment"])
    print(f"  locator@{COARSE:g}mm: containment={_lv.containment.mean():.3f}  oracle={_lv.oracle.mean():.3f}"
          f"  | cont<0.5: {(_lv.containment < 0.5).sum()}  oracle<0.5: {(_lv.oracle < 0.5).sum()}"
          f"  empty-coarse (NaN): {LOC.containment.isna().sum()}")
    return COARSE, FINE, LOC, S, np, pd, plt


@app.cell
def _(FINE, S, np, pd, plt):
    # ── 1. ACCURACY BY TASK (dataset group) ──────────────────────────────────────────────────────
    # Fine (@{FINE}mm) dice grouped by the 37 source tasks (class prefix). Table = macro dice (mean over
    # classes), miss rate, class count, median object size + prompt occupancy — sorted worst→best. The
    # figure is a horizontal box of per-sample fine dice per task (sorted by median) so the spread and
    # the many all-zero tasks (teeth, landmarks, face) are visible next to the few that work (heart
    # chambers, cavities). Tasks track object size + prompt occupancy, not anatomy per se.
    _SF = S[S.spacing == FINE].copy()
    _tab = (_SF.groupby("task").apply(lambda g: pd.Series({
        "macro": g.groupby("class").dice.mean().mean(), "miss": (g.dice < 0.01).mean(),
        "n_cls": g["class"].nunique(), "n": len(g),
        "med_tgt": g.tgt_size.median(), "med_ctxocc": g.ctx_occ.median(),
    }), include_groups=False).sort_values("macro"))
    print(f"fine (@{FINE:g}mm) dice by task (worst→best):\n{_tab.to_string()}")

    _order = _tab.index.tolist()                # worst at bottom of the horizontal axis
    _y = np.arange(len(_order))
    _fig, _ax = plt.subplots(figsize=(11, max(6, 0.3 * len(_order))))
    _ax.boxplot([_SF[_SF.task == t].dice.values for t in _order], vert=False, widths=0.6,
                showmeans=True, flierprops=dict(marker=".", ms=2, alpha=0.3))
    for _i, _t in enumerate(_order):            # jittered strip
        _v = _SF[_SF.task == _t].dice.values
        _ax.scatter(_v, np.random.normal(_i + 1, 0.08, len(_v)), s=6, alpha=0.25,
                    color="tab:blue", zorder=1)
    _ax.set_yticks(_y + 1)
    _ax.set_yticklabels([f"{t} ({int(_tab.loc[t, 'n_cls'])})" for t in _order], fontsize=7)
    _ax.set(xlabel=f"fine dice @{FINE:g}mm", xlim=(-0.02, 1.02),
            title="(1) per-sample fine dice by task (sorted worst→best; n classes in ())")
    _ax.grid(alpha=0.3, axis="x")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(FINE, S, np, plt):
    # ── 2. ACCURACY BY CLASS ─────────────────────────────────────────────────────────────────────
    # Per-class macro fine dice over all 285 classes: the histogram + ECDF show the bimodal outcome —
    # a large mass near zero (novel structures the model never reconstructs) and a smaller shoulder
    # around 0.4–0.6. The ranked bars list the worst and best classes (task/structure), coloured by
    # object size so the "worst = tiny" / "best = large" pattern is explicit.
    _pc = (S[S.spacing == FINE].groupby("class")
             .agg(dice=("dice", "mean"), task=("task", "first"),
                  tgt=("tgt_size", "median")).reset_index())
    _n0 = (_pc.dice < 0.05).mean()
    print(f"per-class fine dice: n={len(_pc)}  mean={_pc.dice.mean():.3f}  median={_pc.dice.median():.3f}"
          f"  frac<0.05={_n0:.2f}  frac>0.5={ (_pc.dice > 0.5).mean():.2f}")
    print("\nworst 15 classes:")
    print(_pc.nsmallest(15, "dice")[["class", "dice", "tgt"]].to_string(index=False))
    print("\nbest 15 classes:")
    print(_pc.nlargest(15, "dice")[["class", "dice", "tgt"]].to_string(index=False))

    _fig, _axes = plt.subplots(2, 2, figsize=(15, 10))
    # (a) histogram of per-class macro dice
    _axes[0, 0].hist(_pc.dice, bins=30, color="tab:blue", alpha=0.8, edgecolor="k", linewidth=0.3)
    _axes[0, 0].axvline(_pc.dice.median(), color="k", ls="--", lw=1, label=f"median {_pc.dice.median():.2f}")
    _axes[0, 0].set(xlabel=f"per-class fine dice @{FINE:g}mm", ylabel="n classes",
                    title="(a) per-class dice histogram (285 classes)")
    _axes[0, 0].legend(fontsize=8); _axes[0, 0].grid(alpha=0.3)
    # (b) ECDF
    _s = np.sort(_pc.dice.values)
    _axes[0, 1].plot(_s, np.arange(1, len(_s) + 1) / len(_s), lw=1.8, color="tab:purple")
    _axes[0, 1].set(xlabel=f"per-class fine dice @{FINE:g}mm", ylabel="ECDF",
                    title="(b) per-class dice ECDF")
    _axes[0, 1].grid(alpha=0.3)
    # (c) worst 20 + (d) best 20 ranked bars, coloured by log size
    import matplotlib.cm as cm
    _norm = plt.Normalize(np.log10(_pc.tgt.clip(lower=1)).min(), np.log10(_pc.tgt.clip(lower=1)).max())
    for _ax, _sub, _ttl in [(_axes[1, 0], _pc.nsmallest(20, "dice"), "(c) worst 20 classes"),
                            (_axes[1, 1], _pc.nlargest(20, "dice")[::-1], "(d) best 20 classes")]:
        _cols = cm.viridis(_norm(np.log10(_sub.tgt.clip(lower=1))))
        _ax.barh(np.arange(len(_sub)), _sub.dice.values, color=_cols)
        _ax.set_yticks(np.arange(len(_sub)))
        _ax.set_yticklabels(_sub["class"].tolist(), fontsize=6)
        _ax.set(xlabel=f"fine dice @{FINE:g}mm", title=_ttl + " (colour=log size)")
        _ax.grid(alpha=0.3, axis="x")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(FINE, S, np, pd, plt):
    # ── 3. ACCURACY BY OBJECT SIZE / PROMPT OCCUPANCY ────────────────────────────────────────────
    # The dominant driver. Per-sample fine dice vs target size (tgt_size) and vs prompt occupancy
    # (ctx_occ, the K-shot context foreground fraction), with binned-mean lines + Spearman ρ. Small
    # objects occupy ~1e-4 of the fixed 128³ crop → near-empty prompts → near-zero dice; the largest
    # objects rebound in miss-rate (huge diffuse shells: face, body, cavities). Miss-rate and a
    # size-sextile box make the two-tailed failure explicit.
    _SF = S[S.spacing == FINE].copy()
    _fig, _axes = plt.subplots(2, 2, figsize=(14, 9)); _axes = _axes.ravel()

    def _driver_panel(ax, col, label):
        _d = _SF[[col, "dice"]].dropna(); _d = _d[_d[col] > 0]
        _rho = _d.dice.corr(np.log10(_d[col]), method="spearman")
        ax.scatter(_d[col], _d.dice, s=6, alpha=0.12, color="tab:blue")
        _grp = _d.groupby(pd.qcut(_d[col], 8, duplicates="drop"), observed=True)
        ax.plot(_grp[col].median(), _grp.dice.mean(), "-o", color="tab:red", ms=4, label="binned mean")
        ax.set_xscale("log")
        ax.set(xlabel=label, ylabel=f"fine dice @{FINE:g}mm", title=f"dice vs {label}  (ρ={_rho:+.2f})")
        ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")

    _driver_panel(_axes[0], "tgt_size", "target size (fg voxels)")
    _driver_panel(_axes[1], "ctx_occ", "prompt occupancy (ctx_occ)")

    # (c) miss-rate vs size sextile
    _SF["szbin"] = pd.qcut(_SF.tgt_size.clip(lower=1), 6, duplicates="drop")
    _mb = _SF.groupby("szbin", observed=True).agg(miss=("dice", lambda x: (x < 0.01).mean()),
                                                  dice=("dice", "mean"), n=("dice", "size"))
    _x = np.arange(len(_mb))
    _axes[2].bar(_x, _mb.miss.values, color="tab:orange", alpha=0.85)
    _axes[2].set_xticks(_x)
    _axes[2].set_xticklabels([f"{int(b.left)}–{int(b.right)}" for b in _mb.index], rotation=30,
                             ha="right", fontsize=7)
    _axes[2].set(ylabel="complete-miss rate (dice<0.01)", xlabel="target size sextile (voxels)",
                 title="(c) miss rate vs size (two-tailed: tiny + huge)")
    _axes[2].grid(alpha=0.3, axis="y")
    # (d) dice box per size sextile
    _bins = list(_mb.index)
    _axes[3].boxplot([_SF[_SF.szbin == b].dice.values for b in _bins], showmeans=True,
                     flierprops=dict(marker=".", ms=2, alpha=0.3))
    _axes[3].set_xticklabels([f"{int(b.left)}–{int(b.right)}" for b in _bins], rotation=30,
                             ha="right", fontsize=7)
    _axes[3].set(ylabel=f"fine dice @{FINE:g}mm", xlabel="target size sextile (voxels)",
                 title="(d) fine dice by size sextile", ylim=(-0.02, 1.02))
    _axes[3].grid(alpha=0.3, axis="y")
    _fig.suptitle("(3) fine accuracy vs object size & prompt occupancy", y=1.002)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(COARSE, FINE, LOC, np, plt):
    # ── 4. LOCALIZATION vs SEGMENTATION FAILURE ──────────────────────────────────────────────────
    # Is failure the locator's fault or the fine segmenter's? containment@{COARSE}mm is high (~0.83) and
    # weakly tied to accuracy, so it is mostly SEGMENTATION failure: classes are framed correctly but the
    # fine mask is empty/poor for novel appearance. Each class is tagged by dominant failure mode:
    #   empty-coarse   — coarse pass produced no foreground (containment is NaN)
    #   diffuse/ceiling— oracle<0.5: object too big/diffuse for the fine window even centred (face, body,
    #                    tissue, effusions) — not localizable as an object
    #   localization   — containment<0.5 (coarse centroid misses) but oracle ok
    #   segmentation   — well-framed (containment≥0.5) yet fine dice<0.15
    #   ok             — fine dice≥0.15
    # Panel (a): containment vs fine dice, colour=log size. Panel (b): class counts + mean dice per mode.
    _L = LOC.copy()
    def _mode(r):
        if not np.isfinite(r.containment):
            return "empty-coarse"
        if np.isfinite(r.oracle) and r.oracle < 0.5:
            return "diffuse/ceiling"
        if r.containment < 0.5:
            return "localization"
        if (not np.isfinite(r.dice_fine)) or r.dice_fine < 0.15:
            return "segmentation"
        return "ok"
    _L["mode"] = _L.apply(_mode, axis=1)
    _order = ["ok", "segmentation", "localization", "diffuse/ceiling", "empty-coarse"]
    _mc = {"ok": "tab:green", "segmentation": "tab:red", "localization": "tab:orange",
           "diffuse/ceiling": "tab:purple", "empty-coarse": "0.4"}
    _summ = _L.groupby("mode").agg(n=("class", "size"), mean_dice=("dice_fine", "mean"),
                                   med_tgt=("tgt_size", "median")).reindex(_order).dropna(how="all")
    print("failure-mode taxonomy (per class):\n" + _summ.to_string())
    _wl = _L[_L.containment > 0.8]
    print(f"\nwell-localized (containment>0.8): {len(_wl)} classes; still failing (dice<0.15): "
          f"{int((_wl.dice_fine < 0.15).sum())} ({(_wl.dice_fine < 0.15).mean() * 100:.0f}%)  "
          f"→ segmentation, not localization, is the bottleneck")
    _lv = _L.dropna(subset=["containment", "dice_fine"])
    print(f"corr(containment, fine dice) spearman = {_lv.containment.corr(_lv.dice_fine, method='spearman'):+.2f}")

    _fig, (_a0, _a1) = plt.subplots(1, 2, figsize=(15, 6.5))
    # (a) containment vs fine dice, colour = log size
    _d = _L.dropna(subset=["containment", "dice_fine"])
    _sc = _a0.scatter(_d.containment, _d.dice_fine, s=34, c=np.log10(_d.tgt_size.clip(lower=1)),
                      cmap="viridis", alpha=0.85, edgecolor="k", linewidth=0.2, zorder=3)
    _a0.axvline(0.5, color="0.6", lw=0.8, ls=":"); _a0.axhline(0.15, color="0.6", lw=0.8, ls=":")
    _fig.colorbar(_sc, ax=_a0, label="log10 target size")
    _a0.set(xlabel=f"coarse containment @{COARSE:g}mm", ylabel=f"fine dice @{FINE:g}mm",
            title="(a) containment vs fine dice (colour=size); high cont + low dice = seg failure")
    _a0.grid(alpha=0.3)
    # (b) class counts per failure mode + mean dice
    _modes = [m for m in _order if m in _summ.index]
    _x = np.arange(len(_modes))
    _b = _a1.bar(_x, [_summ.loc[m, "n"] for m in _modes], color=[_mc[m] for m in _modes])
    for _i, _m in enumerate(_modes):
        _md = _summ.loc[_m, "mean_dice"]
        _a1.annotate(f"dice≈{_md:.2f}" if np.isfinite(_md) else "dice≈0",
                     (_i, _summ.loc[_m, "n"]), ha="center", va="bottom", fontsize=8)
    _a1.set_xticks(_x); _a1.set_xticklabels(_modes, rotation=20, ha="right", fontsize=8)
    _a1.set(ylabel="n classes", title="(b) classes per failure mode (mean fine dice annotated)")
    _a1.grid(alpha=0.3, axis="y")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _():
    # ── Contents ─────────────────────────────────────────────────────────────────────────────────
    # ONE eval run (RUN_ID in cell 0): patchset3d on totalseg_more_labels, 285 novel held-out classes,
    # spacing_sweep=[4,1.5]+locator. Focus = the FINE (@1.5mm) accuracy distribution + failure reasons.
    # Size drivers come from the cases table (no totalseg_geometry_extract — different dataset); per-class
    # locator containment from run.summary.
    # cell 0:  fetch + cache cases table + summary locator/aggregates; task/structure split; header.
    # cell 1:  accuracy BY TASK — macro/miss table + per-sample dice box per task (sorted worst→best).
    # cell 2:  accuracy BY CLASS — per-class dice histogram + ECDF; worst/best-20 ranked bars (colour=size).
    # cell 3:  accuracy BY SIZE — dice vs tgt_size & ctx_occ (Spearman) + miss-rate/box per size sextile.
    # cell 4:  LOCALIZATION vs SEGMENTATION — containment vs fine dice; failure-mode taxonomy
    #          (empty-coarse / diffuse-ceiling / localization / segmentation / ok).
    # Edit RUN_ID in cell 0.
    print("tables + figures: cells 1-4")
    return


if __name__ == "__main__":
    app.run()
