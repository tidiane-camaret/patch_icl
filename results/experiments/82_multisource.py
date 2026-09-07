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
    # ── multisource CT+MRI in-context training — val sample-table analysis (2 runs) ──────────────
    # Both runs = `experiment=81_multisource_ct_mri` (per-task modality regime 1/3 ct / 1/3 mri /
    # 1/3 forced cross; 128^3, K=1, log-uniform [1.5,6] mm train pitch, eval fixed 3 mm; 2000
    # seeded tasks/epoch; goal_mask.p=0.4 ops=[dilate,erode,boundary,sobel]; +data.ram_cache).
    # They differ in train-class set + init:
    #   "81 warmstart/subset"  per_source_train_classes=[balanced, train]  (CT balanced subset +
    #       23 MRI train classes; held-out classes exist -> real in_train split), WARM-STARTED
    #       from the exp80 CT-only checkpoint. Its `epoch` column continues exp80's counter, so
    #       eval points are 200..300 (~110 finetune epochs).
    #   "82 scratch/all-cls"   per_source_train_classes=[all, all]  (every CT+MRI class trained
    #       -> in_train uniformly True, no unseen signal), trained FROM SCRATCH, evals 0..130.
    # This notebook reads each local wandb run dir directly (one val/samples_*.table.json per eval
    # epoch) — no wandb API. Regime + modality pair come from the `detail` column
    # ("<regime> <tgt_mod><-<ctx_mod>", evaluate.py::_sample_detail). nsd is not computed for this
    # eval path (all NaN); self_ctx off, spacing fixed 3 mm.
    import json
    import re
    from pathlib import Path

    import pandas as pd
    import matplotlib.pyplot as plt

    pd.set_option("display.width", 220)
    pd.set_option("display.float_format", lambda v: f"{v:.3f}")

    _WB = Path(__file__).parents[2] / "wandb"
    RUNS = {                                         # label -> wandb run dir
        "81 warmstart/subset": "run-20260906_162641-qjmpcm1h",
        "82 scratch/all-cls":  "run-20260906_233143-kr49tqzo",
    }

    def _load(run_dir, label):
        _tbl = _WB / run_dir / "files" / "media" / "table" / "val"
        _paths = sorted(_tbl.glob("samples_*_*.table.json"),
                        key=lambda p: int(re.search(r"samples_(\d+)_", p.name).group(1)))
        _parts = []
        for _p in _paths:
            _d = json.loads(_p.read_text())
            _parts.append(pd.DataFrame(_d["data"], columns=_d["columns"]))
        _df = pd.concat(_parts, ignore_index=True)
        _df["run"] = label
        _s = _df["detail"].str.split(" ", n=1, expand=True)
        _df["regime"] = _s[0]
        _pp = _s[1].str.replace(" [fb]", "", regex=False).str.split("<-", expand=True)
        _df["tgt_mod"], _df["ctx_mod"] = _pp[0], _pp[1]
        _df["is_last"] = _df["epoch"] == _df["epoch"].max()
        return _df

    DF = pd.concat([_load(v, k) for k, v in RUNS.items()], ignore_index=True)
    L = DF[DF["is_last"]].copy()                        # last-epoch snapshot per run

    for _lab in RUNS:
        _g = DF[DF["run"] == _lab]
        _l = _g[_g["is_last"]]
        _eps = sorted(_g.epoch.unique())
        _it = _l.in_train
        print(f"\n=== {_lab}  ({RUNS[_lab]}) ===")
        print(f"evals {_eps[0]}..{_eps[-1]} ({len(_eps)})  | last-epoch rows {len(_l)}  "
              f"classes {_l['class'].nunique()}  | in_train T/F {int((_it == True).sum())}"
              f"/{int((_it == False).sum())}  | cross fb {int(_l.detail.str.contains(r'\\[fb\\]').sum())}")
        print(f"  overall: dice mean={_l.dice.mean():.3f} median={_l.dice.median():.3f} "
              f"soft={_l.soft_dice.mean():.3f} miss(<0.05)={ (_l.dice < 0.05).mean():.2f}")
        print(_l.groupby("regime").dice.agg(n="size", mean="mean", median="median",
              miss=lambda x: (x < 0.05).mean(), hit=lambda x: (x > 0.5).mean()).to_string())
    return DF, L, RUNS, pd, plt


@app.cell
def _(DF, RUNS, plt):
    # ── 1. TRAINING TREND — per-epoch val dice by regime, one panel per run ──────────────────────
    # 82 (scratch): all three regimes climb from ~0 and are still rising at ep130 (not converged);
    #   `cross` starts below both same-modality regimes and overtakes mri←mri ~ep110.
    # 81 (warm-start): the CT regime enters at ~0.40 and stays ~flat over 100 finetune epochs
    #   (it inherits the exp80 CT-only checkpoint and sits near this arch's CT ceiling) — ALL the
    #   finetune gain is MRI (0.17→0.27) + cross (0.24→0.35). Warm-start is worth ~+0.10 overall
    #   dice vs scratch at comparable budget (0.345 @ep300 vs 0.243 @ep130).
    _fig, _axes = plt.subplots(1, len(RUNS), figsize=(6.2 * len(RUNS), 4.2), squeeze=False)
    for _ax, _lab in zip(_axes[0], RUNS):
        _g = DF[DF["run"] == _lab]
        _t = _g.groupby(["epoch", "regime"]).dice.mean().unstack()
        _t["all"] = _g.groupby("epoch").dice.mean()
        print(f"\n{_lab} — val dice by regime per epoch:\n{_t.round(3).to_string()}")
        for _c, _col in [("ct", "tab:blue"), ("mri", "tab:red"),
                         ("cross", "tab:green"), ("all", "0.4")]:
            _ax.plot(_t.index, _t[_c], marker="o", ms=3, color=_col,
                     ls="--" if _c == "all" else "-", lw=2 if _c == "all" else 1.4, label=_c)
        _ax.set(xlabel="epoch (global counter)", ylabel="val dice (micro mean)",
                title=_lab, ylim=(0, 0.75))
        _ax.legend(fontsize=8)
        _ax.grid(alpha=0.3)
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(L, RUNS, pd, plt):
    # ── 2. CROSS-MODALITY DECOMPOSITION (last epoch) ─────────────────────────────────────────────
    # Score by modality PAIR (target<-context) + each marginal. Both runs: the score tracks the
    # TARGET modality (tgt-CT >> tgt-MRI) and is nearly flat in the CONTEXT modality — in 81
    # `ct←mri` (0.41) is identical to `ct←ct` (0.41). In-context matching is modality-robust;
    # MRI targets simply being harder to segment is the limiter, not the CT<->MRI transfer.
    _pair_order = ["ct<-ct", "ct<-mri", "mri<-ct", "mri<-mri"]
    _marg = []
    for _lab in RUNS:
        _l = L[L["run"] == _lab]
        _mp = _l.tgt_mod + "<-" + _l.ctx_mod
        print(f"\n{_lab}:")
        print(_l.groupby(_mp).dice.agg(n="size", mean="mean", median="median")
              .reindex(_pair_order).to_string())
        _marg.append({"run": _lab,
                      "tgt=ct": _l[_l.tgt_mod == "ct"].dice.mean(),
                      "tgt=mri": _l[_l.tgt_mod == "mri"].dice.mean(),
                      "ctx=ct": _l[_l.ctx_mod == "ct"].dice.mean(),
                      "ctx=mri": _l[_l.ctx_mod == "mri"].dice.mean()})
    _mk = pd.DataFrame(_marg).set_index("run")
    print("\ntarget / context modality marginals:\n" + _mk.to_string())

    _fig, _ax = plt.subplots(1, 2, figsize=(13, 4.2))
    _pv = (L.assign(pair=L.tgt_mod + "<-" + L.ctx_mod)
             .pivot_table(index="pair", columns="run", values="dice", aggfunc="mean")
             .reindex(_pair_order))
    _pv.plot.bar(ax=_ax[0], rot=0)
    _ax[0].set(ylabel="mean dice", xlabel="target<-context", title="(2a) dice by modality pair")
    _ax[0].grid(alpha=0.3, axis="y")
    _mk.T.plot.bar(ax=_ax[1], rot=0)
    _ax[1].set(ylabel="mean dice", xlabel="",
               title="(2b) target vs context modality marginals")
    _ax[1].grid(alpha=0.3, axis="y")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(DF, L, RUNS, plt):
    # ── 3. SEEN vs UNSEEN (in_train) ────────────────────────────────────────────────────────────
    # Only the 81 warm-start run has a held-out class split (82 trains on `all` -> in_train all
    # True, skipped). 81 DOES generalise to unseen classes but with a penalty (overall seen 0.37
    # vs unseen 0.28; CT 0.46 vs 0.33; MRI 0.30 vs 0.19 / median 0.08 — the weakest cell). The
    # gap WIDENS over finetuning: seen climbs steadily while unseen is ~flat (0.25→0.28 over 100
    # epochs) — the model specialises to its train-class set rather than improving general
    # in-context ability.
    for _lab in RUNS:
        _l = L[L["run"] == _lab]
        if _l.in_train.nunique() < 2:
            print(f"{_lab}: in_train all {_l.in_train.iloc[0]} — no seen/unseen split")
            continue
        print(f"\n{_lab} last-epoch dice by (target modality, in_train):")
        print(_l.groupby(["tgt_mod", "in_train"]).dice.agg(n="size", mean="mean",
              median="median", miss=lambda x: (x < 0.05).mean()).to_string())

    _split = [l for l in RUNS if L[L["run"] == l].in_train.nunique() >= 2]
    _fig = None
    if _split:
        _lab = _split[0]
        _tr = (DF[DF["run"] == _lab].groupby(["epoch", "in_train"]).dice.mean().unstack()
               .rename(columns={True: "seen", False: "unseen"}))
        print(f"\n{_lab} — seen vs unseen dice per epoch:\n{_tr.round(3).to_string()}")
        _fig, _ax = plt.subplots(figsize=(6.5, 4))
        _tr[["seen", "unseen"]].plot(ax=_ax, marker="o", ms=4,
                                     color=["tab:green", "tab:red"])
        _ax.set(xlabel="epoch (global counter)", ylabel="val dice",
                title=f"(3) {_lab}: seen vs unseen classes — gap widens over finetuning")
        _ax.grid(alpha=0.3)
        _fig.tight_layout()
    _fig
    return


@app.cell
def _(L, RUNS, pd, plt):
    # ── 4. WHAT FAILS — object size + per-class (last epoch) ─────────────────────────────────────
    # (a) Size is the dominant axis in both runs; the warm-start lifts every size quintile,
    #     especially the smallest (<~285 vox: 0.03→0.12). The tiny quintile (~20% of eval) is
    #     still near-failure.
    # (b) Worst classes (81, n>=15) = thin/tubular/small — iliac & carotid vessels, portal vein,
    #     adrenals, prostate, duodenum, esophagus. Best = large compact organs (heart, spleen,
    #     brain, liver, spinal_cord). `prostate` on MRI ~0.02, but in 81 it is an UNSEEN MRI
    #     class AND tiny -> likely unseen+size rather than a totalsegmri label bug.
    _rows = []
    for _lab in RUNS:
        _l = L[L["run"] == _lab]
        _q = pd.qcut(_l.tgt_size, 5, duplicates="drop")
        _rows.append(_l.groupby(_q, observed=True).dice.mean().reset_index(drop=True).rename(_lab))
    _sz = pd.concat(_rows, axis=1)
    _sz.index = [f"Q{i + 1}" for i in _sz.index]
    print("mean dice by target-size quintile (per run):\n" + _sz.to_string())

    _lab = [l for l in RUNS if "warmstart" in l][0]
    _pc = L[L["run"] == _lab].groupby("class").agg(
        n=("dice", "size"), dice=("dice", "mean"),
        tgt=("tgt_size", "median"), seen=("in_train", "first"))
    _big = _pc[_pc.n >= 15].sort_values("dice")
    print(f"\n{_lab} — worst 15 classes (n>=15):\n{_big.head(15).to_string()}")
    print(f"\n{_lab} — best 10 classes:\n{_big.tail(10).to_string()}")

    _fig, _ax = plt.subplots(1, 2, figsize=(13, 5))
    _sz.plot.bar(ax=_ax[0], rot=0)
    _ax[0].set(xlabel="target size quintile", ylabel="mean dice", title="(4a) dice vs object size")
    _ax[0].grid(alpha=0.3, axis="y")
    _wb = pd.concat([_big.head(12), _big.tail(12)])
    _ax[1].barh(range(len(_wb)), _wb.dice.values, color=["tab:red"] * 12 + ["tab:green"] * 12)
    _ax[1].set_yticks(range(len(_wb)))
    _ax[1].set_yticklabels(_wb.index, fontsize=6)
    _ax[1].set(xlabel="mean dice", title=f"(4b) {_lab}: worst / best 12 classes (n>=15)")
    _ax[1].grid(alpha=0.3, axis="x")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _():
    # ── Contents ────────────────────────────────────────────────────────────────────────────────
    # TWO training runs of experiment=81_multisource_ct_mri (multisource CT+MRI, per-task modality
    # regime), read from their local wandb run dirs (one val sample-table per eval epoch):
    #   "81 warmstart/subset" (qjmpcm1h) — [balanced,train] classes, warm-started from the exp80
    #        CT-only checkpoint (epoch counter continues -> evals 200..300).
    #   "82 scratch/all-cls"  (kr49tqzo) — [all,all] classes, from scratch (evals 0..130;
    #        in_train all True).
    # cell 0: load both runs; parse detail -> regime/tgt_mod/ctx_mod; per-run header + dice by regime.
    # cell 1: TRAINING TREND — per-epoch dice by regime, one panel per run. Warm-start = flat-high
    #         CT + all gain in MRI/cross; scratch still climbing at ep130.
    # cell 2: CROSS-MODALITY DECOMPOSITION — dice by modality pair + target/context marginals.
    #         Score tracks TARGET modality; CONTEXT modality ~irrelevant (81 ct←mri == ct←ct).
    # cell 3: SEEN vs UNSEEN (in_train) — 81 only. Generalises to unseen with a ~0.09 penalty;
    #         MRI-unseen weakest (0.19); gap widens over finetuning (unseen ~flat).
    # cell 4: WHAT FAILS — dice by size quintile (both runs); worst/best classes (81). Warm-start
    #         lifts every size quintile; failures are thin/tubular/small structures.
    print("tables + figures: cells 1-4")
    return


if __name__ == "__main__":
    app.run()
