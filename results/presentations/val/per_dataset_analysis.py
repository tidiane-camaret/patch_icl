import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
    # Per-dataset eval comparison across the 7 eval-expansion sources (isles22, shifts_ms,
    # msd_hippocampus, msd_prostate, atlas_v2, gnc_kidney, hu_lwk1). Compares:
    #   - patchset3d exp92_orig     (.../3d_train/2026-09-11_92_multisource_synth/best.pt)
    #   - patchset3d exp_cascade_register (.../checkpoints/2026-09-14_92_multisource_synth_cascade_register/best.pt,
    #     arch.cascade_registers=true, see docs/datasets/eval_expansion_status.md)
    #   - medverse released weights, 3 inference modes (harness cascade prior=none/pred,
    #     native autoregressive_inference)
    # Rescans experiments/3d/eval.py's eval.json outputs live every run, so re-running this
    # notebook after a new eval.py invocation picks it up automatically -- no manual CSV
    # exports to keep in sync. See docs/datasets/eval_expansion_status.md for full writeups.
    import marimo as mo

    mo.md(
        "# Per-dataset eval comparison\n"
        "patchset3d (exp92_orig vs exp_cascade_register) vs. released Medverse "
        "(harness cascade prior=none/pred, native autoregressive), across all 7 "
        "eval-expansion sources. Rescans `eval.json` outputs on every run."
    )
    return (mo,)


@app.cell
def _():
    import glob
    import json
    import os
    import re
    from collections import defaultdict

    import pandas as pd

    EVAL_ROOT = (
        "/nfs/data/nii/data1/Analysis/camaret___in_context_segmentation/"
        "ANALYSIS_20251122/results/patch_icl/3d_eval"
    )

    # class name -> source. "stroke_lesion" is shared by isles22/atlas_v2 -- disambiguated
    # by n_samples below (isles22 n~247-250, atlas_v2 n=654).
    SOURCE_BY_CLASS = {
        "l1_center": "hu_lwk1",
        "ms_lesion": "shifts_ms",
        "hippocampus_anterior": "msd_hippocampus",
        "hippocampus_posterior": "msd_hippocampus",
        "prostate_pz": "msd_prostate",
        "prostate_tz": "msd_prostate",
        "hyper_r": "gnc_kidney", "hyper_l": "gnc_kidney",
        "hypo_r": "gnc_kidney", "hypo_l": "gnc_kidney",
        "complex_r": "gnc_kidney", "complex_l": "gnc_kidney",
        "hyper_cyst_r": "gnc_kidney", "hyper_cyst_l": "gnc_kidney",
        "hyper_mask_l": "gnc_kidney",
        "hypo_mask_r": "gnc_kidney", "hypo_mask_l": "gnc_kidney",
        "complex_cyst_r": "gnc_kidney", "complex_cyst_l": "gnc_kidney",
    }

    # checkpoint path suffix -> short tag used throughout the eval-expansion docs.
    CKPT_TAG = {
        "2026-09-11_92_multisource_synth/best.pt": "exp92_orig",
        "2026-09-14_92_multisource_synth_cascade_register/best.pt": "exp_cascade_register",
    }
    return CKPT_TAG, EVAL_ROOT, SOURCE_BY_CLASS, defaultdict, glob, json, os, pd, re


@app.cell
def _(CKPT_TAG, SOURCE_BY_CLASS, defaultdict, glob, json, os, pd, re, EVAL_ROOT):
    def _classify_run(run_dir, has_cascade):
        """(mode, query_prior) from the wandb-derived output-dir naming convention used
        throughout the 2026-09-14 medverse/cascade_register sweeps; falls back to the
        exp92 default (cascade always used the trained pred-mixture, never overridden) for
        older auto-named wandb runs (e.g. 'hopeful-frog-85')."""
        rd = re.sub(r"^\d{4}-\d{2}-\d{2}_", "", run_dir)
        if rd.startswith("medverse_ar_"):
            return "native_ar", ""
        if rd.startswith("medverse_cascade_"):
            return "cascade", ("pred" if rd.endswith("_pred") else "none")
        if rd.startswith("cascreg_single_"):
            return "single", ""
        if rd.startswith("cascreg_cascade_"):
            return "cascade", "pred"
        return ("cascade", "pred") if has_cascade else ("single", "")

    def load_eval_df():
        """Scan every eval.json under EVAL_ROOT, keep only rows from our 7 tracked sources
        + our 2 tracked patchset3d checkpoints + released medverse, dedupe by keeping the
        most-recently-written run per (source, model, checkpoint, mode, query_prior, class)."""
        best = {}
        for ej in glob.glob(os.path.join(EVAL_ROOT, "*", "eval.json")):
            run_dir = os.path.basename(os.path.dirname(ej))
            try:
                d = json.load(open(ej))
            except Exception:
                continue
            model = d.get("model")
            ckpt = d.get("config", {}).get("checkpoint") or ""
            ckpt_tag = next((v for k, v in CKPT_TAG.items() if ckpt.endswith(k)), None)
            is_medverse_released = model == "medverse" and not ckpt
            if not (ckpt_tag or is_medverse_released):
                continue
            for row in d.get("rows", []):
                cls = row.get("class", "")
                src = SOURCE_BY_CLASS.get(cls)
                if src is None and cls == "stroke_lesion":
                    src = "atlas_v2" if row.get("n_samples", 0) > 400 else "isles22"
                if src is None:
                    continue
                has_cascade = any(k.startswith("dice_r") for k in row)
                mode, qp = _classify_run(run_dir, has_cascade)
                key = (src, model, ckpt_tag or "released", mode, qp, cls)
                if key not in best or run_dir > best[key]["run_dir"]:
                    best[key] = {
                        "source": src, "model": model, "checkpoint": ckpt_tag or "released",
                        "mode": mode, "query_prior": qp, "class": cls,
                        "n_samples": row.get("n_samples"),
                        "mean_dice": row.get("mean_dice"), "std_dice": row.get("std_dice"),
                        "mean_nsd": row.get("mean_nsd"), "mean_time_ms": row.get("mean_time_ms"),
                        "gflops": row.get("gflops"),
                        "dice_r_levels": ";".join(
                            f"{k[6:]}={row[k]:.4f}" for k in row if k.startswith("dice_r")
                        ),
                        "run_dir": run_dir,
                    }
        df = pd.DataFrame(best.values())
        df["variant"] = (
            df["model"] + " / " + df["checkpoint"] + " / " + df["mode"]
            + df["query_prior"].apply(lambda q: f"({q})" if q else "")
        )
        return df

    per_class_df = load_eval_df()
    per_class_df
    return (per_class_df,)


@app.cell
def _(mo):
    mo.md(
        "**Note**: exp92_orig's own hu_lwk1 cascade result (Dice 0.1287, NSD 0.0729, "
        "`[6,3,1]`) is documented in `docs/datasets/hu_lwk1.md` #8 but its `eval.json` "
        "was not relocatable on disk (likely overwritten by a later run reusing the same "
        "output-dir name) -- it will show as missing below until re-run."
    )
    return


@app.cell
def _(per_class_df, pd):
    # VRAM is NOT plotted: no eval.py/evaluate.py/cascade.py path ever measures
    # torch.cuda.max_memory_allocated() (checked -- only .synchronize() calls exist, no
    # memory stats logged to eval.json/wandb). mean_time_ms and gflops ARE real per-run
    # measurements and stand in as the closest tracked cost proxies; see the note cell below.
    summary_df = (
        per_class_df.groupby(["source", "model", "checkpoint", "mode", "query_prior", "variant"],
                              as_index=False)
        .agg(n_classes=("class", "count"),
             macro_dice=("mean_dice", "mean"),
             macro_nsd=("mean_nsd", "mean"),
             mean_time_ms=("mean_time_ms", "mean"),
             gflops=("gflops", "mean"))
        .sort_values(["source", "macro_dice"], ascending=[True, False])
        .reset_index(drop=True)
    )
    for _col in ("macro_dice", "macro_nsd", "mean_time_ms", "gflops"):
        summary_df[_col] = summary_df[_col].round(4)
    summary_df
    return (summary_df,)


@app.cell
def _(mo):
    mo.md(
        "**No VRAM data**: nothing in `experiments/3d/eval.py` / `evaluate.py` / "
        "`cascade.py` ever calls `torch.cuda.max_memory_allocated()` or logs memory to "
        "`eval.json`/wandb -- it was never measured for any of these runs. `mean_time_ms` "
        "(wall-clock/sample, real) and `gflops` (measured via `FlopCounterMode`, real) are "
        "plotted below as the closest tracked cost proxies. VRAM is mostly a function of "
        "`(model, mode, image_size, batch_size)`, not the dataset itself, so it would be "
        "one number per variant rather than a per-dataset axis -- happy to add a one-off "
        "`torch.cuda.max_memory_allocated()` benchmark pass per variant if still wanted."
    )
    return


@app.cell
def _(plt, summary_df):
    # One figure per metric, one subplot per dataset (7), every variant as a horizontal bar --
    # no dropdown, everything visible at once. Shared across the 4 metric cells below.
    _SOURCES = sorted(summary_df["source"].unique())

    def _variant_color(variant):
        if "exp92_orig" in variant:
            return "#8a8f98"
        if "exp_cascade_register" in variant:
            return "#2a78d6"
        return "#eb6834"

    def plot_metric_grid(col, label, fmt):
        n = len(_SOURCES)
        ncols = 4
        nrows = -(-n // ncols)  # ceil
        fig, axes = plt.subplots(nrows, ncols, figsize=(22, 3.1 * nrows), dpi=140)
        axes = axes.flatten()
        for ax, source in zip(axes, _SOURCES):
            sel = summary_df[summary_df["source"] == source].sort_values(col, ascending=True)
            colors = [_variant_color(v) for v in sel["variant"]]
            bars = ax.barh(sel["variant"], sel[col], color=colors)
            xmax = max(sel[col].max() * 1.3, 1e-6)
            for bar, val in zip(bars, sel[col]):
                ax.text(bar.get_width() + 0.02 * xmax, bar.get_y() + bar.get_height() / 2,
                        fmt.format(val), va="center", fontsize=7)
            ax.set_title(source, fontsize=10)
            ax.set_xlim(0, xmax)
            ax.tick_params(axis="y", labelsize=7)
            ax.tick_params(axis="x", labelsize=7)
        for ax in axes[n:]:
            ax.axis("off")
        fig.suptitle(label, fontsize=14)
        fig.tight_layout()
        return fig

    return (plot_metric_grid,)


@app.cell
def _(plot_metric_grid):
    plot_metric_grid("macro_dice", "macro Dice -- all datasets, all variants", "{:.3f}")
    return


@app.cell
def _(plot_metric_grid):
    plot_metric_grid("macro_nsd", "macro NSD -- all datasets, all variants", "{:.3f}")
    return


@app.cell
def _(plot_metric_grid):
    plot_metric_grid("mean_time_ms", "ms / sample -- all datasets, all variants", "{:.0f}")
    return


@app.cell
def _(plot_metric_grid):
    plot_metric_grid("gflops", "GFLOPs / sample -- all datasets, all variants", "{:.0f}")
    return


@app.cell
def _(per_class_df):
    per_class_df.sort_values(["source", "variant", "class"])
    return


@app.cell
def _():
    import matplotlib.pyplot as plt

    return (plt,)


@app.cell
def _(mo):
    mo.md(
        "## What drives Dice?\n"
        "Per-(source, class) task properties, extracted directly from native-grid ground "
        "truth via each source's `provider.native_gt(subject, cls)` (up to 40 sampled "
        "subjects/class, seeded) -- **not** copied from prose descriptions in the per-source "
        "docs. Precomputed by `compute_class_properties.py` into `class_props_cache.json` "
        "(~4 min, gnc_kidney's 13 classes dominate the cost) since re-scanning NFS on every "
        "notebook open would make it slow; re-run that script if a source's labels change or "
        "a new source is added."
    )
    return


@app.cell
def _(json, os, pd):
    _cache_path = os.path.join(os.path.dirname(__file__), "class_props_cache.json")
    class_props_df = pd.DataFrame(json.load(open(_cache_path)))
    class_props_df
    return (class_props_df,)


@app.cell
def _(class_props_df, per_class_df, pd):
    merged_df = per_class_df.merge(class_props_df, on=["source", "class"], how="left",
                                    suffixes=("", "_prop"))
    return (merged_df,)


@app.cell
def _(merged_df, plt):
    # Dice vs. each extracted task property, every (source, class, variant) row as one point,
    # colored by checkpoint family (same convention as the metric grids above). Mask size and
    # bbox extent on a log-x axis (span orders of magnitude across sources).
    def _color(variant):
        if "exp92_orig" in variant:
            return "#8a8f98"
        if "exp_cascade_register" in variant:
            return "#2a78d6"
        return "#eb6834"

    _colors = merged_df["variant"].map(_color)
    _panels = [
        ("mean_voxels", "mask size (voxels, log)", True),
        ("mean_bbox_extent_mm", "bbox max extent (mm, log)", True),
        ("frac_single_component", "frac. single-component (blob<-->scattered)", False),
        ("mean_contrast", "fg/bg intensity contrast", False),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(22, 5), dpi=140)
    for ax, (col, label, logx) in zip(axes, _panels):
        ax.scatter(merged_df[col], merged_df["mean_dice"], c=_colors, s=22, alpha=0.75,
                   edgecolors="white", linewidths=0.3)
        if logx:
            ax.set_xscale("log")
        ax.set_xlabel(label)
        ax.set_ylabel("Dice")
    fig.suptitle("Dice vs. task properties -- every (source, class, variant) row "
                "(gray=exp92_orig, blue=exp_cascade_register, orange=medverse)")
    fig.tight_layout()
    fig
    return


@app.cell
def _(mo):
    mo.md(
        "**Modality** (CT vs. MRI) isn't a scatter axis -- only hu_lwk1 is CT (6/7 sources "
        "are MRI), so it collapses to a single point per checkpoint family rather than a "
        "trend. See the per-dataset grids above for the CT-vs-MRI split explicitly (hu_lwk1's "
        "panel vs. the rest)."
    )
    return


@app.cell
def _(merged_df, pd):
    # Spearman rank correlation of each task property against Dice, computed within each
    # model/checkpoint family separately (pooling all sources/classes/query_prior variants of
    # that family) -- a monotonic-only, outlier-robust "what predicts Dice" summary.
    _props = ["mean_voxels", "mean_bbox_extent_mm", "frac_single_component", "mean_contrast",
             "mean_n_components"]

    def _family(variant):
        if "exp92_orig" in variant:
            return "patchset3d / exp92_orig"
        if "exp_cascade_register" in variant:
            return "patchset3d / exp_cascade_register"
        return "medverse / released"

    _fam = merged_df.assign(family=merged_df["variant"].map(_family))
    _rows = []
    for fam, g in _fam.groupby("family"):
        row = {"family": fam, "n": len(g)}
        for p in _props:
            row[p] = round(g["mean_dice"].corr(g[p], method="spearman"), 3)
        _rows.append(row)
    correlation_df = pd.DataFrame(_rows).sort_values("family").reset_index(drop=True)
    correlation_df
    return


if __name__ == "__main__":
    app.run()
