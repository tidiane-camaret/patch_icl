"""
controlSynth difficulty benchmark — probe every knob with the UniverSeg baseline.

Goal: measure how each controlSynth difficulty knob (build geometry + live appearance)
affects task difficulty for a *zero-training* in-context baseline (UniverSeg, pretrained),
and surface WHY a value is easy/hard (shrinking foreground vs. degraded context vs.
ambiguity), not just that it is.

Method — one-factor-at-a-time (OFAT): pin every knob at a moderate baseline, then sweep
ONE knob across a value grid while the rest stay fixed, so each knob's marginal effect on
Dice is isolated. Morphology-specific knobs (thinness/tortuosity/branching for tubular,
scattered_count/clustering for scattered) are swept with that morphology fixed; the rest
use a clean `blob`. Live knobs reuse one frozen geometry bank (only the per-subject path
changes), so they're cheap.

Each evaluated subject records its full synth param vector + realized stats
(foreground fraction, mean target↔context Dice) + Dice, so the per-sample CSV supports
arbitrary post-hoc analysis. The script also writes a summary table, difficulty curves,
and a text report ranking knobs by Dice sensitivity with per-knob driver correlations.

Usage:
    .venv311/bin/python experiments/2d/synth_benchmark.py
    .venv311/bin/python experiments/2d/synth_benchmark.py --num_tasks 96 --subjects 32
    .venv311/bin/python experiments/2d/synth_benchmark.py --knobs region_size noise_level morphology
    .venv311/bin/python experiments/2d/synth_benchmark.py --quick      # tiny smoke run
"""

import argparse
import copy
import datetime
import sys
from math import ceil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from hydra import compose, initialize_config_dir
from omegaconf import open_dict
from scipy.stats import spearmanr
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))   # experiments/2d
import common                                               # sets src paths; build_dataset/make_loader
from common import DEVICE, build_dataset, hard_dice, make_loader
from src.datasets.controlSynth import dataset as cs_dataset  # for _BANK_CACHE control


# ── Baseline (everything-else-neutral) + sweep grids ──────────────────────────
# Pinned at a moderately-EASY operating point so the swept knob is the only varying
# source of difficulty. Override via --baseline-* if needed (kept in code for clarity).

# Named baselines (`--baseline`). `moderate` was run 1; `easy` lifts the operating
# point off UniverSeg's floor — bigger region + LOW foreground_contrast (run 1 showed
# the contrast knob raises *background* saturation, so 0.1 is the easy/clean end) — to
# give the geometry-detail knobs headroom to reveal whether they actually drive difficulty.
BUILD_PRESETS = {
    "moderate": dict(morphology="blob", thinness=0.5, tortuosity=0.4, branching_density=0.5,
                     region_size=0.30, boundary_complexity=0.20, scattered_count=8,
                     scattered_clustering=0.0, task_ambiguity=0.0),
    "easy":     dict(morphology="blob", thinness=0.5, tortuosity=0.4, branching_density=0.5,
                     region_size=0.60, boundary_complexity=0.20, scattered_count=8,
                     scattered_clustering=0.0, task_ambiguity=0.0),
}
LIVE_PRESETS = {
    "moderate": dict(support_query_shift=0.3, foreground_contrast=0.7, texture_heterogeneity=0.2,
                     noise_level=0.2, context_copy_fraction=0.0, context_consistency=1.0,
                     task_ambiguity_intensity=0.0),
    "easy":     dict(support_query_shift=0.3, foreground_contrast=0.1, texture_heterogeneity=0.2,
                     noise_level=0.2, context_copy_fraction=0.0, context_consistency=1.0,
                     task_ambiguity_intensity=0.0),
}

# Active baseline (reassigned in main from the chosen preset / --set overrides).
BUILD_DEFAULTS = dict(BUILD_PRESETS["moderate"])
LIVE_DEFAULTS = dict(LIVE_PRESETS["moderate"])

# kind: build (frozen geometry -> bank rebuilt per value) | live (bank reused).
# morph: morphology to fix while sweeping (None = sweep morphology itself).
SWEEPS = [
    dict(name="morphology",             kind="build", morph=None,
         values=["blob", "elongated", "annular", "tubular", "scattered"]),
    dict(name="region_size",            kind="build", morph="blob",
         values=[0.05, 0.15, 0.30, 0.50, 0.70, 0.90]),
    dict(name="boundary_complexity",    kind="build", morph="blob",
         values=[0.0, 0.25, 0.5, 0.75, 1.0]),
    dict(name="task_ambiguity",         kind="build", morph="blob",
         values=[0.0, 0.2, 0.4, 0.6, 0.8]),
    dict(name="thinness",               kind="build", morph="tubular",
         values=[0.0, 0.25, 0.5, 0.75, 1.0]),
    dict(name="tortuosity",             kind="build", morph="tubular",
         values=[0.0, 0.25, 0.5, 0.75, 1.0]),
    dict(name="branching_density",      kind="build", morph="tubular",
         values=[0.0, 0.25, 0.5, 0.75, 1.0]),
    dict(name="scattered_count",        kind="build", morph="scattered",
         values=[2, 4, 8, 16, 32]),
    dict(name="scattered_clustering",   kind="build", morph="scattered",
         values=[0.0, 0.33, 0.66, 1.0]),
    dict(name="support_query_shift",    kind="live",  morph="blob",
         values=[0.0, 0.3, 0.6, 1.0]),
    dict(name="foreground_contrast",    kind="live",  morph="blob",
         values=[0.1, 0.3, 0.5, 0.7, 1.0]),
    dict(name="texture_heterogeneity",  kind="live",  morph="blob",
         values=[0.0, 0.3, 0.6, 1.0]),
    dict(name="noise_level",            kind="live",  morph="blob",
         values=[0.0, 0.25, 0.5, 0.75, 1.0]),
    dict(name="context_copy_fraction",  kind="live",  morph="blob",
         values=[0.0, 0.25, 0.5, 1.0]),
    dict(name="context_consistency",    kind="live",  morph="blob",
         values=[1.0, 0.75, 0.5, 0.25, 0.0]),
    dict(name="task_ambiguity_intensity", kind="live", morph="blob",
         values=[0.0, 0.3, 0.6, 1.0]),
]

# Scalar param columns present in every per-sample row (geo + live), for the CSV.
# `morphology` is excluded — it is categorical and recorded as its own column.
PARAM_KEYS = [k for k in BUILD_DEFAULTS if k != "morphology"] + list(LIVE_DEFAULTS)


# ── cfg construction ──────────────────────────────────────────────────────────

def base_cfg(args):
    cfg_dir = str(_ROOT / "configs" / "experiment" / "2d")
    with initialize_config_dir(config_dir=cfg_dir, version_base=None):
        cfg = compose(config_name="eval_base", overrides=[
            "data.source=synthetic",
            f"data.image_size={args.image_size}",
            f"data.context_size={args.context_size}",
            "data.split=val",
            "eval.max_per_label=null",
            f"eval.batch_size={args.batch_size}",
            f"eval.workers={args.workers}",
            f"synth.diversity.num_tasks={args.num_tasks}",
            f"synth.sampling.eval_subjects_per_task={args.subjects}",
        ])
    return cfg


def point_cfg(base, sweep, value):
    """A cfg with all knobs pinned at baseline and the swept one set to `value`."""
    cfg = copy.deepcopy(base)
    with open_dict(cfg):
        for k, v in BUILD_DEFAULTS.items():
            cfg.synth.build[k] = v
        for k, v in LIVE_DEFAULTS.items():
            cfg.synth.live[k] = v
        cfg.synth.build.mode = "fixed"
        cfg.synth.build.sampled = {}
        cfg.synth.build.n_bins = 1
        if sweep["name"] == "morphology":
            cfg.synth.build.morphology = value
        else:
            cfg.synth.build.morphology = sweep["morph"]
            if sweep["kind"] == "build":
                cfg.synth.build[sweep["name"]] = value
            else:
                cfg.synth.live[sweep["name"]] = value
    return cfg


# ── evaluation ────────────────────────────────────────────────────────────────

def _num(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    return x


@torch.no_grad()
def eval_loader(model, cfg):
    """Run UniverSeg over one deterministic synth val grid; one tagless row per subject."""
    ds = build_dataset(cfg, "val")
    loader = make_loader(ds, cfg, "val", shuffle=False)
    rows = []
    for batch in loader:
        if batch is None:
            continue
        images  = batch["image"].to(DEVICE, non_blocking=True)
        ctx_in  = batch["context_in"].to(DEVICE, non_blocking=True)
        ctx_out = batch["context_out"]
        with torch.autocast(device_type=DEVICE.type, enabled=DEVICE.type == "cuda"):
            out = model(images, context_in=ctx_in,
                        context_out=ctx_out.to(DEVICE, non_blocking=True), mode="val")
        preds = (out["final_logit"] > 0).float().cpu()        # (B, 1, H, W)
        metas, K = batch["meta"], ctx_out.shape[1]
        for b in range(len(metas)):
            label = batch["label"][b, 0]
            d  = hard_dice(preds[b, 0], label)
            fg = float((label > 0).float().mean())
            cd = float(np.nanmean([hard_dice(label, ctx_out[b, k, 0]) for k in range(K)]))
            m, diff, axis = metas[b], dict(metas[b]["difficulty"]), metas[b]["axis"]
            rows.append({
                "morphology": m["morphology"], "task_id": int(m["task_id"]),
                "subject_index": int(m["subject_index"]),
                **{k: _num(diff.get(k)) for k in PARAM_KEYS},
                "axis_identification": float(axis["identification"]),
                "axis_segmentation": float(axis["segmentation"]),
                "fg_frac": fg, "ctx_dice": cd, "dice": d,
            })
    return rows


def evaluate_point(model, cfg, sweep, value):
    rows = eval_loader(model, cfg)
    for r in rows:
        r["kind"], r["knob"], r["value"] = sweep["kind"], sweep["name"], _num(value)
    return rows


# ── analysis & reporting ──────────────────────────────────────────────────────

def _spear(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    if ok.sum() < 3 or np.ptp(x[ok]) == 0 or np.ptp(y[ok]) == 0:
        return float("nan")
    return float(spearmanr(x[ok], y[ok]).statistic)


def summarize(df):
    g = df.groupby(["kind", "knob", "value"], sort=False)
    agg = g.agg(
        n=("dice", "size"),
        dice_mean=("dice", "mean"),
        dice_median=("dice", "median"),
        dice_std=("dice", "std"),
        fail_rate=("dice", lambda s: float((s < 0.1).mean())),    # total-miss fraction
        fg_frac=("fg_frac", "mean"),
        ctx_dice=("ctx_dice", "mean"),
    ).reset_index()
    return agg


def write_report(df, agg, path, args):
    lines = []
    w = lines.append
    w("controlSynth difficulty benchmark — UniverSeg baseline")
    w(f"  baseline_preset={args.baseline}  overrides={args.overrides}")
    w(f"  image_size={args.image_size}  context_size={args.context_size}  "
      f"num_tasks={args.num_tasks}  subjects/task={args.subjects}  N={len(df)} subjects")
    w(f"  baseline build={BUILD_DEFAULTS}")
    w(f"  baseline live ={LIVE_DEFAULTS}")
    w("")

    # Knob sensitivity: spread of mean Dice across a knob's values (how much the knob
    # moves difficulty at all), with the easiest/hardest value called out.
    w("=" * 78)
    w("KNOB SENSITIVITY (ranked by Dice spread across swept values)")
    w("=" * 78)
    w(f"{'knob':<24}{'kind':<7}{'Δdice':>7}  easiest -> hardest")
    rank = []
    for (kind, knob), sub in agg.groupby(["kind", "knob"], sort=False):
        sub = sub.reset_index(drop=True)
        i_hi, i_lo = sub.dice_mean.idxmax(), sub.dice_mean.idxmin()
        rng = float(sub.dice_mean.max() - sub.dice_mean.min())
        rank.append((rng, kind, knob, sub.loc[i_hi], sub.loc[i_lo]))
    for rng, kind, knob, hi, lo in sorted(rank, key=lambda r: -r[0]):
        w(f"{knob:<24}{kind:<7}{rng:>7.3f}  "
          f"{hi.value}={hi.dice_mean:.3f}  ->  {lo.value}={lo.dice_mean:.3f}")
    w("")

    # Per-knob drivers: within a knob's rows, does Dice track the foreground fraction
    # (hard because the target shrinks) or context Dice (hard because context is
    # uninformative)? |rho|>=0.4 gets an inline interpretation.
    w("=" * 78)
    w("WHY: per-knob Spearman(Dice, driver) within each sweep")
    w("=" * 78)
    w(f"{'knob':<24}{'rho(fg_frac)':>13}{'rho(ctx_dice)':>14}  note")
    for (kind, knob), sub in df.groupby(["kind", "knob"], sort=False):
        r_fg, r_cd = _spear(sub.dice, sub.fg_frac), _spear(sub.dice, sub.ctx_dice)
        notes = []
        if not np.isnan(r_fg) and abs(r_fg) >= 0.4:
            notes.append("foreground-size-mediated" if r_fg > 0 else "inverse-fg")
        if not np.isnan(r_cd) and abs(r_cd) >= 0.4:
            notes.append("context-informativeness-mediated" if r_cd > 0 else "inverse-ctx")
        w(f"{knob:<24}{r_fg:>13.3f}{r_cd:>14.3f}  {', '.join(notes)}")
    w("")

    # Pooled global drivers (confounded by OFAT pinning — directional only).
    w("=" * 78)
    w("GLOBAL drivers (pooled across all sweeps; directional, confounded by OFAT)")
    w("=" * 78)
    drivers = ["fg_frac", "ctx_dice", "foreground_contrast", "noise_level",
               "task_ambiguity", "task_ambiguity_intensity", "region_size",
               "texture_heterogeneity", "support_query_shift", "boundary_complexity"]
    for col in drivers:
        if col in df:
            w(f"  rho(Dice, {col:<24}) = {_spear(df.dice, df[col]):.3f}")
    w("")

    # Morphology headline.
    if "morphology" in agg.knob.values:
        w("MORPHOLOGY mean Dice:")
        mo = agg[agg.knob == "morphology"].sort_values("dice_mean", ascending=False)
        for _, r in mo.iterrows():
            w(f"  {str(r.value):<12} dice={r.dice_mean:.3f}  fail={r.fail_rate:.2f}  "
              f"fg={r.fg_frac:.3f}  ctx={r.ctx_dice:.3f}")
        w("")

    Path(path).write_text("\n".join(lines))
    print("\n".join(lines))


def plot_curves(agg, out_dir):
    numeric = agg[agg.knob != "morphology"].copy()
    knobs = list(dict.fromkeys(numeric.knob))
    if knobs:
        ncol = 4
        nrow = ceil(len(knobs) / ncol)
        fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3.2 * nrow), squeeze=False)
        for ax, knob in zip(axes.flat, knobs):
            sub = numeric[numeric.knob == knob].copy()
            sub["value"] = sub.value.astype(float)
            sub = sub.sort_values("value")
            kind = sub.kind.iloc[0]
            ax.plot(sub.value, sub.dice_mean, "-o", color="C0")
            ax.fill_between(sub.value, sub.dice_mean - sub.dice_std.fillna(0),
                            sub.dice_mean + sub.dice_std.fillna(0), alpha=0.15, color="C0")
            ax.plot(sub.value, sub.fail_rate, "--s", color="C3", alpha=0.6, label="fail rate")
            ax.set_title(f"{knob} ({kind})", fontsize=9)
            ax.set_ylim(-0.02, 1.02)
            ax.set_xlabel("value"); ax.grid(alpha=0.3)
        for ax in axes.flat[len(knobs):]:
            ax.axis("off")
        axes.flat[0].set_ylabel("Dice (blue) / fail rate (red)")
        fig.suptitle("controlSynth difficulty curves — UniverSeg baseline", fontsize=11)
        fig.tight_layout()
        fig.savefig(out_dir / "difficulty_curves.png", dpi=120, bbox_inches="tight")
        plt.close(fig)

    mo = agg[agg.knob == "morphology"]
    if len(mo):
        fig, ax = plt.subplots(figsize=(6, 3.5))
        mo = mo.sort_values("dice_mean")
        ax.barh([str(v) for v in mo.value], mo.dice_mean, color="C0")
        ax.set_xlim(0, 1); ax.set_xlabel("mean Dice"); ax.set_title("Morphology difficulty")
        for y, (d, f) in enumerate(zip(mo.dice_mean, mo.fail_rate)):
            ax.text(d + 0.01, y, f"{d:.2f} (fail {f:.2f})", va="center", fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / "morphology_difficulty.png", dpi=120, bbox_inches="tight")
        plt.close(fig)


# ── 2D interaction grid ───────────────────────────────────────────────────────
# Jointly sweep two knobs to expose interactions OFAT can't — notably the spec's
# identification axis: build `task_ambiguity` makes background regions share the fg
# SHAPE; live `task_ambiguity_intensity` makes them share the fg INTENSITY. Each alone
# is separable (by intensity / by shape); only TOGETHER are distractors confusable, so
# the model must use context to identify the true fg. A Dice drop confined to the
# joint-high corner = real identification difficulty.

DEFAULT_GRID = ["build:task_ambiguity=0.0,0.3,0.6,0.9",
                "live:task_ambiguity_intensity=0.0,0.5,1.0"]


def parse_axis(spec):
    """'build:task_ambiguity=0.0,0.3,0.6,0.9' -> dict(kind, name, values)."""
    head, _, vals = spec.partition("=")
    kind, _, name = head.partition(":")
    cast = int if name == "scattered_count" else float
    return dict(kind=kind, name=name, values=[cast(v) for v in vals.split(",")])


def grid_cfg(base, axis_vals):
    cfg = copy.deepcopy(base)
    with open_dict(cfg):
        for k, v in BUILD_DEFAULTS.items():
            cfg.synth.build[k] = v
        for k, v in LIVE_DEFAULTS.items():
            cfg.synth.live[k] = v
        cfg.synth.build.mode = "fixed"
        cfg.synth.build.sampled = {}
        cfg.synth.build.n_bins = 1
        cfg.synth.build.morphology = BUILD_DEFAULTS["morphology"]
        for kind, name, val in axis_vals:
            (cfg.synth.build if kind == "build" else cfg.synth.live)[name] = val
    return cfg


def run_grid(model, base, a1, a2):
    if a1["kind"] == "live" and a2["kind"] == "build":      # build axis outer (bank rebuild)
        a1, a2 = a2, a1
    rows = []
    for v1 in tqdm(a1["values"], desc=a1["name"], dynamic_ncols=True):
        if a1["kind"] == "build":
            cs_dataset._BANK_CACHE.clear()
        for v2 in a2["values"]:
            cfg = grid_cfg(base, [(a1["kind"], a1["name"], v1), (a2["kind"], a2["name"], v2)])
            r = eval_loader(model, cfg)
            for x in r:
                x["ax1"], x["ax1_value"] = a1["name"], _num(v1)
                x["ax2"], x["ax2_value"] = a2["name"], _num(v2)
            rows.extend(r)
            md = float(np.nanmean([x["dice"] for x in r])) if r else float("nan")
            tqdm.write(f"  {a1['name']}={v1!s:<5} {a2['name']}={v2!s:<5} "
                       f"n={len(r):<4} dice={md:.3f}")
    return rows, a1, a2


def grid_report(df, a1, a2, out_dir):
    piv  = df.pivot_table(index="ax1_value", columns="ax2_value", values="dice", aggfunc="mean")
    fail = df.pivot_table(index="ax1_value", columns="ax2_value", values="dice",
                          aggfunc=lambda s: float((s < 0.1).mean()))
    piv.to_csv(out_dir / "grid_dice.csv")

    fig, ax = plt.subplots(figsize=(1.5 + 1.1 * piv.shape[1], 1.5 + 1.0 * piv.shape[0]))
    im = ax.imshow(piv.values, origin="lower", cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(piv.shape[1])); ax.set_xticklabels([f"{c:g}" for c in piv.columns])
    ax.set_yticks(range(piv.shape[0])); ax.set_yticklabels([f"{r:g}" for r in piv.index])
    ax.set_xlabel(a2["name"]); ax.set_ylabel(a1["name"])
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color="w" if v < 0.6 else "k", fontsize=9)
    fig.colorbar(im, ax=ax, label="mean Dice")
    ax.set_title(f"{a1['name']} × {a2['name']} (UniverSeg Dice)")
    fig.tight_layout(); fig.savefig(out_dir / "grid_heatmap.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    lo1, hi1, lo2, hi2 = piv.index.min(), piv.index.max(), piv.columns.min(), piv.columns.max()
    base_d, both_d = piv.loc[lo1, lo2], piv.loc[hi1, hi2]
    a1_only, a2_only = piv.loc[hi1, lo2], piv.loc[lo1, hi2]
    inter = (both_d - base_d) - (a1_only - base_d) - (a2_only - base_d)
    lines = [
        f"controlSynth interaction grid — {a1['name']} (rows) × {a2['name']} (cols)",
        f"  N={len(df)} subjects", "", "mean Dice:", piv.round(3).to_string(),
        "", "fail rate (Dice<0.1):", fail.round(3).to_string(), "",
        f"baseline           ({a1['name']}={lo1:g}, {a2['name']}={lo2:g}): {base_d:.3f}",
        f"{a1['name']} alone (={hi1:g}):  {a1_only:.3f}  (Δ={a1_only - base_d:+.3f})",
        f"{a2['name']} alone (={hi2:g}):  {a2_only:.3f}  (Δ={a2_only - base_d:+.3f})",
        f"both high          (={hi1:g}, ={hi2:g}): {both_d:.3f}  (Δ={both_d - base_d:+.3f})",
        "",
        f"INTERACTION (joint − sum of marginals) = {inter:+.3f}  "
        "(negative => super-additive difficulty only when combined = real identification axis)",
    ]
    (out_dir / "grid_report.txt").write_text("\n".join(lines))
    print("\n".join(lines))


# ── qualitative panels ────────────────────────────────────────────────────────
# For each knob, render the target (GT green / pred red) + context images at several
# values, annotated with the UniverSeg Dice — so the difficulty curves can be read
# against what the images actually look like.

def select_values(values, k=4):
    if len(values) <= k:
        return list(values)
    idx = sorted(set(np.linspace(0, len(values) - 1, k).round().astype(int)))
    return [values[i] for i in idx]


@torch.no_grad()
def predict_one(model, item):
    img = item["image"].unsqueeze(0).to(DEVICE)
    ci  = item["context_in"].unsqueeze(0).to(DEVICE)
    co  = item["context_out"].unsqueeze(0).to(DEVICE)
    with torch.autocast(device_type=DEVICE.type, enabled=DEVICE.type == "cuda"):
        out = model(img, context_in=ci, context_out=co, mode="val")
    pred = (out["final_logit"] > 0).float().cpu()[0, 0]
    return pred, hard_dice(pred, item["label"][0])


def plot_knob_panel(model, base, sweep, out_dir, n_mean=8):
    values = select_values(sweep["values"], k=5 if sweep["name"] == "morphology" else 4)
    K = base.data.context_size
    ncol = 1 + K
    fig, axes = plt.subplots(len(values), ncol, figsize=(2.4 * ncol, 2.4 * len(values)),
                             squeeze=False)
    for r, value in enumerate(values):
        if sweep["kind"] == "build":
            cs_dataset._BANK_CACHE.clear()
        ds = build_dataset(point_cfg(base, sweep, value), "val")
        mean_d = float(np.nanmean([predict_one(model, ds[j])[1]
                                   for j in range(min(n_mean, len(ds)))]))
        item = ds[0]
        pred, _ = predict_one(model, item)
        timg, tseg = item["image"][0].numpy(), item["label"][0].numpy()

        ax = axes[r, 0]
        ax.imshow(timg, cmap="gray", vmin=0, vmax=1)
        if tseg.max() > 0.5:
            ax.contour(tseg, levels=[0.5], colors="lime", linewidths=1.3)
        if pred.numpy().max() > 0.5:
            ax.contour(pred.numpy(), levels=[0.5], colors="red", linewidths=1.0)
        ax.set_ylabel(f"{sweep['name']}={value}\nDice {mean_d:.2f}", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
        if r == 0:
            ax.set_title("target: GT(green)/pred(red)", fontsize=8)
        for k in range(K):
            ax = axes[r, 1 + k]
            cimg, cseg = item["context_in"][k, 0].numpy(), item["context_out"][k, 0].numpy()
            ax.imshow(cimg, cmap="gray", vmin=0, vmax=1)
            ax.imshow(np.ma.masked_where(cseg < 0.5, cseg), cmap="autumn", alpha=0.45,
                      vmin=0, vmax=1)
            ax.axis("off")
            if r == 0:
                ax.set_title(f"context {k}", fontsize=8)
    fig.suptitle(f"controlSynth — {sweep['name']} ({sweep['kind']}, baseline={getattr(plot_knob_panel, '_bl', 'easy')})",
                 fontsize=11)
    fig.tight_layout()
    p = out_dir / "panels" / f"{sweep['name']}.png"
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return p


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_tasks", type=int, default=48)
    ap.add_argument("--subjects", type=int, default=24, help="eval subjects per task")
    ap.add_argument("--image_size", type=int, default=128)
    ap.add_argument("--context_size", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--knobs", nargs="*", default=None, help="subset of knob names; default all")
    ap.add_argument("--baseline", choices=list(BUILD_PRESETS), default="moderate",
                    help="pinned operating point for non-swept knobs")
    ap.add_argument("--set", nargs="*", default=None, metavar="knob=value", dest="overrides",
                    help="override individual baseline knobs, e.g. --set region_size=0.5 noise_level=0.4")
    ap.add_argument("--grid", nargs="*", default=None, metavar="kind:knob=v,v,v",
                    help="2D interaction grid of two axes; bare --grid uses the ambiguity grid "
                         f"({' × '.join(DEFAULT_GRID)})")
    ap.add_argument("--plot", action="store_true",
                    help="render qualitative target+context panels per knob (with UniverSeg Dice)")
    ap.add_argument("--out_dir", type=str, default=None)
    ap.add_argument("--quick", action="store_true", help="tiny run (smoke test)")
    args = ap.parse_args()

    global BUILD_DEFAULTS, LIVE_DEFAULTS
    BUILD_DEFAULTS = dict(BUILD_PRESETS[args.baseline])
    LIVE_DEFAULTS = dict(LIVE_PRESETS[args.baseline])
    for kv in (args.overrides or []):
        k, _, v = kv.partition("=")
        if k in BUILD_DEFAULTS:
            tgt, cast = BUILD_DEFAULTS, (int if k == "scattered_count" else float)
        elif k in LIVE_DEFAULTS:
            tgt, cast = LIVE_DEFAULTS, float
        elif k == "morphology":
            tgt, cast = BUILD_DEFAULTS, str
        else:
            raise SystemExit(f"--set: unknown knob {k!r}")
        tgt[k] = cast(v)

    if args.quick:
        args.num_tasks, args.subjects, args.workers = 12, 4, 0
        if args.knobs is None:
            args.knobs = ["morphology", "region_size", "noise_level"]

    sweeps = SWEEPS if args.knobs is None else [s for s in SWEEPS if s["name"] in args.knobs]
    if not sweeps:
        raise SystemExit(f"no matching knobs in {args.knobs!r}")

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir or (_ROOT / "results" / "2d" / "synth_benchmark" / stamp))
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output -> {out_dir}")

    if DEVICE.type == "cuda":
        torch.backends.cudnn.benchmark = True

    print(f"Loading UniverSeg (size={args.image_size})...")
    from src.models.universeg_baseline import UniverSegBaseline
    model = UniverSegBaseline(pretrained=True, input_size=args.image_size).to(DEVICE).eval()

    base = base_cfg(args)

    if args.plot:
        plot_knob_panel._bl = args.baseline
        print(f"Rendering qualitative panels (baseline={args.baseline})...")
        for sweep in sweeps:
            if sweep["kind"] == "build":
                cs_dataset._BANK_CACHE.clear()
            p = plot_knob_panel(model, base, sweep, out_dir)
            print(f"  {sweep['name']:<22} -> {p}")
        print(f"\nPanels -> {out_dir / 'panels'}")
        return

    if args.grid is not None:
        specs = args.grid if len(args.grid) >= 2 else DEFAULT_GRID
        a1, a2 = parse_axis(specs[0]), parse_axis(specs[1])
        print(f"Grid: {a1['kind']}:{a1['name']}={a1['values']} × "
              f"{a2['kind']}:{a2['name']}={a2['values']}  (baseline={args.baseline})")
        rows, a1, a2 = run_grid(model, base, a1, a2)
        df = pd.DataFrame(rows)
        df.to_csv(out_dir / "per_sample_grid.csv", index=False)
        grid_report(df, a1, a2, out_dir)
        print(f"\nWrote: per_sample_grid.csv ({len(df)} rows), grid_dice.csv, "
              f"grid_heatmap.png, grid_report.txt\n-> {out_dir}")
        return

    all_rows = []
    for sweep in sweeps:
        # Bound memory: build-knob banks differ per value (drop the previous sweep's);
        # a live sweep keeps one bank and reuses it across values (same build_spec).
        cs_dataset._BANK_CACHE.clear()
        for value in tqdm(sweep["values"], desc=f"{sweep['name']:<22}", dynamic_ncols=True):
            cfg = point_cfg(base, sweep, value)
            rows = evaluate_point(model, cfg, sweep, value)
            all_rows.extend(rows)
            md = float(np.nanmean([r["dice"] for r in rows])) if rows else float("nan")
            tqdm.write(f"  {sweep['name']:<22} = {str(value):<10} "
                       f"n={len(rows):<4} mean Dice={md:.3f}")

    df = pd.DataFrame(all_rows)
    df.to_csv(out_dir / "per_sample.csv", index=False)
    agg = summarize(df)
    agg.to_csv(out_dir / "summary.csv", index=False)
    plot_curves(agg, out_dir)
    write_report(df, agg, out_dir / "report.txt", args)
    print(f"\nWrote: per_sample.csv ({len(df)} rows), summary.csv, "
          f"difficulty_curves.png, morphology_difficulty.png, report.txt\n-> {out_dir}")


if __name__ == "__main__":
    main()
