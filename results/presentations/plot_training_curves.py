"""Val-Dice training curve comparison between two wandb runs (+ optional continuation runs).

Pulls history from the wandb API, stitches each model's original + continuation
run into one curve, and plots val/dice vs epoch. Continuation runs that reset
their own epoch counter (weights-only warm start) are shifted by `splice_epoch`
so the x-axis reads as cumulative epochs; the plot itself carries no indication
of where a splice happened.

Usage: edit the RUNS config below, then `python plot_training_curves.py`.
"""
import wandb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ENTITY = "tidiane-camaret-ndir-universit-tsklinikum-freiburg"
PROJECT = "patchset_train"

# label -> {color, run ids [original, continuation?], splice_epoch (only used if
# the continuation run's own epoch counter restarts at 0)}
RUNS = {
    "patchset (ours)": {
        "color": "#2a78d6",
        "run_ids": ["v48ucp2c"],  # 84_cascade_varspacing_GCP_h100, from scratch
        "splice_epoch": 0,
    },
    "medverse": {
        "color": "#eb6834",
        "run_ids": ["dm8f4jor"],  # 91_medverse_multisource_intensity_augs, from orig_weights
        "splice_epoch": 0,
    },
}

CUT_AT_SHORTEST = True  # trim all curves to the shortest run's max epoch
OUT_PATH = "results/presentations/patchset_vs_medverse_cascade_training_curves.png"

TEXT_PRIMARY = "#0b0b0b"
SURFACE = "#fcfcfb"
GRID = "#e4e3dd"


def fetch_val_dice(run_id):
    """Return sorted [(epoch, val/dice), ...] for one wandb run."""
    run = wandb.Api().run(f"{ENTITY}/{PROJECT}/{run_id}")
    hist = run.history(keys=["epoch", "val/dice"], pandas=False)
    return sorted((h["epoch"], h["val/dice"]) for h in hist if "val/dice" in h and "epoch" in h)


def stitched_curve(run_ids, splice_epoch):
    """Concatenate original + continuation run(s), offsetting continuation epochs
    by splice_epoch and de-duping the shared splice point."""
    points = fetch_val_dice(run_ids[0])
    for rid in run_ids[1:]:
        cont = fetch_val_dice(rid)
        offset = splice_epoch if cont[0][0] == 0 else 0
        points += [(e + offset, v) for e, v in cont]
    seen, deduped = set(), []
    for e, v in points:
        if e in seen:
            continue
        seen.add(e)
        deduped.append((e, v))
    return deduped


def main():
    curves = {label: stitched_curve(cfg["run_ids"], cfg["splice_epoch"]) for label, cfg in RUNS.items()}

    if CUT_AT_SHORTEST:
        max_epoch = min(curve[-1][0] for curve in curves.values())
        curves = {label: [(e, v) for e, v in curve if e <= max_epoch] for label, curve in curves.items()}

    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["font.size"] = 18

    fig, ax = plt.subplots(figsize=(12.5, 7), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    for label, cfg in RUNS.items():
        color = cfg["color"]
        x, y = zip(*curves[label])
        ax.plot(x, y, color=color, linewidth=2.4, marker="o", markersize=5.5,
                 markerfacecolor=SURFACE, markeredgecolor=color, markeredgewidth=1.6,
                 zorder=3, label=label)
        ax.annotate(f"{label}\nval dice {y[-1]:.3f}", xy=(x[-1], y[-1]), xytext=(8, 0),
                    textcoords="offset points", color=color, fontsize=17,
                    fontweight="bold", va="center", ha="left")

    ax.legend(loc="upper left", frameon=False, labelcolor=TEXT_PRIMARY, fontsize=18)
    ax.set_xlabel("Epoch", fontsize=18)
    ax.set_ylabel("Dice", fontsize=18)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(GRID)
    ax.tick_params(labelsize=16)
    ax.set_xlim(left=0, right=max(c[-1][0] for c in curves.values()) + 22)
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    fig.savefig(OUT_PATH, facecolor=SURFACE)
    print("saved", OUT_PATH)


if __name__ == "__main__":
    main()
