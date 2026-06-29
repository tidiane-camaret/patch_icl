"""Focused Dice-extrapolation plot for the most informative axis: foreground_contrast.

Info-preserving (ctx_dice flat), monotone, largest clean generalization gap. The model
trained at a SINGLE point (0.50); everything off it is extrapolation. Shows mean Dice
+/- SE, per-morphology curves, and the extrapolation regions on each side of 0.50.
"""
import os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = os.path.dirname(__file__)
TRAIN = 0.50
# value -> csv file (anchor.csv is the 0.50 train point)
files = {0.05: "contrast_0.05.csv", 0.20: "contrast_0.20.csv", 0.35: "contrast_0.35.csv",
         0.50: "anchor.csv", 0.65: "contrast_0.65.csv", 0.80: "contrast_0.80.csv"}
morphs = ["blob", "elongated", "annular", "tubular", "scattered"]

vals = sorted(files)
mean, se, ctx = [], [], []
permorph = {m: [] for m in morphs}
for v in vals:
    d = pd.read_csv(os.path.join(D, files[v]))
    mean.append(d.dice_native.mean())
    se.append(d.dice_native.std() / np.sqrt(len(d)))
    ctx.append(d.ctx_dice.mean())
    g = d.groupby("morphology").dice_native.mean()
    for m in morphs:
        permorph[m].append(g.get(m, np.nan))
mean, se, ctx = map(np.array, (mean, se, ctx))

fig, ax = plt.subplots(figsize=(9, 6))

# extrapolation shading: anything left/right of the single training point
ax.axvspan(min(vals) - 0.02, TRAIN, color="tab:red", alpha=0.05)
ax.axvspan(TRAIN, max(vals) + 0.02, color="tab:red", alpha=0.05)
ax.axvline(TRAIN, color="tab:green", ls="--", lw=2)
ax.text(TRAIN, 0.79, " trained here\n (single point)", color="tab:green",
        fontsize=10, ha="left", va="top")
ax.text(0.06, 0.10, "← extrapolation\n(fainter fg)", color="tab:red", fontsize=9, alpha=0.8)
ax.text(0.74, 0.10, "extrapolation →\n(stronger fg)", color="tab:red", fontsize=9,
        alpha=0.8, ha="right")

# per-morphology (thin)
for m in morphs:
    ax.plot(vals, permorph[m], "-", lw=1, alpha=0.5, label=f"{m}")

# main curve + SE band
ax.fill_between(vals, mean - se, mean + se, color="tab:blue", alpha=0.20)
ax.plot(vals, mean, "o-", color="tab:blue", lw=2.5, ms=8, label="mean (all morphs)", zorder=5)
for v, mu in zip(vals, mean):
    ax.annotate(f"{mu:.3f}", (v, mu), textcoords="offset points", xytext=(0, 10),
                ha="center", fontsize=8, color="tab:blue")

# ctx_dice (info content) — flat, on twin axis to prove info is preserved
ax2 = ax.twinx()
ax2.plot(vals, ctx, "s:", color="tab:gray", alpha=0.7, label="ctx_dice (info)")
ax2.set_ylabel("ctx_dice (realized context overlap)", color="tab:gray")
ax2.set_ylim(0, 0.5); ax2.tick_params(axis="y", colors="tab:gray")
ax2.text(0.82, ctx[-1] + 0.02, "context info FLAT\n→ pure model brittleness",
         color="tab:gray", fontsize=8, ha="right")

ax.set_xlabel("foreground_contrast")
ax.set_ylabel("mean Dice (dice_native)")
ax.set_ylim(0, 0.85); ax.set_xlim(min(vals) - 0.02, max(vals) + 0.02)
ax.grid(alpha=0.3)
ax.set_title("Dice extrapolation vs foreground_contrast\n"
             "imagepfn_zoom trained on hard_diverse (single train point = 0.50)")
ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
fig.tight_layout()
out = os.path.join(D, "extrapolation_contrast.png")
fig.savefig(out, dpi=130, bbox_inches="tight")
print(f"Saved -> {out}")
print(f"train(0.50)={mean[vals.index(0.50)]:.3f} | faint(0.05)={mean[0]:.3f} "
      f"(-{mean[vals.index(0.50)]-mean[0]:.3f}) | strong(0.80)={mean[-1]:.3f} "
      f"(+{mean[-1]-mean[vals.index(0.50)]:.3f})")
