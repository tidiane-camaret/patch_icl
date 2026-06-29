"""Dice-extrapolation plot for noise_level (mean Dice only, no per-shape).

Info-preserving axis (ctx_dice flat), monotone. Model trained at a single point (0.40);
everything off it is extrapolation.
"""
import os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = os.path.dirname(__file__)
TRAIN = 0.40
files = {0.10: "noise_0.10.csv", 0.25: "noise_0.25.csv", 0.40: "anchor.csv",
         0.55: "noise_0.55.csv", 0.70: "noise_0.70.csv", 0.85: "noise_0.85.csv",
         1.00: "noise_1.00.csv"}

vals = sorted(files)
mean, se, ctx = [], [], []
for v in vals:
    d = pd.read_csv(os.path.join(D, files[v]))
    mean.append(d.dice_native.mean())
    se.append(d.dice_native.std() / np.sqrt(len(d)))
    ctx.append(d.ctx_dice.mean())
mean, se, ctx = map(np.array, (mean, se, ctx))

fig, ax = plt.subplots(figsize=(9, 6))
ax.axvspan(min(vals) - 0.02, TRAIN, color="tab:red", alpha=0.05)
ax.axvspan(TRAIN, max(vals) + 0.02, color="tab:red", alpha=0.05)
ax.axvline(TRAIN, color="tab:green", ls="--", lw=2)
ax.text(TRAIN, 0.79, " trained here\n (single point)", color="tab:green",
        fontsize=10, ha="left", va="top")
ax.text(0.12, 0.10, "← extrapolation\n(cleaner)", color="tab:red", fontsize=9, alpha=0.8)
ax.text(0.97, 0.10, "extrapolation →\n(noisier)", color="tab:red", fontsize=9,
        alpha=0.8, ha="right")

ax.fill_between(vals, mean - se, mean + se, color="tab:blue", alpha=0.20)
ax.plot(vals, mean, "o-", color="tab:blue", lw=2.5, ms=8, label="mean Dice", zorder=5)
for v, mu in zip(vals, mean):
    ax.annotate(f"{mu:.3f}", (v, mu), textcoords="offset points", xytext=(0, 10),
                ha="center", fontsize=8, color="tab:blue")

ax2 = ax.twinx()
ax2.plot(vals, ctx, "s:", color="tab:gray", alpha=0.7, label="ctx_dice (info)")
ax2.set_ylabel("ctx_dice (realized context overlap)", color="tab:gray")
ax2.set_ylim(0, 0.5); ax2.tick_params(axis="y", colors="tab:gray")
ax2.text(0.98, ctx[-1] + 0.02, "context info FLAT\n→ pure model brittleness",
         color="tab:gray", fontsize=8, ha="right")

ax.set_xlabel("noise_level")
ax.set_ylabel("mean Dice (dice_native)")
ax.set_ylim(0, 0.85); ax.set_xlim(min(vals) - 0.02, max(vals) + 0.02)
ax.grid(alpha=0.3)
ax.set_title("Dice extrapolation vs noise_level\n"
             "imagepfn_zoom trained on hard_diverse (single train point = 0.40)")
ax.legend(loc="lower left", fontsize=9, framealpha=0.9)
fig.tight_layout()
out = os.path.join(D, "extrapolation_noise.png")
fig.savefig(out, dpi=130, bbox_inches="tight")
print(f"Saved -> {out}")
print(f"train(0.40)={mean[vals.index(0.40)]:.3f} | clean(0.10)={mean[0]:.3f} "
      f"(+{mean[0]-mean[vals.index(0.40)]:.3f}) | noisy(1.00)={mean[-1]:.3f} "
      f"(-{mean[vals.index(0.40)]-mean[-1]:.3f})")
