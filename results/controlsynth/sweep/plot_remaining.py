"""Mean-Dice extrapolation panels for the remaining swept axes (no per-shape).

Blue = mean Dice +/- SE. Grey dotted (twin axis) = ctx_dice (realized context overlap):
flat => info-preserving (drop is pure model brittleness); falling => the shift also
removes usable context (degradation partly inherent). Green dashed = single training
value (live axes) or green band = training range (build axes).
"""
import os, glob, re, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = os.path.dirname(__file__)

# axis key -> (label, train value or None, train range or None, anchor-is-train?)
AX = {
    "texture":     ("texture_heterogeneity", 0.35, None),
    "ambint":      ("task_ambiguity_intensity", 0.60, None),
    "shift":       ("support_query_shift", 0.50, None),
    "scale":       ("support_query_scale", 0.45, None),
    "translate":   ("support_query_translate", 0.05, None),
    "consistency": ("context_consistency", 0.90, None),
    "regionsize":  ("region_size", None, (0.12, 0.62)),
    "taskamb":     ("task_ambiguity (build)", None, (0.30, 0.80)),
}
LIVE = {"texture", "ambint", "shift", "scale", "translate", "consistency"}

def load_axis(key, train):
    rows = []
    for f in glob.glob(os.path.join(D, f"{key}_*.csv")):
        m = re.match(rf"{key}_([0-9.]+)$", os.path.basename(f)[:-4])
        if m:
            rows.append((float(m.group(1)), f))
    if key in LIVE:  # shared anchor = the single training point
        rows.append((train, os.path.join(D, "anchor.csv")))
    rows.sort()
    vals, mean, se, ctx = [], [], [], []
    for v, f in rows:
        d = pd.read_csv(f)
        vals.append(v); mean.append(d.dice_native.mean())
        se.append(d.dice_native.std() / np.sqrt(len(d))); ctx.append(d.ctx_dice.mean())
    return map(np.array, (vals, mean, se, ctx))

fig, axes = plt.subplots(2, 4, figsize=(20, 9), squeeze=False)
for i, (key, (label, tv, tr)) in enumerate(AX.items()):
    ax = axes[i // 4][i % 4]
    vals, mean, se, ctx = load_axis(key, tv)
    ax.fill_between(vals, mean - se, mean + se, color="tab:blue", alpha=0.20)
    ax.plot(vals, mean, "o-", color="tab:blue", lw=2.3, ms=6, zorder=5)
    for v, mu in zip(vals, mean):
        ax.annotate(f"{mu:.2f}", (v, mu), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=7, color="tab:blue")
    if tv is not None:
        ax.axvline(tv, color="tab:green", ls="--", lw=2)
    if tr is not None:
        ax.axvspan(tr[0], tr[1], color="tab:green", alpha=0.12)
    ax2 = ax.twinx()
    ax2.plot(vals, ctx, "s:", color="tab:gray", alpha=0.7)
    ax2.set_ylim(0, 0.5); ax2.tick_params(axis="y", colors="tab:gray", labelsize=8)
    flat = (ctx.max() - ctx.min()) < 0.03
    ax.set_title(f"{label}   [{'info-preserving' if flat else 'ctx_dice falls'}]", fontsize=11)
    ax.set_xlabel(key); ax.set_ylabel("mean Dice"); ax.set_ylim(0, 0.85); ax.grid(alpha=0.3)

fig.suptitle("Per-axis Dice extrapolation (mean only) — imagepfn_zoom / hard_diverse "
             "(anchor=0.647). Green=train; grey dotted=ctx_dice info content", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.97])
out = os.path.join(D, "extrapolation_remaining.png")
fig.savefig(out, dpi=110, bbox_inches="tight")
print(f"Saved -> {out}")
