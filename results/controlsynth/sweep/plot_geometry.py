"""Mean-Dice extrapolation for the geometry/context-pose axes
support_query_translate, region_size, support_query_scale (no ctx_dice).

Blue = mean Dice +/- SE. Green dashed = single training value (live axes);
green band = training range (build axis region_size).
"""
import os, glob, re, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = os.path.dirname(__file__)

# key -> (label, train value or None, train range or None)
AX = {
    "translate":  ("support_query_translate", 0.05, None),
    "regionsize": ("region_size", None, (0.12, 0.62)),
    "scale":      ("support_query_scale", 0.45, None),
}
LIVE = {"translate", "scale"}  # share the single in-dist anchor (0.647)

def load_axis(key, train):
    rows = []
    for f in glob.glob(os.path.join(D, f"{key}_*.csv")):
        m = re.match(rf"{key}_([0-9.]+)$", os.path.basename(f)[:-4])
        if m:
            rows.append((float(m.group(1)), f))
    if key in LIVE:
        rows.append((train, os.path.join(D, "anchor.csv")))
    rows.sort()
    vals, mean, se = [], [], []
    for v, f in rows:
        d = pd.read_csv(f)
        vals.append(v); mean.append(d.dice_native.mean())
        se.append(d.dice_native.std() / np.sqrt(len(d)))
    return map(np.array, (vals, mean, se))

fig, axes = plt.subplots(1, 3, figsize=(16, 5), squeeze=False)
for i, (key, (label, tv, tr)) in enumerate(AX.items()):
    ax = axes[0][i]
    vals, mean, se = load_axis(key, tv)
    ax.fill_between(vals, mean - se, mean + se, color="tab:blue", alpha=0.20)
    ax.plot(vals, mean, "o-", color="tab:blue", lw=2.5, ms=7, zorder=5)
    for v, mu in zip(vals, mean):
        ax.annotate(f"{mu:.2f}", (v, mu), textcoords="offset points", xytext=(0, 9),
                    ha="center", fontsize=8, color="tab:blue")
    if tv is not None:
        ax.axvline(tv, color="tab:green", ls="--", lw=2, label="trained value")
    if tr is not None:
        ax.axvspan(tr[0], tr[1], color="tab:green", alpha=0.12, label="trained range")
    ax.set_title(label, fontsize=12)
    ax.set_xlabel(key); ax.set_ylabel("mean Dice"); ax.set_ylim(0, 0.85)
    ax.grid(alpha=0.3); ax.legend(loc="lower left", fontsize=9)

fig.suptitle("Geometry / context-pose extrapolation (mean Dice) — "
             "imagepfn_zoom / hard_diverse (anchor=0.647)", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.95])
out = os.path.join(D, "extrapolation_geometry.png")
fig.savefig(out, dpi=120, bbox_inches="tight")
print(f"Saved -> {out}")
