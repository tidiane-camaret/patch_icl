"""Aggregate the per-axis OOD sweep CSVs into per-axis Dice curves + a summary plot.

Each CSV is one operating point (one knob value, rest in-distribution). We read the
mean dice_native (+ realized ctx_dice/fg_frac) per file and group by axis. The
in-distribution training value of each axis is marked; anchor.csv is the shared
in-dist point for every LIVE axis.
"""
import glob, re, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = os.path.dirname(__file__)

# axis -> (training value, list of swept values come from filenames)
TRAIN = {
    "noise": 0.40, "contrast": 0.50, "texture": 0.35, "shift": 0.50,
    "scale": 0.45, "translate": 0.05, "consistency": 0.90, "ambint": 0.60,
    "regionsize": None, "taskamb": None,   # build axes: no single train value (drawn from a range)
}
TRAIN_RANGE = {"regionsize": (0.12, 0.62), "taskamb": (0.30, 0.80)}
LIVE_AXES = {"noise", "contrast", "texture", "shift", "scale", "translate", "consistency", "ambint"}
LABEL = {
    "noise": "noise_level", "contrast": "foreground_contrast",
    "texture": "texture_heterogeneity", "shift": "support_query_shift",
    "scale": "support_query_scale", "translate": "support_query_translate",
    "consistency": "context_consistency", "ambint": "task_ambiguity_intensity",
    "regionsize": "region_size", "taskamb": "task_ambiguity (build)",
}

def stat(path):
    d = pd.read_csv(path)
    return dict(dice=d.dice_native.mean(), dice_med=d.dice_native.median(),
                fail=(d.dice_native < 0.1).mean(), ctx=d.ctx_dice.mean(),
                fg=d.fg_frac.mean(), n=len(d))

# anchor (shared in-dist point for live axes)
anchor = stat(os.path.join(D, "anchor.csv"))

rows = []
for f in sorted(glob.glob(os.path.join(D, "*.csv"))):
    base = os.path.basename(f)[:-4]
    m = re.match(r"([a-z]+)_([0-9.]+)$", base)
    if not m:
        continue
    axis, val = m.group(1), float(m.group(2))
    s = stat(f)
    rows.append(dict(axis=axis, val=val, **s))

df = pd.DataFrame(rows)

# add the shared anchor as the in-dist point of each live axis (at its train value)
for ax in LIVE_AXES:
    if ax in df.axis.values:
        df = pd.concat([df, pd.DataFrame([dict(axis=ax, val=TRAIN[ax], **anchor)])],
                       ignore_index=True)
df = df.sort_values(["axis", "val"]).reset_index(drop=True)

# ── text report ──
print(f"Shared in-dist anchor (hard_diverse @ num_tasks=2000): "
      f"dice={anchor['dice']:.3f}  fail={anchor['fail']*100:.0f}%  ctx_dice={anchor['ctx']:.3f}  n={anchor['n']}\n")
order = ["noise", "contrast", "texture", "shift", "scale", "translate",
         "consistency", "ambint", "regionsize", "taskamb"]
for ax in order:
    sub = df[df.axis == ax]
    if sub.empty:
        continue
    tv = TRAIN.get(ax)
    tr = TRAIN_RANGE.get(ax)
    tag = f"train={tv}" if tv is not None else f"train-range={tr}"
    print(f"=== {LABEL[ax]:26s} ({tag}) ===")
    print(f"  {'value':>7} {'dice':>6} {'median':>7} {'fail%':>6} {'ctx_dice':>9}")
    for _, r in sub.iterrows():
        star = ""
        if tv is not None and abs(r.val - tv) < 1e-6:
            star = "  <- train"
        elif tr is not None and tr[0] - 1e-6 <= r.val <= tr[1] + 1e-6:
            star = "  (in train range)"
        print(f"  {r.val:7.2f} {r.dice:6.3f} {r.dice_med:7.3f} {r.fail*100:6.0f} {r.ctx:9.3f}{star}")
    print()

# ── plot: one panel per axis, dice vs knob value ──
fig, axes = plt.subplots(2, 5, figsize=(22, 8), squeeze=False)
for i, ax in enumerate(order):
    a = axes[i // 5][i % 5]
    sub = df[df.axis == ax]
    if sub.empty:
        a.axis("off"); continue
    a.plot(sub.val, sub.dice, "o-", color="tab:blue", label="dice_native")
    a.plot(sub.val, sub.ctx, "s--", color="tab:gray", alpha=0.6, label="ctx_dice (info)")
    tv = TRAIN.get(ax)
    if tv is not None:
        a.axvline(tv, color="tab:green", ls=":", lw=2, label="train value")
    elif ax in TRAIN_RANGE:
        lo, hi = TRAIN_RANGE[ax]
        a.axvspan(lo, hi, color="tab:green", alpha=0.12, label="train range")
    a.axhline(anchor["dice"], color="tab:orange", ls=":", lw=1, alpha=0.7)
    a.set_title(LABEL[ax]); a.set_xlabel("knob value"); a.set_ylabel("mean Dice")
    a.set_ylim(0, 0.8); a.grid(alpha=0.3)
    if i == 0:
        a.legend(fontsize=7, loc="lower center")
fig.suptitle("Per-axis OOD sweep — imagepfn_zoom trained on hard_diverse "
             f"(in-dist anchor dice={anchor['dice']:.3f}, dotted orange)", fontsize=13)
fig.tight_layout()
out = os.path.join(D, "sweep_curves.png")
fig.savefig(out, dpi=110, bbox_inches="tight")
print(f"Saved plot -> {out}")

# also dump the aggregate table
df.to_csv(os.path.join(D, "sweep_summary.csv"), index=False)
print(f"Saved table -> {os.path.join(D, 'sweep_summary.csv')}")
