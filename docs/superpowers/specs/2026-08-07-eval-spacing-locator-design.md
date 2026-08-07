# Eval spacing locator — design

**Date:** 2026-08-07
**Status:** approved, pending implementation

## Goal

Layer a coarse→fine **localization** metric on top of the eval spacing sweep
(`docs/superpowers/specs/2026-08-07-eval-spacing-sweep-design.md`). For a
descending sweep such as `[4, 2]`, use the coarse-spacing (4 mm) target
prediction to position a fine-spacing (2 mm) window, then measure how much of
the target's ground truth falls inside that window. This quantifies whether the
coarse prediction is a good enough locator to drive a future two-stage
(coarse-locate → fine-apply) cascade — **without** running the model at the fine
spacing. It is a pure quality-of-localization measurement.

## Context (what already exists)

- **Spacing sweep:** `evaluate_spacing_sweep(model, cfg, classes, spacings, *, split, fig_dir)`
  (experiments/3d/evaluate.py) loops physical crop spacings, building one
  constant-spacing loader per `s` via `make_eval_loader(cfg, classes, split, spacing=s)`
  and calling the shared `evaluate_classes` with that loader. Gated behind
  `cfg.eval.spacing_sweep`; guarded to the totalseg + `use_crop=true` path by
  `_assert_sweep_supported(cfg)` (experiments/3d/eval.py).
- **Crop geometry:** under `use_crop=true` an item is a cube of `T³` voxels
  (`T = data.image_size` per axis) at `crop_spacing_mm`/voxel, so its physical
  field of view is `T · s` mm. Eval crops are deterministic and centered
  (`eval.crop_jitter=0`) — the target object sits near the crop center, and the
  same `idx` yields the same task + contexts at every spacing.
- **Per-sample tensors in `evaluate_classes`:** for each batch it computes the
  hard prediction `pred` (B,D,H,W) via `model.predict`, has `label` (B,D,H,W)
  the crop GT, and — only when a `logits_fn` is passed — a soft `prob`
  (B,1,D,H,W) = `_to_prob(logits, output_is_prob)`. It builds one `case` dict per
  sample and already records `case["spacing"]`. eval.py's benchmark path passes
  **no** `logits_fn`, so `prob` is absent there today.
- **Soft-prob interface:** train.py's val step obtains logits with
  `logits_fn=model.train_forward` and probabilities with
  `output_is_prob=model_output_is_prob(cfg)` (train.py:101). Reuse that same
  interface here rather than inventing a new one.

**Consequence:** the whole metric is computable inside the *coarse* pass from
`prob` + `label` + the ratio `s_f/s_c`. The fine spacing only sets the box size.
No fine-spacing model forward is required.

## Decisions

- **Scope:** containment metric only (no chained fine-spacing prediction).
- **Locator center:** probability-weighted centroid over **all** voxels,
  `c = Σ p·coord / Σ p`. No thresholds, no tunable knobs. `Σ p ≈ 0` (below a
  tiny numeric epsilon) → `locator_empty=True`, center = crop center.
- **Prob source:** `model.train_forward` via the existing `logits_fn` path
  (with `output_is_prob=model_output_is_prob(cfg)`). A model that lacks
  `train_forward` degrades to the hard-`pred`-mask centroid with a one-time
  warning.
- **Oracle baseline:** the same-size box centered on the GT foreground centroid
  — the best containment the fine FOV allows; report the `oracle − pred` gap.
- **Where it lives:** `evaluate_spacing_sweep` drives it; `evaluate_classes`
  gains one optional, default-`None` `locator_ratio` param that gates a small
  containment block. `locator_ratio=None` keeps `evaluate_classes` byte-identical
  (train.py's val step and the plain sweep are unaffected).
- **Enable flag:** `cfg.eval.spacing_locator` (bool, default `false`). It is a
  feature toggle, not a tuning parameter.

## Design

### 1. Config surface

New optional key `cfg.eval.spacing_locator` (bool, default `false`). When
`true`:
- `cfg.eval.spacing_sweep` must be set and contain at least one **descending**
  consecutive step (`spacings[i+1] < spacings[i]`); otherwise fail fast.
- All the existing sweep guards still apply (totalseg + `use_crop=true`).

`_assert_sweep_supported(cfg)` (experiments/3d/eval.py) gains a locator branch:
when `cfg.eval.get("spacing_locator")` is truthy it additionally requires a
non-empty `spacing_sweep` with a descending step, raising a `ValueError` that
names `spacing_locator` otherwise. When `spacing_locator` is falsy the guard is
unchanged.

`false`/absent → the sweep behaves exactly as today (no extra forward, no new
output).

### 2. Geometry & metric (per descending pair, inside the coarse pass)

Given a coarse pass at spacing `s_c` with a next-finer spacing `s_f < s_c`, and
`ratio = s_f / s_c ∈ (0, 1)`:

- **Box size** (per axis `a ∈ {D,H,W}`): `b_a = max(1, round(T_a · ratio))`
  voxels, where `T_a` is the coarse crop's extent on that axis. The fine box is a
  `b_D × b_H × b_W` sub-region of the coarse `T³` grid (it captures the physical
  cube `T·s_f` mm, i.e. the fine crop's FOV, expressed in coarse voxels).
- **Box placement:** center the box at the locator center `c` (voxel coords,
  §3). Its half-open bounds on axis `a` are `[lo_a, lo_a + b_a)` with
  `lo_a = round(c_a − b_a/2)`, then clamped so `0 ≤ lo_a` and `lo_a + b_a ≤ T_a`
  (shift the whole box inward when `c` is near an edge, preserving size `b_a`).
- **Containment (pred):** `containment = |GT_fg ∩ box| / |GT_fg|`, counting
  foreground voxels of the coarse-crop GT (`label > 0`). This is the fraction of
  the object's physical volume that a fine window placed by the locator would
  capture. `|GT_fg| == 0` (object absent from the crop) → containment is `NaN`
  and the sample is skipped from the pair's mean.
- **Localization error:** `loc_err_mm = ‖c − centroid(GT_fg)‖₂ · s_c` (distance
  from the locator center to the true object centroid, in mm). `NaN` when
  `|GT_fg| == 0`.

Containment is a physical-fraction quantity, so computing it on the coarse voxel
grid (rather than re-rendering GT at `s_f`) is faithful up to sub-voxel aliasing
of thin structures — negligible for a volume fraction.

### 3. Locator center

From the coarse-pass probability volume `p` (shape `T³`, values in `[0,1]`):

```
s = p.sum()
if s < EPS:                      # no signal anywhere
    center = (T_D/2, T_H/2, T_W/2)   # blind crop center
    locator_empty = True
else:
    center = Σ_voxel p·coord / s      # prob-weighted centroid, all voxels
    locator_empty = False
```

`coord` is the voxel index vector; the sum runs over every voxel (no gate).
`EPS` is a fixed numeric guard (e.g. `1e-6`), not a user knob.

`p` comes from the soft `prob` that `evaluate_classes` computes when
`logits_fn=model.train_forward` and `output_is_prob=model_output_is_prob(cfg)`
are supplied by the sweep. If the eval model has no `train_forward`, the sweep
passes `logits_fn=None`; `evaluate_classes` then has no `prob`, so the locator
falls back to the **hard predicted mask** centroid (`p := pred`, a 0/1 volume),
and `evaluate_spacing_sweep` prints a one-time warning that soft centroids are
unavailable for this model.

### 4. Oracle baseline

Recompute the box centered on `centroid(GT_fg)` (same size `b`, same clamping)
and report `containment_oracle = |GT_fg ∩ box_oracle| / |GT_fg|`. This is the
maximum containment achievable at this FOV with a perfect locator, so:

- `containment` low **and** `oracle` low → the fine FOV is simply too small for
  this object (a single zoom cannot capture it).
- `containment` low but `oracle` high → the locator misplaced the box.

The per-(class, pair) summary reports both and their gap `oracle − pred`.

### 5. Data flow / code changes

**`evaluate_classes` (experiments/3d/evaluate.py)** — one new optional param:

```
def evaluate_classes(model, cfg, classes, *, split=None, fig_dir=None,
                     loader=None, logits_fn=None, loss_fn=None, grid_res=None,
                     output_is_prob=False, autocast=False, reuse_logits=False,
                     locator_ratio: float | None = None):
```

When `locator_ratio is not None`, in the per-sample loop (where `pred`, `label`,
and — if `logits_fn` was given — `prob` are already in scope) compute the §2/§3
quantities from `prob[i,0]` (or `pred[i]` when `prob is None`) and `label[i]`,
and store into the case:

```
case["containment"]        = round(float(containment), 4)   # NaN if GT empty
case["containment_oracle"] = round(float(containment_oracle), 4)
case["locator_empty"]      = bool(locator_empty)
case["loc_err_mm"]         = round(float(loc_err_mm), 2)     # NaN if GT empty
```

`locator_ratio is None` (every caller today: train.py val step, plain sweep,
plain eval) → the block is skipped and behaviour is byte-identical.

A small module-level helper does the geometry so the loop stays readable:

```
def _locator_containment(prob, label, ratio):
    """Return (containment, containment_oracle, locator_empty, loc_err_mm).

    prob:  (D,H,W) soft probability or 0/1 hard mask (locator weights)
    label: (D,H,W) GT; foreground = label > 0
    ratio: s_fine / s_coarse in (0,1) -> box side = round(T*ratio) per axis
    NaNs for containment/loc_err_mm when GT has no foreground.
    """
```

**`evaluate_spacing_sweep` (experiments/3d/evaluate.py)** — when the locator is
on, drive the soft-prob path and the per-pass ratio:

```
def evaluate_spacing_sweep(model, cfg, classes, spacings, *, split=None,
                           fig_dir=None, locator=False):
    from common import make_eval_loader
    lf = op = None
    if locator:
        lf = getattr(model, "train_forward", None)
        if lf is None:
            print("  [warn] model has no train_forward; locator uses the hard "
                  "predicted mask centroid (no soft prob).")
        else:
            from train import model_output_is_prob
            op = model_output_is_prob(cfg)
    rows, cases = [], []
    for i, s in enumerate(spacings):
        # locator toward the next-finer spacing, if any
        ratio = None
        if locator and i + 1 < len(spacings) and spacings[i + 1] < s:
            ratio = spacings[i + 1] / s
        loader = make_eval_loader(cfg, classes, split=split or cfg.eval.split, spacing=s)
        rows_s, cases_s = evaluate_classes(
            model, cfg, classes, loader=loader,
            fig_dir=fig_dir if i == 0 else None,
            logits_fn=(lf if ratio is not None else None),
            output_is_prob=(op or False),
            locator_ratio=ratio)
        for r in rows_s:
            r["spacing"] = s
            if ratio is not None:
                r["locator_to"] = spacings[i + 1]
        rows.extend(rows_s)
        cases.extend(cases_s)
    return rows, cases
```

`logits_fn` is passed only on passes that actually run the locator (`ratio is
not None`), so non-locator passes (e.g. the finest spacing) keep the single
`predict` forward. The last spacing has no finer successor → no locator, no
extra forward.

**`_summarize` (experiments/3d/evaluate.py)** — extend the per-class row
aggregation so that, when cases carry `containment`, the row also gets
`mean_containment`, `mean_containment_oracle`, `mean_loc_err_mm` (nan-mean over
cases with non-NaN containment), `n_locator` (count of such cases), and
`n_locator_empty`. Cases without those keys (non-locator runs) leave the row
unchanged.

**`eval.py` (experiments/3d/eval.py)** — thread the flag and extend output:

```
sweep   = cfg.eval.get("spacing_sweep")
locator = bool(cfg.eval.get("spacing_locator"))
if sweep:
    _assert_sweep_supported(cfg)                 # now also validates locator
    spacings = list(sweep)
    rows, all_cases = evaluate_spacing_sweep(model, cfg, classes, spacings,
                                             fig_dir=fig_dir, locator=locator)
else:
    rows, all_cases = evaluate_classes(model, cfg, classes, fig_dir=fig_dir)
```

Output (all conditional on `locator`, so non-locator output is unchanged):
- **Console:** per-row print appends `cont=<pred> (orc=<oracle>)` when the row has
  `mean_containment`; add a headline block
  `pair s_c->s_f : mean_containment  (oracle <o>, gap <g>, n=<n>, empty=<e>)`.
- **eval.csv / eval.json:** rows already carry the new keys, so `eval.json` gets
  them free; add `mean_containment,mean_containment_oracle,mean_loc_err_mm,locator_to`
  columns to `eval.csv` only when `locator`.
- **wandb:** per-(class, pair) scalars `class/{cls}/containment@{s_c}` and
  `.../containment_oracle@{s_c}`; the per-sample `build_sample_table(all_cases)`
  already logs the new case columns.

### 6. Config docs

Document `eval.spacing_locator: false` in `configs/experiment/3d/eval.yaml`,
next to `spacing_sweep`, explaining: coarse→fine localization metric; needs a
descending `spacing_sweep`; adds one extra (soft-prob) forward per non-final
spacing; totalseg + `use_crop` only. Log the feature in `docs/logs.md`.

## Cost / risk

- Extra compute: one additional `train_forward` per sample on every non-final
  spacing pass (only when `spacing_locator=true`). No extra forward otherwise.
- No change to any non-locator path: train.py val step, plain eval, and the
  plain spacing sweep are byte-identical (`locator_ratio` defaults to `None`,
  `spacing_locator` defaults to `false`).
- Containment on the coarse grid is a faithful physical-fraction proxy for the
  fine-spacing GT (sub-voxel aliasing only).

## Out of scope

- Chaining the model at the fine spacing inside the located box (full two-stage
  cascade) and reporting fine Dice.
- Rendering the GT at the fine spacing (the coarse-grid physical fraction
  suffices).
- Any tunable locator parameters (thresholds, temperature) — plain all-voxel
  soft centroid only.
- Non-descending sweep steps (`s_f ≥ s_c`) — no locator computed for them.
