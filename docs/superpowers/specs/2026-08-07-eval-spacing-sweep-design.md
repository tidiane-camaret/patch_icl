# Eval spacing sweep — design

**Date:** 2026-08-07
**Status:** approved, pending implementation

## Goal

During benchmark eval, evaluate each sample at a *set* of physical crop spacings
(e.g. 1.5, 2.5, 3.5 mm) instead of one, to characterise how Dice varies with
spacing (FOV / resolution). Output per-(class, spacing) summaries.

## Context (how spacing flows today)

- `TotalSegInContextDataset` crops at a physical spacing. Under `use_crop=True`
  the output is always `T³` at `crop_spacing_mm`/voxel; the FOV = `T * crop_spacing_mm`.
- Per-item spacing override already exists: `__getitem__` accepts an `(idx, spacing)`
  index tuple and stores it in `_cur_crop_spacing`, which `_crop_mm` returns. One
  `__getitem__` call produces the target **and** its K contexts at that one spacing.
- `SpacingBatchSampler(sampler, batch_size, spacing_range, seed)` (experiments/3d/common.py)
  emits those tuples, one log-uniform spacing per batch. `[s, s]` → constant `s`.
- Eval is deterministic: `make_eval_loader` builds the dataset with `eval_seed`
  (per-item context shuffle + crop jitter seeded by `(eval_seed, idx)`) and
  `shuffle=False`. With `eval.crop_jitter=0` crops are centered. `idx` is stable
  across spacing passes → same task + same context subjects, only spacing varies.
- `evaluate_classes` (experiments/3d/evaluate.py) already writes `case["spacing"]`
  from `batch["spacing"][i,0]`, and `build_sample_table` already has a `spacing`
  column. The spacing-aware model already receives `spacing=batch["spacing"][0,0]`
  via `sp_kw`.

**Consequence:** model compute/time is identical across spacings (same `T³` shape);
this is a pure quality-vs-spacing curve.

## Decisions

- **Context spacing:** matched — target and its K contexts are cropped at the same
  `s` (what `SpacingBatchSampler` already does; matches training).
- **Output:** per-(class, spacing) CSV + table. (The wandb per-sample table already
  carries a `spacing` column, so per-sample detail comes for free.)
- **Scope:** benchmark `eval.py` only. `evaluate_classes` stays untouched
  (byte-identical; shared with train.py's val step). No change to train val.
- **Figures:** when `save_figures` is on, save the one-figure-per-class set only on
  the **first** spacing pass (avoids filename churn / overwrites).

## Design

### 1. Config surface

New optional key `cfg.eval.spacing_sweep`: a list of mm floats, e.g. `[1.5, 2.5, 3.5]`.
`null`/absent → current single-spacing behaviour, unchanged.

Guards (fail fast with a clear message when `spacing_sweep` is set):
- `data.use_crop` must be `true` (spacing override is a no-op otherwise — the resized
  path reports `_get_spacing`, ignoring `_cur_crop_spacing`).
- Source must be the totalseg path that `make_eval_loader` builds directly
  (`TotalSegInContextDataset`). Sources routed through `build_dataset`
  (omnisynth3d / anchor_synth3d / totalseg_more_labels) are unsupported → error.

Recommend (documented, not forced): set `eval.crop_jitter=0` for a fully controlled
sweep (centered crops; only spacing changes across passes).

### 2. Loader — constant-spacing pass

`make_eval_loader(cfg, classes, split, spacing: float | None = None)`:
- `spacing is None` → today's plain `DataLoader(ds, batch_size, shuffle=False, ...)`.
- `spacing = s` → `DataLoader(ds, batch_sampler=SpacingBatchSampler(
  SequentialSampler(ds), batch_size, [s, s]), ...)`.

`[s, s]` makes every batch that constant `s`. The `(idx, s)` tuples travel through
the sampler into `__getitem__` **inside worker processes**, so crops + reported
`spacing` follow. (Driving spacing via the sampler — not by mutating
`ds.crop_spacing_mm` — is required: attribute mutation in the main process does not
reach `num_workers>0` workers.) `SequentialSampler` preserves deterministic order.

Same DataLoader kwargs as the plain path (workers, pin_memory, persistent_workers,
prefetch_factor). Only totalseg (the direct-build branch) accepts `spacing`; the
`build_dataset`-routed sources ignore it (guarded out in §1 anyway).

### 3. Sweep driver

New `evaluate_spacing_sweep(model, cfg, classes, spacings, *, split=None, fig_dir=None)`
in `evaluate.py`:

```python
rows, cases = [], []
for i, s in enumerate(spacings):
    loader = make_eval_loader(cfg, classes, split=split, spacing=s)
    rows_s, cases_s = evaluate_classes(
        model, cfg, classes, loader=loader,
        fig_dir=fig_dir if i == 0 else None)   # figures: first spacing only
    for r in rows_s:
        r["spacing"] = s
    rows.extend(rows_s)
    cases.extend(cases_s)                        # cases already carry case["spacing"]=s
return rows, cases
```

`evaluate_classes` is called with a prebuilt `loader`, so it is unmodified. Rows
become per-(class, spacing); `cases` is the concatenation of all passes.

### 4. eval.py wiring & output

`main()`:
```python
sweep = cfg.eval.get("spacing_sweep")
if sweep:
    _assert_sweep_supported(cfg)                 # use_crop + totalseg-path guards (§1)
    rows, all_cases = evaluate_spacing_sweep(model, cfg, classes,
                                             list(sweep), fig_dir=fig_dir)
else:
    rows, all_cases = evaluate_classes(model, cfg, classes, fig_dir=fig_dir)
```

Output changes (all conditional on sweep, so the single-spacing output is unchanged):
- **eval.csv / eval.json**: add a `spacing` column; one summary row per (class, spacing).
- **Console**: existing per-row print gains the spacing; add a headline
  `spacing → mean_dice` summary block (aggregate curve).
- **wandb**: per-(class, spacing) scalars keyed `class/{cls}/mean_dice@{s}`
  (and `@{s}` variants for std/time); per-spacing `mean_dice@{s}`. The
  `build_sample_table(all_cases)` table is logged as-is — its `spacing` column
  already gives full per-(class, subject, spacing) detail.

## Cost / risk

- Wall time ≈ `len(spacings)` × current eval time. Same GFLOPs/sample at every spacing.
- No change to any single-spacing code path (train val, plain eval) — new behaviour
  is gated entirely behind `cfg.eval.spacing_sweep`.
- `SpacingBatchSampler` reused unchanged (its `spacing_range` param is generic).

## Out of scope

- Dice-vs-spacing plot generation (the CSV/table suffice; can add later).
- Sweeping in train.py's val step.
- Non-crop / non-totalseg sources.
