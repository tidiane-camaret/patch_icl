# Eval Spacing Locator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a coarse→fine localization metric to the eval spacing sweep: use each coarse-spacing target prediction to place a fine-spacing window and measure how much of the target GT it contains, gated behind `cfg.eval.spacing_locator`.

**Architecture:** A pure geometry helper `_locator_containment` computes, from a coarse prediction volume + GT + the fine/coarse spacing ratio, the containment fraction of a prob-weighted-centroid-placed box, an oracle (GT-centroid) box, and the localization error. `evaluate_classes` gains a default-`None` `locator_ratio` param that gates a per-sample block calling the helper; `evaluate_spacing_sweep` drives it with `model.train_forward` (soft prob) toward the next-finer spacing; `_summarize`, the guard, `eval.py`, and config carry the results through.

**Tech Stack:** PyTorch tensors, NumPy, Hydra/OmegaConf, pytest, Python 3.10+ (`float | None`).

## Global Constraints

- Spec: `docs/superpowers/specs/2026-08-07-eval-spacing-locator-design.md` (approved).
- `cfg.eval.spacing_locator` defaults `false`; `locator_ratio` defaults `None`. With both off, `evaluate_classes`, `evaluate_spacing_sweep`, `_summarize`, the guard, and `eval.py` output MUST be byte-identical to today. This is the overriding constraint.
- Locator center = probability-weighted centroid over **all** voxels (`Σ p·coord / Σ p`); `Σ p < 1e-6` → `locator_empty=True`, center = crop center. No thresholds, no tunable knobs.
- Prob source = `model.train_forward` via the existing `logits_fn`/`output_is_prob` path (same as train.py's val step). Model lacking `train_forward` → hard-`pred`-mask centroid + a one-time warning (do NOT error).
- Box side per axis = `max(1, round(T_a · ratio))`, `ratio = s_fine/s_coarse ∈ (0,1)`; box clamped fully inside `[0, T_a]`.
- Containment = `|GT_fg ∩ box| / |GT_fg|`, `GT_fg = label > 0`; NaN when `|GT_fg| == 0` (skip from means). The helper returns localization error in **voxels**; the caller multiplies by the coarse spacing to get `loc_err_mm`.
- Locator only for consecutive **descending** sweep steps (`spacings[i+1] < spacings[i]`); the finest spacing has no successor → no locator, no extra forward.
- Project rule (CLAUDE.md): "Write tests only when necessary." The pure geometry helper and the guard get pytests; wiring is verified with inspection snippets. Log changes to `docs/logs.md`.
- `git` is available. End commit messages with `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.

---

### Task 1: `_locator_containment` geometry helper

**Files:**
- Modify: `experiments/3d/evaluate.py` (add module-level helper after `_summarize`, which ends at line 277)
- Test: `experiments/3d/tests/test_locator.py`

**Interfaces:**
- Produces: `_locator_containment(prob, label, ratio) -> (containment, containment_oracle, locator_empty, loc_err_vox)`. `prob`/`label` are `(D,H,W)` tensors; `ratio ∈ (0,1)`. Returns two floats (NaN when `label>0` is empty), a bool, and the centroid-to-GT-centroid distance in voxels (NaN when GT empty).

- [ ] **Step 1: Write the failing test**

Create `experiments/3d/tests/test_locator.py`:

```python
"""Geometry unit tests for the coarse->fine locator containment helper."""
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # sibling evaluate.py

from evaluate import _locator_containment  # noqa: E402


def _cube(T, sl):
    v = torch.zeros((T, T, T))
    v[sl, sl, sl] = 1.0
    return v


def test_perfect_locator_object_fits():
    # T=8, ratio 0.5 -> box side 4 centered; object 2^3 at center fits entirely.
    obj = _cube(8, slice(3, 5))
    cont, orc, empty, err = _locator_containment(obj, obj, 0.5)
    assert cont == 1.0 and orc == 1.0
    assert empty is False
    assert err < 1e-6


def test_object_larger_than_box():
    # GT fills all 8^3=512; centered box 4^3=64 -> containment 64/512 = 0.125.
    gt = torch.ones((8, 8, 8))
    cont, orc, empty, err = _locator_containment(gt, gt, 0.5)
    assert abs(cont - 0.125) < 1e-6
    assert abs(orc - 0.125) < 1e-6
    assert empty is False


def test_empty_prob_falls_back_to_center():
    prob = torch.zeros((8, 8, 8))
    gt = _cube(8, slice(3, 5))
    cont, orc, empty, err = _locator_containment(prob, gt, 0.5)
    assert empty is True
    # Center fallback = crop center, which coincides with the centered object -> full.
    assert cont == 1.0


def test_offset_prediction_low_containment_high_oracle():
    # GT in one corner, prediction blob in the opposite corner.
    gt = _cube(8, slice(0, 2))
    prob = _cube(8, slice(6, 8))
    cont, orc, empty, err = _locator_containment(prob, gt, 0.5)
    assert cont == 0.0            # pred box misses the GT corner
    assert orc == 1.0            # oracle box centered on GT captures it
    assert err > 5.0             # centroids ~6 voxels apart per axis


def test_gt_empty_returns_nan():
    prob = _cube(8, slice(3, 5))
    gt = torch.zeros((8, 8, 8))
    cont, orc, empty, err = _locator_containment(prob, gt, 0.5)
    assert math.isnan(cont) and math.isnan(orc) and math.isnan(err)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd experiments/3d && python -m pytest tests/test_locator.py -v`
Expected: FAIL — `ImportError: cannot import name '_locator_containment' from 'evaluate'`.

- [ ] **Step 3: Implement the helper**

In `experiments/3d/evaluate.py`, add after `_summarize` (after line 277, before the blank line preceding `evaluate_spacing_sweep`). `np` is already imported at module top:

```python
def _locator_containment(prob, label, ratio):
    """Coarse->fine locator containment for one sample (pure geometry).

    prob : (D,H,W) tensor — soft probability or 0/1 hard mask (locator weights).
    label: (D,H,W) tensor — GT; foreground = label > 0.
    ratio: s_fine / s_coarse in (0,1). Box side per axis = max(1, round(T_a*ratio)).

    Locator center = prob-weighted centroid over ALL voxels; sum(prob) < 1e-6 -> crop
    center + locator_empty=True. The fine box (that side, clamped inside the volume) is
    placed at the locator center; the oracle box is placed at the GT-foreground centroid.
    Returns (containment, containment_oracle, locator_empty, loc_err_vox):
      containment        = |GT_fg ∩ box|        / |GT_fg|   (NaN if no GT foreground)
      containment_oracle = |GT_fg ∩ box_oracle| / |GT_fg|   (NaN if no GT foreground)
      locator_empty      = bool
      loc_err_vox        = ||center - gt_centroid|| in voxels (NaN if no GT foreground).
                           The caller scales by the coarse spacing to get loc_err_mm.
    """
    p = prob.detach().float().cpu().numpy()
    gt = (label.detach().cpu().numpy() > 0)
    T = p.shape                                    # (D, H, W)
    box = [max(1, int(round(t * ratio))) for t in T]
    idx = np.indices(T, dtype=float)               # (3, D, H, W)

    def _frac_in_box(center):
        total = float(gt.sum())
        lo = []
        for a in range(3):
            l = int(round(center[a] - box[a] / 2))
            l = max(0, min(l, T[a] - box[a]))       # clamp so the box fits in [0, T_a]
            lo.append(l)
        sub = gt[lo[0]:lo[0] + box[0], lo[1]:lo[1] + box[1], lo[2]:lo[2] + box[2]]
        return float(sub.sum()) / total

    # Locator center: prob-weighted centroid over all voxels; empty -> crop center.
    s = float(p.sum())
    if s < 1e-6:
        center = np.array([t / 2.0 for t in T])
        locator_empty = True
    else:
        center = np.array([(idx[a] * p).sum() / s for a in range(3)])
        locator_empty = False

    gt_n = float(gt.sum())
    if gt_n == 0.0:
        return float("nan"), float("nan"), locator_empty, float("nan")

    gt_centroid = np.array([(idx[a] * gt).sum() / gt_n for a in range(3)])
    containment = _frac_in_box(center)
    containment_oracle = _frac_in_box(gt_centroid)
    loc_err_vox = float(np.linalg.norm(center - gt_centroid))
    return containment, containment_oracle, locator_empty, loc_err_vox
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd experiments/3d && python -m pytest tests/test_locator.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add experiments/3d/evaluate.py experiments/3d/tests/test_locator.py
git commit -m "feat(eval): add locator containment geometry helper"
```

---

### Task 2: gate the locator into `evaluate_classes` + `_summarize`

**Files:**
- Modify: `experiments/3d/evaluate.py:280-282` (signature), inner loop after line 406, and `_summarize` (250-277)

**Interfaces:**
- Consumes: `_locator_containment` (Task 1).
- Produces: `evaluate_classes(..., locator_ratio: float | None = None)`. When set, each `case` gains `containment`, `containment_oracle` (floats, may be NaN), `locator_empty` (bool), `loc_err_mm` (float, may be NaN). `_summarize` then adds `n_locator`, `n_locator_empty`, `mean_containment`, `mean_containment_oracle`, `mean_loc_err_mm` to the class row (nan-mean over non-NaN cases). All absent when `locator_ratio is None`.

- [ ] **Step 1: Add the `locator_ratio` parameter**

In `experiments/3d/evaluate.py`, change the signature (currently lines 280-282):

```python
def evaluate_classes(model, cfg, classes, *, split=None, fig_dir: Path | None = None,
                     loader=None, logits_fn=None, loss_fn=None, grid_res=None,
                     output_is_prob=False, autocast=False, reuse_logits=False):
```

to add the new last keyword param:

```python
def evaluate_classes(model, cfg, classes, *, split=None, fig_dir: Path | None = None,
                     loader=None, logits_fn=None, loss_fn=None, grid_res=None,
                     output_is_prob=False, autocast=False, reuse_logits=False,
                     locator_ratio: float | None = None):
```

- [ ] **Step 2: Add the gated per-sample locator block**

In the per-sample loop, immediately AFTER the spacing block (evaluate.py:405-406):

```python
            if "spacing" in batch:
                case["spacing"] = round(float(batch["spacing"][i, 0]), 4)
```

insert:

```python
            if locator_ratio is not None:
                # Locate a fine-spacing box from the coarse prediction and measure how much
                # GT it contains. Soft prob when available (logits_fn), else the hard mask.
                lp = prob[i, 0] if prob is not None else pred[i]
                cont, cont_orc, loc_empty, loc_err_vox = _locator_containment(
                    lp, label[i], locator_ratio)
                sp_c = float(batch["spacing"][i, 0]) if "spacing" in batch else 1.0
                case["containment"] = round(float(cont), 4)              # NaN safe: round(nan)=nan
                case["containment_oracle"] = round(float(cont_orc), 4)
                case["locator_empty"] = bool(loc_empty)
                case["loc_err_mm"] = round(loc_err_vox * sp_c, 2)
```

- [ ] **Step 3: Extend `_summarize`**

In `_summarize` (evaluate.py:250-277), immediately BEFORE the final `return row` (line 277), add:

```python
    # Locator containment (only when evaluate_classes ran with locator_ratio set).
    if any("containment" in c for c in cases):
        cont = [c["containment"] for c in cases
                if "containment" in c and not np.isnan(c["containment"])]
        orc = [c["containment_oracle"] for c in cases
               if "containment_oracle" in c and not np.isnan(c["containment_oracle"])]
        err = [c["loc_err_mm"] for c in cases
               if "loc_err_mm" in c and not np.isnan(c["loc_err_mm"])]
        row["n_locator"] = len(cont)
        row["n_locator_empty"] = sum(1 for c in cases if c.get("locator_empty"))
        row["mean_containment"] = round(sum(cont) / len(cont), 4) if cont else float("nan")
        row["mean_containment_oracle"] = round(sum(orc) / len(orc), 4) if orc else float("nan")
        row["mean_loc_err_mm"] = round(sum(err) / len(err), 2) if err else float("nan")
```

- [ ] **Step 4: Verify the gating and byte-identical default by inspection**

Run: `python -c "s=open('experiments/3d/evaluate.py').read(); assert 'locator_ratio: float | None = None' in s; assert 'if locator_ratio is not None:' in s; assert s.count('_locator_containment(') >= 1; assert 'if any(\"containment\" in c for c in cases):' in s; import ast; ast.parse(s); print('ok')"`
Expected: prints `ok` (param added, gated block + summary present, file parses).

- [ ] **Step 5: Run the existing locator + guard tests to confirm nothing regressed**

Run: `cd experiments/3d && python -m pytest tests/test_locator.py tests/test_sweep_guard.py -q`
Expected: PASS (all tests; the helper is unchanged and the new code is import-only at rest).

- [ ] **Step 6: Commit**

```bash
git add experiments/3d/evaluate.py
git commit -m "feat(eval): compute locator containment per sample under locator_ratio"
```

---

### Task 3: drive the locator from `evaluate_spacing_sweep`

**Files:**
- Modify: `experiments/3d/evaluate.py:427-448` (`evaluate_spacing_sweep`)

**Interfaces:**
- Consumes: `evaluate_classes(..., logits_fn, output_is_prob, locator_ratio)` (Task 2); `train.model_output_is_prob(cfg)`.
- Produces: `evaluate_spacing_sweep(model, cfg, classes, spacings, *, split=None, fig_dir=None, locator=False)`. When `locator=True`, coarse passes with a next-finer spacing run with `logits_fn=model.train_forward` and `locator_ratio=spacings[i+1]/spacings[i]`, and their rows also get `r["locator_to"] = spacings[i+1]`.

- [ ] **Step 1: Rewrite the driver with the locator path**

In `experiments/3d/evaluate.py`, replace the whole `evaluate_spacing_sweep` function (lines 427-448) with:

```python
def evaluate_spacing_sweep(model, cfg, classes, spacings, *, split=None, fig_dir=None,
                           locator=False):
    """Run evaluate_classes once per physical crop spacing; tag rows with their spacing.

    Builds a constant-spacing eval loader per `s` (make_eval_loader(..., spacing=s)) and
    calls the shared evaluate_classes with that prebuilt loader. `idx` is stable across
    passes, so each spacing sees the same task + context subjects — only the crop spacing
    changes. Figures are saved on the first spacing only (later passes reuse the filenames).

    When locator=True, each coarse pass that has a next-finer spacing also runs the
    coarse->fine localization metric (see _locator_containment): it forwards a soft
    probability via model.train_forward and passes locator_ratio = s_fine/s_coarse so
    evaluate_classes records per-sample containment. A model without train_forward falls
    back to the hard predicted mask centroid (a one-time warning). The finest spacing has
    no successor, so it runs the plain single-predict path with no extra forward.

    Returns (rows, cases): rows are per-(class, spacing); cases are all passes concatenated.
    """
    from common import make_eval_loader  # local import: common/evaluate are siblings

    lf = op = None
    if locator:
        lf = getattr(model, "train_forward", None)
        if lf is None:
            print("  [warn] model has no train_forward; locator uses the hard predicted "
                  "mask centroid (no soft prob).")
        else:
            from train import model_output_is_prob  # local import: sibling module
            op = model_output_is_prob(cfg)

    rows, cases = [], []
    for i, s in enumerate(spacings):
        ratio = None
        if locator and i + 1 < len(spacings) and spacings[i + 1] < s:
            ratio = spacings[i + 1] / s
        loader = make_eval_loader(cfg, classes, split=split or cfg.eval.split, spacing=s)
        rows_s, cases_s = evaluate_classes(
            model, cfg, classes, loader=loader,
            fig_dir=fig_dir if i == 0 else None,
            logits_fn=(lf if ratio is not None else None),
            output_is_prob=bool(op),
            locator_ratio=ratio)
        for r in rows_s:
            r["spacing"] = s
            if ratio is not None:
                r["locator_to"] = spacings[i + 1]
        rows.extend(rows_s)
        cases.extend(cases_s)
    return rows, cases
```

- [ ] **Step 2: Verify the driver by inspection**

Run: `cd experiments/3d && python -c "import inspect, evaluate; p=list(inspect.signature(evaluate.evaluate_spacing_sweep).parameters); print(p); assert p[-1]=='locator'"`
Expected: prints the params ending in `'locator'`.

- [ ] **Step 3: Verify locator wiring in source**

Run: `python -c "s=open('experiments/3d/evaluate.py').read().split('def evaluate_spacing_sweep')[1]; assert 'getattr(model, \"train_forward\", None)' in s; assert 'spacings[i + 1] / s' in s; assert 'locator_ratio=ratio' in s; assert 'r[\"locator_to\"] = spacings[i + 1]' in s; print('ok')"`
Expected: prints `ok`.

- [ ] **Step 4: Confirm the non-locator path is unchanged**

Run: `python -c "s=open('experiments/3d/evaluate.py').read().split('def evaluate_spacing_sweep')[1]; assert 'logits_fn=(lf if ratio is not None else None)' in s; print('locator=False -> ratio None -> logits_fn None, plain sweep intact')"`
Expected: prints the confirmation (with `locator=False`, `ratio` stays `None`, so `logits_fn=None` and `locator_ratio=None` → byte-identical to the plain sweep).

- [ ] **Step 5: Commit**

```bash
git add experiments/3d/evaluate.py
git commit -m "feat(eval): drive coarse->fine locator from evaluate_spacing_sweep"
```

---

### Task 4: locator guard + test

**Files:**
- Modify: `experiments/3d/eval.py:75-89` (`_assert_sweep_supported`)
- Modify: `experiments/3d/tests/test_sweep_guard.py`

**Interfaces:**
- Consumes: `cfg.eval.spacing_locator`, `cfg.eval.spacing_sweep`.
- Produces: `_assert_sweep_supported(cfg)` additionally raises `ValueError` (mentioning `spacing_locator`) when `spacing_locator` is truthy but `spacing_sweep` has fewer than 2 entries or no descending consecutive step. Unchanged when `spacing_locator` is falsy.

- [ ] **Step 1: Write the failing tests**

In `experiments/3d/tests/test_sweep_guard.py`, update the `_cfg` helper and add locator tests. Replace the existing `_cfg` function:

```python
def _cfg(source="totalseg", use_crop=True):
    return OmegaConf.create({"data": {"source": source, "use_crop": use_crop}})
```

with a version that can carry an `eval` block, then add three tests at the end of the file:

```python
def _cfg(source="totalseg", use_crop=True, **eval_kw):
    return OmegaConf.create(
        {"data": {"source": source, "use_crop": use_crop}, "eval": dict(eval_kw)})


def test_locator_with_descending_sweep_ok():
    _assert_sweep_supported(_cfg(spacing_locator=True, spacing_sweep=[4, 2]))  # no raise


def test_locator_without_descending_step_rejected():
    with pytest.raises(ValueError, match="spacing_locator"):
        _assert_sweep_supported(_cfg(spacing_locator=True, spacing_sweep=[2, 4]))


def test_locator_single_spacing_rejected():
    with pytest.raises(ValueError, match="spacing_locator"):
        _assert_sweep_supported(_cfg(spacing_locator=True, spacing_sweep=[2]))
```

- [ ] **Step 2: Run the tests to verify the new ones fail**

Run: `cd experiments/3d && python -m pytest tests/test_sweep_guard.py -v`
Expected: the three `test_locator_*` tests FAIL (guard does not yet check the locator); the pre-existing sweep tests still PASS.

- [ ] **Step 3: Extend the guard**

In `experiments/3d/eval.py`, at the END of `_assert_sweep_supported` (after the source check that ends at line 89), add:

```python
    if cfg.eval.get("spacing_locator"):
        sweep = cfg.eval.get("spacing_sweep")
        sl = list(sweep) if sweep else []
        if len(sl) < 2 or not any(sl[i + 1] < sl[i] for i in range(len(sl) - 1)):
            raise ValueError(
                f"eval.spacing_locator requires eval.spacing_sweep with at least one "
                f"descending step (e.g. [4, 2]); got {sl!r}.")
```

- [ ] **Step 4: Run the tests to verify all pass**

Run: `cd experiments/3d && python -m pytest tests/test_sweep_guard.py -v`
Expected: PASS (the original sweep tests + the 3 new locator tests).

- [ ] **Step 5: Commit**

```bash
git add experiments/3d/eval.py experiments/3d/tests/test_sweep_guard.py
git commit -m "feat(eval): guard spacing_locator to a descending sweep"
```

---

### Task 5: `eval.py` wiring & output

**Files:**
- Modify: `experiments/3d/eval.py:229-235` (branch), `:242-256` (per-row print + wandb), `:267-276` (headline), CSV block (`:282-287`)

**Interfaces:**
- Consumes: `evaluate_spacing_sweep(..., locator=...)` (Task 3); rows carrying `mean_containment`, `mean_containment_oracle`, `mean_loc_err_mm`, `locator_to`.
- Produces: when `cfg.eval.spacing_locator` is set, console/CSV/wandb gain locator columns; otherwise output is unchanged.

- [ ] **Step 1: Read the locator flag and pass it to the sweep**

In `experiments/3d/eval.py`, replace the sweep branch (lines 229-235):

```python
    sweep = cfg.eval.get("spacing_sweep")
    if sweep:
        _assert_sweep_supported(cfg)
        spacings = list(sweep)
        print(f"  Spacing sweep: {spacings} mm  ({len(spacings)}x eval time)\n")
        rows, all_cases = evaluate_spacing_sweep(model, cfg, classes,
                                                 spacings, fig_dir=fig_dir)
```

with:

```python
    sweep = cfg.eval.get("spacing_sweep")
    locator = bool(cfg.eval.get("spacing_locator"))
    if sweep:
        _assert_sweep_supported(cfg)
        spacings = list(sweep)
        tag = "  (+ coarse->fine locator)" if locator else ""
        print(f"  Spacing sweep: {spacings} mm  ({len(spacings)}x eval time){tag}\n")
        rows, all_cases = evaluate_spacing_sweep(model, cfg, classes, spacings,
                                                 fig_dir=fig_dir, locator=locator)
```

- [ ] **Step 2: Append containment to the per-row print + wandb**

In `experiments/3d/eval.py`, replace the per-row print/log block (lines 250-256) — from `row["gflops"] = round(gflops, 2)` through the `wandb.log({...})` call — with:

```python
        row["gflops"] = round(gflops, 2)
        cont_str = (f"  cont={row['mean_containment']:.3f} (orc={row['mean_containment_oracle']:.3f})"
                    if "mean_containment" in row else "")
        print(f"  {cls:<35s}{sp_str}  dice={row['mean_dice']:.3f} ± {row['std_dice']:.3f}"
              f"  {row['mean_time_ms']:.0f}ms/sample  n={row['n_samples']}{cont_str}")
        if wb_on:
            wandb.log({f"class/{cls}/mean_dice{sp_key}": row["mean_dice"],
                       f"class/{cls}/std_dice{sp_key}": row["std_dice"],
                       f"class/{cls}/mean_time_ms{sp_key}": row["mean_time_ms"]})
            if "mean_containment" in row:
                wandb.log({f"class/{cls}/containment{sp_key}": row["mean_containment"],
                           f"class/{cls}/containment_oracle{sp_key}": row["mean_containment_oracle"]})
```

- [ ] **Step 3: Add the locator headline block**

In `experiments/3d/eval.py`, immediately AFTER the existing `spacing -> mean_dice` block (after line 276, still inside `if valid:`), add:

```python
        if locator:
            print("  pair (coarse->fine) : mean_containment (oracle, gap, n, empty):")
            for r in valid:
                if "mean_containment" in r:
                    gap = r["mean_containment_oracle"] - r["mean_containment"]
                    print(f"    {r['class']:<28s} {r['spacing']:g}->{r['locator_to']:g}mm : "
                          f"{r['mean_containment']:.3f}  (orc {r['mean_containment_oracle']:.3f}, "
                          f"gap {gap:.3f}, n={r['n_locator']}, empty={r['n_locator_empty']})")
```

- [ ] **Step 4: Add locator columns to eval.csv**

In `experiments/3d/eval.py`, replace the CSV build block (lines 282-287):

```python
    sweep_col = ",spacing" if sweep else ""
    csv = [f"model,class,mean_dice,std_dice,mean_time_ms,gflops,n_samples{sweep_col}"]
    csv += [f"{model_name},{r['class']},{r['mean_dice']},{r['std_dice']},"
            f"{r.get('mean_time_ms','')},{r.get('gflops','')},{r['n_samples']}"
            + (f",{r.get('spacing','')}" if sweep else "")
            for r in rows if "mean_dice" in r]
```

with:

```python
    sweep_col = ",spacing" if sweep else ""
    loc_col = ",locator_to,mean_containment,mean_containment_oracle,mean_loc_err_mm" if locator else ""
    csv = [f"model,class,mean_dice,std_dice,mean_time_ms,gflops,n_samples{sweep_col}{loc_col}"]
    csv += [f"{model_name},{r['class']},{r['mean_dice']},{r['std_dice']},"
            f"{r.get('mean_time_ms','')},{r.get('gflops','')},{r['n_samples']}"
            + (f",{r.get('spacing','')}" if sweep else "")
            + (f",{r.get('locator_to','')},{r.get('mean_containment','')},"
               f"{r.get('mean_containment_oracle','')},{r.get('mean_loc_err_mm','')}" if locator else "")
            for r in rows if "mean_dice" in r]
```

- [ ] **Step 5: Verify wiring + non-locator invariance**

Run: `python -c "import ast; s=open('experiments/3d/eval.py').read(); ast.parse(s); assert 'locator = bool(cfg.eval.get(\"spacing_locator\"))' in s; assert 'locator=locator' in s; assert 'mean_containment' in s; assert 'loc_col = ' in s; assert 'model,class,mean_dice,std_dice,mean_time_ms,gflops,n_samples{sweep_col}{loc_col}' in s; print('ok')"`
Expected: prints `ok` (flag read, passed to sweep, containment in output; CSV header base unchanged with `loc_col` empty when locator is falsy).

- [ ] **Step 6: Commit**

```bash
git add experiments/3d/eval.py
git commit -m "feat(eval): surface locator containment in eval.py output"
```

---

### Task 6: config key & docs

**Files:**
- Modify: `configs/experiment/3d/eval.yaml` (document `spacing_locator: false`)
- Modify: `docs/logs.md`

**Interfaces:**
- Consumes: `cfg.eval.spacing_locator` read by `eval.py` (Task 5) and guarded (Task 4).
- Produces: a documented, defaulted-off `spacing_locator: false` key.

- [ ] **Step 1: Locate the sweep key to anchor the edit**

Run: `grep -n "spacing_sweep\|spacing_locator" configs/experiment/3d/eval.yaml`
Expected: shows `spacing_sweep: null` (with its comment) and confirms `spacing_locator` is absent. Note its indentation (2 spaces under `eval:`).

- [ ] **Step 2: Add the documented key**

In `configs/experiment/3d/eval.yaml`, directly AFTER the `spacing_sweep: null` line (and its comment block), add at the same indentation:

```yaml
  # Coarse->fine localization metric layered on spacing_sweep. For each descending
  # consecutive pair (e.g. [4,2]), use the coarse (4mm) target prediction to place a
  # fine-spacing (2mm) box and measure |GT ∩ box| / |GT| (containment), plus an oracle
  # box on the GT centroid. Adds one soft-prob forward per non-final spacing. Requires a
  # spacing_sweep with a descending step (+ totalseg / use_crop). false = sweep only.
  spacing_locator: false
```

- [ ] **Step 3: Verify the key parses**

Run: `python -c "import yaml; ev=yaml.safe_load(open('configs/experiment/3d/eval.yaml'))['eval']; assert ev['spacing_locator'] is False, ev.get('spacing_locator'); print('eval.spacing_locator =', ev['spacing_locator'])"`
Expected: prints `eval.spacing_locator = False`.

- [ ] **Step 4: Log the change**

Append an entry to `docs/logs.md` (date 2026-08-07, matching the existing entry style) describing: coarse→fine locator added (`cfg.eval.spacing_locator`, default off), layered on `spacing_sweep`; per-descending-pair containment + GT-centroid oracle + `loc_err_mm`; soft prob-weighted centroid via `model.train_forward` (hard-mask fallback); per-(class,pair) columns in eval.csv/json + `class/<c>/containment@<s>` wandb scalars; one extra forward per non-final spacing; usage `python experiments/3d/eval.py 'eval.spacing_sweep=[4,2]' eval.spacing_locator=true eval.crop_jitter=0`.

- [ ] **Step 5: Commit**

```bash
git add configs/experiment/3d/eval.yaml docs/logs.md
git commit -m "docs(eval): document eval.spacing_locator config key"
```

---

### Task 7: End-to-end smoke run (GPU/data node)

**Files:** none (verification only)

- [ ] **Step 1: Sweep without locator — must be unchanged**

Run (GPU node with data):
```bash
python experiments/3d/eval.py eval.model=medverse eval.n_subjects=2 \
    data.val_classes='[liver]' 'eval.spacing_sweep=[4,2]' eval.crop_jitter=0 wandb.project=null
```
Expected: two per-spacing rows, no `cont=` in the print, eval.csv header ends `...,spacing` (no locator columns).

- [ ] **Step 2: Sweep with locator — containment reported**

Run:
```bash
python experiments/3d/eval.py eval.model=medverse eval.n_subjects=2 \
    data.val_classes='[liver]' 'eval.spacing_sweep=[4,2]' eval.spacing_locator=true \
    eval.crop_jitter=0 wandb.project=null
```
Expected: the 4mm row prints `cont=… (orc=…)` and a `locator_to=2`; the 2mm row has no containment (finest); a `pair (coarse->fine)` headline block prints; eval.csv header includes `locator_to,mean_containment,mean_containment_oracle,mean_loc_err_mm`.

- [ ] **Step 3: Guard smoke — non-descending sweep errors clearly**

Run:
```bash
python experiments/3d/eval.py 'eval.spacing_sweep=[2,4]' eval.spacing_locator=true \
    eval.n_subjects=1 wandb.project=null
```
Expected: fails fast with `ValueError: eval.spacing_locator requires eval.spacing_sweep with at least one descending step …`.

- [ ] **Step 4: Commit (only if the smoke surfaced doc tweaks)**

```bash
git add -A && git commit -m "chore(eval): note spacing-locator smoke results"
```

(If nothing changed, skip — no empty commit.)
