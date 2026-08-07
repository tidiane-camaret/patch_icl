# Eval Spacing Sweep Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let `experiments/3d/eval.py` evaluate each sample at a *set* of physical crop spacings (e.g. 1.5/2.5/3.5 mm) and report per-(class, spacing) Dice, gated behind `cfg.eval.spacing_sweep`.

**Architecture:** Add a `spacing` param to `make_eval_loader` that swaps the plain `DataLoader` for one driven by `SpacingBatchSampler(SequentialSampler(ds), bs, [s, s])` — a constant-spacing pass that carries `(idx, s)` tuples into worker `__getitem__`. A new `evaluate_spacing_sweep` driver loops the spacings, builds one constant-spacing loader per `s`, calls the **unmodified** `evaluate_classes(loader=…)`, tags each row with its spacing, and saves figures only on the first pass. `eval.py` branches on `cfg.eval.spacing_sweep`, adds a `spacing` column to CSV/JSON, and logs per-`@{s}` wandb scalars.

**Tech Stack:** PyTorch DataLoader/Sampler, Hydra/OmegaConf config, pytest, Python 3.10+ (`float | None` unions).

## Global Constraints

- Spec: `docs/superpowers/specs/2026-08-07-eval-spacing-sweep-design.md` (approved).
- `cfg.eval.spacing_sweep` absent/`null` → **byte-identical** to current single-spacing behaviour. Every change is gated behind it.
- `evaluate_classes` (experiments/3d/evaluate.py) MUST stay untouched — it is shared with `train.py`'s val step. New behaviour goes in a separate `evaluate_spacing_sweep` driver that calls it with a prebuilt `loader`.
- Sweep is supported **only** on the totalseg direct-build path with `data.use_crop=true`. Sources routed through `build_dataset` (omnisynth3d / anchor_synth3d / totalseg_more_labels) and `use_crop=false` must fail fast with a clear message.
- Spacing is driven through the batch-sampler `(idx, spacing)` tuple, never by mutating `ds.crop_spacing_mm` (attribute mutation does not reach `num_workers>0` workers).
- Project rule (CLAUDE.md): "Write tests only when necessary." Only the pure-logic guard gets a pytest; loader/driver wiring is verified with focused smoke snippets. Log changes to `docs/logs.md`.
- `git` is currently unavailable on this node — run each **Commit** step where git is on PATH (e.g. prefix with `! git …` in the session, or commit from a git-capable node). Do not skip the commits; batch them if needed.

---

### Task 1: `make_eval_loader` constant-spacing pass

**Files:**
- Modify: `experiments/3d/common.py:16` (import), `experiments/3d/common.py:243-306` (`make_eval_loader`)

**Interfaces:**
- Consumes: `SpacingBatchSampler(sampler, batch_size, spacing_range, drop_last=False, seed=0)` (common.py:183); `incontext_collate_fn`; `TotalSegInContextDataset`.
- Produces: `make_eval_loader(cfg, classes, split="test", spacing: float | None = None) -> DataLoader`. When `spacing=s`, the returned loader's `batch_sampler` is a `SpacingBatchSampler` over `SequentialSampler(ds)` with `spacing_range=[s, s]`; otherwise behaviour is unchanged.

- [ ] **Step 1: Add `SequentialSampler` to the torch import**

In `experiments/3d/common.py:16`, change:

```python
from torch.utils.data import DataLoader, RandomSampler
```

to:

```python
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
```

- [ ] **Step 2: Add the `spacing` parameter and route it through the totalseg branch**

In `experiments/3d/common.py`, update the signature at line 243:

```python
def make_eval_loader(cfg, classes, split: str = "test", spacing: float | None = None) -> DataLoader:
```

Append to the docstring (after the existing final paragraph, before the code):

```
    `spacing` (mm/voxel) forces every crop in the eval pass to that one physical
    spacing via SpacingBatchSampler([s, s]) over a SequentialSampler — the (idx, s)
    tuples reach __getitem__ inside worker processes (mutating ds.crop_spacing_mm
    would not). Only the totalseg direct-build branch honours it; the build_dataset-
    routed sources (omnisynth3d/anchor_synth3d/totalseg_more_labels) ignore it (the
    sweep is guarded to totalseg in eval.py). None = today's fixed-crop_spacing_mm pass.
```

In the totalseg branch (the code after `_, root, is_mri = _source_root(cfg)` at line 271), replace the final `DataLoader(...)` return (lines 296-306) with a batch-sampler-vs-batch-size split:

```python
    nw = int(e.get("workers", 4))
    common = dict(
        num_workers=nw,
        collate_fn=incontext_collate_fn,
        pin_memory=DEVICE.type == "cuda",
        persistent_workers=nw > 0,
        prefetch_factor=2 if nw > 0 else None,
    )
    if spacing is not None:
        # Constant-spacing pass: SpacingBatchSampler([s, s]) makes every batch that one
        # physical spacing; the (idx, s) tuples travel into worker __getitem__ so both the
        # crop and the reported `spacing` follow. SequentialSampler keeps eval order stable.
        batch_sampler = SpacingBatchSampler(
            SequentialSampler(ds), int(e.get("batch_size", 8)), [spacing, spacing])
        return DataLoader(ds, batch_sampler=batch_sampler, **common)
    return DataLoader(ds, batch_size=int(e.get("batch_size", 8)), shuffle=False, **common)
```

- [ ] **Step 3: Verify the branch wiring by inspection**

Run: `python -c "import ast,sys; src=open('experiments/3d/common.py').read(); assert 'SequentialSampler' in src; assert 'spacing: float | None = None' in src; assert 'SpacingBatchSampler(' in src.split('def make_eval_loader')[1]; print('ok')"`
Expected: prints `ok` (import present, new param present, sampler used inside `make_eval_loader`).

- [ ] **Step 4: Confirm the non-sweep path is unchanged**

Run: `python -c "import re; b=open('experiments/3d/common.py').read().split('def make_eval_loader')[1]; assert 'batch_size=int(e.get(\"batch_size\", 8)), shuffle=False' in b; print('default path intact')"`
Expected: prints `default path intact` (default `spacing=None` still returns the plain `shuffle=False` loader).

- [ ] **Step 5: Commit**

```bash
git add experiments/3d/common.py
git commit -m "feat(eval): add constant-spacing pass to make_eval_loader"
```

---

### Task 2: `evaluate_spacing_sweep` driver

**Files:**
- Modify: `experiments/3d/evaluate.py` (add function after `evaluate_classes`, which ends at line 424)

**Interfaces:**
- Consumes: `make_eval_loader(cfg, classes, split, spacing)` (Task 1); `evaluate_classes(model, cfg, classes, *, loader=…, fig_dir=…)` (evaluate.py:280, unmodified).
- Produces: `evaluate_spacing_sweep(model, cfg, classes, spacings, *, split=None, fig_dir=None) -> (rows, cases)`. `rows` is per-(class, spacing): each dict from `evaluate_classes` plus `r["spacing"] = s`. `cases` is every pass concatenated; each case already carries `case["spacing"]` from `evaluate_classes`.

- [ ] **Step 1: Add the driver**

Append to `experiments/3d/evaluate.py` (after line 424, the end of `evaluate_classes`):

```python
def evaluate_spacing_sweep(model, cfg, classes, spacings, *, split=None, fig_dir=None):
    """Run evaluate_classes once per physical crop spacing; tag rows with their spacing.

    Builds a constant-spacing eval loader per `s` (make_eval_loader(..., spacing=s)) and
    calls the unmodified evaluate_classes with that prebuilt loader. `idx` is stable across
    passes, so each spacing sees the same task + context subjects — only the crop spacing
    changes. Figures are saved on the first spacing only (later passes reuse the filenames).
    Returns (rows, cases): rows are per-(class, spacing); cases are all passes concatenated
    (each case already carries case["spacing"]).
    """
    from common import make_eval_loader  # local import: common/evaluate are siblings

    rows, cases = [], []
    for i, s in enumerate(spacings):
        loader = make_eval_loader(cfg, classes, split=split or cfg.eval.split, spacing=s)
        rows_s, cases_s = evaluate_classes(
            model, cfg, classes, loader=loader,
            fig_dir=fig_dir if i == 0 else None)
        for r in rows_s:
            r["spacing"] = s
        rows.extend(rows_s)
        cases.extend(cases_s)
    return rows, cases
```

- [ ] **Step 2: Verify it imports and has the right signature**

Run: `cd experiments/3d && python -c "import inspect, evaluate; sig=inspect.signature(evaluate.evaluate_spacing_sweep); print(list(sig.parameters))"`
Expected: prints `['model', 'cfg', 'classes', 'spacings', 'split', 'fig_dir']`.

- [ ] **Step 3: Verify figures-first-pass logic by source inspection**

Run: `python -c "s=open('experiments/3d/evaluate.py').read().split('def evaluate_spacing_sweep')[1]; assert 'fig_dir=fig_dir if i == 0 else None' in s; assert 'r[\"spacing\"] = s' in s; print('ok')"`
Expected: prints `ok` (figures gated to `i == 0`; every row tagged with its spacing).

- [ ] **Step 4: Commit**

```bash
git add experiments/3d/evaluate.py
git commit -m "feat(eval): add evaluate_spacing_sweep driver"
```

---

### Task 3: Sweep guard (`_assert_sweep_supported`)

**Files:**
- Modify: `experiments/3d/eval.py` (add helper near the top-level helpers, after `_warn_uninherited_data` which ends at line 72)
- Test: `experiments/3d/tests/test_sweep_guard.py`

**Interfaces:**
- Produces: `_assert_sweep_supported(cfg) -> None`. Raises `ValueError` with a clear message when `cfg.data.use_crop` is not truthy, or when `cfg.data.source` is one of the `build_dataset`-routed sources (`omnisynth3d`, `anchor_synth3d`, `totalseg_more_labels`). Returns `None` on the supported totalseg + crop path.

- [ ] **Step 1: Write the failing test**

Create `experiments/3d/tests/test_sweep_guard.py`:

```python
"""Guard for the eval spacing sweep: totalseg + use_crop only, else a clear ValueError."""
import sys
from pathlib import Path

import pytest
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # sibling eval.py

from eval import _assert_sweep_supported  # noqa: E402


def _cfg(source="totalseg", use_crop=True):
    return OmegaConf.create({"data": {"source": source, "use_crop": use_crop}})


def test_totalseg_crop_ok():
    _assert_sweep_supported(_cfg())  # no raise


def test_use_crop_false_rejected():
    with pytest.raises(ValueError, match="use_crop"):
        _assert_sweep_supported(_cfg(use_crop=False))


@pytest.mark.parametrize("src", ["omnisynth3d", "anchor_synth3d", "totalseg_more_labels"])
def test_unsupported_source_rejected(src):
    with pytest.raises(ValueError, match="spacing_sweep"):
        _assert_sweep_supported(_cfg(source=src))
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd experiments/3d && python -m pytest tests/test_sweep_guard.py -v`
Expected: FAIL — `ImportError: cannot import name '_assert_sweep_supported' from 'eval'`.

- [ ] **Step 3: Implement the guard**

In `experiments/3d/eval.py`, add after `_warn_uninherited_data` (after line 72, before `_build_model`):

```python
def _assert_sweep_supported(cfg: DictConfig) -> None:
    """Fail fast when eval.spacing_sweep is set on an unsupported config.

    The per-spacing crop override only takes effect on the totalseg direct-build path
    with use_crop=true (the resized path ignores _cur_crop_spacing; build_dataset-routed
    sources build their own datasets). Reject anything else with a clear message rather
    than silently producing a single-spacing result."""
    if not cfg.data.get("use_crop"):
        raise ValueError(
            "eval.spacing_sweep requires data.use_crop=true — the crop-spacing override is a "
            "no-op on the pre-resized path (it reports fixed voxel spacing).")
    source = cfg.data.get("source", "totalseg")
    if source in ("omnisynth3d", "anchor_synth3d", "totalseg_more_labels"):
        raise ValueError(
            f"eval.spacing_sweep is unsupported for data.source={source!r} (routed through "
            "build_dataset); it works only on the totalseg direct-build eval path.")
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd experiments/3d && python -m pytest tests/test_sweep_guard.py -v`
Expected: PASS (5 tests: `test_totalseg_crop_ok`, `test_use_crop_false_rejected`, 3× `test_unsupported_source_rejected`).

- [ ] **Step 5: Commit**

```bash
git add experiments/3d/eval.py experiments/3d/tests/test_sweep_guard.py
git commit -m "feat(eval): guard spacing sweep to totalseg + use_crop"
```

---

### Task 4: `eval.py` wiring & output

**Files:**
- Modify: `experiments/3d/eval.py:32` (import), `:211` (branch), `:216-237` (per-row print + wandb + headline), `:240-247` (json/csv)

**Interfaces:**
- Consumes: `evaluate_spacing_sweep` (Task 2), `_assert_sweep_supported` (Task 3), `evaluate_classes` (unchanged).
- Produces: when `cfg.eval.spacing_sweep` is a non-empty list, `rows` carry a `spacing` key; `eval.csv`/`eval.json` gain a `spacing` column; wandb scalars are keyed `class/{cls}/mean_dice@{s}` etc.; console prints an aggregate `spacing → mean_dice` block.

- [ ] **Step 1: Import the sweep driver**

In `experiments/3d/eval.py:32`, change:

```python
from evaluate import measure_flops, evaluate_classes, build_sample_table
```

to:

```python
from evaluate import measure_flops, evaluate_classes, evaluate_spacing_sweep, build_sample_table
```

- [ ] **Step 2: Branch the eval call on the sweep config**

In `experiments/3d/eval.py`, replace line 211:

```python
    rows, all_cases = evaluate_classes(model, cfg, classes, fig_dir=fig_dir)
```

with:

```python
    sweep = cfg.eval.get("spacing_sweep")
    if sweep:
        _assert_sweep_supported(cfg)
        spacings = list(sweep)
        print(f"  Spacing sweep: {spacings} mm  ({len(spacings)}x eval time)\n")
        rows, all_cases = evaluate_spacing_sweep(model, cfg, classes,
                                                 spacings, fig_dir=fig_dir)
    else:
        rows, all_cases = evaluate_classes(model, cfg, classes, fig_dir=fig_dir)
```

- [ ] **Step 3: Add spacing to the per-row print and wandb scalars**

In `experiments/3d/eval.py`, replace the per-row loop (lines 216-227) — the block from `for row in rows:` through the `wandb.log({...})` call — with:

```python
    for row in rows:
        cls = row["class"]
        sp = row.get("spacing")
        sp_str = f" @{sp:g}mm" if sp is not None else ""
        sp_key = f"@{sp:g}" if sp is not None else ""
        if "error" in row:
            print(f"  {cls:<35s}{sp_str}  ERROR: {row['error']}")
            continue
        row["gflops"] = round(gflops, 2)
        print(f"  {cls:<35s}{sp_str}  dice={row['mean_dice']:.3f} ± {row['std_dice']:.3f}"
              f"  {row['mean_time_ms']:.0f}ms/sample  n={row['n_samples']}")
        if wb_on:
            wandb.log({f"class/{cls}/mean_dice{sp_key}": row["mean_dice"],
                       f"class/{cls}/std_dice{sp_key}": row["std_dice"],
                       f"class/{cls}/mean_time_ms{sp_key}": row["mean_time_ms"]})
```

- [ ] **Step 4: Add the per-spacing headline summary**

In `experiments/3d/eval.py`, replace the aggregate block (lines 229-237) — from `valid = [r for r in rows if "mean_dice" in r]` through its `wandb.log({...})` — with:

```python
    valid = [r for r in rows if "mean_dice" in r]
    if valid:
        mean_dice = sum(r["mean_dice"] for r in valid) / len(valid)
        mean_ms   = sum(r["mean_time_ms"] for r in valid) / len(valid)
        print(f"\n  Mean Dice: {mean_dice:.4f}  |  Mean time: {mean_ms:.1f} ms/sample  "
              f"|  GFLOPs: {gflops:.2f}")
        if wb_on:
            wandb.log({"mean_dice": round(mean_dice, 4), "mean_time_ms": round(mean_ms, 1),
                       "gflops": round(gflops, 2), "cases": case_table})
        if sweep:
            # Aggregate curve: mean Dice over classes at each spacing.
            print("  spacing -> mean_dice:")
            for s in spacings:
                vs = [r["mean_dice"] for r in valid if r.get("spacing") == s]
                if vs:
                    md = sum(vs) / len(vs)
                    print(f"    {s:g}mm : {md:.4f}  (n_classes={len(vs)})")
                    if wb_on:
                        wandb.log({f"mean_dice@{s:g}": round(md, 4)})
```

- [ ] **Step 5: Add the spacing column to json/csv**

In `experiments/3d/eval.py`, replace the output block (lines 240-247) — from `(out_dir / "eval.json").write_text(...)` through the `eval.csv` write — with:

```python
    (out_dir / "eval.json").write_text(json.dumps(
        {"model": model_name, "config": OmegaConf.to_container(cfg.eval, resolve=True),
         "rows": rows}, indent=2))
    sweep_col = ",spacing" if sweep else ""
    csv = [f"model,class,mean_dice,std_dice,mean_time_ms,gflops,n_samples{sweep_col}"]
    csv += [f"{model_name},{r['class']},{r['mean_dice']},{r['std_dice']},"
            f"{r.get('mean_time_ms','')},{r.get('gflops','')},{r['n_samples']}"
            + (f",{r.get('spacing','')}" if sweep else "")
            for r in rows if "mean_dice" in r]
    (out_dir / "eval.csv").write_text("\n".join(csv) + "\n")
```

(`rows` already carry `spacing` via `evaluate_spacing_sweep`, so `eval.json` gets the column for free; only the CSV header/row need the conditional column.)

- [ ] **Step 6: Verify the file parses and the wiring is present**

Run: `python -c "import ast; ast.parse(open('experiments/3d/eval.py').read()); s=open('experiments/3d/eval.py').read(); assert 'evaluate_spacing_sweep' in s; assert '_assert_sweep_supported(cfg)' in s; assert 'spacing -> mean_dice' in s; assert 'mean_dice{sp_key}' in s; print('ok')"`
Expected: prints `ok` (valid Python; sweep branch, guard call, headline, and per-`@{s}` wandb key all present).

- [ ] **Step 7: Confirm single-spacing output is untouched**

Run: `python -c "s=open('experiments/3d/eval.py').read(); assert 'else:\n        rows, all_cases = evaluate_classes(model, cfg, classes, fig_dir=fig_dir)' in s; assert 'model,class,mean_dice,std_dice,mean_time_ms,gflops,n_samples{sweep_col}' in s; print('default output intact')"`
Expected: prints `default output intact` (no-sweep branch still calls `evaluate_classes` directly; CSV header base is unchanged, `sweep_col` empty when `sweep` is falsy).

- [ ] **Step 8: Commit**

```bash
git add experiments/3d/eval.py
git commit -m "feat(eval): wire spacing sweep into eval.py output"
```

---

### Task 5: Config surface & docs

**Files:**
- Modify: `configs/experiment/3d/eval.yaml` (document the new key — value stays `null`)
- Modify: `docs/logs.md`

**Interfaces:**
- Consumes: `cfg.eval.spacing_sweep` read by `eval.py` (Task 4).
- Produces: a documented, defaulted-off `eval.spacing_sweep: null` key.

- [ ] **Step 1: Locate the eval config's `eval:` block**

Run: `grep -n "spacing_sweep\|save_figures\|^eval:\|crop_jitter" configs/experiment/3d/eval.yaml`
Expected: shows the `eval:` block keys (e.g. `save_figures`, `crop_jitter`) and confirms `spacing_sweep` is absent. Note a nearby key (e.g. `save_figures`) to anchor the edit.

- [ ] **Step 2: Add the documented key**

In `configs/experiment/3d/eval.yaml`, inside the `eval:` block (next to `save_figures`/`crop_jitter`), add:

```yaml
  # Evaluate every sample at each of these physical crop spacings (mm/voxel), e.g.
  # [1.5, 2.5, 3.5], instead of the single data.crop_spacing_mm. Produces per-(class,
  # spacing) rows in eval.csv/json and class/<c>/mean_dice@<s> wandb scalars. Same
  # GFLOPs/sample at every spacing; wall time is len(list)x. null = single-spacing eval.
  # Requires data.use_crop=true and the totalseg source; pair with crop_jitter=0 for a
  # fully controlled sweep (centered crops, only spacing changes across passes).
  spacing_sweep: null
```

(Match the existing indentation of keys under `eval:` — 2 spaces if the file nests them, else top-level per the file's style.)

- [ ] **Step 3: Verify the key resolves via Hydra**

Run: `cd experiments/3d && python -c "from hydra import compose, initialize_config_dir; from pathlib import Path; initialize_config_dir(config_dir=str(Path('../../configs/experiment/3d').resolve()), version_base='1.3').__enter__(); from hydra import compose as c; cfg=c(config_name='eval'); print('spacing_sweep =', cfg.eval.get('spacing_sweep'))"`
Expected: prints `spacing_sweep = None` (key present, defaults off).

- [ ] **Step 4: Log the change**

Append an entry to `docs/logs.md` describing: eval spacing sweep added (`cfg.eval.spacing_sweep`), gated + off by default, per-(class,spacing) CSV/JSON + `@{s}` wandb scalars, figures saved on first spacing only, guarded to totalseg + `use_crop`; usage example `python experiments/3d/eval.py 'eval.spacing_sweep=[1.5,2.5,3.5]' eval.crop_jitter=0`.

- [ ] **Step 5: Commit**

```bash
git add configs/experiment/3d/eval.yaml docs/logs.md
git commit -m "docs(eval): document eval.spacing_sweep config key"
```

---

### Task 6: End-to-end smoke run

**Files:** none (verification only)

- [ ] **Step 1: Single-spacing regression (no sweep) — must be unchanged**

Run (on a GPU node with data):
```bash
python experiments/3d/eval.py eval.model=medverse eval.n_subjects=2 \
    data.val_classes='[liver]' wandb.project=null
```
Expected: runs as before; `eval.csv` header is `model,class,mean_dice,std_dice,mean_time_ms,gflops,n_samples` (no `spacing` column).

- [ ] **Step 2: Sweep run — per-(class, spacing) output**

Run:
```bash
python experiments/3d/eval.py eval.model=medverse eval.n_subjects=2 \
    data.val_classes='[liver]' 'eval.spacing_sweep=[1.5,2.5]' eval.crop_jitter=0 \
    wandb.project=null
```
Expected: console prints two rows for `liver` (`@1.5mm`, `@2.5mm`) and a `spacing -> mean_dice` block; `eval.csv` header ends with `,spacing` and has one row per (class, spacing); figures dir (if `save_figures`) has each class once (from the 1.5 mm pass only).

- [ ] **Step 3: Guard smoke — unsupported config errors clearly**

Run:
```bash
python experiments/3d/eval.py 'eval.spacing_sweep=[1.5,2.5]' data.use_crop=false eval.n_subjects=1 wandb.project=null
```
Expected: fails fast with `ValueError: eval.spacing_sweep requires data.use_crop=true …` before any eval loop runs.

- [ ] **Step 4: Commit (if any doc tweaks resulted from the smoke run)**

```bash
git add -A
git commit -m "chore(eval): note spacing-sweep smoke results"
```

(If nothing changed, skip — no empty commit.)
