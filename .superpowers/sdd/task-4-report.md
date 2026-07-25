# Task 4 Report: Sweep planner + Hydra driver (run.py) + config

## Status
DONE

## TDD RED/GREEN for plan_sweep

**RED (Step 2):** Ran `python -m pytest tests/test_feature_sim_sweep.py -v` before creating
`run.py`. Result: `ModuleNotFoundError: No module named 'feature_sim.run'` — 1 error,
0 collected. Confirmed RED as expected.

**GREEN (Step 4):** After implementing `experiments/3d/feature_sim/run.py`, re-ran the same
command. Result: `2 passed, 1 warning in 6.60s`. Both tests pass:
- `test_plan_sweep_mode_and_budget` — verifies mode="dense" for 16^3 <= 48^3, mode="point"
  for 64^3 > 48^3, and len==4 for 2 tiers × 2 resolutions.
- `test_plan_sweep_transformer_q_pinned_to_R` — verifies transformer_q tier is pinned to
  res=R=16 mode="dense", deduped to exactly 1 row regardless of resolutions list.

## Import Check Result

```
.venv_nero/bin/python -c "import sys; sys.path.insert(0,'.'); sys.path.insert(0,'experiments/3d'); from feature_sim.run import plan_sweep, main; print('import ok')"
```
Output: `import ok` (plus CUDA version warning, not an error — old driver on this node).
All top-level imports resolve: `common`, `data.totalseg_classes`, `feature_sim.adapters`,
`feature_sim.labels`, `feature_sim.metrics`.

## All Feature-Sim Tests

19/19 passed across all 4 test files:
- `test_feature_sim_metrics.py`: 8 tests
- `test_feature_sim_labels.py`: 4 tests
- `test_feature_sim_adapters.py`: 5 tests
- `test_feature_sim_sweep.py`: 2 tests (new)

## Files Created/Modified

- **Created** `/home/dpxuser/dev/patch_icl/experiments/3d/feature_sim/run.py` — `plan_sweep`
  pure function + `_load_patchset` (mirrors eval.py:55-83) + `_rows_for_task` + `_metric_row`
  + `@hydra.main` driver writing `feature_sim.csv`.
- **Created** `/home/dpxuser/dev/patch_icl/configs/experiment/3d/feature_sim.yaml` — mirrors
  eval.yaml structure (same defaults/searchpath) plus `feature_sim:` block with tiers,
  resolutions, budget, n_fg, n_bg, band.
- **Created** `/home/dpxuser/dev/patch_icl/tests/test_feature_sim_sweep.py` — 2 plan_sweep
  unit tests (verbatim from brief).
- **Modified** `/home/dpxuser/dev/patch_icl/docs/logs.md` — prepended the feature-similarity
  study entry (verbatim from brief Step 7).

## DEFERRED: End-to-End Hydra Smoke Run

No trained PatchSet3D checkpoint is available in this environment
(`results/checkpoints/*/best.pt` is empty). The full smoke run cannot be executed here.

Exact command from the brief (run on a GPU node with a real checkpoint):
```bash
python experiments/3d/feature_sim/run.py \
    eval.checkpoint=$(ls -t results/checkpoints/*/best.pt 2>/dev/null | head -1) \
    eval.n_subjects=2 eval.batch_size=2 \
    feature_sim.resolutions='[16,64]'
```
Expected output: `Done. <N> rows -> .../feature_sim.csv`; CSV has columns
`class,obj_vox,real_dice,tier,res,mode,tier_native_res,K,auroc,soft_dice,ap,margin,retrieval_at1`
with mode=dense for res 16 and mode=point for res 64.

## Self-Review

- `plan_sweep` logic: transformer_q forced to res=R, mode="dense"; other tiers use
  mode="point" iff res^3 > budget; deduplication via (tier, res, mode) set — matches spec.
- config_path in `@hydra.main` is `"../../../configs/experiment/3d"` (run.py is 3 dirs
  deeper than configs/experiment/3d/).
- `_load_patchset` is verbatim from eval.py:55-83 with the same open_dict/arch logic.
- Variable shadowing in point-mode loop (`cls`, `cl`) — the brief's code itself has this;
  not fixed to stay verbatim. In practice `cls` is the class name string from outer scope;
  the inner `cls` list only lives inside the else branch and feeds `cf`/`cl` at the end.
  The final `yield _metric_row(...)` call uses `cls` from the outer scope (class name) —
  correct by Python scoping since the inner `cls` list and outer `cls` string are different
  types. No functional bug, but the name reuse is a minor style concern.
- CSV columns: 13 columns match spec exactly.
- `eval.checkpoint` is required at runtime (raises inside `_load_patchset` if null); this
  is correct — the config sets it to `null` as a placeholder requiring override.

## Concerns

1. **Deferred smoke run**: Full end-to-end validation (CSV output, column check, mode
   switching) requires a real checkpoint + data on a GPU node. Not a blocker for merge.
2. **`cls`/`cl` variable shadowing** in `_rows_for_task` (noted above) — verbatim from brief,
   no bug but could confuse future readers.
3. CUDA driver warning on this node (driver 12030 is older than torch's expectation) — 
   environment issue, not a code issue.

---

## Review Fix Note (commit 25786e6)

Five code-review findings applied to `experiments/3d/feature_sim/run.py`:

1. **Fix 1 (CRITICAL) — cls/cl shadowing**: In `_rows_for_task` point-mode branch, renamed
   the list accumulator `cls` → `ctx_labels` (and `cls.append(ll)` → `ctx_labels.append(ll)`,
   `torch.cat(cls, 0)` → `torch.cat(ctx_labels, 0)`). Prevents the class-name string `cls`
   (used in CSV `class` column) from being clobbered by the inner list accumulator.

2. **Fix 2 (IMPORTANT) — transformer_q context dim mismatch**: Changed context feature
   extraction in the `transformer_q` branch from `"concat"` tier (dim = sum of encoder stage
   channels) to `"img_embed"` tier (dim = transformer embedding dim `e`), so both the target
   query `q` and context features `cf` share the same dim for `prototype_cosine`. Updated
   adjacent comment to explain the tier choice and note approximate ceiling reference nature.

3. **Fix 3 (IMPORTANT) — no stored arch guard in _load_patchset**: Added `elif "arch" not in
   cfg: raise ValueError(...)` guard mirroring `eval.py:71-76`, so older checkpoints without
   a stored arch fail loudly with an actionable message instead of silently proceeding.

4. **Fix 4 (MINOR) — dead transformer_q guard in dense/point yield**: Simplified
   `adapter.native_res("concat" if tier == "transformer_q" else tier, input_res)` to
   `adapter.native_res(tier, input_res)` — the `transformer_q` branch always `continue`s
   earlier and can never reach this yield.

5. **Fix 5 (MINOR) — fail fast on missing checkpoint**: Added early guard in `main()` before
   `make_eval_loader(...)` that raises `ValueError` when `eval.checkpoint` is not set,
   preventing the expensive dataset/loader setup from running unnecessarily.

### Verification commands and output

```
python -m pytest tests/test_feature_sim_sweep.py -v
```
Output:
```
collected 2 items
tests/test_feature_sim_sweep.py::test_plan_sweep_mode_and_budget PASSED  [ 50%]
tests/test_feature_sim_sweep.py::test_plan_sweep_transformer_q_pinned_to_R PASSED [100%]
2 passed in 5.72s
```

```
python -c "import sys; sys.path.insert(0,'.'); sys.path.insert(0,'experiments/3d'); from feature_sim.run import plan_sweep, main; print('import ok')"
```
Output: `import ok`

---

## Final Review Fix Note — wrong batch key `label_name` → `label_names`

**RED (bug confirmed):**
Added `test_rows_for_task_reads_batch_keys_and_shapes` to `tests/test_feature_sim_sweep.py` and
ran against the CURRENT (buggy) code. Result:

```
FAILED tests/test_feature_sim_sweep.py::test_rows_for_task_reads_batch_keys_and_shapes
AssertionError: assert False
 +  where False = all(<generator object ... at 0x7f2065cf01e0>)
```

The `class == "liver"` assertion failed because `item.get("label_name", ["?"])[0]` returned `"?"`
— the key `label_name` does not exist; collate emits `label_names` (plural).

**Fix applied** in `experiments/3d/feature_sim/run.py` line 71:
```python
# Before (WRONG)
cls = item.get("label_name", ["?"])[0]
# After (CORRECT)
cls = item["label_names"][0]
```

**GREEN (all 3 tests pass):**
```
tests/test_feature_sim_sweep.py::test_plan_sweep_mode_and_budget PASSED          [ 33%]
tests/test_feature_sim_sweep.py::test_plan_sweep_transformer_q_pinned_to_R PASSED [ 66%]
tests/test_feature_sim_sweep.py::test_rows_for_task_reads_batch_keys_and_shapes PASSED [100%]
3 passed, 1 warning in 2.59s
```
