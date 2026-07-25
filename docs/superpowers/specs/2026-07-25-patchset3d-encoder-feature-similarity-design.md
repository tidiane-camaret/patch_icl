# PatchSet3D encoder feature-similarity study — design

**Date:** 2026-07-25
**Status:** approved design, pre-implementation
**Author:** brainstormed with Tidiane

## Motivation

PatchSet3D's in-context segmentation rests on one premise: a foreground cell in the
target volume should attend to foreground cells in the K context volumes based on
**content (feature) similarity** (`src/models/patchset3d.py:_attn`). If the conv
encoder does not place same-label cells near each other in feature space, the
transformer has nothing to match on. This study measures **how expressive the encoder
embeddings are for target↔context matching**, so we can (a) understand how encoder
representations drive the ICL process and final Dice, and (b) later compare our trained
encoder against pretrained encoders (SAM/SAME anatomical embeddings, DINOv2 / a 3D
medical FM).

This phase delivers the **measurement machinery** and a first table on our own encoder.
Correlation-with-Dice analysis and concrete pretrained-encoder adapters are phase 2 —
but the interface is built to accept them now.

## Scope

**In scope**
- A pluggable `EncoderAdapter` interface and one concrete implementation wrapping a
  loaded PatchSet3D checkpoint's `.encoder`, exposing features at multiple stages.
- Target↔context matching metrics computed with **no transformer** (pure encoder
  features), so the numbers isolate encoder expressivity:
  - **Prototype cosine** (headline scalar) → AUROC + soft-Dice-at-best-threshold.
  - **FG-match margin** and **top-1 retrieval accuracy** (fine-grained).
- A **representation-tier sweep** (the "which encoder stage" question): per conv stage,
  multi-scale concat, post-standardization, optional post-`img_embed`, and an optional
  post-transformer upper-bound reference.
- A **resolution / pooling sweep** (orthogonal to tier): compute the same metrics at
  grid resolutions finer than `R` (up to native image resolution) to measure how
  avg-pooling to `R³` affects separability. Two modes — **dense** on an `R'³` grid for
  small `R'`, and **point-sampled** (N FG + N BG points, trilinear feature sampling) at
  native resolution where dense is too compute-heavy.
- A driver that loops eval tasks and emits a per-task table (class, object
  size/thickness bucket, per-(tier × resolution) metrics, and the model's real Dice
  column).

**Out of scope (phase 2)**
- Concrete `SAMAnatomicalAdapter` / `DINOAdapter` implementations — only the interface
  is prepared here.
- Correlation plots / statistical analysis of separability vs Dice vs geometry.
- Any training change (contrastive loss, regularizer). Diagnostic only.

## Background: where "features" live in PatchSet3D

Signal path (`src/models/patchset3d.py`):

- `ConvEncoder3D` (`:53`) — image-only (`in_ch=1`) multi-scale conv encoder. Builds a
  `feats` list (stem, then `n_down` stride-2 stages, `:83-85`); each scale is resampled
  to `R³` (default `R=16`) and concatenated on channels → `(B·T, Cf, R,R,R)`,
  `Cf = sum(enc_dims)`. Applied identically to target and all K contexts.
- `_grid_tokens` (`:145`) — flattens each `R³` grid to `N = R³` cell tokens.
- `_attn` (`:180`) — per-batch **support mean/std standardization** (`:186-189`), then
  `img_embed: Cf→e` + Fourier position, `mask_embed` for occupancy, then the dual-axis
  transformer does content-based matching. Query img-column → decoder → per-cell logit.

The encoder-expressivity question lives before the transformer: **do same-label cells
across target and contexts produce similar features?**

## Method

For each in-context task `(target image, K context images, K context masks, target GT)`:

### 1. Feature extraction
Each volume → per-cell feature grid `(C, R', R', R')` via an `EncoderAdapter`, at a
chosen representation **tier** and **resolution** `R'`. `R'=R=16` is the model's token
grid; larger `R'` (up to native) probes finer pooling. Each tier's native feature volume
is resampled to `R'³` with the encoder's own `_down_to` rule. Target and all K contexts
use the same encoder/tier/resolution.

For large `R'` (point-sampled mode, §3) features are not materialized on a full grid:
instead the tier's native feature volume is `grid_sample`-interpolated at N sampled point
coordinates → `(N, C)`.

### 2. Cell / point labeling
- **Dense (`R'³` grid):** downsample each context mask and target GT to `R'³` and
  threshold to FG/BG using the **same occupancy rule as `_occupancy`**
  (`patchset3d.py:151`): `_down_to` avg-pool then threshold at 0.5.
- **Point-sampled (native res):** the loader's native-res mask/GT gives exact labels —
  sample N FG points from mask==1 voxels and N BG points from mask==0 voxels (BG
  optionally restricted to a dilated band around the object; see Open questions). No
  pooling loss, so this is the cleanest FG/BG label.

Target GT labels are used only for scoring, never for the prototype.

### 3. Matching scores (transformer-free)
Let features be L2-normalized per cell. For a task with target cells `t_i` (with GT
labels) and context cells `c_j` (with mask labels):

- **Prototype cosine (headline).** `proto = normalize(mean over context-FG cells)`.
  Score `s_i = cos(t_i, proto)` for every target cell/point → soft pseudo-segmentation.
  Reduce over the target to:
  - **AUROC** of `s_i` separating target FG vs BG (both modes).
  - **soft-Dice at best threshold** (sweep thresholds, take max) — **dense mode only**, a
    feature-only UniverSeg/ProtoSeg segmenter directly comparable to the model's real
    Dice.
  - **average precision** of `s_i` — **point mode only**, replaces soft-Dice (which needs
    a full grid).
- **FG-match margin (fine).** For each target FG cell `t_i`:
  `margin_i = mean_j∈ctxFG cos(t_i, c_j) − mean_j∈ctxBG cos(t_i, c_j)`. Report the
  task-mean margin. >0 means FG cells look more like context FG than context BG.
- **Top-1 retrieval (fine).** For each target FG cell, its nearest context cell over all
  K contexts; retrieval@1 = fraction whose nearest neighbor is context-FG. (NN
  semantic-correspondence methodology, per "A Tale of Two Features".)

All three are pure functions of `(target_feats, target_labels, ctx_feats, ctx_labels)`
and are agnostic to whether rows are grid cells (dense) or sampled points.

### 4a. Representation-tier sweep ("which stage to use")
Every metric is reported per tier so we can see where matching signal emerges:

- **`stage:k`** — individual `ConvEncoder3D` feats (stem = `stage:0`, deeper stages
  `stage:1..n`), each resampled to `R³`. Low = texture, high = semantic.
- **`concat`** — the actual `encoder.forward` output (`Cf`-dim) the model consumes.
- **`concat_std`** — after the support mean/std standardization the transformer sees
  (`_attn:186-189`). Cosine is scale- but not shift-invariant, so raw vs standardized
  can differ; we report both.
- **`img_embed`** (optional) — `e`-dim learned projection, Fourier position **excluded**
  (content only).
- **`transformer_q`** (optional, upper-bound reference) — post-transformer query rep.
  Not "encoder," included as a ceiling on achievable separability. Only defined at
  `R'=R` (the transformer operates on the `R³` token grid).

### 4b. Resolution / pooling sweep ("how pooling affects similarity")
Orthogonal to tier: report every metric at several grid resolutions `R'`:

- `R'=R=16` — dense, the model's operating point (baseline).
- `R'∈{32, 48}` — dense, finer than the token grid.
- `R'=native` — **point-sampled** (dense would be ~2M cells/volume): N FG + N BG points,
  trilinear feature sampling, exact native-res labels.

Interpretation caveat: each tier has a *genuine* native resolution (stem = full 128³,
`stage:k` = 128/2ᵏ, `concat`'s finest genuine info = the stem's). Emitting a tier at an
`R'` finer than its native merely upsamples it — no new detail. So the resolution sweep
is most informative on the stem / early tiers; deeper tiers plateau once `R'` exceeds
their native stride. `run.py` records each tier's native resolution alongside the metrics
so upsampled points are distinguishable from genuine ones.

## Architecture / units

Mirror `experiments/3d/` conventions. Reuse `common.make_eval_loader(cfg, classes,
split=...)` for tasks and eval.py's PatchSet3D checkpoint loader (`eval.py:55-83`:
`torch.load` → rebuild arch from stored `ckpt["arch"]` → strip `_orig_mod.` → load).

```
experiments/3d/feature_sim/
  adapters.py   # EncoderAdapter interface + PatchSet3DEncoderAdapter
  metrics.py    # prototype_cosine, fg_match_margin, retrieval_at1 (pure functions)
  run.py        # Hydra entry: load ckpt, loop eval tasks, emit per-task table
```

### `adapters.py`
- `EncoderAdapter` (interface):
  - `tiers() -> list[str]` — the representation tiers this adapter can emit.
  - `native_res(tier) -> int` — the tier's genuine feature-grid side (for the
    upsampled-vs-genuine flag in the resolution sweep).
  - `features(volumes: Tensor[B,1,D,H,W], tier, res) -> Tensor[B,C,res,res,res]` —
    per-cell grid at `res³`, resampled with the encoder's own `_down_to` rule. Used by
    dense mode.
  - `sample_features(volumes, tier, coords: Tensor[B,N,3]) -> Tensor[B,N,C]` — trilinear
    `grid_sample` of the tier's native feature volume at normalized `coords` (point
    mode).
  - `R: int` property.
- `PatchSet3DEncoderAdapter(model, resolution)`:
  - Runs `ConvEncoder3D` capturing the per-stage `feats` list; resamples each to `R³`.
  - Builds `concat`; applies `_attn`'s standardization for `concat_std` (support =
    context volumes of the task, matching training).
  - `img_embed` / `transformer_q` tiers reuse the model's own layers.
- **Prepared for pretrained (phase 2):** the interface is encoder-agnostic — a future
  `SAMAnatomicalAdapter` / `DINOAdapter` emits native-res embeddings and resamples to
  `R³`. No such class is written now; `run.py` selects the adapter by name so adding one
  is local.

### `metrics.py`
Pure, framework-light functions operating on feature rows + labels (no I/O, no model);
rows are grid cells (dense) or sampled points — the functions don't care:
- `prototype_cosine(target_feats, target_labels, ctx_feats, ctx_labels, mode) -> dict`
  (`{auroc, soft_dice}` dense; `{auroc, ap}` point)
- `fg_match_margin(...) -> float`
- `retrieval_at1(...) -> float`
- `sample_points(mask, n_fg, n_bg, band=None) -> coords[N,3], labels[N]` — native-res
  FG/BG point sampler (used by point mode).
Each metric takes L2-normalized (or normalizes internally) `(N, C)` target and `(M, C)`
context arrays with `{0,1}` labels. Independently unit-testable on tiny synthetic inputs.

### `run.py`
Hydra entry (config under `configs/experiment/3d`, reusing eval config for loader/paths):
1. Load PatchSet3D checkpoint via the eval.py path; build `PatchSet3DEncoderAdapter`.
2. Also keep the full model to get the **real per-task Dice** (`model.predict`) for the
   table's reference column.
3. Loop the eval loader; for each task, sweep the configured **(tier × resolution)**
   grid. Each cell of the sweep runs dense or point mode (auto-selected by a cell-count
   budget, overridable) and computes the three metrics.
4. Emit one row per `(task, tier, resolution)`: `class`, object `size`/`thickness`
   bucket (reuse existing geometry extraction if available; else object voxel count from
   the target GT), `tier`, `res`, `mode`, `tier_native_res`, the metric set
   (`{auroc, soft_dice|ap, margin, retrieval@1}`), and `real_dice` (per-task, repeated).
   Write CSV (long/tidy form) to the run dir; optionally a wandb table (gated by
   `cfg.wandb.project`). N points, resolution list, and budget are Hydra config.

## Data flow

```
eval loader ── task {image, context_in, context_out, label(GT)} ──┐
                                                                   │
  for tier, res in sweep:                                          │
    dense:  adapter.features(vols, tier, res) ── (B,C,res³) ───────┤
            + grid_labels(masks, GT, res) ── res³ FG/BG            │
    point:  sample_points(mask/GT) ── coords,labels               │
            + adapter.sample_features(vols, tier, coords) ─(B,N,C)─┤
                                                                   ▼
                 metrics.{prototype_cosine, fg_match_margin, retrieval_at1}
                                                                   │
  model.predict(task) ── real_dice ────────────────────────────────┤
                                                                   ▼
                               per-(task,tier,res) row → CSV / wandb table
```

## Testing

Write tests only where logic is non-trivial (per repo guidelines), in
`tests/test_feature_sim_metrics.py`:
- **Separable synthetic case:** target/context FG features clustered apart from BG →
  `auroc≈1`, `margin>0`, `retrieval@1≈1`.
- **Random features:** `auroc≈0.5`, `margin≈0`, `retrieval@1≈FG-rate` (sanity floor).
- **grid_labels** occupancy thresholding matches `_occupancy` on a known small mask.
- **sample_points**: returns exactly `n_fg`/`n_bg` points with correct labels on a known
  mask; respects the BG band when set.
- **resolution invariance sanity:** on a separable synthetic volume, dense-at-`R'` and
  point-sampled metrics agree up to sampling noise (both ≈1 auroc).
Adapters/`run.py` are validated by a smoke run on one batch (documented, not a unit
test — needs a checkpoint).

## Open questions (deferred, not blocking)

- Object-geometry bucketing: reuse the existing thickness/geometry extraction (from the
  patchset3d-vs-medverse 3D work) vs. a simple voxel-count proxy — decide at
  implementation from what's already importable.
- Whether `concat_std` support statistics should be per-task (K contexts) or global —
  start per-task to match training.
- BG point sampling: uniform over all mask==0 voxels vs. restricted to a dilated band
  around the object (a harder, more informative near-boundary test). Start uniform;
  expose `band` as an option.
- Dense/point auto-switch budget (cells per volume) — start at a modest cap (e.g. 48³)
  and make it config-overridable.

## Log

Add a `docs/logs.md` entry when implemented (per repo guidelines).
