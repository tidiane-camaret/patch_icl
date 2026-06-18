# Multilevel resolution chain (16→32→64→128) — design

Date: 2026-06-17
Scope: `experiments/2d/multilevel/` (pipeline, train, model) + config. Extends the
current 2-level (stage-1 res-16 → stage-2 res-32) refinement into an N-level coarse-to-
fine chain with per-level weights.

## Goal

Chain multiple resolution levels, e.g. 16→32→64→128. Each hop reuses the **same**
sampling methodology (`sample_patches`: boundary core + fg quota + neighbor fill) and
has its **own set of weights**. The frozen res-16 `ImagePFN` (stage-1) seeds the chain;
each subsequent level is a `PatchSetPFN` that refines a sparse set of sampled cells at a
finer grid.

## Key decisions (from brainstorming)

1. **Training: detached per-level.** Each level trains to refine the DETACHED composite
   of the level below, with its own loss at its sampled cells. No gradient crosses a
   level boundary. All levels train in one run.
2. **Budget: per-level configurable.** `n_total`, `n_fg_core`, `n_fg_core_ctx` are
   per-level lists in config (coverage fraction shrinks with resolution; defer values to
   tuning).
3. **Thinking memory: chained.** Each level's `PatchSetPFN` emits post-transformer
   thinking rows that become the next level's memory (detached). Level-0 memory = the
   stage-1 thinking. All dims are `e=256`, so each level's `stage1_proj` is `e→e`.
4. **Code structure: `refine_level` hop unit + thin `run_chain` driver** (vs a monolith
   or an inline loop). The hop is the testable unit; the driver does the chain
   bookkeeping (upsample, detach, accumulate).

## Architecture

Resolution ladder: `sample.resolutions: [16, 32, 64, 128]`. `resolutions[0]` MUST equal
the frozen stage-1 resolution (assert at startup). The ladder yields `len-1` trained
hops, refining at grids `[32, 64, 128]`. Models = `nn.ModuleList([PatchSetPFN(...) per
hop])`; each hop has its own `mask_patch_size = image_size // grid` (4→2→1) and its own
`stage1_proj`; remaining hyperparams (e, h, l, a, …) shared by default, per-level
overridable later.

```
stage-1 (frozen, res-16)
  ├─ coarse_lr (B,16,16) ───────────────────────────► [dice_r16 metric]
  ├─ think_0 ────────────────┐
  └─ pred_16 ─┐              │
              ▼ upsample→32  ▼
   ┌───────────────────────────────────┐  ← encoder feats@32, GT@32
   │ hop0  PatchSetPFN_0  @grid=32      │     loss0 = BCE+Dice at M0 cells
   │  sample(prev_pred)→M0 → refine     │
   │  composite into upsampled map@32   │
   └──────┬───────────────┬─────────────┘
     refined_32       think_1
      (detach)         (detach)
            ▼ upsample→64  ▼
   ┌───────────────────────────────────┐  ← feats@64, GT@64
   │ hop1  PatchSetPFN_1  @grid=64      │     loss1
   └──────┬───────────────┬─────────────┘
     refined_64       think_2
            ▼ upsample→128 ▼
   ┌───────────────────────────────────┐  ← feats@128, GT@128
   │ hop2  PatchSetPFN_2  @grid=128     │     loss2
   └──────┬─────────────────────────────┘
     refined_128 = final output ──────────────────────► [dice/mean @128, checkpoint]
```

### The hop — `refine_level(...)`

Generalizes today's `build_patch_batch`. One `PatchSetPFN` level.

In:
- `model_L`, `encoder` features at this grid (target + K contexts),
- `coarse_grid` (B, grid²) — prev composite upsampled to this grid, **detached**; serves
  as both the sampling map (prev_pred) and the query prior,
- `prev_think` (B, n_think, e), detached,
- `gt_grid` (B, grid²) — GT pooled to this grid (true labels; for context sampling +
  loss + metrics),
- `level_cfg`: `n_total`, `n_fg_core`, `n_fg_core_ctx`, `tau`, `blur_sigma`, `floor`,
  `temperature`, `mask_prior`.

Steps (same as current single hop):
1. query sampling on `coarse_grid` (or `gt_grid` for the `ds_gt` oracle source),
2. context sampling on the true-GT mask fraction at this grid (`n_fg_core_ctx`),
3. gather features / scalar-or-tile priors / mask tiles,
4. `logits, this_think = model_L(..., stage1_think=prev_think, return_thinking=True)`,
5. composite: `refined = coarse_grid.clone(); refined[qidx] = sigmoid(logits)`.

Out: `refined_grid` (B, grid²), `logits` (B, M), `qry_gt` (B, M), `qidx` (B, M),
`this_think` (B, n_think, e).

### The driver — `run_chain(...)`

Seeds from stage-1 (`coarse_predict` stays as level-0): produces `coarse_lr` (res-16),
`coarse_grid` (res-16 pred), `think_0`. Then loops the ladder:

```
prev_dense, prev_think = stage1 seed (res-16 pred, think_0)
for L, grid in enumerate(resolutions[1:]):
    coarse_grid = upsample(prev_dense, grid)                # bilinear
    feats       = encode_grid(encoder, all_images, grid)
    gt_grid     = pool(label, grid)
    hop = refine_level(models[L], feats, coarse_grid, prev_think, gt_grid, level_cfg[L])
    outputs.append(hop)
    prev_dense, prev_think = hop.refined_grid.detach(), hop.this_think.detach()
return outputs, coarse_lr
```

The final hop's grid = native (128), so its composite is the final output with no extra
upsample.

### Model change — `PatchSetPFN.forward`

Add `return_thinking=False`. When true, also return the post-transformer thinking rows
(the first `n_think` rows of the transformer output) pooled over the 2 columns →
`(B, n_think, e)`. Level-0 think comes from stage-1 (unchanged path).

## Losses

Per-hop `lossₗ = BCE(logitsₗ, qry_gtₗ) + dice_weight · SoftDice(σ(logitsₗ), qry_gtₗ)`.
Total = `Σ wₗ · lossₗ` with per-level `loss_weights` (config; default equal). Detachment
makes each `lossₗ` train only `model_L`.

## Metrics (resolution-honest ladder)

Generalize the current metric block (a hard-coded r16/r32 pair) to loop over the ladder,
using the existing `K_R{res} = f"dice_r{res}/mean"` key construction:
- `dice_r16/mean` — stage-1 @16 (frozen baseline).
- `dice_r32/mean`, `dice_r64/mean` — hop0/hop1 composites at their grids.
- `dice_r128/mean` ≡ `dice/mean` — hop2 composite @128 (native; checkpoint metric).
- Per-hop diagnostics `refine/{hop}/{delta_err, dice_delta, soft_dice_delta, dice_s1,
  dice_s2, ...}` — each hop's composite (s2) vs its upsampled input (s1) on the hop's
  sampled cells.

Checkpoint selection stays on `dice/mean` (final level).

## Config

```yaml
sample:
  resolutions: [16, 32, 64, 128]   # resolutions[0] == stage-1 res (asserted)
  n_total:       [256, 256, 256]   # per hop (len == len(resolutions)-1)
  n_fg_core:     [64, 64, 64]
  n_fg_core_ctx: [160, 160, 160]
  # tau / blur_sigma / floor / temperature: scalars shared, per-level override optional
  train: prev_pred                 # per-hop sampling map source (prev_pred | ds_gt)
  eval:  prev_pred
train:
  loss_weights: [1.0, 1.0, 1.0]    # per hop
```

## File-by-file changes

- `experiments/2d/multilevel/pipeline.py` — add `refine_level` + `run_chain`; keep
  `coarse_predict` as the level-0 seed; move `build_patch_batch`'s logic into
  `refine_level` and **retire `build_patch_batch`** (callers use `run_chain`). The
  single-hop case is just `run_chain` with `resolutions=[16, 32]`.
- `src/models/patchset_pfn.py` — `forward(..., return_thinking=False)` returns pooled
  thinking when requested.
- `experiments/2d/multilevel/train.py` — `train_epoch`/`run_eval` call `run_chain`; sum
  weighted per-level losses; eval loops the dice ladder + per-hop deltas; build models as
  `nn.ModuleList`.
- `configs/experiment/2d/multilevel.yaml` — `sample.resolutions` + per-level budget /
  loss-weight lists.

## Backward compatibility

`resolutions: [16, 32]` reduces to exactly today's single-hop behavior — used to validate
the refactor produces unchanged results.

## Risks / notes

- **Encoder cost at high res:** features at res-128 = a 128×128 grid for B·(K+1) images —
  the main new compute/memory cost as the ladder deepens. The transformer input stays
  fixed-size per hop (M + K·M patches), which is the design's efficiency property.
- **prev_pred offset compounding:** the stage-1 prediction is under-confident with an
  inward-offset 0.5-band (per the value-distribution analysis); this offset feeds each
  hop's sampling map and could compound across 3 hops. Worth watching `dice_delta` per
  hop to confirm each level actually improves.
- **Unsampled cells inherit coarse values:** any cell not sampled at a level keeps the
  upsampled coarser prediction; the boundary-focused sampler assumes interiors are settled
  at coarse levels.

## Testing

- Unit-test `refine_level`: output shapes; composite overwrites exactly the sampled
  cells; unsampled cells equal the upsampled input.
- `run_chain` with `resolutions=[16, 32]` reproduces current single-hop outputs.
- Smoke-test the full 4-level ladder end-to-end on one dataset.
