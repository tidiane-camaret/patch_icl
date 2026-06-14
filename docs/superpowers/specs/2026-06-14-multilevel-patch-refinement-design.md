# Multilevel patch refinement — design

**Date:** 2026-06-14
**Status:** Draft for review
**Location:** `experiments/2d/multilevel/`, `src/models/patchset_pfn.py`, `configs/experiment/2d/multilevel.yaml`

## Goal

Coarse-to-fine 2D in-context segmentation. A frozen stage-1 model predicts at
resolution 16. We then resample the **uncertain** region at resolution 32 and train
a **second** model (a patch-set in-context transformer) to refine those patches.
Success = measurable error reduction on the resampled uncertain region versus the
stage-1 coarse prediction.

Motivation is established by `experiments/2d/patch_error_drivers.py`: on the existing
res-16 model, selecting the top ~20–25% of patches by the **inference-observable**
signal `|pred − 0.5|` captures ~88–92% of total error mass. At resolution 32
(1024 cells), 256 patches ≈ 25%.

## Key decisions (from brainstorming)

1. **Stage-2 is a patch-set in-context model** (TabPFN / nanoTabPFN-shaped), not the
   grid-aligned `ImagePFN`. Rows = sampled patches; cols = `[img-token | mask-token]`.
2. **Patch budget: 256 per image** = **192 closest-to-0.5** + **64 most-certain**,
   applied to *every* image. Target patches (ranked by coarse pred) → **queries**;
   context patches (ranked by their true mask fraction) → **support**. For K context
   images the sequence has `256·(K+1)` patches: 256 queries + `256·K` support.
3. **Certain target patches are queries, not support** — refined and used as a
   regression check (refinement must not degrade already-confident cells).
4. **Patch representation = frozen UniverSeg encoder features** at the res-32 grid
   (`UniverSegFeatureEncoder`, `level=all` → 256-d per cell). Reuses the encoder added
   on 2026-06-14.
5. **Coarse prior is a param** (`coarse_prior`): if true, each query's mask-token is
   the stage-1 coarse prediction (true refinement); if false, a neutral
   support-mean prior (re-prediction).
6. **Positional encoding = 2-D Fourier features (additive), resolution-generalizable.**

## Pipeline (per task: 1 target + K context images)

Order matters — coarse pred and features must be computed on the *augmented* images.

1. **Augment** whole images (reuse `pfn_seg.augment`: geometric on context pairs,
   intensity per image).
2. **Coarse (stage-1)**: frozen res-16 `ImagePFN` (existing checkpoint) →
   target `16×16` logits → sigmoid → upsample to `32×32` = `coarse32` (B, 32, 32).
3. **Features**: frozen `UniverSegFeatureEncoder(out_size=32)` on all `K+1` images →
   `(B, K+1, 256, 32, 32)`; flatten grid → `(B, K+1, 1024, 256)`.
4. **Sample** per image on the `32×32` grid (`R2=32`, 1024 cells):
   - rank cells by `|v − 0.5|`; take the **192 smallest** (uncertain) + **64 largest**
     (certain) → 256 cell indices.
   - target image: `v = coarse32`. context image k: `v = avgpool(mask_k, 32)`.
   - target sampled cells → queries; context sampled cells → support.
5. **Refine (stage-2)**: `PatchSetPFN` predicts the 256 query patch labels.

## `PatchSetPFN` (`src/models/patchset_pfn.py`)

Reuses `ImagePFN`'s `TransformerEncoderLayer` (dual-axis block), `Muon`, LAWA, and
`soft_dice_loss` unchanged. New, focused model file.

**Tokens.** Per patch: `img_tok = img_embed(feature_256)` and
`mask_tok = mask_embed(label_scalar)`, each `→ e`. Two cols per patch:
`[img_tok | mask_tok]`.

**Mask-token value.**
- support (context) patch: true res-32 mask fraction (soft, in [0,1]).
- query (target) patch: `coarse32` at the cell if `coarse_prior=true`, else the mean
  of support mask fractions (nanoTabPFN TargetEncoder analog).

**Feature normalization.** Per-channel standardization of the 256-d encoder features
using **support-patch** statistics (mirrors the encoder path in `ImagePFN`).

**Positional encoding (2-D Fourier, additive, generalizable).** For a patch at grid
cell `(i,j)` on an `R2×R2` grid, normalize `u=(i+0.5)/R2`, `v=(j+0.5)/R2 ∈ (0,1)`.
With `fourier_bands` geometric frequencies `f_b` (e.g. `2^0…2^(L-1)`), build
`[sin(2π f_b u), cos(2π f_b u), sin(2π f_b v), cos(2π f_b v)]_b` (dim `4·L`),
project `Linear(4·L → e)`, and **add to both img and mask tokens**. Shared coordinate
system across all images, so a target query and a context support patch at the same
`(i,j)` share a location code. Resolution-agnostic: normalized coords + fixed
frequencies mean a model trained at `R2=32` applies at other grid resolutions.

**Sequence + attention.** Rows = `[thinking | support (256·K) | query (256)]`;
cols = 2. Reuse the dual-axis block: feature-axis attention mixes a patch's img/mask
cols; sample-axis attention has queries attend to **thinking + support only**
(`sep = n_think + 256·K`), read-only, exactly as `ImagePFN`.

**Decoder.** `Linear(e,h) → GELU → Linear(h,1)` on each query's img-col →
per-query logit. Scatter to a `32×32` map at the queried target cells.

## Training (`experiments/2d/multilevel/train.py`)

- Data: reuse `MedSegBenchDataset` at `image_size=128` (res-32 grid → native P=4;
  encoder pools to 32), `context_size=K`. Same loader/collate as `pfn_seg.py`.
- Stage-1 res-16 `ImagePFN` loaded from `train.stage1_checkpoint`, **frozen**, run
  under `no_grad` per batch. Encoder frozen too.
- Optimizers: `Muon` (transformer 2-D weights) + AdamW (rest); only `requires_grad`
  params (excludes frozen stage-1 + encoder). LAWA checkpoint averaging.
- Loss: `BCE_with_logits + dice_weight · soft_dice_loss` over the **256 query patches**
  (192 uncertain + 64 certain), target = res-32 mask fraction at the queried cells.
- **Refactor:** factor `Muon`, `augment`, `lawa_average`, `soft_dice_loss` out of
  `experiments/2d/pfn_seg.py` into a shared util (e.g. `experiments/2d/pfn_train.py`)
  imported by both scripts. No behavior change to `pfn_seg.py`.

## Metrics (success criterion)

On the val split, on the **sampled target region**, compare stage-2 vs the stage-1
coarse value against res-32 GT (`gt32 = avgpool(label, 32)`):

- **Primary:** mean `|error|` reduction on the **192 uncertain** target patches:
  `Δ = mean|coarse − gt32| − mean|stage2 − gt32|` (want `Δ > 0`).
- Low-res hard/soft Dice on the uncertain region: stage-2 vs coarse.
- **64 certain** target patches: same metrics as a **regression check** — refinement
  must not worsen them.
- Optional whole-map view: composite stage-2 predictions into `coarse32` at the
  queried cells and report full res-32 Dice vs the coarse-only map.

Logged to W&B mirroring existing `dice*/mean` + per-dataset breakdowns.

## Config (`configs/experiment/2d/multilevel.yaml`)

```
model: patchset_pfn
data:   { image_size: 128, context_size: 3, dataset: null }
sample: { grid_res: 32, n_uncertain: 192, n_certain: 64 }
arch:   { e: 256, h: 512, l: 6, a: 4, thinking_rows: 8, residual_decay: 0.95,
          feature_level: all, coarse_prior: true, fourier_bands: 8, compile: true }
train:  { epochs, batch_size, lr, muon_*, adam_wd, warmup_epochs, dice_weight,
          grad_clip, lawa_k, eval_every, workers, seed,
          stage1_checkpoint: results/2d/pfn_seg_low_res_loss/.../best.pt }
eval:   { batch_size, workers, max_per_label, out_dir: results/2d/multilevel }
wandb:  { project, name: null, enabled: true }
```

## Assumptions / defaults (flag on review)

- Stage-1 = the existing res-16 `pfn_seg` checkpoint, frozen, on-the-fly per batch.
- Support labels are **soft** mask fractions; ranking ties broken arbitrarily.
- Including 64 certain queries (25%) prevents a degenerate "always-0.5" refiner by
  keeping the query distribution balanced.
- `compile=true` will graph-break at the frozen encoder (as in `pfn_seg`).

## Out of scope (YAGNI)

- Joint/end-to-end training of stage-1 and stage-2.
- More than two levels (16→32 only).
- RoPE / learned-table PE (2-D Fourier chosen for resolution generalization).
- Iterative re-sampling within stage-2.
