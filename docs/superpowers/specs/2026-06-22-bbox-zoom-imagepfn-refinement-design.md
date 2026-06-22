# Bbox-zoom refinement with a shared ImagePFN architecture

Date: 2026-06-22
Experiment: `experiments/2d/multilevel` (new stage-2 variant, alongside the `PatchSetPFN` chain)

## Problem

The stage-2 refinement chain refines **scattered** patches selected by `sampling.py`
(boundary core + fg-core + neighbor fill). Because the selected cells are not a contiguous
region, the refiner cannot be the stage-1 `ImagePFN` (which assumes a dense N=res² image
grid with spatial positional embeddings and feature-axis spatial attention) — it must be a
separate `PatchSetPFN` arch (rows = scattered patches, Fourier PE on (i,j)).

That confounds any measured refinement gain with the **change of architecture**: we cannot
say whether multi-level refinement helps, or whether `PatchSetPFN`-on-scattered-patches
helps/hurts relative to `ImagePFN`.

## Goal

Make stage-2 use the **same `ImagePFN` architecture** as stage-1, so any measured gain is
attributable to *refinement-by-zoom* (+ explicit prior seeding), not to a different model
class. Achieved by sampling a **contiguous square bbox** instead of scattered cells: the
crop is itself a dense image grid, exactly what `ImagePFN` consumes.

## Decisions (from brainstorming)

- **Structure**: a **chained zoom ladder** — each hop crops tighter and re-segments with
  its own `ImagePFN`, mirroring the existing resolution-chain structure (frozen stage-1;
  per-hop independent training; detached chaining between hops). **Start with a single
  64px hop** (`crop_sizes: [64]`); the `ModuleList` + `run_zoom_chain` loop generalize to
  more hops later with no code change.
- **Bbox policy**: **fixed crop-size schedule** `crop_sizes` (pixels, in the 128-image
  frame), one entry per hop. Uniform zoom factor across samples → fixed-size grids → simple
  compositing.
- **Centering**: target crop centered on the **max-sum window** of the current prediction
  (the s×s square with the largest summed predicted probability — the "largest pred
  values"); each context crop centered on the **max-sum window of that context's GT**
  (densest-GT square; generalizes "GT centroid"). Centers clamped so crops stay in-bounds.
- **Hop init**: each hop's `ImagePFN` is **warm-started from the frozen stage-1 weights**
  (identical arch → `state_dict` loads cleanly), then fine-tuned independently.
- **Query prior**: **seed with the coarse prediction** — the query's mask columns are
  filled with the previous level's prediction cropped+resized to the bbox (not the
  TargetEncoder context-mean). The model corrects an explicit prior (the `ImagePFN` analog
  of `PatchSetPFN`'s `mask_prior=patch`).
- **Features**: **encode once, crop-pool per hop** — the frozen encoder runs once on the
  full 128 images (`encode_maps`); each hop crops+resamples those resolution-independent
  maps to the bbox. No re-encoding of crops. This is both cheaper *and* the clean control:
  the zoom chain consumes the **same** encode-once features the `PatchSetPFN` chain does, so
  the only difference between the two is the architecture.

## Architecture

### New module: `experiments/2d/multilevel/bbox.py` (pure tensor ops, unit-tested)

- `max_sum_window(prob, s) -> centers`: for each `(B,)` prob map on the 128 grid, the s×s
  square (integer top-left, or center) maximizing summed probability. Implemented via a
  box-sum (cumulative-sum / `F.avg_pool2d` with stride 1 over an s×s window) + argmax.
  Centers clamped so `[c-s/2, c+s/2)` stays within `[0, 128)`.
- `gt_window(mask, s) -> centers`: same on a binary GT mask (densest-GT square). Used for
  context crops.
- `crop_resize(x, centers, s, out, mode) -> y`: batched crop of an `(N,C,H,W)` tensor to
  the per-sample s×s bbox, resampled to `out×out`. Implemented with `F.grid_sample` over
  the normalized bbox so it is vectorized across the batch and resolution-agnostic.
- `crop_pool_maps(maps, centers, s, out) -> feat`: applies `crop_resize` to each encoder
  stage map and concatenates → `(N, feature_dim, out, out)`. The zoom analog of
  `pipeline.pool_grid` / `encoder.pool_maps`. NB: `grid_sample` is bilinear point
  resampling rather than area-averaging `adaptive_avg_pool`; acceptable because crops are
  small regions of a modest-resolution map (the per-stage native res is usually ≤ the crop
  in feature pixels, so it is upsampling, not aliasing) — consistent with how the existing
  chain already upsamples maps to grids 64/128 beyond their native size.
- `composite_window(full, patch, centers, s) -> full'`: writes an s×s `patch` (already
  upsampled from the hop's 16×16 output) back into a clone of the `(B,1,128,128)` `full`
  map at the per-sample bbox. Returns a new tensor (input not mutated).

### `ImagePFN` changes (`src/models/pfn_seg_2d.py`) — minimal, default-off

1. Constructor: `use_external_features: bool = False`. When `True` (with `feature_dim`
   given and `image_encoder=None`), build `image_embed = Linear(feature_dim, e)` and **no**
   internal encoder submodule. `forward(image_feats=...)` then consumes precomputed
   `(B,T,N,Cf)` features (still `standardize_by_context` + `image_embed`), skipping
   encoding. Defaults reproduce today's behavior exactly.
2. `forward(..., image_feats=None, seed_query_mask=False)`:
   - `image_feats` given → use it for the image path (bypass encoder).
   - `seed_query_mask=True` → keep the query rows' mask columns as passed (the cropped
     coarse prediction) instead of the TargetEncoder context-mean overwrite.

   Both default to `None`/`False`, so stage-1 and the existing `pfn_seg`/multilevel paths
   are byte-for-byte unchanged.

### New pipeline: `experiments/2d/multilevel/zoom_pipeline.py`

`run_zoom_chain(batch, stage1, encoder, models, cfg, source, stochastic, device)`:
returns `(outputs, coarse_lr)` with the **same output schema** as `pipeline.run_chain`'s
hops (each `o` carries `o["logits"]`, `o["qry_gt"]`, plus zoom-specific `o["bbox"]`,
`o["refined_full"]`) so the train loss loop is shared.

1. Build `all_images = [context_in, query]`, `all_masks = [context_out, 0]` (as `run_chain`).
2. `coarse_lr = stage1(...)` at R0 (frozen); `pred = upsample(coarse_lr → 128)`.
3. `maps = encoder.encode_maps(all_images)` once.
4. For each hop `L` with `s = crop_sizes[L]`:
   - `tgt_center = max_sum_window(pred, s)`; `ctx_centers = gt_window(context_out[:,k], s)`.
   - `image_feats = standardize_by_context(crop_pool_maps(maps, centers, s, out=16), K)`
     stacked over the K context + 1 target rows.
   - mask images: `crop_resize(context_out, ctx_centers, s, 128, nearest)` for context;
     `query prior = crop_resize(pred, tgt_center, s, 128, bilinear)` placed in the query row.
   - `logits16 = models[L](image_feats=..., masks=cropped_masks, sep=K,
     seed_query_mask=True)` → `(B,16,16)`.
   - target: `gt_crop = crop_resize(label, tgt_center, s, 16, bilinear)` (soft, avg of GT
     in the bbox) → `o["qry_gt"]`.
   - composite: `patch = upsample(sigmoid(logits16) → s×s)`;
     `pred = composite_window(pred, patch, tgt_center, s).detach()`.
   - record `o`.
5. Return.

The hop ladder always crops from the **original** 128 image; later hops are progressively
tighter windows. The previous composite supplies only (a) the target center and (b) the
query prior, and is **detached** before the next hop (independent per-hop training).

### Training integration: `experiments/2d/multilevel/train.py` (config switch)

A new key `arch.refine_arch: patchset | imagepfn_zoom` (default `patchset`). When
`patchset`, the existing code path runs unchanged. When `imagepfn_zoom`:

- **Model build**: `model = nn.ModuleList([ImagePFN(use_external_features=True,
  feature_dim=Cf, resolution=R0, image_size=H, input_patch_size=stage1.input_patch_size,
  e/h/l/a/thinking_rows/residual_decay from stage1's arch) for _ in crop_sizes])`. Each hop
  `load_state_dict(stage1_state_filtered_of("image_encoder."), strict=False)` to warm-start.
- **Chain call**: `run_zoom_chain` replaces `run_chain` in `train_epoch` and `run_eval`.
- **Loss**: reuse `patch_loss` (BCE + `dice_weight·soft_dice`) per hop on the dense 16×16
  crop grid; per-hop `loss_weights` sum. (Same loss loop as today, since `o` carries
  `logits`/`qry_gt`.)
- **Reused unchanged**: `load_stage1`, `augment`, Muon/AdamW + cosine scheduler, LAWA,
  wandb, checkpoint save (the saved dict already embeds `arch`/`sample`/`data`).

`muon_params`/`adam_params` split (currently keyed on `"transformer" in name`) still works:
`ImagePFN`'s transformer matrices live under `transformer.*`, so they route to Muon as
before.

### New config: `configs/experiment/2d/multilevel_zoom.yaml`

```yaml
defaults: [train_base, _self_]
model: imagepfn_zoom
arch:
  refine_arch: imagepfn_zoom
  ctx_center: gt_window      # gt_window | centroid (context crop centering)
sample:
  crop_sizes: [64]           # one per hop, pixels in the 128 frame; len = #hops (start: single 64px hop)
train:
  loss_weights: [1.0]        # len == len(crop_sizes)
  stage1_checkpoint: results/2d/pfn_seg_universeg/pfn_seg_USegall_R16q8_e256_l6_k3_think8/best.pt
eval:
  max_per_label: null
```

`multilevel.yaml` (the PatchSetPFN path) is untouched.

## Metrics (zoom eval, `run_eval` zoom branch)

The composite is always at native 128, so:

- `dice/mean`, `dice_soft/mean` — hard / soft Dice of the final composite vs GT @128.
  `dice_soft/mean` remains the **checkpoint-selection** metric (consistent with today).
- `dice_after_hop{L}/mean` — hard Dice of the composite after hop `L` (baseline = stage-1
  composite), so the per-hop improvement curve is readable.
- `refine/hop{L}/dice_delta` — hop `L`'s hard-Dice improvement **inside its own bbox**,
  refined-vs-prior, isolating what the hop added on the region it touched.
- `val/loss` — same per-hop weighted `patch_loss`, mirroring the existing scripts.
- Per-dataset breakdowns reuse the existing `dice/dataset_*` aggregation.

## Edge cases

- **Out-of-bounds**: centers clamped so every crop lies within `[0,128)`.
- **Empty prediction** (max window prob < ε): skip the hop (leave `pred` unchanged for
  those samples) so background-only targets don't get spurious refinement.
- **Object larger than the crop**: only the in-bbox part is refined; choose
  `crop_sizes[0] ≥ 64` to cover typical objects. Outside the bbox the composite keeps the
  previous level's value (by construction).

## Testing

- `bbox.py` unit tests: `max_sum_window` on hand-built maps (single blob, off-center blob,
  border blob → clamped); `crop_resize`/`composite_window` round-trip (crop then composite
  recovers the region); shape/in-bounds invariants.
- `run_zoom_chain` shape/smoke test with a **stub encoder** exposing `encode_maps` (mirrors
  the existing fallback-encoder test), asserting per-hop output schema and final composite
  shape `(B,1,128,128)`.
- `ImagePFN` regression: with defaults (`use_external_features=False`,
  `seed_query_mask=False`, `image_feats=None`) output is unchanged vs current; with
  `use_external_features=True` + `image_feats` it produces the right `(B,res,res)` logits.
- Smoke-run `train.py refine_arch=imagepfn_zoom` for 1 epoch on a tiny subset to confirm
  warm-start loads, the loss decreases, and checkpointing writes.

## Out of scope (YAGNI)

- Adaptive / per-sample bbox sizes (decided against; fixed schedule chosen).
- Re-encoding crops for finer features (decided against; encode-once is the clean control).
- Stage-1 thinking-memory injection into the zoom hops (`ImagePFN` carries its own thinking
  rows; chaining is via the detached composite + query prior).
- Sharing one `ImagePFN` across hops (warm-start-per-hop chosen).
```
