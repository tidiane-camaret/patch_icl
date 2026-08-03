# PrimusEncoder native-grid mode (honor image_size) — design

**Date:** 2026-08-03
**Status:** approved

## Problem

Running `experiments/3d/train.py experiment=30_colipri_encoder data.image_size=[128,128,128]`
logs the same GFLOPs as the 192³ default. The frozen CoLiPri (Primus-M) encoder
resamples every input to a fixed `input_shape=192³` in `PrimusEncoder._preprocess`
(`src/models/primus_encoder.py:158-159`) before the ViT, so the token grid is always
`192/8 = 24³` regardless of `image_size`. Compute is therefore pinned by `input_shape`
and `arch.resolution`, not `image_size`.

Two things force the fixed grid today:
1. Primus builds its EVA RoPE with a fixed `feat_shape=ref_feat_shape=(24,24,24)`
   (`primus.py:101`), so `RotaryEmbeddingCat` caches a fixed 24³ table and sets
   `bands=None`.
2. `Eva._pos_embed` calls `rope.get_embed()` with no shape (`eva.py:186`); the
   shape-aware path is gated behind `dynamic_img_size`, which is `NotImplementedError`.

Neither is a RoPE limitation — RoPE is length-generalizable. `rope.update_feat_shape(grid)`
rebuilds the table for any grid from scratch (no bands needed), confirmed in
`timm/layers/pos_embed_sincos.py:583-592`.

## Goal

Compute/FLOPs scaling study: a knob where smaller `image_size` genuinely means a
smaller ViT token grid → fewer encoder FLOPs, to study the accuracy/compute tradeoff
of the frozen Primus encoder.

## Scope decision

**Only the ViT encoder honors `image_size`.** Downstream is unchanged: after the ViT,
`_down_to(f, self.resolution)` resamples features back to `resolution³` (=24³) exactly
as today, so the patchset transformer and mask decode are untouched. This isolates the
encoder-side effect: how much do compute and accuracy change when the frozen encoder
tokenizes at a coarser grid, with the head held fixed.

Explicitly out of scope: coupling `arch.resolution` to `image_size` (transformer would
also shrink). Can be a follow-up.

## Design

Opt-in flag, default = current fixed-192 behavior, so existing experiments and the
feature-sim path (`experiments/3d/feature_sim/adapters.py`) are unchanged.

### 1. `_preprocess` — stop forcing 192³ (native-grid mode only)
- Intensity renormalization (HU clamp → /1000) is unchanged.
- Replace the unconditional `interpolate(..., input_shape)` with: feed the input at its
  native spatial size when each dim is divisible by the patch size (8); otherwise
  resample to the nearest multiple of 8 (safety net, with a one-time warning).
- Result: `image_size=[128,128,128]` → fed as 128³ → down_projection (stride 8) → 16³
  token grid.

### 2. `_encode` — rebuild RoPE for the actual grid (native-grid mode only)
- Determine the token grid from the down_projection output.
- If it differs from the rope's current `feat_shape`, set
  `rope.ref_feat_shape = grid` then call `rope.update_feat_shape(grid)` before
  `p.eva(x)`. Caches when the grid is stable across a run.

### 3. RoPE reference frame — identity
- Use `ref_feat_shape = feat_shape = grid`. A 16³ grid gets integer positions `0..15`,
  so adjacent tokens are distance 1 apart — the exact local rotary frequency the encoder
  trained on. The smaller grid is a sub-block of the training positional field;
  extrapolation-free. (The rejected alternative, `ref=24`, produces fractional stretched
  positions never seen in training.)

### 4. Wiring
- `PrimusEncoder.__init__`: add `native_grid: bool = False`; store `patch_size` from
  `primus_kwargs["patch_embed_size"]`.
- `src/models/patchset3d.py`: thread `arch.encoder_native_grid` (default `false`) into
  the `PrimusEncoder(...)` construction (~line 130).
- No new experiment file. `encoder_native_grid` defaults to `false` in the model
  schema (`configs/experiment/3d/model/patchset3d.yaml`); the study is run via CLI
  overrides, e.g. `arch.encoder_native_grid=true data.image_size=[128,128,128]` on top
  of `experiment=30_colipri_encoder`. The 192³ baseline stays reproducible unchanged.
- `docs/logs.md`: log the change.

## Expected result

At `image_size=128`, ViT encoder FLOPs ≈ `(16/24)³ ≈ 0.30×` (~70% fewer *encoder*
FLOPs); transformer/decode unchanged. Logged GFLOPs move with `image_size`. Whether the
frozen features stay meaningful at a coarser grid is the empirical question the study
answers.

## Verification

- Unit-ish test: with `native_grid=True`, encoder ViT output (pre-`_down_to`) is
  `(image_size/8)³`; `measure_flops` at 128³ < at 192³.
- With the flag off, encoder output and FLOPs are unchanged (feature-sim / eval paths
  intact).
- `image_size` must be divisible by 8 in native-grid mode (assert / warn + resample).

## Files touched

- `src/models/primus_encoder.py`
- `src/models/patchset3d.py`
- `configs/experiment/3d/model/patchset3d.yaml` (add `arch.encoder_native_grid: false`)
- `docs/logs.md`
