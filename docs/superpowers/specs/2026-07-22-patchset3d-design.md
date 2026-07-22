# PatchSet3D — design

**Date:** 2026-07-22
**Status:** approved (design), pending implementation plan

## Goal

Extend the 2D `PatchSetCNN` (set-of-patches in-context segmentation) to 3D CT
volumes, as a new model `PatchSet3D`. The class name is deliberately
encoder-agnostic — the encoding method may change later, so nothing in the name
ties it to the conv encoder.

Given a target volume and K context (image, mask) pairs for the same class, the
model predicts a low-resolution binary mask over an R³ token grid, tile-decoded to
(R·d)³, upsampled to native resolution for evaluation.

## Scope

**In scope (v1):**
- Single-level dense model at token grid R (default R=16, config-driven).
- Decoder tiling: each query token decodes a d³ block → logit at (R·d)³.
- Direct-call interface + grid-resolution loss (GT pooled DOWN to the logit grid).
- Grid-resolution monitoring metrics: `dice_ds`, `dice_ds_soft`, `cossim`.
- Harness integration into `experiments/3d/train.py` + `evaluate.py`.

**Out of scope (deferred):**
- Multi-level refine (`refine_mode`, `resolutions`, bbox/scatter paths).
- `sim_prior` (max-cosine similarity query prior).
- Muon optimizer + LAWA checkpoint averaging (v1 uses Adam/AdamW only).
- top-k grid metric (noisiest of the four; omitted).

## Reuse boundary

| Piece | Location | Treatment |
|---|---|---|
| Dual-axis transformer core (`TransformerEncoderStack`, `TransformerEncoderLayer`, `ThinkingRows`, `batched_sdpa`, `LowerPrecisionRMSNorm`) | `src/models/pfn_seg_2d.py` | **Reused unchanged.** Operates on `(B, rows, cols, e)` — dimension-free; space is a feature, not a tensor axis. |
| `FourierPositionalEncoding` | `src/models/patchset_pfn.py` | **Generalized in place.** Add `n_axes: int = 2`; `proj = Linear(2 * n_axes * num_bands, e)`. Default `n_axes=2` keeps every 2D caller byte-identical. `forward(ij, grid_res)` already flattens over the axis dim, so passing an `(...,3)` `ijk` with `n_axes=3` works. |
| `ConvEncoder3D` | new `src/models/patchset3d.py` | 3D port of `ConvEncoder`: `Conv3d`, `GroupNorm`, area-pool (down) / trilinear (up) resample of each scale to R³, channel concat. |
| `PatchSet3D` | new `src/models/patchset3d.py` | Single-level model. |

Rationale for a new class over parametrizing `PatchSetCNN`: the valuable, tuned
part (the transformer) is already shared, so a new class duplicates only
mechanical code and carries zero regression risk to the actively-tuned 2D model.
Matches the repo's existing per-dimension grain (`pfn_seg_2d.py`, `experiments/2d`
vs `experiments/3d`).

## `PatchSet3D` forward (data flow)

Interface mirrors 2D:

```
forward(image (B,1,D,H,W), context_in (B,K,1,D,H,W), context_out (B,K,1,D,H,W),
        mode="train") -> {"final_logit": (B,1,Rd,Rd,Rd)}     # Rd = R * d
```

Steps (each the 3D analog of the 2D method it mirrors):

1. **Stack** T=K+1 volumes → `(B·T,1,D,H,W)`.
2. **`ConvEncoder3D`** → `(B·T, Cf, R, R, R)`. R = `arch.resolution` (default 16).
3. **`_grid_tokens`**: flatten spatial → N=R³ tokens/image → support `(B, K·N, Cf)`,
   query `(B, N, Cf)` (image-major, row-major cells). `flatten(2)` is dim-agnostic.
4. **`_occupancy`**: `adaptive_avg_pool3d` context masks → scalar occupancy (p=1) or
   a p³ tile via a 3D `_mask_tiles` (reshape + trilinear resize). `mask_patch_size=p`.
5. **`_tokens`**: `img_embed(feat) + FourierPos(ijk, R)` and
   `mask_embed(occ) + FourierPos(ijk, R)`, stacked as `[img | mask]` columns. Query
   mask-token = support-mean prior. `ij_base` becomes an `ijk` lattice from three
   `arange`s (row-major over R³).
6. **Attention core** — reused unchanged: z-score by support-patch stats, thinking
   rows, optional `context_id_embed`, dual-axis transformer.
7. **Decoder**: `Linear(e→h)→GELU→Linear(h→d³)` per query token → tile the d³ block
   back into `(B,1,Rd,Rd,Rd)` (inverse of 3D `_mask_tiles`). d=1 → one logit/token.

### Config surface (all driven from `arch:`)

`resolution` (R), `mask_patch_decode_size` (d), `enc_dims`, `e`, `h`, `l`, `a`,
`thinking_rows`, `residual_decay`, `fourier_bands`, `mask_patch_size` (p),
`context_id_embed`, `max_context`, `full_attn`, `query_self_attn`.

### Memory note

At R=16, K=4 the sample-axis set is ~20k tokens (K·R³) — heavy. R is the dial: R=8
→ 512 tokens/image. Design targets dense R=16; R stays config-driven so it can be
lowered if it OOMs at the target batch/K.

## Harness integration

Port the 2D unified-trainer philosophy (grid-res loss, direct call) into the 3D
harness rather than the Medverse `.model`/`.train_forward`/`.predict` wrapper
convention. Medverse's path stays byte-identical.

### `experiments/3d/train.py`

- `build_model`: add `model == "patchset3d"` →
  `PatchSet3D(image_size=cfg.data.image_size, **arch)`, returned as a plain
  `nn.Module`. A flag (e.g. `is_patchset = model_name == "patchset3d"`) selects the
  call convention: Medverse keeps `model.model` / `train_forward`; patchset is called
  directly.
- `train_epoch`: for the direct-call model,
  `out = model(img, context_in=cin, context_out=cout, mode="train")`;
  `logit = out["final_logit"]`; a new `_target_like_3d(lbl, logit)` `adaptive_avg_pool3d`s
  GT down to (Rd,Rd,Rd). Loss (`bce_dice` / smooth-L1) computed at grid res — no 128³
  upsample.
- Optimizer/scheduler: reuse existing `build_optimizer` / `build_scheduler`
  (Adam/AdamW + plateau/cosine). No Muon/LAWA in v1.

### `experiments/3d/evaluate.py`

- `PatchSet3D.predict(target, ctx_in, ctx_out)`: run forward, trilinearly upsample the
  grid-res prob to native (D,H,W), threshold at 0.5 → `(B,D,H,W)`. Keeps
  `evaluate_classes` / `validate_mean` working unchanged (eval Dice stays native,
  comparable to Medverse).
- `logits_fn` for the val soft-Dice path: a method returning native-upsampled logits,
  so `soft_dice_binary` is computed against native GT as today.

### Config

New `configs/experiment/3d/patchset3d_train.yaml` with an `arch:` block, mirroring the
2D `patchset_cnn_train` config. Selected via `model=patchset3d` (or as the config's
default `model:` key).

## Grid-resolution metrics

Computed against GT pooled to (Rd,Rd,Rd), in both `train_epoch` and the val step,
logged with the grid-size tag (`@Rd`). 3D ports of the 2D helpers
(`experiments/2d/train.py:86-137`), which are flat-tensor reductions (`flatten(1)`) and
so port with a one-line signature change:

- `dice_ds@Rd` — hard Dice at grid res (pred≥0.5 vs GT>0). Port of `_hard_sum`.
- `dice_ds_soft@Rd` — threshold-free soft Dice. Port of `_soft_sum`.
- `cossim@Rd` — per-sample scale-invariant cosine of (prob, occupancy). Port of
  `_cos_sum`; adds signal at low res where soft-Dice collapses toward mean occupancy.

Logged alongside the existing native `val/dice` (from `.predict`). Best-checkpoint
selection stays on native `val/dice`.

## Testing

Minimal, per repo guideline ("tests only when necessary"):

- `test_patchset3d.py`:
  - forward on a tiny synthetic batch (B=2, K=2, D=H=W=32, R=8, d=2) asserts
    `final_logit` shape `(2,1,16,16,16)`;
  - `predict` returns native `(2,32,32,32)`;
  - loss + backward runs (grads non-None on the encoder and transformer);
  - `FourierPositionalEncoding(n_axes=2)` output unchanged vs current (guards the
    in-place generalization).
- One real smoke run: `python experiments/3d/train.py model=patchset3d
  train.epochs=1 data.max_train_subjects=<small>` to confirm end-to-end wiring.

## Docs

Log the change in `docs/logs.md` per CLAUDE.md.

## Open questions / future work

- Whether dense R=16 fits at a useful batch/K on the target GPU, or R must drop.
- Refine (bbox / scatter) as a follow-up — in 3D the R³ budget may make
  scatter-sampling the primary path rather than an optional second level.
- Muon/LAWA parity with the 2D trainer, if it proves beneficial.
- Swapping the encoder (the reason the class is named `PatchSet3D`, not
  `PatchSetCNN3D`).
