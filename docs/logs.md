# Change log

## 2026-05-24 — MultilevelICL benchmark adapter

Added `src/benchmark_models/multilevel.py` — `MultilevelICLAdapter` wrapping `MultilevelICL` for use in `scripts/eval.py`.

- Loads a multilevel checkpoint and reconstructs both the frozen `STUNetEncoder` and `MultilevelICL` from the stored config.
- Runs N-level coarse-to-fine inference: L0 is a dense forward over the full grid; finer levels sample `NP` sparse patches guided by the previous level's upsampled prediction (`predicted_entropy` at eval time — no GT available for error-based modes).
- Upsamples the final grid prediction back to the input spatial size for fair Dice comparison against 128³ models.

## 2026-05-22 — Multilevel train/val loop generalised to N levels

`experiments/multilevel/train.py`, `experiments/multilevel/model.py`, `experiments/feature_attention/model.py`, `configs/experiment/multilevel.yaml`.

**`MultilevelICL` model changes:**
- `MaskCNN` — shared 3D ConvNet (same-padding, grid-size agnostic) that encodes a soft/binary mask into per-voxel embeddings, interpolated trilinearly to each level's resolution. Replaces the old scalar avg-pool label path when `mask_cnn_dim > 0`.
- `num_registers` — learnable register tokens appended to context K/V and cascaded between levels; `detach_cascade_registers` controls whether gradients cross level boundaries.
- `append_zero_attn` — adds a null K/V slot to cross-attention so target patches can "abstain" from retrieving context.
- `output_dim` added to `PatchICLAttention` (separate from `label_dim`): head always predicts a 1-dim binary mask regardless of label embedding size.

**Training loop:**
- Train and validation loops now handle any number of levels (previously hardcoded to 2); loss weights configurable per level via `train.loss_weights`.
- `_encode_ctx_labels` helper unifies CNN-embedded and scalar label injection paths.
- `soft_labels_train` / `soft_labels_eval` separate: avg-pool float labels during training, binarised at inference.
- New config keys: `mask_cnn_dim`, `soft_labels_train`, `soft_labels_eval`, `num_registers`, `append_zero_attn`, `detach_cascade_registers`.

**`experiments/feature_attention/train_dice46.py`** — standalone training script for `PatchICLAttention` that uses Dice loss instead of BCE (experimental variant).

## 2026-05-21 — 3D RoPE for PatchICLAttention + multilevel patch sampling rewrite

`src/rope3d.py`, `experiments/feature_attention/model.py`, `experiments/multilevel/model.py`, `experiments/multilevel/train.py`.

**3D RoPE (`src/rope3d.py`):**
- `build_rope_cache_3d(max_pos, dim, num_heads)` — builds a `(max_pos, dim//2)` frequency cache; `per_axis` rounded to the nearest multiple of `head_dim` (same fix as in `src/rope.py`).
- `apply_rope_3d(x, coords, cache)` — applies RoPE to a `(B, N, dim)` tensor given integer `(B, N, 3)` d/h/w coordinates; works for any token subset (dense or sparse, no grid-size assumption).

**`PatchICLAttention` gains `pos_encoding="rope3d"`:**
- RoPE applied inside `_mha()` directly to Q and K after projection; falls back gracefully to no PE if coords are not supplied.
- `rope_max_pos` parameter overrides the cache extent (default: `max(grid_size)`).

**`MultilevelICL` switches to RoPE:**
- All levels use `pos_encoding="rope3d"` with a shared cache keyed by `global_max_pos = max(max(grid_size) for all levels)`.
- Removed the old `coord_projs` (`nn.ModuleList` of `Linear(3, embed_dim)`) that injected normalised coordinates externally.

**Patch sampling rewrite (`experiments/multilevel/train.py`):**
- `_gumbel_topk(weights, n, temperature)` — stochastic top-n via Gumbel noise; normalises weights per batch item, replaces the old deterministic entropy/fg/border split.
- `sample_target_patches` now accepts a `mode` argument: `gt_previous_pred_error` (|pred − GT|), `gt_foreground_entropy_balanced` (0.5·GT + 0.5·H(GT)), `predicted_entropy` (H(pred)); controlled by `data.target_sampling`.
- `sample_context_patches` uses `gt_foreground_entropy_balanced` averaged across K context masks; replaces the old fg + border morphological sampling.
- `grid_coords_3d` replaces `make_coords_3d` — returns integer voxel coords (long) rather than normalised floats.

## 2026-05-23 — Quality benchmark for in-context segmentation (eval.py)

Added a segmentation-quality benchmark comparing native models against SOTA.

- `scripts/eval.py` — new evaluation script; argparse CLI; measures per-class and mean Dice on TotalSegmentator test split; saves `results/eval_*.json` and `results/eval_*.csv`.
- `data/benchmark_classes.py` — curated 16-organ class list covering a range of sizes and difficulty.
- `src/benchmark_models/base.py` — `InContextModel` ABC (`predict(target, context_imgs, context_masks) → binary mask`).
- `src/benchmark_models/native.py` — adapters for `ViTInContext3D` (`NativeViT`) and `ResEncInContext3D` (`NativeResEnc`), loading from checkpoint.
- `src/benchmark_models/medverse.py` — adapter for Medverse (local repo at `/nfs/norasys/notebooks/camaret/repos/Medverse`); applies per-volume min-max normalisation; adds channel dim to context masks.
- `src/benchmark_models/__init__.py` — `load_model(name, ...)` registry.

Methods included: native_vit, native_resenc, medverse. UniverSeg (2D-only) and Show&Segment (CVPR 2025, no public code) excluded.

## 2026-05-22 — Fix head-boundary misalignment in 3D RoPE

`src/rope3d.py`, `experiments/feature_attention/model.py`.

`build_rope_cache_3d` previously computed `per_axis = ((dim // 3) // 2) * 2`, which
for `dim=512` gives `per_axis=170` — not a multiple of `head_dim=64`. After the head
split, heads 2 and 5 straddled two axis blocks (d+h and h+w respectively), receiving
mixed-axis rotations.

Fix: added `num_heads` parameter; `per_axis` is now rounded down to the nearest
multiple of `head_dim`. For `dim=512, num_heads=8`: `per_axis=128` (64 pairs/axis),
heads 0–1→d, 2–3→h, 4–5→w, 6–7 unrotated. Default `num_heads=1` preserves old
behaviour for callers that don't pass it. Call in `PatchICLAttention.__init__` updated
to pass `num_heads=num_heads`.

## 2026-05-20 — Synth pipeline fixes in _get_synth_item

Three issues fixed in `TotalSegInContextDataset._get_synth_item`
(`src/totalseg_dataloader_incontext.py`):

- **`use_crop` ignored** (main bug): synth items always loaded the whole-body
  pre-resized 128³ image, giving ~16× worse effective resolution than real-data
  items when `use_crop=True`. Fix: when `use_crop=True`, load native `ct.npy` +
  native synth label file, compute the centroid of the picked supervoxels' union,
  and crop a `T³` patch around it with the same jitter logic as `_load_crop`.
- **`aug_cfg` not guarded**: `self.aug_cfg.synth` raised `AttributeError` when
  no augmentation config was provided (valid for debug runs). Fix: guard with
  `if self.aug_cfg is not None and self.aug_cfg.enabled`, falling back to
  unaug'd identity copies.
- **Slow-path CT preferred `ct.nii.gz` over `ct.npy`**: native numpy is faster
  and avoids the nibabel stack. Fix: check for `ct.npy` first, fall back to
  `ct.nii.gz` only if absent.

## 2026-05-20 — Soft GT for avgpool downsampling

In `process_batch` (`experiments/feature_attention/train.py`), the `> 0` threshold
on `gt_loss` was discarding the continuous coverage fraction produced by avgpool,
making `mask_pool: avg` identical to `max` in practice.

**Fix**: `gt_loss` is now the raw avgpool output (soft float in [0,1]); a separate
`gt_bin = (gt_loss > 0).float()` is kept for norm_dice. Context labels stay binary
(`> 0`) since the model binarizes them internally anyway (`label_dim=1` path uses
`nn.Embedding(2, dim)`). BCE with soft targets is valid and provides proportional
gradient weighting by patch coverage fraction.

## 2026-05-18 — Fix synth elastic augmentation bottleneck

`apply_synth_aug` was generating a full-resolution `(3, D, H, W)` noise field
and smoothing it with `_gaussian_smooth_3d_field` (sigma 8–15 → kernel size
33–61).  For 128³ volumes that's three depthwise conv3d passes on 6M elements
each call, done K+1 times per item.  This starved the DataLoader prefetch
queue and caused periodic training stalls whenever workers drew large sigma.

**Fix** (`src/augmentations.py`, `apply_synth_aug`): replaced the
full-res-randn + Gaussian-conv path with a coarse-grid approach — generate
displacement at `(D/sigma, H/sigma, W/sigma)` resolution and upsample
trilinearly.  Equivalent smooth frequency, ~60× cheaper for sigma=8–15.
The real-label path was unaffected (uses `apply_task_aug` where elastic p=0).

## 2026-05-18 — SegGPT-style random coloring + synthetic label support

**Random coloring** (`--random_coloring`, on by default):
- `TotalSegInContextDataset._apply_coloring`: finds label ids shared across all
  masks in a sample, samples one random RGB per id, returns `(3,D,H,W)` float32
  tensors for `label` and `context_out` (same palette across target + context).
- `PatchICLAttention` gains `label_dim` (1=binary discrete, 3=RGB). Label injection
  uses `nn.Linear(label_dim, C, bias=False)` for RGB so background (black=0) injects
  nothing; output head predicts `label_dim` values per patch.
- Training uses smooth-L1 loss for RGB; metrics use L2 norm of predicted colour as
  scalar probability for AUROC/norm-dice. Validation stays binary (val dataset
  never colorises).

## 2026-05-18 — Synthetic label support in feature_attention/train.py

Added `--synth_method`, `--synth_unions`, and `--p_synth` CLI args to
`experiments/feature_attention/train.py`.  These are forwarded to
`TotalSegInContextDataset` for the training split (val unchanged), enabling
the same supervoxel-based synthetic augmentation path already supported in
`scripts/train.py`.

## 2026-05-18 — PatchICLAttention improvements from TabPFN comparison

Rewrote `experiments/feature_attention/model.py` with five changes derived from
comparing against TabPFN v3's ICL stage:

- **K/V normalization (bug fix)**: cross-attention now pre-norms K and V via
  per-layer `kv_norms` (RMSNorm). Previous code normalised only Q, breaking
  pre-norm invariance as encoder features bled into K/V at each layer.
- **Context self-attention**: each layer now runs a full transformer block
  (SA + FFN) on context tokens before the cross-attention. This lets context
  patches interact across K images before being retrieved, analogous to how
  TabPFN's ICL train rows self-attend through train-only K/V.
- **Log-n query scaling**: cross-attention Q is pre-scaled by
  `log(M)/log(n_base)` before `F.scaled_dot_product_attention` (which applies
  `1/sqrt(D)` internally). Calibrates softmax temperature for large context
  sequences (M = K×8³ = 512+). Configurable via `log_n_base` (default 512).
- **Retrieval head K projection**: added `ret_k_proj` separate from `ret_q_proj`,
  decoupling the similarity space from the representation space (mirrors
  TabPFN's `ManyClassDecoder`).
- **Default input_norm → rmsnorm**: encoder features have per-channel scale
  that varies across the 4 encoder levels; normalising at input stabilises
  Q/K dot products.

Switched from `nn.MultiheadAttention` to manual projections +
`F.scaled_dot_product_attention` throughout, enabling Flash Attention and
giving full control over scaling and norming. `train.py` gains
`--no_ctx_self_attn`, `--no_log_n_scaling`, `--log_n_base` flags.

## 2026-05-15 — pluggable encoder + STU-Net

Refactored the encoder out of `resenc_in_context.py` into `src/models/encoders/`:

- `src/models/encoders/resenc.py` — `ResEncEncoder`: nnUNet `ResidualEncoderUNet`
  encoder wrapper (4 stages, 8× downsample, 2-channel image+mask input).
  Identical behaviour to the original inline encoder.

- `src/models/encoders/stunet.py` — `STUNetEncoder`: 6-stage STU-Net image encoder
  (`_ImageEncoder`, conv_blocks_context naming preserved for pretrained weight
  compatibility) + SAM-style 3-D mask encoder (`_Mask3DEncoder`, stride-2 CNN),
  fused at the bottleneck (additive or concat+proj).
  Supports `small | base | large | huge` variants.
  Pretrained TotalSegmentator weights loadable via `stunet_pretrained=<path>`.

- `src/models/resenc_in_context.py` — accepts `encoder_name` ("resenc" | "stunet")
  and builds encoder + decoder dynamically from `encoder.skip_channels`,
  `encoder.bot_features`, `encoder.total_stride`.  Decoder depth scales
  automatically (3 stages for ResEnc, 5 for STU-Net).

- `configs/config.yaml` — added `model.encoder`, `stunet_variant`,
  `stunet_pretrained`, `stunet_freeze`, `mask_fusion`.

- `scripts/train.py` — `build_model` passes new encoder params to
  `ResEncInContext3D`.

## 2026-05-18 — Feature-attention experiment

Added `experiments/feature_attention/` to study the impact of attention mechanism design on patch-level in-context segmentation.

**Motivation**: cosine-similarity retrieval (normed dice 0.324) vs. TabPFN in-context classifier (0.416) showed a clear gap. The goal is to understand which architectural decisions explain it and to train a lightweight learned attention module.

**Files created**:

- `experiments/feature_attention/model.py` — `PatchICLAttention`: learned cross-attention binary classifier. Target patches attend to context patches (K/V from context only, same train-only masking as TabPFN). Eight configurable decisions:
  - `label_injection` — how context binary labels enter tokens: `additive` (TabPFN-style token += label_embed) | `concat` | `gate` | `none`
  - `output_head` — `linear` | `mlp` | `retrieval` (cross-attention Q=tgt, K=label-conditioned ctx, V=scalar labels)
  - `pos_encoding` — `none` | `sinusoidal` (fixed 3D sin/cos) | `learned` (nn.Embedding per grid position)
  - `input_norm` — `none` | `rmsnorm` | `l2`
  - `num_layers`, `num_heads`, `ff_factor`, `dropout`
  - Zero-init residual output projections (stable early training, TabPFN-style)

- `experiments/feature_attention/train.py` — trains `PatchICLAttention` on TotalSegmentator train split with frozen STU-Net encoder. Class-balanced sampling, AUROC-based validation each epoch, checkpoint saved on best val AUROC. W&B logging to `patch_icl_3d_exps`.

- `experiments/feature_attention/run.py` — evaluation script mirroring `feature_similarity/run.py` (same metrics: soft dice, normed dice, AUROC; same visualisation; same W&B logging).

**Also added to `experiments/feature_similarity/run.py`**:
- `--method tabpfn` option using `TabPFNClassifier` as the prediction head
- `--mask_pool` option (`max` / `avg`) for GT downsampling
- `--feature_level all` to concatenate all encoder levels
- W&B logging (project `patch_icl_3d_exps`, per-sample metrics + figures)
- Avg inference time tracking

## 2026-05-15 — PatchICL-style token conditioning

Added three token-level enrichments to `ResEncInContext3D`, inspired by PatchICL v3:

- **Type embeddings** (`target_type_embed` / `context_type_embed`): learnable `(1,1,C)`
  offsets added to bottleneck tokens before Stage 1. Teach the model to distinguish
  target volumes (zero mask) from context volumes (real mask). Init to zero → no effect
  at initialisation.

- **Register tokens** (`num_registers`): `R` learnable global tokens appended to the
  Stage-2 KV sequence alongside the spatial context tokens. Give the target a set of
  global scratchpad slots to attend to. `RoPECrossAttn` updated to apply identity
  rotation (position 0) to register tokens, preserving backward compatibility.

- **Context-first layers** (`num_context_layers`): optional extra `RoPETransformerBlock`
  layers applied only to context tokens before Stage 1. Context enriches itself via
  self-attention before the joint Stage-1 pass — mirrors PatchICL v3's `context_layers`.

Config knobs added to `configs/config.yaml` (`model.num_registers`,
`model.num_context_layers`); both default to 0 (disabled) to preserve existing runs.

To train with STU-Net (requires 128³ volumes):

```bash
python scripts/train.py \
  model.encoder=stunet \
  model.stunet_variant=base \
  model.stunet_pretrained=/path/to/stunet_base.model \
  data.image_size=[128,128,128]
```
