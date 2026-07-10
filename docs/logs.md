# Change log

## 2026-07-10 — refine: `arch.refine_memory` flag (cross-level memory token)

- **Opt-in cross-level memory for the refine pass: `arch.refine_memory` (default off).** When enabled
  in multi-resolution mode, the refine pass prepends a learnable memory token (type-only adapter,
  `mem_type`, no projection) to the sequence before attention. The memory attends to the detached
  coarse pass's thinking rows (`mem_type` + coarse thinking, `.detach()`), allowing the refine
  pass to selectively condition on coarse-level reasoning without backprop through the coarse
  encoder. Mirrors `multilevel/`'s `stage1_think` design but with shared weights.
- Wiring: `PatchSetCNN(..., refine_memory=...)` constructor (Task 1); `_attn(..., mem=...,
  return_think=False)` dual-purpose signature (Task 2 — produces thinking for refine, consumes
  memory token for coarse); config flag + checkpoint rebuild (Task 3 — `build_model` adds to
  `arch` dict, `train_base.yaml` exposes `arch.refine_memory: false`). Single-level mode is inert
  (constructor-only param, no refine forward).
- Tests: round-trip checkpoint via `build_model`, wiring, gradient flow to `mem_type`, detach
  verification. Both refine modes (`reencode`/`encode_once`) supported.

## 2026-07-10 — train: fix compile flag namespace mismatch (`train.compile` → `arch.compile`)

- The compile knob was **still dead**: the config defines it at `arch.compile` (`train_base.yaml:23`)
  but `train.py` read `cfg.train.get("compile", False)` — a key that exists nowhere, so it always
  resolved `False` and the model ran eager regardless of `arch.compile=true`. `pfn_seg.py:327` reads
  the correct `cfg.arch.compile`; `train.py` had drifted. Now reads `cfg.arch.get("compile", False)`.
- Also ported two things from `pfn_seg.py`: (1) compile Muon's `_newtonschulz5_batched` (pure tensor
  ops) alongside `model.transformer`, and (2) the node-local triton/inductor cache guard
  (`TRITON_CACHE_DIR`/`TORCHINDUCTOR_CACHE_DIR` on `/tmp` keyed by hostname) to avoid the NFS GLIBC
  cache-poisoning issue. Scope is unchanged: only `model.transformer` + Newton–Schulz compile; the
  encoder/bbox crop/pool stay eager (graph-break).

## 2026-07-10 — train: wire the (previously dead) `train.compile` flag for patchset_cnn

- `experiments/2d/train.py` defined `train.compile` (via `train_base.yaml`) but never read it — the
  model always ran eager. Now, when `train.compile` is set, it compiles **only** `model.transformer`
  (`torch.compile(..., dynamic=True)`), mirroring `multilevel/train.py` which compiles just its
  trainable PatchSetPFNs. The transformer is pure tensor ops so it graph-compiles cleanly; the
  encoder + bbox crop/pool (`grid_sample`, `adaptive_avg_pool2d` with data-dependent windows) stay
  eager (they graph-break). UniverSeg has no `.transformer` so it is untouched (`hasattr` guard).
- Checkpoint reload made robust to the mid-key `_orig_mod.` prefix that submodule-compile leaves
  (`transformer._orig_mod.…`): warm-start (`train.py`) and `eval_incontext.py` now strip with
  `.replace("_orig_mod.", "")` instead of `removeprefix` (leading-only). Verified the stripped keys
  strict-load into a fresh model and the Muon `"transformer" in n` filter still selects the 2D
  matrices post-compile. (Local `torch.compile` couldn't be exercised — broken `torch._inductor`
  import in this env; runs on the GPU cluster where multilevel already compiles.)

## 2026-07-10 — refine: `arch.refine_mode` flag (reencode | encode_once)

- **New prototype for the 2-level refine pass, behind `arch.refine_mode`** (`patchset_cnn.py`,
  default `reencode` = unchanged behavior):
  - `reencode` (old): re-run the whole model (encoder + attention) on the upsampled crop — every
    stage recomputed at the finer scale. 2× encoder passes.
  - `encode_once` (new, à la `experiments/2d/multilevel/zoom_pipeline`): encode all K+1 images
    ONCE into native multi-scale maps, then run the attention half twice — on the full pooled
    features (coarse) and on the SAME maps cropped to each bbox and pooled back to the T grid
    (refine). ~half the encoder compute; grid_sample crop is differentiable so grad still reaches
    the encoder from the refine loss. Tradeoff: the refine pass reuses stem/shallow detail but
    does NOT recompute deep-stage semantics at the crop scale.
- Refactor enabling this: `ConvEncoder.forward` split into `encode_maps` (native multi-scale list)
  + `pool_maps`; `_segment` split into `_grid_tokens`/`_occupancy` feature build + a shared `_attn`
  (encoder-independent attention half, now reused by the single-level model and both refine passes);
  bbox selection factored into `_select_bbox`. New `bbox_refine.crop_pool_maps` crops native maps
  (origin/s rescaled per map resolution) and pools to the token grid.
- Verified: coarse head identical across modes; encoder stem runs 2× (reencode) vs 1× (encode_once)
  on B·T images; grad reaches encoder+decoder from both heads. Tests in
  `tests/test_patchset_cnn_refine.py` (12 passed). Wire via `arch.refine_mode` in the experiment cfg.

## 2026-07-10 — metrics: tag cossim/top-k with the token grid; resolution-tagged sample table

- **`cossim` and `top{k}` now carry their resolution**: logged as `cossim@{T}` / `top{k}@{T}`
  (val, `evaluate.py`) and `train/cossim@{T}` / `train/top{k}@{T}` (train), where T = the coarse
  token grid (`low_res`, e.g. 32). They were always computed on that grid but were previously
  untagged. `_select_metric` (`train.py`) and the train console top-k lookup were updated to
  find the tagged keys; the cossim fallback now matches `cossim@*`.
- **Per-sample wandb `val/samples` table columns are now resolution-tagged and model-adaptive**
  (built lazily per run in `validate()`, replacing the static `SAMPLE_COLS`):
  - always: `dice` (native hard — fused pred for refine models);
  - non-native (patchset/refine): `dice_ds@{T}`, `dice_ds_soft@{T}` (coarse grid);
  - refine: `dice@{Rf}`, `dice_soft@{Rf}`, `dice_fused@{Rf}` (per-sample, previously only
    aggregated). Native (UniverSeg) tables now carry just `dice` (no meaningless NaN columns).

## 2026-07-10 — metrics: `ds_metric_res` (@R pooled Dice) is now UniverSeg-only

- The fixed-resolution `dice_ds@R` / `dice_ds_soft@R` metrics (from `eval.ds_metric_res`) are
  now computed **only for native models (UniverSeg)**, in both `validate()` (`evaluate.py`) and
  `train_epoch` (`train.py`). Their purpose is to pool UniverSeg's native prediction to R×R so
  it is comparable to patchset_cnn's coarse grid. For non-native patchset_cnn/refine those
  `@R` metrics were always computed on the *coarse-upsampled* pred — confusing and redundant
  with the model's own `dice_ds@{token grid}` — so `ds_metric_res` is now ignored for them.
- Removed the dead `eval.ds_metric_res` from `configs/experiment/2d/2_omnisynth_medseg_refine.yaml`
  (patchset/refine); it lives in `configs/experiment/2d/model/universeg.yaml` (`[32]`), the
  correct home now that it is UniverSeg-only. patchset_cnn/refine still report their native
  coarse grid as `dice_ds@{low_res}` / `dice_ds_soft@{low_res}`.

## 2026-07-10 — refine: `dice` scored on the fused prediction + checkpoint selects on it

- For multi-resolution refine `PatchSetCNN` models, the headline **`dice`** metric (native
  hard Dice) is now computed on the **fused** prediction of the last level (`rg["fused"]`, at
  full H×W) instead of the coarse `final_logit`. Applies in both `validate()` (`evaluate.py`)
  and `train_epoch` (`train.py`, `train/dice`). Non-refine models (UniverSeg, plain
  patchset_cnn) are unchanged.
- **Checkpoint selection** (`_select_metric`, `train.py`) for refine models now selects on
  native `dice` (the fused full-res hard Dice) rather than `dice_fused@{last res}`. Non-refine
  selection (cossim → dice) is unchanged.
- Scoped step-by-step: only `dice` moved to the fused pred for now. `dice_ds@R` /
  `dice_ds_soft@R`, cossim, and top-k still reflect the coarse pass; `dice_fused@R` /
  `dice@R` per-level metrics remain as diagnostics.

## 2026-07-10 — eval: coarse→fine refine qualitative figure (save_refine_figure)

- **`save_refine_figure`** added to `experiments/2d/evaluate.py`: 2×3 PNG panel showing the coarse→fine refinement for a refine `PatchSetCNN` checkpoint. Row 0 = target, row 1 = first context. Col 0: full frame + GT contour + res0 (coarse) pred heatmap + bbox rectangle. Col 1: bbox crop + GT contour + res1 (refine) pred heatmap. Col 2: full frame + GT contour + fused (place_window stitch) pred. All overlaid with lime GT contours and yellow/cyan bbox rectangles.
- **`_refine_overlay_ax`** helper: gray base + optional Reds heatmap (with optional extent for T×T→crop stretch) + optional lime GT contour + bbox `Rectangle` patches. `from matplotlib.patches import Rectangle` added to the imports.
- **Wired into `validate()`**: inside the gated `if figures and fig_key not in saved` block, immediately after `save_figure`, a `if rg is not None` branch writes `{ds}_l{lv}_refine.png` alongside the standard `{ds}_l{lv}.png` panel; optionally logs to `figures_refine/{ds}/label_{lv}` in wandb. Controlled by the existing `eval.save_figures` / `eval.max_figures` gates. Plain checkpoints (`rg is None`) are unaffected.
- **Verified**: unit tests (`tests/test_refine_figure.py`, 2 tests) pass; 1-epoch CPU smoke (omniSynth, bs=2, 8 samples, size=64) trains to `best.pt`, eval re-loads and emits `dice/mean=0.0308` + 4 `*_refine.png` panels in the eval out_dir alongside the standard `*.png` panels.

## 2026-07-10 — Config + docs: multi-resolution refine (arch.resolutions) wired end-to-end

- **`PatchSetCNN` refine reworked to multi-resolution per-level** (`src/models/patchset_cnn.py`, `src/models/bbox_refine.py`): constructor arg is now `resolutions: list[int]` (effective full-image resolutions), replacing the old `refine: bool` / `refine_crop: int` flags. Token count is constant at T = `resolutions[0]`² tokens; the refine crop is derived as `image_size · R0 / Rk` (e.g. 128 · 32/64 = 64 px for `resolutions=[32,64]`). Per-level losses: coarse `@resolutions[0]` + one refine loss per additional level, weighted by `train.refine_loss_weight`. Old `refine`/`refine_crop` arch flags removed from `configs/experiment/2d/model/patchset_cnn.yaml`.
- **`place_window` (replace-stitch)** added to `src/models/bbox_refine.py`: writes the refine crop back into the native-resolution canvas by replace (not additive). Used exclusively inside the `refine_geometry` helper in `experiments/2d/evaluate.py` for metric assembly; no forward-path fusion in the model.
- **`refine_geometry` helper** in `experiments/2d/evaluate.py`: assembles per-level logits into the fused canvas for metric-only reporting. `fuse_window` (additive, logit-space) remains in `bbox_refine` unused, reserved for a future fused loss.
- **Metrics**: `dice@{Rk}` / `dice_soft@{Rk}` per refine level logged in train and val; `dice_fused@{Rk}` (hard and soft) reported via `refine_geometry`. Checkpoint selection is on `dice_fused@{resolutions[-1]}`.
- **Bbox origins are detached** (`argmax` on `sigmoid(coarse).detach()`); gradients flow only through the two `_segment` calls.
- **New experiment leaf** `configs/experiment/2d/2_omnisynth_medseg_refine.yaml`: rewritten to set `arch.resolutions: [32, 64]`, `train.refine_loss_weight: 1.0`, and `eval.ds_metric_res: [16, 32]`. Inherits `1_omnisynth_medseg` via defaults.
- **Old checkpoints**: any `best.pt` carrying `arch.refine` / `arch.refine_crop` (the previous additive-fusion design) will not rebuild under the new constructor — they are throwaway smoke artifacts from the prior plan; retrain.
- **Verified**: dry-run `--cfg job` confirms `model: patchset_cnn`, `resolutions: [32, 64]`, `refine_loss_weight: 1.0`, `ds_metric_res: [16, 32]` all compose. 1-epoch smoke train (64 samples, bs=4) completes without shape errors, logs `val dice_fused@64=...` in the console, and saves `best.pt` at `…/2026-07-10_dummy-dqibaa6e/best.pt`. `eval_incontext.py` reloads and prints `Loaded patchset_cnn (size=128, ctx=1)` with no rebuild error — `arch.resolutions` round-tripped correctly through the checkpoint.

## 2026-07-09 — PatchSetCNN refine mode wired into config + trainer

- **New `PatchSetCNN` mode** (`src/models/patchset_cnn.py`, `src/models/bbox_refine.py`): constructor args `refine: bool` (default `False`) and `refine_crop: int` (default `64`). When `refine=True`, after the coarse 32×32 pass the model crops a `refine_crop × refine_crop` bbox around the predicted foreground (detached `argmax`), runs a second shared-weight `_segment` forward at native resolution inside that crop, and additively fuses the upsampled crop logit back onto the coarse logit → `final_logit` at native `(B,1,H,W)` resolution. When `refine=False`, `final_logit` is the coarse `(B,1,R,R)` as before. Bbox ops live in `src/models/bbox_refine.py`; both modules committed and tested (8 tests pass).
- **`build_model` threading** (`experiments/2d/train.py`): the `patchset_cnn` branch's `arch` dict now includes `"refine": a.get("refine", False)` and `"refine_crop": a.get("refine_crop", 64)`. These are passed to `PatchSetCNN(image_size=..., **arch)` and saved verbatim in `best.pt` so `eval_incontext.py` rebuilds the refine model with zero drift. No train/eval loop changes — the loop already pools GT to `final_logit`'s spatial size; when `refine=True` that size is native `H×W` so the loss and all Dice metrics land at native resolution.
- **Config group default** (`configs/experiment/2d/model/patchset_cnn.yaml`): added `refine: false` and `refine_crop: 64` under `arch:` (backward-compatible; off by default so all existing runs are unaffected).
- **New runnable leaf** (`configs/experiment/2d/2_omnisynth_medseg_refine.yaml`): inherits `1_omnisynth_medseg` via `defaults`, flips `arch.refine: true`, and sets `eval.ds_metric_res: [16, 32]` to preserve coarse-grid Dice alongside the new native Dice. Run with `python experiments/2d/train.py --config-name 2_omnisynth_medseg_refine`; override window with `arch.refine_crop=48` etc.
- **Checkpoint metric**: with `refine=True` the model predicts at native resolution, so `final_logit.shape == lbl.shape` — the loop's native-vs-coarse branch selects `metric="dice"` (not `"cossim"`), and the checkpoint is selected on hard Dice. `ds_metric_res=[16,32]` in the new leaf preserves pooled coarse Dice for comparison.
- **Verified**: dry-run `--cfg job` confirms `model: patchset_cnn`, `refine: true`, `refine_crop: 64`, `resolution: 32`, `ds_metric_res: [16, 32]` all present. 1-epoch smoke train (64 samples, bs=4) completes without shape errors and saves `best.pt`. `eval_incontext.py` reloads and prints `Loaded patchset_cnn (size=128, ctx=1)` with no rebuild error — `arch.refine/refine_crop` round-tripped correctly through the checkpoint.

- 2D trainer: renamed the universeg train metric `train/dice_ds_soft` → `train/dice_soft` (`train.py:331`). It's soft Dice at **native full resolution**, so the `ds` (=downsampled) prefix was a misnomer — it now parallels `train/dice` (native hard) and matches `pfn_seg.py`'s existing `train/dice_soft` (= soft Dice at the model's native pred resolution). Symmetry: universeg logs `dice`/`dice_soft` (native hard/soft); patchset_cnn logs `dice_ds@R`/`dice_ds_soft@R` (coarse hard/soft, correctly `@`-tagged). Scope was train-side only: `evaluate.py`'s `dice_ds_soft@R`/`dice_ds_soft@{low_res}` summaries, the `SAMPLE_COLS` `dice_ds_soft` column, and the `train.py:354` val-summary lookup all refer to genuinely-downsampled values and were left unchanged. Caveat: renames a logged wandb key, so old runs won't align on it.
- 2D configs: new `model/` Hydra group (`configs/experiment/2d/model/{patchset_cnn,universeg}.yaml`, each `# @package _global_` on line 1 so it sets the top-level `model:` string plus that model's `arch`/`train`/`eval` blocks). This makes `model=patchset_cnn|universeg` a real group override. New runnable leaf `1_omnisynth_medseg.yaml` (defaults: `omnisynth_train_base`, `model: patchset_cnn`, `_self_`) bakes the omniSynth-medseg-scene params held fixed across the sweep — `data.context_size=1`, `arch.resolution=32`, `train.topk_k=64`, `synth.source=medseg`, `synth.scene.{p_copy=0,placement=random,max_nb_objects=8,background=image}` — and runs either model: `python experiments/2d/train.py --config-name 1_omnisynth_medseg model=patchset_cnn|universeg <cli overrides>`. Existing `patchset_cnn_train.yaml`/`universeg_train.yaml` thinned to consume the same `model/` group (single source of truth for model defaults; behavior byte-for-byte unchanged — verified via `--cfg job`). Gotcha: the `@package _global_` directive MUST be the first line; a prose `# @package _global_ ...` comment elsewhere in the header is ignored and the group nests under `model.*` instead (caught in verification: `model:` resolved to a mapping and `lr` stayed at the base default).
- 2D trainer: qualitative per-sample val log (ported from the old universeg_train.py into the unified train.py). `validate` now builds a `wandb.Table` (`SAMPLE_COLS`: epoch, dataset, character, target_mode, target_pos, context_pos, transforms, dice) logged as `val/samples`; `_fmt_transforms` renders each aug placement as `r..,s..,dx..,dy..` (`-` / empty for identical/class). Needs per-sample provenance, so: `render_scene` now returns a 4th `info` dict (`target_cells` indices + aligned `target_transforms`) and `affine_jitter` returns `(bitmap, params)` with the sampled rotate/scale/dx/dy; `OmniSynthICLDataset` meta gains `target_cells`/`context_cells` as **(row, col) grid positions** (not flat indices — `divmod(cell, grid)`) plus `target_transforms`. instCopy (`is_copy`/`copy_slot`) kept alongside; copied context slots correctly report the query's cells. test_render.py updated for the 4-tuple/2-tuple returns. Verified: render tests pass, 1-epoch smoke run (train→val→table→ckpt) clean, copy vs non-copy context positions correct. The old train-Dice-monitoring part of universeg_train.py was NOT ported (train.py already has it, generalised with cossim/top-k).
- 2D trainer: added top-k patch-recall metric (`top{k}`, k=`cfg.train.topk_k` default 16). Per sample, recall of the GT-positive patches within the k highest-valued predicted patches: |gt_pos ∩ topk(pred)| / |gt_pos|, where gt_pos = patches with GT>0 capped to k. Reaches 1.0 exactly when ALL true patches are in the model's top-k — denominator is the (sparse) positive count, not k, so a model that found every true patch isn't penalised for GT having < k positives. Purely rank-based on the pred side (threshold/scale-free), complements cossim. Low-res only: gated on the same `logit.shape != lbl.shape` condition as cossim (at native res a patch is a pixel), so it never fires for UniverSeg. `common.topk_overlap` (per-sample) + train `_topk_sum` (batched, on-device scatter/gather); logged for train+val (`train/top{k}`, `top{k}/*`). Verified batched==per-sample, perfect ranking=1.0 (incl. n_pos>k rows), miss-1-of-5=0.8, empty-GT rows skipped. Caveat: when patch count ≤ k (e.g. R=4 → 16 patches, k=16) pred top-k = all patches → recall trivially 1.0; only informative when R×R > k (e.g. R=16 → 256).
- 2D trainer: log the full resolved Hydra config to wandb (`OmegaConf.to_container(cfg, resolve=True)`) instead of a hand-picked flat dict. All config (train.*, data.*, synth.*, arch.*, ...) is now captured as the single source of truth; removed the flat overlay + `scene`/`target_mode` helper vars that only fed it (no retro-compat kept). `ckpt_meta` unchanged — still saved into the checkpoint. Note: wandb keys are now nested (e.g. `train.lr`, `data.image_size`) rather than flat (`lr`, `image_size`); update any saved wandb filters/groupings accordingly.
- 2D trainer: skip cossim at native prediction resolution. cossim only exists because at low logit res the soft/hard Dice targets collapse; when the logit is already at native res (UniverSeg → logit==GT res) it's redundant with Dice. `train_epoch`/`validate` now compute cossim only when `logit.shape[-2:] != lbl.shape[-2:]` (validate returns empty cos dicts). Checkpoint metric follows: `cossim` at low res, native hard `dice` otherwise (`metric` var; drives `val/best_{metric}` and the "Done." print). train/cossim + train postfix `cos` only shown when computed. So UniverSeg (native) now checkpoints on hard Dice; patchset (low-res) unchanged.
- PatchSetCNN: added optional per-context-image identity embedding (`context_id_embed`, default False; `max_context`=16). A learned tag per context-image slot is added to all N patches of that image (both img/mask cols, image-major order); the target image gets its own learned `qry_id` tag. Previously the support was a fully permutation-invariant patch set with only within-image (i,j) position — so two context images with identical content (e.g. instCopy duplicates) produced byte-identical tokens and were indistinguishable to attention. Now they can be grouped/told apart. Verified: off→identical imgs give identical tokens (0.0), on→differ by the tag (0.088). Init: ctx_id/qry_id at σ=0.1 (small-norm rich-regime sweet spot for Adam per Ito et al. 2025 on learnable PE init; nn.Embedding default σ=1 = lazy regime, σ≲0.05 also hurts under Adam). Fourier (i,j) PE left as-is (fixed 2D sinusoid = the paper's best-performing "ground-truth" encoding for known grid structure). Wired through train.build_model + ckpt meta; enabled in patchset_cnn_train.yaml. No eval.py change (no patchset_cnn eval path yet).
- 2D trainer: added cosine-similarity metric (`cossim`) and made it the checkpoint-selection metric. Diagnosed why low-res patchset training shows near-zero dice: the loss/metric target is avg-pooled soft occupancy, and at R=4 a thin Omniglot glyph fills only ~3-9% of a 32px cell, so `dice_ds_soft` is pinned at its mean-occupancy ceiling (~0.05) and native hard `dice` is structurally 0 (prob fit to ~0.03 occupancy never reaches 0.5) — even for a PERFECT predictor. Cosine `Σ(p·g)/(‖p‖‖g‖)` is scale-invariant → real 0→1 signal (verified: perfect=1.0, untrained~0.24, converged copy-task=1.0), no threshold/hyperparam. common.cosine_sim + train `_cos_sum`; logged for train+val (`train/cossim`, `cossim/*`); dice metrics still logged. The copy task itself was learning fine (BCE 0.69→0.013 in ~50 steps) — only the metric was misleading.
- PatchSetCNN: decoupled token resolution R from enc_dims. ConvEncoder depth is now len(dims)-1 (architecture), producing features at the encoder's natural res H/2**(len-1); each scale is then resampled (area-pool down / bilinear up) to R×R. R no longer constrains enc_dims length or requires image_size = R × pow2. Default (128, R16, 4 dims) unchanged (resample is identity). ConvEncoder no longer takes image_size.
- omniSynth: instCopy copy-tasks — OmniSceneConfig.p_copy (default 0.9) / n_copy (default 1); train-only, isolated rng; copy slot(s) are an EXACT copy of the query scene; meta gains is_copy/copy_slot. Eval byte-identical.

## 2026-06-30 — BiomedParseDataset init: skip the ~1.8h NFS glob/stat via the store index
- Symptom: `eval.py data.source=biomedparse` "dataloading takes forever" — init never
  reached the eval loop. Cause was entirely in `BiomedParseDataset.__init__`, not decode:
  it rebuilt the sample index by globbing the raw PNG tree (`_discover_sources` at depths
  1–4 = 67s) and calling `os.path.exists` on the sibling image of every one of ~179k masks.
  Measured ~37ms/stat over this NFS mount → ~6600s (~1.8h) of pure stat() before iteration.
- The pre-resized store already ships a per-dataset `index_{S}.npz` (written by
  `scripts/datasets/biomedparse/to_npz.py`) recording image/mask paths relative to
  data_root in the exact discovery order. The fast pixel-path read those stores but the
  index was still rebuilt from scratch every run.
- Fix (`src/datasets/biomedparse.py`): when `use_npy`, discover datasets from the store
  tree (`_discover_stores`: ~41 dirs under `_npy/<split>`) and rebuild each sample index
  from its `index_{S}.npz` (`_index_from_npz`) — zero PNG globs, zero per-mask stats. The
  raw PNG-tree scan (`_discover_sources` + per-mask `os.path.exists`) is kept as the
  fallback when `use_npy=False` or no store exists for the requested size.
- Result: test-split init 179,284 samples / 41 datasets in **35.6s** (was >900s, timed out),
  all on the npy fast-path. Sample count + ordering match the PNG path (to_npz uses the
  identical `_collect` discovery, so `index_{S}.npz` rows align with the image/mask stacks).
  Self-test (PNG fallback) still passes; item shapes/ranges/binary masks spot-checked.

## 2026-06-29 — omniSynth: in-context Omniglot grid dataset wired into 2D pipeline
- New dataset `src/datasets/omniSynth/`: in-context 2D segmentation from Omniglot characters
  arranged on a grid. A task = one target character class; each scene is a 4x4 grid where
  k cells contain the target (mask = ink pixels of the target characters within those cells)
  and the remaining cells hold distractor characters from other classes.
- Wired as `data.source=omnisynth` in `experiments/2d/common.py:build_dataset`.
- Hydra config: `configs/experiment/2d/synth/omniglot.yaml` (under `@package synth`).
  Key knobs: `target_mode` (identical | aug | class — controls what "same target" means),
  `k_min`/`k_max` (number of target cells per scene), train=background alphabets,
  val/test=evaluation alphabets (split by `val_test_split`).
- `paths.omniglot` added to `configs/config.yaml` and both cluster configs
  (`configs/cluster/nfs.yaml`, `configs/cluster/meta.yaml`).
- Integration test: `src/datasets/omniSynth/test_integration.py` (run directly, no pytest).

## 2026-06-28 — OOD generalization study of imagepfn_zoom (hard_diverse checkpoint)
- Probed how the `2026-06-22_kind-durian-59` imagepfn_zoom checkpoint (trained on
  `synth=hard_diverse`) generalizes outside its training distribution, on the held-out
  val shape pool. In-dist anchor Dice = 0.648.
- New eval configs: `configs/experiment/2d/synth/{ood_appearance,ood_conditions}.yaml`.
  Appearance-only OOD (noise/contrast/texture pushed OOD, context correspondence kept
  in-dist so ctx_dice unchanged) → Dice 0.648→0.261: a CLEAN −0.39 generalization gap.
  Combined OOD → 0.071 but ctx_dice collapsed 0.164→0.045, so partly an artifact (no
  usable context), not pure generalization.
- Per-axis sweep (`results/controlsynth/sweep/run_sweep.sh`, 50 runs, num_tasks=2000):
  each knob varied around its single training point. Used realized ctx_dice to split
  info-preserving axes (drop = model brittleness) from ctx_dice-falling axes (drop partly
  inherent). Findings: genuine weaknesses are appearance knobs only —
  foreground_contrast (−0.30) ≫ noise (−0.19) ≈ texture (−0.14); support_query_scale is
  robust to 2.5× trained; region_size extrapolates UPWARD better than in-range; ambiguity
  axes handled gracefully. No catastrophic overfit (easy-side extrapolation flat-or-better).
- Recommendation: randomize live appearance knobs during training (currently pinned to
  single scalars in hard_diverse). Full writeup + figures: `results/controlsynth/sweep/SUMMARY.md`.

## 2026-06-24 — BiomedParse dataloading: benchmark + pre-resize pipeline
- Benchmarked src/datasets/biomedparse.py loading. Bottleneck is purely PNG decode:
  every image/mask is a 1024x1024 RGBA PNG; decode->128 grayscale = ~33 ms/img, all CPU
  (zlib inflate + RGBA->L; resize ~2 ms, NFS negligible — cold≈warm, header-open 0.36 ms).
  Each __getitem__ does up to 2*(K+1)=8 decodes. Real shuffled DataLoader throughput:
  2.7 / 22 / 33 / 104 img/s at num_workers 0 / 8 / 16 / 32 — fully decode-bound, scales
  linearly with cores. Corpus = 1.23M PNGs (MSD 605k, amos22 240k dominate).
- Fix: decode once, store pre-resized uint8. memmap'd .npy reload = 0.043 ms/img (~760x).
  At 128px the whole corpus is ~20 GB (fits the 995 GB RAM / OS page cache); one-time
  convert ≈ 20 min on 32 cores.
- New scripts/datasets/biomedparse/to_npz.py: parallel PNG->uint8, per-dataset memmap-able
  images_{S}.npy / masks_{S}.npy stacks + index_{S}.npz (paths rel to data_root). NOT a
  monolithic .npz like totalseg2d's — NpzFile can't be memmapped, so 32 persistent workers
  would each hold a full RAM copy; standalone .npy lets them share one OS-cached copy
  (COW-safe). Row order + resize semantics reproduce the dataset exactly (image L/BILINEAR/
  /255; mask L/NEAREST/>0); verified bit-exact (max diff 0.0) and row==image_idx aligned.
- BiomedParseDataset memmap fast-path (done): new use_npy/npy_root args; per dataset,
  _maybe_load_store() memmaps images_{S}.npy/masks_{S}.npy from <data_root>/_npy and uses
  them iff shapes match the discovered (n_imgs, n_masks) — else that dataset silently
  decodes PNGs (per-dataset fallback). Rows align by construction (same sorted-glob order):
  image row == img_idx, mask row == per-ds kept-mask counter (self._mask_row, populated
  beside self.mask_path, so no new COW-heavy dict). Verified bit-exact vs decode (max diff
  0.0 over images/labels/context). Throughput on ACDC+BreastUS+amos22 @128/K=3: 142 -> 0.20
  ms/item single-process (~700x); DataLoader now FASTEST at num_workers=0 (2822 img/s) and
  SLOWER with workers (1545 @16) — load is no longer the bottleneck, so worker IPC dominates
  (revisit worker count once heavy augmentation is in the pipeline). Store built at 128px
  for both splits (~5 GB, 41 dataset sources each).

## 2026-06-24 — eval.py: imagepfn_zoom backend
- experiments/2d/eval.py now evaluates zoom-chain checkpoints (model=imagepfn_zoom),
  mirroring the patchset_pfn (is_multilevel) backend: reads arch+sample+stage1 path from
  the best.pt, loads the frozen stage-1 ImagePFN + UniverSeg encoder, rebuilds the
  warm-started external-features ImagePFN hops (new helper build_zoom_chain — same shapes
  as train.py:build_zoom_models, no warm-start since trained weights are loaded), and
  drives them with run_zoom_chain. Prediction = final hop's refined_full (already native
  H, no upsample); scored with the same hard/soft Dice. Figure low-res panel uses the
  stage-1 coarse seed (R0×R0). Added "imagepfn_zoom" to the top-of-file src-precedence
  argv gate. Run: python experiments/2d/eval.py model=imagepfn_zoom eval.checkpoint=<best.pt>

## 2026-06-22 — Zoom-refinement variant (shared ImagePFN arch)
- New stage-2 refinement path selectable via arch.refine_arch=imagepfn_zoom (default
  patchset, unchanged). Instead of PatchSetPFN on scattered patches, a warm-started
  ImagePFN refines a contiguous square crop ("zoom"), so the refiner is the SAME arch as
  stage-1 — isolating refinement from the change of model class.
- experiments/2d/multilevel/bbox.py: max_sum_window / gt_window (s×s square of largest
  predicted mass / densest GT), crop_resize (batched grid_sample crop+resample),
  composite_window (write crop back). Unit-tested in test_bbox.py.
- src/models/pfn_seg_2d.py ImagePFN: backward-compatible use_external_features (consume
  precomputed features, no internal encoder) + forward(image_feats=, seed_query_mask=).
  Defaults reproduce prior behavior; test_imagepfn_modes.py + test_pipeline.py green.
- experiments/2d/multilevel/zoom_pipeline.py run_zoom_chain: frozen stage-1 coarse pred →
  encode maps ONCE → per hop crop-pool maps to the bbox (same encode-once features the
  PatchSetPFN chain uses), seed the query with the cropped coarse pred, composite the
  upsampled R0 output back at native H. test_zoom_pipeline.py.
- train.py: refine_arch switch (build_zoom_models warm-start, chain_fn dispatch in
  train_epoch, run_eval_zoom with native-H dice + in-bbox refine delta). Config
  configs/experiment/2d/multilevel_zoom.yaml (single 64px hop). Verified 1-epoch smoke run.
- bbox.max_sum_window / gt_window: empty maps (densest s×s window holds <0.5 mass — e.g. an
  all-background prediction or empty GT) CENTER the crop instead of collapsing to corner
  (0,0). Per-sample within a batch. Covered by test_bbox.py (empty + mixed-batch cases).

## 2026-06-22 — eval.py: allow eval-time override of pfn_seg token-grid resolution

- The pfn_seg_2d eval branch rebuilt ImagePFN using `resolution`/`input_patch_size`
  read *only* from the checkpoint's stored `arch`, so passing `arch.resolution=...`
  on the eval CLI was silently ignored.
- Added `eval.resolution` / `eval.input_patch_size` (default null → use ckpt) to
  eval_base.yaml; eval.py now prefers them over the checkpoint's arch and reflects
  the value actually used in the logged run_cfg. ImagePFN's learnable
  `pos_embed` is shaped (1,1,resolution²,e), so overriding to a resolution the
  weights weren't trained at fails loudly at load_state_dict (verified) — no silent
  wrong-weight eval. (patchset_pfn resolution comes from cfg.sample.resolutions, a
  separate mechanism — unchanged.)

## 2026-06-22 — Deterministic eval context for the real-image 2D datasets

- medsegbench / biomedparse / totalseg2d picked their K context pairs via
  `cow_index.sample_context`, which used the global `random` module. That is
  worker-safe (PyTorch reseeds stdlib `random` per worker, unlike numpy), but it
  also meant eval context drifted run-to-run and epoch-to-epoch — context-sampling
  variance leaking into eval Dice.
- `sample_context(cand, exclude, k, rng=None)` now takes an optional
  `random.Random`. Each dataset gained `deterministic` (default `split != "train"`,
  mirroring controlSynth) and passes `random.Random(idx)` on non-train splits, so a
  given eval target always draws the same context. Train keeps `rng=None` → global
  module → fresh, worker-distinct draws each epoch. controlSynth already had its own
  deterministic eval path (SeedSequence), so it was unchanged.

## 2026-06-22 — Fix UnboundLocalError in pfn_seg.run_eval

- `run_eval` accumulated val loss into `total_loss`/`nl` but referenced an
  undefined `val_loss` in the `tqdm.write` print (line 215), crashing the first
  eval. `val_loss = total_loss / max(nl, 1)` was only computed later (before the
  wandb.log). Moved that computation up before the print and removed the
  now-duplicate line.

## 2026-06-20 — Multilevel encoder: encode-once-pool-many (lossless ~2× speedup)

- **Problem**: `pipeline.run_chain` re-ran the FULL frozen encoder once per hop
  (`encode_grid(encoder, all_images, grid)` for each grid in resolutions[1:], e.g.
  32/64/128 → 3 forwards). But the encoder's stage maps are resolution-independent —
  `out_size` only feeds the final `adaptive_avg_pool2d`. So 2 of every 3 encoder passes
  were redundant.
- **Fix**: split both encoders into `encode_maps(images)` (resolution-independent stage
  maps) + `pool_maps(maps, out_size)` (pool/concat → reduce). `forward` now = pool∘encode
  (unchanged for single-resolution callers: ImagePFN, eval feature paths). `run_chain`
  calls `encode_maps` ONCE, then `pool_grid` per hop. Fallback to per-hop `encode_grid`
  for plain-callable encoders without `encode_maps` (e.g. test stubs).
- **Bit-exact**: UniverSeg and DINOv3 (incl. stage_l2norm and reduce=random) match the
  old per-hop output to max|Δ|=0 at every grid; unfitted-PCA guard preserved in pool_maps.
- **Speedup**: UniverSeg, 64 imgs, hops [32,64,128]: 4792 → 2232 ms (~2.1×) on CPU.
- **Tradeoff**: the 4 native-res stage maps are now held across the hop loop (stage0
  dominates, ~Bᵀ·64·H·W). Worth it vs 3× encoder compute; note if GPU mem-bound.
- Not the encoder context path (that's load-bearing, see prior analysis); this is the
  redundant re-encode across resolutions.

## 2026-06-20 — Unify encoder-feature normalization across pfn_seg / multilevel

- **Problem**: encoder (UniverSeg/DINOv3) features were normalized inconsistently.
  `ImagePFN` (pfn_seg stage-1) per-channel z-scored features by *context-row* stats
  (matching the TabPFN feature_sim backend in eval.py), but the multilevel pipeline's
  `encode_grid` fed *raw* encoder magnitudes straight into PatchSetPFN — no normalization.
  Worst for DINOv3 `feature_level=all`, whose 4 concatenated stages differ ~10–100× in
  magnitude (a known imbalance the `encoder_stage_l2norm` knob only partly addressed).
- **Fix**: single shared helper `standardize_by_context(feat, n_context)` in
  `src/models/pfn_seg_2d.py` — per-channel z-score using the first `n_context` rows'
  stats, applied to all rows (query standardized in the context's frame), clamp ±10.
  `ImagePFN`'s encoder path now calls it (raw-pixel path keeps its scalar normalization);
  `pipeline.encode_grid` now calls it too (signature gains `n_context`; `run_chain` passes K).
- **DINOv3 note**: per-channel z-score makes the cross-stage *magnitude* imbalance moot,
  so `encoder_stage_l2norm` is now largely redundant (it only rebalances channel *counts*,
  a representational weighting choice, not a scale bug). `encoder_imagenet_norm` is an
  input-side normalization (gray→RGB, ImageNet mean/std) and is orthogonal — unchanged.
- Verified: test_pipeline passes; helper gives ctx per-channel mean~0/std~1; both ImagePFN
  paths run. No retraining done — existing checkpoints will see a shifted feature frame at
  the multilevel chain, so chains should be re-trained/re-evaluated on the unified path.

## 2026-06-19 — controlSynth: fix context_copy & noise_level; qualitative panels

- **`context_copy_fraction` collapse FIXED** (`dataset.py`). At copy=1.0 UniverSeg gave
  Dice exactly 0. Diagnostics (UniverSeg forward) showed it is NOT pixel-identity: even a
  noised near-copy and mask-rolled variants collapsed, while real contexts scored 0.86.
  Root cause: copying the target *frame* makes the **background** match the query as well
  as the foreground, so the fg loses its distinctiveness for a context-matcher (the fg is
  normally "the only region consistent across contexts"). Redefined the knob: a fraction
  of contexts are now **pristine exemplars** — rendered with `shift_scale=0.1` (near-zero
  deformation) but a **fresh background** (`_make_subject(..., shift_scale=)`; copy logic
  moved out of `_apply_context_difficulty` → renamed `_apply_context_consistency`). Now
  non-degenerate (copy=1.0 → Dice ~0.80); a mild ease knob (the easy baseline is already
  context-saturated, so little headroom — eases more at harder operating points).
- **`noise_level` now monotone-correct** (0.0 easiest 0.80 → 1.0 hardest 0.68). The earlier
  baseline-dependent sign-flip resolved itself once the contrast redesign made the fg
  properly salient (images less OOD-pathological), so added noise only ever hurts. No
  separate noise code change needed.
- **Qualitative panels** (`synth_benchmark.py --plot`): per knob, renders the target
  (GT green / pred red contours) + K context images (fg mask overlay) at several knob
  values, annotated with mean UniverSeg Dice → `results/2d/synth_benchmark/<ts>/panels/
  <knob>.png`. Lets the difficulty curves be read against what the images look like.

## 2026-06-19 — controlSynth difficulty study: findings + foreground_contrast redesign

Four-round sensitivity study (UniverSeg baseline) consolidated in
`docs/datasets/controlSynth_difficulty_findings.md`. Headline: context-quality
(consistency/shift) and `region_size` are the real difficulty levers; the spec's
identification axis (`task_ambiguity`) is **inert** for a context-matcher (confirmed via
ambiguity×intensity and ambiguity×region_size grids — penalty ~−0.05 at every fg size);
`foreground_contrast` was inverted and `context_copy_fraction` collapses UniverSeg to 0.

**`foreground_contrast` redesign** (`src/datasets/controlSynth/appearance.py`,
`config.py`, `task.py`, `dataset.py`):
- Original bug: `gmm_fill` pushed *background* regions to the [0,1] extremes as contrast
  rose (bg saturation 5%→52%), leaving the fg a bland mid-grey blob → higher contrast was
  *harder* (Dice 0.60→0.23, inverted).
- Fix (a): background means now stay in a fixed central band [0.25,0.75]; the foreground
  is pushed `gap(contrast)` toward an extreme, so the **fg** owns the salient extremes.
- Fix (b, deeper): the fg's side is now a **task-level constant**
  (`task.make_base_geometry` records `meta["appearance_sign"]`, threaded through
  `dataset._make_subject` → `gmm_fill(..., fg_sign=)`). Previously the per-subject random
  extreme made high contrast push each subject's fg to an *independent* side, so the
  context no longer matched the target (the true cause of the inversion).
- `map_contrast_gap` range trimmed 0.55→0.40 (max gap 0.45) so extreme contrast stays
  clear of the background band rather than saturating into its noise tail.
- Result: correctly oriented axis — low contrast hardest (Dice 0.80, fg
  intensity-invisible → found via shape/context), rising to a ~0.85 plateau.

## 2026-06-19 — controlSynth difficulty benchmark (UniverSeg baseline)

New `experiments/2d/synth_benchmark.py` — measures how each controlSynth knob affects
task difficulty for the zero-training UniverSeg baseline, and surfaces *why* a value is
easy/hard (not just that it is).

- **Method = one-factor-at-a-time (OFAT)**: pin every knob at a moderate baseline
  (`BUILD_DEFAULTS`/`LIVE_DEFAULTS` in the script), sweep ONE knob across a value grid,
  so each knob's marginal effect on Dice is isolated. Morphology-specific knobs
  (thinness/tortuosity/branching → tubular; scattered_count/clustering → scattered) are
  swept with that morphology fixed; the rest use a clean `blob`. Live knobs reuse one
  frozen geometry bank (only the per-subject path changes) → cheap.
- **Per-subject record**: each evaluated subject logs its full param vector (all build +
  live knobs), realized stats (`fg_frac`, mean target↔context Dice `ctx_dice`), axis
  loadings, and Dice → `per_sample.csv` for arbitrary offline analysis.
- **Outputs** (→ `results/2d/synth_benchmark/<ts>/`): `per_sample.csv`, `summary.csv`
  (per knob×value: n, dice mean/median/std, fail_rate=Dice<0.1, fg_frac, ctx_dice),
  `difficulty_curves.png` (Dice + fail-rate vs value per knob), `morphology_difficulty.png`,
  and `report.txt` — knob sensitivity ranking (Dice spread, easiest→hardest value),
  per-knob Spearman(Dice, fg_frac)/Spearman(Dice, ctx_dice) to attribute difficulty to
  foreground-shrinkage vs. context-informativeness, and pooled global drivers.
- CLI: `--num_tasks/--subjects/--image_size/--context_size/--batch_size/--workers`,
  `--knobs <subset>`, `--quick` (smoke). Default 48 tasks × 24 subjects/task.
- **Plumbing**: `common.collate` now passes through per-element `meta` when the dataset
  provides it (controlSynth attaches its per-subject knob vector there); additive and
  guarded, so non-synth loaders are unaffected. `eval.py` gains an opt-in
  `eval.synth_csv` that dumps the same per-element synth-param rows from a normal eval
  run (mirrors the existing `patch_csv` pattern).
- Verified end-to-end with `--quick` (UniverSeg loads, sweeps run, all 5 artifacts
  written, CSV carries the full knob vector per subject).

## 2026-06-19 — Unify 2D dataset-source loading

Deduplicated the dataset-loading logic that had drifted across the `experiments/2d`
scripts (three near-identical loader builders with divergent source support: eval
`common.build_loader` did all 3 sources, `pfn_seg` did medsegbench+synthetic,
`multilevel/train` did medsegbench only).

- **`common.py`**: two new shared functions are now the single source of truth.
  - `build_dataset(cfg, split)` — dispatches on `cfg.data.source`
    (`medsegbench | biomedparse | synthetic`); folds in the controlSynth config
    wiring (previously duplicated in `common` *and* `pfn_seg._build_synth_dataset`).
    biomedparse/controlSynth imported lazily so the medsegbench path stays light.
  - `make_loader(ds, cfg, split, shuffle)` — shared subsample + `TaggedDataset`/
    `collate` + sampler policy: non-train splits subsample to `eval.max_per_label`;
    train uses `train.batch_size/workers` + optional `RandomSampler`
    (`data.max_train_samples`); val/test use `eval.batch_size/workers`.
  - `build_loader(cfg)` (eval) = `make_loader(build_dataset(cfg, cfg.data.split), …)`.
- **`pfn_seg.py` / `multilevel/train.py`**: `build_split_loader` collapses to
  `make_loader(build_dataset(cfg, split), cfg, split, shuffle)`. Removed the
  copy-pasted source dispatch, synth wiring, subsample block, and DataLoader assembly.
  Net effect: **every source is now available to every script** (multilevel/pfn_seg
  can train on biomedparse or synthetic; eval already could).
- **Configs**: the `synth:` block moved to a shared Hydra group
  `configs/experiment/2d/synth/default.yaml` (`@package synth`), pulled into
  `base`, `pfn_seg`, and `multilevel` via `defaults: [synth: default, _self_]`
  (`feature_sim` inherits it through `base`). Added `data.source` to `multilevel.yaml`.
  One place to edit synth defaults instead of three.
- Verified: all 4 configs compose with `cfg.synth` present; the unified path builds
  working train (bs=train) and val (bs=eval, difficulty-binned names) synthetic
  loaders and yields correctly-shaped batches.

## 2026-06-19 — controlSynth V1: difficulty-controlled synthetic in-context dataset

New package `src/datasets/controlSynth/` — a minimal on-the-fly procedural generator
for studying the impact of training on synthetic data (`experiments/2d/pfn_seg.py`).
Implements the *minimal generator* slice of `docs/datasets/controlSynth.md`.

- **Three orthogonal config axes** (`config.py`): `DiversityConfig` (num_tasks),
  `DifficultyBuildSpec` (frozen geometry), `DifficultyLiveConfig` (per-subject), plus
  documented monotone `[0,1]→param` mappings so a single difficulty knob can be swept
  with the rest pinned.
- **All five morphologies** (`shapes/`): blob/elongated/annular (`blob.py`), tubular via
  space colonization → caliber taper → vectorized capsule rasterization (`vessel.py`),
  scattered point process hardcore↔Poisson↔clustered (`scattered.py`); plus
  morphology-independent `boundary.py`, `area.py`, and `distractors.py` (geometry side of
  `task_ambiguity`).
- **No LMDB.** Base geometry is precomputed in RAM at dataset init (`geometry.GeometryBank`,
  the in-memory analog of the spec's store), deterministic in `master_seed`; only the cheap
  live path (deform → GMM fill → noise from a precomputed Perlin bank) runs per item.
  Vessels are therefore generated once per task, never per `__getitem__`.
- **Determinism** (`dataset.SynthICLDataset`): train draws fresh entropy (infinite
  subjects); val/test derive every subject seed from
  `(eval_seed_namespace, task_id, sample_index)` → byte-identical eval set. Task ids are
  split into disjoint train/val/test pools (held-out anatomies). Returns the MedSegBench
  4-key dict + `meta`, and exposes `.samples`, so existing `TaggedDataset`/`collate` work
  unchanged.
- **Integration**: `data.source=synthetic` switch in `pfn_seg.py:build_split_loader` and
  `common.py:build_loader`; a `synth:` config block in `configs/experiment/2d/pfn_seg.yaml`.
- **Difficulty-stratified metrics**: val `.samples` names carry the swept difficulty tag
  (`difficulty_tag()` from `build_spec.bin_factor`, e.g. `synth/blob/amb0.40`), so the
  existing per-dataset grouping in `run_eval` logs `dice/dataset/synth/blob/amb0.40` (and
  `dice_ds`/`dice_ds_soft` variants) with no eval-loop changes. `dice/mean` is unaffected.
- **Run naming**: `pfn_seg.py` now lets W&B auto-generate the run name (single
  `wandb.init(mode=…)` call) and saves checkpoints under `{date}_{wandb_run_name}` — the
  same convention as `multilevel/train.py`. The full `synth` config is logged to W&B so a
  difficulty sweep is comparable across runs by `config.synth.*`.
- **Verification** (`scripts/controlsynth_smoke.py`): visual grids per morphology
  (`results/controlsynth/*.png`), val determinism, and a DataLoader+ImagePFN forward all
  pass. End-to-end sanity: easy config reaches val Dice 0.85 in 8 epochs; hard config
  (small region + `task_ambiguity`) stays near 0 as expected.
- **Task diversity in one run**: `build.morphology` accepts a `{type: weight}` mixture
  (spans all five shape families); `mode=per_task_sampled` draws the factors listed in
  `build.sampled` (`{factor: [lo, hi]}`) per task, so a single run spans a difficulty
  range. The val grid then bins `bin_factor` into `n_bins` buckets → difficulty-response
  curve from one run (keys like `dice/dataset/synth/tubular/amb0.30`). NB: inline-dict
  yaml needs spaces (`{blob: 1, …}`); CLI overrides of `sampled` use `++` to replace the
  empty default (`'++synth.build.sampled={task_ambiguity:[0.0,0.8]}'`).
- **Deferred to later sub-projects** (per spec): LMDB store + precompute CLI,
  `eval_harness.py` + oracle UNets, clDice/NSD/size-stratified metrics, `mixed.py`
  (real+synth MixedDataLoader + curriculum). `resolve_difficulty` mode `binned` (a fixed
  labeled eval grid) still raises `NotImplementedError`; `fixed` and `per_task_sampled`
  are wired.
- *Known V1 limitation*: vessel realized area is dominated by the min-caliber floor
  (region_size weak for tubular), and thin vessels fragment under deformation — vessel
  Dice is indicative only until clDice lands.

## 2026-06-18 — Cheap channel-dim reduction for DINOv3 features

`DINOv3FeatureEncoder` gained an `encoder_reduce` knob (applied after pooling, so the
reported `feature_dim` — and thus the downstream `image_embed` input — shrinks):
- `none` (default), `grouppool:<d>` (adaptive_avg_pool1d over channels, zero params,
  no fit), `random:<d>` (frozen Gaussian / Johnson–Lindenstrauss projection, no fit),
  `pca:<d>` (PCA fit once on data, cached to disk).
- `encoder_stage_l2norm` L2-normalizes each stage map before the `all` concat, fixing
  the channel-count/scale imbalance between stages (zero params).
- PCA: `ensure_pca(image_iter)` loads the cached projection
  (`<hf_cache>/reductions/dinov3_<variant>_l<level>_<raw|l2>_pca<d>.pt`) or fits + caches
  it. `reduce_proj`/`reduce_mean`/`reduce_fitted` are registered buffers → for the
  ImagePFN path they ride in the checkpoint state_dict; the multilevel chain encoder
  (not in the checkpoint) re-loads from the disk cache at eval. Fit is wired into
  `pfn_seg.py` and `multilevel/train.py` (guarded by `needs_pca_fit`) and the
  `eval.py` chain branch (cache-hit, iterator untouched).
- Cost is negligible — all reductions add ~1–2 ms to the ~39 ms encode (B=64, 128px,
  bf16); the backbone dominates. The savings are downstream: `image_embed` input
  1920→d, smaller cached/transferred features, and less overfit on a 1920-wide input.
  Select via e.g. `arch.image_encoder=dinov3 arch.encoder_reduce=pca:256`.

## 2026-06-18 — Encoder benchmark: UniverSeg vs DINOv3 (FLOPs / VRAM / latency)

`experiments/2d/bench_encoders.py` times the frozen encoders under matched conditions
(same batch / image size / token grid, same precision, same compile) reporting forward
GFLOPs (FlopCounterMode), peak VRAM, ms/iter. RTX A4000, level=`all`, B=64.
- **128px → grid 16**, bf16: UniverSeg 260 GFLOPs / 657 MiB / 21.8 ms ;
  DINOv3-cnvnxt-base 642 GFLOPs / 658 MiB / 39.0 ms. DINOv3 ≈ 2.5× FLOPs, 1.8× latency.
- All four ConvNeXt variants (`all`, B=64, 128px, bf16): added `tiny`/`small` to `_DIMS`
  (both dims 96/192/384/768 → 1440ch; differ only in stage-2 depth 27 vs 9):

  | variant | feat_dim | GFLOPs | VRAM MiB | ms/iter |
  |---|---|---|---|---|
  | dinov3-tiny  | 1440 | 186  | 356  | 17.5 |
  | dinov3-small | 1440 | 363  | 439  | 27.3 |
  | dinov3-base  | 1920 | 642  | 658  | 39.1 |
  | dinov3-large | 2880 | 1436 | 1223 | 63.0 |
  | (universeg)  | 256  | 260  | 657  | 21.8 |

  **dinov3-tiny is both faster (17.5 vs 21.8 ms) and lighter (356 vs 657 MiB) than
  UniverSeg** while giving a 1440-dim feature — an attractive default. tiny vs small is
  same feature_dim at ~half the FLOPs (stage-2 depth).
- **256px → grid 32**, bf16: UniverSeg 1039 GFLOPs / **2613 MiB** / 86.7 ms ;
  DINOv3 2567 GFLOPs / **1606 MiB** / 142 ms. VRAM **crosses over**: UniverSeg keeps
  high-res 64-ch feature maps through all stages (activation-heavy), while DINOv3's stem
  downsamples /4 immediately → param-heavy but activation-light, so it scales better in
  memory with resolution despite more FLOPs.
- `compile bf16` == `eager bf16` to within noise: both encoders are `@torch.compiler.disable`
  (the pipeline runs them eager; adaptive_avg_pool with symbolic windows can't lower), so
  `torch.compile` graph-breaks at the encoder and gives no speedup. Compute lives in the
  frozen backbone, not in launch overhead.
- feature_dim asymmetry at `all`: UniverSeg 256 vs DINOv3 1920 — inherent to the concat;
  for a compute-matched A/B pick a single `feature_level` per encoder.

## 2026-06-18 — DINOv3 ConvNeXt feature encoder + `image_encoder` factory

Added `DINOv3FeatureEncoder` (`src/models/pretrained_encoders.py`) to study the impact
of encoder architecture/pretraining on the in-context segmentation model. Frozen
`facebook/dinov3-convnext-{base,large}-pretrain-lvd1689m` backbone (ConvNeXt CNN, DINOv3
SSL on LVD-1689M), matching `UniverSegFeatureEncoder`'s interface exactly:
`forward(images, out_size) → (N, feature_dim, out_size, out_size)`, so it drops into the
existing `image_encoder` injection seam in `ImagePFN` / the multilevel chain.
- Fully convolutional → size-agnostic (runs at native H, pooled to the token grid),
  same property the Strategy-A eval relies on. Stage maps: channels
  `[128,256,512,1024]` (base) at strides `[4,8,16,32]`; `level` picks a stage (0=highest
  res) or `"all"` → concat (feature_dim 1920 base / 2880 large).
- Adapts DINOv3's input contract: 1-ch grayscale → repeated ×3, optional ImageNet
  normalization (`encoder_imagenet_norm`, default on; off to let ImagePFN's own
  per-context standardization be the sole norm). Loaded `local_files_only` from the NFS
  HF cache (`…/ANALYSIS_20251122/checkpoints`).
- New `build_image_encoder(arch, device)` factory dispatches on `arch.image_encoder`
  (`patch`→none, `universeg`, `dinov3`/`dinov3-base`/`dinov3-large`) and returns
  `(encoder, feature_dim)`. Replaced all 5 hand-rolled encoder-construction branches
  (`pfn_seg.py`, `multilevel/train.py` ×2, `eval.py` ×3). The multilevel **chain**
  encoder defaults to `universeg` (back-compat) and is overridable via
  `arch.image_encoder=dinov3`.
- Select at train/eval time, e.g. `arch.image_encoder=dinov3 arch.feature_level=2`.
  Note feature_dim changes (UniverSeg `all`=256 vs DINOv3 `all`=1920), which sets the
  `image_embed`/PatchSetPFN input dim — a fresh checkpoint, not warm-startable from a
  UniverSeg one.

## 2026-06-18 — Strategy-A eval: encode at a different resolution, grids fixed

New `eval.encode_size` knob (`base.yaml`, `null` = checkpoint size) lets `pfn_seg_2d`
and `patchset_pfn` eval **feed images at a different size while keeping every token grid
fixed**. Only the frozen, fully-convolutional UniverSeg encoder runs at the new size,
then pools into the unchanged grids; the output is upsampled to native for scoring.
Decouples *encoder input resolution* from the baked patch/token grid.
- `eval.py`: model is built at the checkpoint's `model_size`; the loader serves
  `encode_size`. Stage-1 (and the pfn_seg model) get `patch_size` rescaled to
  `encode_size // resolution` so `Hp = H//P` stays == `resolution` (else the patch
  count ≠ `self.N` and the forward reshape crashes — see the H=256 probe). The chain
  builds `PatchSetPFN.mask_patch_size` from `model_size`, not the served size. Final
  prediction upsampled from the model grid (128) to native before Dice.
- `multilevel/pipeline.py`: `refine_level` takes the patch-tile size `p` from
  `model.mask_patch_size` instead of `label.shape[-1] // grid_res`, so a larger eval
  image is resized down inside `_mask_tiles` rather than producing a `p×p` that
  mismatches `mask_embed`. No-op at the training size (the two are equal).
- wandb config now logs **`encoder_input_size`** (served pixels) and **`model_size`**
  (the resolution the token grids were built at) instead of the ambiguous `image_size`,
  across all four backends. Run name appends the encoder size only when it differs:
  `patchset_pfn_s128e256_k3` (model 128, encoder 256) vs `patchset_pfn_s128_k3` default.
- `encode_size` must be divisible by the stage-1 resolution (16). Attention sequence
  lengths are identical to default; only the conv encode scale (and thus FLOPs/time)
  changes — this measures feature-scale robustness, not higher-res output (output grid
  stays 128). Caveat: feature normalization uses support stats, which shift off the
  training scale, so expect small ± rather than a guaranteed gain.
- Verified on `fast-microwave-31` / `busi val` (same 4 samples, seed 0):
  chain 128→`dice 0.9002`, 257 GFLOPs, 234 ms ; chain 256→`dice 0.9225`, 452 GFLOPs,
  532 ms. pfn_seg 128→`0.8318`, 85 GFLOPs ; pfn_seg 256→`0.8692`, 133 GFLOPs. Default
  (`encode_size=null`) is a bit-for-bit no-op.

Launch: `… model=patchset_pfn eval.checkpoint=<run>/best.pt
eval.stage1_checkpoint=<stage1>/best.pt eval.encode_size=256`

## 2026-06-18 — `patchset_pfn` (multilevel chain) backend in the 2D eval

`experiments/2d/eval.py` gains a 4th backend, `model=patchset_pfn`, so the multilevel
coarse→fine chain is benchmarked under the **same conditions** as the other models
(per-dataset Dice, inference ms/item, FLOPs). Final native-resolution Dice only.
- Reconstructs the full system the way `multilevel/train.py` does: frozen stage-1 ImagePFN
  (`load_stage1`, duplicated from train) + frozen `UniverSegFeatureEncoder` + a `ModuleList`
  of trained `PatchSetPFN` hops. `arch` + `sample` (ladder/budgets) are read from the
  checkpoint and injected into `cfg`; the chain runs via `pipeline.run_chain` with
  `cfg.sample.eval` and `eval_deterministic`. Prediction = `outputs[-1]["refined_grid"]`
  reshaped to native (final hop grid == image_size), i.e. exactly training's
  `dice_r{native}/mean`.
- State-dict load uses `.replace("_orig_mod.", "")` (compiled `ModuleList` buries the prefix
  mid-key). FLOPs counted over the whole chain (stage-1 + encoder + all hops).
- The stage-1 path isn't in older checkpoints → new `eval.stage1_checkpoint` fallback
  (`base.yaml`); `train.py` now also records `stage1_checkpoint` in `best.pt` for future runs.
- `src`-shadowing guard (patch_icl src must beat ic_segmentation's) extended to fire on
  `patchset_pfn` too. image_size + context_size synced from the checkpoint.
- Verified end-to-end on `fast-microwave-31` (ladder 16→32→64→128): `busi val` →
  `dice/mean=0.9002`, 285 ms/item, FLOPs logged.

Launch:
`.venv311/bin/python experiments/2d/eval.py model=patchset_pfn
eval.checkpoint=<run>/best.pt eval.stage1_checkpoint=<stage1>/best.pt
data.split=val eval.max_per_label=20`

## 2026-06-18 — Wired BiomedParse into the 2D eval + macro-average

`experiments/2d/eval.py` can now run on BiomedParse (non-breaking for MedSegBench):
- New `data.source: medsegbench | biomedparse` knob (`configs/experiment/2d/base.yaml`);
  `common.build_loader` branches on it. BiomedParse has only `train`/`test`, so pass
  `data.split=test`.
- `common.log_summary` now also emits **`{prefix}/macro`** — the mean over per-`(dataset,
  label_value)` cell means (for BiomedParse = `(dataset, target)`). Weights every cell equally
  so multi-label datasets can't dominate the headline like the per-sample micro-average lets
  them (the original m2caiseg 0.33-vs-0.55 weighting bug). Additive: `dice/mean` (micro) still
  printed alongside `dice/macro`.
- Verified end-to-end: `eval.py data.source=biomedparse data.split=test data.dataset=DRIVE
  data.image_size=128` → UniverSeg 0.385 Dice (DRIVE retinal vessel; thin tubular, low as
  expected); loader→model→metrics→summary all run, wandb offline.

Launch (all datasets, capped, macro-averaged):
`.venv311/bin/python experiments/2d/eval.py data.source=biomedparse data.split=test
data.image_size=128 eval.max_per_label=20`

## 2026-06-18 — BiomedParse: cells keyed by (dataset, modality, target)

`cell_of` now returns `(dataset, modality, target)` (was `(modality, target)`) — added
`dataset_of`. Keeps each source separate (no cross-dataset pooling): e.g. amos22 CT-liver and
another CT dataset's liver are distinct cells. Dataset key carries sublevels (`amos22/CT`,
`Radiography/COVID`, `MSD/Task01_*`), so those are already first-class cells. Real 6-dataset
subset: 26 → 27 cells. This is the unit a macro-averaged eval should weight equally.

## 2026-06-18 — BiomedParse dataloader speed: benchmark + ~3x lazy-path optimization

Benchmarked dataloading throughput vs MedSegBench (`experiments/2d/bench_dataloader.py`,
batch_size=32, workers=16, steady-state samples/s after warmup batch):

| image_size | MedSegBench | BiomedParse (before) | BiomedParse (after) |
|-----------:|------------:|---------------------:|--------------------:|
| 128        | ~2160       | ~28 (cold) / ~40     | **~99**             |
| 256        | ~580        | ~41                  | **~92**             |
| 512        | ~161        | ~41                  | **~89**             |

Root cause of the gap: MedSegBench loads pre-resized npz into RAM (per-item = just `/255`),
while BiomedParse decodes a **1024×1024 RGBA PNG per image off NFS**. Micro-bench: decode alone
= **39 ms/img** (irreducible for PNG); the old path added ~27 ms (np channel-mean + torch
interpolate). Optimizations in `src/datasets/biomedparse.py`:
- Decode→gray→resize **inside PIL** (`convert("L")` + `resize`) instead of full-array mean +
  `F.interpolate` (66→45 ms/img).
- **Cache the small resized tensors** (not the 1024² source) keyed by path — context images are
  reused heavily across samples, so hits skip the decode entirely. → ~2.5–3.5× overall.
- Cache budget is now **size-aware** (`cache_size=None` → ~64 MB/worker) so 512px × 16 workers
  doesn't OOM.

BiomedParse is now flat ~90/s (decode-bound). Still ~20× under MedSegBench because each unique
image is a fresh 1024² NFS decode. **Real fix if needed (e.g. for training): pre-resized packed
cache** (one npz/npy per source in RAM, mirroring MedSegBench's fast path) — deferred; for *eval*
~90/s overlapped with GPU forward is likely sufficient.

## 2026-06-18 — BiomedParseData dataloader: reconciled with real extracted layout

Inspected the extracted data at `.../ANALYSIS_20251122/data/biomedparse` (29 datasets;
amos22/MSD still populating) and fixed `src/datasets/biomedparse.py` to match reality — the
prototype's flat-path assumption would have found **zero** data. Findings + changes:
- **Double-nesting:** real layout is `<root>/<DATASET>/<DATASET>/{train,test,...}`, and three
  datasets add one sublevel — `amos22/amos22/CT/…` (modality), `MSD/MSD/Task01_*/…` (task),
  `Radiography/Radiography/{Normal,COVID,Viral_Pneumonia,Lung_Opacity}/…` (class). Added
  `_discover_sources` (depth-bounded glob, depths 1–4) + `_collapse_key` → dataset keys like
  `ACDC`, `amos22/CT`, `Radiography/COVID` (context never mixes across sublevels).
- **`absent.png` sentinel** ("target not present") in PanNuke/kits23/amos22 mask dirs — now
  skipped explicitly.
- `DATA_ROOT` corrected `biomedparse_datasets` → `biomedparse`.
- Filename parsing **confirmed correct** against real names: target = last `_`-token (`+`→space,
  e.g. `left+heart+ventricle`); modalities use `-` internally (`MRI-T1-Gd`) so `_`-split is safe.
- Self-test fixture rewritten to mirror double-nest + a modality sublevel + sentinel + orphan
  mask; passes (20 samples, 5 cells). Real-data smoke test over DRIVE/GlaS/ISIC/REFUGE/amos22/
  Radiography: 41,272 samples, **26 (modality×target) cells** (CT/X-Ray/dermoscopy/fundus/
  pathology), shapes/range/context all correct.
- **Still open:** wire into `experiments/2d/eval.py` `build_loader`; add macro-average over
  `cell_of` cells in the eval aggregation.

## 2026-06-17 — BiomedParseData in-context dataloader (prototype)

`src/datasets/biomedparse.py` (new). `BiomedParseDataset` mirrors `MedSegBenchDataset`'s
contract (`samples` = `(dataset, image_idx, target_int)` 3-tuples; `__getitem__` →
`image/label/context_in/context_out`) so it drops into the 2D eval via `common.TaggedDataset`/
`collate`. Reads the official on-disk layout `<root>/<DATASET>/{train,train_mask,test,test_mask}/*.png`
(populate with `huggingface-cli download microsoft/BiomedParseData`); each `*_mask/` PNG =
one (image, modality, site, target) eval unit, parsed from the filename
(`[IMAGE]_[MODALITY]_[SITE]_[TARGET].png`, `+`→space). PNGs loaded lazily with an LRU cache;
images grayscaled + resized (bilinear) and `/255`, masks `!=0`→fg (nearest). Context = K others
sharing `(dataset, target)`, with-replacement if scarce.
Exposes `modality_of/target_of/cell_of` → the `(modality × target)` grid for macro-averaging
(the point vs MedSegBench's flat micro-average; see prior entry). Ships a synthetic-fixture
`__main__` self-test (no download needed) — passes: 14 samples, 3 cells, shapes/dtype/range
asserted. **Follow-up:** one-line branch in `experiments/2d/eval.py` `build_loader` to select it.

## 2026-06-17 — 2d eval: UniverSeg MedSegBench Dice gap diagnosed (weighting, not model); benchmark research

Investigated why `experiments/2d/eval.py` reports UniverSeg Dice 0.33 on MedSegBench while
`ic_segmentation/scripts/eval.py` reports 0.55 on the same model+data. **Root cause: eval-set
sample weighting, not the model or preprocessing.** Both paths build the *same* `UniverSegBaseline`
(input_size=128) and read the *same* npz directory; per-(dataset,label) Dice agrees between them.
The divergence is aggregation:
- `src/datasets/medsegbench.py` emits one sample per **(image, label_value)**, so the 18-organ
  `m2caiseg` dataset alone = **5,501 / 13,237 samples (42%)** of the eval, mostly near-zero rare
  organs → drags the micro-average to 0.33.
- ic emits one **random label per image** and caps per-dataset → those empty classes are
  down-weighted → 0.55.
Reproduced on a matched capped subset: patch_icl **0.25** vs ic **0.42** (same ~1.6× ratio).
- **Ruled out (A/B tested):** image normalization. Swapping patch_icl's plain `/255` `_to_tensor`
  for ic's percentile-[1,99] clip gave **0.233 vs 0.249** on the identical subset — slightly worse,
  not the cause. Change reverted. (Note: both pipelines also pass through the wrapper's per-sample
  min-max, which masks most of the normalization difference.)
- **Secondary (open):** RGB dermoscopy datasets genuinely differ beyond weighting (isic2016
  0.50 vs 0.73, isic2018 0.49 vs 0.70) — likely the RGB→gray path (patch_icl averages channels to
  uint8 at load vs ic per-sample float) and/or small-N context-sampling variance.
- **Takeaways:** (1) score in-context eval with **macro-average per (modality × shape-class)** +
  per-task sample cap, not flat per-sample micro-average; (2) move to a diverse, balanced
  multi-modality 2D benchmark. Candidate datasets researched and specced in `docs/datasets.md`.

## 2026-06-17 — 2d/multilevel: N-level resolution chain (16→32→64→128)

Extended the 2-level (res-16→res-32) refinement into a configurable coarse-to-fine chain
with per-level weights. Spec/plan: `docs/superpowers/specs/2026-06-17-multilevel-resolution-chain-design.md`,
`docs/superpowers/plans/2026-06-17-multilevel-resolution-chain.md`.
- `pipeline.py`: new `composite_predictions`, `refine_level` (one hop: sample→gather→
  forward→composite), `run_chain` (driver: seeds from stage-1, loops `sample.resolutions`,
  upsamples + DETACHES `refined_grid`/`this_think` between levels, no_grads frozen
  stage-1/encoder). `build_patch_batch` retired.
- `patchset_pfn.py`: `forward(..., return_thinking=True)` returns post-transformer thinking
  pooled over columns → chained as the next level's memory.
- `train.py`: models are now an `nn.ModuleList` (one `PatchSetPFN` per hop, own
  `mask_patch_size` 4→2→1 and `stage1_proj`); per-hop weighted losses; `run_eval` loops a
  true-resolution Dice ladder (`dice_r16/r32/r64/r128/mean`, `dice/mean`=final native) +
  per-hop `refine/hop{L}/{delta_err,dice_delta,soft_dice_delta}`. Checkpoint on `dice/mean`.
  Asserts `resolutions[0] == stage-1 res`. Compile wraps each submodule.
- config: `sample.resolutions`, per-hop `n_total/n_fg_core/n_fg_core_ctx`, `train.loss_weights`.
- Training detached per-level (each hop refines the detached composite below it). Verified:
  unit tests green; full 4-level run trains (no NaN, ckpt saves); `resolutions=[16,32]`
  reduces exactly to the old single hop (`dice/mean==dice_r32/mean`).
- Fix: `train.checkpoint` warm-start used `removeprefix("_orig_mod.")`, but a compiled
  ModuleList saves keys as `"{L}._orig_mod...."` (prefix is mid-key, not leading) → it
  matched 0 tensors silently for default `compile=true` chain checkpoints. Switched to
  `replace("_orig_mod.","")` (verified 0/74 → 74/74) + a warning when 0 tensors load.

## 2026-06-16 — 2d/multilevel: dice/mean checkpoint selection + pfn_seg-matched coarse baseline

Stage-2 `train.py` now selects the best checkpoint on native `dice/mean` (the deployment
metric, pfn_seg-comparable) instead of `refine/uncertain/delta_err` (kept as diagnostic;
it's a local L1-reduction on boundary cells that can diverge from native Dice and ignores
certain-cell regression). `run_eval` returns `dice/mean` for selection.

Metric naming made resolution-honest (a `_r16`/`_r32` suffix now means the dice is
computed AT that resolution, verified: dice@r16 0.896 ≠ s1@128 0.782, dice@r32 0.093 ≠
s2@128 0.139):
  - `dice_r16/mean` = stage-1 pred @res-16 vs GT@16 (== pfn_seg low-res dice).
  - `dice_r32/mean` = refined map @res-32 vs GT@32 (new).
  - `dice/mean` = stage-2 refined 32→128 @128 (checkpoint/headline, pfn_seg-comparable).
  - `dice_s1/mean` = stage-1 16→128 @128 (pfn_seg headline baseline); `dice/margin_vs_s1`.
  - refine scopes are res-32 stage comparisons → suffixed by STAGE: `refine/{scope}/dice_s1`
    vs `dice_s2`, `soft_dice_s1/s2`, `refine/certain_err_s1/s2`. Also log per-sample
    improvements `refine/{scope}/dice_delta` and `soft_dice_delta` (s2-s1, >0=better,
    nanmean of paired diffs like delta_err).
Rationale: r16/r32 reserved strictly for true-resolution dices; where two maps are
compared at one shared resolution (native-128 pair, res-32 refine scopes) resolution
can't distinguish them, so they carry s1/s2.
Resolution numbers in the metric keys are derived, not hardcoded: R1 = stage-1 native
res (`round(stage1.N**0.5)`), R2 = `cfg.sample.grid_res`, H = `cfg.data.image_size`;
keys built as `f"dice_r{R1}/mean"` etc.
Fixed the coarse baseline: `dice_coarse/mean` was computing stage-1 via res-16→32→128
(double upsample); now uses the stage-1 res-16 map upsampled 16→128 DIRECTLY, exactly as
pfn_seg.py (same bilinear/align_corners/hard_dice), so it equals pfn_seg's stage-1 number.
Pipeline `coarse_predict` now also returns `coarse_lowres` (B,R1,R1) for this. Smoke-tested
1 epoch on busi: plumbing OK, coarse baseline 0.78.

## 2026-06-16 — 2d/multilevel: target n_fg_core sweep (weak lever)

prev_pred --stats, n_fg_core 64/128/192: fg captured 66.2%→69.1%→69.5% (saturates at
128), bnd→miss FLAT 31.5% throughout (fg_core takes budget from neighbors, never the
top-priority boundary core). Gain lands only on easy datasets with spare budget; hard
thin-structure datasets unchanged (tnbcnuclei fg→miss 51.9→53.4, monusac 52.8→54.5) —
already budget-saturated by their large boundary band, residual fg→miss is structural
(objects > 256 patches at res-32). Much weaker than the context lever (64→160 gave +9pts
on ds_gt) because prev_pred under-confidence already puts fg interior in the boundary
band (fg→core 60% at nfg=64), making the explicit fg quota partly redundant on target.
**Decision: keep context heavy (160), target light — leave target 64 or nudge to 96
(~+2pts free, keeps neighbor budget; >128 the neighbor tier vanishes for no gain).**

## 2026-06-16 — plot_sampling.py: --hist value-distribution diagnostic

Added `--hist` mode: per-cell value distribution (per-dataset + total) for GT and the
source map, with `%@0`/`%@1`/`%mid`, mean, and `|v-0.5|<tau` band fractions + a 10-bin
histogram. Run with `--source prev_pred` to get both GT and prediction in one pass.
**Findings (full val, res-32):** distribution is extremely bimodal. GT: 84.6%@0 /
7.8%mid / 7.6%@1 — the boundary band ≈ all fractional cells (`%mid`≈`|.5|<0.45`).
prev_pred: 75.6%@0 / 21.2%mid / 3.2%@1 — same mean (0.11) but `%@1` HALVED and `%mid`
2.7×: the stage-1 model is under-confident on fg (interiors predicted ~0.7-0.9), so
the 0.5-band is biased inward and mixes true edge with under-predicted interior →
mechanism behind the ~31% target boundary-miss floor. Fixed `tau` gives a ~70×-variable
boundary-core size across datasets (idrib 0.5% → tnbcnuclei 35% at |.5|<0.30), so the
boundary/fg/neighbor budget split is set by the data, not by config (tnbcnuclei's
boundary core alone > n_total).

**Quota-cap experiment (tested → REJECTED).** Added opt-in `--n_boundary_core` (caps the
tau band at the N cells closest to 0.5; default 0 = uncapped = unchanged). prev_pred
--stats, cap 0/96/64: capping WORSENS bnd→miss (31.5%→37.6%→40.8%) and barely moves
fg→miss (33.9%→33.2%) — net negative, even on the thin-structure datasets it targeted
(tnbcnuclei fg→miss 54→62, bnd→miss 62→64). Two reasons: (1) freed budget flows to
NEIGHBORS not fg_core (fixed 64), trading high-value boundary cells for low-value
neighbor fill; (2) on prev_pred the under-confident prediction puts fg interior (0.7-0.9)
INSIDE the tau band, so it covers boundary AND fg at once — capping loses both. The
variable tau allocation is a feature: high-boundary datasets need more boundary budget.
Kept the arg as a diagnostic only; real sampling.py unchanged.

## 2026-06-16 — 2d/multilevel: per-role fg quota (context n_fg_core)

Split `n_fg_core` by sampler role. Target and context both call `sample_patches`, but
with opposite goals: target ranks cells by the stage-1 `coarse_flat` (prev_pred,
uncertainty regime), context ranks by the TRUE GT mask fraction (`ctx_frac`). Context
wants a large share of the foreground sampled for class info, not boundary recall.
Added `sample.n_fg_core_ctx: 160` (`configs/experiment/2d/multilevel.yaml`);
`pipeline.py` context call now uses `s.get("n_fg_core_ctx", s.n_fg_core)` (target call
unchanged at 64). Backward-compatible — absent key falls back to `n_fg_core`. Motivated
by the ds_gt fg-coverage sweep: fg captured 72%→81% going 64→160, boundary miss flat
~28% (heavy fg costs the boundary nothing on the GT map). Diagnostic: `--source ds_gt`
in plot_sampling.py.

## 2026-06-16 — plot_sampling.py: fg-sourced neighbor diagnostic

`sample_patches` neighbor field now diffuses from `boundary_core ∪ fg_core` (was
boundary core only), and gained a `boundary_tier` arg + `--no_boundary` CLI flag to
disable the tau tier (tau→0). Added `[C] boundary coverage` block to `--stats`:
of true-boundary cells (`0<gt<1`), the % reaching core / neighbor / missed. Lets the
head-to-head run by invoking the script with different flags (no `--compare` mode):
`--n_fg_core 0` = boundary-only baseline, `--n_fg_core N` = union,
`--n_fg_core N --no_boundary` = pure fg-sourced neighbors. Spec:
`docs/superpowers/specs/2026-06-16-fg-sourced-neighbor-experiment-design.md`.
**Result (prev_pred, full val, pooled):** dropping the boundary tier (fg-sourced
neighbors only) raises bnd→miss 31.4%→47.8% — unanimous across all 35 datasets;
worst on compact/sharp objects (covid19radio 3.8→42.7, isic2016 5.7→40.1,
pandental 16.4→68.6). The `core_b ∪ fg_core` neighbor-union does NOT help the
boundary: union bnd→miss 31.6% ≈ baseline 31.4%, and union's bnd→neigh (4.7%) is
LOWER than baseline's (6.4%) — boundary is held up entirely by the boundary-core
tier. The fg_core quota's value is interior coverage (fg→miss 45.7%→30.6%), not
boundary. **Conclusion: boundary tier is load-bearing; fg-sourced neighbors are
not. The res-16 map (offset 0.5-band) is the ~31% boundary-miss floor, not the
sampler.**

## 2026-06-16 — 2d/multilevel: configurable mask-token form (arch.mask_prior)

Replaced `arch.coarse_prior` (bool) with `arch.mask_prior: false | scalar | patch` in
`PatchSetPFN` + pipeline. `false`/`scalar` keep the scalar mask-token (neutral support-mean
vs coarse-pred query prior). `patch` makes the mask-token a `p×p` mask tile
(`mask_embed=Linear(p²,e)`, `p=image_size//grid_res` auto): support tiles = native GT under
each cell (exact boundary geometry, richer than the avg-pooled fraction), query tiles =
upsampled coarse prior. New `pipeline._mask_tiles` helper; `qry_coarse` scalar kept as the
metrics baseline. Caveat: patch mode ties `mask_embed` to grid_res (not cross-resolution
generalizable); query tile carries no detail below the res-16 stage-1. Tests
(`test_patchset.py`, `test_pipeline.py`) updated + patch-mode cases added; 1-epoch smoke run
in patch mode verified (params +3840 = (p²-1)·e).

## 2026-06-16 — 2d/multilevel: sampling-procedure redesign + diagnostic

Designed a new stage-2 patch sampler (spec:
`docs/superpowers/specs/2026-06-16-multilevel-patch-sampling-design.md`): threshold
boundary core + fixed random-fg-core quota + blurred-field Gumbel-top-k neighbor fill,
replacing the fixed n_uncertain/n_certain split. Built `experiments/2d/multilevel/plot_sampling.py`
to visualize and measure it on MedSegBench val (`--stats`, `--sweep`), with a `--source
ds_gt|prev_pred` toggle that samples from either res-32 GT or the real frozen stage-1
prediction (res-16→32) at the correct resolutions.

Sweep findings (full val, 13,237 imgs, fg from true GT): tuned defaults `tau=0.30,
sigma=1.0, floor=0.005, n_fg_core=64` (M=256, grid=32) → fg→miss ~36%, matching the GT
oracle. `tau` cannot be read off the oracle (0.45 best for ds_gt but regresses under
prev_pred as the neighbor fill collapses on the misplaced predicted boundary).

Implemented: `sampling.py` now exposes `sample_patches` + `gaussian_blur` (old
`sample_patch_indices` retired); `pipeline.build_patch_batch` uses it for both query and
support paths with `qry_is_uncertain` = boundary core (excludes fg-core), plus a
`stochastic` flag (train True, eval `not eval_deterministic`); config `sample.*` replaced
`n_uncertain/n_certain` with `n_total/tau/n_fg_core/blur_sigma/floor/temperature/
eval_deterministic`; `train.py` threads `stochastic`. Run name now comes from wandb
(auto-generated); checkpoints save under `{date}_{run_name}` (e.g. `2026-05-22_deft-field-72`).
Tests (`test_sampling.py`, `test_pipeline.py`) updated and passing; 1-epoch smoke run of
`train.py` train+eval+checkpoint verified.

## 2026-06-16 — 2d/multilevel: configurable query sampling map (prev_pred | ds_gt)

Isolate the stage-2 patch-selection signal from the model. Added `sample.train` and
`sample.eval` (`prev_pred` | `ds_gt`) to `configs/experiment/2d/multilevel.yaml`.
`build_patch_batch` now takes `sampling_source`: `prev_pred` ranks query cells by the
stage-1 coarse prediction (current/deployable behaviour, default), `ds_gt` ranks by the
downsampled target GT (oracle upper-bound — leaks labels, eval numbers optimistic).
Only query selection changes; `qry_prior` and the eval fusion base stay on the coarse
map, so the model input is held fixed across the two modes. Run name gains a
`_smp-{train}-{eval}` tag. Files: `pipeline.py`, `train.py`, `multilevel.yaml`.

## 2026-06-15 — pfn_seg: fix CUDA grid-limit crash at resolution≥32

`src/models/pfn_seg_2d.py`. Running with `arch.resolution=32` crashed in the
sample-axis attention with `CUDA error: invalid configuration argument`.

- **Root cause**: the sample-axis (cross-image) attention flattens to a batch of
  `B·2·resolution²` independent tiny (seq=12) attention problems. Flash /
  mem-efficient SDPA launch one CUDA grid-Y block per batch element, and
  `gridDim.y` is hardware-capped at 65535. At R32 with `batch_size=32` the batch
  is `32·2·1024 = 65536`, exactly one over the cap (R16 was 16384, fine).
  Confirmed by minimal repro: flash SDPA OK at batch 65535, crashes at 65536.
- **Fix**: `batched_sdpa()` helper splits the batch into equal chunks under the
  cap when needed. Keeps the fused kernel (the math backend works but materializes
  the full score tensor at ~2× memory). `int()` pins the symbolic batch to a
  concrete value so the chunk count is a Python int that `torch.compile`
  (`dynamic=True`) unrolls statically — a plain `range(0, B, step)` over a
  symbolic `B` fails to compile.
- **Note (separate constraint, not a bug)**: R32 + UniverSeg encoder + 6 layers
  at `batch_size=32` OOMs on a 24 GB card (~23 GB needed); R32 has 4× the tokens
  of R16. Use a smaller `batch_size` (verified training proceeds end-to-end at
  `batch_size=8`) or gradient accumulation for the full run.

## 2026-06-14 — multilevel: log native `dice/mean` for direct comparison to pfn_seg

`experiments/2d/multilevel/train.py`. `run_eval` now also logs `dice/mean` — native-resolution
hard Dice over the whole val set, aggregated identically to `pfn_seg.py` (mean over all samples,
plus `dice/dataset/<name>`). The stage-2 "final" map = coarse res-32 map with sampled cells
overwritten by stage-2 preds, upsampled to native (128). `dice_coarse/mean` logs the coarse-only
baseline (≈ the stage-1 model's own dice/mean) so the lift is visible in one place. This makes the
2-stage pipeline directly comparable to the single-stage ImagePFN number in `results/report.md`.

## 2026-06-14 — multilevel: within-image spatial reasoning (query↔query attention)

`src/models/pfn_seg_2d.py`, `src/models/patchset_pfn.py`, `experiments/2d/multilevel/train.py`,
`configs/experiment/2d/multilevel.yaml`.

`arch.query_self_attn` (default true): lets stage-2 query patches attend to **each other**
(keyed by Fourier PE), restoring within-image spatial coherence that the patch-set form
otherwise lacks (each query was classified independently given the support). No label leak —
query mask-tokens carry the coarse prior, not GT. Implemented as an asymmetric sample-axis
mask: support rows attend only to the train set `[:sep_t]`; query rows attend to train set +
all queries. `TransformerEncoderLayer`/`Stack` gained an optional `attn_mask` (default None =
prior behavior, so `ImagePFN` stage-1 is untouched). Memory cost ~0 (+0.01 GB) since the
sample axis is batched over only 2 columns. Verified: a coupling test (perturbing one query's
feature moves another query's output only when enabled), ImagePFN regression, and compiled
(`torch.compile(dynamic=True)`) forward+backward — the dynamic (r×r) bool mask lowers cleanly.

## 2026-06-14 — multilevel: critical target-leak fix + soft-dice / full-image metrics

`experiments/2d/multilevel/pipeline.py`, `experiments/2d/multilevel/train.py`.

**Bug fix (critical):** `build_patch_batch` derived the target fraction `gt32` from
`all_masks[:, -1]`, but the query mask is zeroed there (so stage-1 doesn't see the answer)
— so the supervision target was **all zeros**. Symptoms: train loss collapsed to 0.0000,
a spurious large positive `Δerr`, and NaN dice. Now `gt32` is pooled from the real `label`.

**Metrics:** `run_eval` now reports, per scope `{uncertain (192), sampled (256), full (1024
via compositing stage-2 preds into the coarse map at sampled cells)}`: `delta_err`,
hard `dice_stage2/coarse`, and **soft** `soft_dice_stage2/coarse` (vs the coarse baseline).
Pipeline returns `qry_idx`, `coarse_full`, `gt_full` to support this. Robust `nanmean`
removes the `Mean of empty slice` warning. Headline for checkpointing stays
`refine/uncertain/delta_err`.

## 2026-06-14 — multilevel patch refinement (stage-2 PatchSetPFN)

New experiment `experiments/2d/multilevel/`. A frozen res-16 ImagePFN (stage 1) +
frozen UniverSeg encoder produce a coarse target prediction and res-32 features; we
sample 256 patches/image (192 closest-to-0.5 + 64 most-certain) — target→query,
context→support — and train `src/models/patchset_pfn.py:PatchSetPFN` (nanoTabPFN-shaped:
rows=patches, cols=[img|mask], 2-D Fourier PE, query-attends-to-support) to refine the
query patches. Metric: `refine/delta_err_uncertain` = |error| reduction on the uncertain
target region vs the stage-1 coarse value (64 certain patches kept as a regression check).
`arch.coarse_prior` toggles using the coarse pred as the query mask prior (else neutral 0).

**Stage-1 thinking memory** (`arch.use_stage1_thinking`, default true): `ImagePFN.forward`
gains `return_thinking` → its post-transformer thinking rows mean-pooled over columns
`(B, n_think, e1)`. `PatchSetPFN` projects these `e1→e` (+ a learned type token) and prepends
them as extra **support** rows (inside `sep`), so query patches attend to stage-1's latent
task summary via the existing sample-axis attention. `stage1_dim`=stage-1's `e`, read from
`stage1.thinking.tokens` at construction; new proj/type params are fresh (warm-start-safe).

Files: `src/models/patchset_pfn.py` (FourierPositionalEncoding + PatchSetPFN),
`experiments/2d/multilevel/{sampling,pipeline,train}.py` (+ `test_*.py`),
`configs/experiment/2d/multilevel.yaml`. Reuses the shared utils in `pfn_train.py`.
Run: `python experiments/2d/multilevel/train.py [arch.coarse_prior=false]`.
Spec/plan: `docs/superpowers/{specs,plans}/2026-06-14-multilevel-patch-refinement*.md`.

## 2026-06-14 — Factor shared training utils into pfn_train.py

`experiments/2d/pfn_train.py` (new), `experiments/2d/pfn_seg.py`.

Extracted `_newtonschulz5_batched`, `Muon`, `augment`, `lawa_average`, `soft_dice_loss` from
`pfn_seg.py` into a new shared module `pfn_train.py`. `pfn_seg.py` now imports them from there.
No behaviour change; prepares for reuse in `experiments/2d/multilevel/train.py`.

## 2026-06-14 — pfn_seg_2d: optional frozen pretrained image encoder (UniverSeg features)

`src/models/pretrained_encoders.py` (new), `src/models/pfn_seg_2d.py`,
`experiments/2d/pfn_seg.py`, `experiments/2d/eval.py`, `configs/experiment/2d/pfn_seg.yaml`.

New `arch.image_encoder: patch | universeg`. With `universeg`, the image path becomes a
frozen UniverSeg encoder → `resolution×resolution` feature grid → `Linear(feature_dim, e)`,
replacing the raw-pixel patchify+embed. Mirrors the `feature_sim` eval backend
(`encode_images` + `extract_features_batch`); `arch.feature_level` selects the level
(`all` → 256ch). The mask path is unchanged (raw P×P patches → Q×Q → `Linear(Q², e)`).

Design: the encoder is **injected** into `ImagePFN` (new `image_encoder`/`feature_dim` args),
not imported by it, so `pfn_seg_2d.py` stays torch-only and free of the `src`-package
shadowing (UniverSeg loads from its own `/home/dpxuser/repos/UniverSeg` checkout). It runs
under `no_grad` and is frozen — gradients reach `image_embed` but not the encoder; its params
are filtered out of the Muon/Adam groups via `requires_grad`. Image features are normalized
**per channel** with context-image stats (vs. the single-scalar norm on raw pixels). `eval.py`
rebuilds + injects the encoder when the checkpoint's arch says `universeg`.
`encoder_resize_to_input` (default false) gates resizing inputs to 128² before the fully-conv encoder.
The encoder's `forward` is wrapped in `torch.compiler.disable` (frozen/no_grad — nothing to compile):
under `torch.compile(dynamic=True)` its `adaptive_avg_pool2d` otherwise gets symbolic window sizes
inductor can't lower (`cannot determine truth value of Relational`). Dynamo graph-breaks there; the
transformer still compiles.

`pfn_seg.py` warm-start (`train.checkpoint`) is now **tolerant**: it loads only tensors whose
name+shape match the current model (`strict=False`), so a checkpoint from before these changes
still warm-starts. Switching to a pretrained encoder changes `image_embed` (Q²→feature_dim) and
adds frozen `image_encoder.*` weights — those keep their fresh/pretrained values; shared weights
(mask_embed, pos_embed, thinking, transformer, decoder) transfer. Logs loaded/skipped/fresh counts.

## 2026-06-14 — pfn_seg_2d: `resolution` param decouples output grid from encoder input dim

`src/models/pfn_seg_2d.py`, `experiments/2d/pfn_seg.py`, `experiments/2d/eval.py`,
`configs/experiment/2d/pfn_seg.yaml`.

Replaced `arch.patch_size` with `arch.resolution` (+ `arch.input_patch_size`, default 8).
`resolution` = patches per side = output grid `Hp`; effective patch size `P = image_size //
resolution` (128//16 = 8). Each native `P×P` patch is now resized to `Q×Q` (`input_patch_size`)
via `F.interpolate` in `patchify` before embedding, so `image_embed`/`mask_embed` always take a
fixed `Q²` input regardless of `P`. This lets the output resolution change (e.g. res 8→32, P
16→4) while the patch encoder stays fixed at 8×8.

`eval.py` derives `resolution`/`input_patch_size` from old `patch_size` checkpoints for
back-compat. Run name now `pfn_seg_R{res}q{Q}_...`.

## 2026-06-12 — pfn_seg_2d eval backend: fixes, config consolidation, warm-start, comparison

`experiments/2d/eval.py`, `experiments/2d/pfn_seg.py`, `configs/experiment/2d/base.yaml`,
`configs/experiment/2d/pfn_seg.yaml`. Deleted `configs/experiment/2d/pfn_seg_eval.yaml`.

Finished and validated the `pfn_seg_2d` eval backend added on 2026-06-11.

**`src` package-shadowing fixes** — `common.py` puts `/home/dpxuser/ic_segmentation` on
`sys.path`, whose `src` package shadows patch_icl's. Both are regular packages, so only
one wins per process. Two failures resolved (pfn_seg runs only):
- `ModuleNotFoundError: src.datasets` — `common`'s `from src.datasets.medsegbench` import
  resolved to ic_segmentation (no `datasets`). Fix: when `"pfn_seg"` is in `sys.argv`,
  pre-import patch_icl's `src.datasets.medsegbench` before `common`, caching the right `src`.
- `ModuleNotFoundError: src.models.pfn_seg_2d` — ic_segmentation's `src.models` lacks it.
  Fix: load `pfn_seg_2d.py` directly by file path via `importlib.util` (torch-only deps).
- Both guards key off `"pfn_seg"` in argv, so universeg / feature_sim runs are untouched.

**Config consolidation** — deleted the 3-line `pfn_seg_eval.yaml`; eval now reads
`base.yaml` with overrides. Added `eval.checkpoint: null` to `base.yaml` (Hydra rejects
undeclared CLI keys). New invocation:
```bash
python experiments/2d/eval.py model=pfn_seg_2d eval.checkpoint=results/2d/<run>.pt
```

**Warm-start / resume** — `pfn_seg.py` gains `train.checkpoint`: loads weights (bare
state_dict or `{"model": ...}`, strips `_orig_mod.`) into a fresh model before training.
Used to resume the pre-format-change `results/2d/best.pt` for one epoch (54 741 samples,
val Dice 0.4348), resaved in the new format as `results/2d/pfn_seg_resumed.pt`.

**Validation** — full val eval of `pfn_seg_resumed.pt`: mean Dice 0.4393 over 1141 samples
(matches trainer within sampling noise), 68.20 GFLOPs, ~3.6 ms/item. See `results/report.md`
for the pfn_seg_2d vs UniverSeg comparison (mean Dice 0.524 vs 0.334 over 35 datasets).

## 2026-06-11 — pfn_seg: augmentations, compile flag, progress bars

`experiments/2d/pfn_seg.py`, `src/models/pfn_seg_2d.py`, `configs/experiment/2d/pfn_seg.yaml`, `configs/augmentations/medsegbench.yaml` (new), `experiments/2d/plot_aug.py` (new).

**`torch.compile` made optional (`arch.compile: false`)**:
- `TransformerEncoderLayer.forward` had `@torch.compile(dynamic=True)` and `_newtonschulz5_batched` had `@torch.compile` baked in.  On first run, 6 layers × 2 (fwd+bwd) Triton compilations caused a silent multi-minute hang that looked like a freeze.
- Both decorators removed.  Compile is now opt-in: `arch.compile=true` wraps the full model and `_newtonschulz5_batched` via `torch.compile(dynamic=True)` after construction.

**GPU augmentations (`augment()` in `pfn_seg.py`)**:
- New `configs/augmentations/medsegbench.yaml` — canonical light-aug config: geometric (hflip p=0.5, vflip p=0.5, rotate ±20° p=0.5) and intensity (brightness Δ0.15, contrast ×[0.8–1.2], gamma [0.75–1.33], noise σ=0.04).
- Geometric ops applied **to context pairs only** (joint image+mask via `F.affine_grid`/`F.grid_sample`) so the query GT is never misaligned; intensity ops applied independently to all images.
- All ops are batched GPU tensor operations (`torch.where`, `flip`, `affine_grid`/`grid_sample`); no per-sample Python loops.
- `pfn_seg.yaml` gains a `defaults` list loading `augmentations/medsegbench@aug`; `hydra.main` config_path widened to `../../configs` so Hydra resolves the group.  Disable at runtime with `aug.enabled=false`.

**Progress bars**:
- `train_epoch`: `tqdm` per-batch with running `loss` postfix.
- `run_eval`: existing `tqdm` extended with running `dice` postfix.
- `main` epoch loop: outer `tqdm` with `loss / lr / best` postfix updated after each epoch and eval.
- All `print()` calls inside the loop replaced with `tqdm.write()`.

**Visualisation**: `experiments/2d/plot_aug.py` — loads a small multi-dataset batch and saves a 6-column grid (orig ctx, orig mask, aug ctx, aug mask, orig query, aug query) to `results/datasets/medsegbench_aug.png`.

## 2026-06-11 — ImagePFN: nanoTabPFN-style in-context segmentation model

New files:
- `src/models/pfn_seg_2d.py` — `ImagePFN` model
- `experiments/2d/pfn_seg.py` — training + eval script
- `configs/experiment/2d/pfn_seg.yaml` — config
- `docs/tabpfn/nanotabpfn.md` — paper + repo summary

`ImagePFN` adapts all modded-nanoTabPFN techniques to 2D image segmentation:
- **Dual-axis transformer**: feature-axis attention (spatial, within each image) + sample-axis
  attention (cross-image, per patch position). Tensor layout `(B, rows, cols, e)` where
  rows = images and cols = patches.
- **Asymmetric masking**: query image attends only to thinking + context rows; mirrors TabPFN's
  train/test split in sample-axis attention.
- **Thinking rows**: `n` learnable row tokens prepended to the sequence, broadcast across all
  patch positions; treated as train rows in sample-axis attention.
- **Residual decay**: input to block i scaled by `residual_decay^i` (default 0.95).
- **LowerPrecisionRMSNorm**: pre-norm RMSNorm that upcasts to fp32 for bf16/fp16 inputs.
- **LAWA**: rolling buffer of last `K=10` checkpoints, averaged at eval time only; training
  weights unchanged.
- **Muon optimizer**: applies to all 2D weight matrices inside `transformer`; AdamW + cosine
  LR schedule for everything else (embeddings, norms, decoder).
- Images are patchified (P=8, N=256 patches per image) and normalized using context-image
  statistics per batch (mirrors TabPFN per-column normalization).
- Decoder outputs per-patch logits for query row; upsampled to native resolution for BCE loss.

## 2026-06-11 — feature_sim.py fixes (vectorisation, ensemble averaging, timing)

`experiments/2d/feature_sim.py`

- **Vectorised feature extraction**: replaced `extract_features` + `_pool2d` (which assumed B=1
  via `.squeeze(0)` and was called in an O(B×K) Python loop) with `extract_features_batch`,
  which operates on full `(N, C, H, W)` tensors and concatenates on `dim=1`. The main eval
  loop now calls it once for targets `(B, C', os, os)` and once for contexts `(B*K, C', os, os)`,
  then reshapes; context mask downsampling likewise vectorised via a single
  `F.adaptive_{avg,max}_pool2d` on `(B*K, 1, H, W)`.

- **Probability averaging in `batch_tabpfn`**: was averaging logits before softmax
  (`softmax(sum_logits / n_estimators)`), which differs from the standard ensemble approach.
  Now applies softmax per estimator and accumulates probabilities, dividing at the end.

- **Split timing**: `inference_times` (a single conflated number) replaced by separate
  `encode_times` and `tabpfn_times` lists. Summary now logs `time/encode_ms`,
  `time/tabpfn_ms`, and `time/total_ms` to wandb and prints each separately.

- **balance_ratio warning**: prints a warning at run start when `balance_ratio` is set,
  because it disables batched TabPFN and falls back to serial per-sample inference.

- **FLOPs message**: clarified to say "per sample" (the measured value is for 1+K images,
  not for the full eval batch).

## 2026-05-29 — VoComniNNUNetEncoder (PlainConvUNet/VoComni)

`src/models/encoders/vocomni_nnunet.py` — new encoder wrapping PlainConvUNet
(nnUNet CNN backbone) from the VoComni_nnunet.pt checkpoint (supervised on 20K
CT volumes, 21 classes). 6 stages, channels 32→64→128→256→320→320, total_stride=32.
Loads only the `encoder.*` prefix from the full model checkpoint.
torch.compile default True (~2× speedup vs eager).

`run.py` — added `--encoder vocomni_nnunet`, `--vocomni_nnunet_ckpt`,
`--vocomni_nnunet_compile` args.

## 2026-05-29 — VoComniEncoder (SwinUNETR/VoCo)

`src/models/encoders/vocomni.py` — new encoder wrapping MONAI SwinUNETR with
VoCo/VoComni pretrained weights.  Returns 5 feature levels at strides 2–32.
`skip_channels=[fs, 2fs, 4fs, 8fs]`, `bot_features=16fs`, `total_stride=32`.
Checkpoint loading handles `state_dict`/`student`/`module.`/`backbone.` prefixes.
Supports feature_size 48 (Base, 72M), 96 (Large, 290M), 192 (Huge, 1.2B).

`experiments/feature_similarity/run.py` — added `--encoder vocomni`,
`--vocomni_ckpt`, `--vocomni_feature_size` CLI args; vocomni uses the same
image-only combined-batch path as threedino.

## 2026-05-27 — val_loader pre-filtered to val_items_per_class

`experiments/nninteractive/train.py`.

`val_loader` now wraps a `torch.utils.data.Subset` that pre-selects at most
`val_items_per_class` indices per class from `ds_val.samples` (deterministic,
class-ordered). Previously the full val dataset was loaded (1686 samples / 211
batches) and a batch-level guard inside `validate()` skipped batches only when
**all** items were saturated — so the encoder + attention still ran for items
that were already counted. With the Subset, the loader contains exactly the items
that will be scored (~1410 samples / ~177 batches for 47 classes × 30 items).

Also fixed: `torch.compile(mode="reduce-overhead")` replaced with
`mode="default"` to avoid CUDA-graph segfaults caused by `cascade_registers`
changing from `None` (level 0) to a tensor (levels 1+) across calls to the
shared attention module.

## 2026-05-27 — feature_level list support in extract_features

`experiments/multilevel/train.py`, `configs/experiment/nninteractive.yaml`.

`extract_features` now accepts a list of level indices in addition to `'all'` and a single
integer. This lets a run select any subset of encoder levels without computing or storing
the unused ones.

```
feature_level: [2, 3, 4]   # skip large early stages; concat levels 2+3+4 only
feature_level: 4            # bottleneck only
feature_level: all          # concat all num_stages outputs (original behaviour)
```

Hydra deserialises YAML lists as `omegaconf.ListConfig`, so the isinstance check uses
duck-typing (`not isinstance(level, str) and hasattr(level, "__iter__")`) rather than
`isinstance(level, (list, tuple))`.

`configs/experiment/nninteractive.yaml` updated to `feature_level: [2, 3, 4]` (drops
skip0=32ch@128³ and skip1=64ch@64³, which account for 94% of skip-feature VRAM at B=8 fp16).
`embed_dim` drops from 800 → 704 (128+256+320); auto-detected via dummy forward in `main()`.

---

## 2026-05-27 — torch.compile integration + CUDA graph + RoPE fixes

`experiments/nninteractive/train.py`, `experiments/multilevel/train.py`, `src/rope3d.py`.

### torch.compile integration

Added `compile_model` flag to both training scripts (config key `train.compile_model`).
When enabled on a `shared_weights=True` run, compiles `model.shared_level` (the single
`PatchICLAttention` instance) with `mode="reduce-overhead"` (CUDA graph capture).
The compiled module is kept in a local `compiled_attn` variable — `model.state_dict()`,
optimizer parameters, and checkpoint saving are unaffected.

Default: `compile_model: true` in `nninteractive.yaml`; `false` in `multilevel.yaml` (opt-in).

### CUDA graph cascade_regs fix

`cascade_regs = regs.detach()` shares storage with the CUDA graph's output pool.
Passing that tensor as input to the *next* level's CUDA graph (a sibling graph) triggers:

    RuntimeError: accessing tensor output of CUDAGraphs that has been overwritten by a subsequent run

Fix: `.clone()` after every cascade_regs assignment so the tensor lives in normal GPU
memory and is no longer a view of the CUDA graph output pool:

```python
cascade_regs = regs.detach() if cfg.model.detach_cascade_registers else regs
if compiled_attn is not None:
    cascade_regs = cascade_regs.clone()
```

Applied at all 4 call sites in each train.py (L0 + L1, training + validation).
`torch.compiler.cudagraph_mark_step_begin()` is also called before each compiled forward
as a step-boundary hint for the CUDA graph tree.

### RoPE real-space rotation fix

`src/rope3d.py` `_rotate()` previously used `torch.view_as_complex` / `view_as_real`,
which Torchinductor does not support:

    UserWarning: Torchinductor does not support code generation for complex operators.
    Performance may be worse than eager.

Replaced with equivalent real-space arithmetic:

```python
x0' = x0·cos − x1·sin
x1' = x0·sin + x1·cos
```

No correctness change (fp32 max diff 4.77e-7; fp16 9.77e-4 — normal fp16 rounding).
The new path is fully fuseable by Triton.

---

## 2026-05-27 — Attention compile benchmark (torch.compile on PatchICLAttention)

`/tmp/compile_bench.py` (inline benchmark, not committed).

Benchmarked `torch.compile` modes on `PatchICLAttention` fwd+bwd using `.venv311`
(Python 3.11 + triton 3.2.0 — required for max-autotune; Python 3.12 breaks triton).

Config: B=4, K=1, N=M=512 (dense 8³), embed_dim=800, dim=256, L=8.

| Method | Latency | Speedup | ΔVRAM |
|--------|--------:|--------:|------:|
| baseline eager | 87.8 ms | 1.00× | — |
| **compile reduce-overhead** | **13.8 ms** | **6.34×** | **−499 MB** |
| compile max-autotune | 17.2 ms | 5.11× | −499 MB |

`reduce-overhead` wins (CUDA graph capture); `max-autotune` spends 5 min tuning Triton
kernels for a slightly worse result. Recommendation: add
`torch.compile(model.shared_level, mode="reduce-overhead")` to `train.py`.

## 2026-05-27 — NNInteractive experiment

`experiments/nninteractive/train.py` (new), `configs/experiment/nninteractive.yaml` (new).

New training experiment: same multilevel PatchICLAttention stack as `experiments/multilevel/`
but driven by the frozen `NNInteractiveEncoder` (90 M params) instead of STUNetEncoder.

Key architectural difference: context images are encoded **with their ground-truth masks**
(`encode_context(encoder, ctx_imgs, ctx_masks)`) so the encoder features are already
mask-conditioned at the backbone level. Target images are encoded with a zero mask.
The downstream label injection in PatchICLAttention is kept unchanged.

`embed_dim=800` (feature_level=all, 5 stages) vs 992 for STUNet-base.
MultilevelICL 2.48 M trainable params (encoder frozen).

Usage:
    python experiments/nninteractive/train.py
    python experiments/nninteractive/train.py model.nnint_mask_injection=separate
    python experiments/nninteractive/train.py model.nnint_ckpt=/local/path/...

## 2026-05-27 — NNInteractive pretrained encoder wrapper

`src/models/encoders/nninteractive.py` (new).

`NNInteractiveEncoder` wraps the pretrained ResidualEncoder from the nnInteractive v1.0
checkpoint (383 M params, 8-channel input: 1 image + 7 interaction slots).
Implements the same `(skip_channels, bot_features, total_stride, forward(imgs, masks))`
interface as `STUNetEncoder` so it can drop in as an encoder swap.

Two mask injection modes:
- **ch1** — pack `[image, mask, 0, 0, 0, 0, 0, 0]` as the 8-channel input, using the
  model's native "current segmentation" channel.
- **separate** — image in ch0 only; mask encoded by a SAM-style 3-D CNN
  (`_Mask3DEncoder` from stunet.py) and fused additively at the bottleneck.

`num_stages` controls depth (default 6 → 32× stride; 5 → 16× recommended for 64³/128³).
Encoder weights are frozen by default. nnInteractive installed as editable package into `.venv`.

## 2026-05-27 — Fix seeds3d segfault (float32 → uint8)

`scripts/synth_labels/generate.py`.

`seeds3d` was crashing with a segfault (core dump) on every call, causing forked worker processes to die silently and the pool to hang indefinitely waiting for results. Root cause: `python_3d_seeds` (a pybind11 wrapper around OpenCV's SEEDS) requires **uint8** input normalised to `[0, 255]`; passing `float32` causes an out-of-bounds memory access in the C++ layer. Fix: normalise the volume with `((vol - mn) / (mx - mn + 1e-8) * 255).astype(np.uint8)` before calling `sv.iterate()`.

## 2026-05-26 — MRI normalisation fix + bbox cache parallelisation

`scripts/convert_to_npy.py`, `src/totalseg_dataloader_incontext.py`.

**MRI normalisation (`convert_to_npy.py` `_normalise_mri`)**:
- Lower clip bound changed from hard `0` to the 0.5th percentile of non-zero foreground voxels (matches the 2D zopt dataloader in `ic_segmentation`). Upper bound remains p99.5. Z-score step retained.
- Motivation: `0` is an arbitrary floor that can include background signal and biases the clip range for certain MRI protocols. Using p0.5 of foreground is protocol-agnostic.
- **Pre-resized MRI `.npy` files must be regenerated**: `python scripts/convert_to_npy.py --data .../totalsegmri --modality mri --size 128 128 128 --overwrite`

**Bbox cache parallelisation (`totalseg_dataloader_incontext.py`)**:
- Added module-level `_bbox_for_subject(root, subj)` (picklable worker) — same pattern as `_adj_for_subject`.
- `_load_or_build_bbox_cache` now uses `ProcessPoolExecutor(max_workers=16)` instead of a sequential loop. Critical for TotalSegMRI where native label volumes can exceed 1000 × 1280 × 1900 voxels.
- Existing `.bbox_cache_*.pkl` files are reused as-is; only fresh builds benefit.

**MRI crop size fix (`totalseg_dataloader_incontext.py` `_load_crop` / `_load_crop_multi`)**:
- CT dataset: perfectly isotropic at 1.5 mm → `T × sp_min = 192 mm` always (consistent).
- TotalSegMRI: spacing 0.17–28 mm, up to 120× anisotropy → old formula gave physical crops as small as **21 mm** (zoomed in).
- Fix: `phys_ref = max(T * sp_min, T * 1.5)` — clamps the physical crop to at least 192 mm regardless of how fine the in-plane voxels are. High-res subjects (0.2 mm in-plane) now crop ~960 voxels in-plane and downsample to 128, giving the same physical context as CT.

## 2026-05-26 — `not_benchmark` class splits in resolve_classes

`data/totalseg_classes.py`

- Added `"not_benchmark"` → CT classes in `ALL_CLASSES[:117]` that are **not** in `BENCHMARK_CLASSES` (complement train set for CT).
- Added `"benchmark_mri"` → `MRI_BENCHMARK_CLASSES`.
- Added `"not_benchmark_mri"` → MRI classes in `MRI_ALL_CLASSES` that are **not** in `MRI_BENCHMARK_CLASSES` (complement train set for MRI).
- Use in config: `data.train_classes=not_benchmark data.val_classes=benchmark` to train on held-out classes and evaluate on benchmark.

## 2026-05-25 — Augmentation benchmark and synth_equiv preset

`configs/augmentations/synth_equiv.yaml`, `experiments/multilevel/benchmark_aug.py`.

- **`synth_equiv.yaml`**: new aug preset matching synth-aug strength applied to real labels (task.affine ±25°, scale [0.90–1.50], translate ±0.20, elastic p=0.8; intensity BC+noise+blur matching synth ops but no gamma/sim-low-res). Use with `train.aug_preset=synth_equiv data.p_synth=0.0` to isolate augmentation strength from data-type effect.
- **`experiments/multilevel/benchmark_aug.py`**: comprehensive runtime benchmark comparing custom PyTorch, MONAI, and batchgenerators augmentation pipelines. Sweeps presets × K values, reports per-transform breakdown, worker throughput estimates. Key findings at 128³ (20 reps, n_workers=20):

  | Preset | ms/vol (K=1) | ms/vol (K=3) | batch(K=1, 8 items) |
  |--------|-------------|-------------|---------------------|
  | nnunet (no elastic) | 68 ms | 32 ms | 26 ms |
  | multiverseg | 152 ms | 112 ms | 65 ms |
  | synth_equiv | 216 ms | 233 ms | 102 ms |
  | synth pipeline (p_synth=1) | 419 ms | 465 ms | 143 ms |

  - **Custom PyTorch vs libraries**: affine ≈ MONAI (1.03–1.28× at 128³, batching advantage lost to grid_sample cost), elastic custom is 2–3.5× faster than MONAI, batchgenerators `augment_spatial` is 17–20× slower.
  - **Bottleneck**: affine (~128–140 ms/vol) + elastic (~150 ms/vol) dominate at 128³. nnunet avoids elastic (p=0) and is 3–4× faster.
  - **Synth pipeline**: independent per-copy aug is the training bottleneck at 128³; with 20 workers K=3 gives only 45 sps → 180ms/batch, likely bottlenecking GPU pipeline.
  - **Recommendation**: keep custom PyTorch (already optimal); consider `aug_preset=nnunet` for faster iterations; use `synth_equiv` + `p_synth=0.0` to test augmentation-strength hypothesis.
  - Output saved to `results/aug_benchmark/aug_benchmark_*.json`.

## 2026-05-25 — TotalSegMRI support (convert, synth labels, eval)

`data/totalseg_classes.py`, `data/benchmark_classes.py`, `scripts/convert_to_npy.py`, `scripts/synth_labels/generate.py`, `scripts/eval.py`, `configs/config.yaml`, `configs/cluster/nfs.yaml`, `configs/cluster/meta.yaml`.

- **4 MRI-only classes** appended to `ALL_CLASSES` (indices 118–121): `lung_left`, `lung_right`, `intervertebral_discs`, `vertebrae`. Zero-indexed at the tail so all existing CT label indices are unchanged.
- **`convert_to_npy.py`**: added `--modality {ct,mri}` flag. MRI normalisation clips to `[0, p99.5(foreground)]` then z-scores with foreground mean/std (per-volume, no global constants). Output is still named `ct.npy` so the dataloader needs no changes. Fixed `get_zooms()` tuple vs ndarray bug by using `[float(x) for x in ...]`.
- **`synth_labels/generate.py`**: added `--modality {ct,mri}` flag, routing to `paths.totalsegmri` and loading `mri.nii.gz` with per-volume normalisation as the NIfTI fallback.
- **`eval.py`**: added `--dataset {totalseg,totalsegmri}` flag. `resolve_totalseg_root()` now uses an exact-key regex (`(?<!\w)totalseg:` vs `totalsegmri:`) to prevent prefix collision. Defaults to `MRI_BENCHMARK_CLASSES` when `--dataset totalsegmri`.
- **`MRI_BENCHMARK_CLASSES`**: curated 18-class MRI benchmark covering organs, whole-lung, spine (individual + merged), vasculature, and bones.
- Config: `paths.totalsegmri` added to `config.yaml`, `nfs.yaml`, and `meta.yaml`; `data.dataset: totalseg` default added.

## 2026-05-25 — Shared weights, physical scale injection, and role embeddings for MultilevelICL

`experiments/multilevel/model.py`, `experiments/multilevel/train.py`, `experiments/feature_attention/model.py`, `configs/experiment/multilevel.yaml`, `src/totalseg_dataloader_incontext.py`, `scripts/convert_to_npy.py`.

**Shared weights (`model.shared_weights`)**:
- `MultilevelICL` gains `shared_weights: bool` — when true, a single `PatchICLAttention` instance is used for all spatial levels instead of one per level. Works only with `pos_encoding="rope3d"` (learned PE has grid-size-dependent embedding tables; an assert guards this).
- Checkpoint key remapping on load handles both transition directions: `levels.0.*` → `shared_level.*` (per-level → shared, uses L0 weights as init) and `shared_level.*` → `levels.{i}.*` (shared → per-level, broadcasts to all slots). Happens before the existing shape-filter step so no other changes are needed.

**Physical scale injection (`model.use_scale_embed`)**:
- `ContinuousScaleEncoding` added to `experiments/feature_attention/model.py`: maps `log(scale_mm)` → `dim`-dimensional embedding via log-spaced learnable sinusoidal frequencies (matches `patch_icl_v3`). Added to `PatchICLAttention` behind `use_scale_embed` flag (default off).
- `spacings.json` at the dataset root stores `{"s0000": {"spacing": [dx,dy,dz], "shape": [D,H,W]}, ...}` for all subjects. `convert_to_npy.py` now writes it incrementally (merged with any existing entries). A standalone header-only extraction ran in 16 s for 1228 subjects.
- `TotalSegInContextDataset._load_spacings()` reads `spacings.json` at init and converts to effective mm/voxel: pre-resized path scales by `max_native_shape / T`; crop path uses native spacing as-is. Falls back to 1 mm isotropic if the file is absent.
- `incontext_collate_fn` stacks `"spacing"` → `(B, 3)`. `process_batch` and `validate` compute `scale_mm = (image_size / grid_size) * mean(spacing)` per level and pass it to `model[i](...)`.
- Bug fix: `validate()` sparse-level model call was silently omitting `scale_mm=scale_mm`; fixed.

**Role embeddings (`model.use_role_embed`)**:
- `PatchICLAttention` gains three zero-initialised parameters when `use_role_embed=True`: `tgt_type_embed (1,1,dim)`, `ctx_type_embed (1,1,dim)`, and `ctx_idx_embed Embedding(max_context_size, dim)`. The type embeddings distinguish target from context tokens; `ctx_idx_embed[k]` is added to tokens from context image `k`, letting the model track which context image each token came from. Injected at step 2c (after scale encoding, before label injection), K inferred as `ctx.shape[1] // tgt.shape[1]`. Zero-init means the model starts identically to a checkpoint trained without this flag.
- New config keys: `use_scale_embed: false`, `use_role_embed: false`, `max_context_size: 8`.

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

## ImagePFN checkpoint eval in experiments/2d/eval.py

Added a third backend (`cfg.model=pfn_seg_2d`) to the unified 2D eval script so a
trained `ImagePFN` checkpoint can be evaluated alongside `universeg` and
`universeg_featuresim`.

- `pfn_seg.py` now saves `{model, arch, image_size, context_size}` (was a bare
  `state_dict`) so eval rebuilds the model from the checkpoint alone — no arch
  config to keep in sync. `_orig_mod.` prefix (torch.compile) is stripped on load.
- `eval.py` loads the checkpoint up front, syncs `data.image_size` into cfg before
  building the loader, reconstructs `ImagePFN`, and runs the same recipe as the
  trainer's `run_eval`: bf16 forward → sigmoid → bilinear upsample to native →
  `hard_dice` (native = headline `d_native`, low-res = `d_ds`). Timing reuses the
  shared `inference_times` path.
- Invoked via `base.yaml` + overrides: `model=pfn_seg_2d eval.checkpoint=<path>`.
  (A short-lived `pfn_seg_eval.yaml` was added here then removed on 2026-06-12 — see
  the config-consolidation entry above.)

Note: existing pfn_seg checkpoints saved before this change lack the arch dict and
must be retrained/resaved to be eval-loadable (`train.checkpoint` warm-start, 2026-06-12).

## 2026-06-14 — pfn_seg eval: per-patch error analysis CSV

- Added opt-in `eval.patch_csv` (base.yaml, null = off). When set, the `pfn_seg_2d`
  eval path dumps one CSV row per low-res patch (the `Hp×Hp` grid,
  `Hp = image_size // patch_size`): `dataset, label_value, sample_idx, patch_i,
  patch_j, pred` (sigmoid), `gt` (avg-pool soft fraction), `error` (pred−gt, signed),
  `gt_size` (native fg pixel count), `ctx_dice` (mean native `hard_dice` of target
  GT vs each of the K context masks). Last two are per-sample, replicated per row.
- Zero cost unless `eval.patch_csv` is set; collection reuses tensors already in
  scope and writes once at end via stdlib `csv` (no new deps).

## 2026-06-14 — pfn_seg training: patch-level soft-target loss

- Switched the training objective from native-res binary BCE (on the bilinearly
  upsampled logit grid) to **patch-resolution soft-target supervision**, computed
  directly at the `Hp×Hp` head grid (no upsample):
  `target = adaptive_avg_pool2d(binary_mask, Hp)` (per-patch fg fraction ∈ [0,1]);
  `loss = BCE_with_logits(logits, target) + dice_weight · soft_dice_loss(σ(logits), target)`.
  Soft-target BCE is a proper scoring rule → calibrated to the true patch fraction;
  the soft-Dice term (new `soft_dice_loss`, smoothing eps=1.0) counters background
  imbalance. New config `train.dice_weight` (default 1.0).
- Inference paths (`run_eval`, `eval.py`) still upsample for native-res Dice
  *reporting* only — not a training signal. `dice_ds_soft` now mirrors the objective.
- Architecture unchanged → existing checkpoints resume via the warm-start path
  (`train.checkpoint=...`); verified strict state_dict load on
  `results/2d/pfn_seg_resumed.pt`.
- Soft context masks (symmetric input-side representation) deliberately deferred to
  isolate the loss-term effect first.

- `pfn_seg.py`: set node-local `TRITON_CACHE_DIR`/`TORCHINDUCTOR_CACHE_DIR` (under
  `/tmp/<user>_compile_<hostname>`) before importing torch. `~/.triton` and
  `~/.cache` are on shared NFS, so a `cuda_utils.so` compiled on a GLIBC-2.34 node
  poisoned the cache for GLIBC-2.31 nodes (`GLIBC_2.34 not found` at compile time).
  Keying the cache by hostname makes each node compile its own artifacts.

## 2026-06-14 — pfn_seg patch error-driver analysis

- New script `experiments/2d/patch_error_drivers.py` analyses what drives per-patch
  error in a `patch_analysis.csv` dump, oriented to the refinement-sampling goal
  (separates inference-observable drivers from oracle/GT-only ones). Outputs a text
  report + figures (`patch_error_drivers.txt`, `err_gt_pred_heatmap.png`,
  `err_uncert_modality.png`) next to the CSV. Attaches a best-effort dataset→modality
  map (from MedSegBench identities; not in the data).
- Findings on the soft-loss run (`results/2d/pfn_seg_low_res_loss/.../`):
  - **Error concentrates in boundary patches**: 12.4% of patches (0.05<gt<0.95) hold
    72% of total error mass; pure-bg (81.7%) only 15.5%. Dominant driver = partial
    coverage (boundaryness 4·gt·(1−gt)).
  - **Findable without GT**: `pred_uncert`=4p(1−p) (≡|pred−0.5|) ranks error almost
    perfectly (Spearman ρ=0.988). Top-10/20/30% patches by uncertainty capture
    61/88/96% of error → validates uncertainty-based patch sampling for the refiner.
  - **Blind spot**: confidently-wrong patches (extreme pred, opposite gt) have ~0
    uncertainty → uncertainty sampling misses them; reserve a small quota.
  - Secondary: modality matters (hardest microscopy>dermoscopy>xray; easiest oct/ct/mri;
    hardest datasets robotool/tnbcnuclei/kvasir/brifiseg/monusac — thin/many-object).
    `gt_size` weak (+), `ctx_dice` weak & oracle-only, patch position negligible.
  - Error magnitude only ~⅓–½ predictable (observable R²=0.334, intrinsic R²=0.461):
    rankable, not precisely weightable. NB error≡pred−gt, so a model given both is
    circular (excluded) — drivers come from observable-only and intrinsic models.

## controlSynth: fix gmm_fill IndexError on warped-out foreground (2026-06-20)
- Crash: `synth=hard_diverse` raised `IndexError: index 41 out of bounds for axis 0
  with size 27` in `appearance.gmm_fill` (`lut[fg_label]=fg_mean`). `fg_label=2*num_labels+1`
  (=41 for hard_diverse's `num_labels=20`); the LUT was sized to `labels.max()+1` from
  only the labels *present* in the deformed map.
- Root cause: the elastic warp (`deform`, `support_query_shift=0.50`) can fold a thin/small
  foreground (e.g. tubular, ~1.2% area) entirely out of frame, so `fg_label` exceeds the
  warped map's max present label. Rare (~1/40000 subjects) but inevitable over a 20k-sample
  epoch. Reproduced end-to-end on task 1044 (tubular), `warped.max()=26`.
- Fix: size the LUT to every label `gmm_fill` *writes* (`fg_label` + `distractor_labels`),
  not just those present. Slots for absent labels stay 0 and are never read (`img=lut[label_map]`).
  A vanished-fg subject now renders as all-background with an empty mask — a valid degenerate
  sample, no crash.

## eval: log a checkpoint's training-data provenance (2026-06-20)
- Problem: when evaluating a trained `pfn_seg_2d` / `patchset_pfn` checkpoint, the run had
  no record of what the model was *trained on*. `cfg.data.*` (and thus `run_cfg.source`)
  reflect the *eval* dataset, so a synth-trained vs medsegbench-trained checkpoint were
  indistinguishable in W&B.
- Fix (train side): `pfn_seg.py` and `multilevel/train.py` now embed the full training data
  config in `best.pt` — `"data": OmegaConf.to_container(cfg.data, resolve=True)` plus
  `"synth"` (the controlSynth knob block, or `None` when not `source=synthetic`).
- Fix (eval side): `eval.py` reads `pfn_ckpt.get("data")` / `get("synth")` after loading the
  checkpoint, prints a one-line `Checkpoint trained on: source=…, dataset=…, …` summary, and
  logs them to the W&B run config as `train_data` / `train_synth` (both backends). Old
  checkpoints lacking the keys log `None` and print an "older checkpoint" note — no guessing.

## pfn_seg + multilevel: select best checkpoint on soft Dice at the model's output level (2026-06-20)
- pfn_seg (`experiments/2d/pfn_seg.py`): `run_eval` now returns `dice_ds_soft/mean`
  (low-res soft/shape Dice at the head's native patch grid Hp) instead of `dice/mean`
  (the native-res hard Dice after upsampling preds to image size). All three metrics
  are still logged; only checkpoint selection + the `[best]` message changed.
- multilevel (`experiments/2d/multilevel/train.py`): added per-resolution soft Dice
  `dice_soft_r{res}/mean` (continuous composite vs avg-pooled un-binarized GT, parallel
  to the existing hard `dice_r{res}`), with `dice_soft/mean` = the last computed level.
  Checkpoint selection switched from `dice/mean` (final hard, native) to `dice_soft/mean`.
  Eval line now prints both hard and soft rows; best/summary keys renamed accordingly.
- Rationale: select at the resolution the model is actually supervised at, on a
  threshold-free shape score, rather than an upsampled hard-thresholded metric.

## controlSynth realism: biomedparse mask shape/position/size stats (2026-06-20)
- New `scripts/biomedparse_shape_stats.py`: samples masks balanced across all biomedparse
  datasets (41 train dirs, ~100/ds, N≈3944) and measures position/size/shape. Writes
  `results/controlsynth/biomedparse_shape_stats.{txt,json,png}`. Re-measured synth
  hard_diverse with the same code (`mask_stats`) for a direct gap.
- REAL biomedparse (median / p5..p95):
  center offset 0.158 (p95 0.367; 37% off-center >0.2); area_frac 0.019 (0.0009..0.269,
  ~294x span, heavy small tail); eccentricity 0.81; solidity 0.91; extent 0.59;
  n_cc median 1 (69% single-component, only 31% multi); border-touch 6%.
- SYNTH hard_diverse gap (the "centered + same size + ragged" issue):
  * position too centered: offset p95 0.24 vs 0.37, only 9% off-center vs 37%; centroid
    std ~0.09 vs ~0.14.
  * size too narrow/large: area_frac 0.056 median, ~28x span (region_size floored at 0.30
    → no small-object tail); real is 0.019 median, 294x span.
  * BIGGEST gap = fragmentation: n_cc median 14, 87% multi-component, solidity 0.57,
    extent 0.25 — real masks are mostly single compact regions (median 1 cc, solidity 0.91).
  * border-touch 39% vs 6% — synth foregrounds run off-frame far too often (deformation).
- Next (not yet done): translate targets into generator knobs — random fg placement
  (centroid offset), log-distributed area with a small tail, reduce fragmentation
  (boundary/deformation), keep fg in-frame.

## controlSynth realism: generator + hard_diverse config (2026-06-20, implemented)
- Added 4 backward-compatible DifficultyBuildSpec knobs (defaults = original behavior;
  set only in hard_diverse.yaml so the difficulty-study presets are untouched):
  boundary_amp_scale, boundary_sigma_frac, boundary_keep_largest (shapes/boundary.py),
  position_jitter (new task.place_foreground). Knobs flow via geo_params (no signature change).
  * boundary fix = the big one: roughening shattered shapes (n_cc median ~12); gentler
    amplitude (0.5) + low-freq blur (sqrt(area)*0.4) + keep-largest (blob/elongated/annular
    only) → mostly single compact regions.
  * place_foreground translates the finished shape to centroid ~N(0.5, jitter), clamped to
    keep its bbox in-frame.
- hard_diverse.yaml: boundary_amp_scale=0.5, boundary_sigma_frac=0.4, boundary_keep_largest=true,
  position_jitter=0.15, region_size range [0.30,0.70]→[0.12,0.62] (moderate floor).
- Result (val masks, vs real biomedparse): n_cc med 12→2, solidity 0.57→0.83, area 0.050→0.028
  (real 0.019), offset mean 0.14→0.18 / p95 0.27→0.31 (real 0.37). Sample plot regenerated
  (results/controlsynth/hard_diverse_samples.png) — visibly compact + off-centre.
- CAVEAT (not resolved): border-touch ~34% vs real 6%. Mostly the post-placement deformation
  (shift=0.50, ~6px) pushing inset-2 shapes over the edge + sprawling tubular trees. Would
  need a larger placement inset (vs deform magnitude) or in-frame deformation to fix.

## controlSynth realism: within-(target+context)-set mask distance (2026-06-20)
- New `scripts/context_mask_distance.py`: per in-context set (target label + K context
  masks), averages pairwise overlay_dice, centroid_dist, area_logratio. Run on
  biomedparse (1500 sets) and synth hard_diverse (800 sets), res=128, K=3.
- REAL biomedparse: overlay_dice 0.25 macro (0.18 micro, median 0.10), centroid_dist 0.159,
  area_logratio 1.20 (~2.3x within-set area swing). Many cells ~0 overlap (structure
  relocates entirely across patients: LIDC/colon/polyp/disc).
- SYNTH hard_diverse: overlay_dice 0.27 (median 0.24), centroid_dist 0.105, area_logratio 0.57.
- Finding: synth context sets are too SELF-SIMILAR — real within-set spread comes from
  independent patients; synth derives all K+1 subjects from ONE base geometry varied only
  by per-subject elastic deform (support_query_shift). position_jitter is per-TASK (shared
  by the set) so adds no within-set positional variability. To match real, add per-SUBJECT
  position (centroid 0.105->~0.16) + scale (logratio 0.57->~1.2) jitter — but bounded, else
  the fg stops being the consistent in-context cue. NOT implemented.

## controlSynth realism: per-subject pose jitter (within-set spread) (2026-06-20, implemented)
- Added 2 backward-compat DifficultyLiveConfig knobs (default 0 = original; declared in
  default.yaml live, set in hard_diverse.yaml): support_query_translate (per-subject centroid
  shift std, fraction of image), support_query_scale (per-subject log2 zoom std). Applied via
  new deformation.jitter_pose (affine about fg centroid, clamped in-frame), after deform in
  dataset._make_subject, scaled by shift_scale so pristine contexts stay near-aligned.
- Rationale: support_query_shift only deforms ONE shared base -> synth (target+ctx) sets too
  self-similar. position_jitter was per-TASK (no within-set spread).
- Tuned to tr=0.05, sc=0.45 (closes ~80% of position gap, most of size gap without collapsing
  within-set overlap below real; pushing tr>=0.06 overshoots centroid + drops overlap to ~0.14):
    within-set overlay/centroid/area_logratio: 0.250/0.113/0.546 -> 0.168/0.146/1.042
    (real biomedparse 0.254/0.159/1.202).
  Single-mask shape stats unregressed (area med 0.022, n_cc med 1, solidity ~0.78, offset 0.18).
- Sample plot regenerated: contexts now vary in position+size within each set.

## 2D eval: qualitative figures for all backends (2026-06-21, implemented)
- Generalized experiments/2d/eval.py:save_figure to be backend-agnostic. New signature takes
  pred_native (H,W; soft for feat_sim/pfn_seg/multilevel, binary for universeg) plus an optional
  coarse grid (pred_lowres/gt_lowres). Row 0 = Target+GT | Target+Pred | (GT↓ | Pred↓ only when a
  low-res grid is supplied); Row 1 = K context overlays. Panels built dynamically, squeeze=False,
  so K=1 and the no-low-res (universeg) case both render.
- Prediction resolution per backend (figure source tensors): feat_sim pred_lowres=preds_all[b]
  at output_size, native via bilinear upsample; pfn_seg pred_lowres=preds_lowres[b] at Hp, native
  preds[b]; multilevel pred_lowres=refined_grid at Hg (captured as preds_grid before upsample,
  only shown when Hg!=H), native preds[b]; universeg native binary preds[b,0], no low-res.
- Replaced the feat_sim-only inline save with one unified block in the per-sample loop, gated by
  new config knobs eval.save_figures (bool, default false) + eval.max_figures (cap, default 50).
  Dedup stays one figure per (dataset, label_value). Previously only feat_sim emitted figures and
  always-on; now all four backends can, opt-in.

## 2D eval: wandb-named output folders (2026-06-21, implemented)
- eval.py no longer builds descriptive run names per backend; wandb.init(name=cfg.wandb.name)
  so a null name auto-generates (e.g. "deft-field-72"). out_dir now mirrors pfn_seg.py:
  {eval.out_dir}/{date}_{run_name}, so each run's figures/CSVs land in their own folder instead
  of all dumping into a shared results/2d/outputs. Added eval.figures_to_wandb (default true) to
  toggle wandb upload independently of local PNG saving.

## TotalSegmentator 2D cross-section manifest (2026-06-21, implemented)
- New scripts/build_totalseg_2d_manifest.py: picks ONE axial cross-section per subject that
  is densest in the most common classes, emits one (subject,class) task row per class present.
  Output: results/totalseg2d/manifest_128.csv (17621 rows, 1228 subjects, 110/121 classes;
  splits from meta.csv: 1082 train / 57 val / 89 test; ~14 tasks/subject).
- Axial axis = AXIS 2 of label_{S}x{S}x{S}.npy (verified, non-obvious): a true cross-section
  cuts <=2 vertebrae, only axis 2 satisfies this (axes 0/1 span 7-18 vertebrae = coronal/sagittal).
  Axis-2 slices are also organ-richest.
- Selection score = soft area-ramp: sum over classes (area>=noise_floor) of
  global_weight[class] * min(1, area/area_cap). global_weight = occurrences/max from
  label_stats.csv. Ramp rewards substantial presence (not grazing many organs by a few px) and
  is robust (top slices form one tight cluster; a hard min_area threshold flips between distant
  abdominal/pelvic slices near ties, and a tiny floor selects fragment-heavy edge slices).
  Defaults noise_floor=10, area_cap=100. Task emission decoupled: emit classes with
  area>=task_min_area (default 25), so slivers don't become degenerate targets nor sway selection.
- Selected slices land on the thoraco-abdominal belt (lungs, heart, aorta, autochthon, liver,
  spleen, kidneys, pancreas, stomach, IVC) — matching the global most-common classes.
- NOTE for the dataset step: ct_{S}x{S}x{S}.npy is float16 and pre-normalized (~z-scored, range
  ~[-1.7, 3.3]), NOT [0,1] — a TotalSeg2D dataset must min-max/clip-rescale per slice to match the
  MedSegBench [0,1] convention before serving to the 2D in-context pipeline.

## TotalSegmentator 2D slice export to npz (2026-06-21, implemented)
- New scripts/totalseg2d/to_npz.py: exports ONE axial cross-section per subject to a single
  npz for the 2D experiments. Two deliberate specs vs convert_to_npy.py:
  1. FIXED mm/pixel (default 2.0mm, size 256 -> 512mm FOV), centered on the label. Fixes the
     cross-subject scale drift from longest-axis->cube normalization (s0919 vs s1158 was ~4.5x
     frame-area mismatch in the 128^3 cubes; now ~1.4x, i.e. genuine anatomy only).
  2. RAW int16 HU, no normalization (same bytes as float16, exact). Clip/z-score/windowing is
     deferred to the dataloader for flexibility. Disk/time cost of deferring measured ~0:
     int16 raw == float16 size; ~0.5ms/slice to normalize at load.
- Reads native label.npy (already canonical, axis2=axial) + ct.nii.gz (for raw HU only) +
  spacings.json. Slice selection = soft area-ramp over global class freq (build_totalseg_2d_manifest),
  but areas in PHYSICAL units (voxel count x in-plane mm^2 -> output px at mm_per_px) so argmax is
  FOV-consistent and self-contained (supersedes the cube-based manifest's z for storage).
- Per-slice render: in-plane ndi.zoom to mm_per_px (Gaussian AA on image downsample, nearest for
  label) then center on label centroid into a size x size grid; image pad = AIR_HU (-1024).
- npz schema per split: {split}_images int16 (N,size,size), {split}_label uint8, {split}_subjects,
  {split}_z, {split}_spacing; scalars mm_per_px/size/air_hu + class_names. savez_compressed.
- Smoke: 12 test subjects in 11s, 0 errors; verified raw HU range, scale fix, centering visually.

## TotalSeg2D in-context dataloader (2026-06-21, implemented)
- New src/datasets/totalseg2d.py: TotalSeg2DDataset mirrors MedSegBenchDataset's interface
  (returns {image,label,context_in,context_out}; exposes .samples/.label_index) so it plugs into
  the 2D pipeline via common.build_dataset with data.source="totalseg2d".
- Reads totalseg2d_{stored_size}.npz (to_npz output). Differences vs medsegbench:
  * Images are raw int16 HU -> normalized HERE: clip to data.hu_window (default [-1000,1000]) then
    min-max to [0,1] (clipping also tames extreme out-of-FOV HU outliers, e.g. -8653 seen in data).
  * Loads the 256px export (512mm FOV) and resizes to data.image_size (bilinear img / nearest label),
    decoupling model res from FOV. The 128px export is a tighter 256mm FOV (clips bodies) — avoided.
  * data.min_area (default 16 px @ image_size): a (subject,class) pair is a sample only above this,
    so tiny slivers aren't degenerate targets.
  * Single "dataset" name; label_value = TotalSeg class index 1..117.
- Wired into experiments/2d/common.py build_dataset (new totalseg2d branch + error-msg update).
- Verified with configs/augmentations/medsegbench.yaml via pfn_train.augment: geometric (flip/rotate)
  hits context pairs only, intensity (brightness/contrast/gamma/noise) hits all, query mask untouched,
  outputs stay in [0,1]. Plot: results/totalseg2d/dataloader_aug_samples.png (orig vs aug).

## 2D aug: single shared config + strong preset (2026-06-22)
- De-duplicated the 2D aug config. The identical `aug:` block was inlined in both
  configs/experiment/2d/pfn_seg.yaml and multilevel.yaml; replaced each with `aug_preset: 2d`
  and a code-load (`OmegaConf.load(configs/augmentations/<preset>.yaml)` merged into cfg.aug in
  main(), mirroring the 3D experiments/multilevel/train.py pattern). New canonical file:
  configs/augmentations/2d.yaml (2D schema: enabled/geometric/intensity; the old medsegbench.yaml
  is now an unused orphan). CLI field override uses the +-prefix, e.g. +aug.enabled=false.
- Extended pfn_train.augment() with literature-backed, backward-compatible ops (off unless the
  preset sets the keys; base 2d.yaml takes the identical legacy path — affine theta reduces to the
  old rotate-only matrix):
  * task.invert: episode-wide intensity inversion (UniverSeg "task augmentation"). Intensity-only,
    shared across all K+1, so the query image can be touched without desyncing its GT.
  * geometric.scale/translate: folded into the existing rotate affine (one grid_sample); context only.
  * geometric.elastic: smooth displacement field (low-res random -> bilinear upsample = smoothing);
    context pairs, image bilinear / mask nearest.
  * intensity.bias_field: smooth multiplicative inhomogeneity exp(field*strength) (SynthICL); all images.
  * query_perturb: extra independent noise on the query slot only (Iris "imperfect reference").
  * geometric.crop: random-resized crop — sample an in-bounds sub-window (relative size
    s∈[min_scale,1], shift bounded by 1-s so it never leaves the image → no border padding)
    and resize to full H×W. Context only; lets each context show a different region.
    In 2d_strong: p=0.5, min_scale=0.75. Verified a solid context stays unpadded post-crop.
  Geometric stays context-only by design: the training target is read from the un-augmented batch,
  so moving the query would break query/GT alignment. class dropout (Iris) is dataset-level, N/A here.
- New preset configs/augmentations/2d_strong.yaml enables all of the above (use aug_preset=2d_strong).
- Verified: both presets through augment() preserve shape, keep masks binary, leave the query mask slot
  zero, and clamp images to [0,1]. A/B vs 2d.yaml still TODO (gains may be aug strength, not transform).

## COW-safe dataset indexes (2026-06-22, implemented)
- Cause of DataLoader worker RAM creep / periodic stalls: forked workers share memory
  copy-on-write; COW copies a page only on WRITE, and the only writes during iteration are
  CPython refcount bumps on PyObject headers. The big image/label numpy buffers are read-only
  (safe), but the per-sample `list[(ds,idx,lv)]` and the `dict[..]->list[tuple]` context-lookup
  structures get refcount-churned every __getitem__ -> pages fork per worker -> RSS climbs over
  the run (persistent_workers never resets).
- Fix: src/datasets/cow_index.py — SampleIndex stores the (ds_id,img_idx,label) triples as 3
  contiguous int32 arrays + a small ds_names list, but still behaves like list[(ds,idx,lv)]
  (len/index/iter/subset) so common.TaggedDataset/collate/eval are unchanged. build_candidate_index
  groups image idxs by (ds_id,label) into read-only int32 arrays (replaces label_index/group_index).
  sample_context picks K context idxs from a candidate array via the `random` module (positions, not
  objects). Workers now read only numpy buffers -> no per-element refcount COW.
- Applied to src/datasets/{medsegbench,totalseg2d,biomedparse}.py (__init__ build + __getitem__
  lookup; biomedparse diversity tags + self-test updated). common.make_loader eval-subsampling uses
  SampleIndex.subset (no list rebuild).
- Verified: SampleIndex unit (tuple iface + subset), biomedparse self-test, totalseg2d/medsegbench
  item correctness, and 2 epochs through 3 persistent workers via TaggedDataset+collate.
- NB: still the bigger lever for the user's 48-worker/64GB case is fewer workers (~8-12) — an
  all-in-RAM dataset with cheap __getitem__ is GPU-bound; this fix removes the per-worker creep that
  made the stalls recur every few epochs.

## 2026-06-22 — Log val loss in 2D training scripts
- experiments/2d/pfn_seg.py and experiments/2d/multilevel/train.py now compute and log
  `val/loss` in their `run_eval`, accumulated with the same objective each uses for train
  loss (pfn_seg: BCE + dice_weight*soft_dice on the avg-pooled patch target; multilevel:
  the per-hop `loss_weights`-weighted `patch_loss` sum over chain outputs). Logged to wandb
  alongside the dice metrics and added to the per-epoch eval summary line.

## 2026-06-22 — Shared train_base.yaml for 2D training configs
- New configs/experiment/2d/train_base.yaml holds params common to pfn_seg.yaml and
  2d/multilevel.yaml (full data block; arch-core e/h/l/a/feature_level/thinking_rows/
  residual_decay/compile; full train block; eval.batch_size/workers/out_dir; wandb;
  aug_preset). Mirrors the eval-side base.yaml pattern (defaults: [synth: default, _self_]).
- pfn_seg.yaml / 2d/multilevel.yaml now do `defaults: [train_base, _self_]` and keep only
  model-specific + differing keys (encoders/resolution for pfn_seg; sample block, mask_prior/
  stage1 thinking/loss_weights/stage1_checkpoint for multilevel; per-config max_per_label).
- eval.out_dir set to .../2d_train for BOTH training scripts (multilevel previously .../2d).
- Verified composition with `--cfg job`: resolved configs match prior values (out_dir aside).
  Eval-only base.yaml and its consumers (feature_sim.yaml etc.) are untouched.
- Renamed configs/experiment/2d/base.yaml → eval_base.yaml for symmetry with train_base.yaml.
  Updated refs: feature_sim.yaml (defaults), eval.py + synth_benchmark.py (config_name).
  Verified eval.py / feature_sim.py compose unchanged via `--cfg job`.
- eval.py universeg backend: dice_ds / dice_ds_soft now logged as NaN instead of a copy of
  the native dice. UniverSeg emits only a native-res binary mask (no low-res/soft coarse grid),
  so for binary pred+GT both low-res metrics were mathematically equal to dice — misleadingly
  implying a shape score. log_summary's NaN-filtered aggregation drops them. Other backends
  (pfn_seg/feature_sim/patchset_pfn) score a genuine coarse grid and are unchanged.

## 2026-06-22 — Train-accuracy metrics in 2D training (near-zero cost)
- New common.batch_dice_sums(prob, target): vectorised, on-device soft + hard Dice SUMS +
  valid-row COUNTS, same per-row semantics as soft_dice/hard_dice (hard binarises both at
  0.5; empty pred+GT rows skipped). Verified numerically equal to the per-sample helpers
  (eval-style GT binarization). Lets train accuracy accumulate on GPU and sync once/epoch.
- pfn_seg.py train_epoch now returns (loss, train_dice_soft, train_dice) computed from the
  logits/target the forward already produced (reuses the sigmoid feeding the dice loss) —
  no extra forward, no per-batch GPU→CPU stall. Logged as train/dice_soft and train/dice,
  mirroring val dice_ds_soft / dice_ds so the train↔val gap is directly readable.
- multilevel/train.py train_epoch returns per-hop dicts; logs train/dice_soft_r{grid}/mean
  and train/dice_r{grid}/mean (mirrors val dice_soft_r* naming). Measured on each hop's
  sampled query patches (what it trains on), so reads slightly above the full-grid val metric.
- universeg_train.py: train Dice was never logged (train_epoch only returned loss; the
  soft-dice term was folded into the loss, and log_summary ran only on val output). Added
  per-batch hard_dice accumulation over train predictions; train_epoch now returns
  (loss, train_dice), logged as train/dice and shown in the per-epoch console line + tqdm
  postfix. Monitoring only (under no_grad), independent of synth=omniglot / pretrained flags.
- New model src/models/patchset_cnn.py (PatchSetCNN): trainable single-stream conv
  encoder downsamples each image to an R×R grid (default 16), then ImagePFN's dual-axis
  TransformerEncoderStack reasons in-context over [img | mask-occupancy] columns. Mask
  token = scalar avg-pool occupancy (Linear(1→e)); query mask = context-mean prior. Head
  is a per-patch Linear → R×R logits (no decoder/upsampling); loss vs avg-pooled GT.
- Fused experiments/2d/{universeg_train.py + patchset_cnn trainer} → experiments/2d/train.py.
  One model-agnostic loop: both models called as model(img, context_in, context_out, mode)
  -> {"final_logit"}; GT is avg-pooled to the logit's spatial size (no-op for UniverSeg
  H×W, R×R downsample for PatchSetCNN). Model selected by cfg.model (universeg | patchset_cnn).
  New config configs/experiment/2d/patchset_cnn_train.yaml; universeg_train.yaml kept.
  Removed universeg_train.py.
- Removed ic_segmentation dependency entirely. universeg_baseline lived only in
  ic_segmentation/src/models and was reached via common.py putting ic_segmentation on
  sys.path, whose src/__init__.py shadowed patch_icl's src. Vendored universeg_baseline.py
  into src/models/, dropped the ic_segmentation sys.path insert in common.py, and removed
  the shadowing workarounds (eval.py importlib-by-path load of pfn_seg_2d → plain import;
  stale "cache before ic_segmentation" comments in pfn_seg.py/multilevel/train.py/
  plot_synth_samples.py). `src` is now unambiguously patch_icl.
- Low-res Dice metric: hard Dice at R×R is degenerate for thin omniglot strokes (avg-pooled
  GT binarized at 0.5 is ~empty → NaN). train.py now logs BOTH soft Dice (threshold-free,
  prob vs occupancy) and hard Dice (pred≥0.5, GT>0) for train and val, and selects the best
  checkpoint on soft (dice_soft/mean). Mirrors multilevel/train.py's dice_soft/dice pair.
- PatchSetCNN encoder reworked to UniverSeg per-stage widths + multi-scale concat:
  enc_dims default [64,64,64,64] (UniverSeg v1 encoder_blocks). ConvEncoder now keeps
  the stem + every stage feature map, avg-pools each to R×R, and concatenates along
  channels (out_ch = sum(enc_dims) = 256) instead of returning only the last stage.
  img_embed = Linear(sum(enc_dims) → e) adapts automatically; per-patch tokens now carry
  low- through high-level features. ~4.44M params at e=256.
- train.py metric consistency: `dice` (hard) is now computed at the ORIGINAL resolution
  — preds upscaled (bilinear) to H×W, thresholded ≥0.5, vs the full-res GT (>0). This is a
  no-op for native-res UniverSeg, so the headline `dice` is directly comparable across
  models/resolutions. The soft metric (prob vs pooled occupancy at the model's logit res)
  is renamed dice_soft → `dice_ds_soft` (downsampled). wandb: train/dice_ds_soft, train/dice,
  dice_ds_soft/*, dice/*; best-checkpoint still on dice_ds_soft/mean. NB: a 16×16 PatchSetCNN
  upscaled to 128 cannot recover sub-patch thin strokes, so its native `dice` is capped — it
  reads the achievable native accuracy, while dice_ds_soft reflects the training objective.
- PatchSetCNN switched from image-grid to SET-of-patches attention (PatchSetPFN layout).
  Motivation: the image-grid layout's cross-image (sample-axis) attention is position-
  locked — query patch (i,j) attends only to context patch (i,j) — which is wasteful when
  objects aren't spatially aligned across images (omniSynth chars in random cells). New
  layout: rows = thinking + support patches (K·N) + query patches (N); cols = [img|mask].
  Sample-axis attention is now full-set content matching (query patches attend to all
  support patches); the patch (i,j) is injected as a Fourier positional FEATURE
  (FourierPositionalEncoding reused from patchset_pfn) instead of being enforced by
  structure. Replaced the learned pos_embed; per-channel feature standardization now uses
  support-patch stats. Query mask token = support-mean occupancy prior.
  Supporting fix: pfn_seg_2d TransformerEncoderLayer feature-axis now uses batched_sdpa
  (was plain SDPA) so the b*rows grid stays under the CUDA gridDim.y cap (65535) when rows
  = all patches (e.g. B=64, R=16 → 65600 rows). No-op for existing small-batch ImagePFN use.
- PatchSetCNN: added query_self_attn flag (arch.query_self_attn, default false), mirroring
  PatchSetPFN. When true, query patches attend to each other (within-target spatial
  reasoning) in addition to the support set, via an (r×r) bool attn_mask passed to the
  sample-axis. Queries carry the prior (not GT) so query↔query attention leaks no labels.

## 2026-07-03 — omnisynth_train_base.yaml between train_base and the omniSynth leaves
- universeg_train.yaml and patchset_cnn_train.yaml duplicated every non-model key
  (data.source=omnisynth, the 50-epoch/batch-64 schedule, eval batch/workers/max_per_label).
  Factored the shared keys into a new intermediate config
  configs/experiment/2d/omnisynth_train_base.yaml (defaults: [train_base, override
  synth: omniglot, _self_]). Leaves now carry only model-specific keys: `model`, their
  `arch` block (patchset only; universeg ignores arch), `train.lr`, universeg `pretrained`.
- train_base.yaml deliberately left untouched so the medsegbench/ImagePFN configs
  (pfn_seg, multilevel, multilevel_zoom) keep inheriting source=medsegbench + the long
  200-epoch schedule + the transformer arch block.
- Side effect: the intermediate defaults `synth: omniglot`, so `synth.scene.*` overrides
  (e.g. synth.scene.p_copy / synth.scene.grid) now work without passing synth=omniglot.
  Verified both leaves resolve to their pre-refactor values via `train.py --cfg job`.

## 2026-07-03 — Store full PatchSetCNN arch in the checkpoint
- train.py:build_model previously saved only a partial patchset_cnn ckpt_meta
  (resolution/enc_dims/query_self_attn/context_id_embed/max_context), omitting the
  transformer knobs (e/h/l/a/thinking_rows/residual_decay/fourier_bands) — so a best.pt
  could not be rebuilt without also supplying the training config.
- Now build_model constructs a single `arch` dict (all PatchSetCNN kwargs except
  image_size), uses it to instantiate the model, AND returns it as ckpt_meta -> the
  checkpoint stores it under `arch` (nested, mirroring pfn_seg_2d). A focused eval script
  can rebuild via PatchSetCNN(image_size=ckpt["image_size"], **ckpt["arch"]). Verified the
  round-trip yields an identical state_dict (keys+shapes). image_size stays at the ckpt
  top level (unchanged). Nothing else consumes patchset_cnn checkpoints, so the shape
  change (flat keys -> nested `arch`) is safe.

## 2026-07-03 — Shared validate() + focused eval_incontext.py (universeg / patchset_cnn)
- Extracted the per-epoch eval loop into experiments/2d/evaluate.py as a single shared
  validate() used by BOTH train.py and the new eval script (metrics coherent by construction).
  It computes native `dice` plus, for low-res models, `dice_ds`/`dice_ds_soft`/`cossim`/`top{k}`,
  fills an adaptive per-sample table (source-aware `_sample_detail`), and gates figures /
  patch_csv / synth_csv / a one-shot FLOPs count behind opt-in args.
- train.py now imports validate/_target_like/_upsample_to from evaluate.py (dropped its own
  copies + _fmt_transforms/SAMPLE_COLS and 6 now-dead imports). Behavior unchanged: low-res
  models checkpoint on cossim, native on dice (verified by 1-epoch smoke runs).
- eval.py DRY: its save_figure/_overlay_ax/_heatmap_ax now live in evaluate.py (eval.py imports
  save_figure). Its 5-backend dispatch is otherwise untouched.
- New experiments/2d/eval_incontext.py: thin Hydra wrapper (reuses eval_base.yaml) that loads a
  train.py checkpoint, dispatches on model_name, rebuilds universeg or patchset_cnn (via
  **ckpt["arch"]), and runs the shared validate() with figures/CSVs/FLOPs on. Fails loudly on
  pre-arch patchset_cnn checkpoints. Injects the checkpoint's stored `synth` block into cfg.synth
  so eval reproduces the model's exact training distribution (eval_base defaults to the
  controlSynth synth schema; omniSynth models need the omniglot schema — without this the eval
  silently ran on default scene config and scored a misleading ~0.07 vs the correct ~0.86).

## 2026-07-03 — eval_incontext.py: CLI omniSynth overrides (OOD eval)
- Added configs/experiment/2d/eval_incontext.yaml (defaults: [eval_base, override synth: omniglot,
  _self_]) and pointed the wrapper's @hydra.main at it. Defaulting the synth group to omniglot lets
  `synth.scene.*` compose (eval_base's controlSynth default has no `scene` -> "not in struct").
- Wrapper now uses the checkpoint's stored synth as the BASE, then merges CLI `synth.*` overrides on
  top (read from HydraConfig.get().overrides.task) so they win. Unspecified keys still reproduce the
  training distribution. Enables OOD eval, e.g. `synth.scene.grid=2 synth.scene.target_mode=class`.

## 4_weaknesses.py — universeg vs patchset_cnn weakness analysis (omnisynth)
- Added marimo analysis cells (runs uixvcpny=universeg, kq1cent3=patchset_cnn). The two eval
  tables have different schemas: universeg has explicit cols; patchset packs everything into a
  `detail` string ("char mode= cells= tf="). Cells parse `detail`, unify into one long df, decode
  `transforms` -> rot/scale/dx/dy features, and build a paired merge on
  (dataset, character, target_pos, transforms).
- Findings: universeg mean/median Dice 0.76/0.91 (bimodal: 52% >0.9, 10% <0.1); patchset_cnn
  0.46/0.50, unimodal peak ~0.5, 0% samples >0.9. Universeg wins 82.6% of paired samples.
- Both models: Dice *increases* with |rotation| — the (0,5]deg bin is the hardest
  (uni 0.57, pat 0.41), rising to ~0.88/0.50 by 15-20deg. target_pos has ~no effect.
  Spearman(dice,|rot|): uni +0.35, pat +0.22; scale/translation weak.
- NB: marimo/wandb/seaborn live in `.venv` (py3.12), not `.venv311`. Verification figs:
  results/experiments/4w_fig{1_dist,2_rot,3_paired}.png.

## 4_weaknesses.py — driver analysis (what drives Dice for each model)
- Enriched the paired frame (`mg`): borrowed universeg's `context_pos` onto patchset (same
  seeded samples) to get target-context grid distance for both; parsed transforms; added
  per-character object size = mean ink fraction of the Omniglot glyph (via
  src/datasets/omniSynth/bank.py over val+test pools, keyed by class_id in "alphabet/class_id").
- Quantified with eta^2 (variance-explained). The two models are driven by DIFFERENT things:
    driver              universeg  patchset_cnn
    object size (ink)      0.008     0.094
    character id           0.050     0.242
    dataset/alphabet       0.015     0.107
    target |rotation|      0.174     0.073
    scale deviation        0.009     0.015
    translation            0.033     0.010
    tgt-ctx distance       0.001     0.001
- universeg: accuracy driven by AUGMENTATION (rotation); robust to object identity/size.
  patchset_cnn: driven by OBJECT IDENTITY/SIZE (character 0.24, size 0.09, dataset 0.11);
  fails on small/thin glyphs (ink-quintile Dice 0.37->0.51 vs universeg 0.72->0.81).
- target-context grid distance is negligible for both (eta^2 0.001; ~flat curve).
- Rotation paradox (more |rot| -> higher Dice) is NOT a size confound:
  spearman(|rot|, ink_frac)=0.004; independent. Since target rotation is independent of the
  context, this looks like a target-mask rendering/Dice-geometry effect (thin axis-aligned
  strokes at |rot|~0), not a target-context similarity effect. Open question.
- Cross-model per-character difficulty correlates only rho=0.35 (dataset-level 0.46): partly
  shared difficulty, but the models fail on different objects. patchset uniformly below diagonal.
- Figures: results/experiments/4w_fig{4_drivers,5_size_dist,6_charscatter}.png.

## omniSynth eval logging: self-contained context + real (x,y) positions
- evaluate.py `_sample_detail`: the newer source-agnostic `detail` schema was dropping
  `context_cells`. Added `ctx=` (context cells) and `sub=` (subject_index) so an
  omniSynth-only run is self-contained (context position + per-sample join key). Kept the
  free-form `tf=` last for easy parsing.
- New: real post-aug glyph positions (not just grid cell indices).
  - render.py `render_scene` now returns `info["target_positions"]`: per target cell, the
    ink centre-of-mass in full-canvas coords normalised to [0,1] (x right, y down), via
    `_centroid_xy` (falls back to cell centre if a placement has no ink). Captures cell +
    shift + rotate/scale + glyph asymmetry — the continuous counterpart of target_cells.
  - dataset.py meta gains `target_positions` + `context_positions` (aligned with the cells).
  - `_sample_detail` logs them as `pos=(x,y)...` / `cpos=...` (via `_fmt_positions`).
- Motivation: grid-cell tgt-ctx distance was too coarse (eta^2~0.001 in 4_weaknesses.py);
  continuous positions enable a real distance driver. Existing runs unaffected (re-run needed).
- Verified: render + dataset tests pass; deterministic val sample renders correct centroids
  (cell (0,1) center x0.75,y0.25 -> centroid 0.726,0.247 under dx-0.07 shift).

## Fixed-resolution downsampled Dice for native-res models (UniverSeg)
- New config knob `eval.ds_metric_res` (in universeg_train.yaml, default `[16]`, list-capable,
  null/[] disables). UniverSeg predicts at native full-res so dice_ds/dice_ds_soft are NaN;
  this also logs hard + soft Dice with GT and pred both avg-pooled to RxR (hard thresholded at
  >=0.5), matching patchset_cnn's low-res grid recipe.
- evaluate.py: `validate(..., ds_metric_res=None)` + `_as_res_list` helper. Per batch, pools
  prob_nat & lbl to each R (batched), accumulates per-(dataset,label), emits summary keys
  `dice_ds@{R}/*` and `dice_ds_soft@{R}/*` via log_summary. Sample table (SAMPLE_COLS) unchanged.
- train.py: `train_epoch` reads `cfg.eval.ds_metric_res`, accumulates on-device running sums per
  R (reuses _soft_sum/_hard_sum on pooled maps), returns them as a 6th dict element; main logs
  `train/dice_ds@{R}` + `train/dice_ds_soft@{R}`. Checkpoint selection metric unchanged (native dice).
- Cost (measured, B=64, 128->16 on GPU): +0.30 ms/batch = 0.12% of a train step, 0.28% of eval;
  ~0.05s/epoch train, ~0.15s/full val. Negligible.
- Verified: 1-epoch smoke run (grid=2, ds_metric_res=[16,32]) logs all four metric families
  clean; other validate/train_epoch callers unaffected (default None / distinct local funcs).

## omniSynth: larger, overlapping glyphs via cell_margin (wired + allows <0)
- Problem: chars filled only 80% of their cell (hardcoded 0.1 margin in bank._to_bitmap) and
  cells tiled the canvas non-overlapping -> glyphs looked far apart, little ink/training signal.
- cell_margin is now threaded through (was dead config): glyph size = (1-2*margin)*cell.
  margin>0 insets inside the cell (old look); 0 fills it; <0 makes the glyph LARGER than its
  cell and overflow into neighbours.
- bank.py: _to_bitmap renders the glyph at inner=(1-2*margin)*cell into a tile=max(cell,inner)
  (so margin>=0 stays exactly cell-sized -> byte-identical + tests pass). get_or_build_bank /
  OmniglotBank take cell_margin (in the cache key). dataset.py passes scene.cell_margin.
- render.py: cells no longer written as disjoint slices. Each glyph is pasted centred on its
  cell centre via _paste (np.maximum union, clipped to canvas) so oversized glyphs overlap
  instead of overwrite; order-independent. Target mask = union of target tiles; positions =
  centroid of the pasted (clipped) target ink. New _paste/_paste_centroid; dropped _centroid_xy.
- config omniglot.yaml: cell_margin -0.15 (glyph 1.3x cell, small overlap). Tunable; visual
  sweep saved to results/experiments/omnisynth_margins.png (ink fraction 0.046 -> ~0.14 at -0.15).
- Backward compat: OmniSceneConfig default stays 0.1 -> all existing tests unchanged. Added
  test_oversized_tiles_overflow_and_union. 22 omniSynth tests pass; 1-epoch train smoke clean.

## Consistent dice_ds@{R} naming for patchset_cnn (match universeg)
- Patchset's coarse-grid metrics were logged unqualified (dice_ds, dice_ds_soft) while universeg
  used the resolution suffix (dice_ds@32). Now both carry @{R}.
- evaluate.py validate: capture `low_res` = non-native model's logit side length; log its coarse
  Dice as `dice_ds@{low_res}` / `dice_ds_soft@{low_res}` (skip entirely for native universeg, which
  previously emitted all-NaN dice_ds/*). Resolution is auto-detected, no config needed for patchset.
- train.py train_epoch: detect non-native low_res, accumulate hard Dice at that grid, and return
  `dice_ds@{low_res}` + `dice_ds_soft@{low_res}` in the metrics dict; returns low_res. main skips the
  generic `train/dice_ds_soft` for non-native models (universeg keeps it = full-res soft). tqdm.write
  and eval_incontext.py print now look up the dice_ds(_soft) key dynamically (any @R).
- eval_incontext.py: also passes ds_metric_res through so standalone eval honors the knob.
- Per-sample SAMPLE_COLS table keeps generic `dice_ds`/`dice_ds_soft` columns (fixed schema); only
  the scalar summary/train metric names got the @R suffix. Legacy eval.py untouched (separate script).
- Verified: patchset smoke logs hard@16/soft@16; universeg smoke logs hard@32/soft@32 (ds_metric_res
  =[32]); checkpoint selection unchanged (cossim / native dice).

## omniSynth: fix border-cell glyph cropping from cell_margin<0
- Bug: with cell_margin<0, oversized glyph tiles in the 12/16 border cells overflowed the canvas
  and _paste clipped them -> 13% of glyphs cropped >1% at margin=-0.15 (up to 42%), 25% at -0.25.
  Interior overlap was fine; only canvas-edge glyphs were cut.
- Fix (render.py _clamp_center): nudge each glyph's paste centre inward so its square tile stays
  fully on-canvas (border glyphs shift <=~5px inward for -0.15). No-op for cell-sized tiles
  (margin>=0). Applied to targets and distractors; positions computed from the clamped paste.
- Verified: crop rate 13%/25% -> 0% at -0.15/-0.25; 22 omniSynth tests pass; glyphs now whole
  (results/experiments/omnisynth_crop_fixed.png vs omnisynth_crop.png).

- train.py (2d): added optional warm-start via `train.checkpoint` (already `null` in
  train_base.yaml). Tolerant load — keeps only name+shape-matching tensors, strips
  `_orig_mod.` prefix, accepts bare state_dict or `{"model": ...}` (mirrors pfn_seg.py).
  Enables retraining, e.g. `... train.checkpoint=.../best.pt`.

## 2026-07-06 — 3D eval: single multi-class loader
- `evaluate.evaluate_classes` now builds ONE dataset over all val classes
  (`common.make_eval_loader`) instead of one loader per class. The scan/bbox
  caches load once per eval instead of ~40×, and the "Loaded scan/bbox cache"
  spam is gone. Results grouped back per class via each sample's `label_name`;
  (rows, cases) shape unchanged, so eval.py + train.py val step are untouched.
- `make_loader(cfg, cls)` kept as a thin wrapper over `make_eval_loader([cls])`.
- Classes with no samples now yield an `{"class", "error": "no samples"}` row.

## 2026-07-06 — Medverse: train-from-scratch option
- Added `train.random_init` (default false). When true, `MedverseModel` builds the
  net from the checkpoint's `hyper_parameters` via `LightningModel(hparams)` and
  skips loading the pretrained `state_dict` -> fresh random weights, same arch.
- `base_ckpt: null` still means "stock pretrained Medverse.ckpt"; `random_init: true`
  is the way to ignore pretrained weights entirely. Wired in train.py build_model.

## 2026-07-06 — 3D train: build val loader once
- The multi-class val loader was rebuilt every eval epoch (dataset + scan/bbox
  caches reloaded each time). `evaluate_classes` now accepts an optional prebuilt
  `loader`; `validate_mean` threads it through; `main()` builds `val_loader` once
  via `make_eval_loader` before the epoch loop and reuses it. Eliminates the
  per-epoch "Loaded scan/bbox cache" reloads. eval.py (no loader arg) unchanged.

## 2026-07-07 — TODO: PatchSetCNN refinement-pass ideas (design only)
- Documented two orthogonal extensions in docs/TODO.md (no code yet):
  (1) patch-level `sampling` maps for context images — drop the sample-axis mask
  for full `r×r` attention (~free, breaks "context read-only"), decode all
  `(K+1)·N` rows via one shared 2-ch `(mask, sampling)` head, discard ctx masks;
  (2) high-res outputs via the Medverse "pool QK / hi-res V" trick — coarse `A`,
  per-patch `f×f` sub-cells folded into V channels, `A@V` + `pixel_shuffle(f)` →
  `R·f` map (only `AV` width + V mem scale by `f²`; scores unchanged; no V proj).
  They compose: hi-res read-out with `q_tok=s_tok` yields ctx maps at high res.
  Reference: Medverse MultiContextSpatialCrossAttention3D (/home/dpxuser/repos/Medverse).

## 2026-07-08 — omniSynth: random glyph placement option
- Added `scene.placement` (grid | random) to `OmniSceneConfig`. `grid` (default)
  keeps the existing cell-centred layout; `random` pastes each of the grid*grid
  glyphs at a uniform-random canvas centre (overlap allowed — union blending keeps
  every glyph). Only the centre choice changes in `render_scene`; k targets,
  samplers, masks, provenance, copy-task logic all unchanged.
- preview_omnisynth.py now prints `placement=`; draw with
  `scene.placement=random`. Samples: results/omnisynth_preview_{random,grid}.png.

## 2026-07-08 — omniSynth: max_nb_objects cap
- Added `scene.max_nb_objects` to `OmniSceneConfig` (0 = no cap). Caps total glyphs
  (targets + distractors) at `min(grid*grid, max_nb_objects)`; k is clamped to
  `[1, n_obj]`. `render_scene` fills a random subset of `n_obj` cells (first k of the
  permutation are targets) and skips the rest — so a cap yields a random subset of
  occupied grid cells (grid mode) or fewer scattered glyphs (random mode).
- preview_omnisynth.py prints `max_obj=`; e.g. `scene.max_nb_objects=4`. Samples:
  results/omnisynth_preview_{random,grid}_cap4.png.

## 2026-07-08 — omniSynth: surface placement + max_nb_objects in configs
- Added `scene.placement` and `scene.max_nb_objects` to the omniglot scene block
  (configs/experiment/2d/synth/omniglot.yaml), the single source both train.py and
  evaluate.py/eval_incontext.py pull via common.build_dataset (OmniSceneConfig(
  **dict(s.scene)) forwards every key — no whitelist). Set at CLI, e.g.
  `synth.scene.placement=random synth.scene.max_nb_objects=4`.

## 2026-07-08 — omniSynth: explore MedSegBench as an object source
- New script experiments/2d/synth/explore_medseg_objects.py: crops every connected
  component of each MedSegBench mask to its bbox and characterises it as a candidate
  omniSynth "object" (binary mask = ink/label + intensity patch = texture). Reports
  per-object bbox/area/fill/intensity/contrast (CSV + text + histograms) and a
  montage of extracted objects (img | masked | mask). Outputs: results/2d/medseg_objects/.
- Findings (train, 128, min_area=16, ≤120 imgs/ds, 20k objects across 35 ds):
  three object regimes — single big blob (busi/isic/kvasir/wbc/pandental,
  ~1 obj/img, bbox 0.24–0.98 of image), dense small instances (cellnuclei/monusac/
  nuset/tnbcnuclei/dynamicnuclear, 19–47 obj/img, bbox ~0.06–0.08), and thin vessel
  networks that shatter into tiny fragments (drive/chasedb1/bbbc010, low fill ~0.2).
  fill median 0.72 (blob-like); intensity+contrast span the full range (some ds have
  darker-than-bg objects, e.g. busi/isic contrast ≈ -60; dynamicnuclear img ≈ black).
- Implications for the bank: filter by min_area + maybe min_fill to drop vessel
  fragments; pick instance-rich datasets for many-object scenes; carry the intensity
  patch (not just binary) so pasted objects keep realistic texture/contrast.

## 2026-07-08 — omniSynth: MedSegBench object source
- New object bank src/datasets/omniSynth/bank_medseg.py (MedSegObjectBank), a drop-in
  alternative to OmniglotBank exposing the same task_ids/get/alphabet interface. A
  "class" = (dataset, label_value) [alphabet(cid)="<dataset>/label_<lv>"]; a rendition
  = one image's WHOLE binary mask for that label (all connected components kept, so
  multi-component objects stay intact), bbox-cropped + resized into a cell tile, stored
  as [2,tile,tile] float32: ch0 = intensity (0..1, zeroed outside mask), ch1 = binary
  mask. Classes split train/val/test by seeded permutation (novel-class, like Omniglot).
- render.py generalised to 2-channel renditions via _split(tile): glyph 2D bitmap ->
  img==mask; medseg [2,h,w] -> paste ch0 into image, ch1 into mask. affine_jitter is
  channel-aware (intensity kept continuous + re-masked, mask re-binarised). Glyph path
  and all 21 existing tests unchanged.
- Config: new OmniMedSegConfig + `medseg` block in synth/omniglot.yaml (source=omniglot
  default). Flip with synth.medseg.source=medseg (+ datasets/train_frac/... overrides).
  Wired through common.build_dataset, dataset.py (bank selector), preview_omnisynth.py.
- Tools: experiments/2d/synth/preview_medseg_bank.py (rendition montage) and
  explore_medseg_objects.py (object stats). Tests: test_bank_medseg.py + 2 render tests.
  Previews: results/omnisynth_medseg_{random,grid}.png, results/medseg_bank_preview.png.
- Usage: python experiments/2d/train.py synth=omniglot synth.medseg.source=medseg \
    synth.scene.placement=random synth.scene.max_nb_objects=6

## 2026-07-08 — omniSynth medseg: split by MedSegBench image splits
- Changed MedSegObjectBank splitting (superseding the earlier seeded novel-class split).
  The bank is now scoped to one split: the "train" bank reads each dataset's TRAIN
  images, the "val" bank its VAL images (no test set — val doubles as test). Config:
  replaced train_frac/val_test_split/datasets/master_seed with train_datasets and
  val_datasets ([] = all) selecting which datasets feed each split. Default (both [])
  = same classes in train and val, but drawn from different underlying images.
- get_or_build_medseg_bank / MedSegObjectBank take `split`; dataset.py passes its split.
  task_ids() returns the single scoped pool. Fixes the earlier empty-val edge case with
  few datasets. Updated omniglot.yaml medseg block, preview scripts, test_bank_medseg.py.

## 2026-07-08 — omniSynth medseg: canvas-relative object sizing
- Objects no longer uniformly resized to the cell. New OmniMedSegConfig.size_mode:
  - canvas (default): each object tile is scaled by canvas/source so it keeps its size
    relative to the full canvas (bbox that filled f of its source image -> ~f of the
    canvas), aspect ratio preserved, centred in a square tile. Per-placement aug then
    jitters around this. size_scale multiplies the preserved size.
  - cell: previous behaviour (every object resized to (1-2*margin)*cell, glyph-like).
- Wiring: MedSegObjectBank / get_or_build_medseg_bank take image_size (canvas side;
  defaults to source_size); dataset.py passes it; cache key includes it. No render
  changes — variable/oversized tiles were already handled by _clamp_center + union paste.
- Verified: tile px varies per object (busi med 30, isic 88, cellnuclei/drive ~full
  canvas since their whole-mask bbox spans the image) and rescales exactly with
  image_size (128->64 halves sizes). Previews: results/omnisynth_medseg_{canvas,cell}.png.

## 2026-07-08 — omniSynth: random background (dark objects stay visible)
- Problem: dark medseg object textures vanished on the black canvas. Fix mirrors
  controlSynth's non-black GMM background (bg_center~0.5 off the extremes).
- OmniSceneConfig.background: "black" (default, zero canvas, unchanged) | "random"
  (smooth low-freq grey field via upsampled random + gaussian noise; bg_intensity,
  bg_structure, bg_noise knobs). "black" draws no rng, preserving existing seeds.
- render.py: image now composited with premultiplied-alpha _composite (canvas*(1-mask)
  + img_tile) instead of np.maximum, so an object's true texture (bright OR dark)
  REPLACES the background under its mask rather than being hidden by a lighter bg.
  Label mask keeps its union paste. For a black canvas _composite == the old maximum,
  so the glyph path and all prior tests are unchanged (30 pass).
- Verified: busi target texture (0.005..0.46, incl. near-black) now visible over a
  0.41 grey bg. Preview: results/omnisynth_medseg_bg_random.png.

## 2026-07-08 — omniSynth: cheap anti-overlap for random placement
- Rejection sampling in render_scene: for random placement, each object is sampled
  first, then up to placement_tries candidate centres are drawn and the least-overlapping
  (vs a running boolean occupancy of already-placed masks) is kept — accepted early once
  overlap fraction <= placement_max_overlap. New OmniSceneConfig fields placement_tries
  (default 1 = fully random, no rejection; yaml sets 8) + placement_max_overlap (0.25).
  Helpers _tile_slices/_occupy/_overlap_frac/_place_random; O(objects*tries), tiny cost.
- Loop refactor: object sampled before its position (so its mask drives placement). Grid
  path unchanged incl. rng order (position uses no rng); random path rng order shifts by
  one (object sample now precedes the position draw) — deterministic, just different scenes.
- Verified on real medseg objects (5 datasets, size_scale 0.5, 6 obj/scene): overlapped-
  pixel fraction 0.101 (tries=1) -> 0.033 (4) -> 0.031 (8) -> 0.028 (16); ~3x less overlap
  at the tries=8 default. Previews: results/omnisynth_medseg_tries{1,8}.png. 31 tests pass.

## 2026-07-08 — omniSynth: BiomedParse object source
- Added source=biomedparse alongside medseg. New bank_biomedparse.py
  (BiomedParseObjectBank), same task_ids/get/alphabet interface + [2,tile,tile]
  rendition format. class = (dataset, target) with target parsed from the mask
  filename (_parse_mask_stem, '+'->space); alphabet(cid)="<ds_key>/<target>".
  Reads the pre-resized store <root>/<split>/<ds>/{images,masks}_{size}.npy +
  index_{size}.npz (reusing biomedparse._discover_stores/_index_from_npz); mask row
  -> image row via filename stem. split: train->train store, val/test->test store.
- Factored shared sizing/tiling into bank_common.py (make_object_tile + crop_to_tile);
  MedSegObjectBank refactored to use it (behaviour identical, its 4 tests still pass).
- Config: source now omniglot|medseg|biomedparse; added biomedparse_root. dataset.py
  branches on source; yaml documents it. common.build_dataset unchanged (OmniMedSegConfig
  carries the selector). 41 datasets / 109 (dataset,target) classes (test split).
- Verified: bank builds (train 6 / val 4 on a subset), scene renders
  (results/omnisynth_biomedparse_random.png), train config builds 36/35 classes.
  Tests: test_bank_biomedparse.py (3). Full suite 34 passed.
- Usage: python experiments/2d/train.py synth=omniglot synth.medseg.source=biomedparse \
    synth.scene.placement=random synth.scene.background=random

## 2026-07-08 — omniSynth: un-nest object source selector
- Moved the object-source selector from synth.medseg.source to a top-level
  synth.source (omniglot | medseg | biomedparse); the `medseg` block now holds only
  the real-object params (shared by medseg + biomedparse). Removed `source` from
  OmniMedSegConfig; added a `source` arg to OmniSynthICLDataset (selector uses it).
  common.build_dataset reads s.source; preview_omnisynth.py reads cfg.source. Updated
  omniglot.yaml + bank tests. All 34 tests pass; medseg/biomedparse/omniglot build
  end-to-end via synth.source.

## 2026-07-08 — eval: option to drop per-class metric keys
- Added eval.log_per_class (eval_incontext.yaml, default false there) to suppress the
  per-dataset (dice/dataset/*) and per-class (dice/class/*) breakdown keys in the wandb
  summary — one-per-letter/object-class entries that the per-sample table already covers.
  mean + macro aggregates and the console table are always kept.
- Threaded as per_group through validate() -> log_summary() (common.py); defaults True so
  train.py and other eval scripts are unchanged. eval_incontext.py passes
  cfg.eval.get('log_per_class', True). Verified: per_group=False yields only */mean + */macro
  (identical values), dropping all per-group keys.

## 2026-07-08 — eval: accurate inference timing (cuda sync + warmup)
- evaluate.py validate(): time/inference_ms now brackets the per-batch forward with
  torch.cuda.synchronize() (CUDA only), so it measures real GPU forward compute instead
  of async kernel-launch time. The first timed batch is dropped as warmup (cudnn.benchmark
  autotune / allocator / lazy init). No-op on CPU. FLOPs batch still excluded from timing.

## 2026-07-09 — omniSynth: real-image backgrounds (scene.background=image)
- New background mode: scene.background=image uses a random real full image from
  medseg/biomedparse as the canvas (objects composited over it, as before). Independent
  of the object source (e.g. biomedparse objects on medseg image backgrounds, or omniglot
  glyphs on real backgrounds). New scene fields: bg_source (medseg|biomedparse),
  bg_datasets ([]=all), bg_max_images (pool cap). Roots/source_size reused from the medseg
  block; split maps train->train, val/test->val (medseg)/test (biomedparse).
- bank_background.py (BackgroundBank + get_or_build_background_bank): pools full images
  per split (budget spread across datasets; medseg copies the kept slice, biomedparse
  keeps memmaps), samples one and resizes to the canvas on demand. Process-shared cache.
- render_scene/_make_background take an optional background_sampler; dataset.py builds the
  bank when background=image and threads sampler.sample. render stays PIL-free (sampler
  returns canvas-sized image). Objects still painted over via _composite (dark textures
  visible on any background).
- Tests: test_bank_background.py (2) + a render test; full suite 37 passed. Preview:
  results/omnisynth_bg_image.png. Usage: synth.scene.background=image
  synth.scene.bg_source=medseg [synth.scene.bg_datasets=[...]].

## 2026-07-09 — omniSynth: draw targets last (on top of distractors)
- render_scene now defers target image-compositing to after all distractors, so a
  distractor overlapping a target can no longer overwrite the target's texture in the
  image. Placement, occupancy (anti-overlap), rng order, the label mask and target
  positions/transforms are unchanged — only the target/distractor overlap pixels in the
  image differ (target wins). Targets keep their mutual paste order.
- Test: test_targets_drawn_over_distractors (oversized fully-overlapping tiles ->
  target texture 0.5 survives under 0.9 distractors). Render tests 13 passed.

## 2026-07-09 — 2D trainer: Muon + LAWA for patchset_cnn
- The `muon_*`/`lawa_k` keys in `train_base.yaml` were dead in the unified
  `experiments/2d/train.py` (only pfn_seg.py used them). Wired them in so patchset_cnn
  trains with the same recipe as pfn_seg.py: Muon (Newton-Schulz orthogonalized grads)
  on the transformer 2D weight matrices (`p.ndim==2 and "transformer" in n` → 8 tensors:
  qkv_col/qkv_row/mlp.0/mlp.2 per block), AdamW on everything else (encoder convs,
  embeds, pos, decoder, thinking, ctx-id), plus LAWA checkpoint averaging.
- Always-on for patchset_cnn, gated by `is_patchset = model_name=="patchset_cnn"`.
  universeg keeps its exact prior path (single AdamW, no LAWA) — its Muon group would be
  empty and both Muon and the LAWA queue are skipped, so its baseline is unchanged.
- Scheduler (cosine+warmup, per-batch) drives AdamW only; Muon LR is constant
  (`muon_lr_scale·lr`). `train_epoch` now takes an `optimizers` list (zero_grad/step loop).
- LAWA: deque(maxlen=lawa_k), push CPU state_dict each epoch; before `validate` average
  the queue into the model (eval + any best-checkpoint save use averaged weights), then
  restore raw weights so training continues from them. Epoch-1 (len≤1) averaging is a
  no-op via lawa_average's own guard.
- Verified on the real PatchSetCNN (CPU): param split 8 Muon / 31 AdamW, 3 train steps
  decrease loss, LAWA average changes weights and restore recovers them exactly.
- Comparison caveat: this changes the patchset_cnn recipe, so its current checkpoints are
  not a clean optimizer before/after vs universeg. Design doc:
  docs/superpowers/specs/2026-07-09-muon-lawa-patchset-cnn-design.md

## 2026-07-09 — omniSynth: fix empty masks for tiny objects
- Symptom: ~4/10 preview samples (medseg, canvas sizing, size_scale<1) had fully empty
  target+context masks — all from tiny/sparse classes (idrib microaneurysms, m2caiseg
  rare labels). min_mask_px filters the SOURCE mask, but the mask vanished downstream.
- Two vanishing points, both bilinear-resize + 0.5 threshold on sparse masks:
  1. bank_common.make_object_tile: canvas downsizing blurred tiny masks below 0.5.
     Fix: if the >=0.5 mask is empty, fall back to >0 (keep any coverage); crop_to_tile
     also drops any rendition whose tile mask is still empty (excludes hopeless classes).
  2. render.affine_jitter (target_mode=aug): the warp re-thresholded at 0.5 and could
     blur/push a tiny mask off-tile. Fix: >0 fallback, then if still empty return the
     un-jittered base (bank guarantees it is non-empty).
- Result: empty target masks 4/10 -> 0/200. Full suite 38 passed.
