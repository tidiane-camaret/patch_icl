# Change log

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
