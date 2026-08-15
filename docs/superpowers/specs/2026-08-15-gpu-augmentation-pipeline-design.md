# GPU Augmentation Pipeline — Design

**Date:** 2026-08-15
**Status:** Approved (design), pending implementation plan
**Author:** Tidiane Camaret (with Claude)

## Summary

Move the 3D in-context augmentation pipeline off CPU DataLoader workers and into
a single batched GPU stage that runs in the training loop between `batch.to(device)`
and `model(...)`. The current pipeline (`src/augmentations.py`, called per-item in
`src/totalseg_dataloader_incontext.py.__getitem__`) is CPU-bound: the benchmark
(`experiments/3d/bench_cpu_aug.py`) measured per-item aug at 439 ms (no GIN) rising
to 1623 ms with IPA at 128³, which caps throughput near ~1.25 steps/s @ B=8 — below
the model's ~3.5 steps/s. Since the ops are pure torch (`grid_sample`, `conv3d`,
`interpolate`), they batch cheaply on GPU.

Exact reproduction of the current pipeline is a **non-goal** — this is a redesign.

## Goals / Non-goals

**Goals**
- One unified GPU augmentation module covering all train-time aug: shared geometric
  task aug, per-volume intensity, GIN/IPA, synth heavy aug, self-context jitter.
- Own batched torch ops (no new deps; no MONAI/Kornia). Preserve the in-context
  semantics no library models: one geometric transform *shared* across the K+1
  volumes of a task, independent per-volume intensity.
- Large throughput win vs the CPU path; free workers to do I/O + crop + resample only.
- Behind a config flag so the CPU path remains a fallback during rollout.

**Non-goals**
- Bit-exact reproduction of current augmentation output.
- Differentiability / gradient flow through augs (run under `no_grad`, in-place OK).
- Moving *label synthesis* (ellipsoid / supervoxel generation) to GPU — it is
  I/O + indexing bound and stays on CPU workers.
- Changing the model, loss, or eval path.

## Key insight: augmentation is entangled with task construction

Augmentation today does more than transform pixels — it *builds task structure*:

- **Synth** (`__getitem__` ~1082–1089): `apply_synth_aug` creates the K+1 divergent
  copies from ONE supervoxel volume via heavy independent aug.
- **Self-context** (~1214–1265): clones the *post-aug* target into K contexts, then
  re-augments each clone (`apply_per_image_aug` + `apply_intensity_aug`).
- **Real** (~1202–1212): shared `apply_task_aug` across K+1, then per-volume
  `apply_intensity_aug`.

The redesign **separates task construction from augmentation**:

- **CPU workers keep**: subject/class sampling, context selection, crop, resample,
  I/O, label synthesis (ellipsoid/supervoxel), and *raw replication* (for synth /
  self-context the "contexts" become raw clones of the source).
- **GPU engine takes**: all geometric + intensity + GIN/IPA + synth-heavy +
  self-context jitter transforms.

## Architecture

```
DataLoader worker (CPU)                     Training loop (GPU)
─────────────────────                       ───────────────────
sample subj/class, crop, resample,          batch = {k: v.to(device)}
load real contexts / replicate clones  ─►   GpuAugmentor(batch, training=True)
synth label synthesis (I/O)                   ├─ task geometric (shared per task)
emit RAW volumes + per-item aug_mode          ├─ per-volume intensity + GIN/IPA
                                              ├─ synth heavy aug (mode=synth)
                                              └─ self-context jitter (mode=self_ctx)
                                            model(image, context_in, context_out, …)
```

Collate is **unchanged** (uniform target + K contexts); a per-item `aug_mode` field
is added and stacked.

## Component design

### 1. Dataset changes (`src/totalseg_dataloader_incontext.py`)

New constructor param `defer_aug_to_gpu: bool = False`. When true, `__getitem__`
(both the synth path `_get_synth_item` and the main path):

- **Skips** `apply_task_aug`, `apply_intensity_aug`, `apply_synth_aug`,
  `apply_per_image_aug`.
- Still performs subject/context sampling, crop, resample, **label synthesis**
  (ellipsoid/supervoxel) and **raw replication** for synth/self-context (the K
  contexts become raw `.clone()`s of the source instead of augmented ones).
- Adds `item["aug_mode"]`: an int code in `{0:real, 1:synth, 2:self_context}`.
  For self-context-synth, the synthetic label is already placed on the target grid
  by the existing CPU code (unchanged) — only the *aug* of the clones defers.
- `synth_coord` / `synth_radii` logging is unchanged (computed at construction).

`incontext_collate_fn` stacks `aug_mode` into `(B,)` int tensor when present.

### 2. `GpuAugmentor` (`src/gpu_augment.py`)

Plain callable holding `aug_cfg` (the existing `augmentations` DictConfig).

```
class GpuAugmentor:
    def __init__(self, aug_cfg): ...
    def __call__(self, batch: dict, training: bool) -> dict:
        if not training or not aug_cfg.enabled:
            return batch                      # eval / disabled → identity
        # under torch.no_grad():
        #   stack image + context_in -> (B*(K+1), 1, D, H, W), group index = task_id
        #   stack label + context_out -> (B*(K+1), D, H, W)
        #   dispatch per aug_mode (masked sub-batches)
        #   scatter back into batch tensors
        return batch
```

Shapes in the batch: `image (B,1,D,H,W)`, `label (B,D,H,W)`,
`context_in (B,K,1,D,H,W)`, `context_out (B,K,D,H,W)`, `aug_mode (B,)`.
The augmentor forms a `(B, K+1, …)` view so volume `0` of each group is the target.

### 3. Batched primitives (own torch, non-differentiable)

- **Geometric, shared per task**: sample `B` transforms (flip / affine / elastic).
  Build one θ per task, expand across the K+1 group → a single
  `F.affine_grid` + `F.grid_sample` over `(B*(K+1), …)`; images bilinear
  (`padding_mode=border`), masks nearest (`padding_mode=zeros`), masks cast to
  float then back to long. Elastic = coarse random displacement field per task,
  upsampled, added to the base grid (same construction as `apply_task_aug`).
  This is the K+1-sharing done batched.

- **Intensity, per volume**: brightness / contrast / gamma / gaussian-noise are
  elementwise → fully vectorized with per-volume params of shape
  `(B*(K+1),1,1,1,1)` broadcast over the stack. Per-op probability handled by a
  per-volume Bernoulli mask (apply everywhere, then `where(mask, aug, orig)`), or
  by selecting the active subset.

- **Blur / GIN / IPA**: `F.conv3d` with `groups = B*(K+1)` and per-volume random
  kernels stacked into `(groups*out, 1, k,k,k)` — batched grouped conv, more
  efficient than the current per-item Python loop. GIN keeps its frob-norm renorm
  and clamp to `[CT_NORM_MIN, CT_NORM_MAX]`; IPA blends `ipa_copies` grouped-conv
  GIN outputs with a coarse-field mask (batched form of `_ipa_blend_3d`).

- **Simulate-low-resolution**: batched `interpolate` down/up per active subset.

### 4. Mode dispatch

Modes partition the batch (typically uniform per run — e.g. exp-42 has every train
item as self-context-synth). Process each present mode on its masked sub-batch:

- `real`: shared geometric (per task) → per-volume intensity (all K+1).
- `synth`: independent heavy per-volume geometric + intensity (the divergence that
  `apply_synth_aug` produced across the K+1 clones); uses `aug_cfg.synth`.
- `self_context`: shared geometric on all K+1 → per-image jitter (`aug_cfg.per_image`)
  on the K context clones → intensity (`aug_cfg.intensity`) on all, gated by the
  existing `self_context_{per_image,intensity}` toggles (plumbed as engine flags).

### 5. Config

- New `augmentations.gpu: false` (default). When true:
  - `build_dataset` passes `defer_aug_to_gpu=True` for the train dataset.
  - `train.py` constructs a `GpuAugmentor(cfg.augmentations)` and calls it after
    moving the batch to device, before `model(...)`.
- All existing `augmentations.{task,intensity,synth,per_image}` and
  `intensity.gin` sections are consumed verbatim as engine params — no schema churn.
- Eval already builds datasets with `aug_cfg=None`; the augmentor is train-only.
- Self-context toggles (`data.self_context.augs.{per_image,intensity}`) are forwarded
  to the augmentor.

### 6. Training-loop integration (`experiments/3d/train.py`)

Refactor `train_epoch` so the batch dict is moved to device once (helper
`_to_device(batch)`), then `batch = augmentor(batch, training=True)` before the
forward. The current inline `.to(DEVICE)` calls at the model call site are replaced
by the single move. Eval (`evaluate.py`) is untouched (no augmentor).

## Testing & rollout

Behind `augmentations.gpu` (default false) so the CPU path is the fallback.

- **Distributional equivalence**: on a fixed batch, GPU aug output statistics
  (mean/std/histogram of image; mask voxel counts) match the CPU pipeline within
  tolerance — *not* bit-exact (per non-goal).
- **Mask integrity**: labels stay integer; geometric moves image + mask together
  (a planted mask blob lands in the same place in image and mask after affine).
- **K+1 sharing**: all volumes in a task receive the *same* geometric transform
  (target/context correspondence preserved for `real`/`self_context`).
- **Mode dispatch**: a mixed-mode batch routes each item to the right regime.
- **Eval identity**: `training=False` returns the batch unchanged.
- **Throughput microbench**: GPU aug ms vs the CPU numbers in
  `docs/logs.md` (2026-08-15 GIN/IPA CPU cost) at B=8, 128³.

## Risks & mitigations

- **Mixed-mode batches** reduce batching efficiency → process per mode; B is small
  (≤8), so ≤3 sub-batches. Acceptable.
- **GPU memory**: `B*(K+1)` volumes of 128³ + transient grids/kernels. At B=8,K=3
  that's 32 volumes ≈ 268 MB fp32 + grids; fits alongside the model. Monitor;
  process geometric grid in the group-batched form (one grid, reused for image+mask).
- **Randomness**: use a `torch.Generator(device=cuda)` seeded per epoch/step for
  loose determinism. Eval is aug-free so the eval-reproducibility fix
  (per-item `eval_seed`) is unaffected.
- **CPU worker under-utilization**: workers now only do I/O/crop/resample; may need
  fewer workers. Tune `num_workers` after profiling.

## Future work (not in v1)

- **Approach B (minimal-source + GPU replication)**: dataset emits 1 source volume
  for synth/self-context (instead of K+1 raw clones) and the augmentor replicates on
  GPU, cutting host→device transfer. Needs a ragged/padded collate. Strict transfer
  optimization over this design with the same engine — add if profiling shows the
  pipeline is transfer-bound.
- Optional differentiable mode for GIN-IPA-style consistency regularization.
```
