# Cascade training for PatchSet3D (v2 pipeline)

Date: 2026-08-30
Status: design approved, pending implementation plan

## Motivation

Today `experiments/3d/train.py` trains PatchSet3D at a single physical crop pitch
(`data.crop_spacing_mm`): each item is an organ-centred `T³` crop at that spacing, the
model predicts once, one loss, one backward. Coarse→fine refinement exists only at eval
time (`eval.spacing_cascade` in `experiments/3d/evaluate.py`) and only on the **v1**
dataloader — it is a scoring construct (stitched native Dice over two loader sweeps),
not a training signal.

We want the model to learn the coarse→fine behaviour it is scored on: predict at a
coarse spacing, then predict again at a finer spacing on a crop re-centred on the coarse
prediction, with an independent loss at each level. This is expected to help the
size-stalled small-organ regime (adrenals etc. are sub-cell at 3 mm — see
`docs/logs.md` 2026-08-26 and the exp57 config notes).

## Scope

- **In**: N-level cascade forward + per-level loss in the **v2** training path
  (`src/incontext_dataset_v2.py` + `src/providers/totalseg.py` + PatchSet3D), geometric
  augmentation replayed consistently across levels (flip / affine / elastic / deform),
  matching cascade metric in the v2 training-time val loop, a new experiment config.
- **Out**: throughput optimisation of the in-loop re-crop (a `ThreadPoolExecutor` or a
  GPU-side re-crop) — deferred; run first, optimise later. v1 dataloader. Medverse.
  Changing `experiments/3d/eval.py`'s standalone `spacing_cascade` path. Differentiating
  through the crop centre (it is an integer index op — inherently detached).

## Decisions (from brainstorming)

| Question | Decision |
| --- | --- |
| How is the level-i target crop produced? | Re-cropped **in the train loop** by calling `provider.load(..., center=predicted_COM, spacing=s_i)` directly — exact eval semantics. |
| Geometric aug consistency across levels | One set of geometric params per task, **replayed** at every level; **full** flip/affine/elastic/deform support. The level-(i-1) predicted centre-of-mass is mapped back through the composed sampling grid to native voxels. |
| Number of levels | General **N-level** loop (`data.cascade_spacings: [s0, s1, ...]`); ship + test with 2. |
| Val loop | Also runs the cascade; reports per-level + stitched native Dice; checkpoint-selection metric becomes the stitched native Dice. |
| Loss aggregation | `Σ wᵢ · loss_i`, weights configurable (`train.cascade_loss_weights`), single backward. |
| Code structure | New `experiments/3d/cascade.py` module, shared by the train loop and the val loop. PatchSet3D.forward stays single-level. |
| Config | **New** experiment yaml (`59_*`), exp57 left intact. |
| Throughput | Accept the synchronous in-loop I/O for now; optimise after a first run. |

## Current pipeline (as inspected 2026-08-30)

Path for `experiment=57_organs_encoder_from_scratch` → `dataset=d1` → `data.loader_v2=true`:

1. `common.train_loader(cfg)` → `common.build_dataset` builds `InContextDataset`
   (`src/incontext_dataset_v2.py`) over a `TotalSegProvider` (`src/providers/totalseg.py`).
   `crop_spacing_mm` = `data.crop_spacing_mm` (3 for exp57). `data.train_spacing_range` is
   `null` → no `SpacingBatchSampler`, every crop at 3 mm.
2. `InContextDataset.__getitem__(idx)`: per-item `rng` (global `random` in training),
   samples class + subject (class-balanced), calls
   `provider.load(subj, cls, LoadRequest(rng, crop_spacing))` for the target + K contexts.
   `LoadRequest.center` **already exists** ("cascade fine-crop seam") but v2 always passes
   `None` → provider uses the GT-bbox centroid (`self._bbox`). Returns
   `image (1,T,T,T)`, `label (T,T,T)`, `context_in (K,1,T,T,T)`, `context_out (K,T,T,T)`,
   `subject`, `context_subjects`, `label_name`, `spacing (3,)`, `crop_geom (4,3)`,
   `aug_mode (=0)`.
3. `TotalSegProvider.load` → `crop_and_place` / `crop_and_place_cached`:
   `organ_crop_arrays(center, crop_mm=crop_spacing_mm, jitter=self.crop_jitter, rng)` cuts a
   crop of physical extent `T·crop_spacing_mm` about `center`, resamples to `T³`,
   centre-pads thin axes. `crop_geom = [starts, crop_sizes, out_sizes, pad_lo]` (native
   voxel units). The fast path reads a pre-resampled `ct_raw_{crop_spacing:g}mm.npy` image
   cache when present (pitch must equal `crop_spacing`), else full-res `ct_raw.npy`.
4. Augmentation: exp57 inherits `augmentations=calibrated` → `augmentations.gpu=true`, so
   `InContextDataset._aug_active()` is `False` (`defer_aug=True`) → CPU aug skipped. The
   batch reaches `train_epoch` un-augmented; `GpuAugmentor` (`src/gpu_augment.py`) runs in
   the loop: `_stack_task` → `_geometric(group_size=T)` (shared flips + one affine `theta`
   per task + optional elastic + optional deform, all drawn inline from a per-step
   `torch.Generator(seed+step)`) → `_batched_intensity` (per-volume) → `_unstack_task`.
   Flips are applied as `tensor.flip()` **before** the affine/elastic/deform sampling grid
   is built; the grid is a single `F.affine_grid` + additive displacement fields, consumed
   by one `F.grid_sample`.
5. `train_epoch`: one `_autocast()` forward
   `model(image, context_in, context_out, spacing=<batch[0] scalar or None>)` →
   `final_logit (B,1,G,G,G)`, `G = resolution · mask_patch_decode_size`.
   `target = target_like(label.unsqueeze(1), logits)` pools GT to the grid. One `loss_fn`,
   one `backward`.
6. `validate_mean` → `evaluate_classes` (`experiments/3d/evaluate.py`) over a prebuilt
   `val_loader`, using `model.train_forward` (native-res logits) for soft Dice / loss and
   `model.predict` (or reused logits) for hard Dice; produces per-class rows, a per-sample
   `cases` table, seen/unseen macro split. `val/dice` = plain mean over val classes drives
   best-checkpoint saving.

Reusable pieces already in `experiments/3d/evaluate.py`:
- `_predicted_native_center(prob, geom)` — grid-space soft-prob centroid → native voxel
  centre via `native[a] = starts[a] + (g[a] - pad_lo[a]) / max(1,out_sizes[a]) * crop_sizes[a]`;
  returns `"volume_center"` when `prob.sum() < 1e-6`.
- `_stitched_native_dice(base_pg, over_pg, root)` — composite coarse preds into the native
  volume, overwrite with finer preds, Dice vs `label.npy == class_idx`.
- `_refit_into_coarse(pred_fine, geom_c, geom_f)` — exact fine→coarse grid remap through
  both padded crop geometries.

## Design

### 1. Config surface

```yaml
data:
  cascade_spacings: [3, 1.5]        # null (default) = cascade OFF. List of >=2 crop pitches
                                    # (mm), coarsest first. Level 0 = GT-centred target +
                                    # contexts. Level i>0 = target re-cropped on level i-1's
                                    # predicted COM; contexts stay GT-centred.
  cascade_crop_jitter: 0            # crop jitter (native voxels) for re-cropped levels >=1.
                                    # 0 = respect the predicted COM exactly (matches eval).
train:
  cascade_loss_weights: [1.0, 1.0]  # per-level; len must equal len(data.cascade_spacings).
```

Validation (fail fast in `main` / `common`, with actionable messages):
- `cascade_spacings` set ⇒ `model == patchset3d`, `data.loader_v2 == true`,
  `data.source in _TOTALSEG_SOURCES`.
- `data.train_spacing_range` must be `null` when `cascade_spacings` is set (both own the
  per-batch physical spacing).
- `data.crop_spacing_mm == cascade_spacings[0]` (level-0 geometry: `image_size`,
  `target_like`, the provider crop and the model all agree on the coarse pitch).
- `len(cascade_spacings) >= 2` and strictly decreasing (coarse→fine); warn (not error) if
  not strictly decreasing.
- `len(cascade_loss_weights) == len(cascade_spacings)` (default: all `1.0` when the key is
  absent).

### 2. Dataloader — one small addition

No new dataset class and no change to how the **level-0** loader is built: it is the
existing v2 `InContextDataset` at `cascade_spacings[0]`, `defer_aug=True`,
`class_balanced` as configured. `InContextDataset.__getitem__` already returns `subject`,
`context_subjects`, `label_name`, `crop_geom`, `spacing`, `aug_mode` for every item
(including class-balanced ones) — that is the complete task descriptor the runner needs to
re-crop levels ≥1.

Change:
- `src/incontext_dataset_v2.py`: `LoadRequest` gains `jitter: Optional[int] = None`.
  `None` = provider default (`self.crop_jitter`); an int overrides it. (v2's non-cascade
  path never sets it → unchanged.)
- `src/providers/totalseg.py`: `load()` threads `req.jitter` (when not `None`) into both
  `crop_and_place` / `crop_and_place_cached` as the `jitter` argument. Everything else
  unchanged; `req.center` is already honoured.

### 3. `experiments/3d/cascade.py` (new)

Sibling of `common.py` / `evaluate.py` (same `sys.path` insert in `train.py`). Owns the
N-level orchestration; imports `_predicted_native_center`, `_stitched_native_dice`,
`_refit_into_coarse` from `evaluate`.

```python
@dataclass
class CascadeResult:
    logits:  list[Tensor]   # per level, (B,1,G,G,G)
    targets: list[Tensor]   # per level, grid GT (target_like)
    geoms:   list[Tensor]   # per level, (B,4,3) target crop_geom
    hard_preds: list[Tensor]  # per level, (B,D,H,W) native-res binary (val only; None in train)
    centers: list[list]     # per level>=1, native COM per b (tuple or None==GT-centroid fallback)
    empty_frac: float       # fraction of (level, b) COM inversions that hit the empty-prob fallback

def run_cascade(model, provider, batch, augmentor, spacings, *, device, training,
                step, seed, loss_fn=None, is_prob=False, want_hard_preds=False)
    -> CascadeResult
```

**Level 0**

1. Move `batch` image/label/context tensors to `device`.
2. Assert `batch["aug_mode"]` all `0` (v2 REAL tasks only — no synth/self-context group).
3. If `augmentor` is not `None` (training):
   `aug0, geo_state0 = augmentor.apply(batch, geo_gen=Gen(seed+step),
   int_gen=Gen(seed+step+INT_OFFSET), capture=True)` where
   `geo_state0 = GeoState(grid=(B*T,T,T,T,3) or None, flips=(B*T,3) bool)`.
   `grid` is the composed affine+elastic+deform sampling grid built inside `_geometric`
   right before its `grid_sample` (normalised coords, `grid_sample` xyz axis order);
   `flips` records the per-volume axis flips applied before it.
   If `augmentor is None` (val): `aug0 = batch`, `geo_state0 = GeoState(None, zeros)`.
4. `logit0 = model(aug0.image, aug0.context_in, aug0.context_out, spacing=spacings[0]).float()`
   (`spacing` scalar only forwarded when `model.spacing_aware`).
5. `target0 = target_like(aug0.label.unsqueeze(1), logit0)`.
6. `prob0 = sigmoid(logit0)` (or `clamp(0,1)` when `is_prob`), upsampled/pooled to the
   target-crop grid `T³` for the centroid (grid-res centroid then scaled by `T/G` is
   equivalent and cheaper — use that).

**COM inversion — `invert_geo_center(g_aug, geo_state_row, crop_geom_row, T) -> native tuple | None`**

- `g_aug` = prob-weighted centroid `(d,h,w)` in the level-0 **augmented** target grid.
- If `prob.sum() < 1e-6` → return `None` (level-1 target falls back to the provider's GT
  centroid; counted in `empty_frac`).
- Undo flips: for each axis `a` with `flips[a]` true, `g = (T-1) - g`.
- If `grid` is not `None`: trilinearly sample `grid[target_row]` at voxel `g`
  (convert `g` → normalised, sample, read back the xyz→dhw coord, convert normalised →
  voxel) → `g_pre`, the coord in the **pre-aug** level-0 crop grid. Else `g_pre = g`.
- Native: `native[a] = round(starts[a] + (g_pre[a] - pad_lo[a]) / max(1,out_sizes[a]) *
  crop_sizes[a])` from `crop_geom_row` (identical formula to `_predicted_native_center`);
  clamp `>= 0`.
- With `grid is None` and `flips` all false (the val / no-aug case) this is byte-identical
  to calling `_predicted_native_center(prob, crop_geom_row)`.

**Level i (1 ≤ i < N)**

7. For each `b` in the batch, on the **main process**:
   - target: `provider.load(subjects[b], label_names[b],
     LoadRequest(rng=random.Random((seed,step,i,b)), crop_spacing_mm=spacings[i],
     center=centers[i][b], jitter=cascade_crop_jitter))`
     (`center=None` when the inversion returned `None`).
   - K contexts: `provider.load(context_subjects[b][k], label_names[b],
     LoadRequest(rng=random.Random((seed,step,i,b,k)), crop_spacing_mm=spacings[i],
     center=None, jitter=cascade_crop_jitter))`.
   - collate the B items with `incontext_collate_fn` → `batch_i`.
8. Move `batch_i` to `device`. Augment:
   `augi, geo_state_i = augmentor.apply(batch_i, geo_gen=Gen(seed+step),
   int_gen=Gen(seed+step+INT_OFFSET+i), capture=(i < N-1))`.
   **Same `geo_gen` seed as level 0** ⇒ identical flip / affine / elastic / deform draws:
   every level's stack is `(B*T, 1, T, T, T)` so `_geometric`'s RNG consumption is
   position-for-position identical. Intensity uses a per-level seed (independent appearance
   per level is fine — only geometry must match). Val: `augi = batch_i`, identity geo_state.
9. `logit_i`, `target_i = target_like(...)` as in level 0.
10. If `i < N-1`: `centers[i+1] = [invert_geo_center(centroid(prob_i[b]),
    geo_state_i.row(b_target), crop_geom_i[b], T) for b]`.

**Hard preds (val only, `want_hard_preds=True`)**: per level, upsample `logit_i` to the
crop's native size (`crop_sizes` from `crop_geom_i`, via the provider's out/pad geometry)
and threshold — the array `_stitched_native_dice` expects, paired with `crop_geom_i`.

Return `CascadeResult`.

### 4. `GpuAugmentor` refactor (`src/gpu_augment.py`)

Goal: let `run_cascade` (a) inject its own generators, (b) get `(grid, flips)` back,
(c) replay identical geometry — without changing the existing single-call path.

- `_geometric(vols, masks, group_size, cfg, gen, *, capture=False)`:
  - record `flips` per group/axis as it applies them (currently just mutates the tensor);
    expand to `(N,3)` bool aligned with `vols` rows.
  - after composing `grid` (affine + elastic + deform) and **before** `grid_sample`, stash
    it when `capture=True`.
  - return `(vols, masks, GeoState(grid, flips))` when `capture`, else `(vols, masks)` as
    today.
- New `GpuAugmentor.apply(batch, *, geo_gen, int_gen, capture=False) -> (batch, GeoState|None)`:
  the REAL-mode body of today's `__call__` (v2 has only REAL tasks) with the two injected
  generators instead of the internal `self._step` one.
- `GpuAugmentor.__call__` keeps its current signature and behaviour (its own generator,
  all three aug modes, no capture) — non-cascade runs are byte-identical. It may delegate
  its REAL branch to `apply` internally, but that is an optional tidy, not required.
- `INT_OFFSET` is a large fixed constant so `seed+step` (geo) and `seed+step+INT_OFFSET+i`
  (intensity) never collide across plausible step counts.

### 5. `train_epoch` integration (`experiments/3d/train.py`)

When `cfg.data.get("cascade_spacings")`:
- build a `GpuAugmentor` regardless of `augmentations.gpu` (cascade needs the capture/replay
  API; exp59 sets `augmentations.gpu=true` anyway).
- per step: `res = run_cascade(model, loader.dataset.provider, batch, augmentor, spacings,
  device=DEVICE, training=True, step=(epoch * steps_per_epoch + n), seed=cfg.train.seed,
  loss_fn=loss_fn, is_prob=is_prob)` — `step` is any per-step monotonic int; it seeds the
  aug generators so geometry is reproducible-per-step and replayed across levels.
- `loss = sum(w_i * loss_fn(res.logits[i], res.targets[i]) for i in range(N))`;
  `loss.backward()`; optimiser + scheduler steps unchanged.
- non-finite guard runs on the concatenation of all levels' logits.
- logging: `train/loss_l{i}`, `train/dice_l{i}` (`_hard_dice` per level),
  `train/cascade_empty_frac`. `train/loss` = summed loss; `train/dice` = finest level's
  hard Dice (keeps the existing key meaningful). Grid metrics (`hard_sum`/`soft_sum`/
  `cos_sum`) computed on the finest level only.
- `profile_timing`: add a `recrop` bucket (perf_counter around the per-level provider
  loads) so the synchronous I/O cost is visible from day one.

Non-cascade path is untouched (the whole block is behind the `cascade_spacings` check).

### 6. Val integration

`validate_mean` branches when `cfg.data.get("cascade_spacings")` and patchset3d:
`evaluate_cascade(model, cfg, val_classes, loader=val_loader, ...)` in `cascade.py`.

- iterate the level-0 `val_loader` (built at `cascade_spacings[0]`, `eval_seed` set,
  `crop_jitter=eval.crop_jitter` — unchanged `make_eval_loader` call).
- per batch: `run_cascade(..., training=False, augmentor=None, want_hard_preds=True)`
  (no aug ⇒ COM inversion == `_predicted_native_center`).
- per class, aggregate:
  - per-level native-res hard Dice → `val/dice_l{i}/<class>` and macro `val/dice_l{i}`.
  - **stitched native Dice**: feed the per-(subj,cls) `(hard_pred, crop_geom)` of level 0
    as `base_pg` and of the finest level as `over_pg` to `_stitched_native_dice` (extend to
    fold in intermediate levels: apply them in coarse→fine order, each overwriting the
    previous) → `val/dice_cascade/<class>` and macro `val/dice_cascade`.
- `val/dice` (drives best-checkpoint saving) = macro `val/dice_cascade` when cascade is on.
- seen/unseen macro split + `build_sample_table` derive from the stitched per-case Dice
  (one `case` per (subj,cls) with `dice` = stitched value, plus `dice_l{i}` columns).
- soft-Dice / val-loss reporting: computed at the finest level (comparable scale note like
  the existing one in `validate_mean`).

`evaluate.py`'s standalone `spacing_cascade` path is not touched.

### 7. New experiment config

`configs/experiment/3d/experiment/59_organs_cascade_from_scratch.yaml`:

```yaml
# @package _global_
# 59_organs_cascade_from_scratch — exp57 + coarse->fine cascade training.
#
#   python experiments/3d/train.py experiment=59_organs_cascade_from_scratch
#
defaults:
  - 57_organs_encoder_from_scratch
  - _self_

data:
  crop_spacing_mm: 3            # == cascade_spacings[0]
  cascade_spacings: [3, 1.5]    # 2 levels: 3 mm coarse -> 1.5 mm fine (re-centred on the coarse COM)
  cascade_crop_jitter: 0

train:
  cascade_loss_weights: [1.0, 1.0]

wandb:
  name: 59_organs_cascade_from_scratch
```

exp57 stays exactly as it is.

### 8. Files touched

| File | Change |
| --- | --- |
| `experiments/3d/cascade.py` | **new** — `run_cascade`, `invert_geo_center`, `GeoState`, `CascadeResult`, `evaluate_cascade` |
| `src/incontext_dataset_v2.py` | `LoadRequest.jitter: Optional[int] = None` |
| `src/providers/totalseg.py` | `load()` threads `req.jitter` into `crop_and_place[_cached]` |
| `src/gpu_augment.py` | `_geometric` capture of `(grid, flips)` + injected generator; new `GpuAugmentor.apply`; `__call__` unchanged |
| `experiments/3d/train.py` | `train_epoch` + `validate_mean` branch on `data.cascade_spacings`; config asserts in `main`; `recrop` profile bucket |
| `experiments/3d/common.py` | assert `cascade_spacings` xor `train_spacing_range`; `crop_spacing_mm == cascade_spacings[0]` |
| `configs/experiment/3d/experiment/59_organs_cascade_from_scratch.yaml` | **new** |
| `docs/logs.md` | log the change |

## Risks & mitigations

1. **Throughput** — each level ≥1 does `B·(K+1)` synchronous `provider.load` calls on the
   GPU thread (16/step at B=4, K=3). Mitigations deferred by decision; the `recrop` profile
   bucket quantifies it on the first run. **Pre-requisite**: a `ct_raw_{s:g}mm.npy` image
   cache must exist for every `cascade_spacings` value (e.g. `ct_raw_1.5mm.npy`) or the
   provider falls back to full-res `ct_raw.npy` per load (much slower). Add the cache-build
   step to the run checklist.
2. **Deform inversion accuracy** — `invert_geo_center` point-samples the composed grid: a
   local linearisation of a nonlinear warp. Sufficient for choosing a crop centre (±1
   voxel at the coarse pitch is immaterial); never used for a pixel-exact mapping.
3. **Compiled model called N× per step** — identical shapes across levels, no recompile.
4. **RNG-replay assumption** — identical geometry across levels holds only while every
   level's augmentation stack is exactly `(B·T, 1, T, T, T)`. It is (all levels share `T`,
   `B`, `K`). `run_cascade` asserts the shape before the level-i `apply` call so a future
   change that breaks it fails loudly instead of silently desyncing the warps.
5. **`class_balanced` re-sampling** — the level-i re-crop must reuse the level-0 task's
   `subject` / `context_subjects` / `label_name`; it reads them from the batch dict, never
   re-samples. Asserted by construction (no rng class/subject draw in `run_cascade`).

## Open questions for spec review

- Intermediate-level handling in the stitched native Dice: apply every level coarse→fine
  (each overwrites the previous), or only level 0 + finest? Design assumes all levels,
  coarse→fine.
- `train/dice` and grid metrics reported at the finest level only — acceptable, or log all
  levels? Design keeps finest for the headline key, `*_l{i}` for the rest.
