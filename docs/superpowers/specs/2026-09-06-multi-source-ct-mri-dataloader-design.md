# Multi-source (CT + MRI) in-context dataloader — design

Date: 2026-09-06
Status: approved (brainstorming), pre-plan

## Goal

Train and eval one in-context segmentation model over **multiple sources
simultaneously**, starting with `totalseg` (CT) and `totalsegmri` (MRI). Per
task: sample a class, then sample the K+1 cases under a **modality regime** drawn
per task:

- `ct`    (~1/3): target + all K contexts are CT
- `mri`   (~1/3): target + all K contexts are MRI
- `cross` (~1/3): forced cross-modality — target one modality, all K contexts the
  other

The point is to measure whether in-context conditioning transfers **across
modalities** (CT context → MRI target and vice versa), on top of the existing
single-source baselines.

This slots onto the `experiment=80_varspacing_hard_tgt_prior` lineage:
single-forward PatchSet3D, `[1.5, 6] mm` log-uniform crop pitch,
`encoder_input_norm=instance` (already wired in exp70/80 precisely "for the
planned CT+MRI joint run").

## Decisions (locked in brainstorming)

| Question | Decision |
|---|---|
| Class set | **Union** of both sources' *train* class specs (CT `balanced` ∪ MRI `train` ≈ 70). A class with no subjects in one modality falls back to sampling the other modality for the regimes that need it. |
| `cross` regime composition | **Forced cross-modality**: with `context_size=1`, target modality X, context modality ¬X (fallback to X only if ¬X has no subjects for the class). |
| Eval structure | **Stochastic mix, deterministic**: same 1/3-1/3-1/3 regime draw, seeded per item index, fixed epoch length. Per-regime / per-modality breakdown read out post-hoc from the wandb sample table. |
| MRI eval set | **Eval MRI on the existing `test` split** (`meta.csv` has only `train`/`test`, no `val`). Per-source split remap: CT eval `val`, MRI eval `test`. No `meta.csv` change. A dedicated MRI `val` split is deferred. |

## Background: how the v2 engine composes sources

`src/incontext_dataset_v2.py`'s `InContextDataset` runs over a single
`VolumeProvider` (`classes`, `subjects_for(cls)`, `load(subject, cls, req)`). It
has an **optional cohort hook**: if the provider defines `assemble_task`, the
engine flips to `cohort_mode` and delegates the entire K+1 selection + load to
`provider.assemble_task(rng, crop_spacing_mm)`, keeping only:

- the per-item RNG (`rng = Random(hash((eval_seed, idx)))` when `eval_seed` is
  set, else the global RNG),
- `(idx, spacing)` unpacking from `SpacingBatchSampler`,
- shared task-aug + per-volume intensity aug over the returned stack
  (`_augment_stacks`, applied only when the item carries an `"image"` key).

`src/providers/synth_gmm.py::SynthGmmProvider` is the existing template: it
exposes `epoch_length`, a `classes` list, a stub `subjects_for` returning `[]`,
and an `assemble_task` that returns the standard item dict. Its eval loader is
built by routing straight through `build_dataset(cfg, split)` (it is in
`make_eval_loader`'s special-case list).

`src/providers/totalseg.py::TotalSegProvider` is **modality-locked** at
construction (`modality="ct"|"mri"`): its own `root`, scan cache, bbox cache,
`spacings.json`, and (MRI only) `ct_stats.json` for per-subject z-score. CT uses
the `CtNormSpec` HU frame; MRI uses `normalize_mri(a, ct_stats[subject])`. Its
`load()` already stamps `LoadResult.modality`, which `incontext_collate_fn`
already forwards as `batch["modality"]` (list[str]).

**Why a cohort provider, not a lighter hook:** the regime split is a joint
constraint over the K+1 draws. The engine's non-cohort path picks the target with
`rng.choice(subjects_for(cls))` and contexts with a filtered lazy-shuffle loop —
there is no seam to say "first pick a regime, then constrain all K+1 modalities."
The cohort hook already hands the provider full ownership of that.

**Cost of the cohort path (accepted):** cohort mode disables the engine's
deterministic per-class `(subject, class)` eval enumeration and
`max_tasks_per_class`. The chosen eval design (stochastic + seeded + fixed
`eval_epoch_length`) does not need them; class coverage is statistical but
reproducible, and `class_balanced` sampling in `assemble_task` keeps every class
represented.

## Components

### 1. `src/providers/multisource.py` — `MultiSourceProvider`

Cohort-hook provider composing N modality-locked `TotalSegProvider` instances
(here: `{"ct": <totalseg>, "mri": <totalsegmri>}`).

```
__init__(sub_providers: dict[str, TotalSegProvider], *,
         context_size: int,
         regime_p=(1/3, 1/3, 1/3),   # (all-first, all-second, cross)
         epoch_length: int)
```

- `self.subs` — ordered dict; key order defines "first"/"second" modality for the
  pure regimes and is stable across runs.
- `self.classes` = sorted union of each sub-provider's `classes`, then dropped to
  those with `>=1` subject in `>=1` sub-provider.
- `self._avail: dict[str, list[str]]` — for each surviving class, the modality
  keys whose `sub.subjects_for(cls)` is non-empty.
- `self.epoch_length` — set by the builder (train: `max_ds_len_train or 1000`;
  eval: `source_mix.eval_epoch_length`).
- `subjects_for(cls) -> []` and `load(...) -> raise` — protocol stubs; the engine
  only calls `assemble_task` in cohort mode.

`assemble_task(rng, crop_spacing_mm) -> dict`:

1. `cls = rng.choice(self.classes)` — class-uniform (= class-balanced).
2. `m0, m1 = list(self.subs)` (e.g. `"ct"`, `"mri"`);
   `regime = rng.choices([m0, m1, "cross"], weights=self.regime_p)[0]`.
   The stored `meta["regime"]` is this value verbatim — a modality key
   (`"ct"` / `"mri"`) for the pure regimes, or `"cross"`.
3. Resolve a modality per slot (`1 + context_size` slots):
   - pure regime (`regime in (m0, m1)`): `want = regime`; if
     `want not in _avail[cls]`, `want = _avail[cls][0]` (fallback). All slots =
     `want`.
   - `cross`: `tgt_mod = rng.choice(_avail[cls])`;
     `ctx_mod = the other key if it is in _avail[cls] else tgt_mod` (fallback).
     Slot 0 = `tgt_mod`, slots 1..K = `ctx_mod`.
4. For each distinct modality in the slot list, draw the required number of
   **distinct** subjects from `self.subs[m].subjects_for(cls)` via a seeded
   shuffle. If the pool is smaller than needed, repeat entries with the same
   `warnings.warn(... "self-context fallback")` message the engine uses, so the
   behaviour and the log line match the non-cohort path.
5. `res = self.subs[m].load(subj, cls, LoadRequest(rng=rng, crop_spacing_mm=crop_spacing_mm))`
   for every slot. `rng` is threaded through so crop jitter stays seeded.
6. Stack and return the standard item dict:
   ```
   {
     "image": tgt.image, "label": tgt.label,
     "context_in": stack([r.image for ctx]),          # (K,1,T,T,T)
     "context_out": stack([r.label for ctx]),         # (K,T,T,T)
     "spacing": tgt.spacing,
     "crop_geom": tgt.crop_geom,
     "subject": tgt_subject,
     "context_subjects": [ctx subjects],
     "label_name": cls,
     "modality": tgt_mod,                              # target modality
     "aug_mode": torch.tensor(0, dtype=torch.long),
     "meta": {"regime": regime, "tgt_mod": tgt_mod, "ctx_mod": ctx_mod},
   }
   ```
7. The engine applies `_augment_stacks` (shared geometric task-aug across the
   whole K+1, per-volume intensity aug) because `"image"` is present.

Determinism: every stochastic choice (class, regime, per-modality subject
shuffle, and each `load`'s crop jitter) is drawn from the single `rng` the engine
passes. With `eval_seed` set, `rng = Random(hash((eval_seed, idx)))` → the item
for a given `idx` is byte-identical across models, runs, workers, and DataLoader
order.

### 2. `configs/experiment/3d/dataset/multisource_ct_mri.yaml`

New Hydra dataset group. Geometry inherited from the d2 varspacing regime.

```yaml
# @package _global_
paths:
  totalseg: ...        # repeated for cluster-independence (also in cluster/*.yaml)
  totalsegmri: ...
data:
  source: multisource
  loader_v2: true
  image_size: [128, 128, 128]
  context_size: 1
  train_spacing_range: [1.5, 6.0]     # per-batch log-uniform (SpacingBatchSampler)
  crop_spacing_mm: 3                  # fixed eval pitch
  mask_downsample: soft              # eval maps back to occupancy (make_eval_loader)
  mask_occupancy_thr: 0.5
  class_balanced: true
  max_ds_len_train: 1000
  ct_norm: null                      # CT frame; MRI sub-provider uses per-subject z-score
  train_classes: union               # sentinel -> union of per-source *train* specs
  val_classes: union
  source_mix:
    sources: [totalseg, totalsegmri]
    modalities: [ct, mri]            # parallel to `sources`; sub-provider modality flag
    per_source_train_classes: [balanced, train]   # spec fed to resolve_classes per source
    per_source_val_classes:   [all, test]
    regime_p: [0.334, 0.333, 0.333]  # (all-ct, all-mri, cross)
    split_map: {totalsegmri: {val: test}}         # CT keeps val; MRI val->test
    eval_epoch_length: 1400          # ~20 tasks/class over ~70 union classes
```

`union` for `train_classes`/`val_classes` is resolved by the wiring below;
`source_mix.per_source_{train,val}_classes` gives the exact per-source spec so
each `resolve_classes` call still gets a real split name / list.

### 3. `configs/experiment/3d/experiment/81_multisource_ct_mri.yaml`

```yaml
# @package _global_
defaults:
  - 80_varspacing_hard_tgt_prior
  - override /dataset: multisource_ct_mri
  - _self_

wandb:
  name: 81_multisource_ct_mri
```

Inherits exp80's `arch.mask_embed=conv`, the hard `query_prior` / `prior_perturb`
preset, `encoder_input_norm=instance`, and the exp70 train schedule.

### 4. Wiring — `experiments/3d/common.py` + `experiments/3d/train.py`

- **`_source_root(cfg)`** (`common.py`): add a `multisource` branch returning
  `("multisource", cfg.paths.totalseg, False)` — CT root as the nominal path
  (used for the node-local compile-cache key and gpu_realize gating, both
  modality-agnostic here). Sub-roots are resolved in the builder from
  `source_mix.sources`.
- **Class resolution**: a shared helper, e.g.
  `resolve_multisource_classes(cfg, which: "train"|"val") -> list[str]`, that
  zips `source_mix.sources` / `per_source_{which}_classes` / `modalities` and
  unions `resolve_classes(spec, paths[src], is_mri=(mod=="mri"))`. Called from:
  - `train.py::_resolve_classes_for` — add a `multisource` branch before the
    `_source_root` call (which would otherwise be fine, but the class spec is
    `union`, not a real split name).
  - `common.py::build_dataset` — same, for the `classes` passed to the provider.
- **`build_dataset(cfg, split)`** (`common.py`): add a `multisource` branch
  *before* the `_TOTALSEG_SOURCES` check:
  1. `is_train = split == "train"`.
  2. For each `(src, mod)` in `zip(sources, modalities)`:
     - `sub_split = source_mix.split_map.get(src, {}).get(split, split)`
     - `classes_spec = per_source_{train|val}_classes[i]`
     - `classes = resolve_classes(classes_spec, paths[src], is_mri=(mod=="mri"))`
     - build `TotalSegProvider(root=paths[src], classes=classes,
       image_size=..., split=sub_split, modality=mod, ct_norm=d.ct_norm,
       crop_spacing_mm=..., crop_jitter=(train vs eval.crop_jitter),
       mask_downsample=(d.mask_downsample for train / "occupancy" if "soft" for
       eval), mask_occupancy_thr=..., ram_cache=False)`.
  3. `provider = MultiSourceProvider({mod: sub for ...},
     context_size=d.context_size,
     regime_p=source_mix.regime_p,
     epoch_length=(d.max_ds_len_train or 1000) if is_train
                  else source_mix.eval_epoch_length)`.
  4. `return InContextDataset(provider, context_size=d.context_size,
     aug_cfg=(cfg.augmentations if is_train else None),
     defer_aug=(is_train and augmentations.gpu),
     crop_spacing_mm=d.crop_spacing_mm,
     eval_seed=(None if is_train else int(cfg.eval.seed)))` — cohort_mode
     auto-detected via `hasattr(provider, "assemble_task")`.
- **`make_eval_loader(cfg, classes, split, spacing)`** (`common.py`): add
  `"multisource"` to the special-case source set that routes through
  `build_dataset(cfg, split)` and returns the forkserver `DataLoader`
  (`incontext_collate_fn`, `persistent_workers`, `prefetch_factor=2`,
  `multiprocessing_context="forkserver"` under CUDA). Fixed-spacing eval: no
  `SpacingBatchSampler` (train.py's val call passes `spacing=None`); the cohort
  `__getitem__` uses `self.crop_spacing_mm` for bare-int indices.
- **`_assert_cascade_supported(cfg)`** (`common.py`): raise for
  `source == "multisource"` together with `cascade_spacings` or
  `gpu_realize_crop` — MRI cannot go through the native-crop GPU-realize path
  (per-subject MRI stats are not carried there;
  `_assert_cascade_supported` already rejects MRI + gpu_realize).

No change to the train step, the model, or `train_loader` — `train_loader`
already builds `RandomSampler(ds)` (length = `epoch_length`, capped by
`max_ds_len_train`) and wraps it in `SpacingBatchSampler` when
`train_spacing_range` is set; the cohort `__getitem__` unpacks `(idx, spacing)`.

### 5. `experiments/3d/evaluate.py::_sample_detail`

Currently returns `""` for anything without an omniSynth `class_id`. Add: if
`meta` carries `"regime"`, return `f"{meta['regime']} {meta['tgt_mod']}<-{meta['ctx_mod']}"`.
This lands in the sample-table `detail` column, so the wandb per-case table
(logged every eval by `build_sample_table`) breaks Dice down by regime and
modality pair with no schema change and no new plumbing. Aggregate + seen/unseen
macro Dice are unchanged.

### 6. Data

No `meta.csv` change. MRI eval uses the existing `test` split via
`source_mix.split_map`. Documented in the dataset config header. A dedicated MRI
`val` split is out of scope here.

## Data flow

```
train_loader
  -> RandomSampler(len = epoch_length, capped by max_ds_len_train)
  -> SpacingBatchSampler  -> (idx, s)  [s ~ logU(1.5, 6)]
  -> InContextDataset.__getitem__  (cohort branch)
       rng = global (train) | Random(hash((eval_seed, idx))) (eval)
       -> MultiSourceProvider.assemble_task(rng, s)
            pick class -> pick regime -> per-slot modality (+fallback)
            -> per modality: seeded distinct-subject draw
            -> TotalSegProvider[mod].load(subj, cls, LoadRequest(rng, s))  x (K+1)
            -> stack + item dict (+ modality, + meta{regime,...})
       -> engine _augment_stacks (task geom shared, intensity per-volume)
  -> incontext_collate_fn  (already forwards modality, meta, crop_geom)
  -> train step  (unchanged; encoder_input_norm=instance z-scores each volume;
                  modality list is not consumed by the encoder)
```

Eval is the same minus aug, minus `SpacingBatchSampler` (fixed 3 mm), with the
seeded per-item RNG and `eval_epoch_length` items.

## Testing

`experiments/3d/_check_multisource.py` (plain script, not pytest — matches the
repo's `_check_*` / `_plot_*` convention):

1. Build the **train** dataset; pull ~300 items:
   - regime frequencies within ~±0.05 of `regime_p`;
   - every `cross` item with both modalities available for its class has
     `item["modality"] != modality(context)` (derive context modality from the
     `meta`); fallback-class `cross` items are allowed to be same-modality and
     are counted separately;
   - both `ct` and `mri` pure regimes actually occur;
   - a class from `_avail == ["ct"]` never produces an MRI slot.
2. Build the **eval** dataset (`eval_seed` set); pull all `eval_epoch_length`
   items:
   - every union class with `_avail` non-empty appears at least once;
   - building the dataset twice and pulling `idx` in each yields an identical
     item (tensor-equal `image` / `label` / `context_in`).
3. `python experiments/3d/plot_dataset_items.py dataset=multisource_ct_mri --split train`
   — visual sanity (CT/MRI look right, masks aligned, `cross` panels show both).

## Known limitations

1. **MRI anisotropy.** TotalSegMRI slice thickness is 5–15 mm
   (`spacings.json`); cropping to a 1.5–6 mm isotropic grid upsamples
   through-plane. This is exactly what the existing `dataset=totalseg_mri` config
   already does (occupancy resample mitigates thin-structure loss); the
   multi-source path does not make it worse.
2. **Regime fallback on CT-only classes.** ~47 of the ~70 union classes have no
   MRI subjects, so their `mri` and `cross` regimes collapse to CT. Their
   effective modality mix is CT-only; expected given the data, surfaced by the
   test's fallback counter and the `detail` column.
3. **MRI eval on `test`.** Spends the MRI test split during training-time
   validation. A dedicated MRI `val` split (carved from `train`) is deferred.
4. **Cohort eval coverage is statistical.** No enumerated per-class task list;
   `eval_epoch_length` is sized for ~20 tasks/class on average. Reproducible, but
   per-class counts vary run-to-run only if `eval_seed` changes.

## Out of scope

- K > 1 multi-source contexts with a per-slot modality mix inside one regime
  (the current `cross` is target-vs-all-contexts).
- More than two sources (the provider is written for N but only CT+MRI is wired
  and tested).
- A third "balanced across regimes" enumerated eval mode.
- Cascade / gpu-realize with MRI.
