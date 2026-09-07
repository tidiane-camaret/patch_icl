# Multi-source (CT + MRI) cascade support — design

## Problem

`data.source=multisource` (the joint CT/MRI cohort provider,
`src/providers/multisource.py`) cannot run in cascade mode. Two blockers:

1. **Interface mismatch.** `MultiSourceProvider` is a *cohort* provider: it
   implements only `assemble_task(rng, crop_spacing_mm)`. `subjects_for()`
   returns `[]`, `load()` raises, and there is no `load_native_crop`. The cascade
   re-crop loop (`experiments/3d/cascade.py::_recrop_level`) needs the opposite:
   given the level-0 batch's `subjects` / `context_subjects` / `label_names`, it
   re-crops *those specific cases* at a finer spacing via
   `provider.load_native_crop(subj, cls, req)` (or `provider.load`). The cohort
   provider exposes no such entry point, and — critically — the level-0 batch
   does not record which sub-provider (modality) each case came from, so even a
   dispatching `load` would not know where to route.

2. **MRI + GPU realize.** `_assert_cascade_supported` (`common.py:281`) rejects
   `gpu_realize_crop=true` for `totalsegmri` because `NativeCrop` carries no
   per-subject normalization stats and `realize_native_crops` applies the global
   CT fingerprint unconditionally. A multisource run is ~1/2 MRI, so it inherits
   this.

`_assert_cascade_supported` (`common.py:248`) currently rejects
`source='multisource'` outright:

```
ValueError: data.cascade_spacings: source 'multisource' is not a v2 TotalSeg
source (['chemotox', 'totalseg', 'totalsegmri']).
```

## Goal

Run the multisource CT/MRI cohort through the N-level PatchSet3D cascade
(`data.cascade_spacings`), keeping the fast NFS-free path the caller asked for
(`gpu_realize_crop=true` + `ram_cache=true`). As a coherent side effect,
single-source `totalsegmri` cascade + `gpu_realize_crop` is also unblocked.

Target command (documented in `docs/logs.md`):

```bash
python experiments/3d/train.py experiment=81_multisource_ct_mri \
  +data.ram_cache=true \
  augmentations.goal_mask.p=0.4 \
  augmentations.goal_mask.ops=[dilate,erode,boundary,sobel] \
  data.source_mix.per_source_train_classes=[all,all] \
  train.checkpoint=/nfs/.../2026-09-06_82_multisource_ct_mri_train_all_classes/best.pt \
  train.batch_size=2 train.lr=1.0e-5 train.warmup_epochs=1 \
  data.train_spacing_range=null \
  data.crop_spacing_mm=6 +data.cascade_spacings=[6,3] \
  +data.cascade_crop_jitter=null +data.gpu_realize_crop=true \
  +data.cascade_query_prior.modes=[pred,none,gt] \
  +data.cascade_query_prior.p=[0.6,0.3,0.1] \
  +data.cascade_query_prior.eval_mode=pred \
  +data.cascade_query_prior_hard=false \
  +train.cascade_loss_weights=[1,1] \
  wandb.name=83_multisource_cascade
```

`data.train_spacing_range=null` is required: the `multisource_ct_mri.yaml`
default sets `train_spacing_range: [1.5, 6.0]`, which
`_assert_cascade_supported` (rightly) forbids alongside `cascade_spacings`.

## Non-goals

- The wandb sample-table regime/modality `detail` column for the **cascade** val
  pass. `evaluate_cascade` builds its own rows/cases and does not thread `meta`;
  wiring that is a separate change.
- Any change to the modality *regime* logic (`regime_p`, the regime-conditional
  class draw). The regime is fixed by level-0's `assemble_task`; re-crop levels
  only follow the modality it chose.
- `MultiSourceProvider.subjects_for` — still returns `[]`; `_recrop_level` never
  calls it.
- CPU-re-crop fallback perf. `gpu_realize_crop=false` for multisource will work
  (dispatching `load`), but it pays the NFS re-crop cost per level and is not
  optimized.

## Design

### Component 1 — `NativeCrop` self-describes its normalization

**File:** `src/providers/totalseg.py` (`NativeCrop`, `build_native_crop`,
`TotalSegProvider.load_native_crop`) and `src/gpu_realize_crop.py`
(`_realize_member`).

- `NativeCrop` gains `norm: CtNormSpec`.
- `build_native_crop` takes a resolved `norm` (replacing the vestigial
  `ct_spec=None` param) and stores it.
- `TotalSegProvider.load_native_crop` passes:
  - **CT**: `self.ct_spec` (already a `CtNormSpec`).
  - **MRI**: `resolve_ct_norm(self._ct_stats[subject])`. `self._ct_stats` is
    already loaded in `__init__` for `modality == "mri"`; its entries are
    `{clip_lo, clip_hi, mean, std}` dicts (`mri_stats`), which `resolve_ct_norm`
    already accepts. `normalize_mri` is pointwise `clip → (x - mean) / std` —
    identical in form to `normalize_ct` — so a per-member `CtNormSpec`
    reproduces it exactly.
- `_realize_member(nc, T, mask_downsample, occ_thr, ct_spec, device)`: use
  `getattr(nc, "norm", None) or ct_spec` for `normalize_ct_gpu`. Backward
  compatible: any payload without `norm` still uses the caller's `ct_spec`.
- `realize_native_crops` signature is unchanged; each member now normalizes
  itself, so a **single batch may mix CT and MRI members** (the `cross` regime).

### Component 2 — `MultiSourceProvider`: load methods + native emission

**File:** `src/providers/multisource.py`.

- `__init__(..., gpu_realize_crop: bool = False)`.
- `assemble_task(rng, crop_spacing_mm)` — modality / class / subject draw
  unchanged. Then branch on `self.gpu_realize_crop`:
  - **True** → return an imageless native payload:
    ```python
    {
      "native_crop": [tgt_nc, *ctx_ncs],   # via self.subs[mod].load_native_crop
      "subject": tgt_subj,
      "context_subjects": list(ctx_subjs),
      "label_name": cls,
      "tgt_modality": tgt_mod,
      "ctx_modality": ctx_mod,
      "aug_mode": torch.tensor(0, dtype=torch.long),
      "meta": {...unchanged...},
    }
    ```
    No `"image"` key → `InContextDataset.__getitem__` cohort branch already
    skips CPU aug (`if "image" in item and self._aug_active()`).
  - **False** → the current painted dict, **plus** explicit `"tgt_modality"` /
    `"ctx_modality"` keys.
- New methods for `_recrop_level`:
  ```python
  def load_native_crop(self, subject, cls, req, *, modality):
      return self.subs[modality].load_native_crop(subject, cls, req)

  def load(self, subject, cls, req, *, modality=None):
      if modality is None:
          raise RuntimeError("MultiSourceProvider.load needs modality= "
                             "(cohort provider; level-0 uses assemble_task)")
      return self.subs[modality].load(subject, cls, req)
  ```
- All K contexts in a task share one modality by construction
  (`ctx = [_load(ctx_mod, s) for s in ctx_subjs]`), so `ctx_modality` is a
  **scalar per row**, not a per-k list.

### Component 3 — modality threading to the re-crop loop

**Files:** `src/gpu_realize_crop.py` (`native_crop_collate_fn`),
`src/totalseg_dataloader_incontext.py` (`incontext_collate_fn`),
`experiments/3d/cascade.py` (`realize_cascade_level0`, `_recrop_level`).

- Both collates: pass `tgt_modality` / `ctx_modality` through **when present**
  (guarded on `"tgt_modality" in batch[0]` — single-source items lack them):
  `out["tgt_modality"] = [b["tgt_modality"] for b in batch]` (and `ctx_modality`).
- `realize_cascade_level0`: re-attach `tgt_modality` / `ctx_modality` next to the
  `subjects` / `context_subjects` / `label_names` passthrough it already does.
  `run_cascade` replaces `batch` with this realized dict before calling
  `_recrop_level(provider, batch, ...)`, and `_recrop_level` reads the modality
  keys from it.
- `_recrop_level`: each `tasks` entry gets a 6th field `mod`:
  - target row `b` (k == -1) → `(batch.get("tgt_modality") or [None]*B)[b]`
  - its contexts → `(batch.get("ctx_modality") or [None]*B)[b]`
  Update the two unpack sites (`_load_nc`, `_load`) and the
  `for (b, k, *_), r in zip(tasks, results)` reassembly (star-unpack already
  tolerates the extra field; `_regroup` keys on `t[0]`, unaffected).
  `_load_nc` / `_load` call `provider.load_native_crop(..., modality=mod)` /
  `provider.load(..., modality=mod)` **only when `mod is not None`**, so
  single-source `TotalSegProvider.load*` signatures stay untouched.

### Component 4 — `build_dataset`, `_assert_cascade_supported`, RAM cache

**File:** `experiments/3d/common.py`.

- multisource branch: construct
  `MultiSourceProvider(subs, ..., gpu_realize_crop=_realize)` with the same
  train-only formula as the `loader_v2` branch:
  `_casc = bool(d.get("cascade_spacings"))`,
  `_realize = (split == "train") and bool(d.get("gpu_realize_crop", _casc))`.
  Eval provider stays painted (`gpu_realize_crop=False`), matching the existing
  "eval loader never emits native-crop payloads" rule; eval levels ≥ 1 still
  route through `_recrop_level`.
- Each sub-`TotalSegProvider` gets
  `ram_cache=(_realize and bool(d.get("ram_cache", False)))` (currently hard
  `False`). This is **two** RAM caches (CT + MRI) → roughly 2× the single-source
  ~35 GB footprint; expected given `ram_cache=true`, noted in `docs/logs.md`.
- `_assert_cascade_supported`:
  - Allow `source == "multisource"` (add to the `_TOTALSEG_SOURCES` membership
    check, e.g. `_TOTALSEG_SOURCES | {"multisource"}`).
  - **Remove** the `_gr and source == "totalsegmri"` block (lines 281–285):
    Component 1 makes MRI GPU-realize correct.

### Component 5 — eval

`evaluate_cascade` already passes `loader.dataset.provider` (the
`MultiSourceProvider`) into `run_cascade`, and its eval loader uses
`incontext_collate_fn`. With Components 2–3 the painted level-0 carries
`tgt_modality` / `ctx_modality` and `_recrop_level` routes automatically. No
change to `evaluate_cascade` itself.

## Data flow

**Train, `gpu_realize_crop=true`:**

```
InContextDataset.__getitem__ (cohort)
  -> MultiSourceProvider.assemble_task            # native payload, no "image"
       -> subs[tgt_mod].load_native_crop(...)     # NativeCrop.norm set per modality
       -> subs[ctx_mod].load_native_crop(...)
  -> native_crop_collate_fn                       # + tgt_modality / ctx_modality
train_epoch: batch = crop_realizer(batch)
  -> realize_cascade_level0
       -> realize_native_crops                    # each member self-normalizes
       -> re-attach subjects / *_modality / label_names / aug_mode
run_cascade(model, MultiSourceProvider, batch, ...)
  level 0: forward on realized batch
  level i>0: _recrop_level(provider, batch, centers, spacing_i, realize_crop=True)
       -> per task: mod = batch[tgt|ctx _modality][b]
       -> provider.load_native_crop(subj, cls, req, modality=mod)
       -> realize_native_crops                    # mixed CT/MRI members OK
```

**Eval (cascade val):** level 0 painted from `assemble_task`
(`incontext_collate_fn` carries `*_modality`); levels > 0 identical to train's
`_recrop_level` path.

## Error handling

- `MultiSourceProvider.load` without `modality=` raises `RuntimeError` with a
  message pointing at `assemble_task` (guards a future non-realize misuse).
- `_recrop_level` only forwards `modality=` when the batch actually carries the
  keys, so `source=totalseg` / `totalsegmri` single-source cascade is
  byte-identical to today.
- `resolve_ct_norm(self._ct_stats[subject])` raises `KeyError` if an MRI subject
  is missing stats — desired (loud) failure; the existing MRI `load` path
  already assumes the sidecar is present.

## Testing

- **`src/providers/test_multisource.py`** (extend, fake sub-providers):
  - `gpu_realize_crop=True`: `assemble_task` returns `native_crop` of length
    K+1, no `"image"` key, `tgt_modality` / `ctx_modality` present and
    consistent with `meta["tgt_mod"]` / `meta["ctx_mod"]`.
  - `load_native_crop(s, c, req, modality="mri")` and
    `load(s, c, req, modality="ct")` dispatch to the matching fake sub-provider
    (record the call); `load(...)` without `modality` raises.
  - a `cross` regime (forced) yields `tgt_modality != ctx_modality`.
- **`_realize_member` norm test** (`src/` realize test, new or extend): build a
  `NativeCrop` with a known `norm` `CtNormSpec` over a toy array; assert the
  realized image equals `normalize_ct` / `normalize_mri` applied to the same
  array + resample.
- **`experiments/3d/_check_multisource.py`** (extend, NFS-guarded): build the
  dataset with the cascade overrides; assert the provider is a
  `MultiSourceProvider` with `gpu_realize_crop` set for the train split, and
  that `assemble_task` yields a native payload.
- **GPU smoke** (if `nvidia-smi` shows a free card on this node): the target
  command with `train.epochs=1 data.max_train_subjects=8
  data.max_ds_len_train=16`, expecting a clean cascade step + val pass. Else
  CPU-only (the three checks above).

## Files touched

| File | Change |
|------|--------|
| `src/providers/totalseg.py` | `NativeCrop.norm`; populate it (CT spec / per-subject MRI spec) in `build_native_crop` + `load_native_crop` |
| `src/gpu_realize_crop.py` | `_realize_member` honours `nc.norm`; `native_crop_collate_fn` passes `tgt_modality` / `ctx_modality` |
| `src/providers/multisource.py` | `gpu_realize_crop` ctor flag; native branch in `assemble_task` + `*_modality` keys; `load` / `load_native_crop` dispatch |
| `src/totalseg_dataloader_incontext.py` | `incontext_collate_fn` passes `tgt_modality` / `ctx_modality` |
| `experiments/3d/cascade.py` | `realize_cascade_level0` re-attaches `*_modality`; `_recrop_level` per-task modality routing |
| `experiments/3d/common.py` | multisource branch: `gpu_realize_crop` + `ram_cache` to provider/sub-providers; `_assert_cascade_supported` allows `multisource`, drops the MRI+realize block |
| `src/providers/test_multisource.py` | new routing / native-emission cases |
| `experiments/3d/_check_multisource.py` | cascade-override build assertion |
| `docs/logs.md` | change log + runnable command |
