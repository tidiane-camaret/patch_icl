# Modality-agnostic input normalization — design

Date: 2026-09-03
Status: approved (brainstorming), pending spec review
Driver: future CT + MRI joint training; prep seams landed against
`experiments/3d/experiment/70_patchset_varspacing_6_1_5.yaml` (must stay byte-identical).

## Problem

We will later train on CT **and** MRI in one run, with **fully mixed cross-modal
in-context tasks** (a task's target and its K context pairs may be different modalities).
The current pipeline hard-assumes one global CT frame (`fingerprint_1228`) at three sites:

1. **Provider** (`src/providers/totalseg.py:230`): `self.modality` is a single value for the
   whole dataset; `load()` applies `normalize_ct(ct_spec)` OR `normalize_mri(per-subject
   stats)` — never both in one provider.
2. **GPU augmentor** (`src/gpu_augment.py`): every intensity op ends
   `.clamp(CT_NORM_MIN, CT_NORM_MAX)` (≈ `[-1.66, 3.44]`), constants derived from
   `DEFAULT_CT_NORM` (`src/totalseg_dataset.py:88-89`). Hard guard at `gpu_augment.py:412`
   raises unless `resolve_ct_norm(ct_norm) == DEFAULT_CT_NORM`.
3. **Encoder input norm** (`_norm` in `plainconv_ts.py` / `resenc_ts.py` / `nnunet_ts.py`):
   the `zscore` / `reframe` modes do `hu = x * loader_spec.std + loader_spec.mean` to invert
   "the loader frame" back to HU, with `loader_spec = resolve_ct_norm(loader_ct_norm)` and
   `loader_ct_norm` **hardcoded to `None` → `fingerprint_1228`** — never wired to config.

Goal: the model is **architecturally invariant** to which normalization produced its input
— its first op re-standardizes whatever arrives, with **no metadata / frame descriptor
passed** and no per-modality branching in the encoder path.

## Established facts (verified 2026-09-03)

- `data.ct_norm` reaches the v2 provider via `common.py:317` / `:705`
  (`ct_norm=d.get("ct_norm")`); `null` → `DEFAULT_CT_NORM` =
  `CtNormSpec(clip_lo=-1007, clip_hi=1573, mean=-167.3, std=505.8)`.
- `normalize_ct` (`totalseg_dataset.py:96`) = `clip(HU,[lo,hi])` then
  `(x - mean)/std` with **fixed global** mean/std → CT crops are *not* per-volume z-scored
  (a bone-heavy crop keeps a higher mean than a lung crop).
- `normalize_mri` (`totalseg_dataset.py:126`) uses **per-subject whole-volume** stats from a
  sidecar (`mri_stats`: fg p0.5/p99.5 clip, then fg mean/std) — also not per-crop.
- `modality` today is provider-wide, set from `cfg.data.source` via `_source_root`'s
  `is_mri` (`common.py:316`, `:549`, `:704`, `:754`). `assert modality in ("ct","mri")`
  (`totalseg.py:173`).
- `LoadResult` (v2, `src/incontext_dataset_v2.py:31`) carries `image, label, spacing,
  crop_geom` — **no modality, no frame**. `NativeCrop` (`totalseg.py:31`) likewise.
- The batch dict has `image, label, context_in, context_out, spacing, aug_mode, subjects,
  label_names` — no modality.
- `build_model` (`train.py:401`) passes only `encoder_input_norm` → encoder `input_norm`
  kwarg (`patchset3d.py:241/255/269`). `loader_ct_norm` / `target_ct_norm` are never
  passed by `patchset3d.py`; they sit at the encoder defaults (`None` / `"d297"`).
- `_INPUT_NORMS = ("passthrough", "reframe", "zscore")` is duplicated in all three conv
  encoders; `_norm` bodies are near-identical (differ only in the non-`zscore` target:
  plainconv/resenc use `target_ct_norm`, nnunet_ts uses its plans `CTNormalization`).
- `_assert_cascade_supported` (`common.py:240`) blocks MRI + `gpu_realize_crop`: "the
  NativeCrop payload carries no modality and `realize_native_crops` applies the CT
  fingerprint unconditionally".
- `PatchSet3D._feat_norm` (`patchset3d.py:508`) is a **separate** stage (encoder-feature
  z-score, not intensity); `m2_patchset_decoder` uses `feat_norm: self`.

## Decisions (from brainstorming)

1. **Loose dataloader contract, strict model stem.** The dataloader does *sane
   per-modality* normalization only (CT: clip → z-score; MRI: per-subject p0.5/p99.5 clip →
   z-score; modality N: its own clip → z-score). Output is "roughly zero-centred, roughly
   unit-scale, finite" — **not** a guaranteed exact frame. The model's input stem owns the
   single canonical frame and never inspects or trusts the incoming one.
2. **No metadata contract on the encoder path.** A `modality` string rides the
   `LoadResult` / batch for augmentation and analysis, but `PatchSet3D.forward` and the
   encoders ignore it.
3. **Invariant primitive = per-volume renorm.** `instance` mode: `(x - x.mean) / (x.std +
   eps)` over spatial dims per (B,C), optional learned per-channel affine (γ,β). No HU
   inversion. A `robust` variant (median / IQR) is specified but optional.
4. **Scope now = doc + no-op seams.** Land the low-risk seams so the later change is
   config-only; change no current behavior.

## Design

### 1. The canonical input contract

| Stage | Owner | Guarantee |
|---|---|---|
| Per-modality normalization | dataloader / provider | zero-ish centre, unit-ish scale, finite; modality-specific work is the **clip step** only |
| Intensity augmentation | GPU augmentor | operates in a frame it is *told*, not one it assumes |
| Canonical frame | **model input stem** | the sole strict standardization; identical output distribution regardless of upstream modality |
| Feature normalization | `PatchSet3D._feat_norm` | unchanged; `self` mode recommended for cross-modal tasks |

The contract between dataloader and model is deliberately weak. The stem is the single
place that must be correct.

### 2. Model — the invariant stem

New shared module **`src/models/encoders/_input_norm.py`**:

```
class InputRenorm(nn.Module):
    # mode: passthrough | zscore | reframe | instance | robust
    # zscore/reframe: EXISTING logic, moved verbatim (HU inversion via loader_spec,
    #                 then per-volume z-score / target reframe). Kept for back-compat.
    # instance (new): x.float(); per-(B,C) spatial mean/std standardize; NO HU inversion;
    #                 optional affine=True -> single learned (gamma, beta) for 1-ch input.
    # robust  (new, optional): per-(B,C) median / IQR in place of mean / std.
```

- `_INPUT_NORMS` enum + the `_norm` body move here; `plainconv_ts` / `resenc_ts` /
  `nnunet_ts` each hold an `InputRenorm` instance and call it. This de-dups ~30 lines per
  encoder. `InputRenorm` stays a plain `nn.Module`: `nnunet_ts` passes its plans
  `CTNormalization` (clip + mean/std) in as the `target_spec` (a `CtNormSpec`), so the
  `reframe` branch is shared code and its output is unchanged — no subclassing.
- `instance` defaults `affine=False` (pure standardize). The learned (γ,β) is a config
  sub-flag (`arch.encoder_input_norm_affine`, default false), so `instance` with the flag
  off is deterministic and parameter-free.
- Config: `arch.encoder_input_norm` gains `instance` / `robust`. **Per-encoder defaults
  unchanged** (`plainconv_ts` = `zscore`, `resenc_ts` = `passthrough`, `nnunet_ts` =
  `reframe`).
- For a mixed run: `arch.encoder_input_norm=instance`. `instance` ≈ today's `zscore` minus
  the `hu = x*loader.std + loader.mean` line — i.e. it drops the one hardcoded,
  modality-specific step.
- Placement: exactly where `_norm` sits today — first op inside the encoder forward, before
  the conv stem, after augmentation.

### 3. GPU augmentor — de-pin from the CT frame

- **Seam now**: replace bare `CT_NORM_MIN` / `CT_NORM_MAX` references in `gpu_augment.py`
  with instance attrs `self._clamp_lo` / `self._clamp_hi`, initialized from the CT frame
  by default. Add `GpuAugmentor(clamp_frame=...)`. Downgrade the `ct_norm != DEFAULT`
  hard-raise to: raise **unless** `clamp_frame` is explicitly supplied. Default path
  byte-identical.
- **Target**: a per-item `[0,1]` working space. At the top of intensity aug, map each item
  to `[0,1]` by its own robust percentiles (e.g. p0.5 / p99.5), run every op in `[0,1]`,
  map back. This is per-item (not per-batch) — required because a cross-modal task puts
  different frames on adjacent batch items. `gamma` / `simulate_low_resolution` already
  work in a `[0,1]`-remapped space, so most ops need only their clamp bounds changed.

### 4. Provider / dataloader — per-subject modality

- `TotalSegProvider.modality` (scalar) → per-subject: a `{subject: "ct"|"mri"}` map from a
  manifest or dir convention; `load()` / `load_native_crop()` pick `normalize_ct` vs
  `normalize_mri` per crop by the crop's subject.
- Preferred shape: a **`MultiModalProvider`** wrapping N single-modality providers, routing
  by subject over a unified subject pool. Cross-modal tasks need one pool with per-subject
  modality; a wrapper is cleaner than threading dual state through `TotalSegProvider`.
- Lifting the MRI + `gpu_realize_crop` block (`common.py:240`) depends on the `modality`
  seam (§5.2) + `realize_native_crops` taking a per-crop spec instead of the CT fingerprint.
  Out of scope for the seam PR; noted so the assertion text can point here.

### 5. No-op seams to land now

1. **`src/models/encoders/_input_norm.py`** — the shared stem module; refactor the three
   conv encoders onto it; add the `instance` mode (unreachable until selected). `robust`
   may land as a thin mean/std → median/IQR swap in the same PR or be deferred if it
   complicates the refactor — not on the critical path. Parity test: `zscore` / `reframe`
   / `passthrough` outputs bit-identical to pre-refactor on a fixed input, for each encoder.
2. **`modality` field** — `modality: str = "ct"` on `LoadResult` and `NativeCrop`,
   populated with the provider's current single value; `incontext_collate_fn` emits
   `batch["modality"]` (list[str] length B). Nothing downstream reads it.
   **Target-only, by design for this seam.** `batch["modality"]` is one string per task
   (the target's). The K context pairs' modalities are not carried — the non-cross-modal
   v1 of the follow-up is target-labelled tasks, and a mixed provider that assembles a
   cross-modal task (spec problem statement) is what introduces per-context modality: it
   will widen this to a `(B, K+1)` structure (`batch["modality"]` → `list[list[str]]`,
   plus a `context_modality` on the item dict, parallel to `context_in`). That widening
   is a change to an inert field the mixed-provider PR already touches — accepted here so
   the seam stays a scalar until there is a consumer to shape it against.
3. **`gpu_augment.py`** — `clamp_frame` constructor arg; clamp bounds become instance
   attrs; guard relaxed as in §3. `GpuAugmentor()` with no new arg → identical.

### 6. Migration & validation

- **Now**: every current config (exp 70 included) resolves and runs identically. Verify:
  `python experiments/3d/train.py experiment=70_patchset_varspacing_6_1_5 --cfg job` diff
  = empty; a 1-step forward-parity check (fixed seed) on exp 70 before/after the seam PR;
  the per-encoder `_norm` parity test in §5.1.
- **Later (mixed run)**: new `dataset=` group with a multi-modal source +
  `MultiModalProvider`; `arch.encoder_input_norm=instance`; `GpuAugmentor(clamp_frame=...)`
  (or the `[0,1]` working-space augmentor). No encoder-path code change.

## Out of scope

- The `MultiModalProvider` itself and the multi-modal `dataset=` group (later PR).
- The augmentor `[0,1]` working-space rewrite (later PR; only the `clamp_frame` seam now).
- Lifting MRI + `gpu_realize_crop`.
- Per-modality tuning of `feat_norm` / the transformer — flagged (`feat_norm: self` is the
  recommended cross-modal default) but not changed.
- Any change to `data.ct_norm` semantics or the `fingerprint_1228` / `d297` presets.
