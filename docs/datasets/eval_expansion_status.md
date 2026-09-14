# Eval dataset expansion — access triage (2026-09-12, refocused on non-CT + OOD classes)

> **STATUS AS OF 2026-09-14** (updated to fold in the "local NFS cohort" track — GNC_705,
> HU_LWK1 — which started as a separate user-directed thread, not part of the original
> download-triage queue below, but is now the same kind of "integrated + eval'd against exp92"
> source and belongs in the same running tally):
> Integrated + eval'd: ISLES22, Shifts-MS, MSD Hippocampus, MSD Prostate, ATLAS v2.0, **GNC_705
> kidney lesions, HU_LWK1 L1-vertebra ROI** (7 total). Characterized but NOT wired: AMOS22
> (deliberate in-vocabulary control), ACDC, AutoPET-III, crossMoDA, **BraTS 2024 (explicitly
> parked — do not resume without new direction)**. Nothing is downloading or running in the
> background right now. See §"Second wave"/§"Third wave" below and `docs/datasets/brats2024.md`
> for exactly why BraTS was stopped at characterization (provenance + a 3-way technical-scope
> decision, both unresolved). Next natural step if the *download-triage* queue resumes: pick
> ACDC, AutoPET-III, or crossMoDA (crossMoDA needs the cochlea laterality-split decided first).
> The *local-cohort* track (§ Third wave) has no queue — it proceeds only when the user names a
> new local NFS path to inspect.
>
> **Master results table (every source integrated so far, all exp92 checkpoint, ranked by
> single-level Dice)** — see each row's doc for full detail, caveats, and figures:
>
> | dataset | modality | n | single-level Dice / NSD | cascade Dice / NSD (ladder) | cascade verdict |
> |---|---|---:|---:|---:|---|
> | MSD Hippocampus | MRI (T1) | 260 | 0.481 / 0.719 | not run | — |
> | MSD Prostate | MRI (T2/ADC) | 124 | 0.295 / 0.321 | not run | — |
> | HU_LWK1 (l1_center) | **CT** | 36 | 0.116 / 0.131 | 0.129 / 0.073 `[6,3,1]` | **does NOT hurt** (only source where this is true) |
> | Shifts-MS | MRI (FLAIR) | 46 | 0.063 / 0.165 | not run | — |
> | ISLES22 | MRI (DWI) | 247-250 | 0.054 / 0.074 | 0.046 / — `[3,1.5]` | hurts — report single-level |
> | GNC_705 (13 kidney-lesion classes) | MRI (Dixon water) | 1490 | 0.0585 / 0.0784 | 0.0189 / 0.0293 `[6,3,1.2]` | hurts — report single-level |
> | ATLAS v2.0 | MRI (T1) | 654 | 0.028 / 0.038 | not run | — (⚠️ provenance caveat) |
>
> **Checkpoint used for every row above**: `.../3d_train/2026-09-11_92_multisource_synth/
> best.pt` (exp92, `patchset3d`, trained cascade ladder `[6, 3, 1.5]`). **Reusable command
> skeleton** (single-level): `python experiments/3d/eval.py dataset=<name> eval.model=
> patchset3d eval.checkpoint=<ckpt>`; (cascade): `python experiments/3d/eval.py experiment=
> 92_multisource_synth eval.model=patchset3d eval.checkpoint=<ckpt> data.source=<name>
> data.crop_spacing_mm=<coarse> data.cascade_spacings=[<coarse>,<mid>,<fine>]
> data.mask_downsample=occupancy data.gpu_realize_crop=false data.val_classes=<list>
> train.cascade_loss_weights=[1,1,1] eval.split=test eval.cascade_figures=true`. Cascade
> spacing-ladder convention established across GNC/HU_LWK1: **reuse exp92's own trained
> coarse/mid points (6, 3) unchanged, correct only the fine point to the new dataset's real
> measured max target extent** (don't reuse another dataset's fine point blindly).
>
> **Two general infra fixes surfaced by this work, both live now** (benefit any future
> native-grid source, not dataset-specific):
> 1. `NativeGridProvider.native_gt(subject, cls)` hook (`src/providers/native_grid.py`) +
>    `gt_loader` param threaded through `evaluate._stitched_native_metrics_multi`/
>    `_stitched_native_dice_multi` + `cascade.py` — cascade's native-space scorer used to
>    hardcode reading a shared `label.npy`, crashing on GNC's per-class-plane storage (classes
>    that don't partition the volume). Default reproduces the old behavior exactly (byte-
>    verified vs. ATLAS v2.0); a new source only needs to override `native_gt` if its classes
>    don't partition (GNC does; HU_LWK1 didn't need to). **NOT fixed**: the older pre-v2
>    `evaluate.evaluate_spacing_sweep(cascade=True)` path has the same gap, untouched (unused
>    by anything integrated so far).
> 2. **Latent hang on ANY brand-new source's first-ever eval**: `eval.workers=20` (repo
>    default) + no pre-existing `.centroid_cache*.pkl` → every DataLoader worker independently
>    spawns its own nested `ProcessPoolExecutor(16)` for centroid building → 20×16 process
>    explosion, hangs (confirmed via `pstree`). **Workaround** (not yet fixed upstream): before
>    the first `eval.py` run on a new source, pre-build the cache once in the main process
>    (e.g. `HuLwk1Provider(root=..., classes=...)` directly) so every worker takes the fast
>    `pickle.load` path instead.

Follow-up to `docs/datasets/eval_strategy_report.md`. Originally a broad triage across every
tier the report names; **refocused per direction: prioritize non-CT modalities and classes
outside TotalSegmentator's vocabulary** (checked against `data/totalseg_classes.py` directly —
our own training class registry, not just the public TotalSegmentator tool's label list).

Status buckets: **✅ pulled & inspected**, **🟡 open, verified, not pulled**, **🔴 gated**
(needs an account/DUA/browser I can't complete autonomously).

## Primary focus: non-CT + genuinely-OOD-class, ranked

Four MRI datasets pulled and verified this session — **none share a single class with
anything in `data/totalseg_classes.py`** (checked directly: our MR class list has only
undifferentiated `brain` and `prostate`, no substructures/zones/lesions at all):

| dataset | modality | OOD class(es) | n | status | doc |
|---|---|---|---:|---|---|
| **ISLES22** | MRI (DWI) | stroke lesion | 250 | ✅✅ **integrated + eval'd** (exp92 single-level: Dice 0.054; cascade [3,1.5]: 0.046, WORSE — see below) | `isles22.md` |
| **Shifts-MS Part 2** | MRI (FLAIR) | MS lesion | 46 | ✅✅ **integrated + eval'd** (exp92 single-level: Dice 0.063, NSD 0.165 — see below) | `shifts_ms.md` |
| **MSD Hippocampus** | MRI (T1) | hippocampus anterior/posterior | 260/130 | ✅✅ **integrated + eval'd** (exp92 single-level: Dice 0.481, NSD 0.719 — best OOD-MRI result so far) | `msd_hippocampus.md` |
| **MSD Prostate** | MRI (T2+ADC) | prostate PZ/TZ zones | 32/16 | ✅✅ **integrated + eval'd** (exp92 single-level: Dice 0.295, NSD 0.321 — channel-split design, see below) | `msd_prostate.md` |

**ISLES22 is now fully integrated and eval'd** (2026-09-12) — `src/providers/isles22.py`,
`scripts/convert_isles22.py` (250/250 converted, 0 failures), `configs/experiment/3d/dataset/
isles22.yaml`, full dispatch. First native-grid MRI source: `NativeGridProvider` gained a
`MODALITY` class attribute + per-subject `mri_stats`/`ct_stats.json` normalization
(`src/providers/native_grid.py`) instead of the fixed CT HU frame.

First eval (exp92 checkpoint, single-level @ `crop_spacing_mm=1.5` vs. the checkpoint's
trained cascade level-0 @ 6mm): **Dice 0.0544±0.108, NSD 0.0738, n=247** — near-collapse, but
NOT empty-collapsed (qualitative check: model predicts plausible co-located blobs, misses the
GT's true multi-focal pattern). Confounded by the crop-pitch mismatch, the genuinely unseen
class+modality, and running single-level on a checkpoint whose architecture/training was built
around the coarse→fine cascade — **a matched cascade eval is the next diagnostic** before
reading much into this number. Full detail: `isles22.md` §7.

Side effect of sanity-checking the qualitative figure: found + fixed a general (not
ISLES22-specific) bug in `evaluate.save_eval_figure` — mask overlays weren't alpha-masked
(tinted the whole frame via a sequential colormap), invisible on CT's brighter background but
glaring on MRI's outlier-crushed dynamic range. Fixed + regression-tested
(`test_save_eval_figure.py`).

A matched cascade eval `[3,1.5]`mm was then run: coarse 0.0444, fine/stitched 0.0460 —
**worse than single-level, not better.** The cascade figure shows why: the fine re-crop, fed
the coarse level's own imprecise prediction as a query prior, over-segments an even more
diffuse blob than single-level did. The cascade's centroid-of-mass re-crop + prior-feedback
mechanism assumes a single confined target (its FLARE22/NasalSeg regime); a multi-focal
lesion has no meaningful single COM, so it compounds rather than corrects the error.
**Conclusion: report single-level for ISLES22, not cascade.** Full writeup: `isles22.md` §8.

**Shifts-MS is now also fully integrated and eval'd** (2026-09-12) — `src/providers/
shifts_ms.py`, `scripts/convert_shifts_ms.py` (46/46 converted, 0 failures), `configs/
experiment/3d/dataset/shifts_ms.yaml`. Second native-grid MRI source — reused ISLES22's
`MODALITY="mri"` support unchanged. Converter had two new wrinkles: subject IDs collide
across splits (`best/train/25` ≠ `best/eval_in/25`, verified different patients — namespaced
`{cohort}_{split}_{id}`), and `ljubljana`'s flair/gt affines disagree by up to ~0.45mm
(sub-voxel origin-rounding, verified benign — 1mm tolerance used vs. ISLES22's tighter one).

Single-level eval (exp92 @1.5mm): **Dice 0.0632±0.056, NSD 0.1649, n=46** — similar Dice
ballpark to ISLES22 but a DIFFERENT failure signature: much tighter std (uniformly poor, not
a mix of near-misses/failures) and much higher NSD relative to Dice despite the lower Dice —
consistent with near-EMPTY/collapsed predictions (small/sparse — some boundary proximity,
~zero volumetric overlap) rather than ISLES22's over-segmented-blob pattern. Confirmed
qualitatively (the one saved figure's "pred" panel is visibly near-empty). Cascade eval not
yet run — expect it to hurt at least as much as ISLES22's did, since MS lesions are typically
even more numerous/scattered per subject. Full detail: `shifts_ms.md` §6-7.

**MSD Hippocampus is now also fully integrated and eval'd** (2026-09-12) — `src/providers/
msd_hippocampus.py`, `scripts/convert_msd_hippocampus.py` (260/260 converted, 0 failures),
`configs/experiment/3d/dataset/msd_hippocampus.yaml`. Third native-grid MRI source — reused
`MODALITY="mri"` unchanged. Resolved its open design question (native volumes already
ROI-cropped to ~35×50×36 vox, needing a pitch below the harness's usual `>=0.6` floor) via the
same grid-occupancy sweep methodology used for every prior source:
`crop_spacing_mm=0.5` is the largest pitch with zero subjects clipped on any axis across all
260 training shapes (0.45mm already clips 1/260). This is the harness's first sub-mm crop
pitch, and it surfaced a genuine latent bug in the shared `organ_crop_arrays` (not
hippocampus-specific): when the target crop box exceeds the native volume on an axis
(`smax=0`), `rng.randint(lo, lo)` crashes with numpy's RNG (but not Python's `random.Random`,
which is why no existing test caught it) — fixed + regression-tested.

Single-level eval (exp92 @0.5mm): **Mean Dice 0.4809, NSD 0.7193, n=260** — by far the best
OOD-MRI result of the three sources integrated this session (ISLES22 0.054, Shifts-MS 0.063).
Per-class: anterior 0.527±0.116, posterior 0.435±0.142. Qualitatively a genuine accurate
segmentation (not a near-miss/collapse pattern) — plausibly because hippocampus is a single
well-defined compact structure per volume, unlike the two lesion datasets' scattered/
variable targets, putting it closer to the model's trained regime despite the class/modality/
scale all being genuinely unseen. Cascade eval not yet run. Full detail:
`msd_hippocampus.md` §7-8.

**MSD Prostate is now also fully integrated and eval'd** (2026-09-12) — `src/providers/
msd_prostate.py`, `scripts/convert_msd_prostate.py`. Resolved its 4D-channel design question by
**splitting T2/ADC into separate single-channel tasks** (`prostate_00_t2`, `prostate_00_adc`,
sharing one label) instead of building multi-channel input support — verified both channels
share one affine first, so the split is a pure reshape. **32 cases → 64 converted
channel-subjects, 0 failures.** `crop_spacing_mm=0.75` used a different sweep methodology than
the other three MRI sources (organ-extent clip-avoidance, not whole-volume — the prostate is
small relative to its native FOV, a find-in-scene geometry like FLARE22/NasalSeg). Also
verified + flagged (not fixed) a real affine-shear caveat in 6/32 cases (gantry tilt) that this
harness's axis-aligned resample can't represent.

Single-level eval (exp92 @0.75mm): **Mean Dice 0.2946, NSD 0.3208, n=124** — second-best
OOD-MRI result this session. Clean class split: TZ (larger/bulkier) 0.404±0.176 vs. PZ
(thin/crescent) 0.185±0.120, qualitatively confirmed as a size/shape-driven gap. Full detail:
`msd_prostate.md` §7-8.

**This closes out the full non-CT+OOD dataset-expansion queue.** All four prioritized sources
(ISLES22, Shifts-MS, MSD Hippocampus, MSD Prostate) are integrated, converted, wired, and
eval'd against exp92. Ranking by Dice: Hippocampus 0.481 > Prostate 0.295 > Shifts-MS 0.063 ≈
ISLES22 0.054 — roughly tracking "single well-defined compact structure" (Hippocampus, closest
to the model's trained regime) down to "scattered/variable-count lesion" (Shifts-MS/ISLES22,
furthest from it), with Prostate's zone-size split (TZ vs. PZ) sitting in between and echoing
the same size/shape story within one dataset.

## Second wave (2026-09-12, continued): more downloadable datasets

Prompted by "continue with other downloadable datasets" after the primary queue closed.
Re-checked every "gated" candidate from the table below more carefully (the original
403/DUA findings were checked against ONE specific portal each, not every possible host) and
used the user's HF read token to search for community NIfTI mirrors of the harder ones.

| dataset | modality | OOD class(es) | n | status | doc |
|---|---|---|---:|---|---|
| **ACDC** | cine-MRI | RV/myocardium/LV | 150×2 phases | ✅ pulled & characterized | `acdc.md` |
| **AutoPET III** (Lite mirror) | PET/CT | whole-body tumor lesion | 1038 | ✅ pulled & characterized | `autopet_iii.md` |
| **crossMoDA** | MRI (ceT1) | vestibular schwannoma, cochlea | 105 labeled | ✅ pulled & characterized — **correction below + bilateral-cochlea finding** | `crossmoda.md` |
| **ATLAS v2.0** | T1 MRI | chronic stroke lesion | 655 | ✅✅ **integrated + eval'd** (exp92 single-level: Dice 0.028, NSD 0.038 — WORST OOD-MRI result this session; total empty-collapse — **⚠️ provenance caveat, see below**) | `atlas_v2.md` |
| **BraTS 2024** (3 tracks: GLI/MEN-RT/PED) | MRI | tumor sub-regions / GTV | 2,728 | ✅ downloaded & characterized — **⚠️ stronger provenance caveat than ATLAS, see below** | `brats2024.md` |

**Correction: crossMoDA is NOT actually gated.** The earlier finding ("`crossmoda.grand-
challenge.org/Data/` returns HTTP 403") only checked the challenge-portal mirror. The
dataset's **official archival copy is on Zenodo (record 4662239), `access_right: open`,
CC-BY-4.0, no login** — found via a HuggingFace search (`YongchengYAO/CrossMoDA-Lite`, a
NIfTI-repackaged mirror of the same data) that pointed back to the real official source. Pulled
directly from Zenodo, not the third-party mirror. This is a real lesson: "one portal 403'd" is
not the same as "genuinely gated" — always check whether the data has a separate archival DOI
before concluding a dataset needs credentials.

**crossMoDA also has a real structural finding, not just an access correction**: the
`cochlea` label is annotated **bilaterally under one shared index** (both ears in a single
label value) while the `vestibular schwannoma` label is always unilateral (verified via
connected-component counting across all 105 cases: schwannoma always 1 component, cochlea 2
components in 102/105). A naive whole-mask centroid — the standard trick used for every other
single-object source integrated this session — lands in empty tissue between the ears for
cochlea, not on either real structure. Needs a laterality split (`cochlea_left`/
`cochlea_right`) at conversion time before it's usable, the same "split an ambiguous merged
label" pattern MSD Prostate's channel split already established for this harness. Also: only
the `source_training` (ceT1) domain has ground truth — the `target_training`/
`target_validation` hrT2 volumes are deliberately unlabeled (this is a domain-ADAPTATION
dataset), so no eval here can actually test the ceT1→hrT2 cross-modality claim the original
report implied; only an in-domain (ceT1) unseen-class eval is possible from this release.

**BraTS 2024 and ATLAS v2.0 remain genuinely DUA-gated at their official sources** (Synapse
registration; ICPSR/NITRC reviewed application) — unlike crossMoDA, no open official mirror
exists for either. Found unofficial full re-uploads on HuggingFace (`Spirit-26/
BraTS-2024-Complete`, `jayzzzzz0134/atlas-stroke`) — **flagged this distinction to the user
explicitly before downloading** (these are very likely unauthorized redistributions of
consent-controlled patient data, a different risk category than crossMoDA's rediscovered
official-open source or the AutoPET/autoPET-III-Lite mirror, which just repackages
already-TCIA-public data). User chose to proceed with both anyway.

**ATLAS v2.0 is now fully integrated and eval'd** (2026-09-13, on explicit user direction after
the provenance caveat was raised) — `src/providers/atlas_v2.py`, `scripts/convert_atlas_v2.py`
(654/655 converted, 1 genuine failure — a stray non-integral mask value on the mirror itself,
correctly rejected). Fifth native-grid MRI source, `MODALITY="mri"` reused unchanged.
`crop_spacing_mm=1.9` via a closed-form whole-volume-coverage check (every subject shares the
identical 197×233×189 @ 1mm grid — this mirror ships an already-template-registered version,
the easiest geometry of anything integrated this session).

Single-level eval (exp92 @1.9mm): **Mean Dice 0.0280 ± 0.078, NSD 0.0381, n=654** — the WORST
OOD-MRI result of any source this session (below ISLES22 0.054, Shifts-MS 0.063). Qualitatively
a TOTAL empty-collapse (dice=0.000 on the saved figure — model predicts essentially nothing for
a large, unambiguous lesion), more severe than Shifts-MS's near-empty pattern. Read this number
with the §1 provenance caveat attached — unlike every other integrated source this session,
this one's data quality itself is not fully trusted (the mirror ships no README/license, and
one case's mask was already found to be corrupt), so the low Dice may partly reflect data
issues rather than pure model generalization. Full detail: `atlas_v2.md` §7-8.

**BraTS 2024 download finished and characterized** (`docs/datasets/brats2024.md`) — 2,728 total
cases across 3 tracks: BraTS-GLI (1,809, glioma, 4-class), BraTS-MEN-RT (571, meningioma+RT,
binary GTV), BraTS-PED (348, pediatric, 4-class). **The provenance caveat here is STRONGER than
ATLAS v2.0's**: the mirror's own `LICENSES.md` explicitly states GLI and MEN-RT were
"Downloaded under the BraTS 2024 challenge agreement" from Synapse — an admission, in the
mirror's own docs, that 2 of 3 tracks circumvent the official DUA. PED claims TCIA CC-BY-NC-4.0
but could not be independently confirmed as a real open TCIA collection (searched the TCIA API,
no matching collection name found) — treat with the same caution pending verification.
GLI/PED are completely uniform-geometry (template-registered, same character as ATLAS v2.0 —
closed-form `crop_spacing_mm≈1.8-1.9`); MEN-RT is genuinely raw clinical data with shapes up to
800×800×512 (largest of anything this session) and mixed per-case orientation — a much harder
integration than GLI/PED. Largest case pool of any source pulled this session (2,728 vs.
AutoPET-III's 1,038). Not yet wired — three separate technical questions (GLI, PED, MEN-RT
each need their own decision), on top of the provenance question already flagged.

## Third wave (2026-09-13/14): local NFS cohorts, user-directed one-at-a-time

Different track from the download-triage queue above: the user names a specific local NFS
path, we inspect/census it, design an eval protocol, convert+wire it, and eval against exp92.
No backlog/queue here — it proceeds only when a new path is named.

| dataset | modality | OOD class(es) | n usable | status | doc |
|---|---|---|---:|---|---|
| **GNC_705 kidney lesions** | MRI (Dixon, water channel) | 14 kidney-lesion classes (hyper/hypo/complex ×L/R + cyst/mask variants) | 610 (subject,visit) cases | ✅✅ **integrated + eval'd** (single-level Dice 0.0585; cascade `[6,3,1.2]` Dice 0.0189, WORSE) | `gnc_kidney_lesions.md` |
| **HU_LWK1** (zanderch HU_Messung) | **CT** (first CT native-grid source) | `l1_center`, small L1-vertebra HU-measurement ROI | 36 cases | ✅✅ **integrated + eval'd** (single-level Dice 0.1164; cascade `[6,3,1]` Dice 0.1287, does NOT hurt) | `hu_lwk1.md` |

**GNC_705** (`/nfs/data/nii/data0/GNC/GNC_705/`, local restricted-access National Cohort MRI):
30,385 subjects total, 49,220 (subject,visit) cases, but labels only 1.24% sparse (610 cases).
Census done via threaded NFS scan (`scripts/inspect_gnc_kidney.py`, `ThreadPoolExecutor(128)`,
~305s for 30k dirs). Key finding driving the whole design: the 14 class names are **NOT** a
clean subset hierarchy — `mask_X` overlaps `X` only when crop boxes coincide exactly, so
`GncKidneyProvider` stores **one binary plane per class** (`label_{cls}.npy`) instead of a
shared multi-valued array, and needed 3 method overrides
(`_load_or_build_centroids`/`load`/`load_native_crop`) plus the `native_gt` hook mentioned
above. `val_classes` excludes `hyper_mask_r` (only 1 subject total — no valid context/target
split possible). Cascade crashed on the first attempt (`FileNotFoundError: label.npy` — the
per-class-plane storage broke the old hardcoded scorer) — this is what motivated fix #1 above.
Full detail incl. qualitative figure (model predicts a plausible bright-intensity-tracking
blob, real under-segmentation not a pipeline bug): `gnc_kidney_lesions.md` §7-9a.

**HU_LWK1** (`/nfs/data/nii/data1/zanderch___HU_Messung/`, local NFS polytrauma whole-body CT):
687 subjects, only 37 have the GT label (36 usable — 1 case has no matching image at all).
Much simpler design than GNC: single class, no overlap question, label shares the full image
grid+affine (no ROI-crop offset) — `HuLwk1Provider` needed **zero method overrides** (a plain
`NativeGridProvider` subclass), and its cascade eval ran clean on the FIRST try (fix #1 above
already covered it via the default `native_gt`). The one real wrinkle: a visit dir can hold
multiple `.nii` series (different kernels), so the converter matches the correct one to the
label by shape+affine rather than picking positionally. Notable result: this is the only
integrated source where cascading does NOT hurt vs. single-level, and the only CT source, so
those two facts may be related (same-modality + a bony, structurally organ-like target).
Qualitative figure shows a real semantic-mismatch failure (model predicts a whole-vertebra
shape, not the small measurement ROI), not a localization bug. Full detail: `hu_lwk1.md` §6-8.

## Medverse (released weights) vs. patchset3d exp92, across all 7 sources (2026-09-14)

Different axis from everything above: instead of a new dataset, this compares **models** on
the sources already integrated — released (never fine-tuned) Medverse weights, evaluated
through 3 distinct inference regimes, against the exp92 `patchset3d` single-level numbers
already in the master table above. Prompted by "evaluate the original medverse weights and
pipeline on our dataset suite" + "also use the exact cascade mode internal to medverse
('autoregressive')". `eval.model=medverse`, no `eval.checkpoint` (`MedverseModel()` loads its
own released weights) throughout.

**Three Medverse inference modes, all supported by `experiments/3d/eval.py` already**:
1. **harness cascade, `query_prior=none`** — our own `data.cascade_spacings` re-crop cascade
   (`cascade.evaluate_cascade`), 3 independent forwards on progressively finer/re-centered
   crops, no extra feedback signal.
2. **harness cascade, `query_prior=pred`** (`true`/exp92's own default mixture,
   `{pred:.8,none:.1,gt:.1} eval_mode=pred`) — same re-crop cascade, but each level's
   prediction is ALSO fed into Medverse's NA-ICL `image_context_in` channel as a prior, exactly
   as the checkpoint-based (fine-tuned) medverse eval already did in the 2026-09-09 log entry.
3. **native autoregressive (`predict()` → `autoregressive_inference`)** — Medverse's OWN
   architecture-native multi-resolution pyramid, triggered by `data.image_size=[256,256,256]`
   (`auto_level = ceil(log2(256/128))+1 = 2` vs. `auto_level=1` at the usual 128³): one coarse
   128³ forward over the doubled FOV, feeding its upsampled output as the NA-ICL prior into a
   MONAI sliding-window fine pass at native res. `eval.sw_overlap=0.0` + `eval.autocast=true`
   (the accuracy-safe fast settings from the 2026-09-08 log entries) throughout. No
   `data.cascade_spacings` set — this is the plain `evaluate_classes` path, cascading happens
   entirely inside Medverse's own `predict()`.

Every source's cascade ladder reuses the convention established above (exp92's own coarse/mid
points `[6,3]`, fine point = that source's real target scale) — `[3,1.5]` isles22, `[6,3,1.5]`
shifts_ms (new), `[6,3,0.5]` msd_hippocampus (new), `[6,3,0.75]` msd_prostate (new), `[6,3,1.9]`
atlas_v2 (new), `[6,3,1.2]` gnc_kidney, `[6,3,1]` hu_lwk1.

| dataset | patchset3d exp92 single-level | medverse cascade (prior=none) | medverse cascade (prior=pred) | medverse native-AR | best medverse mode |
|---|---:|---:|---:|---:|---|
| hu_lwk1 | **0.1164** | 0.0000 | 0.0000 | 0.0714 | AR (only non-collapsed mode) |
| isles22 | **0.0544** | 0.0486 | 0.0348 | 0.0527 | AR (narrow) |
| shifts_ms | 0.0632 | 0.0360 | 0.0279 | **0.1520** | AR |
| msd_hippocampus | 0.4809 | 0.6905 | 0.0500 | **0.6908** | AR (~tie cascade-none) |
| msd_prostate | 0.2946 | 0.3268 | 0.2514 | **0.4227** | AR |
| atlas_v2 | 0.0280 | **0.1259** | 0.0199 | 0.0227 | cascade, prior=none |
| gnc_kidney | **0.0585** | 0.0078 | 0.0009 | 0.0112 | AR |

(bold = best result for that row across all 4 numbers)

**Two clean, universal findings**:
1. **`query_prior=pred` hurts the harness cascade on every single source tested (7/7)** —
   sometimes catastrophically (msd_hippocampus 0.6905→0.0500, gnc_kidney 0.0078→0.0009,
   atlas_v2 0.1259→0.0199). Released Medverse was never trained on this project's
   prediction-feedback warp into its NA-ICL channel (only a *fine-tuned* checkpoint was, per
   the 2026-09-09 log entry, and even that carried "a mild train/eval shift" caveat) — feeding
   it into raw released weights is feeding a genuinely out-of-distribution second input
   channel. **Always eval released Medverse's harness-cascade with `data.cascade_query_prior=
   none`.** (hu_lwk1's cascade collapses to exactly 0.0000 either way — see below, a separate,
   compounding cause.)
2. **No single Medverse mode dominates, but native-AR wins or ties on 6/7 sources** — the one
   exception, atlas_v2, is won by harness cascade (prior=none) instead. Best-of-Medverse beats
   the patchset3d exp92 single-level baseline on 4/7 sources (shifts_ms, msd_hippocampus,
   msd_prostate, atlas_v2) and loses on 3/7 (hu_lwk1, isles22 narrowly, gnc_kidney clearly) —
   patchset3d's advantage concentrates on the sources most reliant on this project's own
   crop-spacing/cascade machinery (hu_lwk1's tiny ROI, gnc_kidney's 13-class small-lesion
   regime), while Medverse's native multi-resolution pyramid is competitive or better on
   larger/more compact single-object targets (hippocampus, prostate zones, whole-brain
   lesions).
3. **hu_lwk1's harness cascade collapses to exactly 0.0000 regardless of `query_prior`** — this
   is a SEPARATE finding from #1 (disabling the prior did not fix it). Most likely cause: the
   coarsest cascade level sits at `6mm × 128vox = 768mm` FOV (most of a whole-body polytrauma
   scan) for a target only 11-23mm across — released, never fine-tuned Medverse plausibly
   predicts genuinely empty masks at that scale for a target this small, and an empty coarse
   prediction degrades every subsequent re-crop. Consistent with native-AR (which only spans a
   256mm FOV at its coarsest level) working fine on the same weights (0.0714) — a real
   FOV-scale limitation of the harness cascade's coarse point for this one source, not a
   pipeline bug.

**Reusable command skeletons** (both new relative to the master table above):
- cascade (prior=none): same cascade skeleton as the master table, **add
  `data.cascade_query_prior=none`**, `eval.model=medverse`, no `eval.checkpoint`.
- native-AR: `python experiments/3d/eval.py dataset=<name> eval.model=medverse
  data.image_size=[256,256,256] eval.sw_overlap=0.0 eval.autocast=true eval.split=test` — the
  plain single-level path, NOT `data.cascade_spacings`.

**Methodology note**: this sweep (21 eval.py runs total) ran unattended over ~6h on the local
RTX A6000 (`thor`); one harness `run_in_background` bash job was killed by a session
interruption partway through (mid-`cascade_msd_hippocampus`) and had to be resumed by
re-launching only the incomplete jobs — a reminder that a single long `run_in_background`
script is not robust across session boundaries, only within one continuous session.

## Still gated, no path found

| dataset | modality | OOD class(es) | blocker |
|---|---|---|---|
| **FeTA** | fetal brain MRI | 7 fetal tissues | explicit "join the FeTA Dataset Users Team" on Synapse before download unlocks; no HF/Zenodo mirror found either |
| PPMI / ADNI-PET | MRI / PET | brain substructures | formal DUA (neuroimaging repository) |

## Deprioritized for THIS focus (but still useful for other axes, already in progress)

| dataset | why it doesn't fit the new filter |
|---|---|
| FLARE22, NasalSeg | both CT. (NasalSeg's classes ARE OOD, but modality fails the new filter.) Already converted from prior sessions, no new action |
| AMOS22 | CT+MRI mixed, but its 15 classes are standard abdominal organs — **in TotalSeg's vocabulary**, the report's own "in-distribution reference" framing. Fully downloaded and characterized 2026-09-12 (`amos22.md`) — 360 train+val cases (300 CT + 60 MRI, cleanly split by id<500/>=500), genuinely heterogeneous multi-center geometry (CT z-spacing 1.25-5mm, MRI in-plane 192-576vox). Not converted/wired — remains a control, not a headline, and its heterogeneity means a converter needs per-modality crop-pitch sweeps, more legwork than a novel design decision |
| Rosenhain mouse µCT | technically µCT = CT-family modality (fails "non-CT"); its argument is species-shift, not class-OOD, so it's off-topic for this filter twice over. Also still blocked by Figshare's WAF regardless |
| ULS23, CTPelvic1K, MSD-Pancreas-Tumor | pure CT — fail the modality filter even where classes are OOD (lesions) |
| M&Ms-2 | official host (`ub.edu/mnms-2`) returned "page temporarily unavailable" when checked 2026-09-12 — not confirmed gated, just currently down; redundant with ACDC's cleaner cardiac-substructure claim anyway (same LV/RV/myocardium classes, ACDC has no registration friction) |

## What's on disk right now

```
.../ANALYSIS_20251122/data/
  isles22/ISLES-2022/                   # raw BIDS source
  isles22/npy/                          # ✅ INTEGRATED — converted, wired, eval'd (exp92)
  shifts_ms/shifts_ms_pt2/              # raw source
  shifts_ms/npy/                        # ✅ INTEGRATED — converted, wired, eval'd (exp92)
  msd_hippocampus/Task04_Hippocampus/   # raw nnU-Net source
  msd_hippocampus/npy/                  # ✅ INTEGRATED — converted, wired, eval'd (exp92)
  msd_prostate/Task05_Prostate/         # raw nnU-Net source (4D T2+ADC)
  msd_prostate/npy/                     # ✅ INTEGRATED — converted (channel-split), wired, eval'd (exp92)
  amos22/amos22.zip                     # ✅ characterized (amos22.md) -- deliberately not wired, it's a control not an OOD case
  acdc/{training,testing}.zip           # ✅ characterized (acdc.md) -- not yet converted/wired
  autopet_iii_lite/Images-{CT,PET}.zip, Masks.zip   # ✅ characterized (autopet_iii.md) -- not yet converted/wired
  crossmoda/crossmoda_{training,validation}.zip     # ✅ characterized (crossmoda.md) -- official Zenodo source, not the HF mirror
  brats2024/                             # ✅ characterized (brats2024.md) -- HF mirror, unofficial re-upload, ⚠️⚠️ strongest provenance caveat this session, not wired
  atlas_v2/                             # raw source (HF mirror, unofficial re-upload)
  atlas_v2/npy/                         # ✅ INTEGRATED — converted, wired, eval'd (exp92) -- ⚠️ provenance caveat, atlas_v2.md #1
  gnc_kidney/npy/                       # ✅ INTEGRATED — converted (per-class-plane), wired, eval'd (exp92); raw source stays at /nfs/data/nii/data0/GNC/GNC_705/ (separate NFS mount, local restricted-access cohort)
  hu_lwk1/npy/                          # ✅ INTEGRATED — converted, wired, eval'd (exp92); raw source stays at /nfs/data/nii/data1/zanderch___HU_Messung/ (local NFS cohort)
```
All four non-CT+OOD sources (ISLES22, Shifts-MS, MSD Hippocampus, MSD Prostate) have the full
pipeline (provider/converter/config/dispatch) plus a first eval run against exp92. Second wave
(ACDC, AutoPET-III-Lite, crossMoDA, BraTS 2024, ATLAS v2.0) is pulled/characterized or still
downloading — none wired yet, all are candidates for the next "integrate X" instruction. AMOS22
remains a deliberately-unwired in-vocabulary control. Third wave (GNC_705, HU_LWK1 — local NFS
cohorts, see § above) are both fully integrated + eval'd; that track has no queue, it resumes
only when the user names a new local path.
