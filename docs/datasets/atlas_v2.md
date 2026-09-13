# ATLAS v2.0 (Anatomical Tracings of Lesions After Stroke)

*Chronic stroke lesion segmentation, T1 MRI, multi-site.* Downloaded 2026-09-13 via an
unofficial HuggingFace mirror, following up on `docs/datasets/eval_strategy_report.md`'s
Tier-2 lesion pick. **Genuinely OOD class, verified against our own training vocabulary**: no
stroke/lesion class of any kind exists in `data/totalseg_classes.py`. Not yet wired into the
harness — characterization only.

## 1. Access — ⚠️ PROVENANCE CAVEAT: the official source is genuinely DUA-gated

**Verified this session: BOTH of ATLAS v2.0's known official/semi-official mirrors require a
reviewed application, no open path exists.** The ICPSR host is a restricted-use DUA; the
NITRC/INDI "preprocessed" copy also requires a Google-Form application plus a reviewed
encryption code before download unlocks. This is real patient neuroimaging data with a genuine
consent/DUA framework behind it — unlike crossMoDA (corrected elsewhere this session: that one
turned out to have a genuinely open official Zenodo archive once the actual DOI was found).

**What was actually downloaded**: an unofficial full re-upload on HuggingFace
(`jayzzzzz0134/atlas-stroke`, not gated, no README, no license tag, 655 image/mask pairs +
`train.jsonl`) — this is very likely an **unauthorized redistribution of DUA-gated,
consent-controlled patient data**, not a legitimate open mirror. **This distinction was flagged
explicitly to the user before downloading** (a direct question, not a silent decision either
way); the user chose to proceed with the download for their own research use. Recorded here so
this provenance risk is visible to anyone reading this doc later, independent of what gets done
with the data. Do not treat this as equivalent to the fully-open sources characterized
elsewhere in `docs/datasets/` (ISLES22, Shifts-MS, MSD tasks, ACDC, AutoPET-III-Lite, crossMoDA)
— those all have a genuinely open official or archival source; this one does not.

## 2. Composition — smaller than the official release, single class

655 (image, mask) pairs — smaller than the report's cited 1,271 for the full official release
(this mirror is likely only the GT-bearing "Release 1" training portion, not the full
challenge dataset including test/tracking splits without public GT — not confirmed against an
official manifest given the access situation, so treat the exact provenance/completeness of
this subset as unverified). Single class: `stroke lesion` (per `train.jsonl`'s
`"label": ["stroke lesion"]`).

`sub-r0XXsYYY` naming: **33 distinct site codes** (`r001`-`r0XX`) across the 655 subjects, site
sizes ranging from 5 to 111 subjects per site — genuine multi-site heterogeneity, matching the
report's framing of ATLAS as a large multi-center chronic-lesion set.

## 3. Geometry — VERIFIED, uniform and pre-processed (unlike every other source this session)

All 655 pairs: orientation **RAS** uniformly, shape **exactly 197×233×189**, spacing **exactly
1×1×1mm** — completely uniform across all 655 subjects and all 33 sites, unlike every other raw
clinical source pulled this session (which all show real inter-subject shape/spacing
variation). This uniformity strongly suggests the mirror ships an already-registered/
resampled-to-a-template version (consistent with ATLAS v2.0's documented preprocessing
pipeline — defaced, registered to MNI-152 space), not raw per-site acquisitions. **A genuinely
easy source to integrate geometrically** if the provenance question is resolved — no per-site
orientation/spacing sweep needed at all, unlike ACDC/AMOS22/AutoPET-III's real heterogeneity.

## 4. Labels — verified, extreme size range, no empty masks

All 655/655 masks non-empty. Lesion volume spans **13mm³ to 496,656mm³** — the largest
dynamic range of any lesion dataset pulled this session (ISLES22: 40mm³-482mL; AutoPET-III:
100mm³-2.86mL), consistent with "chronic stroke" covering everything from small lacunar
infarcts to massive territorial strokes.

## 5. Intensity — verified, standard T1 MRI

float32, verified range e.g. [0.0, 151.6] on a sample case (post-normalization, consistent
with the uniform-geometry finding above suggesting preprocessed data) — needs the same
per-subject `mri_stats`/`normalize_mri` mechanism as every other MRI source, though the
already-narrow/normalized range here may need less aggressive per-subject rescaling than raw
scanner-unit sources like ISLES22.

## 6. Fit as an eval set

- **Genuinely unseen class**, largest lesion-size dynamic range of anything pulled this
  session — would stress-test the size-generalization story raised repeatedly in this
  session's other lesion docs (ISLES22/Shifts-MS's small-to-scattered lesions vs. this
  dataset's small-to-massive range).
- **Geometrically the easiest source to wire** technically (§3) — but that ease is entirely
  gated behind the provenance question in §1, which is a policy/ethics decision, not an
  engineering one.
- **Multi-site (33 sites) gives a natural held-out-site generalization axis** if ever used,
  similar in spirit to Shifts-MS's built-in in/out-of-distribution cohort split.

## 7. Integration (implemented, 2026-09-13, on explicit user direction)

The provenance question in §1 was raised explicitly to the user before any integration work;
the user directed to proceed anyway. Wired as an eval-only v2 source, same shape as
ISLES22/Shifts-MS — `NativeGridProvider` `MODALITY="mri"`, no new code needed.

```bash
python scripts/convert_atlas_v2.py --workers 16       # .nii.gz -> native RAS .npy + mri_stats
python experiments/3d/eval.py dataset=atlas_v2 eval.model=patchset3d eval.checkpoint=<ckpt>
```

| piece | file |
|---|---|
| converter | `scripts/convert_atlas_v2.py` |
| provider | `src/providers/atlas_v2.py` (`AtlasV2Provider`, `MODALITY="mri"`) |
| config | `configs/experiment/3d/dataset/atlas_v2.yaml` |
| dispatch | `experiments/3d/common.py`, `experiments/3d/eval.py` (same branches the other 4 MRI sources use) |

**`crop_spacing_mm=1.9`** — a whole-volume-coverage sweep (like ISLES22/Shifts-MS, not the
organ-extent-clip-avoidance sweep used for MSD Prostate/ACDC): since every subject shares the
identical 197×233×189 @ 1mm grid, this reduces to a single closed-form check across 3 axes.
1.9mm (FOV=243.2mm) is the smallest pitch with zero clipping on all 3 axes; 1.8mm already
clips the 233mm axis by <1%.

**Converted: 654/655 cases, 1 genuine failure** — `sub-r039s002`'s mask contains a stray
non-integral value (0.01 instead of 1, a data-quality artifact on the mirror itself, not a
conversion bug), correctly rejected by the strict integral-mask check rather than silently
coerced. `ct_raw.npy` float32 T1; `label.npy` uint8 0/1.

Visual sanity check: `results/3d/atlas_v2_items.png` — plausible chronic-stroke lesion
patterns across a wide size range (small subcortical to large hemispheric), consistent with
§4's volume census.

Raw data stays at `/nfs/.../ANALYSIS_20251122/data/atlas_v2/` (`images/`, `masks/`,
`train.jsonl`); converted native-grid `.npy` at `paths.atlas_v2`
(`/nfs/.../data/atlas_v2/npy/`).

## 8. First eval result (exp92 checkpoint, single-level, 2026-09-13)

```
python experiments/3d/eval.py dataset=atlas_v2 eval.model=patchset3d \
  eval.checkpoint=.../3d_train/2026-09-11_92_multisource_synth/best.pt
```
**Mean Dice 0.0280 ± 0.078, NSD 0.0381, n=654** (wandb `stoic-voice-79`) — the WORST OOD-MRI
result of any source integrated this session (below ISLES22's 0.054 and Shifts-MS's 0.063; far
below MSD Hippocampus's 0.481 or MSD Prostate's 0.295).

Qualitatively (`figures/stroke_lesion_sub-r001s001.png`, dice=0.000): a **total empty-collapse**
— the model predicts essentially nothing for a large, unambiguous hemispheric lesion clearly
visible even to visual inspection, despite the K=1 context example showing a smaller but
clearly-marked lesion at a different location. This is a more SEVERE version of Shifts-MS's
near-empty-collapse failure mode (Shifts-MS at least produced small sparse fragments; here
there's effectively nothing).

Plausible confounds, not yet disentangled:
- **`crop_spacing_mm=1.9` vs. the checkpoint's trained cascade level-0 @ 6mm** — the same
  crop-pitch mismatch flagged for every other source this session, but here the effect looks
  qualitatively different (collapse, not over-segmentation) — worth comparing against a
  cascade eval to see if a coarser first pass at least achieves non-trivial recall before the
  fine level's precision-tuned behavior kicks in.
- **The provenance caveat (§1) may matter here too**: if this mirror's masks or images have
  systematic quality issues beyond the one integral-value failure already caught, this Dice
  number may partly reflect data quality rather than pure model generalization. Not
  investigated further given the provenance concern already flagged — this number should be
  read with that caveat attached, not treated as a clean generalization result the way
  ISLES22/Shifts-MS/Hippocampus/Prostate's numbers can be.
- Untested: cascade eval, `cascade_query_prior` ablations.
