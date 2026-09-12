# Shifts MS Lesion Segmentation (Part 2, openly-licensed cohorts)

*Shifts Challenge 2022 white-matter multiple sclerosis lesion dataset — MRI, no TotalSeg
overlap.* Downloaded and inspected 2026-09-12, sourced while following up on
`docs/datasets/eval_strategy_report.md`'s MS-lesion recommendation (the report names
MSLesSeg/MS3SEG/MSSEG generically; this is the specific one that's actually openly
downloadable without a DUA). Not yet wired into the harness.

## 1. Access — open for Part 2 only; Part 1 is DUA-gated

The full Shifts MS dataset has two halves on Zenodo:
- **Part 1** (`zenodo.org/records/7051658`, MSSEG-1/OFSEP cohorts, 52 cases) — released only
  under an **OFSEP Data Usage Agreement**, cannot be redistributed. **Gated.**
- **Part 2** (`zenodo.org/records/7051692`, `access_right: open`) — the CC-licensed cohorts
  (Best/ISBI-2015 + Ljubljana/PubMRI), **941 MB**, no login:
```
curl -LO "https://zenodo.org/records/7051692/files/shifts_ms_pt2.zip?download=1"
```

## 2. Composition — two cohorts, built-in domain shift

| cohort | split(s) | n |
|---|---|---:|
| `best` (ISBI 2015) | train / dev_in / eval_in | 10 / 2 / 9 = 21 |
| `ljubljana` (PubMRI) | dev_out | 25 |

**46 cases total.** The `_in`/`_out` split naming is deliberate: this dataset is *designed* as
a distributional-shift benchmark (per the Shifts paper) — `ljubljana` is the out-of-distribution
cohort relative to `best`'s training/dev/eval splits. That's a second, independent domain-shift
axis on top of whatever cross-dataset shift we'd already be measuring by evaluating on it.

Each case carries **6 co-registered volumes**: `t1`, `t2`, `flair`, `pd` (best cohort) /
`t1ce` (ljubljana cohort), `fg_mask` (brain foreground mask), `gt` (lesion mask), plus
`individual_annotators/` (multi-rater masks — useful for an inter-rater-variability read, not
just a single GT).

## 3. Geometry — VERIFIED, and the two cohorts disagree on orientation

| cohort | orientation | shape | spacing |
|---|---|---|---|
| `best` | **LAS** | 212×212×~152 | 1×1×1 mm |
| `ljubljana` | **LPS** | ~154-173×241×241 | 1×1×1 mm |

Both isotropic 1mm, but genuinely **different orientation conventions across cohorts** —
unlike every other source in the harness (each of which is internally orientation-uniform).
A converter here needs a per-cohort (not per-dataset) orientation fix, and — like NasalSeg's
LPS flip — should verify it empirically (e.g. via lesion-vs-`fg_mask` consistency) rather than
trust the header blindly.

## 4. Labels — dense, real lesion burden, no empty masks

All 46 cases have a non-empty lesion mask (unlike ISLES22's 3/250 empty cases). Median lesion
volume 6300–18300 mm³ depending on split/cohort (a few to ~2 mL) — smaller and less spread than
ISLES22's stroke lesions (40mm³–482mL), consistent with MS plaques vs. acute stroke infarcts
being a different lesion-size regime.

## 5. Fit as an eval set

- **Genuinely OOD class, confirmed against our OWN training vocabulary, not just
  TotalSegmentator's public tool**: checked `data/totalseg_classes.py` directly (the class
  registry this project's models actually train against) — no hippocampus, no prostate zones,
  no lesion/stroke/MS class of any kind anywhere in either the CT or MR class lists (only
  undifferentiated whole-organ entries like `"brain"`, `"prostate"`). This holds for all four
  MRI datasets pulled this session (Hippocampus, ISLES22, MSD Prostate, this one) — none of
  them share a single class with what the checkpoints in this repo have ever been trained on.
- **Multi-sequence + multi-rater**: 6 volumes/case plus per-annotator masks is richer than
  ISLES22's 3-sequence setup — useful if we ever want a "does the model's uncertainty track
  inter-rater disagreement" probe, not just a point Dice.
- **Small pool**: 46 cases (21 in-distribution-ish, 25 explicitly out-of-distribution) is the
  smallest of the four MRI pulls — fine for a first eval pass, thin for anything requiring a
  seen/unseen macro split within the dataset itself.

## 6. Integration (implemented, 2026-09-12)

Wired as an eval-only v2 source, same shape as ISLES22 (both are `NativeGridProvider`
`MODALITY="mri"` sources — no new code needed in `native_grid.py`, that piece was already
built for ISLES22).

```bash
python scripts/convert_shifts_ms.py --workers 16       # .nii.gz -> native RAS .npy + mri_stats
python experiments/3d/eval.py dataset=shifts_ms eval.model=patchset3d eval.checkpoint=<ckpt>
```

| piece | file |
|---|---|
| converter (FLAIR only, per-cohort orientation) | `scripts/convert_shifts_ms.py` |
| provider | `src/providers/shifts_ms.py` (`ShiftsMsProvider`, `MODALITY="mri"`) |
| config | `configs/experiment/3d/dataset/shifts_ms.yaml` |
| dispatch | `experiments/3d/common.py`, `experiments/3d/eval.py` (same branches ISLES22 uses) |

**Two things this converter had to handle that ISLES22 didn't:**
1. **Non-unique subject IDs across splits.** `best/train/25` and `best/eval_in/25` are
   different patients (verified: different shape 153 vs 151 slices, different SHA content)
   despite sharing the numeral "25" — subjects are namespaced `{cohort}_{split}_{id}`
   (`best_train_0025`, `best_eval_in_0025`, ...).
2. **Per-cohort orientation, handled generically.** `best`=LAS, `ljubljana`=LPS (verified per
   file via `aff2axcodes`, not assumed) — `nib.as_closest_canonical` derives the correct flip
   for either automatically since both are plain NIfTI with real affines (no NasalSeg-style
   manual axis math needed). Confirmed post-conversion: `('R','A','S')` for all 46 subjects.
   Also found: `ljubljana`'s flair/gt affines disagree by up to ~0.45mm across all 25 cases (a
   sub-voxel origin-rounding artifact from how that cohort was independently resampled, `best`
   has ~1e-8mm agreement) — real but benign, so the converter's affine-agreement check uses a
   1mm tolerance for this source instead of ISLES22's tighter one.

Converted: **46/46 cases, 0 failures**, 0 empty lesion masks (this cohort's masks are always
non-empty, unlike ISLES22's 3/250). `ct_raw.npy` float32 raw FLAIR (unnormalized — that
happens at load time from `ct_stats.json`); `label.npy` uint8 0/1.

**`crop_spacing_mm: 1.5`** — same reasoning and same value as ISLES22 (round-trip
grid-occupancy sweep: 100% in-plane fill at both median and largest-FOV subjects, 80-90% on
the shortest axis).

Visual sanity check: `results/3d/shifts_ms_items.png` — both cohorts render correctly, and the
periventricular lesion distribution (clustered around the ventricles) is visibly the classic
MS pattern, a good sign the mask/orientation pipeline is right.

No new tests needed: the `MODALITY="mri"` code path is shared with ISLES22 and already covered
by `test_native_grid_provider.py`'s MRI-modality cases.

Raw source stays at `/nfs/.../ANALYSIS_20251122/data/shifts_ms/shifts_ms_pt2/`; converted
native-grid `.npy` at `paths.shifts_ms` (`/nfs/.../data/shifts_ms/npy/`).

## 7. First eval result (exp92 checkpoint, single-level, 2026-09-12)

```
python experiments/3d/eval.py dataset=shifts_ms eval.model=patchset3d \
  eval.checkpoint=.../3d_train/2026-09-11_92_multisource_synth/best.pt
```
**Mean Dice 0.0632 ± 0.056, NSD 0.1649, n=46** (wandb `winter-terrain-76`). Same ballpark as
ISLES22 (0.0544 Dice) but a NOTABLY different failure signature:
- **Much tighter std** (0.056 vs. ISLES22's 0.108) — more uniformly poor across subjects, not
  a mix of near-misses and outright failures.
- **Higher NSD relative to Dice** (0.165 vs. ISLES22's 0.074, despite a LOWER Dice) — NSD only
  measures boundary proximity, so this is consistent with the model producing very SMALL/
  sparse predictions (a small blob close to a true small lesion scores non-trivial NSD but
  near-zero Dice on the volumetric overlap) rather than ISLES22's large-diffuse-blob
  overshoot.
- Qualitatively (`figures/ms_lesion_best_dev_in_0006.png`, dice=0.089): the "pred" panel is
  visibly near-EMPTY on this slice — a near-collapse-to-background pattern, the opposite
  failure mode from ISLES22's over-segmentation.

Not yet run: a matched cascade eval (same caveat as ISLES22 applies even more here — MS
lesions are frequently even more numerous/scattered per subject than ISLES22's stroke
lesions, so the COM-recrop mismatch is likely at least as bad).
