# ACDC (Automated Cardiac Diagnosis Challenge)

*Cardiac cine-MRI, MICCAI 2017 — left/right ventricle + myocardium segmentation.* Downloaded
and inspected 2026-09-12, following up on `docs/datasets/eval_strategy_report.md`'s IRIS/M&Ms
domain-shift entries. **Genuinely OOD class, verified against our own training vocabulary**:
`data/totalseg_classes.py` only has an undifferentiated `heart` region (and `TS_SET_CARDIAC`
adds only vessels — aorta, pulmonary vein, brachiocephalic trunk, subclavian artery — no
chambers or myocardium at all), so LV/RV/myocardium substructures are unseen. Non-CT modality
too. Not yet wired into the harness — characterization only.

## 1. Access — genuinely open, no registration despite the challenge being "closed"

The official challenge page (`creatis.insa-lyon.fr/Challenge/acdc`) states the challenge is
closed but the data + full ground truth remain public via a Girder data portal
(`humanheart-project.creatis.insa-lyon.fr/database`), **no login** — confirmed via the portal's
own REST API (`public: true` on the collection). Scriptable:
```bash
curl -L -o training.zip "https://humanheart-project.creatis.insa-lyon.fr/database/api/v1/folder/63721d7073e9f0047faa0525/download"
curl -L -o testing.zip  "https://humanheart-project.creatis.insa-lyon.fr/database/api/v1/folder/6372203a73e9f0047faa117e/download"
```
(folder IDs found by walking the Girder API tree: collection → `database` folder →
`training`/`testing` subfolders.) Only obligation is a mandatory citation (Bernard et al.,
IEEE TMI 2018) shipped in every patient folder — no CC/DUA license field at all, but no
redistribution restriction either.

## 2. Composition — small, clean, perfectly balanced by pathology

**150 patients total** (100 in `training/`, 50 in `testing/` — note both splits ship full GT
here, unlike the original challenge's held-out test; this download is the post-challenge public
release). Verified: exactly **30 patients per pathology group × 5 groups** (NOR healthy, MINF
prior myocardial infarction, DCM dilated cardiomyopathy, HCM hypertrophic cardiomyopathy, RV
abnormal right ventricle) — a genuinely balanced design, unlike every dataset pulled so far
this session.

Per patient: a full `_4d.nii.gz` cine series (no GT) plus two single-frame volumes at
end-diastole (ED) and end-systole (ES) — the only two frames with GT (`_gt.nii.gz`).
`Info.cfg` gives the ED/ES frame indices, pathology group, height/weight. **300 GT volumes**
(150 patients × 2 phases) is the actual (subject, phase) pool size for a segmentation task.

## 3. Geometry — VERIFIED, thick-slice, small in scale like MSD Prostate

All 300 GT volumes: orientation **LPS** uniformly (needs `nib.as_closest_canonical`, unlike
the RAS-native MSD sources).

| | shape (vox), min/median/max | spacing (mm), min/median/max |
|---|---|---|
| GT volumes | 154×154×6 / 216×256×9.5 / 428×512×21 | 0.7×0.7×5.0 / 1.52×1.52×10.0 / 1.95×1.95×10.0 |

Very thick slices (5-10mm, median 10mm) and few of them (6-21, median ~9.5) — a short-axis
cardiac stack, same character-class problem as MSD Prostate's few-thick-slices geometry, just
more extreme on z (10mm here vs. 3-4mm there).

**Grid-occupancy sweep** (T=128, all 300×3-class label instances, clip-avoidance against each
class's own centroid-relative extent — same methodology as MSD Prostate, since the heart is
small relative to the in-plane FOV): **`crop_spacing_mm=1.2`** is the smallest pitch with zero
of 900 (volume, class) instances clipped (1.0mm already clips 29/900).

## 4. Labels — 3 classes, universal presence

| label | present | meaning |
|---|---:|---|
| 1 (RV) | 300/300 (100%) | right ventricle cavity |
| 2 (myocardium) | 300/300 (100%) | LV myocardial wall |
| 3 (LV) | 300/300 (100%) | left ventricle cavity |

No absence gaps at all — every patient has all 3 structures at both ED and ES, unlike every
other source pulled this session (which all had at least one organ with real absence).

## 5. Intensity — raw scanner units, verified NOT normalized

int16, per-subject range varies substantially (e.g. [0, 658] on one patient, [2, 255] on
another) — needs the same per-subject `mri_stats`/`normalize_mri` mechanism as every other MRI
source in the harness (ISLES22/Shifts-MS/Hippocampus/Prostate), not a fixed frame.

## 6. Fit as an eval set

- **Clean OOD-class claim**, verified directly against our own class registry (not just
  TotalSegmentator's public label list) — no heart chamber/myocardium substructure anywhere.
- **Perfectly balanced by pathology** (30/group × 5) is a genuinely useful property this
  session's other sources lack — a natural axis for a "does performance vary by disease
  subtype" breakdown (e.g. does the model do worse on HCM's thickened myocardium or DCM's
  enlarged chambers than on NOR?).
- **Small pool but 2 phases/patient**: 150 patients × 2 phases (ED/ES) × 3 classes = up to 900
  (subject, class) tasks at K=1 if ED and ES are treated as separate "subjects" (same trick as
  MSD Prostate's channel split, applied to cardiac phase instead of MR sequence) — reasonable
  middle ground between NasalSeg (535) and MSD Hippocampus (520 before the split trick, more
  after).
- **Established, comparable protocol**: ACDC is one of the most widely used cardiac
  segmentation benchmarks in the literature (IRIS's own domain-shift suite uses it per the
  report) — an eval number here is directly comparable to a large body of published work, more
  so than any bespoke source in this harness.

## 7. Integration — NOT YET DONE

No provider/converter/config yet. Structurally straightforward — another `NativeGridProvider`
subclass (`MODALITY="mri"`), same shape as every MRI source integrated this session. Two design
choices to make explicit when wiring:
1. **ED vs. ES as separate subjects?** Mirrors MSD Prostate's T2/ADC channel-split precedent —
   treating each phase as its own single-frame "subject" (`patient001_ed`, `patient001_es`)
   avoids ever needing the `_4d.nii.gz` series at all (only ED/ES have GT anyway).
2. **`crop_spacing_mm=1.2`** per §3's sweep.

Extracted to a scratch directory for this characterization pass, not yet staged as a permanent
converted dataset. Raw zips kept at `/nfs/.../ANALYSIS_20251122/data/acdc/` (`training.zip`
850MB, `testing.zip` 809MB compressed → ~2.3GB extracted).
