# zanderch HU_Messung (`hu_lwk1`) — L1 vertebra HU-measurement cohort

Local NFS cohort, `/nfs/data/nii/data1/zanderch___HU_Messung/`, NOT a public download.
Inspected + integrated + eval'd 2026-09-14.

## 1. Scale & layout

687 top-level subject dirs, each with one or more visits:
`{subj}/{visit}/{series}/*.nii` (one or more raw CT series per visit — different
reconstruction kernels/phases, e.g. `Polytrauma_3_0_Bf39_3` vs `Polytrauma_2_0_Br59`),
plus `{visit}/preds/` (an existing external auto-segmentation pipeline's OWN outputs —
`HU_LWK1_pred.nii.gz`, `vertebrae_L1.nii.gz` — not GT, not read here) and `{visit}/DICOM/`.

Ground-truth label: `{visit}/HU_LWK1.nii.gz`, present for only **37 visits** (36 usable — see
§3). No multi-visit duplication beyond 2 subjects (`27441734`, `21711659`) that each have GT
at 2 separate visits — these are kept as independent cases (`{subj}_{visit}` id).

## 2. Image — whole-body polytrauma CT

Real HU-unit CT (uint16 on disk, header slope/intercept resolves to standard HU), one 3D
volume per series, always `(512, 512, D)` in-plane with highly variable `D` (189-789 slices)
— these are "polytrauma" trauma protocol scans, so scan length varies hugely (short
abdomen-only to full head-to-pelvis). In-plane spacing varies per case (0.645-0.977mm,
36-case range), Z-spacing is 3mm for 35/36 cases (one case at 2mm). This is the **first CT
(not MRI) native-grid eval source** added to this family (ISLES22/Shifts-MS/MSD/ATLAS/GNC are
all MRI) — same modality patchset3d actually trained on, unlike those.

A visit dir can hold **multiple** `.nii` series (different kernels or genuinely different
acquisitions); the label shares the exact shape+affine of exactly one of them, found by
matching rather than picked positionally (`scripts/convert_hu_lwk1.py:_find_image`).

## 3. Label — a small HU-measurement ROI, NOT a vertebra segmentation

`HU_LWK1.nii.gz` shares the **full image grid + affine** of its matching series (no ROI-crop
offset, unlike GNC — verified `affine_match=True` on all 36 usable cases after
`nib.as_closest_canonical` reorientation). Content is a small blob at the L1 vertebral body
centrum, placed there for an HU density measurement, not a vertebra outline:

| | min | max |
|---|---:|---:|
| voxel count | 309 | 2841 |
| max bbox extent (mm) | 11.0 | 23.4 |

Single binary class throughout (`unique_vals == [0, 1]` on every case) — no multi-class
overlap question like GNC, so no per-class-plane storage is needed; a single shared
`label.npy` (standard `NativeGridProvider` convention) is exact.

**Excluded**: `100999/NII/HU_LWK1.nii.gz` — this case has no image series anywhere under its
case dir (only the label + an old pipeline's `preds/`), a broken/incomplete case rather than a
conversion bug. 36/37 label files convert cleanly.

## 4. Label sparsity

36/687 subjects (5.2%) have this class — much less sparse than GNC (1.24%), but still a small
eval-only cohort (no train/val split attempted, no synth).

## 5. Conversion

`scripts/convert_hu_lwk1.py` (mirrors `scripts/convert_flare22.py`'s native-grid template —
plain single shared-array design, `int16` losslessness check, full 4x4 affine stored):

```
python scripts/convert_hu_lwk1.py --workers 16
```
Layout: `{subj}_{visit}/ct_raw.npy` (int16, native grid, HU) + `label.npy` (uint8 0/1) +
root `spacings.json`. 36/36 usable cases converted, 0 unexpected failures (only the 1
no-image case above, anticipated). Provider: `src/providers/hu_lwk1.py`'s `HuLwk1Provider` —
a **plain** `NativeGridProvider` subclass, no method overrides needed (unlike
`GncKidneyProvider`): single partitioning class, default `native_gt`/`load`/
`load_native_crop` are already correct.

## 6. Eval protocol — crop_spacing_mm / cascade_spacings, "find fitting ones"

The target here is small and near-point-like (max extent 11-23.4mm across all 36 cases) sitting
inside a huge, highly variable-length whole-body scan — the same "small target in a much
larger scene" framing as GNC/FLARE22/NasalSeg, not ISLES22/ATLAS's "whole lesion barely fits
the frame" framing (see `docs/datasets/gnc_kidney_lesions.md` §9 for that distinction). Per the
cascade mechanism (`experiments/3d/cascade.py`), the COARSEST level's crop is always centered
on the ground-truth centroid (`centers[0] = [None]*B` -> provider's own centroid-cache
fallback) — cascading here tests progressive-refinement quality given an already-known rough
location, not blind whole-body search, so spacing choice is about how much surrounding anatomy
each level shows, not about guaranteeing the crop contains the target from an uninformed start.

Chosen ladder **`[6, 3, 1]` mm** — reusing exp92's own trained coarse/mid points (`[6, 3,
1.5]`, `configs/experiment/3d/experiment/88_cascade.yaml`) unchanged (same rationale as GNC:
clipping was never the binding constraint at these pitches, so there's no reason to
extrapolate the model off spacings it was actually trained on), with only the **fine** point
corrected to this dataset's real measured target scale: 128 voxels @ 1mm = 128mm FOV, a ~5.5x
margin over the largest target (23.4mm) while staying close to native in-plane resolution
(0.645-0.977mm) — enough room to see the target vertebra's own shape (useful in-context
signal) without the frame being dominated by irrelevant unrelated anatomy. Single-level
baseline uses this same 1mm spacing, matching precedent (GNC/ISLES22 both baseline at their
cascade's finest point).

## 7. Single-level eval result (exp92 checkpoint, `crop_spacing_mm=1`, 2026-09-14)

```
python experiments/3d/eval.py dataset=hu_lwk1 eval.model=patchset3d \
  eval.checkpoint=.../3d_train/2026-09-11_92_multisource_synth/best.pt
```
**Mean Dice 0.1164 ± 0.082, NSD 0.1313, n=36** (wandb `hopeful-frog-85`), 884ms/sample,
3891 GFLOPs. Notably higher than GNC (0.0585) or ISLES22 (0.0544) / Shifts-MS (0.0632) — this
checkpoint's other MRI eval sources — consistent with `hu_lwk1` being CT (the modality it
actually trained on), even though `l1_center` itself is a completely novel class.

## 8. Cascade eval `[6, 3, 1]` mm (2026-09-14)

```
python experiments/3d/eval.py experiment=92_multisource_synth eval.model=patchset3d \
  eval.checkpoint=<ckpt> data.source=hu_lwk1 data.crop_spacing_mm=6 \
  data.cascade_spacings=[6,3,1] data.mask_downsample=occupancy data.gpu_realize_crop=false \
  data.val_classes=[l1_center] train.cascade_loss_weights=[1,1,1] eval.split=test \
  eval.cascade_figures=true
```
Ran clean on the first attempt (no crash) — the `native_gt`/`gt_loader` generalization built
for GNC (`docs/datasets/gnc_kidney_lesions.md` §9a) already covers a plain single-class source
like this one via `NativeGridProvider`'s default `native_gt`, with zero extra code needed here.

**Mean Dice 0.1287, NSD 0.0729, n=36**, 1538.5ms/sample, 11673.27 GFLOPs. Per-level stitched
dice: **6mm 0.0414 -> 3mm 0.0747 -> 1mm 0.1301**.

Unlike GNC/ISLES22 (where cascading made native-space Dice WORSE than single-level), the
finest cascade level here (0.1301) is essentially on par with — slightly better than — the
single-level baseline (0.1164): progressive refinement doesn't hurt on this source. Plausible
reason: same-modality CT (not an OOD MRI sequence) plus a well-defined, bony, structurally
"organ-like" landmark that patchset3d already has strong shape priors for (see below) is easier
to progressively re-center on than a diffuse/isointense lesion.

**Qualitative figure** (`figures/cascade/l1_center_3to1mm.png`,
`figures/cascade/l1_center_6to3mm.png`, subj `19927245_20220630115323`): the model's fine-level
prediction is a plausible, well-formed blob sitting on the correct vertebra, but it is shaped
like — and roughly the size of — a **whole vertebral body**, not the small measurement ROI the
GT actually marks. This is a genuine semantic mismatch, not a localization or pipeline
failure: patchset3d was trained on TotalSegmentator, which has whole-vertebra classes, so it
recognizes "this is a vertebra" from the in-context signal and produces a vertebra-shaped
segmentation — a sensible generalization, just not the object this eval class actually is.
That explains the moderate-not-collapsed Dice (~0.13): partial overlap from spatial proximity
(same vertebra), not true agreement on the target's actual extent.

## 9. Open questions (not blocking)

- Whether combining `l1_center` with a full-vertebra-segmentation class in the SAME in-context
  task (so the model has to disambiguate "small ROI" vs "whole organ" from the mask shape of
  the K context examples, rather than only ever seeing this one class) would reduce the
  systematic over-segmentation seen in §8 — untested.
- Only 36 cases total; no context-disjoint-from-target concern (unlike GNC's 1-subject
  classes) since there is only one class here and 36 >> 2, but the eval `n=36` is inherently
  small/noisy (see the ±0.082 Dice std).
