# ISLES 2022

*Ischemic Stroke Lesion Segmentation 2022 — multi-center MRI stroke lesion, Tier-2 "arbitrary
lesion" pick in `docs/datasets/eval_strategy_report.md`.* Downloaded and inspected 2026-09-12.
Not yet wired into the harness.

## 1. Access — fully open, zero friction

Zenodo, `access_right: open`, single archive, no account:
```
curl -LO "https://zenodo.org/records/7153326/files/ISLES-2022.zip?download=1"   # 1.69 GB
```
License: per Zenodo record (CC-style, see `LICENSE` in the archive). DOI
`10.5281/zenodo.7153326`.

## 2. Composition — BIDS layout, 250 subjects

`sub-strokecaseNNNN/ses-0001/{anat/*_FLAIR.nii.gz, dwi/*_dwi.nii.gz, dwi/*_adc.nii.gz}` +
`derivatives/sub-strokecaseNNNN/ses-0001/*_msk.nii.gz` (lesion mask) + `*_snp.png` (preview).
`participants.tsv` lists exactly **250** subjects; **all 250 have both image and mask files**
(no missing pairs) — this is the "public train" half of the full 400-case dataset (the other
150 are the hidden test set, per the report).

Three co-registered sequences per subject: **DWI (b=1000), ADC, FLAIR** — a genuine
multi-sequence-MRI source, distinct from every other dataset in the harness (all CT so far,
plus a single-sequence T1 for the not-yet-integrated MSD Hippocampus).

## 3. Geometry — VERIFIED, multi-vendor variability is real

Orientation **LAS** (uniform across all 250) — the mirror of the RAS convention our
converters (`convert_to_npy.py`, `convert_flare22.py`, `convert_nasalseg.py`) all normalize
to; a converter here needs the RAS flip step, same idea as NasalSeg's LPS→RAS but a different
axis pair (need to re-verify which axes flip for LAS→RAS specifically, don't assume it's the
same two axes as NasalSeg's LPS case).

| | min | median | max |
|---|---|---|---|
| shape (vox) | 112×112×25 | 112×112×72 | 256×256×76 |
| spacing (mm) | 0.88×0.88×2.0 | 2.0×2.0×2.0 | 2.0×2.0×5.0 |

Real multi-site heterogeneity (per the report's "multi-vendor variability" framing) — unlike
NasalSeg/Hippocampus's single-protocol uniformity, shape and spacing both vary meaningfully
case to case, closer to FLARE22's variability story but on MRI.

## 4. Labels — single lesion class, real prevalence spread

**3/250 subjects have an entirely empty lesion mask** (worth checking whether these are
genuine negative/near-total-recovery cases or GT gaps before using them — the report doesn't
flag this, so it's a new finding). Among the 247 with a lesion:

| | value |
|---|---|
| min lesion volume | 40 mm³ (near sub-voxel — a real small-lesion stress test) |
| median | 7168 mm³ |
| max | 482,152 mm³ (~482 mL — a massive stroke) |

This volume spread (4 orders of magnitude) is far wider than anything in TotalSeg/FLARE22/
NasalSeg's organ-class targets — exactly the "arbitrary lesion, no fixed shape/size prior"
argument the report leads with for Tier 2.

## 5. Fit as an eval set

- **Clean Tier-2 lesion argument**: a fixed-class model has no output channel for "stroke
  lesion"; in-context can define it from one example. No TotalSegmentator overlap at all
  (lesions aren't in its label set).
- **Multi-sequence input**: DWI/ADC/FLAIR gives 3 natural "which sequence is the context/target
  drawn from" ablations our K=1 in-context setup doesn't currently have a slot for (every
  existing provider is single-channel CT). Simplest first cut: pick one sequence (DWI is the
  acute-stroke-sensitive one) and treat this like any other single-channel MRI source.
- **The 3 empty-mask cases** need a decision before wiring: drop them (matches how we already
  drop degenerate cases elsewhere) or keep them as an explicit "no lesion present" probe.

## 6. Integration (implemented, 2026-09-12)

Wired as an eval-only v2 source, same shape as FLARE22/NasalSeg — plus one new piece:
**ISLES22 is the first native-grid MRI source**, so `NativeGridProvider` gained a
`MODALITY` class attribute (default `"ct"`, unchanged for FLARE22/NasalSeg) and a per-subject
normalization branch for `MODALITY="mri"`: `mri_stats`/`normalize_mri` (the same convention
`TotalSegProvider`'s totalsegmri branch already used) instead of the fixed global CT HU frame
— MRI intensity has no fixed clip/z-score window, so each subject needs its own stats,
computed at convert time and stored in a `ct_stats.json` sidecar.

```bash
python scripts/convert_isles22.py --workers 16          # BIDS .nii.gz -> native RAS .npy + per-subject mri_stats
python experiments/3d/eval.py dataset=isles22 eval.model=patchset3d eval.checkpoint=<ckpt>
```

| piece | file |
|---|---|
| converter (DWI only, LAS→RAS via `nib.as_closest_canonical`) | `scripts/convert_isles22.py` |
| provider | `src/providers/isles22.py` (`Isles22Provider`, `MODALITY="mri"`) |
| MRI-modality support (new) | `src/providers/native_grid.py` — `MODALITY`, `_load_ct_stats`, `_normalize_fn`/`_norm_spec` |
| config | `configs/experiment/3d/dataset/isles22.yaml` |
| dispatch | `experiments/3d/common.py` (`_source_root`, `build_dataset`, `make_eval_loader`, cascade-source allowlist), `experiments/3d/eval.py` |
| tests | `experiments/3d/tests/test_native_grid_provider.py` (MRI-modality cases) |

**The LAS→RAS reorientation is verified, not assumed**: unlike NasalSeg's NRRD (no valid
nibabel affine, needed hand-rolled axis math), ISLES22 is plain NIfTI with a real affine, so
`nib.as_closest_canonical` derives the correct flip directly — confirmed post-conversion via
`nib.aff2axcodes` on the written `spacings.json` affine (`('R','A','S')` for every subject).
Image/mask affine agreement was checked per-case at convert time (`np.allclose`, no silent
misalignment).

Converted: **250/250 cases, 0 failures** (3 with an all-zero lesion mask, matching the raw
BIDS release exactly — kept, not dropped; the centroid cache already excludes them from
`subjects_for("stroke_lesion")`). `ct_raw.npy` is float32 raw DWI intensity (not clipped/
z-scored — that happens at load time from `ct_stats.json`); `label.npy` uint8 0/1.

**`crop_spacing_mm: 1.5`** — round-trip grid-occupancy sweep (median subject FOV
224×224×144mm, largest 256×256×144mm): 100% in-plane / 75% z fill at 1.5mm for both, so even
the largest observed lesion (482mL, ~96mm sphere-equivalent) stays well inside a single crop.

Visual sanity check: `results/3d/isles22_items.png` (`plot_dataset_items.py dataset=isles22`)
— DWI windowing looks correct (bright acute-infarct signal, normal gyral/sulcal pattern), and
lesion masks track the hyperintense regions.

Raw BIDS source stays at `/nfs/.../ANALYSIS_20251122/data/isles22/ISLES-2022/`; converted
native-grid `.npy` at `paths.isles22` (`/nfs/.../data/isles22/npy/`).

## 7. First eval result (exp92 checkpoint, single-level, 2026-09-12)

```
python experiments/3d/eval.py dataset=isles22 eval.model=patchset3d \
  eval.checkpoint=.../3d_train/2026-09-11_92_multisource_synth/best.pt
```
**Mean Dice 0.0544 ± 0.108, NSD 0.0738, n=247** (wandb `dutiful-bird-74`). Near-collapse — a
stack of real confounds, not necessarily (or not only) "the model can't do this at all":
- `crop_spacing_mm=1.5` vs. the checkpoint's trained level-0 pitch of `6` (`_warn_uninherited_data`
  flags this automatically) — single-level, not the cascade the checkpoint was actually
  trained/optimized around.
- Genuinely unseen class (stroke lesion) on a genuinely different modality-regime (DWI brain
  MRI) than the checkpoint's training mix (mostly CT + whole-body-ish `totalsegmri`, not
  brain-focused).
- Qualitatively (`figures/stroke_lesion_sub-strokecase0001.png`, dice=0.053): the model isn't
  collapsed to empty — it predicts 2 plausible contiguous blobs roughly co-located with the
  true lesion region, but misses the GT's true multi-focal scattered pattern entirely. A
  shape/count mismatch, not a localization failure.

**Not yet isolated which confound dominates.** The natural next diagnostic is a matched
cascade eval (`data.cascade_spacings=[6,3,1.5]` or similar, mirroring the FLARE22/NasalSeg
cascade follow-ups) before reading much into this number either way.

## 8. Cascade eval `[3, 1.5]` mm — cascading does NOT help here (2026-09-12)

```
python experiments/3d/eval.py experiment=92_multisource_synth eval.model=patchset3d \
  eval.checkpoint=<exp92 best.pt> data.source=isles22 data.crop_spacing_mm=3 \
  data.cascade_spacings=[3,1.5] data.mask_downsample=occupancy data.gpu_realize_crop=false \
  data.val_classes=all train.cascade_loss_weights=[1,1] eval.split=test eval.cascade_figures=true
```
(`data.mask_downsample=occupancy` is a required override here — composing
`experiment=92_multisource_synth` drags in the training-time `mask_downsample: soft`, and the
FLARE22/NasalSeg/ISLES22 branch of `build_dataset` has no soft→occupancy eval remap the way
the totalseg-specific v2 loader branch does. Left alone, eval would silently score against a
soft partial-volume target.)

**Result: coarse (3mm) 0.0444, fine/stitched (1.5mm) 0.0460, NSD 0.0622, n=247** — slightly
*worse* than the §7 single-level baseline (0.0544 Dice, 0.0738 NSD), not better.

The cascade figure (`figures/cascade/stroke_lesion_3to1.5mm.png`) shows why: the coarse (3mm)
pass correctly flags "something's here" in roughly the right region, but the fine (1.5mm)
re-crop — now zoomed in and fed the coarse level's own imprecise prediction as a query prior
(`eval_mode=pred`, the default) — produces an even MORE diffuse, over-segmented blob than the
single-level baseline did, missing the true scattered multi-focal pattern just as badly. This
matches the structural concern flagged before running it: the cascade's centroid-of-mass
re-crop and prior-feedback mechanism assumes a single confined target (what it worked for on
FLARE22/NasalSeg); a multi-focal lesion has no single COM that means anything, and feeding the
coarse level's already-imprecise prediction forward as a prior compounds the error rather than
sharpening it.

**Conclusion for this checkpoint/dataset pair: prefer the single-level eval (§7) for reporting.**
Cascading is not a free win here the way it was for FLARE22/NasalSeg — it's actively
counterproductive for this failure mode. Untested: whether a gentler step (`[2, 1.5]` — 88-100%
coarse-level in-plane fill vs. `[3,1.5]`'s 59-66%, per the round-trip grid-occupancy sweep) or
`cascade_query_prior=none` (skip the prior-feedback entirely, keep only the re-crop) changes
this conclusion.
