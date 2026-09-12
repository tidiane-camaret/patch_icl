# MSD Task05_Prostate

*Medical Segmentation Decathlon, Task 5 — multi-parametric MRI (T2 + ADC), prostate
transitional/peripheral zone.* Downloaded and inspected 2026-09-12. Matches the report's
"prostate zones" entry in the genuinely-unseen-structures list (§a). Not yet wired into the
harness.

## 1. Access — fully open, zero friction

Same public S3 bucket as Hippocampus (`s3://msd-for-monai`, AWS Registry of Open Data), no
account:
```
curl -LO https://msd-for-monai.s3-us-west-2.amazonaws.com/Task05_Prostate.tar   # 240 MB
```
License CC-BY-SA 4.0 (`dataset.json`), reference Radboud University Nijmegen Medical Centre.

## 2. Composition — small pool

| split | n |
|---|---:|
| train (GT) | **32** |
| test (no GT) | 16 |

Only 64 (subject, class) train tasks at K=1 (32×2 zones) — the smallest pool of anything
inspected so far, well below NasalSeg's 535 or FLARE22's 650.

## 3. Modality — genuinely 4D, unlike everything else in the harness

`dataset.json` declares `"tensorImageSize": "4D"` and `"modality": {"0": "T2", "1": "ADC"}` —
each `imagesTr/*.nii.gz` is a **2-channel volume** (T2-weighted + ADC map stacked on a 4th
axis), not two separate files. Every provider currently in the harness (`TotalSegProvider`,
`NativeGridProvider`) is single-channel; this would be the first genuinely multi-modal-input
source, a different kind of integration work than a single extra `NativeGridProvider`
subclass (which channel(s) does the in-context image slot carry? both, one, or a config
choice?).

## 4. Geometry — VERIFIED

Orientation **RAS** (uniform, all 32). Anisotropic, high in-plane / thick-slice MRI, in the
same character class as FLARE22's anisotropy (just MR instead of CT):

| | min | median | max |
|---|---|---|---|
| shape (X,Y,Z) | 256×256×11 | 320×320×20 | 384×384×24 |
| spacing (mm) | 0.60×0.60×3.0 | 0.62×0.62×3.6 | 0.75×0.75×4.0 |

Very few slices (11–24) — a native FOV that's short in z relative to in-plane, similar in
spirit to FLARE22's abdomen-only z-truncation story, just far more extreme (z-spacing itself
is 3–4mm, vs FLARE22's 2.5mm, on top of only a handful of slices).

## 5. Labels

| label | present | median mm³ |
|---|---:|---:|
| 1 = PZ (peripheral zone) | 32/32 | 15,197 |
| 2 = TZ (transitional zone) | **30/32** | 48,047 |

TZ is absent in 2/32 cases (worth checking whether that's a real anatomical absence, unlikely,
or an annotation gap, before using those 2 cases for the TZ class). TZ volume ~3x PZ's median —
matches known prostate zonal anatomy (TZ enlarges with age/BPH).

## 6. Fit as an eval set

- **Genuinely unseen structures**: neither PZ nor TZ exists anywhere in TotalSegmentator's
  label set (only a single undifferentiated `prostate` region, itself only in a few subtasks) —
  a clean zonal-anatomy unseen-class claim, matching the report's own framing.
- **Small pool, multi-modal input**: 32 training cases and a 2-channel (T2+ADC) volume make
  this a heavier integration lift than NasalSeg/FLARE22 for comparatively little eval volume —
  lower priority to wire than ISLES22 or Hippocampus unless the multi-modal-input question
  itself is something worth building support for generally.

## 7. Integration (implemented, 2026-09-12)

**Design decision: split channels into separate tasks rather than add multi-channel input
support.** The converter treats each case's T2 and ADC channels as two independent
single-channel "subjects" sharing the same PZ/TZ label mask (`prostate_00_t2`,
`prostate_00_adc`) — structurally this is just another `NativeGridProvider` subclass, no new
architecture/dataloader work. **32 cases → 64 converted channel-subjects.** T2 and ADC have
wildly different intensity units anyway (confirmed: T2 raw range [0, 1486], ADC [0, 3619] on
the first case) — independent per-channel `mri_stats` (the same mechanism every other native
MRI source uses) handles this cleanly; there's no shared normalization frame to lose by
splitting.

Verified before committing to the split: both channels in each case's 4D file share **one**
affine (co-registered on one grid) — confirmed by wrapping each channel as its own 3D NIfTI
(`nib.Nifti1Image(data[...,c], affine)`) and round-tripping through `nib.as_closest_canonical`;
identical affine back out. So the split is purely a converter-time reshape, no
resampling/registration risk.

```bash
python scripts/convert_msd_prostate.py --workers 16     # .nii.gz -> native RAS .npy x2ch + mri_stats
python experiments/3d/eval.py dataset=msd_prostate eval.model=patchset3d eval.checkpoint=<ckpt>
```

| piece | file |
|---|---|
| converter (channel-split) | `scripts/convert_msd_prostate.py` |
| provider | `src/providers/msd_prostate.py` (`MsdProstateProvider`, `MODALITY="mri"`) |
| config | `configs/experiment/3d/dataset/msd_prostate.yaml` |
| dispatch | `experiments/3d/common.py`, `experiments/3d/eval.py` (same branches as the other 3 MRI sources) |

**`crop_spacing_mm=0.75`** — a DIFFERENT sweep methodology than ISLES22/Shifts-MS/Hippocampus.
Those three whole-native-volume sources sized the pitch to cover the entire native FOV/volume;
here the prostate is small relative to the native in-plane FOV (median 200×200mm vs. a
~53×45×56mm median label extent per class) — a find-in-scene geometry closer to FLARE22/
NasalSeg. Swept clip-avoidance directly against each (case, class) label's own centroid-relative
extent (62 instances across 32 cases × up to 2 classes): 0.75mm (FOV=96mm) is the smallest
pitch with **zero** instances clipped; 0.7mm already clips 1/62, 0.6mm clips 3/62. Mean organ
fill at 0.75mm ≈ 60%.

**Verified geometric caveat, not fixed**: 6/32 cases (19%) have a non-trivial affine SHEAR (up
to ~42% of the axis norm on Y/Z — gantry-tilted acquisition, common in pelvic MRI to avoid
hip-prosthesis artifacts). Like every converter in this harness, this one takes
`nib.affines.voxel_sizes` (column-norm) as the per-axis spacing and treats the grid as
axis-aligned for the downstream crop/resample — there is no oblique/shear-aware resampling
anywhere in this pipeline. For the sheared minority this introduces a genuine, if modest
(sub-voxel-to-few-mm at this FOV), geometric skew between the header's claimed grid and what a
naive axis-aligned crop extracts. Flagged rather than fixed — proper shear-aware resampling
would be a harness-wide change, out of scope for one dataset's converter.

**Converted: 64/64 channel-subjects from 32/32 cases, 0 failures.** Per-class subject counts
confirm the known TZ-absent-in-2-cases gap propagated correctly: `prostate_pz` 64/64
channel-subjects, `prostate_tz` 60/64 (both channels of the 2 TZ-absent cases correctly
excluded via the standard per-class-centroid-cache mechanism, no special-casing needed).

Visual sanity check: `results/3d/msd_prostate_items.png` — correct T2/ADC visual contrast per
channel-subject, well-localized zonal masks, context/target pairs freely mixing T2 and ADC
context (expected and fine — they're independent "subjects" by design).

Raw source stays at `/nfs/.../ANALYSIS_20251122/data/msd_prostate/Task05_Prostate/`; converted
native-grid `.npy` at `paths.msd_prostate` (`/nfs/.../data/msd_prostate/npy/`).

## 8. First eval result (exp92 checkpoint, single-level, 2026-09-12)

```
python experiments/3d/eval.py dataset=msd_prostate eval.model=patchset3d \
  eval.checkpoint=.../3d_train/2026-09-11_92_multisource_synth/best.pt
```
**Mean Dice 0.2946, NSD 0.3208, n=124** (wandb `likely-sun-78`) — second-best OOD-MRI result
of the four sources integrated this session (Hippocampus 0.481 > **this 0.295** > Shifts-MS
0.063 > ISLES22 0.054). Clear class split:
- `prostate_tz` (transitional zone, the larger/bulkier of the two, ~3x PZ's median volume):
  **0.404±0.176** (nsd 0.336), n=60.
- `prostate_pz` (peripheral zone, thin/crescent-shaped): **0.185±0.120** (nsd 0.305), n=64.

Qualitatively confirms a size/shape-driven gap, not a modality artifact: the TZ example
(`prostate_tz_prostate_00_adc.png`, dice=0.590) shows a roughly-correct, if fragmented,
compact blob; the PZ example (`prostate_pz_prostate_00_adc.png`, dice=0.169) shows the model
finding only small disconnected fragments of the true thin crescent — the same small/thin
-structure instability the eval-strategy report flags generally, and the same pattern
Hippocampus's own anterior>posterior gap hinted at more mildly.

Not yet run: a matched cascade eval; a T2-only vs. ADC-only breakdown (both channels are
pooled into one number above — worth splitting given ADC's very different contrast mechanism,
if this dataset gets revisited).
