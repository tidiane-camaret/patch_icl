# FLARE22 — dataset characterisation and fit as a patchset eval set

*MICCAI FLARE 2022: "Fast, Low-resource, and Accurate oRgan segmentation" — semi-supervised
abdominal multi-organ segmentation in CT.* Explored 2026-08-25; not yet downloaded to NFS.

Sources: [challenge dataset page](https://flare22.grand-challenge.org/Dataset/) ·
[Zenodo 50-labeled train](https://zenodo.org/records/7860267) ·
[challenge paper, Lancet Digital Health 2024 / arXiv:2308.05862](https://arxiv.org/abs/2308.05862) ·
[FLARE-MedFM on HuggingFace](https://huggingface.co/datasets/FLARE-MedFM/FLARE-Task3-DomainAdaption)

## 1. What it is

Semi-supervised challenge: **50 labelled** + **2000 unlabelled** CT scans for training, targeting
**13 abdominal organs**. The point of the challenge was to exploit the unlabelled pool under a
compute/latency budget; for us only the *labelled* portion matters. Aggregated from **>50 medical
groups**, intercontinental (North American / European / Asian test cohorts), pan-cancer, mixed
phases, mixed vendors — the dataset's headline property is **centre and scanner diversity**, not
label novelty.

## 2. Composition

| Split | N | Labels public? | Notes |
|---|---|---|---|
| Train (labelled) | **50** | **yes** (Zenodo, CC-BY-4.0) | pancreas-disease cases; images + pancreas GT from MSD Task07, other 12 organs from AbdomenCT-1K |
| Train (unlabelled) | 2000 | no GT by design | liver/kidney/spleen/pancreas disease |
| Validation ("tuning") | 50 | yes, released post-challenge (grand-challenge / FLARE-MedFM) | liver/kidney/spleen/pancreas disease |
| Test | 200 | **no** — hidden, leaderboard only | + three external cohorts (NA / EU / Asia) |

**Practical eval pool for us: the 50 labelled train cases (+ 50 validation if we grab them) = 50–100
volumes × 13 organs.** That is 650–1300 (subject, class) eval tasks — comfortably more than our
current `eval.n_subjects=50` totalseg runs. Medverse uses exactly 50 FLARE22 cases as its
"unseen centre" held-out split (k=4, 8 repeated context samplings, Dice).

## 3. Labels — VERIFIED on disk

`FLARE22Train/{images,labels}` (nnU-Net layout): `images/FLARE22_Tr_%04d_0000.nii.gz`,
`labels/FLARE22_Tr_%04d.nii.gz`, cases 0001–0050. Labels `uint16`, image and label share shape and
affine in all 50 cases, orientation **RAS** throughout.

Index order confirmed (volumes match the organ, and laterality checked on the RAS x-axis —
id2 lateral to id13 in **50/50** cases, id7 lateral to id8 in **50/50**):

| id | organ | median vol (ml) | range | median z-extent | FOV-truncated |
|---:|---|---:|---|---:|---:|
| 1 | liver | 1566 | 1037–2571 | 168 mm | 5/50 |
| 2 | kidney_right | 193 | 108–420 | 106 mm | 3/50 |
| 3 | spleen | 190 | 52–694 | 82 mm | 0/50 |
| 4 | pancreas | 88 | 42–178 | 84 mm | 0/50 |
| 5 | aorta | 93 | 41–187 | 226 mm | **50/50** |
| 6 | inferior_vena_cava | 85 | 41–128 | 208 mm | 27/50 |
| 7 | adrenal_gland_right | 4.3 | 1.4–9.4 | 38 mm | 0/50 |
| 8 | adrenal_gland_left | 5.5 | 2.4–9.8 | 42 mm | 0/50 |
| 9 | gallbladder | 27.8 | 4.3–117 | 44 mm | 1/50 |
| 10 | esophagus | 15.7 | 7.0–25.8 | 70 mm | **50/50** |
| 11 | stomach | 318 | 140–831 | 92 mm | 0/50 |
| 12 | duodenum | 76 | 44–120 | 92 mm | 1/50 |
| 13 | kidney_left | 196 | 103–381 | 110 mm | 2/50 |

**All 13 organs are present in all 50 cases** — dense annotation, 650 (subject, class) eval tasks
from the labelled set alone, and every case can serve as a context for every class.

## 4. Image characteristics — MEASURED (50 labelled cases)

| | FLARE22 (native) | our TotalSegmentator copy |
|---|---|---|
| in-plane shape | **512×512, all 50** | median 241×231 |
| in-plane spacing | 0.645–0.977 mm (med **0.80**) | **1.5 mm** |
| slice spacing | **2.5 mm** ×47, 4.0 ×1, 5.0 ×2 | **1.5 mm** |
| anisotropy (max/min) | **3.14×** | **1.0× (isotropic, all 1228 subjects)** |
| z slices | 71–113 (med 97) | med 231 |
| FOV | 407×407×**243** mm (abdomen only) | 362×347×347 mm, up to 725 mm in z |
| voxel volume | 1.65 mm³ | 3.375 mm³ |
| HU | min −1024 everywhere, p99 ≈ 270, max 1308–3071 | — |
| orientation | RAS, uniform | — |

The single most important row is **anisotropy**. Our TotalSegmentator copy was resampled to 1.5 mm
isotropic for *every* subject, so the model has seen **zero variance in acquisition anisotropy**;
FLARE22 is natively 3.1× anisotropic. The `use_crop` path resamples a physical crop to
`crop_spacing_mm`, so FLARE22 arrives *downsampled* in-plane (0.8 → 1.5) and *interpolated up* in z
(2.5 → 1.5, ×1.67). Whatever we measure on FLARE22 includes that resampling signature, and no
TotalSegmentator eval can separate it. Sweeping `crop_spacing_mm` over [1.5, 2.5, 4.0] is therefore
not optional — 2.5 mm is the only setting where z is not interpolated.

Secondary: abdomen-only z FOV (243 mm median vs 347 mm) means aorta and esophagus are cut by the
volume border in **50/50** cases and IVC in 27/50 — those three classes are *segments*, not whole
structures, and are not volume-comparable to TotalSegmentator.

## 5. Metrics & reference numbers

Challenge metric: **DSC + NSD** (surface Dice), plus runtime/GPU-memory efficiency scores (irrelevant
to us). Top methods: median DSC **90.0%** on the main test set; 89.5 / 90.9 / 88.3% on the
North American / European / Asian external cohorts. Our `eval.nsd_tolerance_mm: 3.0` matches the
usual convention, so numbers are broadly comparable — but note challenge scores come from
*fully-supervised whole-volume* models, not one-shot in-context, so treat them as a ceiling, not a baseline.

## 6. Fit for patchset — the important part

**Overlap check against our training pool** (`data/totalseg_classes.py`, `train_classes=not_benchmark`):

| FLARE22 organ | in TotalSegmentator | held out of our training (`BENCHMARK_CLASSES`) |
|---|---|---|
| liver, spleen, pancreas, kidney L/R, stomach, gallbladder, esophagus, aorta, IVC, adrenal_gland_left | yes | **yes — clean** |
| **adrenal_gland_right** | yes | **no — trained on** |
| **duodenum** | yes | **no — trained on** |

So 11/13 organs are already class-held-out for us; **adrenal_gland_right and duodenum are seen
classes** and must be reported separately (or dropped) or the "unseen" framing breaks. All 13 are
inside TotalSegmentator's label set, so FLARE22 is **not** an unseen-class benchmark for anyone —
its value is a different axis.

**What claim FLARE22 actually supports:** *unseen centre / acquisition shift on known anatomy.* Same
organs, different hospitals, different scanners, different disease burden, different annotation
protocol (AbdomenCT-1K conventions, not TotalSegmentator's). That is a real generalisation test and
it is the exact role Medverse assigns it — which is the strongest argument for including it: it
gives us a **head-to-head number against our closest 3D-ICL baseline on its own held-out split**.

**What it does not do:** it does not answer the "why not just train one supervised model" critique
(see `eval_strategy_report.md` — a fixed-class nnU-Net has an output channel for every one of these
13 organs, and FLARE22's own leaderboard shows it scores 0.90). Use it as the *comparability*
benchmark, and pair it with the unseen-class / lesion / non-human sets for the structural argument.

**Secondary value — label-definition drift, and it is MEASURED, not hypothetical.**
Median organ volume, FLARE22 vs our TotalSegmentator copy. To keep FOV out of it, the
TotalSegmentator column excludes subjects where the mask touches the volume border
(`scratchpad/ts_vols.py`, 1228 subjects, volumes at 1.5 mm iso):

| organ | TS untruncated (ml) | FLARE22 (ml) | ratio |
|---|---:|---:|---:|
| liver | 1517 | 1566 | **1.03** |
| spleen | 198 | 190 | **0.96** |
| stomach | 271 | 318 | 1.17 |
| adrenal_gland_right | 3.7 | 4.3 | 1.18 |
| inferior_vena_cava | 75 | 85 | 1.14 |
| pancreas | 68 | 88 | **1.30** |
| kidney_left | 149 | 196 | **1.31** |
| adrenal_gland_left | 4.2 | 5.5 | 1.31 |
| gallbladder | 21 | 28 | 1.32 |
| kidney_right | 144 | 193 | **1.34** |
| duodenum | 53 | 76 | **1.43** |
| aorta | 236 | 93 | 0.39 — FOV, ignore |
| esophagus | 34 | 16 | 0.46 — FOV, ignore |

Liver and spleen agree to within 4%, so this is not a global scale or resampling artefact. Seven
compact, fully-contained organs are **systematically ~30% larger** in FLARE22. Partial volume at
2.5 mm slices accounts for only a few percent on a 100 mm-tall kidney, so the bulk is annotation
convention (AbdomenCT-1K vs TotalSegmentator boundaries — hilum, perirenal fat, vessels) plus the
pancreas-cancer cohort. Aorta/esophagus ratios are pure FOV truncation and mean nothing.

**Consequence for reading FLARE22 Dice.** A systematic volume ratio *r* caps Dice at `2/(1+r)` even
for a perfectly-shaped nested prediction: **0.92 at r=1.17, 0.87 at r=1.30, 0.82 at r=1.43.** A
TotalSegmentator-convention model therefore cannot score above ~0.87 on FLARE22 kidneys/pancreas no
matter how good it is. Absolute FLARE22 Dice is only interpretable against that ceiling.

This is also the sharpest ICL-favourable experiment the dataset offers: an in-context model can in
principle *read the convention off the context mask*, where a fixed supervised model cannot. Three
conditions, one dataset:
1. patchset with **FLARE22 contexts** (can it adopt the FLARE22 convention?),
2. patchset with **TotalSegmentator contexts** on FLARE22 targets (does it stay on the old convention?),
3. `eval.model=totalsegmentator` (Route B, context-free nnU-Net) — measures the convention+domain gap
   for a model that is definitionally locked to TotalSegmentator's boundaries, and gives the
   reference the other two are read against.

If (1) > (2) ≈ (3), that is a clean, quantified demonstration of what in-context buys — and it is
independent of the "unseen class" argument entirely.

## 7. Integration (implemented)

Wired as an eval-only v2 source. One-off conversion, then it is a `dataset=` override:

```bash
python scripts/convert_flare22.py --workers 16          # nii.gz -> native-grid .npy
python experiments/3d/eval.py dataset=flare22 eval.model=patchset3d
python experiments/3d/eval.py dataset=flare22 eval.model=patchset3d data.crop_spacing_mm=1.5
```

| piece | file |
|---|---|
| converter (native grid only) | `scripts/convert_flare22.py` |
| provider (crop + resample at load) | `src/providers/flare22.py` |
| config | `configs/experiment/3d/dataset/flare22.yaml` |
| dispatch | `experiments/3d/common.py` (`_source_root`, `build_dataset`, eval loader), `experiments/3d/eval.py` |

**Storage is native and lossless.** `ct_raw.npy` int16 + `label.npy` uint8 + root
`spacings.json` (spacing, shape, **full 4x4 affine**), 2.0 GB for 50 cases. int16 is
bit-exact because the CT is integral with a global range of exactly [-1024, 3071] — the
converter enforces this per case rather than assuming it (50/50 passed). float16 would
NOT be lossless (it cannot represent odd integers > 2048). The affine's translation is
what lets a prediction be written back into the source NIfTI frame.

**All resampling is deferred to the dataloader**, so `crop_spacing_mm` is a config knob,
not a property of the conversion. `Flare22Provider` reuses `crop_and_place`, so crop
geometry is computed in exactly one place across all 3D sources.

Two defaults differ from the totalseg provider, both because the native grid is finer and
anisotropic (on 1.5mm-isotropic totalseg the resample was an identity, which masked both):

* `image_antialias=true` — `F.interpolate` has no 3D `antialias`, so plain trilinear
  point-samples and aliases under 2–4x in-plane decimation. `place_image(..., antialias=True)`
  area-prefilters only the axes being downsampled (so the z UPsample at 1.5mm is unaffected).
  Measured effect on real crops: 4–12% of voxels change.
* `mask_occupancy_thr=0.5` — the v2 totalseg default of 0.1 dilates thin organs (adrenal GT
  volume ~1.7x vs ~1.08x at 0.5) and round-trips worse on every organ.

**`crop_spacing_mm=2.5` is the default**, not 1.5. At 2.5 the crop pitch equals the native z
spacing, so the z resample is an exact identity (confirmed in `crop_geom`: crop_sizes[2] ==
out_sizes[2] == 110), and 128x2.5 = 320 mm truncates nothing. At 1.5 both axes resample and
the FOV clips GT (liver 3.9%, IVC 9.1%, aorta 14.5% of GT voxels fall outside the crop).

### Metrics caveat — still open

`experiments/3d/evaluate.py` computes Dice/NSD on the 128^3 crop grid against the
**resampled** label. That is not FLARE22's metric (native-voxel DSC/NSD) and is not
comparable to published FLARE22 numbers. Round-trip Dice of a *perfect* predictor, i.e.
the native-space ceiling while crop-space scoring reports 1.000
(`experiments/flare22/gt_fidelity.py`, all 50 cases, occ_thr=0.5):

| organ | pitch 1.5 | pitch 2.5 | | organ | pitch 1.5 | pitch 2.5 |
|---|---:|---:|---|---|---:|---:|
| liver | 0.953 | 0.984 | | gallbladder | 0.914 | 0.948 |
| kidney_R / _L | 0.952 / 0.954 | 0.964 / 0.963 | | esophagus | 0.901 | 0.916 |
| spleen | 0.962 | 0.971 | | IVC | 0.902 | 0.938 |
| stomach | 0.952 | 0.974 | | aorta | 0.869 | 0.941 |
| pancreas | 0.925 | 0.951 | | adrenal_R / _L | 0.793 / 0.801 | 0.832 / 0.851 |
| duodenum | 0.923 | 0.940 | | | | |

Crop-space Dice is inflated by 2-21 points, organ-dependent, so it reorders organs.
`crop_geom` (starts, crop_sizes, out_sizes, pad_lo) already inverts the crop exactly and
`Flare22Provider.native_meta(subject)` supplies the native shape/spacing/affine, so
native-space scoring needs no new geometry bookkeeping — only a change in `evaluate.py`.
`nsd_batch` already accepts per-axis spacing; it is currently fed the crop's isotropic
pitch, so the mm tolerance is measured in the wrong physical units.

Also unresolved: whether FLARE22's official NSD uses organ-specific tolerances rather than
our flat `nsd_tolerance_mm: 3.0`; and that the GT-centroid crop is **oracle localization**,
a protocol choice to declare (challenge submissions segment the whole volume).

Not pulled: the 50 validation cases (doubles the pool, needs grand-challenge registration).
