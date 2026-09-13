# GNC_705 Kidney Lesions (German National Cohort)

*Kidney lesion segmentation (hyperdense / hypodense / complex, +cystic components), 4-channel
Dixon MRI.* Inspected 2026-09-13 on user request ("inspect directory ... find a smaller,
balanced, label-rich subset"). **Local restricted-access cohort data, not a public download**:
lives at `/nfs/data/nii/data0/GNC/GNC_705/` — no access/provenance question applies the way it
did for the HF-mirror sources in `eval_expansion_status.md`; this is simply a large local
dataset that hasn't been touched before this session. Characterization + subset selection only
— no converter/provider written yet (matches this session's established pattern: characterize
first, integrate on explicit "integrate X" instruction).

## 1. Scale and layout

**30,385 subjects, 49,220 total (subject, visit) cases** (matches the user's "~49255 cases"
figure). Two parallel trees, joined by subject ID and visit code:

- `data/{subject}/{visit}/*.nii.gz` — hand-drawn lesion ROI masks, direct children of the visit
  dir (deeper subfolders `ML/`, `CYST/`, `META/`, `T2ROI/`, `results/` hold derived/intermediate
  artifacts, not raw labels — excluded from the census by depth).
- `links/{subject}/{visit}/{series_name}/*.nii` — **symlinks** (`{visit}` itself is a symlink to
  `../../../GNC_parent/<hash>/<timestamp>/`) into the raw per-sequence MRI series. `find` does
  not follow a symlink given as an explicit path component unless told to (`-L`/`-H`) — plain
  `ls`/`os.listdir` do follow it; this tripped up the first exploration pass (an empty-looking
  `find -maxdepth 2` on a visit dir) before switching to `os.listdir`.

Visit codes are overwhelmingly `"30"` (30,025) and `"60"` (18,823) — almost certainly two
imaging rounds of the cohort's re-contact design — plus a handful of oddities: `"20"`/`"25"`
(359 total) and a few visit dirs literally named as another subject's ID (`"100040"`, etc., 1
each) — legacy/misfiled data, negligible (<0.01%), not investigated further.

## 2. Image: single 4-channel Dixon MRI file per visit

Each visit has 4 Dixon acquisition **stations** (`3D_GRE_TRA_1..4`, each with `_opp`/`_in`/`_F`
(fat)/`_W` (water) — presumably different Z-coverage slabs), but **station 4 additionally ships
a pre-composed, already-4-channel file**: `3D_GRE_TRA_W_COMPOSED_-<n>_s0NN.nii`, verified shape
**(320, 260, 316, 4)**, spacing **1.40625 × 1.40625 × 3.0mm**, last axis = [opp, in, fat, water]
matching the user's description exactly. This is the natural single image source per visit — no
need to touch stations 1-3 or hand-stitch anything.

**100% coverage verified**: every one of the 610 labeled (subject, visit) pairs (§4) has a
matching `*COMPOSED*.nii` file under some `3D_GRE_TRA_*` station dir. No missing-image risk.

## 3. Labels are tight ROI crops, not full-volume masks — verified affine-diff offset

A label file's shape (e.g. `Hyper_R.nii.gz`: 64×64×32) is much smaller than the full image grid
(320×260×316) — **it's a bounding-box crop around the lesion**, not a full-volume mask, but it
shares the exact same voxel spacing and axis directions as the composed image, just a
translated origin. Diffing the two affines' translation columns and dividing by spacing gives an
**exact integer voxel offset** (verified: e.g. offset (68, 115, 183) for one sample, no
fractional remainder) — meaning a converter can composite the crop into the full-volume grid by
plain array placement, with no interpolation/resampling needed. All label values are strictly
`{0, 1}` (verified on multiple files).

## 4. Extreme label sparsity — 610/49,220 visits (1.24%)

The vast majority of the 49,220 cases have **no** lesion annotation at all. Of the 610 labeled
visits: **every labeled subject has labels at exactly one visit** (0 subjects with labels at
both "30" and "60") — so, for labeling purposes, visits and subjects are 1:1 here.

| canonical class | subjects (of 610 labeled) |
|---|---:|
| `mask_hyper.R.nii.gz` | 1 |
| `mask_hyper.L.nii.gz` | 2 |
| `Complex_R.nii.gz` | 12 |
| `Complex_L.nii.gz` | 18 |
| `Hyper_R.nii.gz` | 53 |
| `Hyper_L.nii.gz` | 61 |
| `mask_hyper.cyst.R.nii.gz` | 68 |
| `mask_hyper.cyst.L.nii.gz` | 73 |
| `mask_complex.cyst.R.nii.gz` | 83 |
| `mask_complex.cyst.L.nii.gz` | 114 |
| `Hypo_R.nii.gz` | 153 |
| `Hypo_L.nii.gz` | 177 |
| `mask_hypo_R.nii.gz` | 294 |
| `mask_hypo_L.nii.gz` | 390 |

Highly imbalanced: 390:1 between the most- and least-common canonical class. Most labeled
subjects carry only 1-2 classes (133 with 1, 280 with 2), but a long tail carries many at once —
up to **11 of the 14 classes on a single subject** (3 subjects with 9, 1 with 11).

**149 distinct non-canonical/legacy basenames** also appear (215 occurrences total) —
`mask_untitled*.nii.gz`, casing variants (`complex_R.nii.gz`), subject-ID-prefixed exports
(`100011_R1_un.nii.gz`), `clone_of_*` — consistent with an evolving annotation pipeline across
years (matches the `ML_fixed/` vs `ML/` subfolder naming seen in the same tree). These are
excluded from the census/subset; if a future converter needs more label yield, that legacy
naming needs its own review pass, not a regex guess.

## 5. Class names are NOT a clean subregion hierarchy — verified via crop-box comparison

Checked whether e.g. `mask_hyper.cyst.R` is always a spatial subset of `Hyper_R` (i.e. "cystic
component of the same lesion") by diffing crop-box origin/shape for several subjects with both:

| pair (same subject) | crop box | voxel count | reading |
|---|---|---|---|
| `Hyper_R` vs `mask_hyper.cyst.R` (130619) | **identical** box | 441 vs 142 | subset — same lesion, cyst sub-mask |
| `Hyper_R` vs `mask_hyper.R` (111225) | **identical** box | 49 vs 28 | subset — same lesion, refined mask |
| `Hypo_R` vs `mask_hypo_R` (130619) | **disjoint**, offset (-112,-15,-14) vox | 1085 vs 3084 | **different lesion instance** |
| `Complex_L` vs `mask_complex.cyst.L` (130619) | different box, offset (-9,0,1) vox | 5460 vs 416 | **different lesion instance** (partial spatial overlap at best) |

**When the crop box matches exactly, the `mask_*` variant is a genuine subregion of the same
lesion. When it doesn't, the two names denote two separate lesions in the same kidney** (a
kidney can have both a hyperdense and hypodense finding, or two hypodense findings, each boxed
independently). **Any per-class in-context task design must treat each of the 14 canonical
names as its own independent lesion-instance class — not assume `mask_X` ⊆ `X` in general.**
Not fully resolved (would need per-subject box-overlap checked at scale, not just 4 samples) —
flagged here so a future converter doesn't silently assume a hierarchy that doesn't hold.

`.cysts.json` sidecars (e.g. `mask_hypo_R.cysts.json`) give per-lesion cyst count and, per cyst,
voxel count / mm³ size / world-space center — useful extra signal (e.g. for filtering
too-small cysts) not yet used in the subset selection below.

## 6. Selected subset: 181 subjects, rarity-weighted greedy balance

`scripts/inspect_gnc_kidney.py --subset --cap 60` — greedy set-cover: score each labeled subject
by `sum(1 / global_class_count)` over its classes (rewards subjects carrying rare classes AND
subjects carrying many classes at once — a multi-label subject is "free" coverage of several
classes for one context example), then add subjects in descending score order, skipping once
every class a subject carries has already hit `min(cap, class_total)`.

**Result, `cap=60`: 181 / 610 labeled subjects selected** (0.6% of the full 30,385-subject
cohort) — every genuinely rare class (≤18 subjects total) keeps **100%** of its available
subjects; every common class is capped at 60-98:

| class | selected / available |
|---|---:|
| `mask_hyper.R` | 1 / 1 |
| `mask_hyper.L` | 2 / 2 |
| `Complex_R` | 12 / 12 |
| `Complex_L` | 18 / 18 |
| `Hyper_R` | 53 / 53 |
| `Hyper_L` | 60 / 61 |
| `mask_hyper.cyst.R` | 61 / 68 |
| `mask_hyper.cyst.L` | 60 / 73 |
| `mask_complex.cyst.R` | 63 / 83 |
| `mask_complex.cyst.L` | 60 / 114 |
| `Hypo_R` | 60 / 153 |
| `Hypo_L` | 66 / 177 |
| `mask_hypo_R` | 77 / 294 |
| `mask_hypo_L` | 98 / 390 |

Mean classes/subject in the selection ≈ 3.4 (vs 1.9 in the full 610-subject labeled pool) — the
greedy score deliberately concentrates on multi-label subjects first. Manifest (subject, visit,
`;`-joined class list): `docs/datasets/gnc_kidney_lesions_subset.csv` (181 rows).

`cap` is a free parameter — raise it to trade subset size for deeper coverage of the common
classes (e.g. `cap=150` would roughly double `mask_hypo_L`/`mask_hypo_R` coverage at the cost of
more total subjects); 60 was chosen here as roughly "the size of the smallest non-degenerate
class" (`Hyper_R`=53), so no common class dominates the selection much more than the naturally
rare ones already do.

## 7. Proposed eval protocol for the 610 labeled cases — DESIGNED, NOT YET BUILT

Goal: evaluate the existing exp92 checkpoint (or any trained checkpoint) OOD on GNC, reusing
every labeled case rather than only the balanced 181-subject subset (§6 remains useful for a
secondary controlled comparison, see step 6 below). Follows the same shape as the ISLES22 /
Shifts-MS / ATLAS v2.0 eval-only integrations (`NativeGridProvider`, no training).

**1. Class list — 13 of the 14 canonical classes, each independent (per §5).**
`mask_hyper.R` (n=1 subject) is **excluded**: in-context eval needs at least one context subject
distinct from the target, so N=1 cannot form a single (context, target) pair. `mask_hyper.L`
(n=2) is kept but flagged **underpowered** (only 2 possible target/context assignments) — report
its number with that caveat attached, don't read it as a stable estimate. No merging of `X` and
`mask_X`/`mask_X.cyst` variants into one class — §5 showed that doesn't hold uniformly.

**2. Data-quality exclusions — verified this session, exclude before eval:**
- **4 outlier label files** (of 1,499 checked) have a bounding box essentially spanning the
  full native XY extent (320×260, matching the whole image, not a lesion crop) — `125816/30/
  mask_hypo_L`, `130579/30/Hypo_R`, `131197/30/{Hyper_L,Hypo_L}`. Same category as ATLAS v2.0's
  single corrupt mask: a real data artifact, not a pipeline bug. A converter should reject any
  label whose bbox covers e.g. >80% of the native volume on 2+ axes, mirroring the strict
  integral-mask check already used for ATLAS.
- **2 subjects (130579, 131197) have 6.0mm Z-spacing** instead of the cohort-typical 3.0mm — a
  converter must read spacing per-subject from the header (as every other provider already
  does), never assume the global constant.

**3. Channel: water (W) only for the first pass**, same "pick one representative sequence"
precedent as ISLES22 (DWI out of 3 co-registered sequences). Water-phase Dixon gives the
clearest fluid/parenchyma contrast for cystic and complex renal lesions among the 4 available
(opp/in/fat/water) and needs no new architecture work — one scalar channel slots into the
existing `[image, mask]` input directly, no channel-split needed for an *eval-only* run (unlike
training, which would need the MSD-Prostate-style split to use more than one channel). A 4-way
channel ablation (does opp/in/fat generalize better than water?) is a reasonable follow-up, not
required for a first number.

**4. Geometry — `crop_spacing_mm=1.4`, computed from real per-file label extents (not
guessed):** read all 1,499 label file headers (shape × spacing, excluding the 4 outliers above)
— max extent on any axis **174.0mm** (95th pct 144.0mm, so the true max is itself a mild
outlier, but a real lesion, not a data artifact). `crop_spacing_mm=1.4` gives FOV=179.2mm at
T=128, zero clipping on the clean set, and **conveniently needs no in-plane resampling at all**
(native in-plane spacing is 1.40625mm — 1.4mm is a de-facto identity crop in X/Y, only Z needs
interpolation from the native 3.0mm slice thickness). This is the organ-extent clip-avoidance
sweep style (MSD Prostate/ACDC), computed exactly rather than assumed uniform like the
whole-volume-coverage sources (this session's first pass wrongly assumed globally-uniform
spacing from a 2-sample check — the 6mm-spacing subjects above were only caught by reading all
1,499 headers).

**5. In-context sampling**: same eval-loader defaults as every other OOD-MRI source this
session (no K override) — keep the protocol comparable to ISLES22/Shifts-MS/Hippocampus/
Prostate/ATLAS's own numbers rather than introducing a new K just for GNC. `eval_seed`
reproducibility (per-item RNG fix, see `project_3d_eval_repro_fix` memory) applies unchanged.

**6. Reporting — per-class + two macro views:**
- Primary: macro-average Dice/NSD across the 13 evaluable classes, full N per class (12-390,
  `mask_hyper.L` flagged n=2), same table style as the other per-class MRI eval writeups.
- Secondary, sanity check only: re-run restricted to the §6 balanced 181-subject subset
  (`docs/datasets/gnc_kidney_lesions_subset.csv`) — if the full-610 and capped-181 macro numbers
  disagree substantially for the common classes, that's a signal the subset's greedy multi-label
  selection (favoring subjects with many simultaneous findings — likely sicker patients) is not
  representative of the full labeled pool, worth knowing before trusting either number alone.

**7. Not yet built** — same three pieces every other source needed: converter (native RAS `.npy`
+ per-subject `mri_stats`, water channel only, with the two exclusion rules from step 2 applied
at conversion time), `NativeGridProvider` subclass (`MODALITY="mri"`, 13-class registry), config
+ `common.py`/`eval.py` dispatch branches. This doc is the design; building it is a separate,
explicit next step.

## 8. Fit as a task / next steps — NOT YET DONE

- **Modality**: `NativeGridProvider MODALITY="mri"` fits directly (per-subject `mri_stats` is
  already the mechanism every prior MRI source uses).
- **Channel count is the real design question**: the model's image input is 2-channel
  `[image, mask]` (single scalar image channel, per `CLAUDE.md`) — GNC's composed file is
  genuinely 4-channel (opp/in/fat/water) **simultaneously per visit**, unlike MSD Prostate's
  T2+ADC case (also 2 channels, solved by the **channel-split** pattern: converting each channel
  into its own single-channel subject sharing the same label, `docs/datasets/msd_prostate.md`
  §7). The same pattern applies here — most likely a **4-way split** per visit (or a smaller
  subset of the 4, e.g. water-only, if in-phase/fat/opp turn out redundant for lesion contrast)
  — worth deciding explicitly before writing a converter, not defaulting to one silently.
  - Anisotropic native grid (1.40625×1.40625×3.0mm) needs its own `crop_spacing_mm` sweep once a
    channel strategy is picked, same organ-extent-clip-avoidance style as MSD Prostate/ACDC given
    labels are small ROIs inside a much larger native FOV (not whole-volume coverage like
    ISLES22/ATLAS).
- **Per-class independence** (§5) should directly inform whatever the (subject, class) task
  definition ends up being — do not collapse `X` and `mask_X.cyst` into one "refine" task without
  first checking box-identity per subject, since it doesn't hold uniformly.
- Reproduce this census: `python scripts/inspect_gnc_kidney.py --scan --analyze --subset
  --check-images` (the `--scan` step is NFS-latency-bound, ~5 min via a 128-thread pool — a
  naive `find -maxdepth 2` over the same tree did not finish in 60s, hence the threaded
  `os.listdir` approach). Raw scan cache (`results/3d/gnc_kidney/scan_result.json`, ~4MB,
  git-ignored) regenerates from `--scan`; only the small derived subset CSV is committed.
