# GNC_705 Kidney Lesions (German National Cohort)

*Kidney lesion segmentation (hyperdense / hypodense / complex, +cystic components), 4-channel
Dixon MRI.* Inspected 2026-09-13 on user request ("inspect directory ... find a smaller,
balanced, label-rich subset"). **Local restricted-access cohort data, not a public download**:
lives at `/nfs/data/nii/data0/GNC/GNC_705/` — no access/provenance question applies the way it
did for the HF-mirror sources in `eval_expansion_status.md`; this is simply a large local
dataset that hadn't been touched before this session. Characterized, subset-selected, then
fully integrated + eval'd (single-level and cascade) on explicit "write converter and eval"
direction — see §7-9.

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

## 7. Integration + eval — IMPLEMENTED AND RUN, 2026-09-13 (two corrections vs. the original design)

Built and run on the explicit "write converter and eval with cascade spacings" instruction.
Same shape as ISLES22/Shifts-MS/ATLAS v2.0 (`NativeGridProvider`, `MODALITY="mri"`, no
training) — but two pieces of the design below turned out wrong once real per-file data was
checked, both caught before they could corrupt a result silently:

**Correction 1 — `crop_spacing_mm` computed from raw file SHAPE was wrong; the fix uses each
label's own nonzero-content bounding box instead.** The original sweep (below) read each
label's `.nii.gz` shape × spacing and found a 174mm max extent, attributing 4 files with a
~320×260 shape to "corrupt whole-volume masks." Loading the actual array content showed those 4
files have a TINY nonzero region (14×14×6 voxels) sitting inside an oversized, un-cropped
canvas — a real annotation-export quirk (some files weren't auto-cropped to their own bounding
box), not corruption. Redone properly (load all 1,499 label arrays, measure the true nonzero
extent): **max content extent is only 153.0mm** (median just 18mm — most lesions are tiny),
giving **`crop_spacing_mm=1.2`** (FOV=153.6mm at T=128, zero clipping) instead of the original
1.4mm guess. The oversized-canvas files need no special-casing at all once extent is measured
correctly — `_make_plane`'s offset+shape placement (below) handles any container size.

**Correction 2 — one binary plane per class, NOT a shared multi-valued `label.npy`.** Painting
every present class into ONE shared array (the format every other multi-class source here uses)
was tried first and produces thousands of overlapping voxels on some subjects (e.g. 7,516 on
one 11-class subject) — confirming §5's finding empirically: a `mask_X.cyst` genuinely sits
inside its `X` superset. Overwriting with a shared array silently shrinks the superset's mask
every time both are present. Fixed by writing `label_{cls}.npy` per PRESENT class instead
(`scripts/convert_gnc_kidney.py`), and giving `GncKidneyProvider` (`src/providers/gnc_kidney.py`)
its own override of the 3 `NativeGridProvider` methods that touch the shared-array path
(`_load_or_build_centroids`, `load`, `load_native_crop`) — each queries its own class's plane
with a fixed `class_idx=1` rather than the shared array's per-class integer.

**Class list — 13 of 14, `mask_hyper.R`/`hyper_mask_r` excluded** (n=1 subject, can't form a
context-disjoint-from-target pair). `hyper_mask_l` (n=2) kept but underpowered.

**Channel — water only** (index 3 of the composed file's 4 channels), **verified by
correlation** against the single-contrast station files (corr>0.99 for all 4: order is
opp/in/fat/water, matching the user's stated order) rather than assumed from the filename.

**Data-quality handling, all caught by defensive checks rather than crashing:** 8 individual
label-plane rejections out of 1,499 (0.5%) on real conversion — 3 from the two
already-known 6mm-Z-spacing subjects (130579, 131197: label affine's Z-scale doesn't match the
image's, so the "pure integer-voxel translation" check correctly refuses to place it rather
than silently misaligning), and 5 more `"plane is empty after clipping"` cases surfaced only at
full-scale conversion (not visible in the 6-subject dry run) — a label's content fell entirely
outside the image canvas after the defensive clip. All 610 (subject,visit) cases still convert
successfully; only these 8 of 1,499 label planes are dropped.

```bash
python scripts/convert_gnc_kidney.py --workers 16
python experiments/3d/eval.py dataset=gnc_kidney eval.model=patchset3d eval.checkpoint=<ckpt>
```

| piece | file |
|---|---|
| converter | `scripts/convert_gnc_kidney.py` |
| provider | `src/providers/gnc_kidney.py` (`GncKidneyProvider`, per-class-plane override) |
| config | `configs/experiment/3d/dataset/gnc_kidney.yaml` |
| dispatch | `experiments/3d/common.py`, `experiments/3d/eval.py` (same branches the other MRI sources use) |

**Gotcha hit on the first eval attempt, not GNC-specific**: with `eval.workers=20` (the repo
default) and NO pre-existing `.centroid_cache_perclass.pkl`, each of the 20 DataLoader worker
processes independently tries to build the centroid cache via its OWN nested
`ProcessPoolExecutor(16)` — a 20×16 process explosion that hangs (confirmed via `pstree`: 30+
idle `multiprocessing.forkserver` processes, main process blocked in `pipe_write`, zero CPU
growth over minutes). This is latent in `NativeGridProvider._load_or_build_centroids` for ANY
source's very first eval run before its cache file exists — GNC just triggered it first this
session because every other source's cache was already warm from earlier runs. **Fix**: build
the cache once in the main process before invoking `eval.py` (`GncKidneyProvider(root=...,
classes=...)` — 19s for 610 subjects), then re-run; every worker takes the fast
`pickle.load` path. Worth fixing upstream (build the cache in the main process before
`DataLoader` workers fork, not lazily per-worker) if another brand-new source hits this again.

Raw data stays at `/nfs/data/nii/data0/GNC/GNC_705/` (`data/`, `links/`); converted native-grid
`.npy` at `paths.gnc_kidney` (`/nfs/.../data/gnc_kidney/npy/`, 32GB, 610 subjects).

## 8. Single-level eval result (exp92 checkpoint, `crop_spacing_mm=1.2`, 2026-09-13)

```
python experiments/3d/eval.py dataset=gnc_kidney eval.model=patchset3d \
  eval.checkpoint=.../3d_train/2026-09-11_92_multisource_synth/best.pt
```
**Mean Dice 0.0585 ± —, NSD 0.0784, n=1490** (wandb `still-wave-81`) — in the same ballpark as
ISLES22 (0.0544) and Shifts-MS (0.0632), below MSD Hippocampus (0.481) and MSD Prostate (0.295).

| class | n | dice | nsd |
|---|---:|---:|---:|
| hyper_r | 53 | 0.028 | 0.056 |
| hyper_l | 60 | 0.039 | 0.072 |
| hypo_r | 152 | 0.037 | 0.051 |
| hypo_l | 176 | 0.042 | 0.057 |
| complex_r | 12 | 0.052 | 0.067 |
| complex_l | 18 | 0.105 | 0.104 |
| hyper_cyst_r | 68 | 0.035 | 0.055 |
| hyper_cyst_l | 73 | 0.051 | 0.075 |
| hyper_mask_l (n=2, underpowered) | 2 | 0.119 | 0.156 |
| hypo_mask_r | 292 | 0.066 | 0.082 |
| hypo_mask_l | 387 | 0.057 | 0.075 |
| complex_cyst_r | 83 | 0.060 | 0.088 |
| complex_cyst_l | 114 | 0.068 | 0.081 |

No class clears 0.12 Dice; `complex_l`/`hyper_mask_l` are the (weak) high points, `hyper_r` the
floor. Consistent with a genuinely OOD failure (unseen class, unseen modality-region combo —
kidney LESIONS, not the whole-organ `kidney_left`/`kidney_right` the checkpoint trained on)
rather than a pipeline bug, given the clean per-class spread and the sane compute cost (392ms/
sample, 3891 GFLOPs — no runaway cost from the larger native volume).

Qualitatively (`figures/hyper_r_100025_v30.png`, dice=0.030): a genuine under-segmentation, not
a pipeline artifact — the hyperintense lesion is directly visible as a bright spot in the raw
water-channel image itself (unsurprising: "hyperintense" is exactly what defines this class on
this sequence), confirming context/target alignment is correct. The model's actual prediction
is a small sliver covering only a fraction of the true lesion (the GT overlay), the same
under-segmentation failure mode seen on the other small-structure OOD-MRI sources this session
rather than a localization miss or empty-collapse.

## 9. Cascade eval `[6, 3, 1.2]` mm — matches exp92's own trained ladder length (2026-09-13)

"Find fitting spacings" for GNC specifically: rather than reusing ISLES22's `[3,1.5]` blind, the
coarse point was checked against GNC's real geometry. Every reasonable coarse pitch from 1.2mm
up to 6mm gives zero clipping here (true max content extent is only 153mm vs. a 6mm-pitch FOV
of 768mm) — unlike ISLES22, where the coarse pitch had to be chosen to just barely contain a
whole hemisphere-sized lesion, GNC's lesions are tiny relative to the full torso-scale composed
image, so clipping was never the binding constraint. That makes GNC's cascade choice closer to
FLARE22/NasalSeg's original organ-localization framing (small target somewhere in a much larger
scene) than to ISLES22/ATLAS's "whole lesion barely fits the frame" framing — so the fitting
ladder here is **exp92's own trained `[6,3,1.5]` ladder, with only the fine point corrected to
this dataset's measured value**: `[6, 3, 1.2]`.

```
python experiments/3d/eval.py experiment=92_multisource_synth eval.model=patchset3d \
  eval.checkpoint=<ckpt> data.source=gnc_kidney data.crop_spacing_mm=6 \
  data.cascade_spacings=[6,3,1.2] data.mask_downsample=occupancy data.gpu_realize_crop=false \
  data.val_classes=<13-class list> train.cascade_loss_weights=[1,1,1] eval.split=test \
  eval.cascade_figures=true
```

**Blocked on the first attempt by a general architectural gap, fixed (see §9a) before this
number is real.** Cascade eval scores the STITCHED NATIVE-VOLUME prediction by loading
`label.npy` directly and checking `== class_idx` (`evaluate._stitched_native_metrics_multi`) —
it bypasses the provider's own `.load()`/`.load_native_crop()` entirely, so GncKidneyProvider's
per-class-plane fix (§7 correction 2) didn't cover it. First run crashed:
`FileNotFoundError: .../100025_v30/label.npy` (187/187 cases had already run — only the final
native-space scoring pass failed). Fixed generally, not just for GNC — see §9a.

**Result (n=1490, all 13 classes, after the fix): Mean Dice 0.0189, NSD 0.0293** — WORSE than
the single-level baseline (0.0585 Dice, §8), the same conclusion ISLES22 reached for its own
cascade attempt. **Caveat, same one ISLES22's writeup flags**: this isn't quite an apples-to-
apples comparison — §8's number is scored in CROP SPACE (the resampled 128³ grid) while this
one is scored in NATIVE SPACE (stitched back to the full native volume, a stricter metric per
`native_grid.py`'s own docstring) — but the qualitative figures (below) show a real, independent
failure mode on top of that scoring difference, not just a metric-space artifact.

| class | n | dice | nsd | dice@6mm | dice@3mm | dice@1.2mm |
|---|---:|---:|---:|---:|---:|---:|
| hyper_r | 53 | 0.013 | 0.026 | 0.007 | 0.013 | 0.014 |
| hyper_l | 60 | 0.013 | 0.020 | 0.009 | 0.010 | 0.017 |
| hypo_r | 152 | 0.023 | 0.032 | 0.014 | 0.018 | 0.024 |
| hypo_l | 176 | 0.018 | 0.028 | 0.014 | 0.016 | 0.019 |
| complex_r | 12 | 0.022 | 0.041 | 0.024 | 0.026 | 0.022 |
| complex_l | 18 | 0.042 | 0.053 | 0.037 | 0.042 | 0.042 |
| hyper_cyst_r | 68 | 0.009 | 0.015 | 0.004 | 0.009 | 0.010 |
| hyper_cyst_l | 73 | 0.013 | 0.022 | 0.005 | 0.007 | 0.013 |
| hyper_mask_l (n=2) | 2 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| hypo_mask_r | 292 | 0.029 | 0.040 | 0.014 | 0.022 | 0.029 |
| hypo_mask_l | 387 | 0.022 | 0.034 | 0.013 | 0.018 | 0.023 |
| complex_cyst_r | 83 | 0.022 | 0.038 | 0.012 | 0.016 | 0.024 |
| complex_cyst_l | 114 | 0.020 | 0.032 | 0.011 | 0.017 | 0.022 |

Every class monotonically improves coarse→fine (`dice@6mm < dice@3mm < dice@1.2mm`) — the
resolution refinement itself works as intended — but the finest per-resolution number (1.2mm)
still lands below the single-level (also 1.2mm) baseline for every class, so refinement isn't
recovering what the extra localization step loses.

Qualitatively (`figures/cascade/hyper_r_3to1.2mm.png`, coarse pass): the SAME
intensity-brightness confound flagged in §8 — the coarse (3mm) prediction is a solid white blob
that exactly matches the lesion's natural hyperintense brightness, not real segmentation — but
here it has a second-order consequence the single-level path doesn't have: the fine crop's
localization (the yellow re-crop box) is centered nowhere near the true lesion, so the fine
(1.2mm) pass never even sees it — panel 3 shows plain kidney parenchyma, no lesion, and the fine
prediction (panel 4) is accordingly empty. **This is the same structural failure mode ISLES22's
cascade doc already identified**: the coarse level's own imprecise prediction feeds forward as
the fine level's re-crop center (`query_prior=pred`, the default), so an imprecise coarse guess
actively misdirects the fine pass rather than the fine pass independently re-finding the target.

**Conclusion, matching ISLES22's: prefer the single-level number (§8) for reporting.** Cascading
adds a genuine failure mode here (bad re-crop from an imprecise coarse pass) on top of the
stricter native-space metric — consistent with this checkpoint's OOD generalization gap being
in *localization/shape*, not *resolution*, for both lesion-segmentation OOD sources tried this
session (ISLES22 stroke lesions, GNC kidney lesions).

## 9a. General fix: cascade scoring now supports a provider without a shared `label.npy`

The crash above is NOT GNC-specific — it's a real gap in shared cascade infrastructure that
would hit ANY future source needing per-class-plane storage (per the user's ask: "we might have
other eval datasets with overlapping labels in the future"). Fixed via a new provider hook
rather than a GNC-only patch:

- **`NativeGridProvider.native_gt(self, subject, cls) -> bool ndarray | None`**
  (`src/providers/native_grid.py`): default implementation reproduces the old
  `label.npy == CLASS_IDX[cls]` read exactly (verified byte-identical against ATLAS v2.0 at fix
  time) — every existing native-grid source (FLARE22, NasalSeg, ISLES22, Shifts-MS, MSD
  Hippocampus/Prostate, ATLAS v2.0) is unaffected. `GncKidneyProvider` overrides it to read its
  own `label_{cls}.npy` plane instead.
- **`evaluate._stitched_native_metrics_multi` / `_stitched_native_dice_multi`** gained an
  optional `gt_loader(subj, cls)` parameter that takes priority over the old `class_idx`-based
  shared-array read when given. `cascade.py`'s call site now does
  `gt_loader = getattr(loader.dataset.provider, "native_gt", None)` and passes it through —
  `None` for TotalSegProvider/MultiSourceProvider (unchanged behavior), the provider's own hook
  for any `NativeGridProvider` subclass.
- **Not touched**: the older `evaluate.evaluate_spacing_sweep(..., cascade=True)` /
  `_stitched_native_dice` path (the pre-v2 cascade mechanism, superseded by `data.cascade_spacings`
  but still present) has the same `label.npy`-only assumption and would need the identical
  `gt_loader` threading if a future overlapping-label source is ever evaluated through it — not
  exercised by this fix since the v2 path is what `data.cascade_spacings` actually uses.

Net effect: any future source whose classes don't partition into one shared array only needs to
override `native_gt` (one method) to get correct cascade scoring for free — no cascade.py or
evaluate.py changes needed per-source.

## 10. Remaining open questions (not blocking the eval numbers above)

- **Channel-split for TRAINING (not eval)**: §7's water-only choice is sufficient for evaluating
  an existing checkpoint, but if GNC were ever used as a TRAINING source, the other 3 Dixon
  channels (opp/in/fat) would need the MSD-Prostate-style channel-split (each channel → its own
  single-channel subject sharing the same label planes) to use more than one channel — not
  attempted here, eval-only doesn't need it.
- **Per-class independence** (§5) should directly inform whatever the (subject, class) task
  definition ends up being if this grows beyond eval — do not collapse `X` and `mask_X.cyst`
  into one "refine" task without first checking box-identity per subject, since it doesn't hold
  uniformly (confirmed empirically at conversion time, §7 correction 2).
- Reproduce the census: `python scripts/inspect_gnc_kidney.py --scan --analyze --subset
  --check-images` (the `--scan` step is NFS-latency-bound, ~5 min via a 128-thread pool — a
  naive `find -maxdepth 2` over the same tree did not finish in 60s, hence the threaded
  `os.listdir` approach). Raw scan cache (`results/3d/gnc_kidney/scan_result.json`, ~4MB,
  git-ignored) regenerates from `--scan`; only the small derived subset CSV is committed.
- Untested: a 4-channel ablation (does opp/in/fat generalize better than water for this
  checkpoint?), and the §6 balanced-181-subset sanity comparison flagged in the original design.
