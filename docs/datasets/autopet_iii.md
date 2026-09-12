# AutoPET III (Lite NIfTI mirror)

*Whole-body FDG/PSMA PET/CT, tumor lesion segmentation.* Downloaded and inspected 2026-09-12,
following up on `docs/datasets/eval_strategy_report.md`'s Tier-2 lesion/cross-modality pick.
**PET is a genuinely new modality axis for this harness** (every other source is CT or MRI) and
tumor lesion is a genuinely unseen class (checked directly against `data/totalseg_classes.py`:
the only tumor-adjacent classes there are `liver_tumor`, `lung_nodules`, `kidney_cyst` — none of
which cover the whole-body multi-organ metastatic lesions this dataset actually contains). Not
yet wired into the harness — characterization only.

## 1. Access — official source is genuinely open; used a repackaged NIfTI mirror

**Official source, verified directly via the TCIA REST API** (no login needed for the query):
the challenge's two cohorts are hosted as public TCIA collections `FDG-PET-CT-Lesions` (900
patients, modalities CT/PT/**SEG** — the ground-truth lesion masks ship as DICOM SEG objects
directly in the collection) and `PSMA-PET-CT-Lesions`. Raw TCIA access means per-series DICOM
download via the NBIA Data Retriever or REST API (`getSeries`/`getImage`), plus DICOM-SEG
decoding for GT — real friction (flagged earlier this session as "worst access friction of
anything checked").

**Used instead**: a community-maintained NIfTI repackaging,
[`YongchengYAO/autoPET-III-Lite`](https://huggingface.co/datasets/YongchengYAO/autoPET-III-Lite)
on HuggingFace (not gated, downloaded anonymously — no HF token needed despite the dataset
card's login instructions). Per its README: "No change to any image or segmentation mask...
Images with empty segmentation masks are excluded from the official release" — i.e. a format
conversion (DICOM→NIfTI, SEG→binary mask) of the same official-open TCIA data, not a
redistribution of DUA-gated content (unlike the BraTS/ATLAS mirrors also found this session —
see `eval_expansion_status.md` for that distinction). License on the mirror is CC BY-NC 4.0
(non-commercial) — worth noting for anything beyond internal research use.
```bash
curl -L -o Images-CT.zip "https://huggingface.co/datasets/YongchengYAO/autoPET-III-Lite/resolve/main/Images-CT.zip"    # 37GB
curl -L -o Images-PET.zip "https://huggingface.co/datasets/YongchengYAO/autoPET-III-Lite/resolve/main/Images-PET.zip"  # 29GB
curl -L -o Masks.zip "https://huggingface.co/datasets/YongchengYAO/autoPET-III-Lite/resolve/main/Masks.zip"            # 9MB
```

## 2. Composition — two cohorts, non-empty lesions only, large pool

**1038 cases, all with a non-empty tumor mask** (empty cases excluded at the mirror's release,
unlike ISLES22's 3/250 empty-included approach):

| cohort | source | n | notes |
|---|---|---:|---|
| `fdg` | Tübingen (UKT) | 501 | melanoma/lymphoma/lung-cancer + some negative-control-derived positives (only non-empty kept) |
| `psma` | Munich (LMU) | 537 | prostate carcinoma |

Each case: one CT volume, one PET volume, one binary lesion mask — genuinely co-registered
(same collection, same study).

## 3. Geometry — VERIFIED, whole-body, the largest volumes in the harness by far

Census across all 1038 mask volumes (proxy for the paired CT/PET geometry — same grid):

| | orientation | shape (vox), min/median/max | spacing (mm), min/median/max |
|---|---|---|---|
| all 1038 | **LAS**, uniformly | 168×168×135 / 256×256×326 / 400×400×**963** | 2.04×2.04×2.0 / 2.73×2.73×3.0 / 4.07×4.07×5.0 |

**963 slices at the max** — whole-body (head-to-thigh or further) coverage, an order of
magnitude larger through-slice than anything else integrated this session (ISLES22/Shifts-MS
top out around 150-250 slices of a single brain). Any converter here needs to think about
memory/decode cost per volume, not just crop-pitch — a raw `.npy` at this scale is genuinely
large per subject.

## 4. Labels — single class, but extreme size variability

Single class (`tumor`, `labels_map={"1": "tumor"}` per the mirror's README) — but volume spans
an enormous range, larger than ISLES22/Shifts-MS's lesion-size spread:

| cohort | n | lesion volume (mm³), min/median/max |
|---|---:|---|
| fdg | 501 | 100 / 99,242 / 2,481,200 |
| psma | 537 | 415 / 43,837 / 2,858,684 |

The max (~2.5-2.9 million mm³, i.e. roughly a 135mm cube of tumor) implies diffuse/multi-organ
metastatic spread in the worst cases — a more extreme version of ISLES22's multi-focal-lesion
problem, on a whole-body canvas instead of a single brain. Expect the same COM-recrop/cascade
mismatch already documented for ISLES22 to be at least as severe here, if not worse.

## 5. Intensity — standard for both modalities, needs a genuinely new normalization branch

- **CT**: real HU, same fixed `normalize_ct`/`DEFAULT_CT_NORM` frame as every CT source.
- **PET (PT)**: **not yet handled by anything in this harness.** PET intensity is typically
  reported in SUV (standardized uptake value) or raw activity concentration, not HU and not
  arbitrary-per-scan like MRI either — it has its own physical unit and normalization
  convention (SUV clipping, e.g. [0, 15] or similar, is standard in the PET-lesion literature).
  `NativeGridProvider`'s `MODALITY` currently only branches `"ct"`/`"mri"` — PET would need a
  **third modality branch** (`MODALITY="pet"`), not a reuse of `mri_stats`. This is the one
  piece of real new code this source would need beyond a mechanical converter port.

## 6. Fit as an eval set

- **Only non-CT-modality axis pulled this session that ISN'T MRI** — PET is a fundamentally
  different acquisition (functional, not anatomical) that no other source here tests. A model
  segmenting PET-defined tumor uptake is a qualitatively different generalization claim than
  anything MRI-based.
- **Genuinely unseen class**: whole-body multi-organ tumor lesions, not covered by
  TotalSegmentator's own tumor-adjacent subtasks (liver/lung/kidney only, and those are organ-
  confined, not whole-body).
- **By far the largest pool** of any source pulled this session (1038 cases vs. AMOS22's 360 or
  MSD Hippocampus's 260) — comfortably enough for held-out splits if that's ever wanted.
- **Real integration cost, not just a design decision**: needs (a) a new PET normalization
  branch in `native_grid.py`, (b) a `crop_spacing_mm`/memory strategy sized for whole-body
  volumes up to 963 slices (very different regime from anything else here), and (c) probably a
  CT-vs-PET-vs-both channel decision mirroring MSD Prostate's T2/ADC question — segment on CT,
  on PET, or fuse both? The tumor is far more PET-conspicuous than CT-conspicuous in most
  cases, so a CT-only in-context task may be a much harder (or differently-shaped) problem than
  the source dataset was designed for.

## 7. Integration — NOT YET DONE

No provider/converter/config yet. This is a bigger lift than any of the four fully-integrated
sources — new PET modality branch, whole-body memory/crop-pitch sizing, and a channel-choice
design question all need resolving first, closer in spirit to MSD Prostate's 4D question than
to a mechanical NativeGridProvider port.

Raw zips kept at `/nfs/.../ANALYSIS_20251122/data/autopet_iii_lite/` (`Images-CT.zip` 37GB,
`Images-PET.zip` 29GB, `Masks.zip` 9MB) — not yet extracted/converted (would need ~66GB+ of
scratch space to unzip in place; only `Masks.zip` was fully extracted for this characterization
pass).
