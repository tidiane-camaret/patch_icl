# BraTS 2024 (Brain Tumor Segmentation)

*Multi-track brain tumor segmentation — glioma, meningioma+radiotherapy, pediatric — MRI.*
Downloaded 2026-09-13 via an unofficial HuggingFace re-upload, following up on
`docs/datasets/eval_strategy_report.md`'s BraTS pick. **Genuinely OOD class, verified against
our own training vocabulary**: no tumor sub-region class of any kind exists in
`data/totalseg_classes.py`. Not yet wired into the harness — characterization only.

## 1. Access — ⚠️ PROVENANCE CAVEAT: mixed, genuinely worse than ATLAS v2.0's

**Verified this session: the official BraTS 2024 release requires Synapse registration**
(`synapse.org/brats`) and a signed challenge data-use agreement — confirmed directly by this
mirror's own `LICENSES.md`, which states GLI and MEN-RT were "Downloaded under the BraTS 2024
challenge agreement" from `Synapse.org (Synapse:syn53708249)`. This is an explicit admission,
in the mirror's own documentation, that two of its three tracks are DUA-gated data being
redistributed outside that agreement:

| track | source per LICENSES.md | provenance read |
|---|---|---|
| BraTS-GLI (glioma, 1,809 cases) | Synapse, "BraTS 2024 challenge agreement" | **DUA-circumventing re-upload** — same category as the ATLAS v2.0 mirror |
| BraTS-MEN-RT (meningioma+RT, 571 cases) | Synapse, "BraTS 2024 challenge agreement" | **DUA-circumventing re-upload** |
| BraTS-PED (pediatric, 348 cases) | "The Cancer Imaging Archive (TCIA)", claimed CC BY-NC 4.0 | **Unverified** — could not find a TCIA public collection matching this name via the TCIA REST API (searched for "brats"/"pediatric"/"brain-tum" in `getCollectionValues`); the challenge's underlying source is reportedly the Children's Brain Tumor Network via TCIA, which may use a different collection name than what's implied here. Treat with the same caution as GLI/MEN-RT until independently confirmed. |

**This was downloaded on explicit user direction**, given the same treatment as ATLAS v2.0:
the DUA-circumvention concern was raised via `AskUserQuestion` before downloading anything, and
the user chose to proceed for their own research use. Recorded here, independent of what gets
done with the data, so the provenance risk is visible to anyone reading this doc later — this
is NOT equivalent to the genuinely open sources characterized elsewhere in `docs/datasets/`.

## 2. Composition — three tracks, very different in character

| track | train | val | modality | sequences | GT format |
|---|---:|---:|---|---|---|
| BraTS-GLI | 1,621 | 188 | MRI | t1n, t1c, t2w, t2f (4-seq) | `seg.nii.gz`, multi-class |
| BraTS-MEN-RT | 500 (+1 additional) | 70 | MRI | **t1c only** (1-seq) | `gtv.nii.gz`, binary |
| BraTS-PED | 257 | 91 | MRI | t1n, t1c, t2w, t2f (4-seq) | `seg.nii.gz`, multi-class |

**2,728 total cases** across all tracks/splits — the largest single pull this session by case
count (ahead of AutoPET-III's 1,038). GLI and PED share the standard 4-sequence BraTS layout;
**MEN-RT is a genuinely different task** — a single contrast-enhanced T1 sequence with a binary
gross-tumor-volume (GTV) mask, the radiotherapy-planning target volume rather than a
multi-region tumor segmentation. Treating MEN-RT as a third, separate integration (not just a
config variant of GLI/PED) is the right frame.

## 3. Geometry — VERIFIED, and GLI/PED are template-uniform while MEN-RT is genuinely raw

| track | orientation | shape (vox) | spacing (mm) |
|---|---|---|---|
| GLI | **LAS**, uniformly | exactly 182×218×182 (all 1,621) | exactly 1×1×1mm (all 1,621) |
| PED | **LPS**, uniformly | exactly 240×240×155 (all 257) | exactly 1×1×1mm (all 257) |
| MEN-RT | **mixed** (473 RAS + 27 LAS of 500) | 160×256×75 to **800×800×512** | 0.338×0.338×0.488 to 1.055×1.055×2.0mm |

GLI and PED are **completely uniform across every single case** — same character as ATLAS
v2.0, confirming these tracks ship an already-template-registered version (standard BraTS
preprocessing: skull-stripped, co-registered, resampled to the SRI24 atlas grid). **MEN-RT is
the opposite** — real per-case orientation and a shape range spanning nearly 3 orders of
magnitude in voxel count (up to 800×800×512, dwarfing even AutoPET-III's whole-body 963-slice
maximum), consistent with raw clinical radiotherapy-planning acquisitions rather than a
standardized research release. Any MEN-RT converter would need real per-file orientation
verification (not a hardcoded flip) and a memory/crop-pitch strategy sized for genuinely
enormous native volumes — closer to AMOS22's heterogeneity problem than to anything else pulled
this session.

## 4. Labels — verified per track, GLI/PED share a 4-class schema, MEN-RT is binary

| track | label | present (of n) |
|---|---:|---|
| GLI (n=1,621) | 1 | 706 (44%) |
| | 2 | 1,618 (99.8%) |
| | 3 | 1,223 (75%) |
| | 4 | 1,374 (85%) |
| PED (n=257) | 1 | 171 (67%) |
| | 2 | 256 (99.6%) |
| | 3 | 83 (32%) |
| | 4 | 52 (20%) |
| MEN-RT (n=500) | 1 (gtv) | 500 (100%) |

GLI and PED share the same 4-value label schema (values 1-4), consistent with BraTS's current
multi-region tumor-subregion convention (necrotic core / edema / non-enhancing tumor core /
enhancing tumor, exact value↔name mapping not independently re-derived here — take from the
official BraTS label definitions if a converter is written, don't assume from this census
alone). **Presence rates differ meaningfully between tracks**: PED shows much lower rates for
labels 3-4 than GLI (32%/20% vs. 75%/85%) — plausible given pediatric gliomas' known biological
difference from adult gliomas (less necrosis/enhancement is a recognized clinical pattern,
consistent with why pediatric tumors get their own BraTS track rather than being pooled with
GLI). MEN-RT's GTV is present in 100% of cases by definition (it's the RT target, not an
optional finding).

## 5. Intensity — verified standard for MRI, per-sequence

- **GLI/PED**: float32, verified range e.g. [0, 8184] on a GLI t1c sample — already-normalized-
  looking dynamic range for the multi-sequence tracks (consistent with the uniform-geometry
  finding suggesting standard BraTS preprocessing).
- **MEN-RT**: int16, verified range e.g. [0, 7286] — raw-er scanner-unit character, consistent
  with its raw-geometry finding above.

Both need the standard per-subject `mri_stats`/`normalize_mri` mechanism, not a fixed frame.

## 6. Fit as an eval set

- **Genuinely unseen classes**: tumor sub-regions (GLI/PED) and GTV (MEN-RT) don't exist in
  TotalSegmentator's or our own vocabulary.
- **By far the largest case pool of anything this session** (2,728 vs. AutoPET-III's 1,038) —
  if the provenance question is resolved, this would give the deepest held-out-subject
  evaluation of any source integrated so far.
- **GLI/PED are the "easy" tracks technically** (uniform template geometry, closed-form
  crop-pitch check — same character as ATLAS v2.0/MSD Hippocampus): a quick sweep at T=128
  shows `crop_spacing_mm=1.8` (GLI) / `1.9` (PED) as the smallest zero-clip pitch.
- **MEN-RT is a genuinely harder integration** than any of the four already-integrated MRI
  sources — real per-case geometry heterogeneity, a single-sequence input (unlike every other
  source, which is single-sequence by choice; here it's single-sequence by the task itself),
  and a binary RT-target-volume label that's a different clinical object than a "lesion" or
  "tumor sub-region" (it includes a clinician's planning margin, not just visible tumor).
- **4-class multi-region task on GLI/PED is itself a design question**: does an in-context task
  here mean one class at a time (e.g., "enhancing tumor" as its own (subject, class) task, the
  pattern every other multi-class source this session uses), or evaluate all 4 jointly? The
  former is consistent with this harness's existing pattern (MSD Hippocampus/Prostate both
  treat sub-structures as independent classes) and is the recommended default if this gets
  wired.

## 7. Integration — NOT YET DONE, three separate technical questions

Given §1-3, this is bigger in scope than a single provider:
1. **GLI**: standard `NativeGridProvider` subclass, `MODALITY="mri"`, `crop_spacing_mm≈1.8`,
   4 classes treated independently — closest to a mechanical port of anything in this doc.
2. **PED**: same shape as GLI, `crop_spacing_mm≈1.9`. Given the shared 4-class schema, GLI and
   PED could plausibly be ONE provider with a track flag, or two separate ones (`SOURCE=
   "brats_gli"` / `"brats_ped"`) — worth deciding explicitly rather than defaulting either way.
3. **MEN-RT**: needs real per-subject geometry handling (no shortcut from GLI/PED's uniformity)
   and its own crop-pitch sweep given the enormous shape range — a genuinely separate
   integration effort, not a config variant.

Given the provenance caveat in §1 is stronger here than ATLAS v2.0's (this mirror's own
documentation explicitly admits circumventing the Synapse DUA for 2 of 3 tracks, vs. ATLAS's
mirror which at least ships no explicit admission either way), any integration decision here
should weigh that. Not investigated further pending explicit direction.

Raw data kept at `/nfs/.../ANALYSIS_20251122/data/brats2024/` (`BraTS-GLI/`, `BraTS-MEN-RT/`,
`BraTS-PED/`, ~97GB total, 11,589 files) — downloaded in full via `huggingface_hub.
snapshot_download`, not yet converted.
