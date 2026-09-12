# AMOS22

*Abdominal Multi-Organ Segmentation 2022 — 15-organ CT+MRI benchmark, multi-center/multi-
vendor/multi-phase.* Unpacked and inspected 2026-09-12 (the zip was already downloaded from a
prior session, staged at `/nfs/.../ANALYSIS_20251122/data/amos22/amos22.zip`, 24.2GB). Unlike
the four sources integrated earlier this session (ISLES22, Shifts-MS, MSD Hippocampus, MSD
Prostate), **AMOS22 is deliberately NOT part of the non-CT+OOD focus** — its 15 organs are all
standard abdominal anatomy already in this project's own training vocabulary
(`data/totalseg_classes.py`), and the majority of its volumes are CT. It was pulled as an
**in-distribution reference/control** (per `docs/datasets/eval_strategy_report.md`'s own
framing), not a headline OOD result. Characterization only — not wired into the harness.

## 1. Access — open, but split across two channels

Official challenge (`amos22.grand-challenge.org`), CC-BY-SA 4.0 (`dataset.json`), citation:
Ji et al. 2022, "AMOS: A Large-Scale Abdominal Multi-Organ Benchmark for Versatile Medical
Image Segmentation" (arXiv:2206.08023). Public download (Zenodo mirror of the challenge data),
no DUA. `imagesTs`/`labelsTs` GT is withheld — evaluated only via the challenge submission
server, same pattern as MSD's `imagesTs`.

## 2. Composition — 15 organs, CT-majority, a real MRI minority baked in

nnU-Net-style layout (`imagesTr`/`labelsTr`, `imagesVa`/`labelsVa`, `imagesTs` no-GT), but
**the id number itself encodes modality**: id < 500 is CT, id >= 500 is MRI (readme convention,
verified directly against the actual files below — `dataset.json`'s own `"modality": {"0":
"CT"}` field is misleading/stale, it does NOT mean the release is CT-only).

| split | n total | CT | MRI | GT? |
|---|---:|---:|---:|---|
| train (`Tr`) | 240 | 200 | 40 | yes |
| val (`Va`) | 120 | 100 | 20 | yes |
| test (`Ts`) | 240 | 199 | 41 | no (challenge-server only) |

Verified by parsing every filename's id, not just trusting the readme's rounder headline
numbers (500 CT / 100 MRI across all three splits combined) — close enough (499/101) that the
small discrepancy is almost certainly the readme rounding, not a real count issue.

**15 organs, one merged sex-dependent label**: spleen, right/left kidney, gall bladder,
esophagus, liver, stomach, aorta, postcava (IVC), pancreas, right/left adrenal gland,
duodenum, bladder, and a merged **`prostate/uterus`** class (id 15 — whichever organ applies
to the subject's sex, not two separate classes).

## 3. Geometry — VERIFIED, genuinely heterogeneous, and orientation splits cleanly by modality

Census across all 360 (train+val) label volumes:

| | orientation | shape (vox), min/median/max | spacing (mm), min/median/max |
|---|---|---|---|
| CT (n=300) | **LAS**, uniformly | 512×512×68 / 512×512×104 / 768×768×353 | 0.45×0.45×1.25 / 0.68×0.68×5.0 / 1.07×1.07×5.0 |
| MRI (n=60) | **RAS**, uniformly | 192×60×64 / 320×245×72 / 576×468×512 | 0.69×0.69×0.82 / 1.19×1.19×3.0 / 1.95×3.0×3.0 |

Two things worth flagging for a future converter:
- **Orientation splits cleanly by modality** (every CT case LAS, every MRI case RAS) — unlike
  Shifts-MS's per-COHORT split or NasalSeg's need for manual axis math, this looks like it'd
  reduce to "flip by modality," but should still be verified via `nib.as_closest_canonical`
  per-file rather than hardcoded, the same discipline applied to every other source here
  (multi-center data has a way of surprising you with a mislabeled outlier).
- **Real multi-vendor heterogeneity, not just anisotropy**: CT z-spacing alone spans 1.25-5mm
  (4x), and the MRI shapes span a 5-8x range per axis (192-576 in-plane, 64-512 through-slice)
  — this is a genuinely harder "one converter, one crop-pitch" problem than any single-site
  source integrated so far (all of ISLES22/Shifts-MS/Hippocampus/Prostate are effectively
  single-protocol per modality). A single `crop_spacing_mm` choice would need its own
  grid-occupancy sweep across BOTH the CT and MRI subsets separately, likely with different
  answers for each (this is exactly TotalSegmentator/FLARE22 territory for CT, but a new
  regime for the MRI minority given how much smaller its z-FOV can be, down to 64 vox).

## 4. Labels — near-universal presence, two organs genuinely sex/FOV-dependent

Census across all 360 (train+val) cases combined (CT+MRI pooled — presence is not
meaningfully different by modality at a glance, not broken out further here):

| label | present | % |
|---|---:|---:|
| liver, aorta, postcava, pancreas, R/L adrenal gland, duodenum, esophagus | 359-360/360 | 100% |
| spleen, R kidney | 357-358/360 | 99% |
| stomach | 357/360 | 99% |
| gall bladder | 337/360 | 94% |
| **bladder** | 296/360 | **82%** |
| **prostate/uterus** | 291/360 | **81%** |

Bladder/prostate-uterus are the only organs with real absence rates — plausible causes:
FOV truncation (some protocols don't extend to the pelvis) for bladder, and the merged
prostate/uterus label being sex-dependent (only one of the two exists per subject, and not
every scan's FOV or annotation protocol necessarily includes it) for the latter. Neither looks
like an annotation-quality problem, just genuine anatomical/protocol variability at this scale
(multi-center, multi-disease, multi-phase per the paper's own framing) — the same kind of
per-class presence gap this harness already handles natively (`subjects_for(cls)`) for every
other source.

## 5. Intensity — verified, standard for both modalities

- **CT**: real HU (int16, verified range e.g. [-1024, 1373] on a sample case) — the same
  `normalize_ct`/`DEFAULT_CT_NORM` fixed frame every other CT source in the harness uses,
  no new normalization work needed.
- **MRI**: arbitrary positive scanner units (int16, verified range e.g. [0, 1093] on a sample
  case), not HU — needs the same per-subject `mri_stats`/`normalize_mri` mechanism as
  ISLES22/Shifts-MS/Hippocampus/Prostate, not a fixed frame.

## 6. Fit as an eval set — deliberately a CONTROL, not an OOD result

- **Every organ is already in-vocabulary**: all 15 labels exist in `data/totalseg_classes.py`
  (checked directly) — this is explicitly the report's own "in-distribution reference" framing,
  the opposite of ISLES22/Shifts-MS/Hippocampus/Prostate's genuinely-unseen-class claim. A
  strong AMOS22 Dice would corroborate that low OOD numbers are really about
  novelty-of-class/modality rather than a broken eval harness; a weak one would be a red flag
  about the harness itself, not about OOD generalization.
- **Mixed-modality in one dataset is a genuinely new axis**: no other source integrated this
  session mixes CT and MRI cases under one name — a real "does in-context matching hold up
  when K context / 1 target could be either modality" question, distinct from every
  single-modality source so far. Worth deciding explicitly whether to eval CT-only, MRI-only,
  or mixed if this gets integrated.
- **Large, clean pool**: 240 train + 120 val with GT (360 usable cases, 15 classes = up to
  5400 (subject, class) tasks before accounting for per-class absence) is comfortably the
  largest single pool inspected this session — an order of magnitude above NasalSeg (535
  tasks) or FLARE22 (650 tasks).
- **The real integration cost is the heterogeneity itself** (§3), not a design question like
  MSD Prostate's 4D channels or Hippocampus's sub-mm pitch — it's "do the CT and MRI grid-
  occupancy sweeps separately, verify orientation per-file anyway," which is more legwork than
  a novel design decision.

## 7. Integration — NOT YET DONE (and not currently prioritized)

Nothing wired yet: no `src/providers/amos22.py`, no converter, no
`configs/experiment/3d/dataset/`. Given §3, a converter would very likely need a
**per-modality** `crop_spacing_mm` (CT and MRI subsets swept separately, mirroring how
`TotalSegProvider` already treats `totalsegmri` as a separate modality branch from its CT
default) — structurally this is still a single `NativeGridProvider` subclass, just with two
crop-pitch defaults selected via the same subject-id convention (`< 500` / `>= 500`) already
used to split modality.

Still zipped at `/nfs/.../ANALYSIS_20251122/data/amos22/amos22.zip` (24.2GB) — this
characterization pass extracted only `dataset.json`/`readme.md`, all 360 train+val label
volumes (for the census above, then deleted — cheap to regenerate, ~44MB), and 2 sample image
volumes (one CT, one MRI, kept under `_peek/` for reference) rather than fully unpacking the
whole archive, since no converter exists yet to consume a full unpack. Not converted to the
harness's native-grid `.npy` format — deprioritized behind the non-CT+OOD queue (now fully
closed, see `eval_expansion_status.md`), since this dataset's value is as an in-distribution
sanity check rather than a new OOD data point.
