# crossMoDA (Cross-Modality Domain Adaptation Challenge)

*Vestibular schwannoma + cochlea segmentation, ceT1 (source) → hrT2 (target) MRI.* Downloaded
and inspected 2026-09-13, following up on `docs/datasets/eval_strategy_report.md`'s Tier-2
cross-modality pick. **Genuinely OOD classes, verified against our own training vocabulary**:
neither vestibular schwannoma (a tumor) nor cochlea (an inner-ear substructure) exists anywhere
in `data/totalseg_classes.py`. Not yet wired into the harness — characterization only.

## 1. Access — CORRECTED this session: genuinely open, not gated

Earlier this session, crossMoDA was flagged gated based on `crossmoda.grand-challenge.org/
Data/` returning HTTP 403. That was checking only the challenge-portal mirror. **The dataset's
real archival copy is on Zenodo, record 4662239, `access_right: open`, CC-BY-4.0, no login** —
found via a HuggingFace search that surfaced a community NIfTI mirror
(`YongchengYAO/CrossMoDA-Lite`) whose README pointed back to the official Zenodo source.
Verified directly via the Zenodo API before trusting it. Downloaded from Zenodo, not the
third-party mirror:
```bash
curl -L -o crossmoda_training.zip   "https://zenodo.org/api/records/4662239/files/crossmoda_training.zip/content"    # 6.07GB
curl -L -o crossmoda_validation.zip "https://zenodo.org/api/records/4662239/files/crossmoda_validation.zip/content"  # 603MB
```
**Lesson for future triage**: one portal returning 403 is not proof a dataset needs
credentials — always check for a separate archival DOI (Zenodo/Figshare/OSF) before concluding
something is gated.

## 2. Composition — only the SOURCE domain is labeled

This is a domain-ADAPTATION dataset, not a plain segmentation set — the labeling asymmetry is
the whole point of the challenge:

| split | domain | sequence | n | labeled? |
|---|---|---|---:|---|
| `source_training` | source | ceT1 (contrast-enhanced T1) | 105 | **yes** |
| `target_training` | target | hrT2 (high-res T2) | 105 | no (unpaired, for domain adaptation) |
| `target_validation` | target | hrT2 | 32 | no (challenge-server only) |

**Only the 105 `source_training` ceT1 cases have ground truth** — the 105 `target_training`
and 32 `target_validation` hrT2 volumes are deliberately unlabeled (the challenge's task is
adapting a ceT1-trained model to unlabeled hrT2). This is a fundamentally different shape than
every other source pulled this session: there's no "target domain" eval possible without
either (a) evaluating only on the labeled ceT1 domain (loses the cross-modality argument
entirely) or (b) needing an external hrT2 GT source. `docs/datasets/eval_strategy_report.md`'s
framing ("ceT1 → hrT2... an extreme example of domain shift") describes a DA training
protocol, not a segmentation eval recipe — this dataset only gives an in-context task a labeled
ceT1 pool to draw K context examples AND targets from; there's no clean way to test "does the
model generalize to hrT2" without separate hrT2 GT this release doesn't provide.

(Note also: this is the crossMoDA **2021** release per the Zenodo record's file names/dates —
the eval-strategy report's 105/105/137 case counts match a LATER edition; this release has 32
in `target_validation`, not 137. Fine for characterization, just don't assume case counts
transfer across crossMoDA years.)

## 3. Geometry — VERIFIED, high-res, small FOV, LPS orientation

All 105 labeled cases: orientation **LPS** uniformly (needs `nib.as_closest_canonical`).

| | shape (vox) | spacing (mm) |
|---|---|---|
| ceT1 (source, all 105) | 512×512×120 (min) to 512×512×160 (max) | 0.41×0.41×1.0 to 0.41×0.41×1.5 |

Very high in-plane resolution (0.41mm) — a temporal-bone-focused head MRI protocol, sharper
than any other MRI source pulled this session (ISLES22/Shifts-MS/Hippocampus are all ≥0.6mm
in-plane).

## 4. Labels — TWO classes with genuinely different structural character

| label | meaning | present | volume (mm³) | connected components |
|---|---|---:|---|---|
| 1 | vestibular schwannoma (tumor) | 105/105 (100%) | 44 – 9,801 (min/max) | **always 1** (unilateral, matches the disease being studied) |
| 2 | cochlea | 105/105 (100%) | 72 – 224 (min/max) | **2 in 102/105 cases** (bilateral — BOTH ears share one label value) |

**Found and verified a structural property that matters a lot for any converter**: cochlea is
annotated **bilaterally under a single label index** — both left and right cochleae get value
`2` in the same mask, confirmed via connected-component counting (`scipy.ndimage.label`) across
all 105 cases: schwannoma is always exactly 1 component (unilateral tumor, as expected
clinically — vestibular schwannoma is essentially always one-sided), cochlea is 2 components in
102/105 cases (3 cases show 3-4 components, likely annotation fragmentation of the same two
structures rather than a real third site — not investigated further here). This means:
- A **centroid-of-mass calculation across the whole label** (the standard trick used for every
  other single-object source integrated this session) is **meaningless for the cochlea class**
  — the centroid of two widely-separated bilateral objects lands in empty tissue between the
  ears, not on either actual cochlea. Confirmed directly: one sample case has cochlea half-
  extent-from-centroid of ~42mm on the axis separating the ears, vs. the individual structure's
  true ~7.5mm diameter.
  Any grid-occupancy sweep or in-context crop strategy for this class needs to either (a) split
  cochlea into `cochlea_left`/`cochlea_right` at conversion time (same "split-the-ambiguous-
  label" pattern already used for MSD Prostate's channel split, applied to laterality instead
  of modality), or (b) pick a connected-component-aware center (nearest component to some
  reference point) rather than a naive full-mask centroid — (a) is almost certainly the
  cleaner fix, matching the general pattern this harness already uses when one label secretly
  contains two distinct things.
- `vestibular schwannoma` alone is well-behaved for the standard crop pipeline (single
  compact object, comparable in size/character to MSD Hippocampus's single-structure regime).

## 5. Intensity — verified standard for MRI

ceT1 is uint16, raw scanner units, verified range e.g. [0, 2634] on a sample case — needs the
same per-subject `mri_stats`/`normalize_mri` mechanism as every other MRI source in this
harness, not a fixed frame.

## 6. Fit as an eval set

- **Genuinely unseen classes on both counts**: neither structure exists in TotalSegmentator's
  or our own vocabulary — a clean claim, same tier as ISLES22/Shifts-MS/Hippocampus/Prostate.
- **The tumor class (schwannoma) is a clean, mechanical port** — single compact object, same
  shape as MSD Hippocampus's regime. The **cochlea class needs the laterality-split fix
  before it's usable** (§4) — not a mechanical port, a real (if small) design decision, closer
  in spirit to MSD Prostate's channel-split question than to a plain converter.
- **The cross-modality argument itself is NOT directly testable from this release** (§2) — only
  ceT1 has GT. An eval here would be "does the model do well at localizing/segmenting
  schwannoma+cochlea in a high-res temporal-bone ceT1 scan," a clean unseen-class claim on its
  own, but NOT the "does it survive ceT1→hrT2 domain shift" claim the report's framing implies,
  since there's no hrT2 ground truth in this release to test against.
- **Smaller pool than most sources this session**: 105 labeled cases × 2 classes = up to 210
  (subject, class) tasks — smaller than NasalSeg (535) or FLARE22 (650), similar order to
  Shifts-MS (46) scaled up, well below AutoPET-III's 1038.

## 7. Integration — NOT YET DONE

No provider/converter/config yet. Structurally this would be:
1. A **converter decision** for cochlea: split into `cochlea_left`/`cochlea_right` via
   connected-component labeling at conversion time (straightforward — `scipy.ndimage.label` +
   assign by which side of the volume's midline each component falls on), OR keep it merged
   and accept the centroid-based crop is unreliable for that one class (not recommended, given
   how easy the split is).
2. Otherwise a standard `NativeGridProvider` subclass (`MODALITY="mri"`), same shape as every
   other MRI source integrated this session — LPS→RAS via `nib.as_closest_canonical`,
   per-subject `mri_stats`.
3. A `crop_spacing_mm` grid-occupancy sweep still needs doing PROPERLY once cochlea is split
   (the sweep run for this doc used the un-split bilateral label, so its "many clipped at small
   pitch" result is an artifact of the centroid problem in §4, not a real pitch requirement —
   re-run after the laterality split, expect something well under 0.5mm given how small both
   structures actually are individually, e.g. 0.41mm in-plane spacing itself suggests staying
   close to native resolution).
4. **Eval scope decision**: report schwannoma+cochlea(split) on the labeled ceT1 domain only
   (honest about what this release supports), don't claim a cross-modality generalization
   result from this dataset alone.

Raw zips kept at `/nfs/.../ANALYSIS_20251122/data/crossmoda/` (`crossmoda_training.zip` 6.07GB,
`crossmoda_validation.zip` 603MB) — not yet extracted/converted beyond the `source_training`
labels pulled for this characterization pass.
