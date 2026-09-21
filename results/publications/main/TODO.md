# TODO

Open items across the draft (`0_abstract.md` … `4_experiments.md`). Mirrored
`\TODO{}` markers in `ics_3d_medical/sec/*.tex` should be resolved by
porting the answer back from here once decided.

## Abstract

- [ ] Headline result: accuracy at what fraction of baseline (Medverse/Iris)
  compute? (`0_abstract.md`)
- [ ] Which categories benefit most from bi-axial attention (small/thin
  structures, as hypothesized)? (`0_abstract.md`)
- [ ] Generalization headline: one-line summary across other CT cohorts,
  MRI, far-OOD. (`0_abstract.md`)

## Introduction

- [ ] Sharpen the framing once we know which contribution carries the
  paper — bi-axial attention, the cascade, or the combination.
  (`1_introduction.md`)
- [ ] Cite our 2D PatchICL paper with a real venue/arXiv id once available
  (currently a placeholder self-citation, `camaretPatchICL2026` in
  `ics_3d_medical/references.bib`).

## Related Work

- [ ] Position against additional image-label fusion designs if relevant
  (e.g. gated cross-attention, FiLM-style conditioning). (`2_related_work.md`)
- [ ] Expand with additional 3D-specific efficiency baselines once we
  decide which ones we actually compare against. (`2_related_work.md`)

## Method

- [ ] Restate/adjust the backbone description to match the final
  architecture diagram. (`3_method.md`, backbone)
- [ ] Formalize bi-axial attention with an equation once the exact
  factorization (interleaved vs. parallel bi-axial layers) is finalized;
  specify how many layers alternate between the two axes. (`3_method.md`,
  image-label interaction)
- [ ] Specify the region-restriction margin/threshold values, and define
  failure-mode handling for when the coarse level misses the structure
  entirely. (`3_method.md`, fine level processing)
- [ ] Decide whether per-level loss weights $\lambda_\ell$ are uniform or
  scheduled, matching the final training config. (`3_method.md`, training
  objective)

## Experiments

**Setup**
- [ ] State the train/held-out anatomical class split (same
  held-in/held-out protocol as the 2D paper, to test generalization to
  unseen classes, not just unseen subjects).
- [ ] State whether Medverse/Iris are retrained on our split or evaluated
  with released weights, and the number of context pairs $K$ used at
  inference.

**Fixed spacing**
- [ ] Fill in Dice/NSD/time/FLOPs/VRAM table for Iris, Medverse, ours.
- [ ] Break down by anatomical category (organs, bones, vessels, muscles)
  once numbers land; confirm/refute the small-object-strength hypothesis.

**Coarse-to-fine**
- [ ] Fill in ablation table: full model vs. −query prior vs. −region
  restriction vs. −register carry (Dice, time, FLOPs).
- [ ] Produce the accuracy/compute Pareto plot as the region-restriction
  margin is swept (full volume → tight crop), analogous to the
  resolution/FLOPs plot in the 2D paper.

**Generalization**
- [ ] Pick and run on held-out CT cohorts.
- [ ] Pick and run on another modality (e.g. TotalSegmentator MRI) with
  CT-only-trained models.
- [ ] Pick and run far-OOD task(s) to probe cascade robustness when the
  coarse-level prior is unreliable.

## Conclusion / front matter (LaTeX only, not yet drafted in markdown)

- [ ] Summarize headline results once finalized (accuracy vs. baselines,
  contribution of each cascade component, generalization findings).
  (`ics_3d_medical/sec/5_conclusion.tex`)
- [ ] Discuss limitations: dependence of region-restricted processing on
  coarse-level localization accuracy, scaling beyond $M$ levels, extending
  register-carried training to other backbones.
  (`ics_3d_medical/sec/5_conclusion.tex`)
- [ ] Funding/acknowledgments statement.
  (`ics_3d_medical/sec/5_conclusion.tex`)

## Housekeeping

- [ ] Once content stabilizes in the `.md` drafts, port changes into
  `ics_3d_medical/sec/*.tex` and drop the resolved `\TODO{}` markers there.
- [ ] Pick a final title (currently a placeholder in
  `ics_3d_medical/main.tex`).
- [ ] Fill in `\paperID`, `\confYear` in `ics_3d_medical/main.tex` once a
  venue is chosen.
