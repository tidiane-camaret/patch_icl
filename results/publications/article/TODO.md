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

See `PERFORMANCE_ANALYSIS.md` for the full evidence synthesis (numbers,
citations, contradictions) behind every item below. That file's confirmed
findings (thickness-driven accuracy story, in-distribution cascade gain,
cascade-compounds-error-OOD pattern) are already grounded in `4_experiments.md`
and don't need re-deriving — the items here are what's still open.

**Blockers (resolve first, everything else depends on these)**
- [ ] **G1 — scope decision:** is "Ours" the converged
  `exp92_multisource_synth` cascade line, or the newer PatchSetV2 line
  (currently unconverged — epoch-0 to dice 0.235, well below exp92's
  0.43–0.58)? All numbers below assume exp92 until this is decided.
- [ ] **G5 — get an Iris baseline running.** Zero Iris numbers exist
  anywhere in the repo (no checkpoint, no eval, no compute bench) despite
  Related Work arguing against its fusion design directly. Either produce
  one (released weights per `docs/README.md`'s repo pointer) or scope all
  comparative claims to "vs. Medverse" only.
- [ ] **G2 — reconcile the two contradictory fixed-spacing notebooks**
  (`20_medverse_patchset3d_comp.py` crop=false → Medverse wins macro Dice
  0.078 vs 0.048, at 17% of Medverse's FLOPs; `36_patchset_medverse_multiscale.py`
  occupancy-mask setting → we win 0.562 vs 0.249). nb20 is cross-validated
  by memory, nb36 is not — fix or retire nb36 rather than citing either
  without resolving why they disagree.
- [ ] **G8 — resolve the finetuned-vs-released-zero-shot confound** in the
  OOD comparison table (our exp92 is CT-only finetuned; Medverse there is
  released multi-modal weights, no finetune). Needed before claiming
  Medverse "generalizes better" on the 4/7 sources where it currently wins.

**Setup**
- [ ] State the train/held-out anatomical class split (same
  held-in/held-out protocol as the 2D paper). Training curves already show
  a real seen/unseen gap (e.g. dice_seen 0.68 vs dice_unseen 0.51), so the
  split exists operationally — just needs stating explicitly.
- [ ] State $K$ (context pairs) used at inference for every reported number
  — not currently stated anywhere we found.

**Fixed spacing**
- [ ] **G4 — build one benchmark that reports Dice/NSD + time/FLOPs/VRAM
  together** for a single matched checkpoint/setting. Currently accuracy
  (nb20) and compute (`results/presentations/perf/RESULTS.md`) come from
  different checkpoints and different eval harnesses; VRAM specifically is
  never measured inside `evaluate.py`/`cascade.py` at all.
  - [ ] Correct the compute framing once G4 lands: current architecture
    (77–110M params) is comparable to/larger than Medverse's 71.1M — the
    "15× smaller" claim is stale (from an old ~4.7M-param checkpoint) and
    must not be reused. What survives at matched bf16 precision is a
    ~1.05–1.3× latency edge and a FLOPs reduction tied to the iris-style
    decoder path specifically, not overall model size.
- [ ] Break down by anatomical category (organs, bones, vessels, muscles)
  in addition to the thickness-quartile split already confirmed
  (near-parity thin, −0.06 to −0.18 thick).

**Coarse-to-fine**
- [ ] **G6 — actually run the ablation table**: full model vs. −query prior
  vs. −region restriction vs. −register carry (Dice, time, FLOPs). What
  exists today is indirect evidence only (Medverse's own query-prior hurts
  its cascade 7/7 OOD sources) — not a controlled ablation of our design.
- [ ] **G7 — run the region-restriction margin sweep** (full volume → tight
  crop) for the Dice-vs-compute Pareto plot; not run at all yet
  (`37_patchset_spacing_locator.py` sweeps spacing, not restriction margin).
- [ ] **G3 — benchmark a full multi-level cascade run end-to-end**
  (time/FLOPs/VRAM including re-crop/GPU-realize overhead between levels);
  only per-level single-pass costs exist today.
- [ ] Finish register-carry training (epoch 36/140 at last check) and get
  the missing GNC_705 cascade cell (3 failed attempts, suspected
  `torch.compile` hang) resolved or explicitly excluded before citing that
  ablation.
- [ ] Verify whether `99_iris_decoder_plainconv_doubling`'s low fully-trained
  dice (0.056, well below the exp92 line's 0.43–0.58) is a genuine
  finding or a training-config bug before using it as an ablation result.

**Generalization**
- [ ] GNC_705 (kidney lesion, MRI) and ISLES22 (stroke lesion, MRI) already
  cover the "far OOD" role and both show the cascade-compounds-error
  pattern — confirm this satisfies the planned far-OOD slot or pick an
  additional source.
- [ ] Other-modality slot is already covered (Shifts-MS, MSD Hippocampus,
  MSD Prostate, ATLAS v2.0) — flag ATLAS v2.0's result (Dice 0.028) as
  possibly confounded by label-corruption, not a clean failure case.
- [ ] Decide whether additional held-out CT cohorts beyond HU_LWK1 are
  needed, or whether HU_LWK1 (the only source where cascade helps) is
  sufficient as the "other CT datasets" evidence.

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
