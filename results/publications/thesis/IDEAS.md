# Ideas (thesis-specific working notes)

Freeform notes on framing/structure choices — distinct from `TODO.md`
(concrete action items) and `PERFORMANCE_ANALYSIS.md`/`arch.md` (evidence
and implementation reference).

## Framing decided so far

- **Scope:** 3D work only. The 2D PatchICL paper (concurrent CVPR
  submission) stays cited as related/prior work, not its own chapter.
- **Three-axis emphasis** (this thesis's organizing principle, distinct
  from the article's "attention design + cascade" framing): image–label
  feature fusion, coarse-to-fine cascade, synthetic task generation. Every
  content chapter (Methodology, Results) is split one file/section per
  axis rather than mirroring the article's structure directly.
- **Title:** "Efficient In-Context Learning for 3D Medical Image
  Segmentation" — foregrounds the compute-efficiency result (17% of
  Medverse's FLOPs), currently the strongest empirical claim.
- **Unresolved experiments:** not a blocker for drafting. Thesis text can
  present the current preliminary state (exp92 line) with gaps flagged as
  limitations/future work in the Discussion chapter, rather than waiting
  for G1–G8 (see `PERFORMANCE_ANALYSIS.md`) to resolve first.

## Open questions / ideas not yet decided

- Fundamentals depth: how much pedagogical detail does the Related Work
  subsection need vs. just citing? (thesis readers aren't assumed to know
  UniverSeg/Iris/Medverse already, unlike a CVPR reviewer)
- Whether the pooling-token ablation (`arch.pool_token`) belongs under
  Results §Fusion (architecture/feature-aggregation framing) or gets its
  own subsection — currently parked under Fusion, see `3_results.md`.
- No architecture diagram exists yet for Axis 3 (synthetic task
  generation). Worth sketching one (MAISI-bank sampling → cohort-shared
  GMM repaint, optionally cross-class-correlated → i.i.d./multi-octave/
  heterogeneity noise → optional procedural shape-mode overwrite) to
  match the other three axes' figures.
- **New (2026-09-23):** a third checkpoint lineage (`108`→`135b`) now has
  the best OOD numbers of anything trained so far (PERFORMANCE_ANALYSIS.md
  §6), on top of an unrelated CT+MRI joint-training result that's the
  cleanest ablation in the whole evidence base. Neither was trained with
  the thesis in mind, and both currently sit at N=1 seed. Open framing
  question: does this lineage become "Ours" (supersedes `exp92` as the
  reported architecture), stay a Discussion/future-work pointer, or get
  promoted only if a replicate seed confirms the CT+MRI result first?
- Degree program / faculty for the front matter (Physics? CS? Scientific
  Computing?) still unknown — affects `information.tex`'s Faculty line.
