# TODO

Open items across the thesis draft (`0_motivation.md` … `5_conclusion.md`,
`IDEAS.md`). Mirrored placeholders in `ics_3d_medical/sections/*.tex`
should be resolved by porting the answer back from here once decided.

## Content chapters — not yet drafted

- [ ] `0_motivation.md` → `motivation.tex`
- [ ] `1_fundamentals.md` → `fundamentals_*.tex` (4 subsections)
- [ ] `3_results.md` → `results_*.tex` (4 subsections) — which numbers to
  show is a separate decision from drafting; see `PERFORMANCE_ANALYSIS.md`
  for the evidence base and its open gaps (G1–G8).
- [ ] `4_discussion.md` → `discussion.tex`
- [ ] `5_conclusion.md` → `conclusion.tex`

## Methodology — drafted, small gaps remain

- [ ] No architecture diagram for Axis 3 (synthetic task generation) —
  `methodology_synth.tex` is text-only, unlike the other three axes.

## Front matter (`ics_3d_medical/sections/information.tex`, `acknowledgements.tex`)

- [ ] Student ID
- [ ] ORCID (or remove `\orcidlink`)
- [ ] Examiner name
- [ ] Supervisor name(s) for acknowledgements
- [ ] Lab/group name
- [ ] Faculty (depends on degree program — Physics / CS / Scientific
  Computing?)
- [ ] Regenerate `assets/cover.pdf` from the official Uni Freiburg LaTeX
  cover kit (currently the template author's own leftover sample,
  "Analysis of Heatpumps" — can't be text-edited directly, needs the
  external source project)

## Housekeeping

- [ ] Decide whether to keep the glossary/symbols list (template README
  suggests dropping it for a simpler thesis) or populate it for real terms
  used in Methodology (RoPE, ICL, GMM, etc.)
- [ ] Once content stabilizes in the `.md` drafts, port changes into
  `ics_3d_medical/sections/*.tex` and drop resolved `\todo{}` markers there
