# TODO

Open items across the thesis draft (`0_motivation.md` … `5_conclusion.md`,
`IDEAS.md`). Mirrored placeholders in `ics_3d_medical/sections/*.tex`
should be resolved by porting the answer back from here once decided.

## Content chapters — not yet drafted

- [x] `0_motivation.md` → `motivation.tex`
- [x] `1_fundamentals.md` → `fundamentals_*.tex` (4 subsections)
- [ ] `3_results.md` → `results_*.tex` (4 subsections) — which numbers to
  show is a separate decision from drafting; see `PERFORMANCE_ANALYSIS.md`
  for the evidence base and its open gaps (G1–G8).
- [ ] `4_discussion.md` → `discussion.tex`
- [ ] `5_conclusion.md` → `conclusion.tex`

## Methodology — drafted, small gaps remain

- [x] Axis 3 (synthetic task generation) description corrected 2026-09-23
  — was describing a supervoxel-repainting generator not actually used by
  any cited checkpoint; now describes the real MAISI-bank GMM system plus
  the cross-class correlation / host-anchored-shape calibrations added
  this session. See `PERFORMANCE_ANALYSIS.md` §6.
- [ ] No architecture diagram for Axis 3 (synthetic task generation) —
  `methodology_synth.tex` is text-only, unlike the other three axes.
  Now needs to show: MAISI-bank sampling → cohort-shared GMM draw (±
  cross-class correlation) → i.i.d./multi-octave/heterogeneity noise →
  optional shape-mode overwrite.
- [ ] New (2026-09-23): decide whether the `108`→`135b` checkpoint
  lineage (§6 of `PERFORMANCE_ANALYSIS.md`) is in scope for the thesis at
  all, given it wasn't trained specifically for it and mixes several
  changes vs. `exp92` (architecture + CT/MRI data mix + synth
  calibration) — see gap G1's restatement. If in scope, needs at minimum
  a second training seed for the CT+MRI ablation (its strongest, most
  citable result) before being reported as a headline finding.

## Front matter (`ics_3d_medical/sections/information.tex`, `acknowledgements.tex`)

- [ ] Student ID
- [ ] ORCID (or remove `\orcidlink`)
- [ ] Examiner name
- [ ] Supervisor name(s) for acknowledgements
- [ ] Lab/group name
- [ ] Faculty (depends on degree program — Physics / CS / Scientific
  Computing?)
- [x] Regenerate `assets/cover.pdf` from the official Uni Freiburg LaTeX
  cover kit — source lives in `front_page/` (title + name filled in,
  compiled, copied to `ics_3d_medical/assets/cover.pdf`). Re-run
  `pdflatex Thesis_Titlepage.tex` there and re-copy if the title changes.
  Minor cosmetic overfull-hbox (~3.7pt) on the title line at this length —
  not fixed since `front_page/setup.tex` is the university's fixed
  corporate-design file and shouldn't be edited.

## Pending eval runs — dataset-coverage gaps (found 2026-09-29 auditing Results)

Cross-checking which of the 11 `tab:ood-sources` datasets each results section
actually reports turned up two real gaps that need new *eval* runs (not new
training) on checkpoints that already exist. The Cascade section's own gap
(ISLES22/Shifts-MS/ATLAS v2.0/GNC\_705 missing from `tab:cascade-ablation`)
did **not** need a new run — those numbers already existed in
`experiments/2b_cascade_val` and are now folded into the table. The two gaps
below still need someone to actually launch `eval.py`.

Dataset config names (`configs/experiment/3d/dataset/*.yaml`): `atlas_v2`,
`flare22`, `gnc_kidney`, `isles22`, `nasalseg`, `shifts_ms`, `totalseg`,
`totalseg_mri`.

### Fusion (results_fusion.tex, tab:fusion-staircase) — 5 datasets × 3 checkpoints = 15 runs

- [x] Runs launched 2026-09-29 — all 15 done, exit 0. Numbers + per-case CSV in
  `experiments/1c_fusion_ood_5sources/` (`runs.json` + `samples.csv`); full
  writeup in `docs/logs.md` 2026-09-29 ("Fusion checkpoints (150/151/152)
  evaluated on the 5 OOD sources they'd never seen"). `medverse_bounded_head`/
  `medverse.compile`/`eval_autocast` turned out to be train-time knobs already
  baked into 152's checkpoint, not eval.py flags needed here — confirmed via
  `1b_fusion_ood`'s own wandb-metadata.json args.
- [ ] **Still open**: merge into `results_fusion.tex`'s `tab:fusion-staircase`
  + prose. Not a straight data copy — these 5 sources are single/few-class
  with ~zero TotalSegmentator overlap, so there's no seen/unseen split like
  the existing TotalSeg/FLARE22 rows, and it's unclear whether they should
  fold into the existing "mean of 5 held-out sources" row (→ mean of 10),
  get their own footnoted mean, or be reported separately given 4/5 are
  near-collapse (Dice 0.01–0.11) unlike NasalSeg's 0.27–0.43. Needs an
  authorial decision, not just pasting numbers in.

### Cascade (results_cascade.tex, tab:cascade-medverse) — 10 runs

**(a) 4 runs — Medverse "matched" column for the 4 added lesion datasets.**
ISLES22/Shifts-MS/ATLAS v2.0/GNC\_705 now appear in `tab:cascade-medverse`
with Cascade and Medverse-native values, but the "Medverse matched" column is
`--` for all four: no depth-matched autoregressive Medverse run exists for
them (only `3c_synthetic_cascade`'s 5 datasets — msd_hippocampus, hu_lwk1,
msd_prostate, flare22, nasalseg — have one). Same depth-matched-AR protocol
as those 5. **Corrected 2026-09-29** (the ladders below were wrong in an
earlier version of this note — see `results_cascade.tex`/`results_synth.tex`
`tab:cascade-medverse`/`tab:synth-cascade-realhost`, whose Spacings columns
had the same error and are now fixed): per `2b_cascade_val`'s own
description, isles22/shifts\_ms/atlas\_v2 use a 2-level "whole-volume-coverage"
ladder (fine spacing = each dataset's own tuned `crop_spacing_mm`, coarse =
2$\times$ that), not the 3-level ladder `3c_synthetic_cascade` used for a
*different* experiment ("Medverse with our cascade" paragraph in
`results_cascade.tex`). Actual
ladders, matching what's now in the tables: `[3,1.5]` (isles22 — unchanged),
`[3,1.5]` (shifts\_ms — same fine spacing as isles22, per its own doc),
`[3.8,1.9]` (atlas\_v2), `[6,3,1.2]` (gnc\_kidney — this one is genuinely
3-level, unchanged). So depth-matching Medverse needs `image_size=256`
($M{=}2$) for isles22/shifts\_ms/atlas\_v2, and `image_size=384` ($M{=}3$)
for gnc\_kidney only.

**(b) 6 runs — TotalSeg CT/MRI: the biggest open gap in this section.**
Why these matter more than (a): `tab:cascade-levels`, `tab:cascade-medverse`,
and `tab:cascade-ablation` all report TotalSeg CT/MRI numbers for *both*
Cascade and Medverse, but none of them trace to any archived run. The only
archived run touching checkpoints 145/146/147 on TotalSeg CT/MRI
(`2a_cascade`) uses a declared 2-level `[4,1.5]` ladder at ~78\,ms/sample —
the tables report a 3-level `[6,3,1.5]` ladder at 278/306\,ms/sample, a
latency gap too large to be sampling noise. Medverse's "matched" value for
these two rows (0.103/0.038) is equally untraceable: `3c_synthetic_cascade`
(the source for every other "matched" cell) explicitly scopes TotalSeg CT/MRI
out as "out of scope for this OOD comparison." Fixing this needs 3 things per
dataset (our model, Medverse matched, Medverse native) $\times$ 2 datasets.

**Corrected 2026-09-29** — the commands below were wrong in an earlier
version of this note (`cascade.py` is a library module `run_cascade`/
`evaluate_cascade` used *by* `eval.py`/`train.py`, not a standalone script
with its own `dataset=` CLI group; every archived cascade run actually goes
through `eval.py experiment=<config> ... data.cascade_spacings=[...]`, per
the worked examples in `docs/datasets/{isles22,gnc_kidney_lesions}.md`).
Also: **`eval.py dataset=totalseg` defaults to `data.p_synth=1`** (old
seeds3d supervoxel synth, unrelated to `synth_gmm` — see memory
`project_eval_totalseg_psynth_gotcha.md`), which silently evaluates on fake
anatomy unless overridden; `dataset=totalseg_mri` is already safe
(`p_synth: 0` by default in its own config), so only the CT run needs the
explicit override.

```
# our model (checkpoint 146, "Cascade" everywhere in this section) —
# the 146_cascade_randomfg_predprior_regoff experiment config already sets
# p_synth=0 (real anatomy), kept explicit below as a defensive belt-and-braces
experiments/3d/eval.py experiment=146_cascade_randomfg_predprior_regoff eval.model=patchset3d \
  eval.checkpoint=<146 best.pt> data.source=totalseg data.p_synth=0 data.crop_spacing_mm=6 \
  data.cascade_spacings=[6,3,1.5] data.mask_downsample=occupancy data.val_classes=all eval.split=test
experiments/3d/eval.py experiment=146_cascade_randomfg_predprior_regoff eval.model=patchset3d \
  eval.checkpoint=<146 best.pt> data.source=totalseg_mri data.crop_spacing_mm=6 \
  data.cascade_spacings=[6,3,1.5] data.mask_downsample=occupancy data.val_classes=all eval.split=test

# Medverse "matched" — depth-matched native-AR, image_size = 128 * 2^(M-1) = 384 for our M=3 ladder
# (same formula 3c_synthetic_cascade already used for hu_lwk1/msd_prostate/flare22, its other M=3 rows)
experiments/3d/eval.py dataset=totalseg     eval.model=medverse data.p_synth=0 data.image_size=[384,384,384] eval.split=test eval.n_subjects=null
experiments/3d/eval.py dataset=totalseg_mri eval.model=medverse data.image_size=[384,384,384] eval.split=test eval.n_subjects=null

# Medverse "native" — Medverse's own standard depth, same image_size=256 used for every other native row
experiments/3d/eval.py dataset=totalseg     eval.model=medverse data.p_synth=0 data.image_size=[256,256,256] eval.split=test eval.n_subjects=null
experiments/3d/eval.py dataset=totalseg_mri eval.model=medverse data.image_size=[256,256,256] eval.split=test eval.n_subjects=null
```
No `eval.checkpoint` for the Medverse runs — released weights load automatically.
146 = wandb run `wrwzs5fs` (`2a_cascade`/`2b_cascade_val`); resolve its
`best.pt` path via that run or `results/checkpoints/`. Whoever runs these
should double check `eval.sw_overlap`/`eval.autocast` against the actual
wandb-recorded args of `3c_synthetic_cascade`'s `medverse_depthmatched_*`
runs before launching — not confirmed here that those two flags were used
for the depth-matched runs specifically (they're documented for the
separate *native*-AR sweep in `eval_expansion_status.md`, which may or may
not be identical).

Once these 6 exist, recompute `tab:cascade-medverse`'s "Mean" latency row
sample-weighted across the full 7 datasets instead of leaving TotalSeg CT/MRI
out of it. This isn't a small effect: tested 2026-09-29 on the 5 datasets
that already are traceable (FLARE22, HU\_LWK1, MSD Prostate, MSD Hippocampus,
NasalSeg, via `2b_cascade_val`), unweighted mean is 257.8\,ms vs.\
sample-weighted 209.2\,ms (MSD Hippocampus and NasalSeg have large sample
counts and low latency, pulling the weighted mean down substantially). Same
reweighting already applied to Fusion's `tab:fusion-staircase` "Mean of 10
held-out sources" row — see that table's footnote for the method.

### Synth (results_synth.tex, tab:synth-cascade-realhost) — 2 runs

Checkpoint `168_cascade_real_host_all_shapes` (Synth + Cascade, resumed from
`167`'s real-host checkpoint, see `docs/logs.md` 2026-09-28) was never
evaluated on TotalSeg CT or TotalSeg MRI — `3c_synthetic_cascade/runs.json`
explicitly notes these are "out of scope" for that experiment. Cheaper than
the Fusion gap: just 2 runs, cascade ladder `[6,3,1.5]` (matching `146`'s own
in-distribution ladder), same protocol as `2a_cascade`/`2b_cascade_val`
(`eval.split=test`, `data.mask_downsample=occupancy`).

```
experiments/3d/cascade.py dataset=totalseg      eval.checkpoint=<168 best.pt> cascade_spacings=[6,3,1.5]
experiments/3d/cascade.py dataset=totalseg_mri  eval.checkpoint=<168 best.pt> cascade_spacings=[6,3,1.5]
```

## Housekeeping

- [ ] Decide whether to keep the glossary/symbols list (template README
  suggests dropping it for a simpler thesis) or populate it for real terms
  used in Methodology (RoPE, ICL, GMM, etc.)
- [ ] Once content stabilizes in the `.md` drafts, port changes into
  `ics_3d_medical/sections/*.tex` and drop resolved `\todo{}` markers there
