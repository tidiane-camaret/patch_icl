# Results (draft)

Source of truth for `ics_3d_medical/sections/results_*.tex`. See
`PERFORMANCE_ANALYSIS.md` for the full evidence synthesis (numbers,
citations, contradictions) behind every claim here — that file is the lab
notebook, this file is the chapter prose. Not yet drafted; which specific
numbers to report is a separate decision from this file-organization pass.

## Experimental Setup
**TODO** (`results_setup.tex`): dataset, baselines, metrics, checkpoint
scope (see PERFORMANCE_ANALYSIS.md §"Scope decision" — exp92 line vs.
PatchSetV2), and open caveats (no Iris baseline yet, finetuned-vs-released
confound).

## Fusion (Axis 1)
**TODO** (`results_fusion.tex`): fixed-spacing accuracy/compute vs.
Medverse, thickness-family breakdown (near-parity thin, losing thick, at
17% of Medverse's FLOPs). Candidate spot for the pooling-token ablation
(`arch.pool_token`, fine vs. coarse) since it's a feature-aggregation
question, not a synthetic-data one.

## Cascade (Axis 2)
**TODO** (`results_cascade.tex`): coarse-to-fine gain (+0.177 macro Dice,
4mm→1.5mm), cascade ablations (query prior / region restriction / register
carry — see PERFORMANCE_ANALYSIS.md gap G6, not yet run as a controlled
ablation), and the OOD compounding-error pattern (helps same-modality CT,
hurts far-OOD MRI lesion tasks).

## Synthetic Task Generation (Axis 3)
**TODO** (`results_synth.tex`): synth_gmm ablation (texture noise beats
i.i.d. noise beats real-only on val Dice, seen/unseen split).
