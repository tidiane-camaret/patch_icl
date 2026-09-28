# Results (draft)

Source of truth for `ics_3d_medical/sections/results_*.tex`. See
`PERFORMANCE_ANALYSIS.md` for the full evidence synthesis (numbers,
citations, contradictions, and everything explicitly flagged as *not*
solid enough to cite) behind every claim here. Numbers below are the ones
that file's own "what's already solid enough to write into the paper now"
list clears — anything with an open gap (G1–G11) keeps its caveat inline
rather than being silently reported as settled.

## Experimental Setup

**Checkpoint scope.** Two lineages are reported here, kept explicitly
separate (gap G1 — not yet resolved which one "Ours" means for the final
thesis):

- **`exp92_multisource_synth`** (dual attention, query-prior init,
  register carry gated off) — the source of every Fusion and Cascade
  number below. `PatchSetV2` (configs 100/101/103/104) is not converged
  (dice 0.006–0.235 at last check, well below `exp92`'s 0.43–0.58 range)
  and is excluded entirely.
- **`108`→`130`→`135b`** — a separate, more recent lineage descending from
  `exp92` (adds `cascade_registers`, `pool_token`, synth texture noise),
  the source of every Synthetic Task Generation number below. Trained
  after and independently of the Fusion/Cascade experiments, at N=1 seed
  throughout (gap G11) — read as a promising newer result, not a
  replacement for the exp92 numbers above it in this chapter.

**Baselines.** Medverse (released weights, and — where noted — finetuned
on the same TotalSegmentator split) is the only baseline with any
empirical numbers anywhere in the repo. No Iris baseline exists (gap
G5) — the Related Work argument against Iris's pixel-shuffle fusion is
architectural, not empirical.

**Metrics.** Dice (macro, per-class-then-averaged unless stated
per-sample) and Normalized Surface Dice (NSD) for accuracy; GFLOPs,
wall-clock latency, and peak VRAM for compute, where available (gap G4 —
VRAM is never measured inside the real eval path, only in a standalone
B=1/K=1 benchmark harness, so it should not be read as representative of
a real K>1 eval run).

**Datasets.** In-distribution: TotalSegmentator (117 classes). Held-out
OOD: 7 eval-expansion sources spanning CT and MRI and classes outside the
TotalSegmentator vocabulary — ISLES22 (MRI DWI, stroke lesion), Shifts-MS
(MRI FLAIR, MS lesion), MSD Hippocampus (MRI T1), MSD Prostate (MRI
T2+ADC), ATLAS v2.0 (MRI T1, chronic stroke lesion — ⚠️ provenance
caveat, unofficial data mirror), GNC_705 (MRI Dixon, kidney lesion),
HU_LWK1 (**CT**, vertebra measurement ROI — the only CT OOD source).

## Fusion (Axis 1)

**Medverse vs. our architecture (single-axis fusion) across six datasets.** The
cold-transformer chain (`150`/`151`/`152`) trains all three architectures under an
identical data pipeline, augmentation recipe, and optimizer/schedule. Rather than
in-distribution accuracy followed by a separate generalization check, all six evaluation
datasets are reported together from one consistent protocol: `eval.py`, fixed 3mm
single-level crops, fp32, `test` split, each model's own best checkpoint (no epoch
matching across arms — `150`/`151` stopped within 3 epochs of each other, `152` trained
to full convergence). Rows are ordered by distance from the training distribution
(TotalSeg CT itself → held-out modality/dataset/vocabulary):

| dataset | Medverse (`152`) | Ours — single-axis fusion (`151`) |
|---|---:|---:|
| TotalSeg CT — seen | 0.348 | **0.574** |
| TotalSeg CT — unseen | 0.221 | **0.408** |
| TotalSeg MRI | 0.178 | **0.232** |
| FLARE22 | 0.518 | **0.684** |
| HU_LWK1 | **0.146** | 0.079 |
| MSD Prostate | **0.377** | 0.205 |
| MSD Hippocampus | 0.172 | **0.184** |
| mean of 5 OOD sources | **0.278** | 0.264 |
| params | **71.1M** | 74.1M |
| GFLOPs | **2362.6** | 3973.1 |
| latency (per-sample, fp32) | 72.7 ms | 73.0 ms |

Single-axis wins 4/6 datasets (both TotalSeg splits, TotalSeg MRI, FLARE22) but Medverse
wins on HU_LWK1 and MSD Prostate, and the two are close on MSD Hippocampus — averaged
across the 5 OOD sources (unweighted by class count), Medverse actually edges ahead
(0.278 vs. 0.264). Latency and GFLOPs are both measured fp32 here, so they're internally
consistent with each other but not with the bf16 numbers reported elsewhere in this
chapter. This sweep carries no `query_prior` asymmetry between the two models (single-level
eval never touches it), unlike the in-distribution-only comparisons in this project's
earlier history.

**Single-axis vs. dual attention, same six datasets.** A direct ablation of the
architecture's own dual-attention design (position axis, cross-context + modality axis,
within-cell img↔mask attention) against an IRIS-style early-fusion alternative: img and mask are
merged into one token per cell via a PixelShuffle trick *before* the transformer
(`arch.dual_axis=false`), with only the position axis (cross-context) attention remaining.
Compute-matched (`151`'s transformer depth raised to `l=9`; params are +52% higher for
single-axis since position-axis-only layers can't buy back modality-axis compute per-parameter).
Both arms warm-start only the encoder/decoder from a shared checkpoint, with the entire
in-context reasoning core randomly initialized for both. Same protocol and dataset order
as the table above:

| dataset | single-axis fusion, l=9 (`151`) | dual attention (`150`) |
|---|---:|---:|
| TotalSeg CT — seen | 0.574 | **0.582** |
| TotalSeg CT — unseen | 0.408 | **0.426** |
| TotalSeg MRI | 0.232 | **0.259** |
| FLARE22 | 0.684 | **0.693** |
| HU_LWK1 | 0.079 | **0.102** |
| MSD Prostate | 0.205 | **0.247** |
| MSD Hippocampus | 0.122 | **0.184** |
| mean of 5 OOD sources | 0.264 | **0.297** |
| params | 74.1M | **48.7M** |
| GFLOPs | 3973.1 | **3892.5** |
| latency (per-sample, fp32) | **73.0 ms** | 72.5 ms |

Dual attention wins all 6/6 datasets — the cleanest, most consistently-generalizing result in
this axis. Not yet settled: neither arm has reached its full training budget, this is N=1
seed per arm, and the params mismatch (+52% for single-axis) leaves a residual capacity
confound even after compute-matching.

*(The comparison below is from an older checkpoint lineage — `exp92`'s whole-body
`use_crop=false` protocol — kept for the thickness-driver analysis and compute table that
build on it; see Experimental Setup's gap G1 on which lineage "Ours" should mean for the
final thesis.)*

**Fixed-spacing accuracy vs. Medverse**, `use_crop=false` whole-body
protocol, 3897 samples, matched checkpoints (`vc7kfdto` / `94nlx7yw`):

| | Medverse | Ours (exp92) |
|---|---:|---:|
| macro Dice (117 classes) | **0.078** | 0.048 |
| classes won | **66/117** | 51/117 |
| per-sample win rate | **49.7%** | 40.2% |
| complete-miss rate | 1.6% | **0.5%** |
| GFLOPs | 2362.6 | **398** (17%) |
| latency | 69.6 ms | 85 ms |

Medverse wins overall macro Dice and more classes; we have a lower
complete-miss rate and win more often per-sample than the macro gap
alone suggests (both true simultaneously: Medverse wins bigger when it
wins, we fail less catastrophically). We run at **17% of Medverse's
FLOPs** at roughly matched wall-clock.

**What drives the gap — thickness, not identity.** Splitting by shape
family: `thick_tube` Δ−0.182, `thick_blob` Δ−0.095 (Medverse wins big);
`mid_sheet` Δ+0.016, `mid_tube1` Δ+0.008 (near parity). By thickness
quartile: thin half Δ−0.005 (**near parity**), thick half Δ−0.059
(Medverse pulls ahead). Independently replicated in the 2D companion
work (patch-based context selection wins thin/small structures) and in a
dedicated 3D geometric-driver study confirming the predictor is object
*thickness*, not identity or intensity contrast.

**Compute at the current architecture** (2026-09-20 benchmark, B=1/K=1/
128³, RTX PRO 6000 Blackwell, native precision per model — **do not
combine with the params/FLOPs numbers above**, which are from an older,
~4.7M-parameter checkpoint; the "15× smaller" framing from that era no
longer holds):

| model | params | GFLOPs | latency | peak VRAM |
|---|---:|---:|---:|---:|
| Medverse (fp32) | 71.1M | 2362.6 | 69.6 ms | 3.30 GB |
| Medverse (bf16) | — | — | 45.5 ms | 2.32 GB |
| Ours, conv-decoder (exp92) | 48.0M | 3891.1 | 43.5 ms (bf16) | 2.50 GB |
| Ours, iris-decoder | 71.9M | uncounted | 36.4 ms (bf16) | 3.36 GB |
| Ours, v2 (101) | 77.4M | 1230.3 | 35.9 ms (bf16) | 2.67 GB |
| Ours, v2 (103, cascade, single-level) | 109.6M | 1993.4 | 42.6 ms (bf16) | 3.78 GB |

At the current architecture, parameter count is comparable-to-larger than
Medverse (48–110M vs. 71.1M), and VRAM is roughly at parity or worse —
**the honest compute claim is "cheaper decoder path via region-restricted
attention" (v2/101: 1230 vs 2363 GFLOPs, ~52%), not "smaller model."** At
matched (bf16) precision the wall-clock advantage narrows to ~1.05–1.27×
from the native-precision 1.6–1.9× — most of that gap is a precision
artifact, not architecture.

## Cascade (Axis 2)

**In-distribution, coarse-to-fine helps cleanly**, replicated two ways:

- Spacing sweep 4mm→1.5mm (3897 samples, 117 classes): macro Dice
  **0.361 → 0.538 (+0.177)**. Locator containment 0.922 mean vs. 0.970
  oracle (0.048 gap = crop-ceiling headroom); containment does *not*
  predict fine-level accuracy — well-localized classes still fail at the
  segmentation step once the coarse crop is roughly right, not at
  localization.
- `exp92` 3-level cascade eval (1157 samples, `[6,3,1.5]`mm): dice
  r1.5/r3/r6 = **0.5716 / 0.5088 / 0.3889** — monotonic coarse-to-fine
  gain, 11673 GFLOPs / 666 ms for the full 3-level pass.

**OOD: cascade helps only when the coarse level is already reasonable and
same-modality; hurts when the coarse level is already failing** — the
clearest generalization finding in this chapter, and (see Synthetic Task
Generation below) one that generalizes across checkpoint lineages:

| source | modality | single-level Dice | cascade Dice | verdict |
|---|---|---:|---:|---|
| HU_LWK1 (L1 vertebra) | **CT** | 0.116 | **0.129** `[6,3,1]` | **helps** |
| ISLES22 | MRI DWI | 0.054 | 0.046 `[3,1.5]` | hurts |
| GNC_705 (kidney lesion) | MRI Dixon | 0.059 | 0.019 `[6,3,1.2]` | hurts badly |

The same compounding-error mechanism shows up in an independent,
indirect check: feeding Medverse's own imperfect coarse prediction
forward as a query prior hurts its cascade universally, 7/7 OOD sources
tested, sometimes catastrophically (MSD Hippocampus 0.691→0.050) — this
is evidence the compounding-error pattern is **generic to coarse-to-fine
cascades**, not specific to this architecture, though it has not yet been
isolated as a controlled ablation on our own query-prior design (gap G6).

**Ours vs. Medverse, six datasets, accuracy and compute.** "Ours" here is
the predicted-prior checkpoint (the best-performing arm — see the
query-prior ablation below); "Medverse" is released weights, native
autoregressive inference, its own coarse-to-fine mechanism, depth
level-matched to our own ladder per source (matching actually *hurts*
Medverse, see finding below — reported anyway as the fairer, matched
comparison). Four sources (ISLES22, Shifts-MS, ATLAS v2.0, GNC\_705) are
excluded — both models score too low there to be informative — and
TotalSeg CT (in-distribution) is excluded for a different reason: its
Medverse run was not completed (gap, not yet closed):

| dataset | Ours | Medverse | latency (Ours) | latency (Medverse) |
|---|---:|---:|---:|---:|
| TotalSeg MRI | **0.506** | 0.038 | 306ms | 4842ms |
| HU\_LWK1 | **0.152** | 0.020 | 439ms | 5336ms |
| MSD Hippocampus | 0.303 | **0.686** | 99ms | 50ms |
| MSD Prostate | **0.375** | 0.357 | 279ms | 5006ms |
| FLARE22 | **0.751** | 0.359 | 311ms | 4894ms |
| NasalSeg | 0.547 | **0.737** | 161ms | 1886ms |
| mean | -- | -- | **266ms** | 3669ms |

Ours wins 4/6 (TotalSeg MRI, HU\_LWK1, MSD Prostate, FLARE22); Medverse
wins 2/6 (MSD Hippocampus, NasalSeg) — both cases where the target is a
single, large, well-defined structure Medverse's full-FOV multi-resolution
pyramid suits well. Compute favors Ours by roughly 14$\times$ on mean
latency, and this is *after* level-matching Medverse's own depth to ours;
level-matching turned out to hurt Medverse's accuracy too (below), so an
unmatched Medverse would lose by even more on both axes at once.

**A real predicted prior beats a perturbed-GT prior.** Same checkpoint
chain, in-distribution TotalSegmentator (`val_classes=all`, 1157 samples,
3-level `[6,3,1.5]`mm ladder, 117 classes):

| config | macro Dice | macro NSD |
|---|---:|---:|
| GT-prior (perturbed) | 0.543 | 0.576 |
| **Predicted prior (real prev. pred)** | **0.586** | **0.631** |

Feeding the model's own real previous prediction forward beats a
perturbed-GT prior on 99/117 classes (median Δ +0.043) — a *prior-source*
ablation (real vs. noisy synthetic), distinct from the Medverse
*prior-presence* finding above; a true no-prior arm is still needed (gap
G6). OOD (9 sources incl. FLARE22/NasalSeg): helps 5/9, hurts 4/9 —
weaker than the clear in-distribution majority.

**Cascade registers: inconsistent, kept off.** Adding cross-level register
carry on top of the predicted-prior checkpoint is small and
protocol-sensitive (−0.006 to −0.020 depending on eval protocol, no
stable class-level pattern, OOD 3/9 help vs. 6/9 hurt) — not a settled
finding, and not part of the default configuration. One OOD number worth
keeping: `hu_lwk1` does **not** regress under registers (0.152→0.155),
superseding an earlier mid-training number (0.1287→0.0935) that should no
longer be cited.

**Level-matching Medverse's AR depth to ours hurts it.** Medverse's AR
level is set by `image_size` alone
(`level = max(1, ceil(log2(max_axis/128)) + 1)`); unlike our cascade,
which narrows field of view per level, Medverse's pyramid keeps the full
volume in view throughout and only refines resolution. Matching its depth
to our own per-source ladder:

| dataset | level change | old AR | matched AR | Δ |
|---|---|---:|---:|---:|
| HU\_LWK1 | 2→3 | 0.071 | 0.020 | **−0.052** |
| FLARE22 | 2→3 | 0.440 | 0.359 | **−0.082** |
| MSD Prostate | 2→3 | 0.423 | 0.357 | **−0.065** |
| MSD Hippocampus | 2→1 | 0.691 | 0.686 | ~flat |
| NasalSeg | 2→2 | 0.735 | 0.737 | ~flat |

Every source with a real depth increase regressed; the two unchanged
stayed flat. Side effect: MSD Prostate's per-model winner flips (Medverse
0.423 → Ours 0.375 once corrected) — the table above already uses the
corrected number.

## Synthetic Task Generation (Axis 3)

> **⚠️ OUTDATED (as of 2026-09-28).** Everything in this axis predates a
> newer checkpoint lineage (`160`→`167`, continuing directly from `135b`)
> that found and fixed a much larger effect on the same 4 weakest OOD
> sources (ISLES22, Shifts-MS, ATLAS v2.0, GNC_705): the synthetic canvas
> itself (`gmm_bank`) was 100% CT-sourced, mismatched to targets that are
> mostly MRI. That supersedes the shape-diversity conclusions below on
> those 4 sources specifically — see "**The sim-to-real canvas fix**"
> below, which replaces them. The texture-noise (105/106/107) and
> CT+MRI-modality-mix (110/130) findings, and the intensity-realism
> calibration methodology, are a different axis of evidence and are
> unaffected — kept as-is.

**Texture noise beats flat noise beats real-only**, TotalSeg val Dice, a
controlled 3-arm chain sharing one starting checkpoint (105 = real-only,
$p_\text{synth}{=}0$; 106 = flat i.i.d. synth noise, $p_\text{synth}{=}0.3$;
107 = multi-octave correlated texture, same $p_\text{synth}$), epoch 50:

| arm | val Dice | seen | unseen |
|---|---:|---:|---:|
| 105 (real-only) | 0.4868 | 0.5737 | 0.3886 |
| 106 (flat-noise synth) | 0.4899 | 0.5676 | 0.4022 |
| **107 (multi-octave texture)** | **0.4955** | **0.5748** | 0.4059 |

107 wins on all three metrics — highest overall and seen-class Dice
(narrowly beating even the real-only arm, i.e. texture erases the
seen-class cost that flat synth alone pays), and unseen-class Dice
essentially tied with its own peak. Neither 106 nor 107 had converged at
epoch 50 (both still climbing); 105 had already plateaued by epoch 10.

**CT+MRI joint training improves OOD generalization on every held-out
source** — the strongest single result in this chapter, from a matched,
single-variable ablation (`108`+ lineage, `110`=CT-only vs. `130`=identical
recipe with a 50/50 CT/MRI training mix):

| source | CT-only | CT+MRI | Δ |
|---|---:|---:|---:|
| ISLES22 | 0.021 | 0.041 | +0.020 |
| Shifts-MS | 0.001 | 0.014 | +0.013 |
| MSD Hippocampus | 0.134 | **0.291** | **+0.157** |
| MSD Prostate | 0.043 | 0.081 | +0.038 |
| ATLAS v2.0 | 0.021 | 0.030 | +0.009 |
| GNC_705 | 0.041 | 0.060 | +0.019 |
| HU_LWK1 (CT) | 0.130 | 0.136 | +0.006 |

All 7 sources improve, including the one CT-only source — this is a
genuine broad generalization gain from training-modality diversity, not
merely "MRI training helps MRI eval." (Caveat: N=1 seed, gap G11; not yet
checked on the `exp92` lineage, gap G10.)

**Synthetic shape-intensity realism was measured against real data, not
guessed — and one earlier internal calibration was caught and corrected
by that measurement.** A procedural shape-mode generator (blob / splatter
/ disk / cylinder, stamped into real host organs) had its shape-vs-host
intensity contrast calibrated against a fresh measurement of real
target-vs-immediately-surrounding-tissue contrast (749 cases across the 7
OOD sources, effect size in ring-standard-deviation units):

| source | contrast (ring-σ) | known radiological pattern |
|---|---:|---|
| ISLES22 (DWI, acute stroke) | +2.42 | classic DWI hyperintensity |
| Shifts-MS (FLAIR, MS lesion) | +1.49 | classic FLAIR hyperintensity |
| GNC_705 (kidney lesion) | +0.49 (σ=1.31) | mixed cyst/complex appearance |
| MSD Prostate (zones) | +0.35 | — |
| HU_LWK1 (vertebra ROI, CT) | −0.07 | ≈0 expected (not a lesion boundary) |
| MSD Hippocampus | −0.51 | — |
| ATLAS v2.0 (T1, chronic stroke) | −0.59 | classic T1 hypointensity |

Every source's sign and rough magnitude matches its known imaging
contrast direction, validating the measurement methodology before it was
used to set the generator's calibration range. This measurement also
caught a design error: an initial *absolute*-offset version of the
host-anchoring calibration was, by this same measurement, roughly
7–12$\times$ too extreme relative to the generator's own per-class
variance scale — corrected to a ratio-of-host-variance design before any
reported checkpoint used it.

**Widening synthetic shape diversity: a real short-budget effect that did
not hold up at a full training budget — reported as a limitation, not a
finding.** Three 31-epoch probes on top of the CT+MRI checkpoint (`131`
widened scatter/multiplicity realism; `132` widened non-scatter shape
diversity as a control; combining both) moved cross-source mean OOD Dice
from baseline 0.093 to 0.112/0.107 — but consolidating the combination at
a full 51-epoch budget (`133`) reverted to baseline (0.091), with one
source (HU_LWK1) reversing from the probes' largest gain to a net loss.
**The 31-epoch numbers should not be cited as a stable effect** — no
replicate seeds exist to separate genuine non-monotonic training dynamics
from noise on the smaller OOD sources (HU_LWK1 n=36, Shifts-MS n=46).

**Headline checkpoint** (`135b`: CT+MRI + all calibrations above,
continued training, stopped at epoch 170 not yet converged):

| source | baseline (`130`) | best 31-epoch probe | `135b` |
|---|---:|---:|---:|
| ISLES22 | 0.041 | 0.077 | 0.050 |
| Shifts-MS | 0.014 | 0.033 | 0.008 |
| MSD Hippocampus | 0.291 | 0.307 | **0.326** |
| MSD Prostate | 0.081 | 0.070 | **0.130** |
| ATLAS v2.0 | 0.030 | 0.027 | 0.028 |
| GNC_705 | 0.060 | 0.069 | **0.082** |
| HU_LWK1 (CT) | 0.136 | 0.223 | 0.196 |
| **mean of 7** | 0.093 | — | **0.117** |

Highest cross-source mean of any checkpoint evaluated, winning 3/7
sources outright — most notably MSD Prostate (>1.6$\times$ the previous
best), the one source that had regressed under every earlier shape
intervention. Still improving on its own in-domain validation metric when
stopped, so this is a snapshot, not a confirmed ceiling (gap G9).

**The sim-to-real canvas fix.** Continuing from `135b`, the same
shape-diversity direction above was pushed further (`160`–`167`) on the
four sources it had never moved: ISLES22, Shifts-MS, ATLAS v2.0, GNC_705
(Dice 0.008–0.082 on `135b`). Widening the shape-family mix and isolating
the highest-signal family (`scatter_field`) at higher dose both landed as
clean nulls — every source stayed within noise of `135b`. The cause,
found directly rather than assumed: the synthetic canvas (`gmm_bank`) the
shapes were painted onto is 100% CT-sourced, while every one of these
four targets is MRI (or Dixon-MRI). The fix swaps the canvas to a real
MRI TotalSegmentator subject's own image (`RealHostShapeProvider`)
instead of the synthetic bank, holding shape family, dose, checkpoint,
and epoch budget fixed — the one single-variable A/B in this whole
progression:

| stage | checkpoint | ISLES22 | Shifts-MS | ATLAS v2.0 | GNC_705 |
|---|---|---:|---:|---:|---:|
| baseline | `135b` | 0.050 | 0.008 | 0.028 | 0.082 |
| 1) synthetic canvas (`gmm_bank`) | `162` | 0.045 | 0.011 | 0.033 | 0.070 |
| 2) real canvas, one family (`scatter_field`) | `163` | **0.098** | **0.058** | 0.032 | 0.105 |
| 3) real canvas, all 7 families | `167` | 0.087 | 0.051 | 0.027 | **0.223** |

Stage 1→2 (`162`→`163`) is the controlled comparison: identical
checkpoint, dose, and epoch budget, only the canvas modality changes.
Every source moves except ATLAS v2.0 (a single, compact, chronic
lesion — the only non-multi-focal target of the four, plausibly outside
what a multiplicity-targeted shape family like `scatter_field` can reach).
Extending the same real-canvas mechanism to all 7 shape families and a
much longer budget (stage 3, `167`, all real CT+MRI hosts across every
organ, stopped at epoch 320/400) keeps compounding on GNC_705
(0.070→0.105→**0.223**, the best result of the whole progression) while
easing back slightly on the two brain-lesion sources from stage 2's peak.

**This did not come at the cost of in-distribution accuracy.** TotalSeg
val Dice (own periodic validation, same checkpoints, `seen`/`unseen` =
macro Dice by trained-class membership, `mri`/`ct` = micro Dice by target
modality):

| stage | val Dice | seen | unseen | MRI | CT |
|---|---:|---:|---:|---:|---:|
| baseline (`135b`, e170) | 0.418 | 0.489 | 0.344 | 0.390 | 0.468 |
| 1) synthetic canvas (`162`, e9) | 0.422 | 0.485 | 0.356 | 0.385 | 0.470 |
| 2) real canvas, one family (`163`, e9) | 0.417 | 0.483 | 0.348 | 0.384 | 0.467 |
| 3) real canvas, all families (`167`, e320) | **0.490** | **0.579** | **0.397** | **0.474** | **0.534** |

In-distribution Dice tracks flat-to-improving throughout — the seen/CT
lead over unseen/MRI persists at roughly the same size at every stage,
and stage 3's much longer budget lifts every one of these five numbers
together. The OOD gains above are additive transfer improvement, not a
trade against in-distribution accuracy.

*(Confounds not controlled in stage 2→3 specifically — epoch budget,
task-level deform, spacing range, added CT hosts all change together
alongside the family-count widening — so that step should be read as "the
mechanism keeps paying off when pushed further," not as its own isolated
ablation; stage 1→2 is the one variable-isolated result. Single seed
throughout, as elsewhere in this chapter. Source data:
`results/publications/thesis/experiments/{3a_synthetic_tasks,
3b_synthetic_tasks_train}/`.)*
