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

- **`exp92_multisource_synth`** (bi-axial attention, query-prior init,
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

**Bi-axial attention vs. early feature fusion.** A direct ablation of the
architecture's own bi-axial design (row-axis cross-context + column-axis
within-volume img↔mask attention) against an IRIS-style early-fusion
alternative: img and mask are merged into one token per cell via a
PixelShuffle trick *before* the transformer (`arch.dual_axis=false`), and
only row-axis (cross-context) attention remains. Compute-matched (the
single-axis arm's transformer layer count is raised to `l=9` so its
GFLOPs land within 3% of the bi-axial baseline's — params could not be
matched simultaneously this way and are +52% higher for single-axis, since
row-axis-only layers are a parameter-inefficient way to buy back the
missing column-axis compute). Both arms warm-start *only* the encoder and
decoder from a shared checkpoint; the entire in-context reasoning core
(transformer, img/mask embed, context/query id, thinking rows, cascade
projection, pool projection) is randomly initialized for **both** arms —
removing the asymmetric-warm-start confound an earlier, uncontrolled
version of this ablation had (one arm fine-tuning an already
~200-epoch-trained transformer, the other only partially warm). Real
anatomy only (`p_synth=0`), CT-only, TotalSeg val, both arms read at a
matched epoch (160) since the single-axis run had not yet reached its full
400-epoch budget at analysis time:

| arm | val Dice | seen | unseen |
|---|---:|---:|---:|
| **bi-axial (dual_axis=true)** | **0.462** | **0.527** | **0.389** |
| single-axis fusion (dual_axis=false, l=9) | 0.448 | 0.510 | 0.377 |

Bi-axial attention wins on all three metrics at matched epoch, matched
compute, and a from-scratch reasoning core — a smaller and better-controlled
gap than the earlier uncontrolled comparison. Not yet a settled result:
single-axis was still training toward its full budget (both arms were
still climbing at epoch 160, neither converged), this is N=1 seed per arm,
and the params mismatch (+52% for single-axis) means a residual capacity
confound remains even after compute-matching.

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

**Register carry hurts; a real predicted prior beats a perturbed-GT
prior** — a completed, controlled 3-arm chain (`145`→`146`→`147`, all
resuming checkpoint `135b` independently, `p_synth=0`, in-distribution
TotalSegmentator, 119 classes / 800 fixed eval samples, macro Dice at
epoch 59/60):

| arm | config | macro Dice | Δ |
|---|---|---:|---:|
| 145 | GT-prior (perturbed), registers off | 0.499 | — |
| 146 | pred-prior (real prev. pred), registers off | **0.525** | **+0.026** |
| 147 | pred-prior, registers on | 0.505 | **−0.020** |

Two findings from one chain. First, feeding the model's own real previous
prediction forward as the query prior beats a perturbed-GT prior on 90/119
classes (median +0.026) — a *prior-source* ablation (real prediction vs.
noisy synthetic), distinct from the Medverse finding above, which is a
*prior-presence* one (real prediction vs. none); the two aren't in
tension, but a true no-prior arm on our own architecture is still needed
(gap G6) before claiming the full ablation. Second,
`cascade_registers` regresses accuracy (77/119 classes, mean −0.020,
worse on held-out classes than seen: −0.030 vs. −0.009) and adds ~2.5%
latency for it. The regression concentrates structurally: the
worst-hit classes are almost all repeated fine anatomy in the same volume
(individual ribs, individual vertebral levels — worst case
`brachiocephalic_vein_left`, which collapses to Dice 0.0 at every epoch
under registers-on), while a smaller set of large, uniquely-shaped
structures (lungs, skull, aorta) actually improve — consistent with
registers interfering with instance disambiguation rather than uniformly
degrading capacity. The region-restriction-margin Pareto sweep (gap G7)
has not been run.

**Neither in-distribution direction replicates on OOD.** The same three
checkpoints (`145`/`146`/`147`), evaluated on 7 held-out sources
(`2b_cascade_val`, each source's own cascade ladder), give a mixed
picture: the pred-prior gain (145→146) helps 3/7 sources (`hu_lwk1`
+0.053, `msd_prostate` +0.065) and hurts 4/7; the register regression
(146→147) helps 2/7 (`hu_lwk1` flat, `msd_hippocampus` +0.097) and hurts
5/7. Notably, `hu_lwk1`'s cascade does **not** regress under registers
here (0.152→0.155) — superseding an earlier, mid-training OOD number
(0.1287→0.0935) that should no longer be cited.

## Synthetic Task Generation (Axis 3)

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
