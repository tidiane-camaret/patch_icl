# Performance analysis for the thesis (working synthesis)

Forked from `results/publications/article/PERFORMANCE_ANALYSIS.md` on
2026-09-23 so the thesis can add analysis/ablations beyond what the CVPR
paper needs. Re-sync manually if the shared evidence base changes in ways
that matter to both documents.

This is a synthesis of every concrete Dice/NSD/time/FLOPs/VRAM number currently
sitting in the repo (`docs/`, `results/experiments/`, `results/presentations/`,
`wandb/`), organized around the thesis's three-axis structure (`IDEAS.md`,
`3_results.md`). Goal: figure out what story the existing evidence
actually supports, where it contradicts itself, and what's missing before we
can write real numbers into the thesis. Nothing here is fabricated — every
number has a file citation. Treat this file as a lab notebook, not prose for
the thesis; port conclusions into `3_results.md` once resolved.

**Scope note:** almost all usable numbers below come from the
`exp92_multisource_synth` / cascade-`[6,3,1.5]` architecture line (the one
described in `3_method.md`: bi-axial attention, query-prior init, register
carry gated off by default). The newest `PatchSetV2` line (configs 100/101/103/104,
git commit `970ab44`) is **not converged** (see §5) and should not be the
source of any publication number yet — but it may be the actual target
architecture if training continues. This is a scope decision the paper needs
to make explicitly (see Gaps, item G1).

---

## 1. Fixed-spacing, in-distribution accuracy — the thickness story

Two notebooks give **contradictory headline verdicts** on the same
TotalSegmentator/117-class comparison, because they use different eval
protocols and checkpoints:

| notebook | protocol | checkpoints | macro Dice: Medverse | macro Dice: ours |
|---|---|---|---|---|
| `20_medverse_patchset3d_comp.py` (`artifacts/20_crop_false_runs.csv`) | `use_crop=false`, whole-body, 3897 samples | patchset `vc7kfdto` / medverse `94nlx7yw` | **0.078** | 0.048 |
| `36_patchset_medverse_multiscale.py` | occupancy-mask setting | patchset `21_nzxp4nw9` (ep192) / medverse `21_czxebwi5` (ep176) | 0.249 | **0.562** |

nb20's numbers match memory `[[project_patchset3d_vs_medverse_3d]]`
(mv 0.088 / ps 0.057, crop=false) closely enough to treat as the same,
reproducible finding. nb36's numbers do not match anything else in the repo
and use a different masking convention (`occupancy-mask`) — **treat nb36 as
non-canonical until someone confirms what protocol it actually measures.**
Do not average or split the difference between these two; pick nb20's
protocol as canonical (it's cross-validated by memory) and either fix or
retire nb36.

**The defensible headline, from nb20 + memory, is not "who wins overall" but
what drives the gap:**

- Medverse wins overall macro Dice (0.078 vs 0.048) and wins more classes
  (66/117 vs 51/117), complete-miss rate is lower for Medverse (1.6% vs
  0.5%... wait, ours has the *lower* miss rate — Medverse misses more often
  but wins bigger when it doesn't miss).
- Per-sample: Medverse wins 49.7%, we win 40.2% — closer than the macro
  average of *individual classes* suggests.
- **Compute:** we run at **17% of Medverse's FLOPs** (398 vs 2363 GFLOPs) at
  roughly the same wall-clock (85ms vs 71ms) — nb20, `artifacts/20_crop_false_runs.csv`.
- **Shape-family breakdown is the real finding:** we lose badly on
  `thick_tube` (Δ−0.182) and `thick_blob` (Δ−0.095), and are roughly at
  parity or ahead on `mid_sheet` (Δ+0.016) and `mid_tube1` (Δ+0.008).
  Splitting by thickness quartile: thin half Δ=−0.005 (**near parity**),
  thick half Δ=−0.059 (Medverse pulls ahead). This is a clean, independently
  replicated result — the 2D paper found the same pattern (patch-based
  selection wins thin/small structures), and the 3D geometric-driver study
  (`[[project_patchset3d_vs_medverse_3d]]`) confirms it's *object thickness*,
  not object identity or contrast, that predicts the winner.

**Hypothesis for why:** bi-axial image/label attention lets label evidence
sharpen image features at every layer without needing a large convolutional
receptive field, so it resolves thin/high-frequency shapes early. Thick,
large-volume structures likely need appearance consistency aggregated over a
large spatial extent (e.g. organ-wide intensity homogeneity) — something a
deep conv stack accumulates naturally across many layers of spatial mixing,
but that our current patch-token granularity may under-resolve. This is
exactly the framing `0_abstract.md`'s "which categories benefit most" TODO
should use — **the answer is: thin/small structures, driven by geometry not
semantics**, and it comes at a large compute discount (17% FLOPs), not a
compute-matched win.

**Gap (G2):** the FLOPs/VRAM numbers here (398 GFLOPs) are from an older,
smaller patchset3d checkpoint (~4.7M params per `docs/logs.md:4936`). The
compute picture for the *current* architecture line is different — see §2.

---

## 2. Compute at fixed spacing — the param-count story doesn't hold for the current architecture

Two vintages of compute benchmark exist and **should not be conflated**:

**Old (`docs/logs.md:4936-4996`, B∈{1,2,4}, older ~4.7M-param patchset3d vs
Medverse 71.1M):**
- 15× fewer params (4.7M vs 71.1M).
- fwd+bwd 63ms vs 185ms (2.9×); epoch 50s+17GB vs 190s+36GB.
- Peak VRAM @B=4: 11.7GB vs 29.5GB.
- This is the number underlying the current draft's framing of "our model" as
  much smaller/cheaper — it is **stale for the architecture we're now
  training**.

**Current (`results/presentations/perf/RESULTS.md`, 2026-09-20, B=1/K=1/128³,
RTX PRO 6000 Blackwell, native precision per model):**

| model | params | GFLOPs | latency | peak VRAM |
|---|---:|---:|---:|---:|
| Medverse (fp32) | 71.1M | 2362.6 | 69.6 ms | 3.30 GB |
| Medverse (bf16, unverified accuracy) | — | — | 45.5 ms | 2.32 GB |
| ours, iris-decoder | 71.9M | uncounted (FlopCounter crash) | 36.4 ms (bf16) | 3.36 GB |
| ours, conv-decoder (92_multisource_synth) | 48.0M | 3891.1 | 43.5 ms (bf16) | 2.50 GB |
| ours, v2 (101) | 77.4M | 1230.3 | 35.9 ms (bf16) | 2.67 GB |
| ours, v2 (103, cascade arch, single-level) | 109.6M | 1993.4 | 42.6 ms (bf16) | 3.78 GB |

**The current architecture has grown to be comparable-to-larger than
Medverse in parameter count** (71.9M–109.6M vs Medverse's 71.1M) — the
"15× smaller" claim is no longer true and must not be reused. What survives:
we're consistently **1.6–1.9× faster wall-clock** at native precision, but
most of that gap is a **bf16-vs-fp32 precision artifact, not architecture**
— Medverse under bf16 drops to 45.5ms, narrowing our advantage to
**~1.05–1.27×**. VRAM is now roughly at parity or slightly worse for us
(3.36–3.78GB vs Medverse's 3.30GB), the opposite of the old finding.

**Revised honest compute claim:** at matched precision, the two model
families cost about the same per forward pass; our FLOPs are lower in the
v2/101 config (1230 vs 2363 GFLOPs, ~52%) via the iris-style m-scale
cross-attend skipping the R³-token transformer, but that's a decoder design
choice (`decoder=iris`/`v2`), not free — it's the same lever Iris itself
uses. **The compute story needs to be told as "cheaper decoder path via
region-restricted/compressed attention," not "smaller model."**

**Gap (G3):** no FLOPs/time/VRAM number for a full **multi-level cascade**
run exists anywhere (`RESULTS.md` explicitly says the 103 row "says nothing
about the cost of a full multi-level cascade run"). We have per-level costs
and can multiply by 3 as an estimate, but re-crop/GPU-realize overhead
between levels is unmeasured (see `[[project_cascade_realize_step_profile]]`
for a partial cost breakdown from a different angle — not the same metric).

**Gap (G4):** VRAM is never actually measured in `evaluate.py`/`cascade.py`
(per `results/presentations/val/per_dataset_analysis.py` inline finding — no
`torch.cuda.max_memory_allocated()` call in the eval path). Every VRAM number
above comes from the standalone `bench_inference_compare.py` harness, at
B=1/K=1, not from a real eval run with K>1 context. **The paper's planned
Table (`tab:fixed-spacing`, accuracy+compute jointly) cannot currently be
produced from one script — accuracy and compute numbers come from disjoint
benchmarks with different checkpoints/settings.**

**Gap (G5) — Iris is entirely absent.** Every single accuracy or compute
number found across all four sources (`docs/`, notebooks, presentations,
wandb) compares us against **Medverse only**. There is no Iris checkpoint,
no Iris eval run, no Iris compute benchmark anywhere in this repo. The
related-work and method sections argue against Iris's pixel-shuffle fusion
design by construction, but there is zero empirical Iris baseline to back
that up. This is the single biggest hole for the "Fixed spacing" experiment
section as scoped in `IDEAS.md` — either (a) get an Iris baseline running
(their released weights/repo, per `docs/README.md`'s pointer to
`/software/notebooks/camaret/repos`), or (b) scope the paper's claims to
"vs. Medverse" only and soften the Iris comparison to a qualitative/related-work-only
argument.

---

## 3. Coarse-to-fine cascade — helps in-distribution; register-carry regresses

**In-distribution, cascade helps, cleanly, in two independent measurements:**

- `37_patchset_spacing_locator.py` (run `05kb6kcc`, 117 classes/3897 samples,
  spacing sweep 4mm→1.5mm): macro Dice **0.361 → 0.538** (+0.177) from
  coarse to fine. Locator containment 0.922 mean, oracle 0.970 (0.048 gap =
  crop-ceiling headroom). Containment does **not** predict fine-level
  accuracy (weak/negative correlation, confounded by object size) — i.e.
  well-localized classes still fail at the *segmentation* step, not the
  localization step. This matches the `more_labels` failure-mode finding
  (`[[project_more_labels_failure]]`): **segmentation quality, not
  localization, is the bottleneck once the coarse level is roughly right.**
- `92_multisource_synth` eval (wandb `run-20260915_121004-ne3cbag7`, 1157
  samples, cascade `[6,3,1.5]`, e4_plainconv_ts encoder): dice
  r1.5/r3/r6 = **0.5716 / 0.5088 / 0.3889** — monotonic coarse-to-fine gain,
  11673 GFLOPs, 666ms total for the 3-level pass.

**Register-carry ablation is now complete and confirmed negative**
(`[[project_cascade_register_carry]]`): experiment `2a_cascade`
(`results/publications/thesis/experiments/2a_cascade/`), a 3-arm chain
`145`→`146`→`147` all resuming checkpoint `135b` independently, trained
to completion (60 epochs, `p_synth=0`, in-distribution TotalSegmentator,
119 classes / 800 fixed eval samples). `146` (pred-prior, registers off)
vs. `147` (pred-prior, registers on): macro Dice **0.5246 → 0.5050**
(−0.020), regressing 77/119 classes, worse on held-out than seen classes
(−0.030 vs. −0.009), and +2.5% inference latency for it. Per-class
breakdown shows the regression is structural, not uniform: worst-hit
classes are almost all repeated fine anatomy in one volume (individual
ribs/vertebral levels, worst case `brachiocephalic_vein_left` — Dice 0.0
at every epoch under registers-on), while large uniquely-shaped
structures (lungs, skull, aorta) improve.

**⚠️ Supersedes an earlier claim:** the mid-training OOD probe previously
cited alongside this result (`hu_lwk1` single-level 0.1287→0.0935) is now
superseded by a matched, fully-trained comparison — see the `146` vs
`147` row in the `2b_cascade_val` table below (0.152→0.155, flat, not a
regression). **Do not cite the 0.1287→0.0935 number any more.**

**OOD sweep of the same 145/146/147 checkpoints** (`2b_cascade_val`,
`results/publications/thesis/experiments/2b_cascade_val/`, each source's
own native cascade ladder, `msd_hippocampus` single-level only), macro
Dice:

| source | n | 145 (GT-prior) | 146 (pred-prior) | 147 (+registers) |
|---|---:|---:|---:|---:|
| hu_lwk1 (CT) | 36 | 0.099 | 0.152 | 0.155 |
| msd_prostate | 124 | 0.310 | 0.375 | 0.358 |
| msd_hippocampus | 520 | 0.319 | 0.302 | 0.400 |
| isles22 | 247 | 0.067 | 0.069 | 0.060 |
| shifts_ms | 46 | 0.035 | 0.031 | 0.030 |
| atlas_v2 | 654 | 0.029 | 0.026 | 0.025 |
| gnc_kidney | 1490 | 0.027 | 0.024 | 0.021 |

Prior-source (145→146): helps 3/7 sources, hurts 4/7. Registers
(146→147): helps 2/7 (`hu_lwk1` flat-positive within noise,
`msd_hippocampus` +0.097), hurts 5/7. Neither in-distribution direction
replicates cleanly OOD.

**Query-prior *source* ablation now exists on our own architecture too —
but it is not the same comparison as the Medverse finding, and does not by
itself close G6.** `query_prior=pred` (feeding the model's own imperfect
coarse prediction forward) hurts Medverse's harness-cascade universally,
7/7 OOD sources, sometimes catastrophically (`msd_hippocampus`
0.6905→0.0500, `docs/datasets/eval_expansion_status.md:339-345`) — that
comparison is *prior present vs. absent*. `2a_cascade`'s `145` (GT-prior,
perturbed) vs. `146` (real prev.-pred prior), both registers off, is a
different comparison — *which non-null prior source is better* — with
both arms already using a prior: macro Dice **0.4988 → 0.5246** (+0.026),
improving 90/119 classes, i.e. a real predicted prior beats a noisy
synthetic (perturbed-GT) one. This is not in tension with the Medverse
result (different comparisons, not opposite findings on the same one),
but it also isn't the "−query-prior" arm G6 asks for — no arm here drops
the prior entirely. **G6 narrows, does not close:** still need a true
no-prior arm on our own architecture to complete "full model vs
−query-prior vs −region-restriction vs −register-carry"; the
register-carry leg of that ablation table is now filled in by this same
experiment (§ above).

**Gap (G7):** the accuracy/compute Pareto sweep as the region-restriction
margin is relaxed (full volume → tight crop) — the plot `4_experiments.md`
promises as "analogous to the resolution/FLOPs Pareto plot in the 2D
paper" — has not been run. `37_patchset_spacing_locator.py` gets close (it
sweeps spacing, not crop margin) but doesn't vary the restriction margin
itself.

---

## 4. Generalization / OOD — cascade helps only same-modality CT, hurts elsewhere

Single-level vs cascade, `exp92` checkpoint, native-grid eval-expansion
sources (`docs/datasets/eval_expansion_status.md`):

| source | modality | single-level Dice/NSD | cascade Dice/NSD | cascade verdict |
|---|---|---|---|---|
| ISLES22 | MRI DWI | 0.054/0.074 | 0.046/— `[3,1.5]` | hurts |
| GNC_705 (kidney lesion) | MRI Dixon | 0.0585/0.0784 | 0.0189/0.0293 `[6,3,1.2]` | **hurts badly** |
| HU_LWK1 (L1 vertebra) | **CT** | 0.1164/0.1313 | **0.1287**/0.0729 `[6,3,1]` | **helps** (only source that isn't hurt) |
| Shifts-MS | MRI FLAIR | 0.063/0.165 | not run | — |
| MSD Hippocampus | MRI T1 | 0.481/0.719 | not run | — |
| MSD Prostate | MRI T2+ADC | 0.295/0.321 | not run | — |
| ATLAS v2.0 | MRI T1 | 0.028/0.038 | not run | ⚠️ possible label-corruption confound |

**The pattern is clean and worth being the paper's headline generalization
finding:** cascade helps when the coarse level's own single-level Dice is
already reasonable and the modality matches training (HU_LWK1, CT, single-level
Dice 0.116), and hurts when the coarse level is already failing
(GNC_705/ISLES22, single-level Dice 0.05-0.06, both MRI, both lesion/pathology
targets far from TotalSegmentator's anatomical-organ training distribution).
This is the same compounding-error mechanism as the Medverse query-prior
finding above — **general to coarse-to-fine cascades, not specific to our
design**, and worth stating as such in Related Work / Discussion rather than
as a weakness unique to us. It directly motivates a concrete future-work /
limitation statement: a confidence-gated fallback to full-volume search when
coarse-level confidence is low.

**Best-of-Medverse (released weights, no finetune) beats our exp92
single-level on 4/7 sources** (shifts_ms, msd_hippocampus, msd_prostate,
atlas_v2), loses on hu_lwk1/isles22/gnc_kidney
(`docs/datasets/eval_expansion_status.md:326-335`). **Gap (G8) — this is a
confounded comparison**: Medverse here is released, multi-modal-pretrained,
zero-shot; ours is finetuned only on TotalSegmentator CT. The TODO.md item
("state whether Medverse/Iris are retrained on our split or evaluated with
released weights") is exactly this confound — it must be resolved (matched
training data, or explicitly caveated) before this 4/7-vs-3/7 framing goes
in the paper, otherwise it reads as "Medverse generalizes better" when it
may just be "Medverse saw more training modalities."

---

## 5. PatchSetV2 — not yet converged, don't cite

The newest architecture line (`experiment=100/101/103/104`, commits
`970ab44`/`132fc8d`) is far from converged: `103_patchset_v2_cascade` was at
epoch 0 (dice 0.006) at last check, `104_patchset_v2_varspacing_hard_tgt_prior`
epoch 133 dice 0.235 — both well below the `exp92` line's 0.43–0.58 range.
The `95-99_iris_decoder*` ablation series is similarly early/undertrained
(dice 0.013–0.056), including one *fully-trained* checkpoint
(`99_iris_decoder_plainconv_doubling`, epoch 139, dice 0.056) that stays far
below the cascade lineage — flag for verification (real ablation finding, or
a training-config bug?) rather than assuming it's a valid negative result.

**Gap (G1, restated):** decide whether the paper's "Ours" is the
`exp92_multisource_synth` line (converged, has real numbers, matches
`3_method.md`'s description modulo register-carry) or the in-progress
`PatchSetV2` line. If it's the latter, none of the accuracy numbers above
are usable yet and the experiments section is blocked on training progress,
not analysis.

---

## 6. CT+MRI joint training + synth intensity/shape calibration (2026-09-23 session) — a new checkpoint lineage

**Scope note:** this section's numbers come from a *third* checkpoint
lineage (configs `108`→`130`→`135`→`135b`), distinct from both `exp92`
(§1–4 above) and `PatchSetV2` (§5). It descends from `exp92` via the
`105`–`121` chain (adds `cascade_registers`, `pool_token`, synth texture
noise on top of `exp92`'s own feature set) and is the most-recently-trained,
best-performing line on the OOD eval-expansion sources — but it changes
several axes at once relative to `exp92` (architecture *and* data mix *and*
synth calibration), so it should not be read as an ablation against §1–4,
only as its own internally-controlled sequence. **This sharpens gap G1**:
"is Ours = exp92 or PatchSetV2" now has a third real candidate.

**CT+MRI joint training — clean, controlled, the strongest single result
of this session.** All experiments up to `108`–`121` trained CT-only
despite already being wired for a configurable CT/MRI task mix. A matched
pair — `110` (CT-only) vs. `130` (identical recipe, only
`regime_p` changed to 50/50 CT/MRI) — isolates the effect on the same 7
eval-expansion OOD sources used in §4:

| source | 110 (CT-only) | 130 (CT+MRI) | Δ |
|---|---:|---:|---:|
| isles22 | 0.0211 | 0.0406 | +0.0195 |
| shifts_ms | 0.0012 | 0.0141 | +0.0129 |
| msd_hippocampus | 0.1342 | 0.2913 | **+0.1571** |
| msd_prostate | 0.0430 | 0.0806 | +0.0376 |
| atlas_v2 | 0.0206 | 0.0298 | +0.0092 |
| gnc_kidney | 0.0412 | 0.0602 | +0.0190 |
| hu_lwk1 (CT) | 0.1297 | 0.1357 | +0.0060 |

CT+MRI mixing improves Dice on **all 7** OOD sources, including the one
CT-only source — i.e. this is not simply "MRI training helps MRI eval,"
it is a genuine broad generalization gain from modality diversity in
training. This directly counters an earlier, *confounded* impression:
comparing the same `130` checkpoint against the older `exp92`/
`cascade_register` baselines (different architecture generation entirely)
had suggested CT+MRI mixing looked *worse* — resolved only once the
matched-lineage ablation was run. **Methodological lesson worth stating
explicitly in Discussion:** an appealing but architecturally-confounded
baseline comparison gave the opposite conclusion from a clean ablation: do
not trust a cross-lineage comparison for a data-mix claim.

**Synth shape-diversity: real short-budget effect, did not hold up at a
real training budget — an honest negative/mixed result.** Widening the
procedural shape generator's parameter ranges (`131`, splatter-scatter
realism motivated by measuring real lesion multiplicity — e.g. MS lesions:
median 54 separate components per case, vs. the generator's un-widened cap
of 6) improved 4/7 OOD sources after a short 31-epoch probe. A paired
diversity-only control (`132`, widening *non-scatter* shape parameters
instead) reproduced most of that gain, showing the dominant mechanism was
generic shape-parameter diversity, not scatter-realism specifically.
Consolidating both changes at a full 51-epoch budget (`133`) **reverted
to baseline on the cross-source mean** (0.0913, vs. baseline `130`'s
0.0932, vs. the two 31-epoch probes' 0.1121/0.1065) — one source
(`hu_lwk1`) even reversed from the probes' biggest win to a net loss.
**Do not cite the 31-epoch shape-diversity probe numbers as a stable
finding** — they read differently at different training budgets, and no
replicate seeds exist to separate genuine training-dynamics
non-monotonicity from noise on the smaller sources (`hu_lwk1` n=36,
`shifts_ms` n=46).

**Intensity/shape realism calibration, grounded in measurement, not
guesswork.** Three previously-unused or newly-built calibration
mechanisms (§ Methodology Axis 3) were wired in together (`135`):
cross-class mean correlation (`mu_group_ids`, pre-existing but never
enabled in any config before this session), cross-class variance
correlation (`sd_group_ids`, did not exist before this session — `sd` was
always independent regardless of `mu_group_ids`), and host-anchored shape
intensity (a shape's own painted intensity is now drawn relative to its
host organ's real mean/std, ratio range calibrated against a fresh
measurement of real target-vs-surrounding-tissue contrast on the 7 OOD
sources — 749 cases, physiologically sensible per-source signatures that
validate the measurement: isles22/DWI stroke lesion +2.42 ring-$\sigma$
hyperintense, atlas_v2/T1 chronic lesion $-0.59$ hypointense, matching
known radiological contrast direction for each). An initial version of
the host-anchoring calibration used an *absolute* intensity offset and
was, by this same later measurement, found to be roughly 7–12$\times$ too
extreme relative to the generator's own per-class variance scale —
caught and corrected before any reported result used it.

**Headline result (`135b`, this checkpoint's own recipe continued to
epoch 170, not yet converged when stopped):**

| dataset | 130 (baseline) | best of 131/132/133 | **135b** |
|---|---:|---:|---:|
| isles22 | 0.0406 | 0.0770 | 0.0499 |
| shifts_ms | 0.0141 | 0.0326 | 0.0080 |
| msd_hippocampus | 0.2913 | 0.3074 | **0.3262** |
| msd_prostate | 0.0806 | 0.0697 | **0.1303** |
| atlas_v2 | 0.0298 | 0.0271 | 0.0281 |
| gnc_kidney | 0.0602 | 0.0692 | **0.0819** |
| hu_lwk1 (CT) | 0.1357 | 0.2229 | 0.1964 |
| **mean of 7** | 0.0932 | — | **0.1173** |

`135b` has the highest cross-source mean of every checkpoint tested this
session, winning 3/7 sources outright — most notably `msd_prostate`
(0.1303, >1.6$\times$ the previous best), the *one* source that had
regressed under every earlier shape-diversity intervention. It was still
improving on its own in-domain validation metric when training was
stopped (val Dice still rising from epoch $\sim$120 onward), so this may
not be its ceiling. Weaker on `isles22`/`shifts_ms`/`hu_lwk1` than the
best short-probe numbers — plausibly the same non-monotonic-with-budget
pattern seen in `133`, not yet disentangled.

**New gaps this section adds:**
- **G9:** `135b` was stopped mid-training, not converged — no replicate,
  no confirmation the reported numbers are a stable endpoint rather than
  a snapshot on a still-moving trajectory.
- **G10:** the CT+MRI ablation (the cleanest result in this section) has
  not been re-checked on the `exp92` architecture lineage — it is only
  demonstrated on the `108`+ line. Unclear how much of the effect is
  architecture-independent vs. specific to features (`cascade_registers`,
  `pool_token`) `exp92` lacks.
- **G11:** no replicate seeds anywhere in this session's checkpoint
  sequence (`130`–`135b`) — every comparison in this section is an N=1
  training run per condition. The shape-diversity non-monotonicity
  finding especially should be read as "not yet stable," not "disproven."

## Prioritized gap list (for `TODO.md`)

1. **G1 — scope decision:** is "Ours" = exp92 line, PatchSetV2, or the
   newer `108`→`135b` line (§6 — best OOD numbers of any lineage so far,
   but changes architecture+data+calibration together vs. exp92, so it
   isn't a drop-in replacement claim without more isolation work). Blocks
   everything else.
2. **G5 — no Iris baseline exists anywhere.** Biggest hole relative to what
   the paper claims to compare against.
3. **G2/G8 — reconcile Medverse-comparison confounds:** crop setting (nb20
   vs nb36), and finetuned-vs-released-zero-shot (OOD table). Pick one
   canonical protocol per experiment and rerun both baselines under it.
4. **G6 (narrowed) — still need a true −query-prior arm.** `2a_cascade`
   (§3) now fills in the register-carry leg and a prior-*source* ablation
   (real pred vs. perturbed-GT), but no arm drops the query prior entirely
   — that comparison, and −region-restriction, are still open.
5. **G3/G4 — build one script that reports accuracy AND compute together**
   (including VRAM, currently unmeasured in the eval path) for a matched
   checkpoint/setting, so `tab:fixed-spacing` can be filled from a single
   source instead of stitched from two unrelated benchmarks.
6. **G7 — run the region-restriction margin sweep** (Pareto plot) promised
   in `4_experiments.md`.
7. Verify whether `99_iris_decoder_plainconv_doubling`'s low dice (0.056,
   fully trained) is a genuine finding or a training-config bug.
8. **G9/G11 — replicate.** Resume `135b` to convergence (it was stopped
   mid-climb) and, separately, re-run at least the CT+MRI ablation (§6,
   the strongest claim in this file) with a second seed before it goes in
   the thesis as a headline result — currently N=1.
9. **G10 — check the CT+MRI joint-training effect on the exp92 lineage
    too**, to know whether it's a general finding or specific to the
    `108`+ feature set.

## What's already solid enough to write into the paper now

- **Thickness-driven accuracy story** (§1): near-parity on thin structures,
  losing on thick ones, at 17% of Medverse's FLOPs — replicated in 2D and 3D,
  cross-validated by memory and by `nb20`.
- **In-distribution cascade gain**: +0.177 macro Dice from 4mm→1.5mm
  coarse-to-fine, replicated in two independent runs (`37` and wandb `92` eval).
- **Cascade compounding-error pattern**: helps on same-modality/reasonable-coarse-Dice
  targets (HU_LWK1), hurts when the coarse level is already failing
  (GNC_705, ISLES22) — generalizable framing backed by the independent
  Medverse query-prior finding.
- **CT+MRI joint training improves OOD generalization** (§6): +Dice on all
  7 eval-expansion sources in a matched, single-variable-changed ablation
  (`110` vs `130`) — the cleanest, most directly citable result in this
  file. Caveat: N=1 seed (item G11), and only demonstrated on the `108`+
  lineage, not cross-checked against `exp92` (item G10).
- **Real-data-grounded synth calibration methodology** (§6, Methodology
  Axis 3): cross-class intensity/variance correlation and host-anchored
  shape-intensity ranges were calibrated against measurements on the
  actual OOD eval sources (not guessed), and the measurement itself
  reproduces known per-source radiological contrast direction — a solid,
  citable methodology contribution independent of whether the resulting
  checkpoint's Dice numbers hold up under more training.
- **`cascade_registers` regresses accuracy in-distribution** (§3):
  completed, controlled ablation (`2a_cascade`, `146` vs `147`), macro
  Dice 0.5246→0.5050 across 119 classes. Caveat: N=1 seed, only
  demonstrated on the `108`→`135b` lineage, and does **not** replicate
  OOD — the matched `2b_cascade_val` sweep of the same checkpoints is
  mixed (helps 2/7 sources, hurts 5/7).
- **Real predicted prior beats a perturbed-GT prior in-distribution**
  (§3): same `2a_cascade` chain, `145` vs `146`, macro Dice
  0.4988→0.5246, 90/119 classes improve. This is a prior-*source*
  ablation, not the still-open prior-*presence* one (G6) — don't conflate
  the two when citing. Also does not replicate OOD (`2b_cascade_val`:
  helps 3/7 sources, hurts 4/7).
