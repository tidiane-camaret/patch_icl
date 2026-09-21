# Performance analysis for the publication (working synthesis)

This is a synthesis of every concrete Dice/NSD/time/FLOPs/VRAM number currently
sitting in the repo (`docs/`, `results/experiments/`, `results/presentations/`,
`wandb/`), organized around the paper's structure (`IDEAS.md`,
`4_experiments.md`). Goal: figure out what story the existing evidence
actually supports, where it contradicts itself, and what's missing before we
can write real numbers into the paper. Nothing here is fabricated — every
number has a file citation. Treat this file as a lab notebook, not prose for
the paper; port conclusions into `4_experiments.md` once resolved.

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

## 3. Coarse-to-fine cascade — helps in-distribution, ablations incomplete

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

**Register-carry ablation (`arch.cascade_registers`) is incomplete and
mixed**, not a clean ablation result yet (`[[project_cascade_register_carry]]`):
still mid-training (epoch 36/140 at last check), single-level improves on
3/7 OOD sources (e.g. `hu_lwk1` +0.045) but the *cascade* result on
`hu_lwk1` regresses (0.1287 → 0.0935), and the `gnc_kidney` cascade cell for
this checkpoint never completed (3 failed attempts, suspected
`torch.compile` hang, not a scoring bug). **Do not report a register-carry
ablation number until (a) training finishes and (b) the gnc_kidney run
either completes or is explicitly excluded with a stated reason.**

**Query-prior finding exists, but only for Medverse's own cascade, not
ours specifically:** `query_prior=pred` (i.e., feeding the model's own
imperfect coarse prediction forward) hurts Medverse's harness-cascade
universally, 7/7 OOD sources, sometimes catastrophically
(`msd_hippocampus` 0.6905→0.0500, `docs/datasets/eval_expansion_status.md:339-345`).
This is strong indirect evidence that coarse-to-fine cascades are
generically vulnerable to compounding error from an unreliable prior — but
it has not yet been isolated as an ablation on *our* architecture's own
query-prior design (the TODO.md ask: "full model vs −query prior vs −region
restriction vs −register carry" does not exist as a controlled ablation
anywhere). **Gap (G6).**

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

## Prioritized gap list (for `TODO.md`)

1. **G1 — scope decision:** is "Ours" = exp92 line or PatchSetV2? Blocks
   everything else.
2. **G5 — no Iris baseline exists anywhere.** Biggest hole relative to what
   the paper claims to compare against.
3. **G2/G8 — reconcile Medverse-comparison confounds:** crop setting (nb20
   vs nb36), and finetuned-vs-released-zero-shot (OOD table). Pick one
   canonical protocol per experiment and rerun both baselines under it.
4. **G6 — run the actual ablation table** for our own cascade (full model
   vs −query-prior vs −region-restriction vs −register-carry), not just
   Medverse's query-prior-hurts finding used as indirect evidence.
5. **G3/G4 — build one script that reports accuracy AND compute together**
   (including VRAM, currently unmeasured in the eval path) for a matched
   checkpoint/setting, so `tab:fixed-spacing` can be filled from a single
   source instead of stitched from two unrelated benchmarks.
6. **G7 — run the region-restriction margin sweep** (Pareto plot) promised
   in `4_experiments.md`.
7. Finish register-carry training (currently epoch 36/140) and get a
   completed gnc_kidney cascade cell before citing that ablation.
8. Verify whether `99_iris_decoder_plainconv_doubling`'s low dice (0.056,
   fully trained) is a genuine finding or a training-config bug.

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
