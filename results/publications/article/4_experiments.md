# Experiments (draft)

See `PERFORMANCE_ANALYSIS.md` for the full evidence synthesis behind every
claim below (file citations, contradictions, and what's still missing).
Numbers here are preliminary/internal — not yet verified as
publication-ready (see gaps flagged inline).

## Setup

- **Dataset:** TotalSegmentator CT. **TODO:** state the train/held-out
  anatomical class split — following the same held-in/held-out protocol
  used in the 2D paper to test in-context generalization to unseen
  classes, not just unseen subjects. (Training curves already show a real
  seen/unseen gap, e.g. dice_seen 0.68 vs dice_unseen 0.51 on an early
  cascade checkpoint — the split exists operationally, just needs stating.)
- **Baselines:** Medverse, Iris. **TODO/blocker:** no Iris baseline exists
  anywhere in the repo yet (`PERFORMANCE_ANALYSIS.md` gap G5) — every
  number we have is vs. Medverse only. Also unresolved: our OOD numbers use
  a finetuned-on-TotalSeg-CT checkpoint vs. Medverse's released,
  multi-modal, zero-shot weights (gap G8) — must be stated explicitly or
  fixed before any head-to-head framing.
- **Scope decision (gap G1):** "Ours" currently means the
  `exp92_multisource_synth` cascade line (bi-axial attention + query prior +
  region restriction, register-carry gated off). The newer PatchSetV2 line
  is not yet converged and is not the source of any number below.
- **Metrics:** Dice, normalized surface distance (NSD) for accuracy;
  wall-clock time, FLOPs, peak VRAM for compute. **Caveat:** VRAM is not
  currently measured anywhere in the eval pipeline (`evaluate.py`/`cascade.py`
  has no memory instrumentation) — only in a separate B=1/K=1 inference
  benchmark. Accuracy and compute numbers currently come from disjoint
  scripts/checkpoints; needs one unified benchmark before the joint table
  below can be filled in cleanly.

## Fixed spacing

- Datasets: TotalSegmentator CT, whole-body `use_crop=false` protocol
  (117 classes, 3897 samples) — see `PERFORMANCE_ANALYSIS.md` §1 for why
  this protocol is canonical and a conflicting notebook (`nb 36`) is not.
- **Preliminary result:** Medverse wins macro Dice overall (0.078 vs 0.048)
  and wins more individual classes, but we run at **17% of its FLOPs**
  (398 vs 2363 GFLOPs) at comparable wall-clock. The overall macro-Dice gap
  is *not* uniform: splitting by object thickness, we're near parity on
  thin structures (Δ≈−0.005) and lose specifically on thick tubes/blobs
  (Δ up to −0.18) — replicated independently in the 2D paper and in a
  separate 3D geometric-driver study. **This is the strength claim: small/thin
  structures at a large compute discount, not an overall win.**
- **Compute, current architecture:** the "much smaller model" framing is
  stale — the current architecture (77–110M params) is comparable to or
  larger than Medverse (71.1M). What holds up at matched (bf16) precision is
  a modest 1.05–1.3× latency edge and a FLOPs reduction specific to the
  iris-style compressed decoder path (1230 vs 2363 GFLOPs), not a
  param-count story.

**TODO:** per-anatomical-category breakdown (organs, bones, vessels,
muscles) beyond the thickness-family split we already have; an Iris row;
one benchmark that reports accuracy+VRAM+FLOPs together for a single
checkpoint/setting (see gaps G2–G5).

## Coarse-to-fine

- Datasets: TotalSegmentator CT.
- **Preliminary result — cascade helps in-distribution, replicated twice:**
  macro Dice **0.361 → 0.538** (+0.177) from 4mm coarse to 1.5mm fine
  (`37_patchset_spacing_locator.py`), and a matching monotonic
  r1.5/r3/r6 = 0.572/0.509/0.389 on the full multisource eval (wandb
  `92_multisource_synth`). Locator containment (0.92 mean) does *not*
  predict fine-level Dice — once the coarse level roughly localizes the
  structure, remaining failures are segmentation quality, not localization,
  matching the `more_labels` failure-mode study.
- **TODO/blocker:** the actual ablation this section promises (full model
  vs −query prior vs −region restriction vs −register carry) has not been
  run as a controlled experiment (gap G6). What exists instead is indirect
  evidence: feeding a model's own imperfect prediction forward as the
  query prior hurts Medverse's own cascade universally (7/7 OOD sources,
  sometimes catastrophically) — suggestive that compounding error is a
  generic cascade risk, not proof of our specific design's ablated
  components.
- Register-carry status: still training (epoch 36/140 at last check),
  mixed early signal — helps single-level Dice on 3/7 OOD sources but
  regresses the cascade result on the one source where it's been fully
  measured (HU_LWK1, 0.1287→0.0935); one OOD cell (GNC_705 cascade) has
  never completed a run (suspected `torch.compile` hang, not a scoring
  issue). Don't cite a register-carry number until training finishes and
  that cell is resolved or explicitly excluded.

**TODO:** ablation table (Dice, time, FLOPs per configuration) once G6 is
run; Pareto plot of Dice vs. compute as the region-restriction margin is
swept (not yet run at all — gap G7); full multi-level cascade compute
(time/FLOPs/VRAM end-to-end, including re-crop overhead) has never been
benchmarked (gap G3).

## Generalization

- **Preliminary result — cascade helps only when the coarse level is
  already roughly right:** across 7 native-grid OOD sources
  (`docs/datasets/eval_expansion_status.md`), cascade *helps* only on
  HU_LWK1 (CT, single-level Dice 0.116 → cascade 0.129) — the one
  same-modality-as-training source with a non-trivial single-level Dice —
  and *hurts* on GNC_705 and ISLES22 (both MRI, both already failing at
  single-level, Dice 0.05–0.06). This mirrors the Medverse query-prior
  finding above: coarse-to-fine cascades compound error when the coarse
  prior is unreliable. Frame as a general cascade limitation (motivates a
  confidence-gated fallback as future work), not a design flaw unique to
  us.
- **Other modalities:** Shifts-MS (FLAIR, 0.063), MSD Hippocampus (T1,
  0.481 — best OOD-MRI result), MSD Prostate (T2/ADC, 0.295), ATLAS v2.0
  (T1, 0.028 — ⚠️ possible label-corruption confound, don't treat as clean
  evidence of failure).
- **TODO/blocker:** best-of-Medverse (released, zero-shot) beats our
  single-level result on 4/7 of these sources, loses on 3 (HU_LWK1,
  ISLES22, GNC_705) — but this comparison is confounded (gap G8: released
  multi-modal Medverse vs. our CT-only-finetuned checkpoint) and needs
  either a matched-training rerun or an explicit caveat before it goes in
  the paper as "generalizes better/worse."
- **Far OOD / pathology tasks:** GNC_705 (kidney lesion) and ISLES22
  (stroke lesion) already serve this role; both show the cascade-hurts
  pattern above. No additional far-OOD source planned beyond these unless
  a gap is found in review.

## Synthetic data ablation (synth_gmm)

- **Setup:** three runs sharing one starting checkpoint and recipe
  (single-level, non-cascade training regime — not the fixed-spacing/
  cascade eval protocol used elsewhere in this doc), varying only the
  synthetic-data knob: real-only ($p_{\text{synth}}{=}0$), synthetic with
  i.i.d. paint noise ($p_{\text{synth}}{=}0.3$), and synthetic with
  multi-octave texture noise (same $p_{\text{synth}}$). Metric: TotalSeg
  val Dice, seen/unseen class macro split, at a shared epoch-50 cutoff.
- **Preliminary result — texture noise beats both alternatives on every
  axis:** val Dice 0.496 (texture) vs. 0.490 (i.i.d. noise) vs. 0.487
  (real-only); seen-class 0.575 vs. 0.568 vs. 0.574; unseen-class 0.406
  vs. 0.402 vs. 0.389. Plain synthetic data alone trades seen-class
  accuracy for unseen-class generalization; multi-octave texture noise
  recovers the seen-class cost while keeping the unseen-class gain.
  Runs (curves not yet converged — both synthetic arms were still rising
  at the cutoff):
  [real-only](https://wandb.ai/tidiane-camaret-ndir-universit-tsklinikum-freiburg/patchset_train/runs/y3kcv5w1) ·
  [i.i.d.-noise synth](https://wandb.ai/tidiane-camaret-ndir-universit-tsklinikum-freiburg/patchset_train/runs/pg7veapf) ·
  [texture-noise synth](https://wandb.ai/tidiane-camaret-ndir-universit-tsklinikum-freiburg/patchset_train/runs/gdap5nuo)
- **TODO/blocker:** prediction figures per arm (pull from the wandb runs
  above once ready); extend all three past epoch 50 to convergence;
  external-cohort OOD numbers for these checkpoints (the `eval.py`
  DataLoader bug blocking this is now fixed, not yet re-run here); this
  ablation uses a different training regime than "Ours" elsewhere in this
  section (gap G1-adjacent) — state that explicitly if/when merged into
  a single results table.

## Pooling token ablation (`arch.pool_token`)

- **Setup:** two further runs continuing the texture-noise checkpoint
  above (same recipe, same epoch-50 cutoff), adding an IRIS-style
  foreground-masked pooling token (one extra prefix row per volume,
  §3's "Synthetic supervision" arch is otherwise unchanged) and varying
  only *which* feature map it pools: `fine` (IRIS's own choice — a
  near-native-resolution stage, real per-K compute cost) vs. `coarse`
  (the same coarse grid features the main transformer already uses,
  essentially free).
- **Preliminary result — both help, `fine` wins by a modest, late-emerging
  margin:** val Dice 0.507 (`fine`) vs. 0.500 (`coarse`) vs. 0.496 (no
  pooling token), at matching epoch. Notably `coarse` *led* `fine`
  through the first third of training before `fine` pulled ahead — a
  weaker and later-emerging version of the effect IRIS's own ablation
  reports for masking after (vs. before) upsampling, not a clean
  replication of their reported magnitude.
  Runs (curves not yet converged):
  [pool=fine](https://wandb.ai/tidiane-camaret-ndir-universit-tsklinikum-freiburg/patchset_train/runs/mr91mta9) ·
  [pool=coarse](https://wandb.ai/tidiane-camaret-ndir-universit-tsklinikum-freiburg/patchset_train/runs/q4umnca3)
- **Per-sample breakdown — the small-object story is real, just concentrated at
  the extreme tail:** paired per-sample Dice (same deterministic 800-task eval
  set across all three checkpoints, confirmed identical `(class, subject,
  context)` at every row) bucketed by target foreground-voxel count. `coarse`'s
  gain over no-pooling is essentially **zero** in the smallest quintile
  (+0.0003, median 42 voxels) and only turns clearly positive from the second
  quintile up (+0.003 to +0.005); `fine`'s gain is positive everywhere and
  highest in that same smallest bucket (+0.017). Tightening to <100 voxels
  (n=166): `fine` beats `coarse` on 58% of tiny samples vs. 54% on the rest.
  This is a direct, sample-level match for *why* IRIS's own ablation frames
  this as a small-object effect — `coarse` pools an already spatially-blended
  R³-grid feature (~stride-8 receptive field per cell), so small structures
  are diluted with neighboring content before pooling ever happens, regardless
  of mask precision. Caveat: the whole-distribution correlation between
  target size and the fine-vs-coarse gap is weak (Spearman ρ=−0.05, p=0.16)
  — this is a tail effect, not a smooth gradient, and the 95% CI on the tiny-
  bucket gap (±0.019) still crosses zero at this sample size.
- **Follow-up — pooling more than one stage:** `pool_token`'s `fine` mode
  originally read only the single finest requested encoder stage even when
  `fine_decode` was configured with more than one (`fine_stage=[0,1]`,
  the decoder's own default); fixed to mask-average every requested stage
  independently and concatenate, mirroring how the decoder itself already
  combines multiple fine_stage maps. Run in progress, continuing the `fine`
  checkpoint above:
  [pool=fine, 2 stages](https://wandb.ai/tidiane-camaret-ndir-universit-tsklinikum-freiburg/patchset_train/runs/xha3lxx2).
- **TODO/blocker:** prediction figures per arm; convergence check (same
  caveat as the synthetic-data ablation above); external-cohort OOD
  numbers; decide whether `coarse`'s near-free compute makes it the
  better default despite the small Dice gap, once VRAM/latency for both
  variants is actually measured (no compute numbers exist yet for either);
  result of the 2-stage `fine` run above.
