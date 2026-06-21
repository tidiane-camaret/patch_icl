# controlSynth difficulty study — findings (UniverSeg baseline)

Synthesis of a four-round sensitivity study of the controlSynth generator, probing
which difficulty knobs actually make tasks hard for an in-context segmenter, in order to
use the generator for the synthetic-data training study (`experiments/2d/pfn_seg.py`).

**Method.** `experiments/2d/synth_benchmark.py` evaluates the **UniverSeg baseline**
(pretrained, zero-training, in-context) so that Dice reflects *intrinsic task difficulty*,
not fitting. One-factor-at-a-time (OFAT) sweeps isolate each knob; a 2D-grid mode probes
interactions. Each subject logs its full knob vector + realized stats (`fg_frac`,
`ctx_dice`) + Dice. Runs at N≈11k (OFAT) / ≈2k (grids).

**Caveat.** All conclusions are relative to UniverSeg, which is out-of-distribution on
synthetic textures. They characterize *the generator's difficulty levers for a
context-matching model*; absolute Dice is not calibrated.

## 1. The operating point matters

UniverSeg's mean Dice on the first ("moderate") baseline was ~0.33 with 42% total
failures — near its floor, which **compressed the dynamic range and hid real drivers**.
Re-running at an **easy** baseline (`region_size=0.6`, `foreground_contrast=0.1`; blob Dice
0.805) revealed that several "inert" knobs were floor artifacts. Always measure knob
sensitivity with headroom.

## 2. Difficulty driver ranking

| Class | Knobs | Verdict |
|---|---|---|
| **Context quality (dominant)** | `context_consistency`, `support_query_shift`, `context_copy_fraction` | The #1 lever. UniverSeg is a context-matcher, so anything that breaks context↔query correspondence dominates. ρ(Dice, ctx_dice) is high for most knobs (shift 0.74, boundary 0.70, morphology 0.68). |
| **Foreground geometry** | **`region_size`** (dominant single knob, +0.27 Dice 0.05→0.6), `boundary_complexity`, `scattered_clustering`, `scattered_count`, `branching_density` | Real. Several were floor-masked at the moderate baseline. |
| **Appearance** | `foreground_contrast` (after fix), `texture_heterogeneity` | Moderate, correctly oriented after the redesign (§4). |
| **Identification / ambiguity** | `task_ambiguity` (shape + intensity distractors) | **Inert** — see §3. |
| **Negligible** | `tortuosity`, `thinness` | < 0.08 Dice spread even with headroom. |

## 3. The identification axis does not bite (key null result)

`task_ambiguity` is the spec's *primary ICL/identification axis*: background regions are
given the foreground's shape (build) and intensity (live), so the model must use context
to pick the true region. **It barely affects UniverSeg.**

- OFAT: Δ ≈ 0.05 at both baselines.
- 2D grid `task_ambiguity × task_ambiguity_intensity` (N=1728): shape distractors alone
  cost −0.06; intensity-side adds only −0.017 on top; interaction −0.017; **zero total
  failures** in the joint-high corner.
- 2D grid `task_ambiguity × region_size` (N=2304): the ambiguity penalty is **constant
  (~−0.05) at every foreground size**, including region_size=0.05 (fg ≈ 0.4% of the image
  with ~13 same-shape/same-intensity decoys). Interaction ≈ 0.

**Conclusion.** With consistent K=3 context, UniverSeg resolves *which* region is the
target by spatial context-matching, regardless of how many look-alikes exist or how small
the target is. Distractor ambiguity is **not** a usable difficulty axis here. To create
genuine identification difficulty you must couple ambiguity with **context degradation**
(inconsistent/mismatched context), not just add decoys.

## 4. Broken/mislabeled knobs (generator issues, now fixed/flagged)

- **`foreground_contrast` was inverted** — FIXED. The original `gmm_fill` pushed
  *background* regions to the [0,1] extremes as contrast rose (bg saturation 5%→52%),
  leaving the foreground a bland mid-grey blob, so higher "contrast" was *harder* (Dice
  0.60→0.23). Two-part fix in `appearance.gmm_fill` + `config.map_contrast_gap`:
  (a) background means now stay in a fixed central band [0.25,0.75]; the foreground is
  pushed `gap(contrast)` toward an extreme, so the **foreground** owns the salient
  extremes; (b) the fg's side is a **task-level constant** (`meta["appearance_sign"]`)
  shared across the task's context+target subjects — otherwise high contrast pushed each
  subject's fg to an *independent* extreme and the context no longer matched the target
  (the deeper cause of the inversion). After the fix the axis is correctly oriented: low
  contrast = hardest (0.80, fg intensity-invisible → found via shape/context), rising to a
  ~0.85 plateau; the gap is capped (max 0.45) so the extreme stays separable.
- **`context_copy_fraction` collapsed to Dice 0** at copy=1.0 — FIXED. Diagnostics showed
  it was not pixel-identity (noised/rolled near-copies also collapsed; real contexts scored
  0.86): copying the target *frame* makes the **background** match the query too, so the fg
  loses the distinctiveness a context-matcher relies on (the fg is normally the only region
  consistent across contexts). Redefined as **pristine exemplars** — a fraction of contexts
  are rendered with near-zero deformation but a **fresh background**. Now non-degenerate
  (copy=1.0 → ~0.80); a mild ease knob (eases more at harder, context-limited baselines).
- **`noise_level` sign flip — RESOLVED** as a side effect of the contrast redesign. It now
  decreases Dice monotonically (0.0 easiest 0.80 → 1.0 hardest 0.68); the earlier
  floor-baseline "noise helps" regime came from pathologically-clean OOD images, which the
  salient-foreground rendering removed.

## 5. Recommendations for the synthetic-data training study

1. **Difficulty curriculum should be built on the levers that work**: `region_size`
   (geometry difficulty) and the **context-quality** family (`context_consistency`,
   `support_query_shift`) for ICL difficulty — these have the largest, cleanest,
   correctly-oriented effects.
2. **Do not rely on `task_ambiguity` to create identification difficulty** for
   context-matching models; if identification difficulty is wanted, degrade context
   alongside it.
3. Treat `tortuosity`, `thinness` as near-neutral. `context_copy_fraction` (now fixed) is a
   mild ease knob — most useful at harder, context-limited baselines.
4. `foreground_contrast` (low=hard) and `noise_level` (high=hard) are now both usable,
   correctly-oriented appearance axes after the generator fixes.
5. Re-validate these rankings with the actual trained ImagePFN model (not just UniverSeg)
   before committing a curriculum — the inertia of the identification axis in particular
   may differ for a model trained on the synthetic distribution.

Reproduce: `python experiments/2d/synth_benchmark.py --baseline easy` (OFAT, all knobs);
`--grid` (ambiguity interaction); `--grid "build:task_ambiguity=..." "build:region_size=..."`
(any 2-knob grid). Outputs → `results/2d/synth_benchmark/<ts>/`.
