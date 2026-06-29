# OOD generalization analysis — `imagepfn_zoom` trained on `hard_diverse`

**Model.** `imagepfn_zoom` checkpoint `2026-06-22_kind-durian-59/best.pt`, trained on
`synth=hard_diverse` (controlSynth). **Eval.** `experiments/2d/eval.py`,
`data.source=synthetic`, held-out `val` shape pool, K=3 context, image_size 128.

**Question.** How does the model perform on tasks *outside* its training distribution?

**Key control — `ctx_dice`.** The realized overlap between the context/support masks and
the target GT (`mean_k hard_dice(target, context_out[k])`, a property of the data, not the
prediction). It separates two reasons Dice can fall under an OOD shift:
- `ctx_dice` **flat** → context still carries full information ⇒ a Dice drop is pure
  **model brittleness** ("info-preserving" axis).
- `ctx_dice` **falls** → the shift itself destroys the context↔target correspondence ⇒
  degradation is partly **inherent** (less signal), not the model's fault.

---

## 1. Distribution-level results

| eval set | mean Dice | median | failures (<0.1) | ctx_dice |
|---|---|---|---|---|
| **in-distribution** (`hard_diverse`, val) | **0.648** | 0.693 | 3.5% | 0.164 |
| **OOD — appearance only** (info preserved) | **0.261** | 0.235 | 33.3% | 0.164 |
| **OOD — combined conditions** | **0.071** | 0.000 | 75.6% | 0.045 |

- **Appearance-only** shift (noise/contrast/texture pushed OOD, context correspondence
  kept in-dist so `ctx_dice` is unchanged) still halves Dice → a *clean* generalization
  gap of **−0.39**: the model has effectively overfit to its training appearance regime.
- **Combined** shift additionally degraded the context-correspondence knobs, collapsing
  `ctx_dice` (0.164→0.045) → near-total failure. This is **partly an artifact**: the task
  became near-impossible (no usable context), not just OOD. Read it as an upper bound on
  worst-case stacking, not a pure generalization number.

Configs: `configs/experiment/2d/synth/{ood_appearance,ood_conditions}.yaml`.
CSVs: `results/controlsynth/{ood_appearance,ood_conditions}.csv`.

---

## 2. Per-axis sweep (the core analysis)

One knob varied at a time, all others held at the in-distribution training point.
`hard_diverse` fixes each **live** knob at a *single* scalar, so the model trained on one
operating point per axis — **both sides are OOD**. Live-knob runs share geometry (one
shared in-dist anchor = **0.647**); build-knob runs change geometry (own anchor).
Reduced to `num_tasks=2000` (400 tasks/run, ~42 s/run; anchor matches the full 5000-task
run's 0.648). Driver: `run_sweep.sh`. Aggregate: `aggregate.py` → `sweep_summary.csv`.

### 2a. Info-preserving axes (ctx_dice flat — clean generalization signal)

| axis | train | easy-side | OOD-hard extreme | Δ Dice (hard) |
|---|---|---|---|---|
| **foreground_contrast** | 0.50 | 0.80 → 0.688 | 0.05 → **0.351** | **−0.30** |
| **noise_level** | 0.40 | 0.10 → 0.665 | 1.00 → **0.461** | **−0.19** |
| **texture_heterogeneity** | 0.35 | 0.05 → 0.663 | 0.95 → **0.504** | **−0.14** |
| **task_ambiguity** (build) | [0.3,0.8] | 0.0 → 0.738 | 1.00 → **0.607** | **−0.13** |
| **task_ambiguity_intensity** | 0.60 | 0.0 → 0.715 | 1.00 → **0.603** | **−0.04** |

Clean, attributable brittleness. Curves are **asymmetric**: flat/saturated on the easy
side, smooth falloff on the hard side — the signature of training at a single point.
**Low foreground contrast is the single biggest unforced generalization gap.**

### 2b. Context-correspondence axes (ctx_dice falls — partly inherent)

| axis | train | OOD extreme | Δ Dice | ctx_dice |
|---|---|---|---|---|
| **support_query_translate** | 0.05 | 0.35 → **0.395** | −0.25 | 0.166 → **0.061** |
| **context_consistency** | 0.90 | 0.30 → **0.436** | −0.21 | 0.166 → **0.073** |
| **support_query_shift** | 0.50 | 0.95 → **0.472** | −0.18 | 0.166 → 0.104 |
| **support_query_scale** | 0.45 | 1.20 → **0.631** | −0.02 | 0.166 → 0.153 |

Translate/consistency/shift hurt most in absolute terms, but each **collapses `ctx_dice`**
— largely information loss. **`support_query_scale` is genuinely robust**: flat out to
1.20 (>2.5× trained), `ctx_dice` barely moves.

### 2c. Geometry — `region_size` (train range 0.12–0.62)

Monotone; the model **extrapolates *upward* better than within range**:
0.95 → **0.796 Dice, 0 failures** (vs 0.647 in-dist). Small objects are the hard end
(0.05 → 0.490) but that's the known sub-patch floor + collapsing context overlap.

---

## 3. Consolidated robustness ranking

| rank | axis | behavior | clean (info-preserving)? |
|---|---|---|---|
| most robust | support_query_scale | flat to 1.2× | — (ctx ~flat) |
| | task_ambiguity_intensity | shallow | ✓ |
| | region_size (upward) | improves | — |
| | task_ambiguity (build) | gentle | ✓ |
| | texture / noise | smooth falloff | ✓ |
| | support_query_shift | moderate | partly info-loss |
| | context_consistency | steep | partly info-loss |
| | **foreground_contrast** | **steepest clean drop (−0.30)** | ✓ |
| most fragile | **support_query_translate** | **steepest overall** | mostly info-loss |

---

## 4. Conclusions

1. **No catastrophic overfitting to the exact training point.** On the *easier* side of
   every axis (less noise, higher contrast, less ambiguity, larger objects) Dice is
   flat-or-better; the model extrapolates fine to benign shifts.
2. **The genuine generalization weaknesses are all appearance knobs:**
   **foreground_contrast ≫ noise ≈ texture**. These are info-preserving, so the drop is
   pure brittleness.
3. **Context-geometry shifts either generalize well (scale) or hurt because they destroy
   information (translate, consistency)** — the latter no model can recover.
4. **Identification ambiguity is handled gracefully** (both ambiguity axes have shallow
   slopes), consistent with the model using context as intended.

### Actionable recommendation
The cheapest robustness win is to **randomize the live appearance knobs during training**
— sample `foreground_contrast`, `noise_level`, `texture_heterogeneity` over ranges instead
of pinning each at one value (which is exactly why the model is sharp-but-brittle around
it). `support_query_scale` already generalizes and needs no extra attention.

---

## 5. Artifacts

- Configs: `configs/experiment/2d/synth/{ood_appearance,ood_conditions}.yaml`
- Sweep driver / aggregation: `run_sweep.sh`, `aggregate.py`, plot scripts
  `plot_contrast.py`, `plot_noise.py`, `plot_remaining.py`
- Per-run CSVs: `results/controlsynth/sweep/*.csv` (one operating point each)
- Summary table: `results/controlsynth/sweep/sweep_summary.csv`
- Figures:
  - `sweep_curves.png` — all 10 axes
  - `extrapolation_contrast.png` — foreground_contrast (most informative, w/ per-morph)
  - `extrapolation_noise.png` — noise_level
  - `extrapolation_remaining.png` — 8 remaining axes (mean Dice)
