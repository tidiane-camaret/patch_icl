# In-Context Medical Image Segmentation from Synthetic Supervision — Mid-Project Report

## Summary

We train in-context segmentation (ICL) models entirely on **synthetic data** — a
supervoxel-based label generator in the spirit of *Scaling In-Context Segmentation with
Hierarchical Supervision* — and evaluate zero-shot on real medical images.

- **3D:** our model outperforms **Medverse** (the only open-source full-3D in-context
  segmenter to date) on all 7 anatomy categories of TotalSegmentator (117 labels).
- **2D:** trained purely on synthetic shapes and evaluated on **BiomedParse**
  (≈70k samples, 41 datasets, 9 modalities), we beat the **UniverSeg** baseline on
  **6 of 9** modality categories.

Our goal is **high accuracy at the lowest possible compute**, which requires understanding
*which ICL mechanisms are actually necessary*. The 2D setting lets us probe the sampling
strategy precisely and — because training is fully synthetic and parameterized — run
**controlled experiments isolating which image properties drive out-of-distribution (OOD)
performance**.

## 1. Motivation and goal

In-context segmentation aims for a single model that segments arbitrary structures from a
few (image, mask) examples, with no per-task fine-tuning. We target **efficiency**:
current ICL stacks combine many mechanisms (large encoders, context cross-attention,
multi-stage refinement) whose individual contributions are unclear. We want the minimal
mechanism set that retains accuracy, for deployment under a tight compute budget.

## 2. Approach

The model learns the in-context matching operation — "given K example masks for a
structure, segment the same structure in a new image" — rather than memorizing anatomy.
Training data is thus decoupled from any real labeling effort, and the task distribution
is directly controllable. We use two complementary synthetic generators.

**3D — supervoxel pseudo-labels on real CT.** Following the hierarchical-supervision
recipe, we run an unsupervised supervoxel segmentation (default SLIC; also grid /
watershed / SEEDS) on each real CT volume, drawing `n_segments ~ U[50, 500]` per subject.
Each supervoxel becomes a pseudo-structure; an optional *union* step merges adjacent
supervoxels (face-connectivity random walk) into fewer, organ-scale blobs. A task picks
one pseudo-label and forms K context (image, mask) pairs plus a target from different
subjects. The images are real CT — only the labels are synthetic — so appearance is
realistic while supervision is free and unlimited.

**2D — procedural difficulty-controlled shapes (`controlSynth`).** A fully procedural
generator (128×128, ~16 distractor regions) that composites a task from scratch:
foreground morphology → enforce area fraction → roughen boundary → Voronoi background with
injected shape-distractors. Appearance is rendered by a per-region GMM intensity fill plus
Perlin+Gaussian noise; each subject (target and each context) is an independent elastic
deformation of the shared base geometry. Its defining feature is that every difficulty
factor is an **explicit, calibrated knob**, split into *build* (geometry, fixed per task)
and *live* (appearance / context-pose, resampled per subject) axes — so a single knob can
be swept as a controlled variable while task diversity and quantity are held fixed. This
is what enables the controlled OOD study in §6.

## 3. 3D results — vs. Medverse on TotalSegmentator (117 labels)

Trained on synthetic supervoxels, evaluated on 117 real TotalSegmentator labels with a
held-out subject set. We outperform Medverse in every category:

| Category | PatchICL | Medverse | Δ |
|---|---|---|---|
| Muscles | **0.488** | 0.178 | +0.310 |
| Bones (Limbs/Shoulder/Pelvis) | **0.481** | 0.255 | +0.227 |
| Organs (Thorax/Head/Spine) | **0.381** | 0.189 | +0.193 |
| Bones (Spine) | **0.180** | 0.024 | +0.156 |
| Vessels | **0.156** | 0.045 | +0.111 |
| Organs (Abd/Pelvis) | **0.333** | 0.244 | +0.089 |
| Bones (Ribs/Sternum) | **0.047** | 0.023 | +0.024 |

## 4. Moving to 2D: faster ablation and broader evaluation

To study mechanisms and sampling under a tractable compute budget, we shifted experiments
to **2D**. This enables (i) fast iteration on the **sampling strategy** and on mechanism
ablations, and (ii) evaluation on far more **diverse real data** than 3D CT alone.

## 5. 2D results — vs. UniverSeg on BiomedParse (9 modalities)

Trained on synthetic 2D shapes, evaluated zero-shot on BiomedParse (≈70k samples,
41 datasets, 9 modalities). PatchICL wins on **6 of 9** modality categories:

| Category | PatchICL | UniverSeg | Δ |
|---|---|---|---|
| Endoscopy | **0.251** | 0.149 | +0.102 |
| Fundus | **0.492** | 0.407 | +0.085 |
| CT | **0.322** | 0.258 | +0.064 |
| Pathology | **0.169** | 0.111 | +0.057 |
| OCT | **0.211** | 0.166 | +0.045 |
| MRI | **0.183** | 0.159 | +0.024 |
| Ultrasound | 0.571 | **0.579** | −0.009 |
| X-Ray | 0.567 | **0.647** | −0.081 |
| Dermoscopy | 0.447 | **0.604** | −0.158 |

We lead on 6/9 categories; UniverSeg leads on Ultrasound, X-Ray, and Dermoscopy.

## 6. Controlled OOD analysis in the synthetic setting

Because training is fully synthetic and parameterized (the **controlSynth** generator), we
hold the segmentation task fixed and shift one image property at a time, then measure how
the model degrades. We take a 2D model trained on a diverse-but-fixed synthetic regime
(`hard_diverse`) and sweep each generative knob around its training value.

**Control.** Each sweep also tracks `ctx_dice`, the realized overlap between the context
masks and the target — a property of the data, not the prediction. When `ctx_dice` stays
flat, the context still carries full information, so a Dice drop reflects the model alone;
when `ctx_dice` falls, the shift itself removes usable signal, so degradation is partly
inherent to the task. This separates model brittleness from increased task difficulty.

**Measured findings.**
- A pure appearance shift (noise / contrast / texture moved out of range, context
  information preserved) lowers Dice from **0.648 → 0.261**, a **−0.39** gap attributable
  to appearance shift alone.
- Among information-preserving axes, the largest drops are foreground **contrast**
  (−0.30), **noise** (−0.19), and **texture** (−0.14).
- Within-set **scale** variation is essentially flat out to ~2.5× the trained value;
  **larger-than-trained objects** score higher than in-distribution; identification
  **ambiguity** degrades only gently.
- On the easier side of every axis the model is flat-or-better (no collapse around its
  training operating point).

Figures: `results/controlsynth/sweep/extrapolation_{contrast,noise,remaining}.png`.
Full breakdown: `results/controlsynth/sweep/SUMMARY.md`.

## 7. Takeaways and next steps

1. Synthetic supervision works: a model trained with zero real labels beats the open 3D
   baseline on all categories and a strong 2D baseline on most modalities.
2. In the controlled synthetic setting, the measured generalization gap is largest along
   **appearance** axes (contrast, noise, texture) and small along geometry / context-pose
   axes. A natural next step is to **randomize appearance during synthetic training**
   (sample contrast/noise/texture over ranges rather than fixing them) and measure the
   effect on real-data modalities.
3. With the 2D harness in place, ablate ICL mechanisms (context cross-attention,
   refinement stages, encoder size) to find the smallest mechanism set that preserves
   accuracy.

---

*Open items: confirm the aggregation used for the reported Dice scores (per-label mean vs.
per-category macro) and annotate the tables accordingly.*
