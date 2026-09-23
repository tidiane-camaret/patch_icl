# Methodology (draft)

Source of truth for `ics_3d_medical/sections/methodology_*.tex`. Edit here
first, then port into the matching `.tex` file (same subheading → same
file: Overview → `methodology_overview.tex`, Fusion → `methodology_fusion.tex`,
etc.).

## Overview (problem setup and backbone)

Given a target volume $I^t$ and a context set $\{(I^c_k, L^c_k)\}_{k=1}^{K}$
of image/mask pairs for the same anatomical class as the (unknown) target
mask, the model predicts a target mask $\hat{L}^t$. Prediction proceeds
through a coarse-to-fine cascade of $M$ resolution levels with decreasing
voxel spacing $r_1 > r_2 > \dots > r_M$; each level produces a full mask
that seeds the next, finer level (see Cascade below).

**Figure:** architecture overview, ours vs. Medverse (early concatenation +
dual U-Net) — `imgs/method/arch_medverse.pdf` / `arch_patchset.pdf`.

**Shared encoder.** All $T = K+1$ volumes of a task — the $K$ context
images and the target — are passed once through a single shared
convolutional encoder (a from-scratch, nnU-Net-style plain-convolution
stack, without external pretraining), producing a feature pyramid per
volume. Two disjoint subsets of its stages are read out: *coarse* stages,
resampled to a common $R \times R \times R$ token grid ($N = R^3$ cells
per volume) and consumed by the attention stack; and *fine* stages, kept
at native resolution and reserved for the decoder's skip connections. This
split keeps the attention stack on a small, fixed-size token grid while
still allowing a full-resolution decoded mask.

**Tokenization.** Each cell of every volume yields two tokens sharing one
row — an image token $z^{\text{img}}$ and a label token $z^{\text{lbl}}$ —
assembled into a single sequence together with a small number of learned
*thinking rows*, shared across all volumes, that give the attention stack
fixed-size read/write capacity not tied to any single spatial cell.
Context rows always carry the ground-truth mask in their label token; the
target's label token instead starts from either a neutral placeholder or
the previous cascade level's prediction (see Cascade), since the target
mask is exactly what is being predicted.

## Fusion (bi-axial image–label attention) — Axis 1

How a model combines image and label evidence is a central design choice.
Medverse concatenates image and label channels before a shared U-Net,
merging the two streams before most of the network's capacity is applied.
Iris instead fuses image and label features once, through a pixel
shuffle–unshuffle operation between context and target. Both commit to a
single fusion point. We instead keep image tokens $z^{\text{img}}$ and
label tokens $z^{\text{lbl}}$ as separate streams throughout the network
and let them exchange information repeatedly, over two attention axes
applied at every layer:

- **Slot axis**: at each cell, the image and label token attend to each
  other, letting label evidence sharpen the image representation and vice
  versa, independently per cell.
- **Row axis**: every cell of every volume — context, target, and the
  thinking rows — attends over the entire token sequence at once, using
  3D axial rotary position embeddings so attention is a function of
  physical distance between cells rather than raw grid index. This single
  joint attention performs within-volume self-attention and cross-context
  matching together, since tokens from every volume already share one
  sequence.

**Figure:** `imgs/method/arch_biaxial.pdf` — both axes within one layer.

Only the target's post-attention *image* token is read out for decoding;
the label token is discarded after the last layer, since it started from
a placeholder rather than real image evidence for the target row. The
decoded image tokens are projected back to the coarse token grid and
progressively fused with the encoder's fine-resolution skip features via
upsampling and convolution, one step per fine stage, into the final
full-resolution mask logits.

## Cascade (coarse-to-fine) — Axis 2

The cascade connects consecutive resolution levels through three
mechanisms, each replacing a coarser design choice made by Medverse or
Iris.

**Query prior.** Medverse feeds an upsampled, perturbed ground-truth mask
into an extra U-Net branch at every step; Iris uses no prior at all. At
level $\ell > 1$, we instead initialize the target's label token directly
from level $\ell{-}1$'s own prediction:

$$z^{\text{lbl}}_{\ell,0} = \mathrm{LblEmbed}\big(\mathrm{Warp}_{\ell-1 \to \ell}(\hat{L}^t_{\ell-1})\big)$$

where $\mathrm{Warp}$ resamples level $\ell{-}1$'s prediction onto level
$\ell$'s crop grid, correcting for the two levels' different centers,
field of view, and any augmentation applied independently to each. This
makes the previous level's belief part of the same token stream that
attends bi-axially with the image, rather than a separate side channel.

**Figure:** `imgs/method/arch_query_prior.pdf` — flow across two levels.

**Region-restricted fine processing.** Medverse and Iris re-scan the
entire volume with a sliding window at every step. Once a coarse level
has localized the structure of interest, we instead restrict the sliding
window at the next, finer level to the coarse prediction's foreground
region, thresholded and padded by a fixed margin. Compute at fine levels
then scales with the size of the structure rather than the size of the
input volume, at the cost of depending on the coarse level's localization
being approximately correct.

**Cross-level training via register tokens.** Medverse and Iris connect
levels only through the rendered prediction feeding the next step, with
no gradient path between them. We add a small set of learned register
tokens $\mathbf{r}_\ell$ that participate in level $\ell$'s attention
alongside the image and label tokens; their final state is carried into
level $\ell{+}1$ via a learned projection and a level-independent type
embedding:

$$\mathbf{r}_{\ell+1,0} = W_r\, \mathbf{r}_{\ell,\text{final}} + \mathbf{e}_{\ell+1}$$

Unlike the query prior, this path is not detached between levels: level
$\ell{+}1$'s loss can backpropagate into level $\ell$'s own attention,
giving the model a direct incentive to carry forward whatever helps the
next level — e.g. uncertainty or cues about neighboring structures —
beyond what the rendered mask alone can encode.

## Synthetic task generation — Axis 3

Real (image, mask) pairs are limited to the anatomical classes and
appearance distribution covered by the training cohort. To broaden
training beyond this, a fraction $p_{\text{synth}}$ of training tasks are
drawn from a synthetic generator instead of real context/target subjects.

**Supervoxel repainting.** A real-anatomy volume is first oversegmented
into supervoxels (unsupervised, appearance-agnostic regions). One
supervoxel is treated as the target class, and its binary mask is
label-perfect by construction. Each of the $K{+}1$ volumes needed for a
task is then produced by repainting the same supervoxel geometry with
independently sampled per-class Gaussian intensities (a Gaussian mixture
over classes), giving $K{+}1$ (image, mask) pairs that share identical
anatomy but differ in appearance — no real cross-subject registration or
annotation is required.

**Paint noise.** The default painting noise is i.i.d. per voxel: it
matches each class's target intensity variance but has near-zero spatial
autocorrelation, unlike real CT/MRI tissue (measured $\approx 0.6$
autocorrelation at native resolution, versus $\approx 0.0$ for i.i.d.
paint). An optional multi-octave variant instead sums several
coarse-to-fine random fields before painting, producing spatially
correlated texture that better matches real tissue statistics while
leaving the per-class intensity variance unchanged.

**TODO:** no architecture diagram exists yet for this axis (supervoxel →
GMM repaint → i.i.d. vs. multi-octave texture pipeline). Worth sketching
one to match the other three axes' figures.

## Training objective

Each cascade level is supervised independently at its own resolution with
a combined binary cross-entropy and soft Dice loss, summed over the $M$
levels:

$$\mathcal{L} = \sum_{\ell=1}^{M} \lambda_\ell \left( \mathcal{L}_{\text{BCE}}^\ell + \mathcal{L}_{\text{Dice}}^\ell \right)$$

Level weights $\lambda_\ell$ are uniform ($\lambda_\ell{=}1$) in the
reference configuration; the query-prior path is detached between levels,
so each level's loss trains its own prediction without back-propagating
through the coarser level's mask.
