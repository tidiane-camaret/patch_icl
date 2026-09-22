# Method (draft)

**Input:** a target volume $I^t$ and a context set of $K$ (image, mask)
pairs for the same anatomical class as the (unknown) target mask.

**Output:** a predicted target mask $\hat{L}^t$, produced by a
coarse-to-fine cascade over $M$ resolution levels $\{r_1, \dots, r_M\}$.

## Backbone

Each volume (target or context) is tokenized by a shared encoder — either
the nnU-Net residual-encoder stages (`ResEncInContext3D`) or a 3D patch
embedding (`ViTInContext3D`) — into a grid of spatial tokens with 3D axial
rotary position embeddings (RoPE). At the bottleneck, a two-stage attention
block first lets tokens attend within their own volume (target and each
context volume independently, sharing weights), then lets the target
cross-attend, read-only, into all K context bottlenecks.

**TODO:** restate/adjust to match the final architecture diagram once
settled.

## Image-label interaction

- **Medverse:** concatenates image and label before the U-Net.
- **Iris:** fuses image features and labels via pixel shuffle-unshuffle.
- **Ours: bi-axial attention.** Rather than folding the label into the
  image as a second channel before encoding, or fusing image/label
  features once, we keep image tokens and label tokens as separate streams
  and attend over two axes:
  - **Patch axis** — standard spatial self-/cross-attention among tokens
    at the same image/label identity (target self-attention, target/context
    cross-attention).
  - **Image/label axis** — at each spatial location, the image and label
    tokens attend to each other, letting label evidence sharpen the
    corresponding image features and vice versa.

  The target's label stream starts from the query prior (below) rather
  than a zero/blank mask, so this axis is informative for the target as
  well as for the contexts.

**TODO:** formalize with an equation once the exact factorization
(interleaved vs. parallel bi-axial layers) is finalized; specify how many
layers alternate between the two axes.

## Coarse-to-fine training

### Query image prior

- **Medverse:** upsampled, perturbed GT fed in as an extra U-Net branch.
- **Iris:** none.
- **Ours:** at level $\ell > 1$, the target's label tokens are initialized
  from the upsampled prediction of level $\ell - 1$:

  $$z^{\text{lbl}}_{\ell,0} = \text{Embed}(\text{Upsample}(\hat{L}^t_{\ell-1}))$$

  This makes the previous level's belief part of the same token stream
  that attends bi-axially with the image, rather than a side channel fused
  later.

### Fine level processing

- **Medverse:** sliding window over the whole volume at every step.
- **Iris:** sliding window over the whole volume.
- **Ours:** sliding window restricted to the previous level's predicted
  zones. Given level $\ell-1$'s prediction, we take its predicted
  foreground region (thresholded and padded by a margin) and restrict the
  sliding window at level $\ell$ to that region. This lets compute at fine
  levels scale with the size of the structure of interest rather than the
  size of the input volume, at the cost of depending on the coarse level's
  localization being approximately correct.

**TODO:** specify margin/threshold values and failure-mode handling (what
happens when the coarse level misses the structure entirely?).

### Cross-level training

- **Medverse:** none (levels connected only through the rendered
  prediction feeding the next autoregressive step).
- **Iris:** none.
- **Ours: register tokens.** To connect the cascade with a gradient path
  that crosses resolution levels, we add a small set of learned register
  tokens $r_\ell$ at each level. Registers participate in the bi-axial
  attention at level $\ell$ alongside image and label tokens, and their
  updated state is carried forward, via a learned projection and a
  level-type embedding, into level $\ell+1$'s register initialization:

  $$r_{\ell+1,0} = W_r \, r_{\ell,\text{final}} + e_{\ell+1}$$

  This lets the model propagate information useful for finer-level
  processing but not fully captured by the rendered segmentation mask
  alone (e.g. uncertainty, or cues about neighboring structures).

## Training objective

Each level is supervised at its own resolution with a combined BCE + soft
Dice loss, summed over levels:

$$\mathcal{L} = \sum_{\ell=1}^{M} \lambda_\ell \, (\mathcal{L}_{\text{BCE}}^\ell + \mathcal{L}_{\text{Dice}}^\ell)$$

**TODO:** state whether level weights $\lambda_\ell$ are uniform or
scheduled, matching the final training config.

## Synthetic supervision

A fraction $p_{\text{synth}}$ of training tasks are drawn from a GMM-based
synthetic generator instead of real (image, mask) pairs: a real-anatomy
supervoxel cohort is repainted with per-class Gaussian intensities, giving
a label-perfect (image, mask) pair with a novel appearance. Paint noise is
i.i.d. per voxel by default; an optional multi-octave correlated variant
(coarse-to-fine random fields, summed) better matches the spatial
autocorrelation of real tissue (measured ≈0.6 in real CT/MRI vs. ≈0.0 for
i.i.d. paint at native resolution) without changing the per-class
intensity variance.

**TODO:** figure of a synth_gmm training pair (i.i.d. vs. multi-octave
texture noise) next to a real (image, mask) pair; equation for the paint
model once finalized.

## Pooling token

Alongside the per-cell tokens (§Backbone), an optional extra "prototype"
row per volume — support and query alike — summarizes that volume's
foreground appearance as a single token: a masked average of a per-volume
feature map, projected to the token width and inserted as an extra prefix
row (same mechanism the register tokens above use). Two variants differ in
*which* feature map is pooled: a near-native-resolution stage (masking
after upsampling, following Iris's finding that this — not masking a
coarser feature — is what preserves small-structure signal), or the same
coarse grid features the backbone's per-cell tokens are already built
from (no extra encoder cost, at the price of pooling an already
spatially-blended feature).

**TODO:** equation/diagram once one variant is picked as default; cite
Iris's masked-pooling ablation directly.
