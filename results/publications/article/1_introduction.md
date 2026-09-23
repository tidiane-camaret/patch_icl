# Introduction (draft)

In-context learning (ICL) recasts segmentation as few-shot matching: given a
target scan and a small set of (image, mask) context pairs illustrating the
structure of interest, the model predicts a mask for the target with no
task-specific fine-tuning. This removes the need to retrain for every new
anatomical class — valuable for structures that are rare, newly defined, or
annotated at only a handful of sites.

Most ICL segmentation models were designed and validated in 2D (UniverSeg,
Iris, Medverse). Moving to native 3D volumes is attractive — CT/MRI are
inherently volumetric, and slice-wise processing discards through-plane
context needed to resolve thin or elongated structures — but it sharpens an
already difficult resolution/compute trade-off. Dense cross-attention
between a target volume and K context volumes scales with the cube of side
length, so a design affordable at 64³ becomes impractical at 256³. Existing
3D-capable approaches each make a different concession:

- **Medverse** processes the whole volume through a sliding window at
  *every* resolution step of its autoregressive cascade, re-visiting the
  entire volume even once coarse levels have already localized the target.
- **Iris** fuses context information into the target through a single-shot
  pixel-shuffle/unshuffle operation, with no iterative refinement across
  scales.

Our own concurrent 2D work on hierarchical patch selection (PatchICL, CVPR
submission) showed that supervising *which regions to process*, rather than
relying on attention alone to discover them, yields a favorable
accuracy/compute trade-off. Here we ask whether the same principle extends
to full 3D volumes, and how the image-label interaction itself should be
designed once both axes (space, and image/label identity) matter jointly.

## Our approach

**Bi-axial image-label attention.** Instead of concatenating image and mask
channels before a shared encoder (Medverse) or fusing them once via pixel
shuffle-unshuffle (Iris), target and context tokens attend jointly along two
axes: a spatial-patch axis (standard within-volume / cross-context
attention) and an image/label axis that lets label information at a given
location refine, and be refined by, the corresponding image features. This
keeps the two streams distinct for longer while allowing early, repeated
exchange.

**Register-carried coarse-to-fine cascade.** The model processes a volume
through a small number of resolution levels, connected by:

1. a **query image prior** — the previous level's upsampled prediction
   initializes the target's label tokens at the next level (rather than a
   separate perturbed-GT branch as in Medverse, or nothing at all as in
   Iris);
2. **region-restricted fine processing** — once a coarse level localizes
   the structure, the sliding window at finer levels is restricted to the
   predicted zone instead of re-scanning the full volume.

To train the cascade as a single model rather than a chain of
independently-supervised stages, we add a small number of learned
**register tokens** that carry hidden state from one level's processing
into the next, giving the optimizer a gradient path that crosses resolution
levels.

## Contributions

1. A bi-axial image-label attention mechanism for in-context 3D
   segmentation, contrasted directly against early-concatenation and
   one-shot fusion baselines.
2. A coarse-to-fine cascade in which the query prior and the fine-level
   region restriction are ablated independently, quantifying each one's
   contribution to the accuracy/compute trade-off.
3. Register-token cross-level training that lets gradients flow across the
   cascade, evaluated against per-level-only supervision.
4. Evaluation on TotalSegmentator at fixed spacing against Medverse and
   Iris (Dice, NSD, time, FLOPs, VRAM), plus generalization experiments on
   other CT datasets, other modalities, and far-OOD tasks.

**TODO:** sharpen the framing once we know which contribution carries the
paper — is it the bi-axial attention, the cascade, or the combination?
