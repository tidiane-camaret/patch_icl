# Related Work (draft)

## In-context medical image segmentation

UniverSeg popularized ICL segmentation with a CrossBlock architecture that
densely cross-attends target and context features at full resolution. Iris
instead encodes each context pair into a compact task embedding and fuses
it into the target features through pixel shuffle-unshuffle operations,
avoiding dense cross-attention but performing the image-label fusion in a
single shot with no coarse-to-fine refinement. Both were designed for 2D
slices or small, fixed-size 3D patches; naively extending them to full 3D
volumes reintroduces the cubic scaling that motivated patch- and
cascade-based alternatives.

## Coarse-to-fine and autoregressive cascades

Medverse performs next-scale autoregressive ICL: at each step, the target
image and the previous step's prediction form an "autoregressive context"
that is processed, together with the semantic (image, mask) context, by
weight-shared U-Net branches, following the next-scale prediction idea of
VAR. Crucially, every autoregressive step still performs a full
sliding-window pass over the entire volume — only the *input* to that pass
changes with scale, not its spatial extent. Deep Neural Patchworks
similarly stacks levels of increasing resolution but targets single-model
supervised segmentation rather than in-context, few-shot prediction.

In contrast, our cascade restricts the sliding window itself to the region
already localized by the previous level, so compute at fine levels scales
with the size of the structure rather than the size of the volume, and
connects levels with register tokens rather than by feeding back only the
rendered prediction.

## Image-label fusion mechanisms

How a model combines image and label information from the context is a
central design axis. Concatenation-based fusion (Medverse, and earlier 2D
CrossBlock-style models) merges the two streams early, before most of the
network's capacity is applied. Iris fuses them once via a
pixel-shuffle/unshuffle operation between context and target. We instead
treat image/label identity as a first-class attention axis, alongside the
spatial-patch axis, allowing the two streams to exchange information
repeatedly across attention layers rather than once at the input or via a
single fusion step.

**TODO:** position against additional fusion designs if relevant, e.g.
gated cross-attention or FiLM-style conditioning.

## Efficient 3D volumetric backbones

Our encoder builds on the nnU-Net residual encoder and 3D axial rotary
position embeddings for within-volume and cross-context self-attention,
following the general trend of combining convolutional feature extraction
with attention-based context aggregation in volumetric segmentation
(UNETR, 3D U-Net).

**TODO:** expand with any additional 3D-specific efficiency baselines we
end up comparing against.
