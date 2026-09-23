# Abstract (draft)

In-context learning (ICL) lets a segmentation model adapt to a new anatomical
structure at inference time from a handful of (image, mask) context pairs,
without any gradient update. Extending ICL segmentation to full 3D volumes is
attractive for CT/MRI workflows but sharpens the resolution/compute
trade-off: dense cross-attention between a target volume and its contexts
scales cubically with side length, forcing prior work to downsample, fall
back to per-slice processing, or tile the volume with independent,
context-blind sliding windows.

We propose a 3D in-context segmentation model built around two ideas:

1. **Bi-axial image-label attention** — target and context tokens exchange
   information jointly along the spatial-patch axis and the image/label
   axis, instead of early concatenation (Medverse) or one-shot
   pixel-shuffle fusion (Iris).
2. **Register-carried coarse-to-fine cascade** — each finer level's label
   tokens are initialized from the previous level's upsampled prediction,
   the sliding window at fine levels is restricted to the previously
   predicted region, and a small set of learned register tokens carries
   hidden state across levels so the cascade trains end-to-end.

We evaluate on TotalSegmentator CT at fixed spacing, reporting Dice, NSD,
and compute (time, FLOPs, VRAM) against Medverse and Iris, and ablate the
query prior, region-restricted fine processing, and cross-level register
training independently.

**TODO:** headline numbers once experiments land — e.g. accuracy at what
fraction of baseline compute, and which categories (small/thin structures?)
benefit most from bi-axial attention.

**TODO:** generalization headline — other CT cohorts, MRI, far-OOD.
