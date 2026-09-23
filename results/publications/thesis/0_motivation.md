# Motivation (draft)

Source of truth for `ics_3d_medical/sections/motivation.tex`.

In-context learning (ICL) recasts segmentation as few-shot matching:
given a target scan and a small set of (image, mask) context pairs
illustrating the structure of interest, a model predicts a mask for the
target with no task-specific fine-tuning. This removes the need to
retrain for every new anatomical class — valuable for structures that are
rare, newly defined, or annotated at only a handful of sites.

Most ICL segmentation models were designed and validated in 2D. Moving to
native 3D volumes is attractive — CT/MRI are inherently volumetric, and
slice-wise processing discards through-plane context needed to resolve
thin or elongated structures — but it sharpens an already difficult
resolution/compute trade-off: dense cross-attention between a target
volume and its context volumes scales with the cube of side length, so a
design affordable at $64^3$ becomes impractical at $256^3$.

This trade-off is not one problem but three, and this thesis addresses
each with its own mechanism:

1. **How should image and label evidence be combined?** Concatenating
   image and mask channels before a shared encoder, or fusing them once
   through a single operation, commits to one fusion point before most of
   the network's capacity is applied. Chapter [Methodology] proposes
   *bi-axial image--label attention*, keeping the two streams separate and
   letting them exchange information repeatedly.
2. **Where should compute be spent?** Re-scanning an entire volume with a
   sliding window at every processing step wastes compute on regions
   already known not to contain the structure of interest. Chapter
   [Methodology] proposes a *coarse-to-fine cascade* that restricts fine
   processing to the region a coarse pass has already localized, connected
   across levels by a query prior and register tokens that carry a
   gradient path between resolutions.
3. **Where does training signal come from beyond limited annotation?**
   Real (image, mask) pairs are limited to the classes and appearance
   distribution covered by the training cohort. Chapter [Methodology]
   proposes a *synthetic task generator* that repaints real anatomical
   geometry with novel, controllable appearance, producing label-perfect
   training tasks beyond the annotated cohort.

**Contributions.**

- A bi-axial image--label attention mechanism for in-context 3D
  segmentation, contrasted against early-concatenation (Medverse) and
  one-shot fusion (Iris) baselines.
- A coarse-to-fine cascade in which the query prior, region-restricted
  fine processing, and cross-level register training are evaluated
  against the sliding-window-over-the-whole-volume design used by prior
  work.
- A GMM-based synthetic task generator with a texture-noise variant that
  better matches real tissue's spatial autocorrelation, evaluated against
  real-only and i.i.d.-noise training.
- Evaluation on TotalSegmentator CT and a range of out-of-distribution
  cohorts and modalities, reporting where each mechanism helps and where
  it does not.

**Outline.** Chapter [Fundamentals] introduces in-context segmentation,
the volumetric encoder and attention machinery the method builds on, and
synthetic-data background, then positions UniverSeg, Iris, and Medverse
against the three axes above. Chapter [Methodology] details the model.
Chapter [Results] evaluates each axis in turn. Chapter [Discussion]
synthesizes what the evidence supports and what remains open, and Chapter
[Conclusion] summarizes.
