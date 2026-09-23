# Fundamentals (draft)

Source of truth for `ics_3d_medical/sections/fundamentals_*.tex`. Not yet
drafted. Unlike the CVPR article's terse Related Work, this chapter should
explain background pedagogically (thesis readers aren't assumed to know
the prior architectures already).

## In-Context Segmentation
**TODO** (`fundamentals_icl.tex`): formal few-shot/in-context segmentation
problem definition; why it removes the need to retrain per class.

## Volumetric Encoders and Attention
**TODO** (`fundamentals_backbone.tex`): nnU-Net residual/plain-conv
encoders, transformer attention basics, 3D axial RoPE — whatever the
Methodology chapter assumes but doesn't re-derive.

## Synthetic Training Data
**TODO** (`fundamentals_synthdata.tex`): domain randomization / synthetic
data background (SynthSeg, GIN/IPA) — grounds Axis 3, which the article's
Related Work never gave space to.

## Related Work
**TODO** (`fundamentals_relatedwork.tex`): UniverSeg, Iris, Medverse
explained in depth (not just one contrasting line each, as in the
article) — end with a table positioning our 3 axes against them.
