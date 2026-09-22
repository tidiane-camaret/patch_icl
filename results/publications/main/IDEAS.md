# Method
## Image-label interaction
- Medverse : concat image and label before u-net
- Iris : fuse images features and labels using pixel shuffle-unshuffle
- Ours : bi-axial attention (patch,img<->label)

## Coarse-to-fine training
### Query image prior
- Medverse : upsampled perturbed GT as extra U-Net branch
- Iris : None
- Ours : upsampled prev. level pred as label token initialisation

### Fine level processing
- Medverse : sliding window over whole volume
- Iris : sliding window over whole volume
- Ours : sliding window over prev. level predicted zones

### Cross-level training
- Medverse : None
- Iris : None
- Ours : gradient flow across level (register tokens)

# Dataset 
Totalsegmentator

# Results
## Fixe spacing
- datasets : Totalseg CT, 3mm spacing crops
- accuracy (dice,nsd) and compute (time, flops, vram)
- show strenghts (e.g. small objects)
## Coarse-to-fine
- datasets : Totalseg CT
- show effect of query prior and cross-level training
- show acc/compute tradeoff when restricting sliding window to prev.level predictions

## Generalization
- other CT datasets
- other modalities (MRI, etc)
- Far OOD tasks

## Compute scaling (token compression)
- v2's compress_m (R³ raw tokens -> m compressed slots) cuts the main transformer's own
  cost hugely (measured: ~20x wall-clock, ~74x FLOPs at the real R=16/m=128 configs) but
  barely moves END-TO-END time (44.4 -> 35.1 ms, 1.27x) or FLOPs-to-time efficiency
- why: transformer is a MINORITY of the budget even uncompressed (13.8 of 44.4 ms, 31%) --
  encoder (flat, resamples a fixed-size input) and decode (untouched, sometimes heavier in
  v2) dominate regardless of token count, so cutting the transformer has a hard ceiling on
  what it can save
- FLOPs cut (3.16x total) overstates the wall-clock win (1.27x): attention matmuls run
  near-peak on tensor cores, conv/decode doesn't -- cutting FLOP-cheap-per-ms attention and
  leaving FLOP-expensive-per-ms conv untouched
- GPU sweep confirms this is a real crossover, not a fluke: transformer FLOPs scale ~linear
  in token count L at small L, bending toward quadratic (local exponent 1.0 -> 1.8) past a
  few thousand tokens -- v1's dense R³ tokens (L=8192 real config) sit deep in the
  super-linear zone, v2's compressed tokens (L=258) sit in the flat/linear zone; pushing R
  alone (R=4->32) blows v1's total time up 12.7x vs v2's 2.0x over the same range
  (results/presentations/perf/bench_feature_scaling.py, feature_scaling_results.json)
- publication angle: acc/compute tradeoff plot should show compute broken into encode/
  attend/decode, not just total -- a token-length ablation only pays off in the total-time
  column once the attend stage is already >30-40% of the budget (i.e. at higher resolution/
  more context than the 128^3/K=1 default), otherwise compression is invisible end-to-end
  despite being real
