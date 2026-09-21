# Experiments (draft)

## Setup

- **Dataset:** TotalSegmentator CT. **TODO:** state the train/held-out
  anatomical class split — following the same held-in/held-out protocol
  used in the 2D paper to test in-context generalization to unseen
  classes, not just unseen subjects.
- **Baselines:** Medverse, Iris. **TODO:** state whether baselines are
  retrained on the same data/split or evaluated with released weights, and
  the number of context pairs K used at inference.
- **Metrics:** Dice, normalized surface distance (NSD) for accuracy;
  wall-clock time, FLOPs, peak VRAM for compute.

## Fixed spacing

- Datasets: TotalSegmentator CT, 3mm spacing crops.
- Report accuracy (Dice, NSD) and compute (time, FLOPs, VRAM) against
  Medverse and Iris, isolating the image-label interaction design
  (bi-axial vs. concat vs. pixel-shuffle) from the cascade, which is
  evaluated separately below.
- Show strengths — expect an advantage on small/thin structures, similar
  to the pattern observed for patch-based selection in the 2D work.

**TODO:** table with per-method Dice/NSD/time/FLOPs/VRAM; break down by
anatomical category (organs, bones, vessels, muscles) once numbers land.

## Coarse-to-fine

- Datasets: TotalSegmentator CT.
- Show the effect of the query prior and of cross-level (register)
  training, ablated independently against: (a) blank label-token init
  instead of the query prior, (b) full-volume sliding window instead of
  region-restricted, (c) independently-supervised levels instead of
  register-carried training.
- Show the accuracy/compute trade-off as the sliding window restriction is
  relaxed (full volume → tight crop around the previous prediction),
  analogous to the resolution/FLOPs Pareto plot in the 2D paper.

**TODO:** ablation table (Dice, time, FLOPs per configuration); Pareto
plot of Dice vs. compute as the region-restriction margin is swept.

## Generalization

- **Other CT datasets:** **TODO** — list held-out CT cohorts and results.
- **Other modalities:** **TODO** — MRI (e.g. TotalSegmentator MRI)
  evaluated with models trained only on CT.
- **Far OOD tasks:** **TODO** — non-anatomical or pathology-driven
  segmentation tasks, probing whether the region-restricted cascade still
  localizes correctly when the coarse level's prior is unreliable.
