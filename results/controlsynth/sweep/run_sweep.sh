#!/usr/bin/env bash
# Per-axis OOD sweep for the imagepfn_zoom model trained on synth=hard_diverse.
# Each run holds everything at the in-distribution hard_diverse operating point and
# varies ONE knob across levels straddling its single training value (*). Live-knob
# runs share geometry (live knobs don't change shapes); build-knob runs change geometry
# and carry their own in-range anchor. Restartable: skips any CSV already written.
set -u
CKPT=/nfs/data/nii/data1/Analysis/camaret___in_context_segmentation/ANALYSIS_20251122/results/patch_icl/2d_train/2026-06-22_kind-durian-59/best.pt
OUT=results/controlsynth/sweep
COMMON="model=imagepfn_zoom eval.checkpoint=$CKPT data.source=synthetic synth=hard_diverse synth.diversity.num_tasks=2000 eval.save_figures=false"

run() {  # name  override...
  local name=$1; shift
  local csv="$OUT/${name}.csv"
  if [[ -f "$csv" ]]; then echo "SKIP $name (exists)"; return; fi
  echo "RUN  $name :: $*"
  WANDB_MODE=offline python experiments/2d/eval.py $COMMON "eval.synth_csv=$csv" "$@" \
    > "$OUT/${name}.log" 2>&1
  echo "DONE $name -> $(grep -m1 'Wrote' "$OUT/${name}.log")"
}

# ── LIVE axes (geometry fixed; anchor.csv = shared in-dist point, * = train value) ──
# noise_level (train 0.40*)
for v in 0.10 0.25 0.55 0.70 0.85 1.00; do run "noise_${v}"        synth.live.noise_level=$v; done
# foreground_contrast (train 0.50*)
for v in 0.05 0.20 0.35 0.65 0.80;      do run "contrast_${v}"     synth.live.foreground_contrast=$v; done
# texture_heterogeneity (train 0.35*)
for v in 0.05 0.20 0.55 0.75 0.95;      do run "texture_${v}"      synth.live.texture_heterogeneity=$v; done
# support_query_shift (train 0.50*)
for v in 0.20 0.35 0.65 0.80 0.95;      do run "shift_${v}"        synth.live.support_query_shift=$v; done
# support_query_scale (train 0.45*)
for v in 0.00 0.20 0.70 0.95 1.20;      do run "scale_${v}"        synth.live.support_query_scale=$v; done
# support_query_translate (train 0.05*)
for v in 0.00 0.15 0.25 0.35;           do run "translate_${v}"    synth.live.support_query_translate=$v; done
# context_consistency (train 0.90*)
for v in 0.30 0.50 0.70 1.00;           do run "consistency_${v}"  synth.live.context_consistency=$v; done
# task_ambiguity_intensity (train 0.60*)
for v in 0.00 0.30 0.80 1.00;           do run "ambint_${v}"       synth.live.task_ambiguity_intensity=$v; done

# ── BUILD axes (geometry changes; own anchor at an in-range fixed value) ──
# region_size: training drew from [0.12,0.62]; fix at a single value to sweep, others in-dist.
for v in 0.05 0.12 0.37 0.62 0.80 0.95; do run "regionsize_${v}"   "synth.build.sampled.region_size=[$v,$v]"; done
# task_ambiguity (build): training drew from [0.30,0.80]
for v in 0.00 0.30 0.55 0.80 1.00;      do run "taskamb_${v}"      "synth.build.sampled.task_ambiguity=[$v,$v]"; done

echo "ALL DONE"
