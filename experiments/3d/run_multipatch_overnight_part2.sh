#!/bin/bash
# Resume phases 2-8 of run_multipatch_overnight.sh -- phase 1 already completed and
# its JSON is safe under results/experiments/3d/multipatch_cascade_probe/
# (20261013_235557_flare22.json). The original process died cleanly (no traceback,
# GPU fully freed) around 03:04, consistent with an environment/container restart,
# not a code bug. Run from the REPO ROOT.
set -uo pipefail

CKPT=/nfs/data/nii/data1/Analysis/camaret___in_context_segmentation/ANALYSIS_20251122/results/patch_icl/3d_train/2026-09-28_168_cascade_real_host_all_shapes/best.pt
SCRIPT=experiments/3d/compare_multipatch_flare22.py
COMMON="eval.model=patchset3d eval.checkpoint=$CKPT eval.batch_size=1 augmentations.enabled=false"

log() { echo "[$(date +%H:%M:%S)] $*"; }

log "=== 2/8: FLARE22 full 13-class, n=[1,2], spacings=[4,3] -- SIZE PROXY A: T=128 fixed (FOV preserved), coarser fine-level resolution ==="
python $SCRIPT $COMMON eval.split=val dataset=flare22 \
  data.crop_spacing_mm=4 +data.cascade_spacings=[4,3] \
  +data.cascade_n_patches_sweep=[1,2]
log "=== 2/8 done (exit $?) ==="

log "=== 3/8: FLARE22 full 13-class, n=[1,2], spacings=[4,6] -- SIZE PROXY A, even coarser ==="
python $SCRIPT $COMMON eval.split=val dataset=flare22 \
  data.crop_spacing_mm=4 +data.cascade_spacings=[4,6] \
  +data.cascade_n_patches_sweep=[1,2]
log "=== 3/8 done (exit $?) ==="

log "=== 4/8: FLARE22 full 13-class, n=[1,2], image_size=[64,64,64] -- SIZE PROXY B ==="
python $SCRIPT $COMMON eval.split=val dataset=flare22 \
  data.image_size=[64,64,64] data.crop_spacing_mm=4 +data.cascade_spacings=[4,1.5] \
  +data.cascade_n_patches_sweep=[1,2]
log "=== 4/8 done (exit $?) ==="

log "=== 5/8: FLARE22 full 13-class, n=[1,2], image_size=[32,32,32] -- SIZE PROXY B, even smaller ==="
python $SCRIPT $COMMON eval.split=val dataset=flare22 \
  data.image_size=[32,32,32] data.crop_spacing_mm=4 +data.cascade_spacings=[4,1.5] \
  +data.cascade_n_patches_sweep=[1,2]
log "=== 5/8 done (exit $?) ==="

log "=== 6/8: totalseg CT+MRI (multisource), n=[1,2,4], mode=[prob,uncertainty], spacings=[4,1.5], eval_epoch_length=150 ==="
python $SCRIPT $COMMON eval.split=val dataset=multisource_ct_mri \
  data.crop_spacing_mm=4 +data.cascade_spacings=[4,1.5] \
  data.source_mix.eval_epoch_length=150 \
  +data.cascade_n_patches_sweep=[1,2,4] +data.cascade_sample_mode_sweep=[prob,uncertainty]
log "=== 6/8 done (exit $?) ==="

log "=== 7/8: totalseg CT+MRI (multisource), n=[1,2], spacings=[4,3] -- SIZE PROXY A, in-distribution ==="
python $SCRIPT $COMMON eval.split=val dataset=multisource_ct_mri \
  data.crop_spacing_mm=4 +data.cascade_spacings=[4,3] \
  data.source_mix.eval_epoch_length=150 \
  +data.cascade_n_patches_sweep=[1,2]
log "=== 7/8 done (exit $?) ==="

log "=== 8/8: totalseg CT+MRI (multisource), n=[1,2], image_size=[64,64,64] -- SIZE PROXY B, in-distribution ==="
python $SCRIPT $COMMON eval.split=val dataset=multisource_ct_mri \
  data.image_size=[64,64,64] data.crop_spacing_mm=4 +data.cascade_spacings=[4,1.5] \
  data.source_mix.eval_epoch_length=150 \
  +data.cascade_n_patches_sweep=[1,2]
log "=== 8/8 done (exit $?) ==="

log "=== ALL REMAINING RUNS COMPLETE ==="
