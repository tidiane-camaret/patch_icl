## patch_icl

# write totalsegmentator imgs/masks as resizes numpy arrays
python scripts/convert_to_npy.py --size 128 128 128 --overwrite
# generate synthetic labels
python scripts/synth_labels/generate.py --method slic --union --overwrite --size 64 64 64 --workers 16
# generate bbox cache 
python scripts/build_bbox_cache.py --workers 16

# train vit_in_context
python scripts/train_vit_in_context.py data.synth_method=slic data.synth_unions=true data.p_synth=0.5 train.checkpoint=/home/dpxuser/dev/patch_icl/results/checkpoints/vit_incontext_best.pt

# train resnetenc_in_context
python scripts/train.py experiment=1_nnunet_augs train.checkpoint=/home/dpxuser/dev/patch_icl/results/checkpoints/resenc_in_context_best.pt

## benchmark 
# uses config.yaml defaults: resenc_in_context, 64³, batch=2, K=3
python scripts/benchmark.py --bench_aug --bench_data
# sweep image sizes with dataloader
python scripts/benchmark.py --bench_data --image_size 64 128
# compare both models
python scripts/benchmark.py --model vit_in_context resenc_in_context --image_size 64 128