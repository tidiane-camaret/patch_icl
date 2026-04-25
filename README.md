# patch_icl

python scripts/convert_to_npy.py --size 64 64 64 

python scripts/synth_labels/generate.py --method slic --union --overwrite --size 64 64 64 --workers 16

python scripts/train_vit_in_context.py data.synth_method=slic data.synth_unions=true data.p_synth=0.5 train.checkpoint=/home/dpxuser/dev/patch_icl/results/checkpoints/vit_incontext_best.pt