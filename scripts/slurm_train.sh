#!/bin/bash
#SBATCH -p ml_gpu-rtx2080
#SBATCH -c 20
#SBATCH --mem 24000
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00


# interactive session : 
#srun -p ml_gpu-rtx2080 -c 20 --mem 48000 --gres=gpu:2 --time=12:00:00 --pty bash 
#srun -p ml_gpu-rtx2080 -c 20 --mem 24000 --time=4:00:00 --pty bash 

# Ensure clean GPU state
nvidia-smi

# run with sbatch scripts/slurm_train.sh
# uv run scripts/totalseg_3d_to_2d_every_n_slice.py cluster=dlclarge max_files_3d_to_2d=500
uv run scripts/train_vit_in_context.py \
    train.checkpoint=/work/dlclarge2/ndirt-SegFM3D/patch_icl/results/vit_incontext_best.pt

