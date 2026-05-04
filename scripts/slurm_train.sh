#!/bin/bash
#SBATCH -p ml_gpu-rtx2080
#SBATCH -c 16
#SBATCH --mem 32000
#SBATCH --gres=gpu:2
#SBATCH --time=12:00:00


# interactive session : 
#srun -p ml_gpu-rtx2080 -c 20 --mem 48000 --gres=gpu:2 --time=12:00:00 --pty bash 
#srun -p ml_gpu-rtx2080 -c 20 --mem 24000 --time=4:00:00 --pty bash 

# Ensure clean GPU state
nvidia-smi

export nnUNet_raw="/work/dlclarge2/ndirt-SegFM3D/patch_icl/results/nnUNet/nnUNet_raw"
export nnUNet_preprocessed="/work/dlclarge2/ndirt-SegFM3D/patch_icl/results/nnUNet/nnUNet_preprocessed"
export nnUNet_results="/work/dlclarge2/ndirt-SegFM3D/patch_icl/results/nnUNet/nnUNet_results"

# run with sbatch scripts/slurm_train.sh
# uv run scripts/totalseg_3d_to_2d_every_n_slice.py cluster=dlclarge max_files_3d_to_2d=500
uv run nnUNetv2_train 1 3d_fullres 0 -p nnUNetPlannerResEncM -num_gpus 2


