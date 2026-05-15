# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

`patch_icl` is a research project for **in-context 3D medical image segmentation** on CT volumes from the TotalSegmentator dataset. Given a target CT volume and K context (image, mask) pairs for the same organ class, the model predicts a binary segmentation mask for the target.

## Code Guidelines

- Write understandable code with short docstrings
- Log changes to docs/logs.md
- Write tests only when necessary
- Related repos in `/software/notebooks/camaret/repos`: Medverse, nnInteractive_fork, PatchWork, Neuroverse3D

## Common commands

```bash
# Data preparation (run once before training)
python scripts/convert_to_npy.py --size 64 64 64          # Convert nii.gz → .npy at 64³
python scripts/synth_labels/generate.py --method slic --union --overwrite --size 64 64 64 --workers 16

# Training (Hydra config, overrides via dot-notation)
python scripts/train.py                                    # ResEncInContext3D (default)
python scripts/train.py model.name=vit_in_context
python scripts/train.py train.epochs=10 data.max_train_subjects=50  # quick debug run
python scripts/train.py train.checkpoint=results/checkpoints/resenc_in_context_best.pt  # resume

# Benchmarking
python scripts/benchmark.py --bench_aug --bench_data
python scripts/benchmark.py --model vit_in_context resenc_in_context --image_size 64 128
```

## Architecture

### Two model variants (both in `src/`)

**`ResEncInContext3D`** (`src/models/resenc_in_context.py`):
- Encoder: nnUNet `ResidualEncoderUNet.encoder` (4 stages, nnUNetResEncM blocks). Input is 2-channel `[image, mask]` — target gets a zero mask channel.
- Bottleneck: Stage-1 within-volume self-attention (RoPE) shared over target + all K contexts; Stage-2 cross-context attention where target cross-attends to all K context bottlenecks (read-only).
- Decoder: U-Net skip-connection decoder (trilinear upsample + concat + conv × 3).
- Uses 3D axial RoPE (`src/rope.py`) — head_dim split into 3 axis chunks; use `rope_theta=100` for small grids (≤32 tokens/axis).

**`ViTInContext3D`** (`src/vit_in_context.py`):
- Patch embed for image + separate patch embed for mask; context tokens = img_tok + mask_tok + pos_embed.
- Same two-stage attention design but without skip connections; decoder is trilinear upsample → 1×1×1 conv.
- Learnable positional embedding (vs. RoPE in ResEnc).

### Data pipeline (`src/totalseg_dataloader_incontext.py`)

`TotalSegInContextDataset` returns `{image, label, context_in, context_out}` per item. Key design decisions:
- **Fast path**: uses pre-resized `ct_{D}x{H}x{W}.npy` and `label_{D}x{H}x{W}.npy` — skips CPU interpolation entirely. Only falls back to native .nii.gz if pre-resized files are absent.
- **Scan cache**: on first init, scans all `label.npy` files to build a `subject→classes` index (pickle, keyed by SHA of subject list). Covers all 117 TotalSegmentator classes so it is valid across any class subset.
- **Synth path**: supervoxel-based synthetic labels (`label_synth_{method}[_union].npy`). At `p_synth` fraction of iterations, loads a supervoxel and creates K+1 independently augmented copies instead of real context subjects.
- **Class-balanced sampling**: samples class uniformly first, then subject — prevents large-anatomy classes from dominating.

### Augmentation (`src/augmentations.py`, `configs/augmentations.yaml`)

Three modes: task aug (same geometric params for all K+1 volumes in a task), intensity aug (independent per volume), synth aug (heavy independent aug per supervoxel copy).

## Configuration (Hydra)

Config entry point: `configs/config.yaml`. Cluster-specific overrides in `configs/cluster/`:
- `nfs.yaml` — NFS cluster (primary dev), `batch_size=16`, `workers=16`
- `meta.yaml` — dlclarge2 cluster, `batch_size=4`, `workers=4`

Data paths (`paths.totalseg`, `paths.nnunet`) are set per-cluster. Override cluster config with `cluster=meta`.

W&B logging is controlled by `train.wandb_project` (set to `null` to disable).

## Data layout

TotalSegmentator root contains `s0000/`, `s0001/`, … each with:
- `ct.nii.gz` — raw CT
- `ct_{D}x{H}x{W}.npy` — pre-resized float16 CT
- `label.npy` — merged label volume (uint8, all 117 classes)
- `label_{D}x{H}x{W}.npy` — pre-resized label
- `label_synth_{method}[_union].npy` — supervoxel labels
- `label_synth_{method}[_union]_{D}x{H}x{W}.npy` — pre-resized supervoxel labels
- `meta.csv` at root level — train/val/test splits (semicolon-delimited, columns `image_id`, `split`)

Checkpoints saved as `results/checkpoints/{model_name}_best.pt`. Loaded with `torch.load` then `removeprefix("_orig_mod.")` to strip `torch.compile` prefix.
