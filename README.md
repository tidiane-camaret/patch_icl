## patch_icl

In-context 3D medical image segmentation: given a target CT/MRI volume and K context
(image, mask) pairs for the same organ, predict a mask for the target. Primary dataset is
TotalSegmentator (CT + MRI); also trains/evals on several OOD sources (ISLES22, Shifts-MS,
ATLAS v2, GNC_705, NasalSeg, ...).

## Inference

Run the trained cascade on real nifti files, works from the shared `patchset` conda env. Full usage: `docs/inference_cli.md`.

```bash
conda activate patchset

# one target
patchset-infer \
  --target tgt_ct.nii.gz \
  --context ctx_ct.nii.gz ctx_mask.nii.gz \
  --checkpoint /nfs/.../best.pt \
  --out pred.nii.gz [--gt gt_mask.nii.gz]

# many targets, one checkpoint load instead of one process per target
# (JSON-Lines manifest, see docs/inference_cli.md)
patchset-infer --manifest tasks.jsonl --checkpoint /nfs/.../best.pt --out-dir preds/

# multi-label (one id-valued context mask -> many organs in one run)
patchset-infer ... --labels 1,5,13 --batch-size 4

# match the checkpoint's own training config -- getting these wrong doesn't error,
# it silently changes what the model sees (the CLI prints a drift warning either way)
#   --crop-spacings S0,S1   coarse->fine FOV schedule in mm/voxel (default 4,1.5)
#   --query-prior pred|none v2 cascade query-mask prior mode (default pred)
#   --mask-downsample occupancy|soft   context-mask downsample (default occupancy)
#   --mask-occupancy-thr    omit it -- auto-inherits the checkpoint's own training value

# perf flags (off by default, try both -- they don't clearly stack):
#   --gpu-realize-crop     GPU-side crop resample, ~25-35% faster on multi-label runs
#   --recrop-workers N     thread fan-out for crop loading, ~2x faster on the plain CPU path
```

After changing code either CLI depends on, re-sync the vendored env copy:
`bash scripts/sync_patchset_env.sh`.

### Add nifti files to a nora project 

```bash
# from the project's filesytem
nora -p camaret___in_context_segmentation --add pred.nii.gz

# outside of the project's filesytem
nora -p camaret___in_context_segmentation --importfiles \
  /nfs/data/nii/data1/camaret___in_context_segmentation/10276934/20250926154617/ML/pred.nii.gz \
  "(?<patients_id>\d+)/(?<studies_id>\d+)/ML/" \
  "ML/"
```

### Creating the `patchset` env

Ampere nodes only (nero, thor, loki — see `requirements-patchset-infer.txt`). **Install
torch first, then `dynamic_network_architectures` with `--no-deps`** — a bare install pulls
a newer torch + a cu13 cuDNN wheel that silently breaks the primus encoder's first conv3d
(`CUDNN_STATUS_NOT_INITIALIZED`, looks intermittent since small convs still work):

```bash
conda create -n patchset python=3.11 -y
conda activate patchset
pip install "torch==2.6.0" "torchvision==0.21.0" --extra-index-url https://download.pytorch.org/whl/cu124
pip install --no-deps dynamic_network_architectures==0.4.4 einops timm
pip install numpy scipy nibabel matplotlib tqdm "hydra-core>=1.3" "omegaconf>=2.3" wandb  # rest of requirements-patchset-infer.txt

python -c "import torch; assert torch.backends.cudnn.version() == 90100"  # sanity check

bash scripts/sync_patchset_env.sh   # populate code + create the patchset-infer launchers
```

## Environment

uv-managed, **node-specific** (`.venv` itself is stale/corrupted — don't use it):

```bash
# Ampere nodes (nero, thor, loki) -- cu124
UV_PROJECT_ENVIRONMENT=.venv_nero uv sync --extra cu124

# Blackwell (odin, sm_120) -- needs cu128
UV_PROJECT_ENVIRONMENT=.venv_blackwell uv sync --extra cu128
```

Nifti CLI inference (above) only needs the separate shared `patchset` conda env — no uv
env or repo checkout required for that path.

## Data prep (run once before training)

```bash
python scripts/convert_to_npy.py --size 128 128 128        # nii.gz -> pre-resized .npy per subject
python scripts/synth_labels/generate.py --method slic --union --overwrite --size 128 128 128 --workers 16
```

## Training

`experiments/3d/train.py`, Hydra-configured, overrides via dot-notation. **Default model
is medverse** — the cascade/patchset3d architecture (the one actually used for inference,
above) is reached via `experiment=<preset>`, not the bare command:

```bash
python experiments/3d/train.py                                    # medverse on totalseg (default)
python experiments/3d/train.py dataset=omnisynth3d                # train on omniSynth-3D
python experiments/3d/train.py train.epochs=10 data.max_train_subjects=50   # quick debug run
python experiments/3d/train.py train.checkpoint=results/checkpoints/best.pt # resume

# patchset3d + v2 cascade (coarse->fine), via a numbered experiment preset --
# see configs/experiment/3d/experiment/ for the full catalog, each with a header
# comment explaining what it changes and why
python experiments/3d/train.py experiment=168_cascade_real_host_all_shapes
```

## Eval / benchmark

```bash
python experiments/3d/eval.py                                      # medverse on totalseg (test)
python experiments/3d/eval.py eval.model=patchset3d eval.checkpoint=/nfs/.../best.pt
python experiments/3d/eval.py dataset=omnisynth3d eval.split=val    # eval on omniSynth-3D

# cross-model comparison on a held-out OOD source
python experiments/3d/eval.py dataset=shifts_ms eval.model=medverse \
  eval.checkpoint=/nfs/.../best.pt data.crop_spacing_mm=3 eval.split=test eval.n_subjects=null
```

## Known gotchas

- **Eval DataLoader CUDA crash** (`CUDA error: initialization error` on the first eval
  batch): fork-after-CUDA-init. Workaround: `eval.workers=0`.
- **`git commit` hangs (D-state)** on this NFS mount, occasionally with a stale
  `index.lock` left behind — don't retry blindly from a subagent; investigate the lock
  before deleting it.
- **No GPU on the dev/login node** — varies by session, check `nvidia-smi` before assuming;
  when there's none, verify by resolving configs / reading `plans.json`/checkpoints rather
  than forwarding a model.

## More detail

- `docs/inference_cli.md` — full inference CLI reference
- `docs/logs.md` — dev log, the actual source of truth for "how was X done / what command
  produced result Y"
- `docs/datasets/` — per-dataset conversion/eval notes
