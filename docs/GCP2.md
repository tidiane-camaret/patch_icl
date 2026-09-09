# Running data preparation and training on GCP

This guide is split into two phases:

1. **CPU data preparation:** download TotalSegmentator, convert the NIfTI files to `.npy`, validate the result, and upload it to Cloud Storage.
2. **TPU training:** restore the prepared dataset on a TPU VM and run training.

The converter is CPU-only, but it imports the shared dataset module, which imports CPU PyTorch. Do not install `torch_xla` or create a TPU for the data-preparation phase.

## 0. Recommended CPU dry-run

Use a normal Compute Engine VM first. For a cheap smoke test, start with `e2-standard-8` (8 vCPUs, 32 GB RAM) and a small `--limit`; it is available broadly and does not require a GPU. For the complete conversion, move to `n2d-standard-16` (16 vCPUs, 64 GB RAM) if available in the chosen zone. The converter uses one process per worker and each process can hold a full volume, so increase `--workers` only after checking memory use.

Google's current machine-family guidance lists E2 as the lowest-cost general-purpose family and N2D as suitable for batch processing. Check availability and current pricing in the selected zone before creating the VM. Spot provisioning can reduce compute cost, but it may interrupt a long conversion; the converter is resumable because existing `.npy` files are skipped.

Run these commands from Cloud Shell or a local machine with an authenticated `gcloud` CLI:

```bash
export PROJECT=atomic-acrobat-308517
export ZONE=us-central1-b
export CPU_VM=patch-icl-prep
export DATA_DISK=patch-icl-prep-data
export BUCKET=atomic-acrobat-totalseg

gcloud config set project "$PROJECT"
gcloud compute instances create "$CPU_VM" \
    --project="$PROJECT" \
    --zone="$ZONE" \
    --machine-type=n2d-standard-32 \
    #--provisioning-model=SPOT \
    --boot-disk-size=50GB \
    --boot-disk-type=pd-balanced \
    --image-family=ubuntu-2404-lts-amd64 \
    --image-project=ubuntu-os-cloud \
    --scopes=cloud-platform

# Size this disk after checking the raw archive and available quota. Keep enough
# space for raw NIfTI files and the generated .npy files at the same time.
gcloud compute disks create "$DATA_DISK" \
    --project="$PROJECT" --zone="$ZONE" \
    --type=pd-balanced --size=500GB
gcloud compute instances attach-disk "$CPU_VM" \
    --project="$PROJECT" --zone="$ZONE" \
    --disk="$DATA_DISK"

gcloud compute ssh "$CPU_VM" --project="$PROJECT" --zone="$ZONE"
```

On the VM, format and mount the data disk once:

```bash
sudo apt-get update
sudo apt-get install -y git unzip wget tmux
lsblk
# Confirm the data disk (usually /dev/sdb) before running mkfs. This erases it.
sudo mkfs.ext4 -F /dev/sdb
sudo mkdir -p /mnt/data
sudo mount /dev/sdb /mnt/data
sudo chown "$USER:$USER" /mnt/data
df -h /mnt/data
```

Install the repository's locked CPU environment. This installs CPU PyTorch, which is required by `src/totalseg_dataset.py`, but no CUDA or TPU packages:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
git clone https://github.com/tidiane-camaret/patch_icl "$HOME/patch_icl"
cd "$HOME/patch_icl"
uv sync --extra cpu --frozen
```

Download and extract the dataset onto the attached disk. If you have already downloaded the ZIP locally, upload it once to Cloud Storage from your local machine or Cloud Shell:

```bash
export PROJECT=atomic-acrobat-308517
export BUCKET=atomic-acrobat-totalseg

gcloud config set project "$PROJECT"
gcloud storage buckets create "gs://$BUCKET" \
    --project="$PROJECT" --location=us-central1  # once
gcloud storage cp \
    ./Totalsegmentator_dataset_v201.zip \
    "gs://$BUCKET/raw/Totalsegmentator_dataset_v201.zip"
```

On the CPU VM, use the bucket copy instead of downloading from Zenodo. The ZIP remains in the bucket and can be reused by future VMs:

```bash
export BUCKET=atomic-acrobat-totalseg
mkdir -p /mnt/data/totalseg
cd /mnt/data
gcloud storage cp \
    "gs://$BUCKET/Totalsegmentator_dataset_v201.zip" \
    ./Totalsegmentator_dataset_v201.zip
unzip Totalsegmentator_dataset_v201.zip -d totalseg
rm Totalsegmentator_dataset_v201.zip
du -sh /mnt/data/totalseg
df -h /mnt/data
```

If the ZIP is not already in the bucket, download it directly from Zenodo instead:

```bash
mkdir -p /mnt/data/totalseg
cd /mnt/data
wget -O totalseg.zip "https://zenodo.org/records/10047292/files/Totalsegmentator_dataset_v201.zip?download=1"
unzip -q totalseg.zip -d totalseg
rm totalseg.zip
du -sh /mnt/data/totalseg
df -h /mnt/data
```

Run the dry-run first. It exercises the real NIfTI loading, CT normalisation, label merging, multiprocessing, and output layout without processing all subjects:

```bash
cd "$HOME/patch_icl"
uv run scripts/convert_to_npy.py \
    --data /mnt/data/totalseg --workers 24

find /mnt/data/totalseg -maxdepth 2 -name 'ct.npy' -o -name 'label.npy'
python - <<'PY'
from pathlib import Path
import numpy as np

root = Path('/mnt/data/totalseg')
subjects = sorted(p for p in root.iterdir() if p.is_dir())[:2]
assert subjects and all((p / 'ct.npy').exists() and (p / 'label.npy').exists() for p in subjects)
for subject in subjects:
    image = np.load(subject / 'ct.npy', mmap_mode='r')
    label = np.load(subject / 'label.npy', mmap_mode='r')
    assert image.shape == label.shape
    assert image.dtype == np.float16 and label.dtype == np.uint8
    print(subject.name, image.shape, image.dtype, label.dtype)
PY
```

If the smoke test passes and there is enough free disk, run the full conversion in `tmux`. Start conservatively; `--workers 8` is a reasonable first setting on a 16-vCPU VM:

```bash
tmux new -s convert
cd "$HOME/patch_icl"
python scripts/convert_to_npy.py \
    --data /mnt/data/totalseg --workers 8
# detach: Ctrl-B D; reattach: tmux attach -t convert
```

Upload only after checking that the conversion ended with `err=0`. This saves the converted `ct.npy` and `label.npy` files in the bucket while excluding the raw `.nii.gz` files. `gcloud storage rsync` is preferred over `gsutil`; it can be rerun after an interruption:

```bash
gcloud storage buckets create "gs://$BUCKET" \
    --project="$PROJECT" --location=us-central1  # once
gcloud storage rsync --recursive --exclude='.*\.nii\.gz$' \
    /mnt/data/totalseg "gs://$BUCKET/data/totalseg"
gcloud storage du --summarize "gs://$BUCKET/totalseg"
```

On a new VM, restore the preprocessed dataset from the bucket with:

```bash
mkdir -p /mnt/data/totalseg
gcloud storage rsync --recursive \
    "gs://$BUCKET/totalseg" /mnt/data/totalseg
```

Delete the CPU VM after verifying the bucket. Keep or delete the data disk deliberately: the disk continues to incur storage charges even when the VM is stopped.

```bash
gcloud compute instances delete "$CPU_VM" --project="$PROJECT" --zone="$ZONE"
# Delete this only if the bucket is the sole copy:
# gcloud compute disks delete "$DATA_DISK" --project="$PROJECT" --zone="$ZONE"
```

## 1. Create a TPU VM

**Use `gcloud compute tpus tpu-vm create`, not `gcloud compute instances create`** — the latter provisions a plain VM with no TPU hardware attached.

```bash
export PROJECT=atomic-acrobat-308517
export ZONE=us-central2-b       # us-central2-b for v4, us-west4-a for v5e
export TPU=patch-icl-tpu

gcloud compute tpus tpu-vm create $TPU \
    --project=$PROJECT \
    --zone=$ZONE \
    --accelerator-type=v4-8 \
    --version=tpu-ubuntu2204-base \
    --preemptible

gcloud compute tpus tpu-vm ssh $TPU --project=$PROJECT --zone=$ZONE
```

> Apply for free TPU quota via the [TPU Research Cloud](https://sites.research.google/trc/about/) before paying on-demand.

**Verify the TPU is real before doing anything else:**

```bash
ls /dev/accel*
```

If this returns `No such file or directory`, the VM has no TPU hardware — delete it and retry. A working TPU VM will show devices like `/dev/accel0`.

## 2. Install dependencies

```bash
sudo apt-get update && sudo apt-get install -y git

pip install "torch_xla[tpu]==2.6.0" \
    -f https://storage.googleapis.com/libtpu-releases/index.html

pip install hydra-core nibabel tqdm matplotlib

# wandb may need a clean reinstall if import errors occur
pip install --force-reinstall wandb protobuf
```

> `torch` 2.6.0 is pre-installed system-wide on TPU VMs — no need to reinstall it.

**Verify torch_xla can see the TPU:**

```bash
python -c "import torch_xla.core.xla_model as xm; print(xm.xla_device())"
```

This should print `xla:0`. If it fails with `Failed to get global TPU topology`, the libtpu version is mismatched — see Troubleshooting below.

## 3. Clone the repo

```bash
git clone https://github.com/tidiane-camaret/patch_icl
cd patch_icl
```

## 4. Get the dataset

**Download and extract (first time):**

```bash
mkdir -p $HOME/data/totalseg
cd $HOME/data

wget -O totalseg.zip "https://zenodo.org/records/10047292/files/Totalsegmentator_dataset_v201.zip?download=1"
unzip -q totalseg.zip -d totalseg
rm totalseg.zip

cd $HOME/patch_icl
```

**Preprocess to `.npy` (runs once, in-place):**

```bash
python scripts/convert_to_npy.py --data $HOME/data/totalseg --workers 8
```

If the disk fills up mid-conversion, free space by deleting raw files for already-converted subjects, then re-run:

```bash
for d in $HOME/data/totalseg/s*/; do
    if [ -f "${d}ct.npy" ] && [ -f "${d}label.npy" ]; then
        rm -f "${d}ct.nii.gz"
        rm -rf "${d}segmentations/"
    fi
done
python scripts/convert_to_npy.py --data $HOME/data/totalseg --workers 8
```

**Optional — cache on GCS to avoid re-downloading on future VMs:**

```bash
# Upload only .npy files (skip raw .nii.gz to save space)
gsutil -m rsync -r -x '.*\\.nii\\.gz$' $HOME/data/totalseg gs://atomic-acrobat-totalseg/totalseg

# On a new VM, restore with:
mkdir -p $HOME/data/totalseg
gsutil -m rsync -r gs://atomic-acrobat-totalseg/totalseg $HOME/data/totalseg
```

## 5. Configure W&B

```bash
wandb login   # paste your API key
```

## 6. Run training

**Always run inside `tmux`** — the first run builds a scan cache (~1228 subjects) which takes several minutes. Without `tmux`, an SSH broken pipe will kill the process before training starts.

```bash
tmux new -s train
# inside tmux:
PJRT_DEVICE=TPU python scripts/train_vit_in_context.py train.tpu=true train.workers=4 paths.totalseg=$HOME/data/totalseg
# detach with Ctrl-B D, reattach later with: tmux attach -t train
```

Run all overrides on a single line — multiline pastes can break in some shells, and Hydra does not expand `~` (use `$HOME` instead).

Common overrides:

```bash
PJRT_DEVICE=TPU python scripts/train_vit_in_context.py train.tpu=true train.workers=4 train.epochs=100 train.batch_size=4 train.run_name=my-run paths.totalseg=$HOME/data/totalseg
```

Resume from a checkpoint:

```bash
PJRT_DEVICE=TPU python scripts/train_vit_in_context.py train.tpu=true train.checkpoint=results/vit_incontext_best.pt paths.totalseg=$HOME/data/totalseg
```

## 7. Delete the VM when done

Run this from your **local machine or Cloud Shell**, not from inside the VM:

```bash
gcloud compute tpus tpu-vm delete $TPU --zone=$ZONE --project=$PROJECT
```

To list all VMs if you've lost track of the name/zone:

```bash
gcloud compute tpus tpu-vm list --project=$PROJECT --zone=$ZONE
gcloud compute instances list --project=$PROJECT   # also catches plain VMs
```

---

## Troubleshooting

**`Failed to get global TPU topology`**
The libtpu version doesn't match torch_xla. Uninstall both libtpu packages and let torch_xla reinstall the correct one:
```bash
pip uninstall -y libtpu libtpu-nightly
pip install "torch_xla[tpu]==2.6.0" -f https://storage.googleapis.com/libtpu-releases/index.html
```

**`/dev/accel*` not found**
The VM has no TPU hardware. This happens when using `gcloud compute instances create` instead of `gcloud compute tpus tpu-vm create`, or when the zone lacks quota. Delete the VM and recreate it with the correct command and zone.

**`No space left on device` during preprocessing**
See the disk-freeing step in section 4.

**`RESOURCE_EXHAUSTED: Ran out of memory in memory space hbm`**
Stage-1 self-attention runs on a batch of `B + B×context_size` sequences, so the 512² attention intermediates scale with batch size. With the default `batch_size=8` and `context_size=3` the model requires ~16.03GB, just over the v4-8 limit of 15.75GB. Use `train.batch_size=4` (or reduce `model.embed_dim=128` for a larger model-size reduction).

**`Broken pipe` / SSH disconnect during startup**
The scan cache build on first run takes several minutes of near-idle SSH output, which causes most SSH clients to drop the connection. Always launch training inside `tmux` (see section 6). If you get disconnected mid-run, reconnect with `gcloud compute tpus tpu-vm ssh $TPU ...` then `tmux attach -t train`.

**`LexerNoViableAltException` from Hydra**
Hydra does not expand `~` in override values. Use `$HOME` instead of `~`, and pass all overrides on a single line.

**Permission denied when deleting VM from inside the instance**
Run the delete command from your local machine or Cloud Shell, not from the SSH session.

---

## Notes

- **bfloat16**: TPU training uses bfloat16 automatically (no GradScaler needed). GPU runs use float16.
- **Preemptible VMs**: add `--preemptible` to the create command for ~3× cheaper spot pricing. Save checkpoints to GCS frequently.
- **DataLoader workers**: keep `train.workers` at 4 or below on TPU VMs (tighter shared memory limits than GPU VMs).
- **Scan cache**: on the first run the dataloader scans all subjects and writes `.scan_cache_*.pkl` to the data root. Subsequent runs load instantly from cache.
