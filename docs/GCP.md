# Running training on GCP TPU

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
gsutil -m rsync -r -x '.*\.nii\.gz$' $HOME/data/totalseg gs://atomic-acrobat-totalseg/totalseg

# On a new VM, restore with:
mkdir -p $HOME/data/totalseg
gsutil -m rsync -r gs://atomic-acrobat-totalseg/totalseg $HOME/data/totalseg
```

## 5. Configure W&B

```bash
wandb login   # paste your API key
```

## 6. Run training

Run all overrides on a single line — multiline pastes can break in some shells, and Hydra does not expand `~` (use `$HOME` instead):

```bash
PJRT_DEVICE=TPU python scripts/train_vit_in_context.py train.tpu=true train.workers=4 paths.totalseg=$HOME/data/totalseg
```

Common overrides:

```bash
PJRT_DEVICE=TPU python scripts/train_vit_in_context.py train.tpu=true train.workers=4 train.epochs=100 train.batch_size=16 train.run_name=my-run paths.totalseg=$HOME/data/totalseg
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
