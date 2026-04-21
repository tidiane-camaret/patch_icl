# Running training on GCP TPU

## 1. Create a TPU VM

```bash
export PROJECT=your-project-id
export ZONE=us-west4-a
export TPU=patch-icl-tpu

gcloud compute tpus tpu-vm create $TPU \
    --project=$PROJECT \
    --zone=$ZONE \
    --accelerator-type=v5litepod-8 \
    --version=v2-alpha-tpuv5-lite \
    --preemptible

gcloud compute tpus tpu-vm ssh $TPU --project=$PROJECT --zone=$ZONE
```

> Apply for free TPU quota via the [TPU Research Cloud](https://sites.research.google/trc/about/) before paying on-demand ($9.60/hr for v5litepod-8).

## 2. Install dependencies

```bash
sudo apt-get update && sudo apt-get install -y git

pip install torch==2.6.0 "torch_xla[tpu]==2.6.0" \
    -f https://storage.googleapis.com/libtpu-releases/index.html

pip install wandb hydra-core nibabel tqdm matplotlib totalsegmentator
```

## 3. Clone the repo

```bash
git clone https://github.com/tidiane-camaret/patch_icl
cd patch_icl
```

## 4. Get the dataset

The TotalSegmentator dataset (~450 GB raw) needs to be downloaded once. Store the preprocessed files on a GCS bucket to avoid re-downloading on each VM.

**First time — download and preprocess:**

```bash
mkdir -p /data/totalseg
totalseg_download_dataset -o /data/totalseg

# Convert .nii.gz to .npy (runs once, in-place)
python scripts/convert_to_npy.py --data /data/totalseg --workers 8

# Upload preprocessed files to GCS (skip raw .nii.gz to save space)
gsutil -m rsync -r -x '.*\.nii\.gz$' /data/totalseg gs://your-bucket/totalseg
```

**Subsequent VMs — sync from GCS:**

```bash
mkdir -p /data/totalseg
gsutil -m rsync -r gs://your-bucket/totalseg /data/totalseg
```

## 5. Configure W&B

```bash
wandb login   # paste your API key
```

## 6. Run training

```bash
PJRT_DEVICE=TPU python scripts/train_vit_in_context.py \
    train.tpu=true \
    train.workers=4 \
    paths.totalseg=/data/totalseg
```

Any config value can be overridden on the command line:

```bash
PJRT_DEVICE=TPU python scripts/train_vit_in_context.py \
    train.tpu=true \
    train.workers=4 \
    train.epochs=100 \
    train.batch_size=16 \
    train.run_name=my-run \
    paths.totalseg=/data/totalseg
```

To resume from a checkpoint:

```bash
PJRT_DEVICE=TPU python scripts/train_vit_in_context.py \
    train.tpu=true \
    train.checkpoint=results/vit_incontext_best.pt \
    paths.totalseg=/data/totalseg
```

## 7. Delete the VM when done

```bash
gcloud compute tpus tpu-vm delete $TPU --zone=$ZONE --project=$PROJECT
```

---

## Notes

- **bfloat16**: TPU training uses bfloat16 automatically (no GradScaler needed). GPU runs use float16.
- **Preemptible VMs**: add `--preemptible` to the create command for ~3× cheaper spot pricing. Save checkpoints to GCS frequently.
- **DataLoader workers**: keep `train.workers` at 4 or below on TPU VMs (tighter shared memory limits than GPU VMs).
- **Scan cache**: on the first run the dataloader scans all subjects and writes `.scan_cache_*.pkl` to the data root. Subsequent runs load instantly from cache.
