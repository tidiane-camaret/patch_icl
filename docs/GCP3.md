export PROJECT=atomic-acrobat-308517
export ZONE=europe-north1-c   # Finland preferred (cheapest H100 ~$1.09/hr); fallback asia-northeast1-c (Tokyo ~$2.58/hr)
export CPU_VM=patch-icl-prep
export GPU_VM=patch-icl-h100
export DATA_DISK=patch-icl-tokyo-data  # permanent 200GB disk in asia-northeast1-c; recreate in target zone if switching
export BUCKET=atomic-acrobat-totalseg


#### DATA PREPARATION ####

# create CPU instance with 32 cores
gcloud compute instances create "$CPU_VM" \
    --project="$PROJECT" \
    --zone="$ZONE" \
    --machine-type=n2d-standard-32 \
    --boot-disk-size=50GB \
    --boot-disk-type=pd-balanced \
    --image-family=ubuntu-2404-lts-amd64 \
    --image-project=ubuntu-os-cloud \
    --scopes=cloud-platform

# attach permanent disk
gcloud compute instances attach-disk "$CPU_VM" \
    --project="$PROJECT" --zone="$ZONE" \
    --disk="$DATA_DISK"

# give bucket writing rights to VM
gcloud compute instances set-service-account "$CPU_VM" \
    --zone=us-central1-b \
    --scopes=cloud-platform

# ssh to vm and mount permanent disk

gcloud compute ssh "$CPU_VM" --project="$PROJECT" --zone="$ZONE"

sudo apt-get update
sudo apt-get install -y git unzip wget tmux

sudo mkdir -p /mnt/data
sudo mount /dev/sdb /mnt/data
sudo chown -R $USER:$USER /mnt/data/
df -h /mnt/data

# preprocess data
tmux
uv run python scripts/convert_to_npy.py --data /mnt/data/totalseg --store-raw --workers 24

# delete VM
gcloud compute instances delete "$CPU_VM" --project="$PROJECT" --zone="$ZONE"


#### TRAINING ####

# Spot price comparison for a3-highgpu-1g (H100 80GB, 26 vCPUs, 234 GiB RAM):
#   europe-north1-c (Finland):  ~$1.09/hr  ← cheapest (try first, often sold out)
#   asia-northeast1-c (Tokyo):  ~$2.58/hr  ← fallback (currently running)
#   us-east5-a (Columbus):      ~$5.55/hr
#   us-central1-b:              ~$6.26/hr
#
# DATA_DISK must exist in the same zone as the instance.
# Create a new permanent disk if switching zones:
#   gcloud compute disks create patch-icl-<zone>-data --zone=$ZONE --size=200GB --type=pd-balanced
# Then populate from GCS (first run only — rsync is incremental after that).

# NVIDIA T4
gcloud compute instances create "$GPU_VM" \
  --project=$PROJECT --zone=$ZONE \
  --machine-type=n1-standard-8 \
  --accelerator=type=nvidia-tesla-t4,count=1 \
  --provisioning-model=SPOT --instance-termination-action=STOP \
  --maintenance-policy=TERMINATE \
  --image-family=common-cu129-ubuntu-2204-nvidia-580 \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=75GB --boot-disk-type=pd-balanced \
  --disk=name=$DATA_DISK,mode=rw,boot=no \
  --scopes=cloud-platform

# NVIDIA A100
gcloud compute instances create "$GPU_VM" \
  --project=$PROJECT --zone=$ZONE --machine-type=a2-highgpu-1g \
  --provisioning-model=SPOT --instance-termination-action=STOP --maintenance-policy=TERMINATE \
  --image-family=common-cu129-ubuntu-2204-nvidia-580 --image-project=deeplearning-platform-release \
  --boot-disk-size=100GB --boot-disk-type=pd-balanced \
  --disk=name=$DATA_DISK,mode=rw,boot=no --scopes=cloud-platform

# NVIDIA H100
gcloud compute instances create "$GPU_VM" \
  --project=$PROJECT --zone=$ZONE \
  --machine-type=a3-highgpu-1g \
  --provisioning-model=SPOT --instance-termination-action=STOP \
  --maintenance-policy=TERMINATE \
  --image-family=common-cu129-ubuntu-2204-nvidia-580 --image-project=deeplearning-platform-release \
  --boot-disk-size=75GB --boot-disk-type=pd-balanced \
  --disk=name=$DATA_DISK,mode=rw,boot=no --scopes=cloud-platform

gcloud compute ssh "$GPU_VM" --project="$PROJECT" --zone="$ZONE"

sudo apt-get update
sudo apt-get install -y git unzip wget tmux build-essential
sudo mkdir -p /mnt/data

# mount permanent disk (data disk is google-persistent-disk-3 on a3-highgpu; use lsblk to confirm)
sudo mount -o discard,defaults /dev/disk/by-id/google-persistent-disk-3 /mnt/data
# auto-mount on VM restart:
echo '/dev/disk/by-id/google-persistent-disk-3 /mnt/data ext4 discard,defaults,nofail 0 2' | sudo tee -a /etc/fstab

# if disk is new (unformatted), format first:
# sudo mkfs.ext4 -m 0 -E lazy_itable_init=0,lazy_journal_init=0,discard /dev/disk/by-id/google-persistent-disk-3

sudo chown -R $USER:$USER /mnt/data/
df -h /mnt/data


# rsync bucket data -> disk (first run or after re-creating disk in a new zone)
cd /mnt/data
mkdir -p results totalseg totalsegmri
gcloud storage rsync gs://atomic-acrobat-totalseg/results results --recursive
gcloud storage rsync gs://atomic-acrobat-totalseg/data/totalseg totalseg --recursive
gcloud storage rsync gs://atomic-acrobat-totalseg/data/totalsegmri totalsegmri --recursive

curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.local/bin/env
git clone https://github.com/tidiane-camaret/patch_icl && cd patch_icl && git checkout feat/incontext-dataloader-v2
uv sync --extra cu124        # or --extra cu128


# run training 
tmux
export NUMEXPR_MAX_THREADS=26 #h100 has 26 cores

uv run python experiments/3d/train.py cluster=gcp experiment=80_varspacing_hard_tgt_prior train.batch_size=2 train.epochs=5 data.max_train_subjects=50

# rsync disk -> bucket data
gcloud storage rsync results gs://atomic-acrobat-totalseg/results --recursive

# delete instance 
gcloud compute instances delete "$GPU_VM" --project="$PROJECT" --zone="$ZONE"
