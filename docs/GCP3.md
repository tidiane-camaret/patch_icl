export PROJECT=atomic-acrobat-308517
export ZONE=us-central1-b
export CPU_VM=patch-icl-prep
export DATA_DISK=patch-icl-prep-data
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

export GPU_VM=patch-icl-h100

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
  --disk=name=patch-icl-prep-data,mode=rw,boot=no \
  --scopes=cloud-platform

# NVIDIA A100

gcloud compute instances create "$GPU_VM" \
  --project=$PROJECT --zone=us-central1-b --machine-type=a2-highgpu-1g \
  --provisioning-model=SPOT --instance-termination-action=STOP --maintenance-policy=TERMINATE \
  --image-family=common-cu129-ubuntu-2204-nvidia-580 --image-project=deeplearning-platform-release \
  --boot-disk-size=100GB --boot-disk-type=pd-balanced \
  --disk=name=patch-icl-prep-data,mode=rw,boot=no --scopes=cloud-platform

# NVIDIA H100

gcloud compute instances create "$GPU_VM" \
  --project=$PROJECT --zone=us-central1-b \
  --machine-type=a3-highgpu-1g \
  --provisioning-model=SPOT --instance-termination-action=STOP \
  --maintenance-policy=TERMINATE \
  --image-family=common-cu129-ubuntu-2204-nvidia-580 --image-project=deeplearning-platform-release \
  --boot-disk-size=200GB --boot-disk-type=pd-balanced \
  --disk=name=patch-icl-prep-data,mode=rw,boot=no --scopes=cloud-platform

gcloud compute ssh "$GPU_VM" --project="$PROJECT" --zone="$ZONE"

sudo apt-get update
sudo apt-get install -y git unzip wget tmux build-essential
sudo mkdir -p /mnt/data
sudo mount /dev/sdb /mnt/data # 

# if disk has another name, search it : 
# lsblk
# ls -l /dev/disk/by-id/google-*
# then : 
# sudo mount -o discard,defaults /dev/disk/by-id/google-persistent-disk-1 /mnt/data

sudo chown -R $USER:$USER /mnt/data/
df -h /mnt/data

curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.local/bin/env
git clone https://github.com/tidiane-camaret/patch_icl && cd patch_icl && git checkout feat/incontext-dataloader-v2
uv sync --extra cu124        # or --extra cu128

tmux
uv run python experiments/3d/train.py cluster=gcp experiment=80_varspacing_hard_tgt_prior train.batch_size=2 train.epochs=5 data.max_train_subjects=50



#delete instance 
gcloud compute instances delete "$GPU_VM" --project="$PROJECT" --zone="$ZONE"
gcloud compute instances delete "$CPU_VM" --project="$PROJECT" --zone="$ZONE"