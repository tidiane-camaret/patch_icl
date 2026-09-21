# Method
## Image-label interaction
- Medverse : concat image and label before u-net
- Iris : fuse images features and labels using pixel shuffle-unshuffle
- Ours : bi-axial attention (patch,img<->label)

## Coarse-to-fine training
### Query image prior
- Medverse : upsampled perturbed GT as extra U-Net branch
- Iris : None
- Ours : upsampled prev. level pred as label token initialisation

### Fine level processing
- Medverse : sliding window over whole volume
- Iris : sliding window over whole volume
- Ours : sliding window over prev. level predicted zones

### Cross-level training
- Medverse : None
- Iris : None
- Ours : gradient flow across level (register tokens)

# Dataset 
Totalsegmentator

# Results
## Fixe spacing
- datasets : Totalseg CT, 3mm spacing crops
- accuracy (dice,nsd) and compute (time, flops, vram)
- show strenghts (e.g. small objects)
## Coarse-to-fine
- datasets : Totalseg CT
- show effect of query prior and cross-level training
- show acc/compute tradeoff when restricting sliding window to prev.level predictions

## Generalization
- other CT datasets
- other modalities (MRI, etc)
- Far OOD tasks
