import numpy as np
import nibabel as nib
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoImageProcessor

# Load the model
model = AutoModel.from_pretrained('fomofo/tap-ct-b-3d', trust_remote_code=True)
preprocessor = AutoImageProcessor.from_pretrained('fomofo/tap-ct-b-3d', trust_remote_code=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


# --- Memory fix: the blocks use `MemEffAttention`, which needs xformers for
# O(L) memory. When xformers is absent it falls back to explicit q@k.T softmax
# (O(L^2)), which OOMs on large volumes: a full forward on raw (179,192,294)
# resizes in-plane to 224x224 and pads depth to 180 -> 35285 tokens, whose
# per-layer attention matrix alone is ~60 GB in fp32. Swap in PyTorch SDPA
# (flash kernel) so a whole-volume forward stays O(L) memory (<2 GB at 35k tok,
# <4 GB at 100k tok on a 48 GB card). See tap_ct_bench.py for the full sweep.
def _sdpa_forward(self, x, attn_bias=None):
    # SDPA applies the 1/sqrt(head_dim) scale internally (== self.scale), so q
    # is NOT pre-scaled here.
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
    q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)  # each (B, heads, N, head_dim)
    x = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, N, C)
    return self.proj_drop(self.proj(x))


for _m in model.modules():
    if _m.__class__.__name__ in ('MemEffAttention', 'Attention'):
        type(_m).forward = _sdpa_forward
        break
# Load image & set orientation to LPS
file_path = '/nfs/data/nii/data1/Analysis/camaret___in_context_segmentation/ANALYSIS_20251122/data/totalseg/s0000/ct.nii.gz'

# 1. Read with nibabel (lenient header parsing)
nib_img = nib.load(file_path)

# 2. Extract the raw numpy array 
# Note: nibabel loads as (X, Y, Z), SimpleITK expects (Z, Y, X), so we transpose it
data = nib_img.get_fdata().T

print(f"Loaded image shape (after transpose): {data.shape}")

# 3. Create the SimpleITK Image
volume = sitk.GetImageFromArray(data)

# 4. Safely transfer spacing and origin (skipping the broken direction matrix)
zooms = nib_img.header.get_zooms()
volume.SetSpacing((float(zooms[0]), float(zooms[1]), float(zooms[2])))

affine = nib_img.affine
volume.SetOrigin((float(affine[0,3]), float(affine[1,3]), float(affine[2,3])))

volume = sitk.DICOMOrient(volume, 'LPS')

# Get array, expand to (B, C, D, H, W) and preprocess
array = sitk.GetArrayFromImage(volume)
array = np.expand_dims(array, axis=(0, 1))
x = preprocessor(array)['pixel_values'].to(device)
# Move to device


print(f"Preprocessed input shape: {x.shape}")
print(f"Preprocessed input value range: min={x.min().item()}, max={x.max().item()})")
# Forward pass
with torch.no_grad():
    output = model.forward(x)

print(f"Output shape: {output.last_hidden_state.shape}")
# OR

# Forward pass with sliding window
from monai.inferers import SlidingWindowInferer

def predictor_fn(x):
    # Reshape the patch tokens to resemble a 3D feature map
    out = model(x, reshape=True)
    return out.last_hidden_state

inferer = SlidingWindowInferer(
    roi_size=[12, 224, 224],
    sw_batch_size=1,
    overlap=0.75,
    mode='gaussian'
)

with torch.no_grad():
    output = inferer(x, predictor_fn)

print(f"Sliding window output shape: {output.shape}")
