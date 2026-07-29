import torch

p = "/nfs/data/nii/data1/Analysis/camaret___in_context_segmentation/ANALYSIS_20251122/results/patch_icl/3d_train/2026-07-29_dark-capybara-204/best.pt"
ck = torch.load(p, map_location="cpu", weights_only=False)
sd = ck["model"]
bad = [k for k, v in sd.items()
       if torch.is_floating_point(v) and (torch.isnan(v).any() or torch.isinf(v).any())]
mx = max(v.abs().max().item() for v in sd.values() if torch.is_floating_point(v))
print("tensors:", len(sd), "| NaN/Inf tensors:", len(bad), bad[:5])
print("max |weight|:", mx)
print("cuda:", torch.cuda.is_available(),
      torch.cuda.device_count() if torch.cuda.is_available() else "")
