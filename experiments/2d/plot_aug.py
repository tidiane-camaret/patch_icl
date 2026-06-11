"""Quick script to visualise augmented training samples → results/datasets/medsegbench_aug.png"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

_ROOT = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, _ROOT)
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.datasets.medsegbench import MedSegBenchDataset
from common import DEVICE, TaggedDataset, collate
from torch.utils.data import DataLoader

# ── inline aug config (mirrors medsegbench.yaml) ──────────────────────────────
from omegaconf import OmegaConf
aug_cfg = OmegaConf.create({
    "enabled": True,
    "geometric": {
        "hflip_p": 0.5,
        "vflip_p": 0.5,
        "rotate": {"p": 0.5, "max_angle_deg": 20.0},
    },
    "intensity": {
        "brightness": {"p": 0.5, "max_delta": 0.15},
        "contrast":   {"p": 0.5, "range": [0.8, 1.2]},
        "gamma":      {"p": 0.3, "range": [0.75, 1.33]},
        "noise":      {"p": 0.3, "std": 0.04},
    },
})

import torch.nn.functional as F

def augment(images, masks, K, cfg):
    B, T, _, H, W = images.shape
    dev = images.device
    BK  = B * K
    c_imgs = images[:, :K].reshape(BK, 1, H, W)
    c_msks = masks[:, :K].reshape(BK, 1, H, W)

    g = cfg.geometric
    if g.hflip_p > 0:
        m = torch.rand(BK, 1, 1, 1, device=dev) < g.hflip_p
        c_imgs = torch.where(m, c_imgs.flip(-1), c_imgs)
        c_msks = torch.where(m, c_msks.flip(-1), c_msks)
    if g.vflip_p > 0:
        m = torch.rand(BK, 1, 1, 1, device=dev) < g.vflip_p
        c_imgs = torch.where(m, c_imgs.flip(-2), c_imgs)
        c_msks = torch.where(m, c_msks.flip(-2), c_msks)
    if g.rotate.p > 0:
        active = torch.rand(BK, device=dev) < g.rotate.p
        angles = (torch.rand(BK, device=dev) * 2 - 1) * g.rotate.max_angle_deg * active.float()
        rad = torch.deg2rad(angles)
        cos_t, sin_t = torch.cos(rad), torch.sin(rad)
        z = torch.zeros_like(cos_t)
        theta = torch.stack([cos_t, -sin_t, z, sin_t, cos_t, z], dim=1).reshape(BK, 2, 3)
        grid  = F.affine_grid(theta, (BK, 1, H, W), align_corners=False)
        c_imgs = F.grid_sample(c_imgs, grid, mode="bilinear", align_corners=False, padding_mode="zeros")
        c_msks = F.grid_sample(c_msks, grid, mode="nearest",  align_corners=False, padding_mode="zeros")

    images = torch.cat([c_imgs.reshape(B, K, 1, H, W), images[:, K:]], dim=1)
    masks  = torch.cat([c_msks.reshape(B, K, 1, H, W), masks[:, K:]],  dim=1)

    BT = B * T
    imgs = images.reshape(BT, 1, H, W)
    ic = cfg.intensity
    if ic.brightness.p > 0:
        m = torch.rand(BT, 1, 1, 1, device=dev) < ic.brightness.p
        d = (torch.rand(BT, 1, 1, 1, device=dev) * 2 - 1) * ic.brightness.max_delta
        imgs = torch.where(m, (imgs + d).clamp(0, 1), imgs)
    if ic.contrast.p > 0:
        lo, hi = ic.contrast.range
        m  = torch.rand(BT, 1, 1, 1, device=dev) < ic.contrast.p
        s  = torch.rand(BT, 1, 1, 1, device=dev) * (hi - lo) + lo
        mu = imgs.mean(dim=(-2, -1), keepdim=True)
        imgs = torch.where(m, ((imgs - mu) * s + mu).clamp(0, 1), imgs)
    if ic.gamma.p > 0:
        lo, hi = ic.gamma.range
        m = torch.rand(BT, 1, 1, 1, device=dev) < ic.gamma.p
        g = torch.rand(BT, 1, 1, 1, device=dev) * (hi - lo) + lo
        imgs = torch.where(m, imgs.clamp(1e-6).pow(g), imgs)
    if ic.noise.p > 0:
        m = torch.rand(BT, 1, 1, 1, device=dev) < ic.noise.p
        n = torch.randn_like(imgs) * ic.noise.std
        imgs = torch.where(m, (imgs + n).clamp(0, 1), imgs)

    return imgs.reshape(B, T, 1, H, W), masks


# ── load one small batch ───────────────────────────────────────────────────────
ds = MedSegBenchDataset(split="train", context_size=1, image_size=128,
                        datasets=["isic2018", "kvasir", "promise12", "bkai-igh",
                                  "drive", "m2caiseg"])
loader = DataLoader(TaggedDataset(ds), batch_size=8, shuffle=True,
                    num_workers=0, collate_fn=collate)
batch = next(iter(loader))

K = 1
ctx_in  = batch["context_in"]   # (B, K, 1, H, W)
ctx_out = batch["context_out"]
img     = batch["image"]         # (B, 1, H, W)

all_images_orig = torch.cat([ctx_in, img.unsqueeze(1)], dim=1)
all_masks_orig  = torch.cat([ctx_out, torch.zeros_like(img.unsqueeze(1))], dim=1)

torch.manual_seed(0)
all_images_aug, all_masks_aug = augment(
    all_images_orig.to(DEVICE), all_masks_orig.to(DEVICE), K, aug_cfg
)
all_images_aug = all_images_aug.cpu()
all_masks_aug  = all_masks_aug.cpu()

# ── plot ──────────────────────────────────────────────────────────────────────
# Columns: ctx_img_orig | ctx_mask_orig | ctx_img_aug | ctx_mask_aug | query_orig | query_aug
N = min(8, len(batch["dataset"]))
cols = 6
fig, axes = plt.subplots(N, cols, figsize=(cols * 2.2, N * 2.2))

col_titles = ["ctx img", "ctx mask", "ctx img (aug)", "ctx mask (aug)", "query img", "query img (aug)"]
for c, title in enumerate(col_titles):
    axes[0, c].set_title(title, fontsize=9)

for i in range(N):
    ds_name = batch["dataset"][i]
    row_data = [
        all_images_orig[i, 0, 0].numpy(),
        all_masks_orig[i, 0, 0].numpy(),
        all_images_aug[i, 0, 0].numpy(),
        all_masks_aug[i, 0, 0].numpy(),
        all_images_orig[i, K, 0].numpy(),
        all_images_aug[i, K, 0].numpy(),
    ]
    cmaps = ["gray", "gray", "gray", "gray", "gray", "gray"]
    for c, (data, cmap) in enumerate(zip(row_data, cmaps)):
        ax = axes[i, c]
        vmax = 1.0 if c in (1, 3) else None
        ax.imshow(data, cmap=cmap, vmin=0, vmax=vmax)
        ax.axis("off")
    axes[i, 0].set_ylabel(ds_name, fontsize=7, rotation=0, labelpad=55, va="center")

plt.suptitle("MedSegBench augmentation preview (K=1 context)", fontsize=11, y=1.01)
plt.tight_layout()
out = Path(_ROOT) / "results/datasets/medsegbench_aug.png"
out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out, dpi=120, bbox_inches="tight")
print(f"Saved → {out}")
