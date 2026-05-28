"""
Visualise a few batch items from TotalSegInContextDataset with multiverseg_v3 augs.

Usage
-----
    python scripts/plot_aug_batch.py
    python scripts/plot_aug_batch.py --n 6 --out /tmp/aug_batch.png
    python scripts/plot_aug_batch.py --aug_preset nnunet
    python scripts/plot_aug_batch.py --cluster meta
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.totalseg_dataloader_incontext import TotalSegInContextDataset, incontext_collate_fn
from src.totalseg_dataset import CT_NORM_MIN, CT_NORM_MAX
from data.totalseg_classes import resolve_classes


def _best_slice(vol: np.ndarray, mask: np.ndarray) -> int:
    """Index of the axial slice with the most foreground voxels."""
    counts = (mask > 0).sum(axis=(1, 2))
    return int(counts.argmax()) if counts.max() > 0 else vol.shape[0] // 2


def _to_uint8(img: np.ndarray) -> np.ndarray:
    """Map z-score volume to uint8 [0, 255]."""
    img = (img - CT_NORM_MIN) / (CT_NORM_MAX - CT_NORM_MIN)
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def _overlay(img: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """RGB overlay: image in grey, mask in red."""
    g = _to_uint8(img)
    rgb = np.stack([g, g, g], axis=-1)
    if mask.max() > 0:
        red = np.zeros_like(rgb)
        red[..., 0] = 220
        fg = (mask > 0)[..., None]
        rgb = (rgb * (1 - alpha * fg) + red * alpha * fg).astype(np.uint8)
    return rgb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n",          type=int,  default=4,   help="number of samples to plot")
    ap.add_argument("--aug_preset", default="multiverseg_v3")
    ap.add_argument("--cluster",    default="nfs")
    ap.add_argument("--out",        default=str(ROOT / "results" /  "aug_benchmark" / "aug_samples.png"))
    args = ap.parse_args()

    base   = OmegaConf.load(ROOT / "configs" / "config.yaml")
    cl_cfg = OmegaConf.load(ROOT / "configs" / "cluster" / f"{args.cluster}.yaml")
    cfg    = OmegaConf.merge(base, cl_cfg)

    aug_yaml = ROOT / "configs" / "augmentations" / f"{args.aug_preset}.yaml"
    aug_cfg  = OmegaConf.load(aug_yaml).augmentations

    data_root = cfg.paths.totalseg
    classes   = resolve_classes("benchmark", data_root, is_mri=False)[:20]  # first 20 benchmark classes

    ds = TotalSegInContextDataset(
        root=data_root,
        classes=classes,
        image_size=(128, 128, 128),
        split="train",
        context_size=1,
        class_balanced=True,
        aug_cfg=aug_cfg,
        use_crop=True,
        p_synth=0.0,   # real labels only so aug effect is clear
    )

    loader = DataLoader(ds, batch_size=args.n, shuffle=True, num_workers=2,
                        collate_fn=incontext_collate_fn)
    batch  = next(iter(loader))

    B  = batch["image"].shape[0]
    K  = batch["context_in"].shape[1]
    # columns: [tgt_img | tgt_overlay | ctx0_img | ctx0_overlay | ctx1_img | ctx1_overlay | ...]
    ncols = 2 + 2 * K
    fig, axes = plt.subplots(B, ncols, figsize=(3 * ncols, 3 * B))
    if B == 1:
        axes = axes[None]

    col_titles = ["target", "target+mask"] + [f"ctx{k}" for k in range(K) for _ in range(2)]

    for b in range(B):
        img  = batch["image"][b, 0].numpy()        # (D, H, W)
        lbl  = batch["label"][b].numpy()           # (D, H, W)
        z    = _best_slice(img, lbl)

        row_data = [
            _to_uint8(img[z]),
            _overlay(img[z], lbl[z]),
        ]
        for k in range(K):
            ctx_img = batch["context_in"][b, k, 0].numpy()
            ctx_lbl = batch["context_out"][b, k].numpy()
            zk = _best_slice(ctx_img, ctx_lbl)
            row_data.append(_to_uint8(ctx_img[zk]))
            row_data.append(_overlay(ctx_img[zk], ctx_lbl[zk]))

        cls_name = batch["label_names"][b]
        for col, (ax, data) in enumerate(zip(axes[b], row_data)):
            ax.imshow(data, cmap="gray" if data.ndim == 2 else None)
            ax.set_xticks([]); ax.set_yticks([])
            if b == 0:
                ax.set_title(col_titles[col], fontsize=8)
            if col == 0:
                ax.set_ylabel(cls_name, fontsize=7, rotation=0, labelpad=50, va="center")

    fig.suptitle(f"aug_preset={args.aug_preset}  |  {B} samples  (real labels, p_synth=0)",
                 fontsize=9)
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
