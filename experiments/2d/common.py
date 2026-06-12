"""
Shared utilities for 2D MedSegBench evaluation scripts.
"""

import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, "/home/dpxuser/ic_segmentation")
sys.path.insert(0, "/home/dpxuser/repos/UniverSeg")

from src.datasets.medsegbench import MedSegBenchDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Dataset helpers ───────────────────────────────────────────────────────────

def collate(batch):
    batch = [b for b in batch if b["context_in"].shape[0] > 0]
    if not batch:
        return None
    return {
        "image":       torch.stack([b["image"]       for b in batch]),
        "label":       torch.stack([b["label"]       for b in batch]),
        "context_in":  torch.stack([b["context_in"]  for b in batch]),
        "context_out": torch.stack([b["context_out"] for b in batch]),
        "dataset":     [b["dataset"]     for b in batch],
        "sample_idx":  [b["sample_idx"]  for b in batch],
        "label_value": [b["label_value"] for b in batch],
    }


class TaggedDataset(torch.utils.data.Dataset):
    """Attaches (dataset, sample_idx, label_value) metadata to each item."""
    def __init__(self, inner):
        self.inner = inner

    def __len__(self):
        return len(self.inner)

    def __getitem__(self, idx):
        item = self.inner[idx]
        ds_name, sample_idx, label_value = self.inner.samples[idx]
        item["dataset"]     = ds_name
        item["sample_idx"]  = sample_idx
        item["label_value"] = label_value
        return item


def build_loader(cfg) -> DataLoader:
    """Build a tagged, collated DataLoader from a Hydra eval config."""
    datasets = [cfg.data.dataset] if cfg.data.dataset else None
    ds = MedSegBenchDataset(
        split=cfg.data.split,
        context_size=cfg.data.context_size,
        image_size=cfg.data.image_size,
        datasets=datasets,
    )
    max_per_label = cfg.eval.get("max_per_label", None)
    if max_per_label:
        groups: dict[tuple, list[int]] = {}
        for i, (ds_name, _, lv) in enumerate(ds.samples):
            key = (ds_name, lv)
            groups.setdefault(key, []).append(i)
        keep: list[int] = []
        for indices in groups.values():
            keep.extend(random.sample(indices, min(max_per_label, len(indices))))
        ds.samples = [ds.samples[i] for i in sorted(keep)]
        print(f"Subsampled to {len(ds.samples)} samples (max {max_per_label} per dataset/label)")
    return DataLoader(
        TaggedDataset(ds),
        batch_size=cfg.eval.batch_size,
        shuffle=False,
        num_workers=cfg.eval.workers,
        collate_fn=collate,
        pin_memory=DEVICE.type == "cuda",
        persistent_workers=cfg.eval.workers > 0,
    )


# ── Metrics ───────────────────────────────────────────────────────────────────

def hard_dice(pred: torch.Tensor, gt: torch.Tensor, threshold: float = 0.5) -> float:
    """Threshold pred at threshold, binarize gt at > 0. Returns NaN for empty pred+gt."""
    p = (pred >= threshold).float()
    g = (gt > 0).float()
    num = 2 * (p * g).sum()
    den = p.sum() + g.sum()
    return float(num / den) if den > 1e-6 else float("nan")


def soft_dice(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """Continuous (threshold-free) Dice: 2·Σ(p·g) / (Σp + Σg).

    Both inputs are soft maps in [0, 1] (a probability map and an avg-pooled GT).
    Measures whether the predicted mass lands where the GT mass is — a "shape"
    score that ignores hard thresholds (but still reflects magnitude/calibration).
    Returns NaN when both maps are empty.
    """
    p = pred.float()
    g = gt.float()
    den = p.sum() + g.sum()
    return float(2 * (p * g).sum() / den) if den > 1e-6 else float("nan")


def downsample_mask(mask: torch.Tensor, output_size: int, mode: str = "avg") -> torch.Tensor:
    """mask: (H, W) → (H', W') using avg or max pool."""
    x = mask.float().unsqueeze(0).unsqueeze(0)
    size = (output_size, output_size)
    if mode == "max":
        return F.adaptive_max_pool2d(x, size).squeeze()
    return F.adaptive_avg_pool2d(x, size).squeeze()


# ── Logging ───────────────────────────────────────────────────────────────────

def log_summary(
    per_ds: dict,
    per_label: dict,
    sample_table=None,
    extra: dict | None = None,
    prefix: str = "dice",
    metric_label: str = "native",
) -> dict:
    """Aggregate NaN-filtered Dice scores, print table, return wandb summary dict.

    Keys are emitted under `prefix` (e.g. "dice" → dice/mean, dice/dataset/*,
    dice/class/*), so the same routine can log both native and downsampled metrics.
    """
    summary = {}

    print(f"\n{'Dataset':>25}  {'N':>5}  {f'Dice ({metric_label})':>14}")
    print("-" * 50)
    all_scores = []
    for name in sorted(per_ds):
        scores = [s for s in per_ds[name] if not np.isnan(s)]
        mean   = float(np.mean(scores)) if scores else float("nan")
        all_scores.extend(scores)
        summary[f"{prefix}/dataset/{name}"] = mean
        print(f"{name:>25}  {len(per_ds[name]):>5}  {mean:>14.4f}")
    print("-" * 50)
    valid   = [s for s in all_scores if not np.isnan(s)]
    overall = float(np.mean(valid)) if valid else float("nan")
    summary[f"{prefix}/mean"] = overall
    print(f"{'MEAN':>25}  {len(all_scores):>5}  {overall:>14.4f}")

    for key, scores in per_label.items():
        valid_cls = [s for s in scores if not np.isnan(s)]
        if valid_cls:
            summary[f"{prefix}/class/{key}"] = float(np.mean(valid_cls))

    if extra:
        summary.update(extra)
    if sample_table is not None:
        summary["samples"] = sample_table
    return summary
