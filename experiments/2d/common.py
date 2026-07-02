"""
Shared utilities for 2D MedSegBench evaluation scripts.
"""

import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, "/home/dpxuser/repos/UniverSeg")

from src.datasets.medsegbench import MedSegBenchDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Dataset helpers ───────────────────────────────────────────────────────────

def collate(batch):
    batch = [b for b in batch if b["context_in"].shape[0] > 0]
    if not batch:
        return None
    out = {
        "image":       torch.stack([b["image"]       for b in batch]),
        "label":       torch.stack([b["label"]       for b in batch]),
        "context_in":  torch.stack([b["context_in"]  for b in batch]),
        "context_out": torch.stack([b["context_out"] for b in batch]),
        "dataset":     [b["dataset"]     for b in batch],
        "sample_idx":  [b["sample_idx"]  for b in batch],
        "label_value": [b["label_value"] for b in batch],
    }
    # Pass through per-element `meta` when the dataset provides it (controlSynth
    # attaches per-subject synth params here). Kept as a Python list of dicts so
    # downstream benchmarks can read every knob value for the batch element.
    if "meta" in batch[0]:
        out["meta"] = [b["meta"] for b in batch]
    return out


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


def build_dataset(cfg, split: str):
    """Construct the raw (untagged) dataset for `split`, dispatching on cfg.data.source.

    The single source of truth for "source -> dataset" wiring; all train/eval
    scripts go through here. Source-specific deps (biomedparse, controlSynth) are
    imported lazily so the common medsegbench path stays import-light.
    """
    source = cfg.data.get("source", "medsegbench")
    datasets = [cfg.data.dataset] if cfg.data.get("dataset", None) else None
    if source == "medsegbench":
        return MedSegBenchDataset(
            split=split, context_size=cfg.data.context_size,
            image_size=cfg.data.image_size, datasets=datasets,
        )
    if source == "biomedparse":
        from src.datasets.biomedparse import BiomedParseDataset
        # biomedparse has only 'train' / 'test' — map the training script's 'val'
        # eval split onto 'test' (its only held-out split).
        bp_split = "test" if split == "val" else split
        return BiomedParseDataset(
            split=bp_split, context_size=cfg.data.context_size,
            image_size=cfg.data.image_size, datasets=datasets,
        )
    if source == "totalseg2d":
        from src.datasets.totalseg2d import TotalSeg2DDataset
        d = cfg.data
        return TotalSeg2DDataset(
            split=split, context_size=cfg.data.context_size,
            image_size=cfg.data.image_size,
            stored_size=d.get("stored_size", 256),
            hu_window=tuple(d.get("hu_window", (-1000.0, 1000.0))),
            min_area=d.get("min_area", 16),
        )
    if source == "synthetic":
        from src.datasets.controlSynth import (
            DifficultyBuildSpec, DifficultyLiveConfig, DiversityConfig, SynthICLDataset,
        )
        s = cfg.synth
        return SynthICLDataset(
            split=split,
            context_size=cfg.data.context_size,
            image_size=cfg.data.image_size,
            diversity=DiversityConfig(
                num_tasks=s.diversity.num_tasks, num_labels=s.diversity.num_labels,
                context_size=cfg.data.context_size, master_seed=s.diversity.master_seed,
                splits=tuple(s.diversity.splits)),
            build_spec=DifficultyBuildSpec(**dict(s.build)),
            difficulty_live=DifficultyLiveConfig(**dict(s.live)),
            epoch_length=s.sampling.epoch_length,
            eval_seed_namespace=s.sampling.eval_seed_namespace,
            eval_subjects_per_task=s.sampling.eval_subjects_per_task,
            noise_bank_size=s.get("noise_bank_size", 256),
        )
    if source == "omnisynth":
        from src.datasets.omniSynth import (
            OmniDiversityConfig, OmniSamplingConfig, OmniSceneConfig, OmniSynthICLDataset,
        )
        s = cfg.synth
        # omniglot_root comes from cfg.paths.omniglot when the config tree includes a
        # `paths` block (train configs do); the standalone eval_base.yaml has none, so
        # fall back to OmniDiversityConfig's default path.
        paths = cfg.get("paths", None)
        div_kwargs = dict(s.diversity)
        if paths is not None and paths.get("omniglot", None):
            div_kwargs["omniglot_root"] = paths.omniglot
        return OmniSynthICLDataset(
            split=split,
            context_size=cfg.data.context_size,
            image_size=cfg.data.image_size,
            diversity=OmniDiversityConfig(**div_kwargs),
            scene=OmniSceneConfig(**dict(s.scene)),
            sampling=OmniSamplingConfig(**dict(s.sampling)),
        )
    raise ValueError(
        f"unknown data.source {source!r} "
        "(medsegbench | biomedparse | totalseg2d | synthetic | omnisynth)")


def make_loader(ds, cfg, split: str, shuffle: bool) -> DataLoader:
    """Wrap a raw dataset in TaggedDataset + collate and build its DataLoader.

    Shared train/eval policy:
      - non-train splits (val/test) are subsampled to cfg.eval.max_per_label per
        (dataset, label_value) cell when set; train is never subsampled.
      - train uses cfg.train.batch_size/workers and an optional RandomSampler
        (cfg.data.max_train_samples); eval/val use cfg.eval.batch_size/workers.
    """
    if split != "train" and cfg.eval.get("max_per_label", None):
        max_per_label = cfg.eval.max_per_label
        groups: dict[tuple, list[int]] = {}
        for i, (ds_name, _, lv) in enumerate(ds.samples):
            groups.setdefault((ds_name, lv), []).append(i)
        keep: list[int] = []
        for indices in groups.values():
            keep.extend(random.sample(indices, min(max_per_label, len(indices))))
        keep = sorted(keep)
        # Keep the COW-safe SampleIndex when present (don't rebuild a list of tuples).
        ds.samples = (ds.samples.subset(keep) if hasattr(ds.samples, "subset")
                      else [ds.samples[i] for i in keep])
        print(f"{split}: subsampled to {len(ds.samples)} samples "
              f"(max {max_per_label} per dataset/label)")

    is_train = split == "train"
    bs = cfg.train.batch_size if is_train else cfg.eval.batch_size
    nw = cfg.train.workers   if is_train else cfg.eval.workers
    max_train = cfg.data.get("max_train_samples", None)
    sampler = (RandomSampler(ds, replacement=False, num_samples=max_train)
               if is_train and max_train is not None else None)
    return DataLoader(
        TaggedDataset(ds),
        batch_size=bs,
        shuffle=(shuffle and sampler is None),
        sampler=sampler,
        num_workers=nw,
        collate_fn=collate,
        pin_memory=DEVICE.type == "cuda",
        persistent_workers=nw > 0,
        prefetch_factor=4 if nw > 0 else None,
    )


def build_loader(cfg) -> DataLoader:
    """Build a tagged, collated DataLoader from a Hydra eval config (cfg.data.split)."""
    split = cfg.data.split
    return make_loader(build_dataset(cfg, split), cfg, split, shuffle=False)


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


def cosine_sim(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-6) -> float:
    """Scale-invariant similarity of two soft maps: Σ(p·g) / (‖p‖·‖g‖).

    Unlike soft_dice, the magnitude cancels (the denominator is quadratic in the
    values, matching the numerator), so it reaches 1.0 whenever pred and GT agree in
    shape/location regardless of the absolute occupancy — a meaningful 0→1 progress
    signal at low prediction resolutions, where avg-pooling a sparse mask yields tiny
    (≪0.5) soft targets that pin soft_dice near its mean-occupancy ceiling. Returns
    NaN when the GT map is empty.
    """
    p = pred.float().flatten()
    g = gt.float().flatten()
    den = p.norm() * g.norm()
    return float((p * g).sum() / den) if den > eps else float("nan")


def topk_overlap(pred: torch.Tensor, gt: torch.Tensor, k: int, eps: float = 1e-6) -> float:
    """Recall of the GT-positive patches within the k highest-valued predicted patches.

    GT maps are sparse (usually < k positive patches), so dividing by k would penalise a
    model that already found every true patch. Instead the denominator is the number of
    GT-positive patches (patches with GT > 0), capped to k: score = |gt_pos ∩ topk(pred)|
    / |gt_pos|. It reaches 1.0 exactly when ALL true patches are among the model's top-k.
    Purely rank-based on the pred side (threshold/scale-free). Meant for low-res patch
    maps — at native res a "patch" is a pixel. Returns NaN when the GT map is empty.
    """
    p = pred.float().flatten()
    g = gt.float().flatten()
    n_pos = int((g > eps).sum())
    if n_pos == 0:
        return float("nan")
    k = min(k, p.numel())
    m = min(n_pos, k)                         # GT-positive count, capped to k
    gi = torch.topk(g, m).indices            # the m highest GT patches (= all positives when n_pos ≤ k)
    pi = torch.topk(p, k).indices
    return float(torch.isin(gi, pi).sum()) / m


def batch_dice_sums(prob: torch.Tensor, target: torch.Tensor, eps: float = 1e-6):
    """Batched soft & hard Dice SUMS for cheap *training-accuracy* logging.

    Same per-row semantics as soft_dice / hard_dice (a row whose pred+GT are both
    empty is skipped), but vectorised and kept on-device: returns running SUMS and
    valid-row COUNTS as 0-dim tensors so the caller accumulates over an epoch and
    syncs (.item()) ONCE, avoiding a per-batch GPU→CPU stall. Computed from the
    logits/target the forward already produced, so there is no extra model pass.

    `prob` is a probability map (post-sigmoid), `target` the soft (avg-pooled) GT;
    both (B, ...) and flattened per row. soft = 2·Σpg/(Σp+Σg); hard binarises both at
    0.5 first. Returns (soft_sum, soft_cnt, hard_sum, hard_cnt).
    """
    p = prob.detach().flatten(1).float()
    g = target.detach().flatten(1).float()
    den_s = p.sum(1) + g.sum(1)
    ok_s  = den_s > eps
    soft  = torch.where(ok_s, 2 * (p * g).sum(1) / den_s.clamp_min(eps), torch.zeros_like(den_s))
    pb, gb = (p >= 0.5).float(), (g >= 0.5).float()
    den_h = pb.sum(1) + gb.sum(1)
    ok_h  = den_h > eps
    hard  = torch.where(ok_h, 2 * (pb * gb).sum(1) / den_h.clamp_min(eps), torch.zeros_like(den_h))
    return soft.sum(), ok_s.sum(), hard.sum(), ok_h.sum()


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

    # Per-cell means + a macro-average over them. A "cell" = one per_label group,
    # i.e. (dataset, label_value) — for BiomedParse this is (dataset, target).
    # Macro weights every cell equally, so multi-label datasets (m2caiseg etc.)
    # can't dominate the headline the way the per-sample micro-average lets them.
    cell_means = []
    for key, scores in per_label.items():
        valid_cls = [s for s in scores if not np.isnan(s)]
        if valid_cls:
            m = float(np.mean(valid_cls))
            summary[f"{prefix}/class/{key}"] = m
            cell_means.append(m)
    macro = float(np.mean(cell_means)) if cell_means else float("nan")
    summary[f"{prefix}/macro"] = macro
    print(f"{'MACRO (per cell)':>25}  {len(cell_means):>5}  {macro:>14.4f}")

    if extra:
        summary.update(extra)
    if sample_table is not None:
        summary["samples"] = sample_table
    return summary
