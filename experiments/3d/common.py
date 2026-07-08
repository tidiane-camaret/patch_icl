"""
Shared dataset wiring for 3D in-context experiments.

`build_dataset(cfg, split)` is the single source of truth for "source -> 3D
dataset" construction, mirroring experiments/2d/common.py. All 3D train / eval /
plot scripts should build their datasets through here so they see exactly the
same data the models are trained on.
"""

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, RandomSampler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data.totalseg_classes import resolve_classes
from src.totalseg_dataloader_incontext import (
    TotalSegInContextDataset,
    incontext_collate_fn,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# data.source values served by TotalSegInContextDataset (differ only in root + classes).
_TOTALSEG_SOURCES = ("totalseg", "totalsegmri")


def _source_root(cfg) -> tuple[str, str, bool]:
    """Resolve (source, root, is_mri) from cfg.data.source — shared by all builders."""
    source = cfg.data.get("source", "totalseg")
    if source not in _TOTALSEG_SOURCES:
        raise ValueError(
            f"unknown data.source {source!r} (expected one of {_TOTALSEG_SOURCES})"
        )
    root = cfg.paths.get(source)
    if root is None:
        raise ValueError(f"cfg.paths.{source} is not set (needed for data.source={source!r})")
    return source, root, source == "totalsegmri"


def build_dataset(cfg, split: str) -> TotalSegInContextDataset:
    """Construct the 3D in-context dataset for `split`, dispatching on cfg.data.source.

    Split-aware, matching scripts/train.py: the 'train' split enables
    augmentation and the synth path; 'val'/'test' disable both.  Every data.*
    knob (including the newer random_coloring / num_labels_per_sample /
    n_synth_merge_*) is forwarded, so the dataset is identical to training.
    """
    d = cfg.data
    _, root, is_mri = _source_root(cfg)
    class_spec = d.train_classes if split == "train" else d.val_classes
    classes = resolve_classes(class_spec, root, is_mri=is_mri)

    is_train = split == "train"
    return TotalSegInContextDataset(
        root=root,
        classes=classes,
        image_size=tuple(d.image_size),
        split=split,
        context_size=d.context_size,
        max_subjects=(d.max_train_subjects if is_train else d.max_val_subjects),
        aug_cfg=(cfg.augmentations if is_train else None),
        synth_method=((d.synth_method or None) if is_train else None),
        synth_unions=d.synth_unions,
        p_synth=(d.p_synth if is_train else 0.0),
        class_balanced=d.class_balanced,
        use_crop=d.use_crop,
        random_coloring=d.get("random_coloring", False),
        num_labels_per_sample=d.get("num_labels_per_sample", 1),
        n_synth_merge_min=d.get("n_synth_merge_min", 1),
        n_synth_merge_max=d.get("n_synth_merge_max", 1),
    )


def train_loader(cfg) -> DataLoader:
    """Multi-class train loader over build_dataset(cfg, "train") — aug + synth on.

    Uses cfg.train.batch_size/workers; optionally caps samples per epoch via
    RandomSampler(cfg.data.max_ds_len_train). Mirrors scripts/train.py.
    """
    ds = build_dataset(cfg, "train")
    nw = int(cfg.train.workers)
    max_len = cfg.data.get("max_ds_len_train", None)
    sampler = None
    if max_len is not None:
        n = min(int(max_len), len(ds))
        sampler = RandomSampler(ds, replacement=False, num_samples=n)
    return DataLoader(
        ds,
        batch_size=int(cfg.train.batch_size),
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=nw,
        collate_fn=incontext_collate_fn,
        pin_memory=DEVICE.type == "cuda",
        persistent_workers=nw > 0,
        prefetch_factor=2 if nw > 0 else None,
    )


def make_eval_loader(cfg, classes, split: str = "test") -> DataLoader:
    """Multi-class eval loader (deterministic, no aug, no synth, class_balanced off).

    Builds ONE dataset over all `classes`, so the scan/bbox caches are loaded once
    instead of once per class. class_balanced=False makes `dataset.samples` a
    deterministic (subject, class) list, and shuffle=False keeps samples grouped
    by class; each item carries its own `label_name` for grouping downstream.

    Sources image_size / context_size / use_crop from cfg.data and
    n_subjects / batch_size / workers from cfg.eval, so the eval set is built from
    the same config surface as training.
    """
    d, e = cfg.data, cfg.eval
    _, root, is_mri = _source_root(cfg)
    ds = TotalSegInContextDataset(
        root=root,
        classes=list(classes),
        image_size=tuple(d.image_size),
        split=split,
        context_size=d.context_size,
        max_subjects=e.get("n_subjects", None),
        aug_cfg=None,
        synth_method=None,
        p_synth=0.0,
        class_balanced=False,
        use_crop=d.use_crop,
    )
    nw = int(e.get("workers", 4))
    return DataLoader(
        ds,
        batch_size=int(e.get("batch_size", 8)),
        shuffle=False,
        num_workers=nw,
        collate_fn=incontext_collate_fn,
        pin_memory=DEVICE.type == "cuda",
        persistent_workers=nw > 0,
        prefetch_factor=2 if nw > 0 else None,
    )


def make_loader(cfg, cls: str, split: str = "test") -> DataLoader:
    """Single-class eval loader — thin wrapper over make_eval_loader([cls])."""
    return make_eval_loader(cfg, [cls], split=split)
