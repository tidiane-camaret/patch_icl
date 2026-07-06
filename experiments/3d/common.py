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
from torch.utils.data import DataLoader

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


def make_loader(cfg, cls: str, split: str = "test") -> DataLoader:
    """Single-class eval loader (deterministic, no aug, no synth, class_balanced off).

    Sources image_size / context_size / use_crop from cfg.data and
    n_subjects / batch_size / workers from cfg.eval, so the eval set is built from
    the same config surface as training. Used by experiments/3d/eval.py per class.
    """
    d, e = cfg.data, cfg.eval
    _, root, is_mri = _source_root(cfg)
    ds = TotalSegInContextDataset(
        root=root,
        classes=[cls],
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
