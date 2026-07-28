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


def resolve_anchor_classes(anchor_cfg, root):
    """Anchor pool for anchor_synth3d: resolve `anchor_classes`, expanding an empty
    list to all 117 TotalSegmentator classes (the documented `[]` = all)."""
    classes = resolve_classes(anchor_cfg.get("anchor_classes") or (), totalseg_root=root)
    if not classes:
        from data.totalseg_classes import ALL_CLASSES
        classes = list(ALL_CLASSES[:117])
    return classes


_ANCHOR_SHAPES = ("blob", "elongated", "tubular")


def anchor_shapes(cfg):
    """Validation 'classes' for anchor_synth3d = the object shapes it emits, which is
    the val grouping key (label_name). `shape=mix` -> all three, else the single shape."""
    shape = cfg.anchor_synth.get("shape", "blob")
    return list(_ANCHOR_SHAPES) if shape == "mix" else [shape]


def build_dataset(cfg, split: str):
    """Construct the 3D in-context dataset for `split`, dispatching on cfg.data.source.

    Split-aware, matching scripts/train.py: the 'train' split enables
    augmentation and the synth path; 'val'/'test' disable both.  Every data.*
    knob (including the newer random_coloring / num_labels_per_sample /
    n_synth_merge_*) is forwarded, so the dataset is identical to training.
    """
    if cfg.data.get("source", "totalseg") == "omnisynth3d":
        from src.datasets.omniSynth.dataset3d import OmniSynth3DICLDataset
        from src.datasets.omniSynth.config import OmniTotalSegConfig
        s = cfg.get("synth3d")
        if s is None:
            raise ValueError("data.source=omnisynth3d requires a `synth3d` config block")
        tiles_root = s.get("tiles_root", None) or cfg.paths.get("totalseg")
        cfg3d = OmniTotalSegConfig(
            tiles_root=tiles_root,
            size=tuple(s.get("size", cfg.data.image_size)),
            classes=tuple(resolve_classes(s.get("classes") or (),
                                          totalseg_root=cfg.paths.get("totalseg"))),
            n_objects=int(s.get("n_objects", 4)),
            k_min=int(s.get("k_min", 1)), k_max=int(s.get("k_max", 2)),
            placement_tries=int(s.get("placement_tries", 4)),
            placement_max_overlap=float(s.get("placement_max_overlap", 0.1)),
            target_mode=s.get("target_mode", "class"),
            background=s.get("background", "black"),
            lru_classes=int(s.get("lru_classes", 64)),
            eval_seed_namespace=int(s.get("eval_seed_namespace", 0)),
            eval_subjects_per_task=int(s.get("eval_subjects_per_task", 4)),
            epoch_length=int(s.get("epoch_length", 10000)),
        )
        return OmniSynth3DICLDataset(split=split, context_size=cfg.data.context_size,
                                     cfg=cfg3d)
    if cfg.data.get("source") == "anchor_synth3d":
        from src.datasets.anchor_synth.dataset3d import AnchorSynth3DICLDataset
        a = cfg.get("anchor_synth")
        if a is None:
            raise ValueError("data.source=anchor_synth3d requires an `anchor_synth` block")
        root = cfg.paths.get("totalseg")
        classes = resolve_anchor_classes(a, root)
        is_train = split == "train"
        return AnchorSynth3DICLDataset(
            root=root, classes=classes, image_size=tuple(cfg.data.image_size),
            split=split, context_size=cfg.data.context_size,
            object_source=a.get("object_source", "blob"),
            shape=a.get("shape", "blob"), n_objects=int(a.get("n_objects", 1)),
            n_anchors=int(a.get("n_anchors", 4)),
            extrapolation=float(a.get("extrapolation", 0.3)),
            weight_concentration=float(a.get("weight_concentration", 1.0)),
            max_select_tries=int(a.get("max_select_tries", 20)),
            object_size_frac_min=float(a.get("object_size_frac_min", 0.3)),
            object_size_frac_max=float(a.get("object_size_frac_max", 0.8)),
            object_size_min_vox=int(a.get("object_size_min_vox", 6)),
            scale_jitter=float(a.get("scale_jitter", 0.15)),
            rotate_jitter=float(a.get("rotate_jitter", 12.0)),
            contrast_delta=float(a.get("contrast_delta", 0.15)),
            edge_blur=float(a.get("edge_blur", 0.08)),
            boundary_complexity=float(a.get("boundary_complexity", 0.0)),
            harmonic_amp=float(a.get("harmonic_amp", 0.30)),
            n_harmonics=int(a.get("n_harmonics", 4)),
            eccentricity=float(a.get("eccentricity", 3.0)),
            eval_subjects_per_task=int(a.get("eval_subjects_per_task", 4)),
            eval_seed_namespace=int(a.get("eval_seed_namespace", 0)),
            epoch_length=int(a.get("epoch_length", 10000)),
            deterministic=(split != "train"),
            aug_cfg=(cfg.get("augmentations") if is_train else None),
            max_subjects=(cfg.data.get("max_train_subjects") if is_train
                          else cfg.data.get("max_val_subjects")))
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
        crop_spacing_mm=d.get("crop_spacing_mm", 1.5),
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
    if d.get("source") in ("omnisynth3d", "anchor_synth3d"):
        # omniSynth3D / anchor_synth3d compose their own deterministic multi-class
        # eval scenes; route through build_dataset (the same dataset the trainer
        # uses, deterministic for val/test). Their pool already spans every anchor/
        # tile-cache class, so the `classes` arg isn't re-applied here — each item
        # carries its own label_name for the same per-class grouping downstream.
        ds = build_dataset(cfg, split)
        nw = int(e.get("workers", 4))
        return DataLoader(
            ds, batch_size=int(e.get("batch_size", 8)), shuffle=False,
            num_workers=nw, collate_fn=incontext_collate_fn,
            pin_memory=DEVICE.type == "cuda", persistent_workers=nw > 0,
            prefetch_factor=2 if nw > 0 else None,
        )
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
        crop_spacing_mm=d.get("crop_spacing_mm", 1.5),
        # eval.crop_jitter=0 makes each (subject, class) crop deterministic (centered,
        # no random offset), so a crop is identical whether it appears as a target or
        # another sample's context — letting the frozen encode cache reuse it within an
        # epoch. Default None keeps the training-time jitter (T//4). See _load_crop.
        crop_jitter=e.get("crop_jitter", None),
        # Deterministic per-item context shuffle + crop jitter (reproducible across
        # models/workers/order); see TotalSegInContextDataset.eval_seed.
        eval_seed=int(e.get("seed", 0)),
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
