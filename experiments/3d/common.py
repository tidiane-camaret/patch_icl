"""
Shared dataset wiring for 3D in-context experiments.

`build_dataset(cfg, split)` is the single source of truth for "source -> 3D
dataset" construction, mirroring experiments/2d/common.py. All 3D train / eval /
plot scripts should build their datasets through here so they see exactly the
same data the models are trained on.
"""

import math
import random
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data.totalseg_classes import resolve_classes, resolve_more_labels_classes
from src.totalseg_dataloader_incontext import (
    TotalSegInContextDataset,
    incontext_collate_fn,
)
from src.totalseg_more_labels_dataset import TotalSegMoreLabelsDataset
from src.chemotox_dataset import ChemoToxBCDataset, BC_NAMES
from src.synth_gen_maisi_dataset import SynthGenMaisiDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# data.source values served by TotalSegInContextDataset (differ only in root + classes).
_TOTALSEG_SOURCES = {"totalseg", "totalsegmri", "chemotox"}


def _source_root(cfg) -> tuple[str, str, bool]:
    """Resolve (source, root, is_mri) from cfg.data.source — shared by all builders."""
    source = cfg.data.get("source", "totalseg")
    if source == "totalseg_more_labels":
        root = cfg.paths.get("totalseg_more_labels")
        if root is None:
            raise ValueError("cfg.paths.totalseg_more_labels is not set "
                             "(needed for data.source=totalseg_more_labels)")
        return source, root, False
    if source in ("flare22", "nasalseg"):
        root = cfg.paths.get(source)
        if root is None:
            raise ValueError(f"cfg.paths.{source} is not set "
                             f"(needed for data.source={source})")
        return source, root, False
    if source == "chemotox_bc":
        root = cfg.paths.get("chemotox")
        if root is None:
            raise ValueError("cfg.paths.chemotox is not set (needed for source=chemotox_bc)")
        return source, root, False
    if source not in _TOTALSEG_SOURCES:
        raise ValueError(
            f"unknown data.source {source!r} (expected one of {_TOTALSEG_SOURCES})"
        )
    root = cfg.paths.get(source)
    if root is None:
        raise ValueError(f"cfg.paths.{source} is not set (needed for data.source={source!r})")
    return source, root, source == "totalsegmri"


def eval_cfg(cfg):
    """Return cfg with data.* overlaid by the optional `data.val` block — an eval-only
    source override so a run can TRAIN on one source and VALIDATE on another (e.g. train
    synth_gmm_maisi, eval totalseg). No `data.val` block -> cfg unchanged (val = train
    source). image_size/context_size are inherited from data.* (model input geometry must
    match), so keep them out of data.val; val-source keys (source, val_classes, use_crop,
    crop_spacing_mm, paths are shared) win in the merge."""
    from omegaconf import OmegaConf
    v = cfg.data.get("val")
    if not v:
        return cfg
    m = cfg.copy()
    m.data = OmegaConf.merge(cfg.data, v)
    return m


def _split_scalar(v, split: str, default: float = 0.0) -> float:
    """Resolve a possibly split-specific probability: a bare scalar applies to both splits,
    else read v.train (split=='train') or v.eval (val/test)."""
    if isinstance(v, (int, float, bool)):
        return float(v)
    return float(v.get("train" if split == "train" else "eval", default))


def _self_context(d, split: str) -> tuple[float, bool, bool, dict | None]:
    """Parse data.self_context -> (p, intensity, per_image, synth_masks) for `split`. Accepts
    the nested {p:{train, eval}, augs:{intensity, per_image}, synth_masks:{p:{train,eval}, ...}}
    form; every `p` may also be a bare scalar (both splits), and the whole block a bare
    scalar/bool (p only). split=='train' reads p.train; any other split (val/test) reads p.eval.
    augs apply to both splits. synth_masks (dict or None, with its own split-specific p already
    resolved to a scalar for `split`) drives the synthetic target label: its `sources`
    (ellipse/supervoxel) and per-source params are forwarded verbatim to the dataset."""
    sc = d.get("self_context", 0.0)
    if isinstance(sc, (int, float, bool)):
        return float(sc), False, False, None
    p_val = _split_scalar(sc.get("p", 0.0), split)
    augs = sc.get("augs", {}) or {}
    sm = sc.get("synth_masks", None)
    if sm is not None:
        from omegaconf import OmegaConf, DictConfig
        if isinstance(sm, DictConfig):
            sm = OmegaConf.to_container(sm, resolve=True)
        else:
            sm = dict(sm)
        sm["p"] = _split_scalar(sm.get("p", 0.0), split)   # collapse split-specific p -> scalar
    return (p_val,
            bool(augs.get("intensity", False)),
            bool(augs.get("per_image", False)),
            sm)


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
    d = cfg.data
    if d.get("source") in ("flare22", "nasalseg"):
        # Sources stored on their native anisotropic grid; the provider crops + resamples
        # to the isotropic model grid per item (src/providers/native_grid.py). v2 only.
        from src.incontext_dataset_v2 import InContextDataset
        _, root, _ = _source_root(cfg)
        if d.get("source") == "flare22":
            from src.providers.flare22 import Flare22Provider as _Provider
        else:
            from src.providers.nasalseg import NasalSegProvider as _Provider
        is_train = split == "train"
        provider = _Provider(
            root=root,
            classes=(d.get("train_classes", "all") if is_train else d.get("val_classes", "all")),
            image_size=tuple(d.image_size),
            max_subjects=(d.get("max_train_subjects") if is_train
                          else d.get("max_val_subjects")),
            crop_spacing_mm=d.get("crop_spacing_mm", 1.5),
            crop_jitter=(d.get("crop_jitter", 0) if is_train
                         else cfg.get("eval", {}).get("crop_jitter", 0)),
            mask_downsample=d.get("mask_downsample", "occupancy"),
            mask_occupancy_thr=d.get("mask_occupancy_thr", 0.5),
            image_antialias=d.get("image_antialias", True))
        return InContextDataset(
            provider, context_size=d.context_size,
            class_balanced=(is_train and d.get("class_balanced", False)),
            aug_cfg=(cfg.augmentations if is_train else None),
            defer_aug=(is_train and bool(cfg.get("augmentations", {}).get("gpu", False))),
            crop_spacing_mm=d.get("crop_spacing_mm", 1.5),
            eval_seed=(None if is_train else int(cfg.get("eval", {}).get("seed", 0))))

    if d.get("loader_v2", False) and d.get("source", "totalseg") in _TOTALSEG_SOURCES:
        from src.incontext_dataset_v2 import InContextDataset
        from src.providers.totalseg import TotalSegProvider
        _, root, is_mri = _source_root(cfg)
        is_train = split == "train"
        class_spec = d.train_classes if is_train else d.val_classes
        classes = resolve_classes(class_spec, root, is_mri=is_mri)
        provider = TotalSegProvider(
            root=root, classes=classes, image_size=tuple(d.image_size),
            split=split, max_subjects=(d.get("max_train_subjects") if is_train
                                       else d.get("max_val_subjects")),
            crop_spacing_mm=d.get("crop_spacing_mm", 1.5),
            crop_jitter=(d.get("crop_jitter") if is_train else cfg.get("eval", {}).get("crop_jitter", 0)),
            mask_downsample=d.get("mask_downsample", "occupancy"),
            mask_occupancy_thr=d.get("mask_occupancy_thr", 0.1),
            modality=("mri" if is_mri else "ct"))
        return InContextDataset(
            provider, context_size=d.context_size,
            class_balanced=(is_train and d.get("class_balanced", False)),
            aug_cfg=(cfg.augmentations if is_train else None),
            defer_aug=(is_train and bool(cfg.get("augmentations", {}).get("gpu", False))),
            crop_spacing_mm=d.get("crop_spacing_mm", 1.5),
            eval_seed=(None if is_train else int(cfg.get("eval", {}).get("seed", 0))))

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
    if cfg.data.get("source") == "totalseg_more_labels":
        d = cfg.data
        root = cfg.paths.get("totalseg_more_labels")
        classes = resolve_more_labels_classes(root, d.val_classes)
        return TotalSegMoreLabelsDataset(
            root=root, classes=classes, image_size=tuple(d.image_size),
            split=split, context_size=d.context_size,
            max_subjects=d.get("max_val_subjects"),
            eval_seed=int(cfg.get("eval", {}).get("seed", 0)),
            use_crop=d.get("use_crop", False),
            crop_spacing_mm=d.get("crop_spacing_mm", 1.5),
            # eval.crop_jitter: null -> T//4 training jitter; 0 -> centered deterministic crops
            crop_jitter=cfg.get("eval", {}).get("crop_jitter", None),
            raw_ct=d.get("raw_ct", False),
        )
    if cfg.data.get("source") == "chemotox_bc":
        d = cfg.data
        root = cfg.paths.get("chemotox")
        return ChemoToxBCDataset(
            root=root, classes=BC_NAMES, image_size=tuple(d.image_size),
            split=split, context_size=d.context_size,
            max_subjects=d.get("max_val_subjects"),
            eval_seed=int(cfg.get("eval", {}).get("seed", 0)),
            use_crop=True, crop_spacing_mm=d.get("crop_spacing_mm", 1.5),
            crop_jitter=cfg.get("eval", {}).get("crop_jitter", None),
        )
    if cfg.data.get("source") == "synth_gen_maisi":
        d = cfg.data
        root = cfg.paths.get("synth_gen_maisi")
        if root is None:
            raise ValueError("cfg.paths.synth_gen_maisi is not set "
                             "(needed for data.source=synth_gen_maisi)")
        # classes: explicit MAISI-name list from config; "all"/None -> every MAISI class
        spec = d.get("train_classes") if split == "train" else d.get("val_classes")
        classes = None if (spec is None or isinstance(spec, str)) else list(spec)
        return SynthGenMaisiDataset(
            root=root, classes=classes, image_size=tuple(d.image_size),
            split=split, context_size=d.context_size,
            max_subjects=d.get("max_train_subjects" if split == "train" else "max_val_subjects"),
            aug_cfg=(cfg.get("augmentations") if split == "train" else None),
            defer_aug_to_gpu=(split == "train" and bool(cfg.get("augmentations", {}).get("gpu", False))),
            class_balanced=d.get("class_balanced", False),
            eval_seed=(None if split == "train" else int(cfg.get("eval", {}).get("seed", 0))),
            use_crop=d.get("use_crop", False),
            crop_spacing_mm=d.get("crop_spacing_mm", 1.5),
            crop_jitter=cfg.get("eval", {}).get("crop_jitter", None) if split != "train" else d.get("crop_jitter"),
            mask_downsample=d.get("mask_downsample", "nearest"),
            mask_occupancy_thr=d.get("mask_occupancy_thr", 0.5),
        )

    if cfg.data.get("source") == "synth_gmm_maisi":
        from omegaconf import OmegaConf
        from src.synth_gmm_maisi_dataset import SynthGmmMaisiDataset
        d = cfg.data
        bank = cfg.paths.get("gmm_bank")
        if bank is None:
            raise ValueError("cfg.paths.gmm_bank is not set (needed for data.source=synth_gmm_maisi)")
        spec = d.get("train_classes") if split == "train" else d.get("val_classes")
        classes = None if (spec is None or isinstance(spec, str)) else list(spec)
        g = cfg.data.get("gmm", {})
        # train iterates epoch_length generative samples; val is capped small (deterministic
        # per idx via eval_seed) — max_val_subjects overrides, default 100.
        length = (int(d.get("epoch_length", 10000)) if split == "train"
                  else int(d.get("max_val_subjects") or 100))
        synth_ds = SynthGmmMaisiDataset(
            bank_dir=bank, image_size=tuple(d.image_size), context_size=d.context_size,
            crop_spacing_mm=d.get("crop_spacing_mm", 1.5),
            crop_jitter=(cfg.get("eval", {}).get("crop_jitter", None) if split != "train"
                         else d.get("crop_jitter")),
            classes=classes, length=length,
            var_max=float(g.get("var_max", 5.0)),
            background_mode=g.get("background_mode", "zero"),
            # match the totalseg training regime: occupancy area-pool + threshold so thin
            # organs survive the crop->grid downsample (nearest drops them at large FOV).
            mask_downsample=d.get("mask_downsample", "occupancy"),
            mask_occupancy_thr=float(d.get("mask_occupancy_thr", 0.1)),
            # class-uniform target sampling (rare organs as often as common) vs mask-frequency
            # prior — mirrors totalseg data.class_balanced.
            class_balanced=bool(d.get("class_balanced", True)),
            # to_container -> plain dict/list so CohortSampler's isinstance checks see native
            # types (a DictConfig/ListConfig randomness spec would slip past dict/list checks).
            # A config that doesn't define data.cohort yields a plain {} default (which
            # to_container rejects), so only unwrap when it's actually an OmegaConf node.
            cohort=(OmegaConf.to_container(d.cohort, resolve=True)
                    if OmegaConf.is_config(d.get("cohort")) else dict(d.get("cohort") or {})),
            # gpu_realize: ship native crops + defer occupancy resample + paint to the GPU
            # train loop (src/gpu_synth_realize). TRAIN only — val is a real source via eval_cfg.
            gpu_realize=(split == "train" and bool(d.get("gpu_realize", False))),
            gpu_realize_max_native=int(d.get("gpu_realize_max_native", 256)),
            eval_seed=(None if split == "train" else int(cfg.get("eval", {}).get("seed", 0))))
        if not d.get("loader_v2", False):
            return synth_ds
        # loader_v2: drive the same cohort dataset through the generic v2 engine via the
        # cohort hook (src/providers/synth_gmm.py) — so synth shares the engine's task+
        # intensity aug path with totalseg-v2. Aug is deferred to GPU (gpu=true) or skipped
        # on eval (aug_cfg=None); gpu_realize items carry no image and stay aug-free here.
        from src.incontext_dataset_v2 import InContextDataset
        from src.providers.synth_gmm import SynthGmmProvider
        is_train = split == "train"
        return InContextDataset(
            SynthGmmProvider(synth_ds), context_size=d.context_size,
            aug_cfg=(cfg.augmentations if is_train else None),
            defer_aug=(is_train and bool(cfg.get("augmentations", {}).get("gpu", False))),
            crop_spacing_mm=d.get("crop_spacing_mm", 1.5),
            eval_seed=(None if is_train else int(cfg.get("eval", {}).get("seed", 0))))

    d = cfg.data
    _sc_p, _sc_int, _sc_pi, _sc_synth = _self_context(d, split)
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
        defer_aug_to_gpu=(is_train and bool(cfg.get("augmentations", {}).get("gpu", False))),
        synth_method=((d.synth_method or None) if is_train else None),
        synth_unions=d.synth_unions,
        p_synth=(d.p_synth if is_train else 0.0),
        class_balanced=d.class_balanced,
        use_crop=d.use_crop,
        crop_spacing_mm=d.get("crop_spacing_mm", 1.5),
        mask_downsample=d.get("mask_downsample", "nearest"),
        mask_occupancy_thr=d.get("mask_occupancy_thr", 0.5),
        random_coloring=d.get("random_coloring", False),
        num_labels_per_sample=d.get("num_labels_per_sample", 1),
        n_synth_merge_min=d.get("n_synth_merge_min", 1),
        n_synth_merge_max=d.get("n_synth_merge_max", 1),
        raw_ct=d.get("raw_ct", False),
        modality=("mri" if is_mri else "ct"),
        self_context=_sc_p,
        self_context_intensity=_sc_int,
        self_context_per_image=_sc_pi,
        self_context_synth=_sc_synth,
    )


class SpacingBatchSampler:
    """Wrap a base sampler into fixed-size batches of (idx, spacing), one spacing per
    batch drawn log-uniformly in [lo, hi] mm. One spacing per batch lets the spacing-aware
    frozen encoder use a single shared (compile-safe) RoPE table for the whole forward,
    while the dataset crops each item at that same physical spacing (content matches rope).
    Log-uniform is scale-natural (equal weight per octave over 1-4 mm)."""

    def __init__(self, sampler, batch_size, spacing_range, drop_last=False, seed=0):
        self.sampler = sampler
        self.batch_size = int(batch_size)
        self.lo, self.hi = float(spacing_range[0]), float(spacing_range[1])
        self.drop_last = drop_last
        self._rng = random.Random(seed)

    def _sample(self) -> float:
        return math.exp(self._rng.uniform(math.log(self.lo), math.log(self.hi)))

    def __iter__(self):
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                s = self._sample()
                yield [(i, s) for i in batch]
                batch = []
        if batch and not self.drop_last:
            s = self._sample()
            yield [(i, s) for i in batch]

    def __len__(self):
        n = len(self.sampler)
        return n // self.batch_size if self.drop_last else (n + self.batch_size - 1) // self.batch_size


def train_loader(cfg) -> DataLoader:
    """Multi-class train loader over build_dataset(cfg, "train") — aug + synth on.

    Uses cfg.train.batch_size/workers; optionally caps samples per epoch via
    RandomSampler(cfg.data.max_ds_len_train). Mirrors scripts/train.py. When
    cfg.data.train_spacing_range is set, batches use one random physical spacing
    each (SpacingBatchSampler) for variable-spacing training. Eval never uses it —
    make_eval_loader always crops at the fixed cfg.data.crop_spacing_mm.
    """
    ds = build_dataset(cfg, "train")
    nw = int(cfg.train.workers)
    max_len = cfg.data.get("max_ds_len_train", None)
    bs = int(cfg.train.batch_size)
    base = (RandomSampler(ds, replacement=False, num_samples=min(int(max_len), len(ds)))
            if max_len is not None else RandomSampler(ds))
    # gpu_realize ships variable-shape native crops -> a list-preserving collate; the
    # SynthRealizer paints them on GPU in the train loop. Otherwise the default stacking collate.
    if cfg.data.get("source") == "synth_gmm_maisi" and cfg.data.get("gpu_realize", False):
        from src.gpu_synth_realize import synth_gpu_collate_fn
        collate = synth_gpu_collate_fn
    else:
        collate = incontext_collate_fn
    common = dict(num_workers=nw, collate_fn=collate,
                  pin_memory=DEVICE.type == "cuda", persistent_workers=nw > 0,
                  prefetch_factor=2 if nw > 0 else None)
    train_spacing_range = cfg.data.get("train_spacing_range", None)
    if train_spacing_range is not None:
        batch_sampler = SpacingBatchSampler(base, bs, train_spacing_range,
                                            seed=int(cfg.train.get("seed", 0)))
        return DataLoader(ds, batch_sampler=batch_sampler, **common)
    return DataLoader(ds, batch_size=bs, sampler=base, **common)


def make_eval_loader(cfg, classes, split: str = "test", spacing: float | None = None) -> DataLoader:
    """Multi-class eval loader (deterministic, no aug, no synth, class_balanced off).

    Builds ONE dataset over all `classes`, so the scan/bbox caches are loaded once
    instead of once per class. class_balanced=False makes `dataset.samples` a
    deterministic (subject, class) list, and shuffle=False keeps samples grouped
    by class; each item carries its own `label_name` for grouping downstream.

    Sources image_size / context_size / use_crop from cfg.data and
    n_subjects / batch_size / workers from cfg.eval, so the eval set is built from
    the same config surface as training.

    `spacing` (mm/voxel) forces every crop in the eval pass to that one physical
    spacing via SpacingBatchSampler([s, s]) over a SequentialSampler — the (idx, s)
    tuples reach __getitem__ inside worker processes (mutating ds.crop_spacing_mm
    would not). Only the totalseg direct-build branch honours it; the build_dataset-
    routed omnisynth3d/anchor_synth3d ignore it; totalseg_more_labels (a
    TotalSegInContextDataset subclass) honours it too. None = fixed-crop_spacing_mm pass.
    """
    d, e = cfg.data, cfg.eval
    _sc_p, _sc_int, _sc_pi, _sc_synth = _self_context(d, split)
    if d.get("source") in ("omnisynth3d", "anchor_synth3d", "totalseg_more_labels",
                            "chemotox_bc", "synth_gmm_maisi", "flare22", "nasalseg"):
        # omniSynth3D / anchor_synth3d / totalseg_more_labels compose their own
        # deterministic multi-class eval datasets; route through build_dataset (the
        # same dataset the trainer uses, deterministic for val/test). Their pool
        # already spans every anchor/tile-cache/more-labels class, so the `classes`
        # arg isn't re-applied here — each item carries its own label_name for
        # the same per-class grouping downstream.
        ds = build_dataset(cfg, split)
        nw = int(e.get("workers", 4))
        common = dict(
            num_workers=nw, collate_fn=incontext_collate_fn,
            pin_memory=DEVICE.type == "cuda", persistent_workers=nw > 0,
            prefetch_factor=2 if nw > 0 else None,
            # forkserver: eval workers spawn from a clean server that never touched CUDA, so
            # they don't inherit the parent's CUDA context (fork does -> the worker aborts in
            # ExchangeDevice with "CUDA error: initialization error" at the first eval batch,
            # since measure_flops/training already init CUDA in the parent). See docs/logs.md.
            multiprocessing_context=("forkserver" if nw > 0 and DEVICE.type == "cuda" else None),
        )
        if spacing is not None:
            # totalseg_more_labels subclasses TotalSegInContextDataset, so it honours the
            # (idx, spacing) crop override (its _load_crop sizes the FOV as T*self._crop_mm);
            # drive a constant-spacing pass like the direct totalseg path. omnisynth3d /
            # anchor_synth3d never reach here with spacing set (guarded out in eval.py).
            batch_sampler = SpacingBatchSampler(
                SequentialSampler(ds), int(e.get("batch_size", 8)), [spacing, spacing])
            return DataLoader(ds, batch_sampler=batch_sampler, **common)
        return DataLoader(ds, batch_size=int(e.get("batch_size", 8)), shuffle=False, **common)
    _, root, is_mri = _source_root(cfg)
    if d.get("loader_v2", False):
        # v2 eval loader: deterministic cross-subject tasks over the raw_ct crop provider
        # (src/incontext_dataset_v2.py). v2 does NOT implement the self_context/synth probes
        # (v2 design non-goals), so any eval self_context settings are ignored here. Mirrors
        # the v1 tail's (idx, spacing) constant-spacing handling below.
        from src.incontext_dataset_v2 import InContextDataset
        from src.providers.totalseg import TotalSegProvider
        provider = TotalSegProvider(
            root=root, classes=list(classes), image_size=tuple(d.image_size),
            split=split, max_subjects=e.get("n_subjects", None),
            crop_spacing_mm=d.get("crop_spacing_mm", 1.5),
            crop_jitter=e.get("crop_jitter", None),
            mask_downsample=d.get("mask_downsample", "occupancy"),
            mask_occupancy_thr=d.get("mask_occupancy_thr", 0.1),
            modality=("mri" if is_mri else "ct"))
        ds_v2 = InContextDataset(
            provider, context_size=d.context_size, class_balanced=False,
            aug_cfg=None, crop_spacing_mm=d.get("crop_spacing_mm", 1.5),
            eval_seed=int(e.get("seed", 0)),
            # eval.tasks_per_class caps the TARGETS per class; eval.n_subjects caps the
            # provider POOL (targets AND context candidates). Use tasks_per_class to bound
            # eval time while leaving n_subjects=null, so every task still retrieves its
            # contexts from the whole split.
            max_tasks_per_class=e.get("tasks_per_class", None))
        nw = int(e.get("workers", 4))
        common = dict(num_workers=nw, collate_fn=incontext_collate_fn,
                      pin_memory=DEVICE.type == "cuda", persistent_workers=nw > 0,
                      prefetch_factor=2 if nw > 0 else None,
                      # forkserver so eval workers don't inherit the parent CUDA context (fork
                      # -> "CUDA error: initialization error" abort at first eval batch). See
                      # the build_dataset branch above / docs/logs.md.
                      multiprocessing_context=("forkserver" if nw > 0 and DEVICE.type == "cuda" else None))
        if spacing is not None:
            batch_sampler = SpacingBatchSampler(
                SequentialSampler(ds_v2), int(e.get("batch_size", 8)), [spacing, spacing])
            return DataLoader(ds_v2, batch_sampler=batch_sampler, **common)
        return DataLoader(ds_v2, batch_size=int(e.get("batch_size", 8)), shuffle=False, **common)
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
        mask_downsample=d.get("mask_downsample", "nearest"),
        mask_occupancy_thr=d.get("mask_occupancy_thr", 0.5),
        # eval.crop_jitter=0 makes each (subject, class) crop deterministic (centered,
        # no random offset), so a crop is identical whether it appears as a target or
        # another sample's context — letting the frozen encode cache reuse it within an
        # epoch. Default None keeps the training-time jitter (T//4). See _load_crop.
        crop_jitter=e.get("crop_jitter", None),
        # Deterministic per-item context shuffle + crop jitter (reproducible across
        # models/workers/order); see TotalSegInContextDataset.eval_seed.
        eval_seed=int(e.get("seed", 0)),
        raw_ct=d.get("raw_ct", False),
        modality=("mri" if is_mri else "ct"),
        self_context=_sc_p,
        self_context_intensity=_sc_int,
        self_context_per_image=_sc_pi,
        self_context_synth=_sc_synth,
    )
    nw = int(e.get("workers", 4))
    common = dict(
        num_workers=nw,
        collate_fn=incontext_collate_fn,
        pin_memory=DEVICE.type == "cuda",
        persistent_workers=nw > 0,
        prefetch_factor=2 if nw > 0 else None,
        # forkserver so eval workers don't inherit the parent CUDA context (fork ->
        # "CUDA error: initialization error" abort at first eval batch). See docs/logs.md.
        multiprocessing_context=("forkserver" if nw > 0 and DEVICE.type == "cuda" else None),
    )
    if spacing is not None:
        # Constant-spacing pass: SpacingBatchSampler([s, s]) makes every batch that one
        # physical spacing; the (idx, s) tuples travel into worker __getitem__ so both the
        # crop and the reported `spacing` follow. SequentialSampler keeps eval order stable.
        batch_sampler = SpacingBatchSampler(
            SequentialSampler(ds), int(e.get("batch_size", 8)), [spacing, spacing])
        return DataLoader(ds, batch_sampler=batch_sampler, **common)
    return DataLoader(ds, batch_size=int(e.get("batch_size", 8)), shuffle=False, **common)


def make_loader(cfg, cls: str, split: str = "test") -> DataLoader:
    """Single-class eval loader — thin wrapper over make_eval_loader([cls])."""
    return make_eval_loader(cfg, [cls], split=split)
