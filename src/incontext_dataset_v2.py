"""Generic in-context segmentation dataset engine (v2).

A source-agnostic `InContextDataset` assembles items from a `VolumeProvider`.
Per-item state flows through `LoadRequest`/`LoadResult`, so there is no mutable
instance side-channel (contrast the v1 `_cur_rng`/`_last_crop_geom`).
"""
import random
import warnings
from dataclasses import dataclass
from typing import Optional, Protocol

import torch
from torch.utils.data import Dataset

from src.totalseg_dataloader_incontext import _lazy_shuffle
from src.augmentations import apply_task_aug, apply_intensity_aug


@dataclass
class LoadRequest:
    rng: random.Random                 # per-item RNG (eval determinism or global)
    crop_spacing_mm: float             # physical crop pitch for THIS item
    center: Optional[tuple] = None     # native-voxel crop center; None -> provider default
                                       # (cascade fine-crop seam; v2 always passes None)


@dataclass
class LoadResult:
    image: torch.Tensor                # (1, T, T, T) f32, normalized
    label: torch.Tensor               # (T, T, T) i64, binary {0,1}
    spacing: torch.Tensor              # (3,) mm/voxel of the output
    crop_geom: torch.Tensor            # (4, 3) i64: starts, crop_sizes, out_sizes, pad_lo


class VolumeProvider(Protocol):
    classes: list
    def subjects_for(self, cls: str) -> list: ...
    def load(self, subject: str, cls: str, req: LoadRequest) -> LoadResult: ...


class InContextDataset(Dataset):
    """Generic in-context task assembler over a VolumeProvider."""

    def __init__(self, provider, context_size=3, class_balanced=False,
                 aug_cfg=None, defer_aug=False, crop_spacing_mm=1.5, eval_seed=None):
        self.provider = provider
        self.context_size = int(context_size)
        self.class_balanced = bool(class_balanced)
        self.aug_cfg = aug_cfg
        self.defer_aug = bool(defer_aug)
        self.crop_spacing_mm = float(crop_spacing_mm)
        self.eval_seed = eval_seed
        self.samples = [(s, c) for c in provider.classes
                        for s in provider.subjects_for(c)]
        self.active_classes = [c for c in provider.classes if provider.subjects_for(c)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        crop_spacing = self.crop_spacing_mm
        if isinstance(idx, (tuple, list)):
            idx, crop_spacing = int(idx[0]), float(idx[1])
        rng = (random.Random(hash((self.eval_seed, idx)))
               if self.eval_seed is not None else random)

        if self.class_balanced:
            cls = rng.choice(self.active_classes)
            subj = rng.choice(self.provider.subjects_for(cls))
        else:
            subj, cls = self.samples[idx]

        req = LoadRequest(rng=rng, crop_spacing_mm=crop_spacing)
        tgt = self.provider.load(subj, cls, req)
        image_t, label_t = tgt.image, tgt.label

        context_in, context_out, ctx_subjects = [], [], []
        candidates = [s for s in self.provider.subjects_for(cls) if s != subj]
        for cs in _lazy_shuffle(rng, candidates):
            if len(context_in) >= self.context_size:
                break
            try:
                r = self.provider.load(cs, cls, LoadRequest(rng, crop_spacing))
            except Exception:
                continue
            context_in.append(r.image); context_out.append(r.label); ctx_subjects.append(cs)

        if not context_in:
            warnings.warn("InContextDataset: no context candidates; self-context "
                          "fallback (metrics leakage-inflated).", stacklevel=2)
            context_in.append(image_t.clone()); context_out.append(label_t.clone())
            ctx_subjects.append(subj)
        while len(context_in) < self.context_size:
            i = rng.randrange(len(context_in))
            context_in.append(context_in[i].clone())
            context_out.append(context_out[i].clone())
            ctx_subjects.append(ctx_subjects[i])

        if self.aug_cfg is not None and getattr(self.aug_cfg, "enabled", False) and not self.defer_aug:
            imgs = torch.cat([image_t.unsqueeze(0), torch.stack(context_in)], dim=0)
            msks = torch.cat([label_t.unsqueeze(0), torch.stack(context_out)], dim=0)
            imgs, msks = apply_task_aug(imgs, msks, self.aug_cfg.task)
            for i in range(imgs.shape[0]):
                imgs[i] = apply_intensity_aug(imgs[i], self.aug_cfg.intensity)
            image_t, label_t = imgs[0], msks[0]
            context_in, context_out = list(imgs[1:]), list(msks[1:])

        return {
            "image": image_t,
            "label": label_t,
            "context_in": torch.stack(context_in),
            "context_out": torch.stack(context_out),
            "subject": subj,
            "context_subjects": ctx_subjects,
            "label_name": cls,
            "spacing": tgt.spacing,
            "aug_mode": torch.tensor(0, dtype=torch.long),
            "crop_geom": tgt.crop_geom,
        }
