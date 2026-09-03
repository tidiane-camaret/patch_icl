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
                                       # (cascade fine-crop seam)
    jitter: Optional[int] = None       # per-load crop-jitter override (native voxels);
                                       # None -> provider default self.crop_jitter.
                                       # Cascade re-crops pass 0 so the predicted COM is exact.


@dataclass
class LoadResult:
    image: torch.Tensor                # (1, T, T, T) f32, normalized
    label: torch.Tensor               # (T, T, T) i64, binary {0,1}
    spacing: torch.Tensor              # (3,) mm/voxel of the output
    crop_geom: torch.Tensor            # (4, 3) i64: starts, crop_sizes, out_sizes, pad_lo
    modality: str = "ct"               # "ct" | "mri" — rides for aug/analysis; encoder path ignores it


class VolumeProvider(Protocol):
    classes: list
    def subjects_for(self, cls: str) -> list: ...
    def load(self, subject: str, cls: str, req: LoadRequest) -> LoadResult: ...


class InContextDataset(Dataset):
    """Generic in-context task assembler over a VolumeProvider."""

    def __init__(self, provider, context_size=3, class_balanced=False,
                 aug_cfg=None, defer_aug=False, crop_spacing_mm=1.5, eval_seed=None,
                 max_tasks_per_class=None, gpu_realize_crop=False):
        self.provider = provider
        self.context_size = int(context_size)
        self.class_balanced = bool(class_balanced)
        self.aug_cfg = aug_cfg
        self.defer_aug = bool(defer_aug)
        self.crop_spacing_mm = float(crop_spacing_mm)
        self.eval_seed = eval_seed
        # gpu_realize_crop: emit an imageless `native_crop` payload (provider.load_native_crop)
        # that the cascade train loop resamples + paints on-GPU (src/gpu_realize_crop). Non-cohort
        # train path only; the engine skips CPU aug for imageless items.
        self.gpu_realize_crop = bool(gpu_realize_crop)
        # Cohort providers (e.g. synth_gmm) sample a whole K+1 cohort + a shared appearance
        # jointly, which the independent target+context load path below cannot express. They
        # implement the optional `assemble_task` hook instead; the engine then owns only aug +
        # the per-item RNG + (idx, spacing) unpacking. No per-subject samples in this mode.
        self.cohort_mode = hasattr(provider, "assemble_task")
        if self.cohort_mode:
            self.samples, self.active_classes = [], []
            self._length = int(provider.epoch_length)
        else:
            # max_tasks_per_class caps how many TARGETS each class contributes, without
            # touching the provider's subject pool — so contexts keep drawing from every
            # candidate in `subjects_for(cls)`. This is what decouples eval COST from eval
            # POOL: the alternative knob, eval.n_subjects, shrinks the provider itself and
            # therefore starves the context sampler too (see common.make_eval_loader).
            # Subsampling is seeded per class from eval_seed, so the task list is identical
            # across models, runs, workers and DataLoader order (cf. the eval-reproducibility
            # fix in docs/logs.md) — and stays sorted, keeping samples grouped by class for
            # the shuffle=False eval pass.
            cap = None if max_tasks_per_class is None else int(max_tasks_per_class)
            self.samples = []
            for c in provider.classes:
                subs = provider.subjects_for(c)
                if cap is not None and len(subs) > cap:
                    subs = sorted(random.Random(f"{eval_seed}:{c}").sample(list(subs), cap))
                self.samples += [(s, c) for s in subs]
            self.active_classes = [c for c in provider.classes if provider.subjects_for(c)]

    def __len__(self):
        return self._length if self.cohort_mode else len(self.samples)

    def _aug_active(self):
        return (self.aug_cfg is not None and getattr(self.aug_cfg, "enabled", False)
                and not self.defer_aug)

    def _augment_stacks(self, image_t, label_t, ctx_in, ctx_out):
        """Shared task (geometric) + per-volume intensity aug over target+contexts.

        ctx_in (K,1,T,T,T), ctx_out (K,T,T,T). Returns the augmented
        (image_t, label_t, ctx_in, ctx_out)."""
        imgs = torch.cat([image_t.unsqueeze(0), ctx_in], dim=0)
        msks = torch.cat([label_t.unsqueeze(0), ctx_out], dim=0)
        imgs, msks = apply_task_aug(imgs, msks, self.aug_cfg.task)
        for i in range(imgs.shape[0]):
            imgs[i] = apply_intensity_aug(imgs[i], self.aug_cfg.intensity)
        return imgs[0], msks[0], imgs[1:], msks[1:]

    def __getitem__(self, idx):
        crop_spacing = self.crop_spacing_mm
        if isinstance(idx, (tuple, list)):
            idx, crop_spacing = int(idx[0]), float(idx[1])
        rng = (random.Random(hash((self.eval_seed, idx)))
               if self.eval_seed is not None else random)

        if self.cohort_mode:
            item = self.provider.assemble_task(rng, crop_spacing)
            # gpu_realize items ship a native-crop payload (no "image") that is painted +
            # augmented on-GPU downstream; only the CPU-paint item gets the engine's aug.
            if "image" in item and self._aug_active():
                (item["image"], item["label"],
                 item["context_in"], item["context_out"]) = self._augment_stacks(
                    item["image"], item["label"], item["context_in"], item["context_out"])
            return item

        if self.class_balanced:
            cls = rng.choice(self.active_classes)
            subj = rng.choice(self.provider.subjects_for(cls))
        else:
            subj, cls = self.samples[idx]

        if self.gpu_realize_crop:
            req = LoadRequest(rng=rng, crop_spacing_mm=crop_spacing)
            tgt = self.provider.load_native_crop(subj, cls, req)
            ctx, ctx_subjects = [], []
            candidates = [s for s in self.provider.subjects_for(cls) if s != subj]
            for cs in _lazy_shuffle(rng, candidates):
                if len(ctx) >= self.context_size:
                    break
                try:
                    nc = self.provider.load_native_crop(cs, cls, LoadRequest(rng, crop_spacing))
                except Exception:
                    continue
                ctx.append(nc); ctx_subjects.append(cs)
            if not ctx:
                warnings.warn("InContextDataset: no context candidates; self-context "
                              "fallback (metrics leakage-inflated).", stacklevel=2)
                ctx.append(tgt); ctx_subjects.append(subj)
            while len(ctx) < self.context_size:
                i = rng.randrange(len(ctx))
                ctx.append(ctx[i]); ctx_subjects.append(ctx_subjects[i])
            # No "image" key: the engine skips CPU aug for imageless items; the real
            # context ids ride along so cascade._recrop_level can re-crop the same
            # contexts at level >= 1.
            return {"native_crop": [tgt, *ctx], "subject": subj,
                    "context_subjects": ctx_subjects, "label_name": cls,
                    "aug_mode": torch.tensor(0, dtype=torch.long)}

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

        ctx_in, ctx_out = torch.stack(context_in), torch.stack(context_out)
        if self._aug_active():
            image_t, label_t, ctx_in, ctx_out = self._augment_stacks(
                image_t, label_t, ctx_in, ctx_out)

        return {
            "image": image_t,
            "label": label_t,
            "context_in": ctx_in,
            "context_out": ctx_out,
            "subject": subj,
            "context_subjects": ctx_subjects,
            "label_name": cls,
            "spacing": tgt.spacing,
            "aug_mode": torch.tensor(0, dtype=torch.long),
            "crop_geom": tgt.crop_geom,
            "modality": tgt.modality,
        }
