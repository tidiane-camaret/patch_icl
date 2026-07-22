"""AnchorSynth3DICLDataset: draws synthetic objects at a consistent position
relative to a shared anchor organ, on real TotalSegmentator CT backgrounds.

Subclasses TotalSegInContextDataset to reuse its scan cache, class-balanced
anchor/subject sampling, and pre-resized fast-path loading. The anchor organ is a
landmark only (never labeled); the label is the drawn object(s). The per-item task
spec (per-object offset/geometry/contrast) is drawn once and shared across the K+1
scenes; only small scale/rotation jitter varies per scene. See
docs/superpowers/specs/2026-07-22-anchor-synth3d-design.md.
"""

import numpy as np
import torch

from src.totalseg_dataloader_incontext import TotalSegInContextDataset
from src.totalseg_dataset import _ALL_CLASSES_IDX
from .shapes import sample_object_spec, render_object, small_rotation, roughen
from .draw import anchor_stats, offset_to_center, place_object


class AnchorSynth3DICLDataset(TotalSegInContextDataset):
    def __init__(self, root, classes, image_size=(128, 128, 128), split="train",
                 context_size=1, object_source="blob", shape="blob", n_objects=1,
                 offset_range=0.6, scale_frac=0.4, scale_jitter=0.15,
                 rotate_jitter=12.0, contrast_delta=0.15, edge_blur=0.08,
                 boundary_complexity=0.0, harmonic_amp=0.30, eccentricity=3.0,
                 n_harmonics=4, deterministic=None, eval_seed_namespace=0,
                 eval_subjects_per_task=4, epoch_length=10000, max_subjects=None):
        if object_source != "blob":
            raise NotImplementedError(
                f"object_source={object_source!r} not implemented in v1 (blob only)")
        super().__init__(root=root, classes=classes, image_size=image_size,
                         split=split, context_size=context_size,
                         max_subjects=max_subjects, class_balanced=True)
        self.object_source = object_source
        self.shape = shape
        self.n_objects = int(n_objects)
        self.offset_range = float(offset_range)
        self.scale_frac = float(scale_frac)
        self.scale_jitter = float(scale_jitter)
        self.rotate_jitter = float(rotate_jitter)
        self.contrast_delta = float(contrast_delta)
        self.edge_blur = float(edge_blur)
        self.boundary_complexity = float(boundary_complexity)
        self.harmonic_amp = float(harmonic_amp)
        self.eccentricity = float(eccentricity)
        self.n_harmonics = int(n_harmonics)
        self.eval_seed_namespace = int(eval_seed_namespace)
        self.eval_subjects_per_task = int(eval_subjects_per_task)
        self.epoch_length = int(epoch_length)
        self.anchor_deterministic = (split != "train") if deterministic is None else deterministic

        if self.anchor_deterministic:
            self._eval_index = [(cls, s) for cls in self.active_classes
                                for s in range(self.eval_subjects_per_task)]
            self._n = len(self._eval_index)
        else:
            self._eval_index = None
            self._n = self.epoch_length

    def __len__(self):
        return self._n

    def _draw_specs(self, rng):
        """Per-item task spec (shared across the K+1 scenes)."""
        specs = []
        for _ in range(self.n_objects):
            specs.append({
                "geom": sample_object_spec(
                    rng, shape=self.shape, eccentricity=self.eccentricity,
                    n_harmonics=self.n_harmonics, harmonic_amp=self.harmonic_amp,
                    edge_blur=self.edge_blur),
                "offset": rng.uniform(-self.offset_range, self.offset_range, size=3),
                "contrast": float(rng.uniform(-1.0, 1.0) * self.contrast_delta),
            })
        return specs

    def _render_subject(self, subj, anchor_cls, specs, scene_rng):
        image_t, anchor_t = self._load(subj, anchor_cls)          # fast path
        img = image_t.squeeze(0).numpy().astype(np.float32)  # (D,H,W)
        label = np.zeros(img.shape, dtype=np.int64)
        stats = anchor_stats(anchor_t.cpu().numpy())
        if stats is not None:
            centroid, extent, _ = stats
            base = self.scale_frac * float(np.mean(extent))
            for lid, spec in enumerate(specs, 1):
                jit = 1.0 + scene_rng.uniform(-self.scale_jitter, self.scale_jitter)
                size = max(3, int(round(base * jit)))
                alpha = render_object(size, spec["geom"],
                                      R_extra=small_rotation(scene_rng, self.rotate_jitter))
                if self.boundary_complexity > 0.0:
                    alpha = roughen(alpha, self.boundary_complexity, scene_rng)
                center = offset_to_center(centroid, extent, spec["offset"],
                                          size, img.shape)
                place_object(img, alpha, center, spec["contrast"], label, lid)
        return torch.from_numpy(img).unsqueeze(0), torch.from_numpy(label)

    def __getitem__(self, idx):
        if self.anchor_deterministic:
            anchor_cls, sample_index = self._eval_index[idx]
            item_rng = np.random.default_rng(np.random.SeedSequence(
                [self.eval_seed_namespace, _ALL_CLASSES_IDX[anchor_cls], sample_index]))
        else:
            item_rng = np.random.default_rng()
            anchor_cls = self.active_classes[item_rng.integers(len(self.active_classes))]

        subs = self.label_to_subjects[anchor_cls]
        ordered = [subs[i] for i in item_rng.permutation(len(subs))]
        target = ordered[0]
        context_pool = ordered[1:]                        # excludes the target (no leakage)
        contexts = list(context_pool[:self.context_size])
        while len(contexts) < self.context_size:
            if context_pool:
                contexts.append(context_pool[int(item_rng.integers(len(context_pool)))])
            else:
                contexts.append(target)                   # only fallback: self-context (rare)
        chosen = [target] + contexts

        specs = self._draw_specs(item_rng)
        scene_seeds = item_rng.integers(0, 2 ** 32, size=len(chosen))
        scenes = [self._render_subject(subj, anchor_cls, specs,
                                       np.random.default_rng(int(s)))
                  for subj, s in zip(chosen, scene_seeds)]

        image_t, label_t = scenes[0]
        ctx = scenes[1:]
        return {
            "image":       image_t,
            "label":       label_t,
            "context_in":  torch.stack([c[0] for c in ctx]),
            "context_out": torch.stack([c[1] for c in ctx]),
            "subject":     chosen[0],
            "label_name":  anchor_cls,
            "spacing":     self._get_spacing(chosen[0]),
            "meta": {"anchor": anchor_cls,
                     "n_objects": self.n_objects,
                     "offsets": [spec["offset"].tolist() for spec in specs],
                     "contrasts": [spec["contrast"] for spec in specs]},
        }
