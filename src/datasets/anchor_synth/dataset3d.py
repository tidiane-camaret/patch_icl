"""AnchorSynth3DICLDataset: draws synthetic objects at a position determined by
a barycentric combination of multiple anchor organ centroids, on real
TotalSegmentator CT backgrounds.

Subclasses TotalSegInContextDataset to reuse its scan cache, class-balanced
anchor/subject sampling, and pre-resized fast-path loading. Anchors are
landmarks only (never labeled); the label is the drawn object(s). The per-item
task spec (per-object weights/geometry/contrast) is drawn once and shared across
the K+1 scenes; only small scale/rotation jitter varies per scene. See
docs/superpowers/specs/2026-07-22-anchor-synth3d-design.md.
"""

import numpy as np
import torch

from src.totalseg_dataloader_incontext import TotalSegInContextDataset
from src.augmentations import apply_task_aug, apply_intensity_aug
from src.totalseg_dataset import _ALL_CLASSES_IDX
from .shapes import sample_object_spec, render_object, small_rotation, roughen
from .draw import anchor_stats, barycentric_center, frame_length, affine_weights
from .draw import place_object


class AnchorSynth3DICLDataset(TotalSegInContextDataset):
    def __init__(self, root, classes, image_size=(128, 128, 128), split="train",
                 context_size=1, object_source="blob", shape="blob", n_objects=1,
                 n_anchors=4, extrapolation=0.3, weight_concentration=1.0,
                 max_select_tries=20, object_size_frac_min=0.3, object_size_frac_max=0.8,
                 object_size_min_vox=6, scale_jitter=0.15, rotate_jitter=12.0,
                 contrast_delta=0.15, edge_blur=0.08, boundary_complexity=0.0,
                 harmonic_amp=0.30, eccentricity=3.0, n_harmonics=4, deterministic=None,
                 eval_seed_namespace=0, eval_subjects_per_task=4, epoch_length=10000,
                 max_subjects=None, aug_cfg=None):
        if object_source != "blob":
            raise NotImplementedError(
                f"object_source={object_source!r} not implemented in v1 (blob only)")
        super().__init__(root=root, classes=classes, image_size=image_size,
                         split=split, context_size=context_size,
                         max_subjects=max_subjects, class_balanced=True)
        self.aug_cfg = aug_cfg          # set AFTER super().__init__ (which nulls it)
        self.object_source = object_source
        self.shape = shape
        self.n_objects = int(n_objects)
        self.n_anchors = int(n_anchors)
        self.extrapolation = float(extrapolation)
        self.weight_concentration = float(weight_concentration)
        self.max_select_tries = int(max_select_tries)
        self.object_size_frac_min = float(object_size_frac_min)
        self.object_size_frac_max = float(object_size_frac_max)
        self.object_size_min_vox = int(object_size_min_vox)
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

        # Co-occurrence structures over the anchor pool (self.classes), from the
        # parent's per-split label_to_subjects. Anchors define POSITION + SIZE only.
        self.subject_sets = {c: set(s) for c, s in self.label_to_subjects.items() if s}
        self.subject_to_classes: dict[str, set] = {}
        for c, subs in self.subject_sets.items():
            for s in subs:
                self.subject_to_classes.setdefault(s, set()).add(c)
        self.eligible_subjects = sorted(
            s for s, cs in self.subject_to_classes.items() if len(cs) >= self.n_anchors)

        if not self.eligible_subjects:
            raise ValueError(
                f"AnchorSynth3DICLDataset: no subject in split {split!r} has "
                f">= n_anchors={self.n_anchors} pool classes (classes={list(self.classes)!r}).")

        if self.anchor_deterministic:
            self._eval_index = [(subj, s) for subj in self.eligible_subjects
                                for s in range(self.eval_subjects_per_task)]
            self._n = len(self._eval_index)
        else:
            self._eval_index = None
            self._n = self.epoch_length

    def __len__(self):
        return self._n

    def _draw_specs(self, rng):
        """Per-item task spec (shared across the K+1 scenes): shape geometry,
        barycentric weights, size fraction, contrast — all anchor-independent."""
        specs = []
        for _ in range(self.n_objects):
            specs.append({
                "geom": sample_object_spec(
                    rng, shape=self.shape, eccentricity=self.eccentricity,
                    n_harmonics=self.n_harmonics, harmonic_amp=self.harmonic_amp,
                    edge_blur=self.edge_blur),
                "weights": affine_weights(rng, self.n_anchors, self.extrapolation,
                                          self.weight_concentration),
                "size_frac": float(rng.uniform(self.object_size_frac_min,
                                               self.object_size_frac_max)),
                "contrast": float(rng.uniform(-1.0, 1.0) * self.contrast_delta),
            })
        return specs

    def _select_anchors(self, subj, rng):
        """Pick n_anchors classes present in `subj`, preferring a set whose mutual
        co-occurrence yields >= context_size other subjects. Returns (anchors, cooccur)."""
        present = sorted(self.subject_to_classes[subj])
        best = None
        for _ in range(self.max_select_tries):
            pick = [present[i] for i in
                    rng.choice(len(present), self.n_anchors, replace=False)]
            cooccur = set.intersection(*(self.subject_sets[c] for c in pick)) - {subj}
            if best is None or len(cooccur) > len(best[1]):
                best = (pick, cooccur)
            if len(cooccur) >= self.context_size:
                break
        return best

    def _load_scene(self, subj):
        """Fast-path load of (ct float32 (D,H,W), full label volume (D,H,W))."""
        subj_dir = self.root / subj
        image = np.load(subj_dir / f"ct_{self._size_str}.npy", mmap_mode="r")
        full = np.load(subj_dir / f"label_{self._size_str}.npy", mmap_mode="r")
        return np.array(image, dtype=np.float32), np.asarray(full)

    def _render_subject(self, subj, anchors, specs, scene_rng):
        img, full = self._load_scene(subj)                 # img is a writable copy
        label = np.zeros(img.shape, dtype=np.int64)
        centroids = []
        for c in anchors:                                  # anchor -> POSITION + SIZE only
            st = anchor_stats(full == _ALL_CLASSES_IDX[c])
            if st is None:                                 # anchor vanished at this res
                return torch.from_numpy(img).unsqueeze(0), torch.from_numpy(label)
            centroids.append(st[0])
        centroids = np.stack(centroids)                    # (n_anchors, 3)
        L = frame_length(centroids)                        # orientation-invariant scale
        for lid, spec in enumerate(specs, 1):
            jit = 1.0 + scene_rng.uniform(-self.scale_jitter, self.scale_jitter)
            size = max(self.object_size_min_vox, int(round(spec["size_frac"] * L * jit)))
            alpha = render_object(size, spec["geom"],
                                  R_extra=small_rotation(scene_rng, self.rotate_jitter))
            if self.boundary_complexity > 0.0:
                alpha = roughen(alpha, self.boundary_complexity, scene_rng)
            center = barycentric_center(centroids, spec["weights"], size, img.shape)
            place_object(img, alpha, center, spec["contrast"], label, lid)
        return torch.from_numpy(img).unsqueeze(0), torch.from_numpy(label)

    def __getitem__(self, idx):
        if self.anchor_deterministic:
            target, sample_index = self._eval_index[idx]
            item_rng = np.random.default_rng(np.random.SeedSequence(
                [self.eval_seed_namespace,
                 self.eligible_subjects.index(target), sample_index]))
        else:
            item_rng = np.random.default_rng()
            target = self.eligible_subjects[item_rng.integers(len(self.eligible_subjects))]

        anchors, cooccur = self._select_anchors(target, item_rng)
        pool = sorted(cooccur)
        contexts = ([pool[i] for i in item_rng.permutation(len(pool))][:self.context_size]
                    if pool else [])
        while len(contexts) < self.context_size:
            if pool:
                contexts.append(pool[int(item_rng.integers(len(pool)))])
            else:
                contexts.append(target)                    # last-resort self-context (rare)
        chosen = [target] + contexts

        specs = self._draw_specs(item_rng)
        scene_seeds = item_rng.integers(0, 2 ** 32, size=len(chosen))
        scenes = [self._render_subject(subj, anchors, specs,
                                       np.random.default_rng(int(s)))
                  for subj, s in zip(chosen, scene_seeds)]

        image_t, label_t = scenes[0]
        context_in = [c[0] for c in scenes[1:]]
        context_out = [c[1] for c in scenes[1:]]

        if self.aug_cfg is not None and self.aug_cfg.enabled and len(context_in) > 0:
            all_images = torch.cat([image_t.unsqueeze(0), torch.stack(context_in)], dim=0)
            all_masks  = torch.cat([label_t.unsqueeze(0), torch.stack(context_out)], dim=0)
            all_images, all_masks = apply_task_aug(all_images, all_masks, self.aug_cfg.task)
            for i in range(all_images.shape[0]):
                all_images[i] = apply_intensity_aug(all_images[i], self.aug_cfg.intensity)
            image_t     = all_images[0]
            label_t     = all_masks[0]
            context_in  = list(all_images[1:])
            context_out = list(all_masks[1:])

        return {
            "image":       image_t,
            "label":       label_t,
            "context_in":  torch.stack(context_in),
            "context_out": torch.stack(context_out),
            "subject":     target,
            "label_name":  specs[0]["geom"].get("shape", self.shape),   # group by shape
            "spacing":     self._get_spacing(target),
            "meta": {"anchors": list(anchors),
                     "n_objects": self.n_objects,
                     "shapes": [s["geom"].get("shape") for s in specs],
                     "weights": [np.asarray(s["weights"]).tolist() for s in specs],
                     "contrasts": [s["contrast"] for s in specs]},
        }
