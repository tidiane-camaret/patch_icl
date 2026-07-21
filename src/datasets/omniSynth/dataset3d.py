"""OmniSynth3DICLDataset: paints bbox-cropped TotalSegmentator organs at random 3D
positions onto a D×H×W canvas, emitting the TotalSegInContextDataset contract
(image/label/context_in/context_out/subject/label_name/spacing) so the existing 3D
pipeline + incontext_collate_fn consume it unchanged.

Reuses OmniSynthICLDataset's deterministic RNG seeding + target-mode resolution, and
render.make_target_sampler/make_distractor_sampler (dimension-agnostic for the
identical|class modes used here). Free placement only; no per-item scipy warps."""

import numpy as np
import torch

from .bank_totalseg import get_or_build_totalseg_bank
from .config import OmniTotalSegConfig
from .dataset import OmniSynthICLDataset
from .render import make_distractor_sampler, make_target_sampler
from .render3d import render_scene_3d

_TARGET_MODES_3D = ("identical", "class")


class OmniSynth3DICLDataset(OmniSynthICLDataset):
    def __init__(self, split="train", context_size=3, cfg=None, deterministic=None):
        self.split = split
        self.context_size = context_size
        self.cfg = cfg or OmniTotalSegConfig()
        if self.cfg.target_mode not in _TARGET_MODES_3D:
            raise ValueError(f"3D target_mode must be identical|class, got "
                             f"{self.cfg.target_mode!r}")
        # The reused parent helpers read self.sampling.eval_seed_namespace and
        # self.scene.target_mode — point both at the single 3D config.
        self.sampling = self.cfg
        self.scene = self.cfg
        self.canvas = tuple(int(v) for v in self.cfg.size)
        self.deterministic = (split != "train") if deterministic is None else deterministic

        self.bank = get_or_build_totalseg_bank(self.cfg.tiles_root, self.cfg.size,
                                               split, tuple(self.cfg.classes),
                                               self.cfg.lru_classes)
        self.pool = self.bank.task_ids(split)
        if not self.pool:
            raise ValueError(f"empty class pool for split {split!r}")

        if self.deterministic:
            self._eval_index = []
            self.samples = []
            for class_id in self.pool:
                for s in range(self.cfg.eval_subjects_per_task):
                    self.samples.append(len(self._eval_index))
                    self._eval_index.append((class_id, s))
        else:
            self._eval_index = None
            self.samples = list(range(self.cfg.epoch_length))

    def __len__(self):
        return len(self.samples)

    def _render(self, rng, target_sampler, distractor_sampler):
        return render_scene_3d(
            rng, self.canvas, self.cfg.n_objects, self.cfg.k_min, self.cfg.k_max,
            target_sampler, distractor_sampler,
            tries=self.cfg.placement_tries, max_overlap=self.cfg.placement_max_overlap,
            background=self.cfg.background)

    def __getitem__(self, idx):
        if self.deterministic:
            class_id, sample_index = self._eval_index[idx]
        else:
            class_id = int(self.pool[np.random.default_rng().integers(len(self.pool))])
            sample_index = idx

        rngs = self._subject_rngs(class_id, sample_index)     # inherited
        base_rng = self._item_rng(class_id, sample_index)     # inherited
        mode = self._resolve_target_mode(base_rng)            # inherited

        target_sampler = make_target_sampler(self.bank, class_id, self.scene,
                                             base_rng, mode=mode)
        distractor_sampler = make_distractor_sampler(self.bank, self.pool, class_id)

        t_img, t_seg, _, _ = self._render(rngs[0], target_sampler, distractor_sampler)
        ctx = [self._render(rngs[1 + i], target_sampler, distractor_sampler)
               for i in range(self.context_size)]

        def _img(a):   # (D,H,W) float -> (1,D,H,W) float32
            return torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)).unsqueeze(0)

        def _lbl(a):   # (D,H,W) float -> (D,H,W) int64 binary
            return torch.from_numpy((np.ascontiguousarray(a) > 0).astype(np.int64))

        return {
            "image":       _img(t_img),
            "label":       _lbl(t_seg),
            "context_in":  torch.stack([_img(c[0]) for c in ctx]),
            "context_out": torch.stack([_lbl(c[1]) for c in ctx]),
            "subject":     f"omni_{int(class_id)}_{int(sample_index)}",
            "label_name":  self.bank.alphabet(class_id),
            "spacing":     torch.ones(3, dtype=torch.float32),
        }
