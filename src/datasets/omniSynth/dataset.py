"""OmniSynthICLDataset: composes grid scenes of Omniglot characters into the
in-context contract (image, label, context_in, context_out + meta), matching
SynthICLDataset so the existing TaggedDataset/collate wrappers work unchanged.

Determinism (mirrors controlSynth): train draws fresh entropy per subject; val/
test derive every subject seed from (eval_seed_namespace, task_id, sample_index)
-> byte-identical eval set. A separate item-level rng (distinct spawn key) fixes
the shared target base bitmap for identical/aug modes across query + contexts.
"""

from collections.abc import Mapping

import numpy as np
import torch
from torch.utils.data import Dataset

from .bank import get_or_build_bank
from .config import OmniDiversityConfig, OmniSamplingConfig, OmniSceneConfig
from .render import make_distractor_sampler, make_target_sampler, render_scene

# target_mode="mix" samples uniformly over these per item.
_TARGET_MODES = ("identical", "aug", "class")


def _to_img_tensor(arr):
    return torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32)).unsqueeze(0)


class OmniSynthICLDataset(Dataset):
    def __init__(self, split="train", context_size=3, image_size=64,
                 diversity=None, scene=None, sampling=None, deterministic=None):
        self.split = split
        self.context_size = context_size
        self.image_size = image_size
        self.diversity = diversity or OmniDiversityConfig()
        self.scene = scene or OmniSceneConfig()
        self.sampling = sampling or OmniSamplingConfig()
        self.deterministic = (split != "train") if deterministic is None else deterministic

        grid = self.scene.grid
        if image_size % grid != 0:
            raise ValueError(f"image_size {image_size} not divisible by grid {grid}")
        self.cell_size = image_size // grid

        self.bank = get_or_build_bank(self.diversity, self.cell_size)
        self.pool = self.bank.task_ids(split)
        if not self.pool:
            raise ValueError(f"empty class pool for split {split!r}")

        if self.deterministic:
            self._eval_index = []                       # idx -> (class_id, subject_index)
            self.samples = []
            for class_id in self.pool:
                alph = self.bank.alphabet(class_id)
                for s in range(self.sampling.eval_subjects_per_task):
                    self.samples.append((f"omniglot/{alph}", len(self._eval_index), 1))
                    self._eval_index.append((class_id, s))
        else:
            self._eval_index = None
            self.samples = [("omniglot/train", i, 1)
                            for i in range(self.sampling.epoch_length)]

    def __len__(self):
        return len(self.samples)

    def _subject_rngs(self, task_id, sample_index):
        n = self.context_size + 1
        if self.deterministic:
            ss = np.random.SeedSequence([int(self.sampling.eval_seed_namespace),
                                         int(task_id), int(sample_index)])
            return [np.random.default_rng(c) for c in ss.spawn(n)]
        return [np.random.default_rng() for _ in range(n)]

    def _item_rng(self, task_id, sample_index):
        """Item-level rng for the shared target base (identical/aug). Distinct
        namespace offset so it never collides with the subject seeds above."""
        if self.deterministic:
            ss = np.random.SeedSequence([int(self.sampling.eval_seed_namespace) + 1,
                                         int(task_id), int(sample_index)])
            return np.random.default_rng(ss)
        return np.random.default_rng()

    def _resolve_target_mode(self, rng):
        """Resolve scene.target_mode to one of identical|aug|class for this item.

        Accepts a fixed string, the literal "mix" (uniform over the three), a
        sequence (uniform), or a {mode: weight} mapping (weighted) — mirroring
        controlSynth's morphology mixture. Drawn once per item so the query and
        all K contexts share the same task definition (and it's deterministic for
        eval via the item rng). Fixed strings don't consume the rng, so eval with a
        fixed mode keeps its byte-identical seeding."""
        tm = self.scene.target_mode
        if isinstance(tm, str):
            if tm != "mix":
                return tm
            return _TARGET_MODES[int(rng.integers(len(_TARGET_MODES)))]
        if isinstance(tm, Mapping):
            keys = list(tm)
            w = np.array([float(tm[k]) for k in keys], dtype=float)
            return keys[int(rng.choice(len(keys), p=w / w.sum()))]
        seq = list(tm)                                   # list / OmegaConf ListConfig
        return seq[int(rng.integers(len(seq)))]

    def __getitem__(self, idx):
        if self.deterministic:
            class_id, sample_index = self._eval_index[idx]
        else:
            class_id = int(self.pool[np.random.default_rng().integers(len(self.pool))])
            sample_index = idx

        rngs = self._subject_rngs(class_id, sample_index)
        base_rng = self._item_rng(class_id, sample_index)
        mode = self._resolve_target_mode(base_rng)

        target_sampler = make_target_sampler(self.bank, class_id, self.scene, base_rng,
                                             mode=mode)
        distractor_sampler = make_distractor_sampler(self.bank, self.pool, class_id)

        def scene(rng):
            return render_scene(rng, self.scene, self.scene.grid, self.cell_size,
                                target_sampler, distractor_sampler)

        t_img, t_seg, t_k = scene(rngs[0])
        ctx = [scene(rngs[1 + i]) for i in range(self.context_size)]

        is_copy = False
        copy_slot = -1
        if not self.deterministic and self.context_size > 0 and self.scene.p_copy > 0.0:
            crng = np.random.default_rng()       # isolated: never perturbs subject/item seeds
            if crng.random() < self.scene.p_copy:
                n = max(1, min(int(self.scene.n_copy), self.context_size))
                slots = crng.permutation(self.context_size)[:n].tolist()
                for j in slots:
                    ctx[j] = (t_img.copy(), t_seg.copy(), t_k)   # exact copy of the query scene
                is_copy = True
                copy_slot = min(slots)

        return {
            "image":       _to_img_tensor(t_img),
            "label":       _to_img_tensor(t_seg),
            "context_in":  torch.stack([_to_img_tensor(c[0]) for c in ctx]),
            "context_out": torch.stack([_to_img_tensor(c[1]) for c in ctx]),
            "meta": {
                "class_id": int(class_id),
                "alphabet": self.bank.alphabet(class_id),
                "subject_index": int(sample_index),
                "target_mode": mode,
                "k_target": int(t_k),
                "is_copy": bool(is_copy),
                "copy_slot": int(copy_slot),
            },
        }
