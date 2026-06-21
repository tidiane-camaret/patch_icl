"""
In-RAM geometry bank: the V1 analog of the spec's LMDB store + precompute.py.

Built once in SynthICLDataset.__init__ (before workers fork, so the arrays are
inherited copy-on-write). Holds every task's base geometry, a precomputed Perlin
noise bank, and the disjoint train/val/test task-id pools. All deterministic in
`master_seed` so a run is reproducible from config alone, and identical whether
built here or (later) offline.
"""

import numpy as np

from .config import DifficultyBuildSpec, DiversityConfig
from .task import make_base_geometry, resolve_difficulty


def _task_rng(master_seed, task_id):
    """Independent, reproducible Generator per task (spec ss6 determinism)."""
    return np.random.default_rng([int(master_seed), int(task_id)])


def split_task_ids(num_tasks, splits, master_seed):
    """Disjoint train/val/test id pools (unseen-anatomy eval, spec ss7)."""
    rng = np.random.default_rng([int(master_seed), 99])
    ids = np.arange(num_tasks)
    rng.shuffle(ids)
    n_tr = int(round(splits[0] * num_tasks))
    n_va = int(round(splits[1] * num_tasks))
    return {
        "train": sorted(ids[:n_tr].tolist()),
        "val":   sorted(ids[n_tr:n_tr + n_va].tolist()),
        "test":  sorted(ids[n_tr + n_va:].tolist()),
    }


def build_noise_bank(n, image_size, master_seed):
    """Precomputed multi-octave Perlin-like fields [n,H,W], each unit-std."""
    from scipy.ndimage import gaussian_filter
    rng = np.random.default_rng([int(master_seed), 7777])
    bank = np.empty((n, image_size, image_size), dtype=np.float32)
    for i in range(n):
        field = np.zeros((image_size, image_size), dtype=np.float32)
        for sigma, w in ((image_size * 0.04, 1.0), (image_size * 0.10, 0.6),
                         (image_size * 0.25, 0.3)):
            field += w * gaussian_filter(rng.standard_normal((image_size, image_size)),
                                         sigma).astype(np.float32)
        field -= field.mean()
        field /= (field.std() + 1e-8)
        bank[i] = field
    return bank


class GeometryBank:
    """Read-only RAM store of base geometries + noise bank + split pools."""

    def __init__(self, diversity: DiversityConfig, build_spec: DifficultyBuildSpec,
                 image_size: int, noise_bank_size: int = 256, verbose: bool = True):
        self.diversity = diversity
        self.image_size = image_size
        self._records = []          # list[(label_map uint8, fg_label int, meta dict)]

        if verbose:
            print(f"[controlSynth] building {diversity.num_tasks} base geometries "
                  f"(morphology={build_spec.morphology}, size={image_size})...")
        for task_id in range(diversity.num_tasks):
            rng = _task_rng(diversity.master_seed, task_id)
            morphology, geo = resolve_difficulty(build_spec, task_id, rng)
            label_map, fg_label, meta = make_base_geometry(
                image_size, morphology, geo, diversity.num_labels, rng)
            meta["task_id"] = task_id
            self._records.append((label_map, fg_label, meta))

        self._splits = split_task_ids(diversity.num_tasks, diversity.splits,
                                      diversity.master_seed)
        self._noise_bank = build_noise_bank(noise_bank_size, image_size,
                                             diversity.master_seed)
        if verbose:
            n = {k: len(v) for k, v in self._splits.items()}
            print(f"[controlSynth] done. task pools: {n}")

    def __len__(self):
        return len(self._records)

    def get(self, task_id):
        """(label_map uint8 [H,W] READ-ONLY, fg_label int, meta dict)."""
        return self._records[task_id]

    def task_ids(self, split):
        return self._splits[split]

    def noise_bank(self):
        return self._noise_bank

    def difficulty_table(self):
        """One meta row per task -> list of dicts (DataFrame-ready)."""
        return [dict(meta, fg_label=fg) for _, fg, meta in self._records]
