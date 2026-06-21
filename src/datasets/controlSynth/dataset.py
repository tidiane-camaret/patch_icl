"""
SynthICLDataset: live path + the pfn_seg integration contract.

Returns the same 4-key dict as MedSegBenchDataset (image, label, context_in,
context_out) plus a `meta` dict, and exposes `.samples` so the existing
TaggedDataset/collate wrappers work unchanged.

Determinism (spec ss6): train draws fresh entropy per subject (infinite subject
diversity); eval derives every subject seed from (eval_seed_namespace, task_id,
sample_index) -> byte-identical val set across runs. Base geometry + noise bank
live in a process-shared GeometryBank built once and inherited by forked workers.
"""

import numpy as np
import torch
from torch.utils.data import Dataset

from .appearance import add_noise, gmm_fill
from .config import (
    DifficultyBuildSpec,
    DifficultyLiveConfig,
    DiversityConfig,
)
from .deformation import deform, jitter_pose
from .geometry import GeometryBank

# Process-level cache so train + val loaders (same config) share one bank.
_BANK_CACHE: dict = {}


def get_or_build_bank(diversity, build_spec, image_size, noise_bank_size=256):
    key = (repr(diversity), repr(build_spec), image_size, noise_bank_size)
    if key not in _BANK_CACHE:
        _BANK_CACHE[key] = GeometryBank(diversity, build_spec, image_size,
                                        noise_bank_size)
    return _BANK_CACHE[key]


def _to_img_tensor(arr):
    return torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32)).unsqueeze(0)


# Short labels for the difficulty factor swept in a run -> goes into the val sample
# name so W&B keys self-document (e.g. dice/dataset/synth/blob/amb0.40).
_FACTOR_ABBR = {
    "task_ambiguity": "amb", "task_ambiguity_intensity": "ambI",
    "noise_level": "noise", "region_size": "rsize", "thinness": "thin",
    "boundary_complexity": "bnd", "foreground_contrast": "fgc",
    "support_query_shift": "shift", "context_consistency": "consist",
    "context_copy_fraction": "copy", "scattered_clustering": "clust",
    "tortuosity": "tort", "branching_density": "branch",
    "texture_heterogeneity": "tex",
}


def difficulty_tag(build_spec, live):
    """`<abbr><value>` for build_spec.bin_factor (read from build or live config).

    Returns None if the factor isn't a scalar we can format (e.g. morphology).
    """
    factor = getattr(build_spec, "bin_factor", "task_ambiguity")
    val = getattr(build_spec, factor, None)
    if val is None:
        val = getattr(live, factor, None)
    if not isinstance(val, (int, float)):
        return None
    abbr = _FACTOR_ABBR.get(factor, factor[:4])
    return f"{abbr}{float(val):.2f}"


def _binned_tag(factor, value, lo, hi, n_bins):
    """`<abbr><bin-center>` — buckets `value` into n_bins over [lo, hi].

    Used by the per_task_sampled val grid so tasks of similar difficulty share a
    metric key (a difficulty-response curve from a single run).
    """
    abbr = _FACTOR_ABBR.get(factor, factor[:4])
    n_bins = max(1, int(n_bins))
    if hi <= lo:
        return f"{abbr}{float(value):.2f}"
    k = min(n_bins - 1, max(0, int((value - lo) / (hi - lo) * n_bins)))
    center = lo + (k + 0.5) * (hi - lo) / n_bins
    return f"{abbr}{center:.2f}"


class SynthICLDataset(Dataset):
    def __init__(
        self,
        split: str = "train",
        context_size: int = 3,
        image_size: int = 128,
        diversity: DiversityConfig = None,
        build_spec: DifficultyBuildSpec = None,
        difficulty_live: DifficultyLiveConfig = None,
        deterministic: bool = None,
        epoch_length: int = 10000,
        eval_seed_namespace: int = 0,
        eval_subjects_per_task: int = 4,
        noise_bank_size: int = 256,
    ):
        self.split = split
        self.context_size = context_size
        self.image_size = image_size
        self.diversity = diversity or DiversityConfig(context_size=context_size)
        self.build_spec = build_spec or DifficultyBuildSpec()
        self.live = difficulty_live or DifficultyLiveConfig()
        # train -> non-deterministic; val/test -> deterministic, unless overridden.
        self.deterministic = (split != "train") if deterministic is None else deterministic
        self.epoch_length = epoch_length
        self.eval_seed_namespace = eval_seed_namespace

        self.bank = get_or_build_bank(self.diversity, self.build_spec, image_size,
                                      noise_bank_size)
        self.pool = self.bank.task_ids(split)
        if not self.pool:
            raise ValueError(f"empty task pool for split {split!r}")

        # `.samples` = (name, sample_idx, label_value) for TaggedDataset/collate.
        if self.deterministic:
            # Val eval grid. The sample name carries the morphology AND a difficulty
            # tag, so run_eval's per-dataset grouping stratifies Dice for free:
            #   fixed            -> one run-level tag (e.g. "synth/blob/amb0.40")
            #   per_task_sampled -> each task binned by its own sampled bin_factor
            #                       value (e.g. "synth/blob/amb0.30") -> difficulty curve
            bf = self.build_spec.bin_factor
            sampled = dict(getattr(self.build_spec, "sampled", {}) or {})
            if self.build_spec.mode == "per_task_sampled" and bf in sampled:
                lo, hi = float(sampled[bf][0]), float(sampled[bf][1])
                tag_for = lambda meta: _binned_tag(
                    bf, meta["geo_params"][bf], lo, hi, self.build_spec.n_bins)
            else:
                fixed = difficulty_tag(self.build_spec, self.live)
                tag_for = lambda meta: fixed
            self._eval_index = []                    # idx -> (task_id, subject_index)
            self.samples = []
            for task_id in self.pool:
                meta = self.bank.get(task_id)[2]
                tag = tag_for(meta)
                suffix = f"/{tag}" if tag else ""
                for s in range(eval_subjects_per_task):
                    self.samples.append((f"synth/{meta['morphology']}{suffix}",
                                         len(self._eval_index), 1))
                    self._eval_index.append((task_id, s))
        else:
            self._eval_index = None
            self.samples = [("synth/train", i, 1) for i in range(epoch_length)]

    def __len__(self):
        return len(self.samples)

    # -- subject construction --------------------------------------------------

    def _subject_rngs(self, task_id, sample_index):
        """K+1 independent Generators (target + contexts)."""
        n = self.context_size + 1
        if self.deterministic:
            ss = np.random.SeedSequence([int(self.eval_seed_namespace),
                                         int(task_id), int(sample_index)])
            return [np.random.default_rng(c) for c in ss.spawn(n)]
        return [np.random.default_rng() for _ in range(n)]

    def _make_subject(self, base, fg, distractors, rng, fg_sign=None, shift_scale=1.0):
        """(img float32[H,W], seg float32[H,W], warped_label_map uint8[H,W]).

        `shift_scale` < 1 makes a near-aligned 'pristine' exemplar (the ease knob);
        the background is still drawn fresh so the foreground stays the only region
        that consistently matches the query (a frame-copy would make the background
        match too, which is anti-informative for a context-matcher).
        """
        warped = deform(base, self.live.support_query_shift * shift_scale, rng, self.image_size)
        # Per-subject pose jitter (position + size) -> within-set spread like real data.
        # Scaled by shift_scale so pristine context exemplars also stay near-aligned.
        warped = jitter_pose(warped, self.live.support_query_translate * shift_scale,
                             self.live.support_query_scale * shift_scale,
                             fg, rng, self.image_size)
        img = gmm_fill(warped, fg, distractors,
                       self.live.foreground_contrast,
                       self.live.texture_heterogeneity,
                       self.live.task_ambiguity_intensity, rng, fg_sign=fg_sign)
        img = add_noise(img, self.live.noise_level, self.bank.noise_bank(), rng)
        seg = (warped == fg).astype(np.float32)
        return img, seg, warped

    def _corrupt_mask(self, warped, fg, distractors, rng):
        """Label-swap analog: point the mask at a wrong region (spec Axis A)."""
        present = [d for d in distractors if (warped == d).any()]
        if not present:
            present = [l for l in np.unique(warped) if l != fg and l != 0]
        if not present:
            return np.zeros_like(warped, dtype=np.float32)
        wrong = present[rng.integers(len(present))]
        return (warped == wrong).astype(np.float32)

    def _apply_context_consistency(self, ctx, fg, distractors, rng):
        """Consistency corruption (harder): point a fraction of context masks at a
        wrong region (spec Axis A). The ease side (copy_fraction) is applied at
        generation time via shift_scale, not here."""
        out_in, out_seg = [], []
        for img, seg, warped in ctx:
            if rng.random() > self.live.context_consistency:
                seg = self._corrupt_mask(warped, fg, distractors, rng)
            out_in.append(img)
            out_seg.append(seg)
        return out_in, out_seg

    # -- item ------------------------------------------------------------------

    def __getitem__(self, idx):
        if self.deterministic:
            task_id, sample_index = self._eval_index[idx]
        else:
            # idx is a placeholder; draw a random task from the train pool.
            task_id = int(self.pool[np.random.default_rng().integers(len(self.pool))])
            sample_index = idx

        base, fg, meta = self.bank.get(task_id)
        distractors = meta["distractor_labels"]
        fg_sign = meta.get("appearance_sign")     # task-level fg intensity side
        rngs = self._subject_rngs(task_id, sample_index)

        target = self._make_subject(base, fg, distractors, rngs[0], fg_sign=fg_sign)
        # context_copy_fraction = fraction of contexts rendered as pristine exemplars
        # (near-zero deformation, fresh background) -> easier, non-degenerate.
        ctx = []
        for i in range(self.context_size):
            pristine = rngs[0].random() < self.live.context_copy_fraction
            ctx.append(self._make_subject(base, fg, distractors, rngs[1 + i],
                                          fg_sign=fg_sign,
                                          shift_scale=0.1 if pristine else 1.0))
        ctx_in, ctx_out = self._apply_context_consistency(ctx, fg, distractors, rngs[0])

        t_img, t_seg, _ = target
        return {
            "image":       _to_img_tensor(t_img),                       # [1,H,W]
            "label":       _to_img_tensor(t_seg),                       # [1,H,W]
            "context_in":  torch.stack([_to_img_tensor(x) for x in ctx_in]),   # [K,1,H,W]
            "context_out": torch.stack([_to_img_tensor(x) for x in ctx_out]),  # [K,1,H,W]
            "meta": {
                "task_id": int(task_id),
                "fg": int(fg),
                "subject_index": int(sample_index),
                "morphology": meta["morphology"],
                "difficulty": {**meta["geo_params"], **vars(self.live)},
                "axis": meta["axis_loadings"],
            },
        }
