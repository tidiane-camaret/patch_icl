"""
Cohort sampler over the compact GMM mask bank (built by build_gmm_mask_bank.py). A cohort
= K+1 "similar" masks that all contain a target class, drawn to be as alike as possible on
three axes at once: body-region span, spacing/FOV, and an all-label by-class-size vector. The three
are fused into one weighted distance; kNN around a random anchor gives the cohort, so pools
never come up empty (unlike hard filters). "Similar masks + one shared GMM draw" = one
scanner/patient family for in-context training.

Mask randomization (per-mask erode/dilate/warp) is a later stage; this only selects.
"""
import pickle
import random
from pathlib import Path

import numpy as np


class CohortSampler:
    # coverage-proxy label ids excluded from the by-class-size distance: 0=air, 200=body
    # envelope. Both encode how much of the FOV is filled, not organ composition (air/fg is
    # 72% of the raw by-class-size L1; body is 43-85% of foreground) — see
    # analyze_cohort_distance.py / docs/logs.md. by_class_size_common_frac then restricts to
    # organs present in >= that fraction of masks (drops annotation presence/absence confound).
    _COVERAGE_IDS = (0, 200)

    def __init__(self, bank_dir, k, w_span=1.0, w_fov=0.02, w_spacing=0.3, w_by_class_size=3.0,
                 randomness=0.0, min_masks_per_class=None, by_class_size_common_frac=0.75,
                 by_class_size_mode="fraction", class_balanced=True):
        """k = context_size (cohort draws k+1 masks). Distance weights fuse region-span
        (idx units), FOV (mm), spacing (mm), and by-class organ-size L1.

        `randomness` is the PER-COHORT diversity dial r in [0,1] (drawn each cohort). r maps
        to the width of the nearest-neighbour pool the cohort is sampled from:
          M = round((k+1) + r*(pool-(k+1)));  r=0 -> the k+1 NEAREST (tight),
          r=1 -> uniform-random from the whole class pool. It accepts:
            float x            -> fixed tightness r=x every cohort;
            [lo, hi]           -> r ~ U[lo,hi] per cohort (continuous spread);
            {p_tight,tight,loose} -> bimodal: prob p_tight -> r=tight (def 0.0),
                                     else r=loose (def 1.0) — the "some similar, some
                                     full random" mixture (sweep p_tight for ablations).

        by_class_size_common_frac restricts the by-class-size vector to organs present in >=
        that fraction of masks (0 = keep all organs), after dropping air+body coverage proxies.
        by_class_size_mode selects what the by-class-size vector measures:
          'fraction' — organ voxel counts renormalized to sum 1 (relative composition,
                        SCALE-INVARIANT: same proportions -> same vector regardless of body size).
          'volume'   — organ physical volume in LITRES (counts × prod(spacing)); scale-AWARE
                        (separates large vs thin patients), big-organ-weighted. Needs a smaller
                        w_by_class_size (~0.5) as its L1 is ~5 L vs ~0.5 for fraction."""
        self.dir = Path(bank_dir)
        with open(self.dir / "index.pkl", "rb") as f:
            idx = pickle.load(f)
        self.k = int(k)
        self.w = dict(span=w_span, fov=w_fov, spacing=w_spacing, by_class_size=w_by_class_size)
        self.randomness = randomness
        # class_balanced picks the target class UNIFORMLY over usable classes (rare organs seen
        # as often as common ones — matches totalseg data.class_balanced); False weights the
        # pick by how many bank masks contain the class (the natural anatomical-frequency prior,
        # body/liver/lungs dominate). Weights built after self.classes below.
        self.class_balanced = bool(class_balanced)

        # drop masks with a degenerate spacing axis (corrupt source affines: exact-0 -> ZeroDivision
        # in organ_crop_arrays, and near-0 like 3e-8 -> giant target_size). 0.05mm floor is well
        # below any real CT (keeps legit 0.1-0.25mm high-res). Keep size_mat row-aligned.
        entries = idx["entries"]
        size_mat = np.asarray(idx["size_mat"], np.float32)
        good = np.array([all(s >= 0.05 for s in e["spacing"]) for e in entries], bool)
        if not good.all():
            print(f"CohortSampler: dropping {int((~good).sum())} masks with bad spacing "
                  f"({[entries[i]['file'] for i in np.where(~good)[0]]})", flush=True)
        self.entries = [e for e, g in zip(entries, good) if g]
        size_mat = size_mat[good]
        n = len(self.entries)

        # per-mask arrays for vectorized distance
        self.span = np.array([e["span"] for e in self.entries], np.float32)          # (N,2)
        self.spacing = np.array([e["spacing"] for e in self.entries], np.float32)     # (N,3)
        self.dim = np.array([e["dim"] for e in self.entries], np.float32)             # (N,3)
        self.fov = self.dim * self.spacing                                            # (N,3) mm

        # class → mask indices that contain it
        self.cls2masks: dict[int, list[int]] = {}
        for i, e in enumerate(self.entries):
            for l in e["label_list"]:
                self.cls2masks.setdefault(l, []).append(i)
        need = min_masks_per_class if min_masks_per_class is not None else self.k + 1
        self.classes = [c for c, m in self.cls2masks.items() if len(m) >= need]

        # --- clean by-class-size matrix (drop coverage proxies, restrict to shared core) — the
        # shared core is computed HERE from this bank, so the full 5164 bank gets its own core
        # rather than a hardcoded 500-subset one. ---
        bcs = size_mat.copy()                                          # (N, maxid) fg-normalized
        keep = np.ones(bcs.shape[1], bool)
        keep[list(self._COVERAGE_IDS)] = False                         # drop air + body
        if by_class_size_common_frac > 0:
            freq = np.zeros(bcs.shape[1], int)
            for e in self.entries:
                for l in e["label_list"]:
                    freq[l] += 1
            keep &= freq >= by_class_size_common_frac * n              # organs in >= frac of masks
        if by_class_size_mode == "volume":
            # absolute physical volume (L): counts = size_vec·fg, fg = prod(dim)/(1+air_frac)
            # (recoverable from index → no rebuild); vol = counts · prod(spacing). No renorm.
            fg = self.dim.prod(1) / (1.0 + bcs[:, 0])                  # (N,) foreground voxels
            vox_l = self.spacing.prod(1) / 1e6                        # mm³/voxel → litres
            self.by_class_size_mat = bcs * (fg * vox_l)[:, None] * keep[None, :]
        elif by_class_size_mode == "fraction":
            bcs = bcs * keep[None, :]
            row = bcs.sum(1, keepdims=True)                            # renorm kept cols → sum 1
            self.by_class_size_mat = np.divide(bcs, row, out=np.zeros_like(bcs), where=row > 0)
        else:
            raise ValueError(f"by_class_size_mode {by_class_size_mode!r} (fraction | volume)")
        self.by_class_size_ncols = int(keep.sum())

    def _dist(self, anchor, pool):
        """Weighted distance from `anchor` to every mask index in `pool` (np array)."""
        d_span = np.abs(self.span[pool] - self.span[anchor]).sum(1)
        d_fov = np.abs(self.fov[pool] - self.fov[anchor]).sum(1)
        d_sp = np.abs(self.spacing[pool] - self.spacing[anchor]).sum(1)
        d_bcs = np.abs(self.by_class_size_mat[pool] - self.by_class_size_mat[anchor]).sum(1)
        return (self.w["span"] * d_span + self.w["fov"] * d_fov
                + self.w["spacing"] * d_sp + self.w["by_class_size"] * d_bcs)

    def _draw_r(self, rng: random.Random) -> float:
        """Per-cohort randomness r in [0,1] from the `randomness` spec (float | [lo,hi] |
        {p_tight,tight,loose})."""
        s = self.randomness
        if isinstance(s, dict):
            tight, loose = float(s.get("tight", 0.0)), float(s.get("loose", 1.0))
            return tight if rng.random() < float(s.get("p_tight", 0.5)) else loose
        if isinstance(s, (list, tuple)):
            return rng.uniform(float(s[0]), float(s[1]))
        return float(s)

    def sample_cohort(self, rng: random.Random | None = None, target_class: int | None = None):
        """Return (target_class, [k+1 entry dicts]) — the cohort. rng defaults to `random`.
        Tightness is drawn per cohort via `_draw_r`: r=0 -> k+1 nearest, r=1 -> uniform random."""
        rng = rng or random
        if target_class is not None:
            cls = target_class
        elif self.class_balanced:
            cls = rng.choice(self.classes)                         # uniform over usable classes
        else:
            # natural-frequency prior: weight by #bank masks holding the class. Weights built
            # here (not cached) so a post-init self.classes filter can't stale them.
            cls = rng.choices(self.classes,
                              weights=[len(self.cls2masks[c]) for c in self.classes])[0]
        pool = np.array(self.cls2masks[cls])
        if len(pool) <= self.k + 1:
            chosen = pool.tolist()
        else:
            r = self._draw_r(rng)
            anchor = int(rng.choice(pool))
            # width of the nearest-neighbour band we draw the cohort from: r scales it from
            # the k+1 tightest up to the whole class pool (uniform random at r=1).
            m = round((self.k + 1) + r * (len(pool) - (self.k + 1)))
            m = int(min(max(m, self.k + 1), len(pool)))
            order = pool[np.argsort(self._dist(anchor, pool), kind="stable")]
            cand = [i for i in order[:m].tolist() if i != anchor]   # anchor (dist 0) always in band
            chosen = [anchor] + rng.sample(cand, self.k)            # k more -> k+1 total
        return cls, [self.entries[i] for i in chosen]

    def cohort_stats(self, cls, cohort):
        """Diagnostic spread within a cohort (lower = tighter)."""
        idx = [self.entries.index(e) for e in cohort]
        return {
            "n": len(idx), "class": cls,
            "span_set": sorted({tuple(self.span[i].astype(int)) for i in idx}),
            "fov_mm_std": float(self.fov[idx].std(0).mean()),
            "spacing_std": float(self.spacing[idx].std(0).mean()),
            "by_class_size_L1_meanpair": float(np.abs(
                self.by_class_size_mat[idx][:, None] - self.by_class_size_mat[idx][None]).sum(-1).mean()),
        }
