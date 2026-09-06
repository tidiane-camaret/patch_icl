"""Multi-source in-context provider (v2 cohort hook).

Composes two modality-locked TotalSegProviders into one provider that draws, per
task, a modality REGIME (all-source-0 / all-source-1 / forced cross), then draws a
class that regime can serve WITHOUT fallback (regime-conditional class draw), and
loads the K+1 cases accordingly. The per-slot modality fallback (a class with no
subjects in the wanted modality) remains only as a rare safety net. Implements the
engine's `assemble_task` hook (like
src/providers/synth_gmm.py), so InContextDataset owns only aug + the per-item RNG.
"""
import warnings

import torch

from src.incontext_dataset_v2 import LoadRequest


class MultiSourceProvider:
    """Cohort-hook provider over exactly two modality-locked sub-providers."""

    def __init__(self, sub_providers, *, context_size, regime_p=(1 / 3, 1 / 3, 1 / 3),
                 epoch_length=1000):
        if len(sub_providers) != 2:
            raise ValueError(f"MultiSourceProvider expects exactly 2 sub-providers, "
                             f"got {list(sub_providers)}")
        self.subs = dict(sub_providers)          # insertion order defines m0, m1
        self._mods = list(self.subs)
        self.context_size = int(context_size)
        self.regime_p = tuple(float(x) for x in regime_p)
        if len(self.regime_p) != 3:
            raise ValueError(f"regime_p needs 3 entries (m0, m1, cross), got {regime_p}")
        self.epoch_length = int(epoch_length)

        all_classes = set()
        for p in self.subs.values():
            all_classes.update(p.classes)
        self._avail = {}
        for c in sorted(all_classes):
            mods = [m for m, p in self.subs.items() if p.subjects_for(c)]
            if mods:
                self._avail[c] = mods
        self.classes = list(self._avail)
        if not self.classes:
            raise ValueError("MultiSourceProvider: no class has subjects in any sub-provider")
        # Regime-conditional class pools: each regime draws only from classes it can
        # serve without fallback (m -> classes with m subjects; "cross" -> both-modality).
        self._classes_by_mod = {m: [c for c in self.classes if m in self._avail[c]]
                                for m in self._mods}
        self._both_classes = [c for c in self.classes if len(self._avail[c]) == 2]

    # --- VolumeProvider protocol stubs (engine uses assemble_task in cohort mode) ---
    def subjects_for(self, cls):
        return []

    def load(self, *a, **k):
        raise RuntimeError("MultiSourceProvider is a cohort provider; use assemble_task")

    # --- helpers ---
    def _draw_subjects(self, rng, mod, cls, n):
        """`n` distinct subjects for (mod, cls); repeat with a warning if the pool is short."""
        pool = list(self.subs[mod].subjects_for(cls))
        rng.shuffle(pool)
        if len(pool) >= n:
            return pool[:n]
        warnings.warn(
            f"MultiSourceProvider: only {len(pool)} {mod} subject(s) for {cls!r}, "
            f"need {n}; repeating (metrics leakage-inflated).", stacklevel=2)
        out = list(pool)
        while len(out) < n:
            out.append(pool[len(out) % len(pool)])
        return out

    # --- cohort hook ---
    def assemble_task(self, rng, crop_spacing_mm):
        m0, m1 = self._mods
        regime = rng.choices([m0, m1, "cross"], weights=self.regime_p, k=1)[0]

        # Draw the class from this regime's no-fallback pool, so the genuine
        # cross-modality rate ~= regime_p[cross] (a uniform pre-regime draw
        # collapsed ~90% of `cross` tasks to pure-CT). Empty pool (no both-modality
        # class at all / a modality with no classes) -> uniform draw + the per-slot
        # fallback below.
        pool = self._both_classes if regime == "cross" else self._classes_by_mod[regime]
        cls = rng.choice(pool) if pool else rng.choice(self.classes)
        avail = self._avail[cls]

        if regime == "cross":
            tgt_mod = rng.choice(avail)
            other = m1 if tgt_mod == m0 else m0
            ctx_mod = other if other in avail else tgt_mod
        else:
            tgt_mod = regime if regime in avail else avail[0]
            ctx_mod = tgt_mod

        k = self.context_size
        if tgt_mod == ctx_mod:
            subs = self._draw_subjects(rng, tgt_mod, cls, k + 1)
            tgt_subj, ctx_subjs = subs[0], subs[1:]
        else:
            tgt_subj = self._draw_subjects(rng, tgt_mod, cls, 1)[0]
            ctx_subjs = self._draw_subjects(rng, ctx_mod, cls, k)

        def _load(mod, subj):
            return self.subs[mod].load(
                subj, cls, LoadRequest(rng=rng, crop_spacing_mm=float(crop_spacing_mm)))

        tgt = _load(tgt_mod, tgt_subj)
        ctx = [_load(ctx_mod, s) for s in ctx_subjs]

        return {
            "image": tgt.image,
            "label": tgt.label,
            "context_in": torch.stack([r.image for r in ctx]),
            "context_out": torch.stack([r.label for r in ctx]),
            "spacing": tgt.spacing,
            "crop_geom": tgt.crop_geom,
            "subject": tgt_subj,
            "context_subjects": list(ctx_subjs),
            "label_name": cls,
            "modality": tgt_mod,
            "aug_mode": torch.tensor(0, dtype=torch.long),
            # `regime` is the pre-fallback draw; `fallback` flags a "cross" draw that
            # landed on a single-modality class and so collapsed to same-modality.
            "meta": {"regime": regime, "tgt_mod": tgt_mod, "ctx_mod": ctx_mod,
                     "fallback": bool(regime == "cross" and tgt_mod == ctx_mod)},
        }
