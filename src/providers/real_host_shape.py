"""Procedural shapes stamped onto REAL host images (not synth_gmm's synthetic GMM-painted
canvas) -- see docs/logs.md 2026-09-27 "sim-to-real gap". Motivation: synth_gmm's shape-mode
host organ is drawn from the gmm_bank, which is 100% CT-sourced (verified directly: every one
of its 21 provenance tags is a CT dataset) -- so a family meant to help transfer to a REAL MRI
OOD source (ISLES22/Shifts-MS, both brain lesions) was never once trained with an MRI-looking
canvas. This module stamps the SAME procedural shapes (shapes3d.primitives, unchanged) onto a
REAL MRI TotalSeg subject's own image, at a REAL local intensity contrast (computed from that
subject's own crop, not a precomputed per-class GMM table) -- closer to the SyntheticTumors/
DiffTumor "synthetic lesion on a real scan" technique this codebase's own HeterogeneitySpec
docstring already cites as precedent, rather than a fully-synthetic canvas.

v0, deliberately simple (flat contrast fill, no texture/heterogeneity layering) to test the
core hypothesis fast, one variable at a time, before adding realism on top if it helps.
"""
import dataclasses
import random

import numpy as np
import torch
from scipy.ndimage import binary_dilation as _binary_dilation

from src.incontext_dataset_v2 import LoadRequest
from src.providers.synth_gmm import TextureSpec, _fractal_value_noise
from src.providers.totalseg import build_native_crop
from src.shapes3d.instantiate import draw_cohort_hyperparams, draw_member_shape, rasterize_shape_in_crop
from src.totalseg_dataloader_incontext import organ_crop_arrays


class RealHostShapeProvider:
    """Cohort hook (assemble_task), same contract as SynthGmmProvider(cascade=True): each
    call returns K+1 NativeCrops of the SAME procedural shape stamped into K+1 DIFFERENT
    real subjects' own images. `host_classes` may be a single class name (e.g. 'brain') or a
    list -- when a list, each COHORT (not each member) draws ONE class uniformly at random,
    then samples its K+1 subjects from that class's own pool, so a cohort's target+context
    still represent "the same task" (same organ) the way every other cohort in this codebase
    does. Added 2026-09-27 (user direction: "all real host organs instead of just brain") --
    163's brain-only result already showed the gain wasn't brain-anatomy-specific (gnc_kidney,
    a kidney source, improved as much as atlas_v2 did), so training across many real organs
    should broaden the "real MRI tissue/noise around a lesion-like region" skill further."""

    def __init__(self, totalseg_provider, shape_spec, *, host_classes="brain", context_size=2,
                epoch_length=1000, max_native=256, jitter=0, texture_spec=None,
                crop_spacing_mm=1.0):
        self.provider = totalseg_provider
        self.shape_spec = shape_spec
        self.context_size = int(context_size)
        self.epoch_length = int(epoch_length)
        self.max_native = max_native
        self.jitter = jitter
        # crop_spacing_mm: FIXED, independent of whatever spacing the training batch happens
        # to sample (train_spacing_range=[3,6] is sized for whole-body CT coverage). Verified
        # 2026-09-27 (user-flagged "heavily anisotropic" plotted cases): at that range, ALL
        # 57 real MRI 'brain' subjects clip on every axis (native brain FOV ~150-260mm vs. a
        # 384-768mm target window) -- every real-host task would be a mostly-air-padded,
        # badly distorted crop. First fix tried 1.5mm (matching ISLES22/Shifts-MS's own real
        # eval pitch) but that still left 48/57 subjects clipping on >=1 axis (still visibly
        # squished in a second visual pass, user-flagged again) -- measured directly across
        # the pitch range and 1.0mm (128mm FOV) covers 48/57 (84%) with ZERO clipping on any
        # axis vs. only 9/57 (16%) at 1.5mm, so this pool's own real geometry drives the
        # choice, not a borrowed number from a different dataset. Also consistent with the
        # earlier finding that finer spacing helps small-structure synthetic Dice generally
        # (docs/logs.md 2026-09-27 spacing sweep) -- no known downside to going finer here.
        # Reused unchanged across every host class here (not re-tuned per organ) -- the
        # per-class FOV filter below naturally drops whatever doesn't fit at this pitch,
        # rather than silently producing a bad crop.
        self.crop_spacing_mm = float(crop_spacing_mm)
        host_classes = [host_classes] if isinstance(host_classes, str) else list(host_classes)
        self.subjects_by_class = self._filter_usable_subjects(host_classes)
        # texture_spec: painted-region noise matches the REAL local tissue's own std (scaled
        # by it, not a fixed amount) so a flat/too-clean patch doesn't give the lesion away
        # for free -- user-flagged 2026-09-27 ("realistic object intensity, else too easy to
        # spot"). n_octaves<=1 (default) = plain i.i.d. noise at that scale, still textured.
        self.texture_spec = texture_spec or TextureSpec()
        self.classes = [f"shape_{f}" for f in shape_spec.family_weights]

    def _subject_fov_cache(self):
        """{subject: min-axis native FOV mm}, built from ONE `spacings.json` read (already
        has both `shape` and `spacing` per subject -- see `providers/totalseg.py::
        _load_spacings`, which only keeps `spacing`). Falls back to `load_raw` (an actual
        mmap open) only for subjects missing from that file, if any.

        Added 2026-09-27: the original per-(subject,class) `load_raw` call was fine for the
        MRI pool (50 classes, 6451 total pairs) but pathological for CT (117 classes heavily
        overlapping the SAME ~1228 subjects -- most CT subjects carry nearly every organ --
        giving ~70k redundant mmap opens, each paying real NFS latency; the smoke test never
        finished inside a 300s budget). FOV depends only on the SUBJECT's own scan geometry,
        never the class, so computing it once per unique subject (not per pair) is both
        correct and the actual fix, not just a cache-shaped workaround."""
        import json
        path = self.provider.root / "spacings.json"
        cache = {}
        if path.exists():
            with open(path) as f:
                raw = json.load(f)
            for s, m in raw.items():
                if "shape" in m and "spacing" in m:
                    cache[s] = min(d * sp for d, sp in zip(m["shape"], m["spacing"]))
        return cache

    def _filter_usable_subjects(self, host_classes):
        """Per host class, drop subjects whose native FOV is too small on any axis to cover
        this provider's own crop window at zero clipping -- a handful of genuinely degenerate/
        partial acquisitions (e.g. one 'brain' subject has only 13 native slices at 3.6mm =
        47mm total z-coverage) would otherwise get badly air-padded/squished REGARDLESS of
        crop_spacing_mm, and with context_size+1 draws per cohort the per-ROW odds of hitting
        at least one such subject is much higher than the per-subject rate suggests (verified:
        picking a finer pitch alone did not visibly fix this in a second visual pass,
        2026-09-27 -- excluding the actual bad subjects does). Classes left with too few usable
        subjects for a full cohort are dropped entirely (not an error -- with many classes some
        are expected to be too rare/small in this pool). One-time cost at init (see
        `_subject_fov_cache`), not per-item."""
        need = self.crop_spacing_mm * self.provider.T
        fov_cache = self._subject_fov_cache()
        by_class = {}
        for cls in host_classes:
            usable = []
            for s in self.provider.subjects_for(cls):
                if s in fov_cache:
                    fov_min = fov_cache[s]
                else:
                    image_np, _, native_sp, _ = self.provider.load_raw(s)
                    fov_min = min(d * sp for d, sp in zip(image_np.shape, native_sp))
                if fov_min >= need:
                    usable.append(s)
            if len(usable) >= self.context_size + 1:
                by_class[cls] = usable
            else:
                print(f"RealHostShapeProvider: dropping host class {cls!r} -- only "
                      f"{len(usable)} subjects have >= {need:.0f}mm FOV on every axis "
                      f"(need >= {self.context_size + 1})", flush=True)
        if not by_class:
            raise ValueError(
                f"none of {host_classes} have >= {self.context_size + 1} usable subjects at "
                f"crop_spacing_mm={self.crop_spacing_mm}")
        print(f"RealHostShapeProvider: {len(by_class)}/{len(host_classes)} host classes usable "
              f"({sum(len(v) for v in by_class.values())} subjects total)", flush=True)
        return by_class

    def _one_member(self, subject, host_cls, cohort_hp, crop_spacing_mm, rng, contrast_ratio,
                    gmm_seed, member_idx, center=None):
        """Builds one member's NativeCrop. Fully reproducible from
        (subject, host_cls, cohort_hp, gmm_seed, member_idx) alone -- `member_nrng` is
        constructed fresh here (not passed in) so a cascade re-crop (`load_native_crop`,
        different call, possibly a different `center`) draws the IDENTICAL member_draw/
        rasterization/texture-noise as the original `assemble_task` call. Mirrors
        `SynthGmmProvider._build_nc`'s own `member_nrng = default_rng([gmm_seed,
        member_idx])` pattern exactly.

        `center`: None (assemble_task, level-0) -> the crop window is centered on the
        shape's own anchor (see below). An explicit value (load_native_crop, cascade
        re-crop) -> the crop window looks there instead, but the shape's true position
        does NOT move with it -- it stays anchored to the host class's own COM, always
        resolved fresh with center_mode="com" regardless of `center`, exactly like
        SynthGmmProvider's `fallback` anchor (see its _build_nc docstring). This is what
        makes the shape the SAME physical object at every cascade level.
        """
        member_nrng = np.random.default_rng([int(gmm_seed), int(member_idx)])
        member_draw = draw_member_shape(member_nrng, cohort_hp, self.shape_spec)

        image_np, label_np, native_sp, norm = self.provider.load_raw(subject)
        anchor_req = LoadRequest(rng=rng, crop_spacing_mm=crop_spacing_mm, center_mode="com")
        shape_anchor = self.provider.resolve_center(subject, host_cls, anchor_req, label_np)
        crop_center = center if center is not None else shape_anchor
        jitter = 0 if center is not None else self.jitter   # exact predicted center on re-crop
        crop_ct_view, crop_lbl_view, out_sizes, pad_lo, geom = organ_crop_arrays(
            image_np, label_np, crop_center, list(native_sp),
            image_size=(self.provider.T,) * 3, crop_mm=crop_spacing_mm,
            jitter=jitter, rng=rng)

        step = (1, 1, 1)
        if self.max_native and max(crop_lbl_view.shape) > self.max_native:
            step = tuple(-(-s // self.max_native) for s in crop_lbl_view.shape)
            crop_ct_view = crop_ct_view[::step[0], ::step[1], ::step[2]]
            crop_lbl_view = crop_lbl_view[::step[0], ::step[1], ::step[2]]
        crop_ct = np.array(crop_ct_view, dtype=np.float32)          # writable copy
        crop_lbl = np.zeros(crop_lbl_view.shape, dtype=np.uint8)    # discard the real label

        spacing = np.asarray(native_sp, dtype=np.float64)
        mm_per_voxel = tuple((spacing * np.asarray(step, dtype=np.float64)).tolist())
        starts = geom[0].numpy().astype(np.float64)
        # Anchored to shape_anchor (level-invariant), NOT crop_center (which moves with the
        # cascade-predicted re-crop) -- see docstring above.
        center_native = (np.asarray(shape_anchor, dtype=np.float64)
                         + np.asarray(member_draw.position_offset_mm, dtype=np.float64) / spacing)
        center_local = (center_native - starts) / np.asarray(step, dtype=np.float64)
        rasterize_shape_in_crop(crop_lbl, 1, member_draw, mm_per_voxel, center_local, rng=member_nrng)

        shape_mask = crop_lbl == 1
        if shape_mask.any():
            # RING local stats (not the whole crop, which mixes in background air / distant
            # tissue and would deflate both the contrast baseline and the noise scale) --
            # mirrors analyze_target_surround_contrast.py's own real-lesion measurement
            # methodology, so the contrast_ratio drawn from host_contrast_ratio_range means
            # the same thing here as it does there.
            ring_vox = max(1, int(round(6.0 / float(np.mean(mm_per_voxel)))))
            ring = _binary_dilation(shape_mask, iterations=ring_vox) & ~shape_mask
            bg = crop_ct[ring] if ring.sum() >= 30 else crop_ct[~shape_mask]
            local_mean, local_std = float(bg.mean()), float(bg.std() + 1e-6)
            # Texture noise (scaled by the REAL local std, not painted flat) so the stamped
            # region isn't suspiciously smooth against real MRI's natural noise -- user-
            # flagged 2026-09-27 ("realistic object intensity, else too easy to spot").
            # member_nrng (not a fresh ambient draw) -- reproducible across cascade levels.
            noise = _fractal_value_noise(crop_ct.shape, member_nrng, mm_per_voxel, self.texture_spec)
            crop_ct[shape_mask] = local_mean + contrast_ratio * local_std + local_std * noise[shape_mask]

        return build_native_crop(crop_ct, crop_lbl, 1, out_sizes, pad_lo, geom,
                                 crop_spacing_mm=float(crop_spacing_mm), norm=norm,
                                 modality=self.provider.modality, max_native=None)

    def assemble_task(self, rng, crop_spacing_mm):
        # `crop_spacing_mm` (the caller's batch-sampled spacing, e.g. from train_spacing_range)
        # is IGNORED on purpose -- see self.crop_spacing_mm's own docstring above.
        crop_spacing_mm = self.crop_spacing_mm
        family = rng.choices(list(self.shape_spec.family_weights),
                             weights=list(self.shape_spec.family_weights.values()))[0]
        spec = self.shape_spec
        gmm_seed = rng.getrandbits(64)
        family_spec = dataclasses.replace(spec, family_weights={family: 1.0})
        cohort_hp = draw_cohort_hyperparams(np.random.default_rng([int(gmm_seed), 999]), family_spec)
        contrast_ratio = random.Random(gmm_seed).uniform(*spec.host_contrast_ratio_range)

        # One host class per COHORT (not per member) -- target+context still represent "the
        # same task" (same organ), matching how every other cohort in this codebase works.
        host_cls = rng.choice(list(self.subjects_by_class))
        chosen = rng.sample(self.subjects_by_class[host_cls], self.context_size + 1)
        ncs = [self._one_member(subj, host_cls, cohort_hp, crop_spacing_mm, rng,
                                contrast_ratio, gmm_seed, i)
               for i, subj in enumerate(chosen)]

        # subject strings: "<real subject id>|<gmm_seed>|<member_idx>|host<host_cls>|<modality>"
        # -- EVERY member (target AND context) carries host_cls + modality (not just the
        # target), fixed 5-field format, mirroring SynthGmmProvider's own `|host<id>` suffix
        # convention -- required for load_native_crop (cascade re-crop) to reconstruct the
        # exact same cohort/member draw for ANY member, not just the target.
        tag = f"host{host_cls}|{self.provider.modality}"
        return {
            "native_crop": ncs,
            "subject": f"{chosen[0]}|{gmm_seed}|0|{tag}",
            "context_subjects": [f"{s}|{gmm_seed}|{i + 1}|{tag}" for i, s in enumerate(chosen[1:])],
            "label_name": f"shape_{family}",
            "aug_mode": torch.tensor(0, dtype=torch.long),
            # "synth", NOT self.provider.modality ("mri"/"ct") -- tgt/ctx_modality's actual
            # purpose is CASCADE RE-CROP ROUTING (totalseg_dataloader_incontext.py:1616,
            # TriSourceProvider.load_native_crop), which branches purely on
            # modality=="synth" -> self.synth (SynthGmmProvider, which dispatches into
            # real_host_providers) vs anything else -> self.multi (the REAL totalseg
            # provider, which has no idea what a real-host encoded subject string is).
            # This task IS reached via that synth branch (TriSourceProvider.assemble_task's
            # p_synth coin flip -> self.synth.assemble_task -> real_host_providers dispatch),
            # so it must report itself the same way SynthGmmProvider's own tasks do, or
            # every cascade re-crop of it gets routed to the wrong provider and crashes
            # (confirmed: FileNotFoundError trying to open a literal encoded subject string
            # as a directory). The REAL per-member modality isn't lost -- it's still on each
            # NativeCrop.modality (build_native_crop, unchanged) and in the subject string's
            # own trailing tag (MixtureShapeProvider.load_native_crop's dispatch key).
            "tgt_modality": "synth", "ctx_modality": "synth",
        }

    def load_native_crop(self, subject, cls, req):
        """Cascade re-crop: re-derive the same cohort/member draw from the encoded subject
        string (see assemble_task's tag format). `req.center` may be a cascade-predicted
        center or None (same-level reconstruction) -- either way it only affects WHERE the
        crop window looks; the shape's own position stays anchored to the host class's
        fixed COM, independent of it -- see _one_member's docstring. Mirrors
        SynthGmmProvider.load_native_crop's contract.

        `req.crop_spacing_mm` (whatever the calling cascade level's own ladder pitch is,
        e.g. 4mm/1.5mm) is IGNORED, same as assemble_task's own caller-spacing param --
        see self.crop_spacing_mm's docstring. A real-host task always renders at this
        provider's own fixed pitch regardless of which cascade level is re-cropping it."""
        subj, gmm_seed_str, member_idx_str, host_part, modality = subject.split("|")
        assert host_part.startswith("host"), f"malformed real-host subject string: {subject!r}"
        host_cls = host_part[len("host"):]
        gmm_seed = int(gmm_seed_str)
        member_idx = int(member_idx_str)
        family = cls[len("shape_"):] if cls.startswith("shape_") else cls

        spec = self.shape_spec
        family_spec = dataclasses.replace(spec, family_weights={family: 1.0})
        cohort_hp = draw_cohort_hyperparams(np.random.default_rng([int(gmm_seed), 999]), family_spec)
        contrast_ratio = random.Random(gmm_seed).uniform(*spec.host_contrast_ratio_range)
        return self._one_member(subj, host_cls, cohort_hp, self.crop_spacing_mm, req.rng,
                                contrast_ratio, gmm_seed, member_idx, center=req.center)


class MixtureShapeProvider:
    """Cohort hook that delegates each `assemble_task` call to ONE child provider, drawn
    per-COHORT (not per-member -- a cohort's target+context always come from the same child,
    same "same task" convention every other cohort-consistency mechanism in this codebase
    follows). Added 2026-09-27 (user direction: "include CT subjects for synthetic task
    painting") to combine a REAL-MRI-hosted `RealHostShapeProvider` with a REAL-CT-hosted one
    under a single `real_host_providers[family]` entry -- `SynthGmmProvider.assemble_task`
    only ever calls one provider per family, so this is the seam that lets two (or more) real
    hosts of different modalities share that one slot."""

    def __init__(self, providers, weights=None):
        self.providers = list(providers)
        self.weights = list(weights) if weights is not None else [1.0] * len(self.providers)
        assert len(self.providers) == len(self.weights) and self.providers
        self.epoch_length = max(getattr(p, "epoch_length", 1000) for p in self.providers)
        seen = []
        for p in self.providers:
            for c in getattr(p, "classes", []):
                if c not in seen:
                    seen.append(c)
        self.classes = seen

    def assemble_task(self, rng, crop_spacing_mm):
        provider = rng.choices(self.providers, weights=self.weights)[0]
        return provider.assemble_task(rng, crop_spacing_mm)

    def load_native_crop(self, subject, cls, req):
        """Cascade re-crop: dispatch to whichever child provider originally produced this
        member, recovered from the modality tag `assemble_task` embeds in every subject
        string (last '|'-field, e.g. 'mri'/'ct') -- NOT re-drawn via `rng.choices` (that
        would be a fresh, unrelated coin flip, not necessarily the same child)."""
        modality = subject.rsplit("|", 1)[-1]
        for p in self.providers:
            if getattr(p.provider, "modality", None) == modality:
                return p.load_native_crop(subject, cls, req)
        raise ValueError(f"MixtureShapeProvider.load_native_crop: no child provider for "
                         f"modality {modality!r} (subject={subject!r})")
