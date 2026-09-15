"""GMM-synth cohort provider for the in-context dataloader v2.

Two modes:
  cascade=False (default): legacy path — wraps SynthGmmMaisiDataset and drives its
    assemble() from the engine's RNG. Used when synth is the sole source.
  cascade=True: builds NativeCrop payloads (same format as TotalSegProvider) so the
    provider plugs into the cascade train loop alongside real sources. Used by
    TriSourceProvider in multisource+synth runs (data.source=multisource, data.p_synth>0).

GMM consistency across cascade levels: the GMM seed is embedded in the subject string
as "<filename>|<gmm_seed>|<member_idx>". load_native_crop re-derives the same cohort-
shared mu/sd from the seed, and the per-member paint nrng from (gmm_seed, member_idx),
giving an identical color palette at every cascade level for the same member.

Shape mode (p_shape > 0, cascade=True only): a `p_shape` coin flip per cohort swaps the
sampled cohort's target for a procedurally generated geometric shape (blob/splatter/
disk/cylinder, see src/shapes3d/) stamped into the sampled hosts' real anatomy under a
new pseudo-class id (data.maisi_classes.SHAPE_ID_TO_FAMILY). The host's real class
still drives crop placement (center resolution against real anatomy); only the painted
target and its label change. Shape size/position are physical (mm), anchored to the
host's own precomputed centroid (`e["cents"]`, level-invariant) rather than the
resolved crop center or any crop-relative fraction -- so the SAME shape (same physical
size and position) is what every cascade level re-crops toward, however that level's
`crop_spacing_mm` differs. See docs/superpowers/specs/
2026-09-14-cohort-consistent-synthetic-shapes-design.md and docs/logs.md 2026-09-15
(world-space redesign, superseding the initial crop-relative one).
"""
from types import SimpleNamespace

import numpy as np
import torch

from data.maisi_classes import MAISI_CLASS_TO_IDX, MAISI_IDX_TO_CLASS, SHAPE_ID_TO_FAMILY
from src.gpu_gmm_intensity import sample_grouped_uniform
from src.providers.totalseg import _resolve_center
from src.shapes3d.instantiate import draw_cohort_hyperparams, draw_member_shape, rasterize_shape_in_crop
from src.shapes3d.spec import ShapeCohortSpec
from src.totalseg_dataloader_incontext import organ_crop_arrays

# Sentinel second seed-key for the cohort-level shape hyperparameter draw (vs. real
# member indices 0, 1, 2, ...) -- np.random.default_rng's SeedSequence coerces every
# entry to uint32, so this must be a valid non-negative int, not -1.
_SHAPE_COHORT_HP_SEED_KEY = 2**32 - 1


class SynthGmmProvider:
    """Cohort-hook provider wrapping a SynthGmmMaisiDataset for InContextDataset."""

    def __init__(self, dataset, *, cascade=False, p_shape=0.0, shape_spec=None):
        self.ds = dataset
        self.epoch_length = len(dataset)
        self.classes = [MAISI_IDX_TO_CLASS.get(c, str(c)) for c in dataset.cs.classes]
        self.cascade = cascade
        self.p_shape = float(p_shape)
        self.shape_spec = shape_spec or ShapeCohortSpec()
        if cascade:
            self._entry_by_file = {e["file"]: e for e in dataset.cs.entries}
            from src.providers.totalseg import NativeCrop
            from src.totalseg_dataset import CtNormSpec
            # Synth images are already in (val - GMM_MEAN) / GMM_STD space, roughly
            # [-1.7, 1.7]. The pass-through norm skips HU-clip+z-score in _realize_member;
            # encoder_input_norm=instance handles per-volume normalization downstream.
            self._SYNTH_NORM = CtNormSpec(clip_lo=-3.0, clip_hi=3.0, mean=0.0, std=1.0)
            self._NativeCrop = NativeCrop

    # --- VolumeProvider protocol stubs ---
    def subjects_for(self, cls):
        return []

    # --- helpers ---
    def _draw_gmm(self, gmm_seed):
        """Re-derive cohort-shared (mu, sd) from seed. Same result as at assemble_task L0."""
        nrng = np.random.default_rng(int(gmm_seed))
        n = self.ds.maxid + 1
        mu = np.empty(n, dtype=np.float32)
        if self.ds.mu_group_ids:
            mu[1:] = sample_grouped_uniform(n - 1, 0.0, 255.0, self.ds.mu_group_ids,
                                            self.ds.mu_group_rho, nrng)
        else:
            mu[1:] = nrng.uniform(0.0, 255.0, size=n - 1)
        sd = np.sqrt(nrng.uniform(0.0, self.ds.var_max, size=n)).astype(np.float32)
        if self.ds.bg_mode == "zero":
            mu[0] = 0.0; sd[0] = 0.0
        else:
            mu[0] = nrng.uniform(0.0, 15.0); sd[0] = 0.5 ** 0.5
        return mu, sd

    def _build_nc(self, e, cls_id, rng, crop_mm, mu, sd, gmm_seed, member_idx, *,
                  center=None, jitter=None, center_mode="com",
                  shape_hp=None, shape_id=None):
        """Crop + paint one MAISI bank entry → NativeCrop. cascade=True required.

        Shape mode (shape_hp/shape_id both set): `cls_id` still drives crop placement
        against `e`'s real anatomy (the host); the painted/returned class becomes
        `shape_id` instead."""
        # per-member nrng keyed to (gmm_seed, member_idx): reproducible at L1 recrops
        member_nrng = np.random.default_rng([int(gmm_seed), int(member_idx)])
        n = self.ds.maxid + 1
        mu_e = (mu + self.ds.between_ratio * sd * member_nrng.standard_normal(n).astype(np.float32)
                if self.ds.between_ratio is not None else mu)

        arr = np.squeeze(np.load(self.ds.cs.dir / "masks" / e["file"], mmap_mode="r"))
        # centroid fallback loaded before center resolution -- random_fg needs `arr` to draw
        # a voxel from (mirrors providers/totalseg.py::_resolve_center; same helper, reused).
        cents = e["cents"].get(cls_id)
        fallback = tuple(cents[:3]) if cents is not None else None
        # fg_samples (precomputed by add_fg_samples_to_bank.py): a bounded per-class voxel
        # subset that lets random_fg draw in O(1) instead of scanning the mmap'd `arr` --
        # see docs/logs.md. None on an un-augmented bank -- _resolve_center falls back to
        # the live scan exactly as before.
        fg_samples = e.get("fg_samples", {}).get(cls_id)
        center = _resolve_center(SimpleNamespace(center=center, center_mode=center_mode, rng=rng),
                                  arr, cls_id, fallback, fg_samples=fg_samples)
        if jitter is None:
            jitter = self.ds.jitter
        _, crop_lbl, out_sizes, pad_lo, geom = organ_crop_arrays(
            arr, arr, center, list(e["spacing"]),
            image_size=(self.ds.T,) * 3, crop_mm=crop_mm, jitter=jitter, rng=rng)
        # Cap the native crop BEFORE the expensive paint (RNG draw + antialias + occupancy
        # resample, all at native res -- see docs/logs.md / memory project_synth_gmm_paint_perf):
        # a wide-FOV level-0 crop can be ~400M-1.9B native voxels here, uncapped.
        # crop_lbl is still a LAZY mmap view here (organ_crop_arrays only slices, doesn't
        # copy) -- stride-slice it BEFORE materializing. Measured on the real gmm_bank: an
        # earlier version of this cap called np.ascontiguousarray on the FULL native crop
        # first and downsampled after, which pays the full materialize cost regardless of
        # any cap (~1.3s of pure memcpy for a 512^3 uint8 crop, vs ~0.1s when the stride
        # happens first on the still-lazy view) -- effectively a no-op cap. `out_sizes`/
        # `pad_lo` describe the TARGET grid placement and don't depend on the source array's
        # shape, so this is a pure cost cut, not a correctness change.
        cap = self.ds.gpu_realize_max_native
        step = (1, 1, 1)
        if cap and max(crop_lbl.shape) > cap:
            step = tuple(-(-s // cap) for s in crop_lbl.shape)   # ceil division
            crop_lbl = crop_lbl[::step[0], ::step[1], ::step[2]]
        crop_lbl = np.ascontiguousarray(crop_lbl, dtype=np.uint8)

        target_cls = cls_id
        if shape_hp is not None:
            member_draw = draw_member_shape(member_nrng, shape_hp, self.shape_spec)
            # np.ascontiguousarray above is a no-op (returns the SAME buffer) whenever
            # crop_lbl is already C-contiguous uint8 -- which includes the common case
            # where the gpu_realize_max_native cap didn't fire, leaving crop_lbl a
            # zero-copy view of the mmap_mode="r" bank file. rasterize_shape_in_crop
            # writes in place, so force an actual copy first if that view is read-only.
            if not crop_lbl.flags.writeable:
                crop_lbl = crop_lbl.copy()
            # World-space anchor: `fallback` (e["cents"][cls_id], the host's own
            # precomputed centroid) is level-invariant -- unlike the resolved `center`
            # above, which can be a live random_fg draw or a cascade-predicted center
            # and so is NOT guaranteed identical across cascade levels. Anchoring the
            # shape's position to `fallback` instead (plus a fixed per-member mm
            # offset) is what makes the shape the SAME physical object at every level.
            spacing = np.asarray(e["spacing"], dtype=np.float64)
            center_native = np.asarray(fallback if fallback is not None else (0.0, 0.0, 0.0),
                                       dtype=np.float64)
            center_native = center_native + (
                np.asarray(member_draw.position_offset_mm, dtype=np.float64) / spacing)
            starts = geom[0].numpy().astype(np.float64)
            center_local = (center_native - starts) / np.asarray(step, dtype=np.float64)
            mm_per_voxel = tuple((spacing * np.asarray(step, dtype=np.float64)).tolist())
            rasterize_shape_in_crop(crop_lbl, shape_id, member_draw, mm_per_voxel,
                                    center_local, member_nrng)
            target_cls = shape_id
            if self.shape_spec.intensity_between_ratio is not None:
                if mu_e is mu:          # avoid mutating the cohort-shared mu array in place
                    mu_e = mu_e.copy()
                fresh_noise = member_nrng.standard_normal()
                mu_e[shape_id] = (mu[shape_id] + self.shape_spec.intensity_between_ratio
                                  * sd[shape_id] * fresh_noise)

        img, mask = self.ds._resample_paint_mask(
            crop_lbl, out_sizes, pad_lo, target_cls, mu_e, sd, member_nrng)

        T = self.ds.T
        return self._NativeCrop(
            image=img[0].half(),
            label_frac=mask.float().half(),
            class_idx=target_cls,
            has_fg=bool(mask.any()),
            out_sizes=[T, T, T],
            pad_lo=[0, 0, 0],
            crop_geom=geom,
            crop_spacing_mm=float(crop_mm),
            decim=(1, 1, 1),
            modality="synth",
            norm=self._SYNTH_NORM,
        )

    # --- cohort hook ---
    def assemble_task(self, rng, crop_spacing_mm):
        """Engine cohort hook: build one in-context item from the engine's per-item RNG."""
        if not self.cascade:
            nrng = np.random.default_rng(rng.getrandbits(64))
            return self.ds.assemble(rng, nrng, float(crop_spacing_mm))

        # cascade mode: return native_crop payload compatible with native_crop_collate_fn
        gmm_seed = rng.getrandbits(64)
        mu, sd = self._draw_gmm(gmm_seed)
        host_cls_id, cohort = self.ds.cs.sample_cohort(rng)

        shape_hp, shape_id = None, None
        if self.p_shape > 0.0 and rng.random() < self.p_shape:
            shape_hp = draw_cohort_hyperparams(
                np.random.default_rng([int(gmm_seed), _SHAPE_COHORT_HP_SEED_KEY]), self.shape_spec)
            family_by_shape_name = {v: k for k, v in SHAPE_ID_TO_FAMILY.items()}
            shape_id = family_by_shape_name[shape_hp.family]

        ncs = [self._build_nc(e, host_cls_id, rng, float(crop_spacing_mm), mu, sd, gmm_seed, i,
                              shape_hp=shape_hp, shape_id=shape_id)
               for i, e in enumerate(cohort)]

        if shape_hp is not None:
            name = MAISI_IDX_TO_CLASS[shape_id]
            host_suffix = f"|host{host_cls_id}"
        else:
            name = MAISI_IDX_TO_CLASS.get(host_cls_id, str(host_cls_id))
            host_suffix = ""

        return {
            "native_crop": ncs,
            "subject": f"{cohort[0]['file']}|{gmm_seed}|0{host_suffix}",
            "context_subjects": [f"{e['file']}|{gmm_seed}|{i + 1}{host_suffix}"
                                  for i, e in enumerate(cohort[1:])],
            "label_name": name,
            "aug_mode": torch.tensor(0, dtype=torch.long),
            "tgt_modality": "synth",
            "ctx_modality": "synth",
        }

    def load_native_crop(self, subject, cls, req):
        """Cascade re-crop: re-derive same GMM + member paint nrng from subject string."""
        if not self.cascade:
            raise RuntimeError("SynthGmmProvider.load_native_crop requires cascade=True")
        parts = subject.split("|")
        if parts[-1].startswith("host"):
            filename, gmm_seed_str, member_idx_str, host_str = parts
            host_cls_id = int(host_str[len("host"):])
        else:
            filename, gmm_seed_str, member_idx_str = parts
            host_cls_id = None
        gmm_seed = int(gmm_seed_str)
        member_idx = int(member_idx_str)
        mu, sd = self._draw_gmm(gmm_seed)
        e = self._entry_by_file[filename]
        cls_id = MAISI_CLASS_TO_IDX.get(cls)
        if cls_id is None:
            try:
                cls_id = int(cls)
            except (ValueError, TypeError):
                raise ValueError(f"SynthGmmProvider.load_native_crop: unknown class {cls!r}")

        shape_hp, shape_id = None, None
        if host_cls_id is not None:
            # req.center may be a cascade-predicted center or None (same-level
            # reconstruction) -- either way it only affects WHERE the crop window
            # looks; the shape's own position is anchored to the host's fixed centroid
            # (see _build_nc), independent of it. See docs/logs.md 2026-09-15.
            shape_hp = draw_cohort_hyperparams(
                np.random.default_rng([int(gmm_seed), _SHAPE_COHORT_HP_SEED_KEY]), self.shape_spec)
            shape_id = cls_id
            assert SHAPE_ID_TO_FAMILY.get(shape_id) == shape_hp.family, (
                f"shape id/family mismatch: cls={cls_id!r} resolved family "
                f"{SHAPE_ID_TO_FAMILY.get(shape_id)!r} != redrawn cohort family "
                f"{shape_hp.family!r} -- mis-routed cls or subject string")
            cls_id = host_cls_id       # resolve center/crop against the real host anchor

        # no jitter for cascade recrops (center is predicted, not default centroid)
        jitter = 0 if req.center is not None else self.ds.jitter
        return self._build_nc(e, cls_id, req.rng, req.crop_spacing_mm, mu, sd,
                               gmm_seed, member_idx, center=req.center, jitter=jitter,
                               center_mode=getattr(req, "center_mode", "com"),
                               shape_hp=shape_hp, shape_id=shape_id)
