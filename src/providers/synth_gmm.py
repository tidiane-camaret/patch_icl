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
"""
import numpy as np
import torch

from data.maisi_classes import MAISI_CLASS_TO_IDX, MAISI_IDX_TO_CLASS
from src.gpu_gmm_intensity import sample_grouped_uniform
from src.totalseg_dataloader_incontext import organ_crop_arrays


class SynthGmmProvider:
    """Cohort-hook provider wrapping a SynthGmmMaisiDataset for InContextDataset."""

    def __init__(self, dataset, *, cascade=False):
        self.ds = dataset
        self.epoch_length = len(dataset)
        self.classes = [MAISI_IDX_TO_CLASS.get(c, str(c)) for c in dataset.cs.classes]
        self.cascade = cascade
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
                  center=None, jitter=None):
        """Crop + paint one MAISI bank entry → NativeCrop. cascade=True required."""
        # per-member nrng keyed to (gmm_seed, member_idx): reproducible at L1 recrops
        member_nrng = np.random.default_rng([int(gmm_seed), int(member_idx)])
        n = self.ds.maxid + 1
        mu_e = (mu + self.ds.between_ratio * sd * member_nrng.standard_normal(n).astype(np.float32)
                if self.ds.between_ratio is not None else mu)

        arr = np.squeeze(np.load(self.ds.cs.dir / "masks" / e["file"], mmap_mode="r"))
        if center is None:
            cents = e["cents"].get(cls_id)
            center = tuple(cents[:3]) if cents is not None else None
        if jitter is None:
            jitter = self.ds.jitter
        _, crop_lbl, out_sizes, pad_lo, geom = organ_crop_arrays(
            arr, arr, center, list(e["spacing"]),
            image_size=(self.ds.T,) * 3, crop_mm=crop_mm, jitter=jitter, rng=rng)
        img, mask = self.ds._resample_paint_mask(
            np.asarray(crop_lbl), out_sizes, pad_lo, cls_id, mu_e, sd, member_nrng)

        T = self.ds.T
        return self._NativeCrop(
            image=img[0].half(),
            label_frac=mask.float().half(),
            class_idx=cls_id,
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
        cls_id, cohort = self.ds.cs.sample_cohort(rng)
        ncs = [self._build_nc(e, cls_id, rng, float(crop_spacing_mm), mu, sd, gmm_seed, i)
               for i, e in enumerate(cohort)]
        name = MAISI_IDX_TO_CLASS.get(cls_id, str(cls_id))
        return {
            "native_crop": ncs,
            "subject": f"{cohort[0]['file']}|{gmm_seed}|0",
            "context_subjects": [f"{e['file']}|{gmm_seed}|{i + 1}"
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
        filename, gmm_seed_str, member_idx_str = subject.rsplit("|", 2)
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
        # no jitter for cascade recrops (center is predicted, not default centroid)
        jitter = 0 if req.center is not None else self.ds.jitter
        return self._build_nc(e, cls_id, req.rng, req.crop_spacing_mm, mu, sd,
                               gmm_seed, member_idx, center=req.center, jitter=jitter)
