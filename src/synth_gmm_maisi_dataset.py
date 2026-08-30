"""
Minimal GMM-synth in-context dataset (wires into experiments/3d/train.py via
data.source=synth_gmm_maisi). Each item is a COHORT of K+1 similar masks drawn from the
MAISI mask bank (CohortSampler), organ-cropped around a shared target class and painted
with ONE cohort-shared per-label Gaussian (SynthSeg) indexed by the shared MAISI id — so an
organ keeps a consistent shade across target+context ("one scanner"), while different items
= different scanners. Returns the same item dict as TotalSegInContextDataset so it plugs
into incontext_collate_fn / train.py unchanged.

Paint is CPU (flat GMM, vectorized LUT gather — no train.py changes; downstream texture can
come from the existing GPU bias/noise aug). The GPU intensity stage (src/gpu_gmm_intensity)
is the drop-in fast path for later when paint moves into the train loop.
"""
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from data.maisi_classes import MAISI_IDX_TO_CLASS
from src.gmm_cohort_sampler import CohortSampler
from src.gpu_gmm_intensity import maisi_ids_to_indices, resolve_between_ratio, sample_grouped_uniform
from src.totalseg_dataloader_incontext import (organ_crop_arrays, place_image, place_label,
                                                resample_binary)

BODY_ID = 200
# fixed 0-255 -> ~zero-mean/unit-std bridge (uniform-ish over 0..255: mean 128, std ~74).
# fixed (not per-volume) so the cohort-shared appearance survives normalization.
GMM_MEAN, GMM_STD = 128.0, 74.0


class SynthGmmMaisiDataset(Dataset):
    def __init__(self, bank_dir, image_size=(128, 128, 128), context_size=4,
                 crop_spacing_mm=1.5, crop_jitter=None, classes=None, length=10000,
                 var_max=5.0, background_mode="zero", eval_seed=None, maxid=200,
                 cohort=None, mask_downsample="occupancy", mask_occupancy_thr=0.1,
                 class_balanced=True, gpu_realize=False, gpu_realize_max_native=256,
                 paint_mask_aligned=False, mu_group_ids=None, mu_group_rho=None,
                 sd_between_ratio=None):
        assert image_size[0] == image_size[1] == image_size[2], "cubic crops only"
        self.T = int(image_size[0])
        self.k = int(context_size)
        self.crop_mm = float(crop_spacing_mm)
        self.jitter = int(crop_jitter) if crop_jitter is not None else self.T // 4
        self.length = int(length)
        self.var_max = float(var_max)
        self.bg_mode = background_mode
        self.maxid = int(maxid)
        self.eval_seed = eval_seed
        # How the native-res multiclass crop is resized to grid res. "occupancy" area-pools
        # each present id to its foreground fraction and keeps the argmax id clearing
        # mask_occupancy_thr per voxel (thin organs survive; matches totalseg training);
        # "nearest" point-samples (thin organs vanish at large FOV). The paint is drawn FROM
        # this resampled map, so image and mask stay consistent either way.
        assert mask_downsample in ("nearest", "occupancy"), mask_downsample
        self.mask_downsample = mask_downsample
        self.mask_occupancy_thr = float(mask_occupancy_thr)
        # GPU-realize mode: ship the NATIVE-res multiclass crop (uint8) + placement geometry +
        # the cohort GMM draw instead of a CPU-painted image, so the ~15 s/item occupancy
        # area-pool (per present class, native res) and the paint run batched on GPU in the
        # train loop (src/gpu_synth_realize). gpu_realize_max_native caps the shipped native
        # side (nearest pre-downsample above it) to bound H2D transfer + GPU memory at large
        # crop_spacing (FOV); occupancy native->grid still runs on GPU below that cap. 0/None
        # = uncapped. Only meaningful for the TRAIN split (val is a real source via eval_cfg).
        self.gpu_realize = bool(gpu_realize)
        self.gpu_realize_max_native = int(gpu_realize_max_native or 0)
        # paint_mask_aligned: when True, overwrites the paint map in mask=1 voxels with the target
        # class BEFORE painting, ensuring consistent intensity within the supervision mask. This
        # eliminates artificial boundary variance from neighboring structures bleeding into the
        # occupancy-expanded mask region. Default False for backward compat.
        self.paint_mask_aligned = bool(paint_mask_aligned)
        # mu_group_ids/mu_group_rho: inject REALISTIC INTENSITY CORRELATION between a FIXED
        # subset of label ids every cohort (see src.gpu_gmm_intensity.sample_grouped_uniform
        # + CT_GROUP_MAISI_IDS/CT_GROUP_RHO, calibrated from real TotalSegmentator CT via
        # analyze_totalseg_intensity.py) instead of every slot's mu being fully independent.
        # Motivation: fully-independent per-slot mu lets every label be told apart by shade
        # alone (e.g. two ribs painted at unrelated brightnesses), which real scans never do
        # (same-tissue structures share a HU band) and makes localization artificially easy.
        # Membership is FIXED (the same real MAISI ids every cohort -- reshuffling it per
        # cohort was tried and rejected, see docs/logs.md: it collapses the real block-
        # diagonal correlation structure into a flat, anatomy-blind bump). The per-slot
        # marginal stays EXACTLY Uniform(0,255) either way and the shared VALUE each group
        # takes is still a fresh random draw every cohort -- no real HU value or persistent
        # brightness-to-identity mapping is ever injected, only "these ids are the same
        # tissue type". None/empty (default) = today's fully-independent behavior, unchanged.
        # mu_group_ids takes 1-based MAISI ids (e.g. CT_GROUP_MAISI_IDS); converted to 0-based
        # positions into mu[1:] once here (maisi_ids_to_indices) rather than every assemble().
        self.mu_group_ids = (maisi_ids_to_indices(mu_group_ids) if mu_group_ids else ())
        self.mu_group_rho = tuple(mu_group_rho) if mu_group_rho else ()
        assert len(self.mu_group_ids) == len(self.mu_group_rho), \
            (self.mu_group_ids, self.mu_group_rho)
        # sd_between_ratio: inject REAL INTRA-COHORT (member-to-member) intensity variance,
        # currently ~absent (mu[c] is shared verbatim by every member; sd[c] is a per-VOXEL
        # noise scale, so any organ with >~a few hundred voxels has its per-member mean
        # converge to mu[c], averaging away). Real CT splits a class's total spread into
        # between-subject (patient-to-patient, e.g. IV-contrast timing shifts a vessel's whole
        # characteristic HU) and within-scan voxel texture (what sd[c] already models); their
        # RATIO is NOT constant across classes (docs/logs.md "real intra-cohort variance
        # analysis" -- vascular ~1.1, organ/lung ~0.6, bone/muscle ~0.4). Modeled as a second
        # Gaussian level on top of sd[c]'s existing texture role (unchanged): each MEMBER e
        # (not just each voxel) gets its own mu_e = mu + ratio*sd*eps_e, eps_e ~ N(0,1) fresh
        # per member; sd[c] itself is untouched. None (default) = ratio=0 everywhere = today's
        # behavior exactly (no extra randomness consumed). 'ct' = CT_BETWEEN_WITHIN_GROUPS
        # preset (6-family lookup, see src.gpu_gmm_intensity). Or pass an explicit
        # (maxid+1,)-length array.
        self.between_ratio = resolve_between_ratio(sd_between_ratio, self.maxid)
        # cohort-sampling knobs (distance weights + diversity) -> CohortSampler; empty = defaults.
        # class_balanced (uniform-over-classes vs mask-frequency prior) is a top-level knob
        # mirroring totalseg data.class_balanced, passed alongside the cohort dict.
        self.cs = CohortSampler(bank_dir, k=self.k, class_balanced=bool(class_balanced),
                                **dict(cohort or {}))
        # optional class restriction (list of MAISI ids or names); default all usable classes
        if classes:
            ids = {c if isinstance(c, int) else _name_to_id(c) for c in classes}
            self.cs.classes = [c for c in self.cs.classes if c in ids]
        assert self.cs.classes, "no usable classes in bank after filtering"
        _md = (self.mask_downsample
               + (f"(thr={self.mask_occupancy_thr})" if self.mask_downsample == "occupancy" else ""))
        _gr = (f"GPU(cap={self.gpu_realize_max_native or 'off'})" if self.gpu_realize else "CPU")
        _pa = f" | paint_align={self.paint_mask_aligned}" if self.paint_mask_aligned else ""
        _mg = (f" | mu_groups={[len(g) for g in self.mu_group_ids]}@rho{self.mu_group_rho}"
              if self.mu_group_ids else "")
        _br = (f" | between_ratio[{self.between_ratio.min():.2f},{self.between_ratio.max():.2f}]"
              if self.between_ratio is not None else "")
        print(f"SynthGmmMaisiDataset: {len(self.cs.entries)} masks | "
              f"{len(self.cs.classes)} classes | K={self.k} | T={self.T} | "
              f"crop={self.crop_mm}mm | var_max={self.var_max} | mask={_md} | "
              f"class_balanced={self.cs.class_balanced} | realize={_gr}{_pa}{_mg}{_br} | "
              f"len={self.length}", flush=True)

    def __len__(self):
        return self.length

    def _resample_paint_mask(self, crop_lbl, out_sizes, pad_lo, target_cls, mu, sd, nrng):
        """Native-res multiclass crop (ints) -> placed (image (1,T,T,T) f32, mask (T,T,T) long).

        Mirrors the REAL totalseg image path (src/providers/totalseg.place_image): paint the
        GMM Gaussian at NATIVE resolution (a continuous field, like a real CT) and resample
        THAT with trilinear (+ area anti-alias pre-filter) — i.e. treat the painted volume
        exactly like a real image — instead of nearest-resampling the discrete label and
        painting after. This gives genuine partial-volume-style blending at organ boundaries
        (a boundary voxel is a physically-weighted mix of neighboring tissues' colors) instead
        of a hard nearest-sample cliff, which could disagree with the mask's occupancy
        footprint (docs/logs.md).

        The MASK stays a separate resample of the native BINARY target map via
        `mask_downsample` ("occupancy" area-pools + thresholds at mask_occupancy_thr, so a low
        thr GROWS a thin/small target, matching totalseg's resample_binary; "nearest" point-
        samples). Non-empty guard is inside resample_binary.

        When `paint_mask_aligned=True`, the image is overwritten in mask=1 voxels with a fresh
        draw of the target class's own Gaussian, so the supervised region is never contaminated
        by a blended neighbor at the boundary (trades the physically-realistic blend for a
        clean interior signal, same intent as before — see docs/logs.md)."""
        noise = nrng.standard_normal(crop_lbl.shape).astype(np.float32)
        paint_native = mu[crop_lbl] + sd[crop_lbl] * noise                     # continuous, native res
        paint_native = (paint_native - GMM_MEAN) / GMM_STD          # normalize BEFORE resample
        img = place_image(paint_native, out_sizes, pad_lo, self.T, antialias=True)

        mask = resample_binary(crop_lbl == target_cls, tuple(out_sizes),
                               mode=self.mask_downsample, occ_thr=self.mask_occupancy_thr)
        mask = place_label(mask, out_sizes, pad_lo, self.T)

        if self.paint_mask_aligned:
            fresh = nrng.standard_normal(tuple(img.shape[1:])).astype(np.float32)
            target_val = (mu[target_cls] + sd[target_cls] * fresh - GMM_MEAN) / GMM_STD
            img = torch.where(mask.bool().unsqueeze(0),
                              torch.from_numpy(target_val).unsqueeze(0), img)
        return img, mask

    def _crop_paint_mask(self, e, cls, rng, crop_mm, mu, sd, nrng):
        """Organ-centred T³ (image (1,T,T,T) f32, mask (T,T,T) long) around class `cls`."""
        arr = np.squeeze(np.load(self.cs.dir / "masks" / e["file"], mmap_mode="r"))
        center = tuple(e["cents"][cls][:3])
        _, crop_lbl, out_sizes, pad_lo, _ = organ_crop_arrays(
            arr, arr, center, e["spacing"], image_size=(self.T,) * 3,
            crop_mm=crop_mm, jitter=self.jitter, rng=rng)
        return self._resample_paint_mask(np.asarray(crop_lbl), out_sizes, pad_lo, cls, mu, sd, nrng)

    def _native_crop(self, e, cls, rng, crop_mm):
        """GPU-realize worker payload: the NATIVE-res multiclass crop (uint8, ids 0..maxid)
        around class `cls`, plus placement geometry (out_sizes, pad_lo). No resample/paint —
        both run on GPU. Caps the native side at gpu_realize_max_native via a cheap nearest
        pre-downsample so H2D transfer + GPU one-hot memory stay bounded at large crop FOV."""
        arr = np.squeeze(np.load(self.cs.dir / "masks" / e["file"], mmap_mode="r"))
        center = tuple(e["cents"][cls][:3])
        _, crop_lbl, out_sizes, pad_lo, _ = organ_crop_arrays(
            arr, arr, center, e["spacing"], image_size=(self.T,) * 3,
            crop_mm=crop_mm, jitter=self.jitter, rng=rng)
        native = np.ascontiguousarray(crop_lbl, dtype=np.uint8)         # ids 0..maxid<=255
        cap = self.gpu_realize_max_native
        if cap and max(native.shape) > cap:                            # bound transfer/mem
            new = tuple(min(cap, s) for s in native.shape)             # still >= out_sizes (<=T<=cap)
            native = (F.interpolate(torch.from_numpy(native.astype(np.float32))[None, None],
                                    size=new, mode="nearest")[0, 0].to(torch.uint8).numpy())
        return (torch.from_numpy(native),
                torch.tensor(out_sizes, dtype=torch.long),
                torch.tensor(pad_lo, dtype=torch.long))

    def __getitem__(self, idx):
        # The spacing batch sampler indexes with (idx, spacing) so every item in a batch
        # crops (and reports) the same physical spacing (train_spacing_range); a plain int
        # → fixed crop_spacing_mm.
        if isinstance(idx, (tuple, list)):
            idx, crop_mm = int(idx[0]), float(idx[1])
        else:
            crop_mm = self.crop_mm
        # train: fresh entropy each call (generative); eval: deterministic per idx.
        if self.eval_seed is not None:
            rng = random.Random(hash((self.eval_seed, idx)) & 0xFFFFFFFF)
            nrng = np.random.default_rng(hash((self.eval_seed, idx, "n")) & 0xFFFFFFFF)
        else:
            rng, nrng = random, np.random.default_rng()
        return self.assemble(rng, nrng, crop_mm)

    def assemble(self, rng, nrng, crop_mm):
        """Build one in-context item from an already-chosen (rng, nrng) and physical spacing.

        Split out from __getitem__ so the v2 cohort provider (src/providers/synth_gmm.py)
        can drive it from the InContextDataset engine's per-item RNG, sharing this single
        cohort-sample + shared-GMM + crop/paint implementation. `rng` selects the cohort and
        crops; `nrng` draws the cohort-shared GMM and the paint noise."""
        cls, cohort = self.cs.sample_cohort(rng)

        # cohort-shared GMM draw (indexed by shared MAISI id 0..maxid); id 0 = air (fixed).
        # mu[1:] is EITHER fully independent (default) OR grouped-correlated (mu_group_ids
        # set, FIXED membership) -- see sample_grouped_uniform: same exact per-slot marginal
        # either way, only the joint structure across those fixed ids changes.
        n = self.maxid + 1
        mu = np.empty(n, dtype=np.float32)
        if self.mu_group_ids:
            mu[1:] = sample_grouped_uniform(n - 1, 0.0, 255.0, self.mu_group_ids,
                                            self.mu_group_rho, nrng)
        else:
            mu[1:] = nrng.uniform(0.0, 255.0, size=n - 1)
        sd = np.sqrt(nrng.uniform(0.0, self.var_max, size=n)).astype(np.float32)
        if self.bg_mode == "zero":
            mu[0] = 0.0; sd[0] = 0.0
        else:
            mu[0] = nrng.uniform(0.0, 15.0); sd[0] = 0.5 ** 0.5

        name = MAISI_IDX_TO_CLASS.get(cls, str(cls))

        # GPU-realize: ship native crops + geometry + the cohort GMM draw; occupancy
        # resample + paint (incl. the per-member mu_e perturbation, if between_ratio is set --
        # a fixed calibration table, so SynthRealizer is constructed with it directly rather
        # than shipping it per-item) happen on GPU (src/gpu_synth_realize.SynthRealizer).
        if self.gpu_realize:
            natives, outs, pads = [], [], []
            for e in cohort:
                nat, osz, plo = self._native_crop(e, cls, rng, crop_mm)
                natives.append(nat); outs.append(osz); pads.append(plo)
            return {
                "native_lbls": natives,                       # list K+1 uint8 (variable shape)
                "out_sizes": torch.stack(outs),               # (K+1,3)
                "pad_lo": torch.stack(pads),                  # (K+1,3)
                "cls": int(cls),
                "gmm_mu": torch.from_numpy(mu),               # (maxid+1,) cohort-shared
                "gmm_sd": torch.from_numpy(sd),               # (maxid+1,)
                "spacing": torch.full((3,), crop_mm, dtype=torch.float32),
                "subject": cohort[0]["file"], "label_name": name,
                "context_subjects": [e["file"] for e in cohort[1:]],
                "aug_mode": torch.tensor(0, dtype=torch.long),
            }

        imgs, masks = [], []
        for e in cohort:
            # Real intra-cohort variance (docs/logs.md): each MEMBER gets its own mu_e, on top
            # of the cohort-shared mu -- sd stays the cohort-shared per-voxel texture scale.
            # Disabled (between_ratio is None) -> mu_e is mu, no extra randomness consumed
            # (bit-identical to before this feature existed).
            mu_e = (mu + self.between_ratio * sd * nrng.standard_normal(n).astype(np.float32)
                   if self.between_ratio is not None else mu)
            img, mask = self._crop_paint_mask(e, cls, rng, crop_mm, mu_e, sd, nrng)
            imgs.append(img.float())                                          # (1,T,T,T) trilinear-blended
            masks.append(mask)                                                # (T,T,T) binary (occupancy)
        return {
            "image": imgs[0], "label": masks[0],
            "context_in": torch.stack(imgs[1:]),          # (K,1,T,T,T)
            "context_out": torch.stack(masks[1:]),        # (K,T,T,T)
            "spacing": torch.full((3,), crop_mm, dtype=torch.float32),
            "subject": cohort[0]["file"], "label_name": name,
            "context_subjects": [e["file"] for e in cohort[1:]],
            "aug_mode": torch.tensor(0, dtype=torch.long),
        }


def _name_to_id(name):
    """Convert a class name (any format) to MAISI index.

    Accepts both MAISI names ('left kidney') and TotalSeg names ('kidney_left')
    via the unified class registry.
    """
    from data.class_registry import to_maisi_idx, normalize_lenient
    # Try the class registry first (handles both vocabularies)
    try:
        canon = normalize_lenient(name)
        maisi_idx = to_maisi_idx(canon)
        if maisi_idx is not None:
            return maisi_idx
    except KeyError:
        pass
    # Fallback to direct MAISI lookup for any edge cases
    from data.maisi_classes import MAISI_CLASS_TO_IDX
    if name in MAISI_CLASS_TO_IDX:
        return MAISI_CLASS_TO_IDX[name]
    raise KeyError(f"{name!r} is not a known MAISI class. Use MAISI names "
                   f"('left kidney'), TotalSeg names ('kidney_left'), or MAISI ids.")
