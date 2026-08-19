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
from src.totalseg_dataloader_incontext import organ_crop_arrays, place_label

BODY_ID = 200
# fixed 0-255 -> ~zero-mean/unit-std bridge (uniform-ish over 0..255: mean 128, std ~74).
# fixed (not per-volume) so the cohort-shared appearance survives normalization.
GMM_MEAN, GMM_STD = 128.0, 74.0


class SynthGmmMaisiDataset(Dataset):
    def __init__(self, bank_dir, image_size=(128, 128, 128), context_size=4,
                 crop_spacing_mm=1.5, crop_jitter=None, classes=None, length=10000,
                 var_max=5.0, background_mode="zero", eval_seed=None, maxid=200,
                 cohort=None, mask_downsample="occupancy", mask_occupancy_thr=0.1,
                 class_balanced=True):
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
        print(f"SynthGmmMaisiDataset: {len(self.cs.entries)} masks | "
              f"{len(self.cs.classes)} classes | K={self.k} | T={self.T} | "
              f"crop={self.crop_mm}mm | var_max={self.var_max} | mask={_md} | "
              f"class_balanced={self.cs.class_balanced} | len={self.length}", flush=True)

    def __len__(self):
        return self.length

    def _resample_multiclass(self, crop_lbl, out_sizes, target_cls):
        """Resize a native-res multiclass crop (ints) to out_sizes (long tensor).

        "occupancy": area-pool each present id to its foreground fraction, assign per voxel
        the argmax id whose fraction clears mask_occupancy_thr (small ids survive, don't lose
        ties to larger neighbours); a non-empty guard keeps the target class's densest voxel
        if thresholding erased it (a synth item with an empty target is dead weight).
        "nearest": point-sample (thin ids can vanish under heavy downsampling)."""
        size = tuple(int(s) for s in out_sizes)
        if self.mask_downsample != "occupancy":
            t = torch.from_numpy(np.ascontiguousarray(crop_lbl, np.float32))[None, None]
            return F.interpolate(t, size=size, mode="nearest")[0, 0].long()
        out = torch.zeros(size, dtype=torch.long)
        best = torch.zeros(size)
        tgt_frac = None
        for i in (int(v) for v in np.unique(crop_lbl) if v != 0):
            bi = torch.from_numpy(np.ascontiguousarray(crop_lbl == i, np.float32))
            frac = F.interpolate(bi[None, None], size=size, mode="area")[0, 0]
            take = (frac >= self.mask_occupancy_thr) & (frac > best)
            out[take] = i
            best = torch.maximum(best, frac)
            if i == target_cls:
                tgt_frac = frac
        if tgt_frac is not None and not bool((out == target_cls).any()):
            out.view(-1)[int(tgt_frac.argmax())] = target_cls     # never emit an empty target
        return out

    def _crop_multiclass(self, e, cls, rng, crop_mm):
        """Organ-centred T³ multiclass label (long) around class `cls` in mask entry `e`."""
        arr = np.squeeze(np.load(self.cs.dir / "masks" / e["file"], mmap_mode="r"))
        center = tuple(e["cents"][cls][:3])
        _, crop_lbl, out_sizes, pad_lo, _ = organ_crop_arrays(
            arr, arr, center, e["spacing"], image_size=(self.T,) * 3,
            crop_mm=crop_mm, jitter=self.jitter, rng=rng)
        small = self._resample_multiclass(np.asarray(crop_lbl), out_sizes, cls)
        return place_label(small, out_sizes, pad_lo, self.T)          # (T,T,T) multiclass

    def _paint(self, lab_np, mu, sd, nrng):
        """Flat shared-id GMM: img = mu[lab] + sd[lab]*noise, then fixed 0-255->z bridge."""
        img = mu[lab_np] + sd[lab_np] * nrng.standard_normal(lab_np.shape, dtype=np.float32)
        return (img - GMM_MEAN) / GMM_STD

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
        cls, cohort = self.cs.sample_cohort(rng)

        # cohort-shared GMM draw (indexed by shared MAISI id 0..maxid); id 0 = air (fixed)
        n = self.maxid + 1
        mu = nrng.uniform(0.0, 255.0, size=n).astype(np.float32)
        sd = np.sqrt(nrng.uniform(0.0, self.var_max, size=n)).astype(np.float32)
        if self.bg_mode == "zero":
            mu[0] = 0.0; sd[0] = 0.0
        else:
            mu[0] = nrng.uniform(0.0, 15.0); sd[0] = 0.5 ** 0.5

        imgs, masks = [], []
        for e in cohort:
            lab = self._crop_multiclass(e, cls, rng, crop_mm)
            lab_np = lab.numpy().astype(np.int64)
            img = torch.from_numpy(self._paint(lab_np, mu, sd, nrng))[None]   # (1,T,T,T)
            imgs.append(img.float())
            masks.append((lab == cls).long())                                 # (T,T,T) binary
        name = MAISI_IDX_TO_CLASS.get(cls, str(cls))
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
    from data.maisi_classes import MAISI_CLASS_TO_IDX
    return MAISI_CLASS_TO_IDX[name]
