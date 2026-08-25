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
                 class_balanced=True, gpu_realize=False, gpu_realize_max_native=256):
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
        print(f"SynthGmmMaisiDataset: {len(self.cs.entries)} masks | "
              f"{len(self.cs.classes)} classes | K={self.k} | T={self.T} | "
              f"crop={self.crop_mm}mm | var_max={self.var_max} | mask={_md} | "
              f"class_balanced={self.cs.class_balanced} | realize={_gr} | "
              f"len={self.length}", flush=True)

    def __len__(self):
        return self.length

    def _resample_paint_mask(self, crop_lbl, out_sizes, target_cls):
        """Native-res multiclass crop (ints) -> (paint_lab, mask) at out_sizes (long tensors).

        The two play different roles, so they resize differently (mask_downsample only ever
        applied to a downsample that needs small structures preserved is the MASK):
          paint_lab: full multiclass, NEAREST. It only drives the per-voxel GMM shade, so it is
                     treated like an IMAGE — no enlargement; each boundary voxel takes one label.
          mask:      target-class BINARY under `mask_downsample`. "occupancy" area-pools the
                     target fraction and keeps voxels clearing mask_occupancy_thr, so a low thr
                     GROWS a small/thin target (matches totalseg resample_binary); "nearest"
                     point-samples. Non-empty guard. Rim voxels (mask=1 but paint!=target) are
                     realistic partial-volume hard cases — intentional, not a bug."""
        size = tuple(int(s) for s in out_sizes)
        paint = F.interpolate(
            torch.from_numpy(np.ascontiguousarray(crop_lbl, np.float32))[None, None],
            size=size, mode="nearest")[0, 0].long()
        if self.mask_downsample == "occupancy":
            bi = torch.from_numpy(np.ascontiguousarray(crop_lbl == target_cls, np.float32))
            frac = F.interpolate(bi[None, None], size=size, mode="area")[0, 0]
            mask = frac >= self.mask_occupancy_thr
            if not bool(mask.any()) and bool((crop_lbl == target_cls).any()):
                mask.view(-1)[int(frac.argmax())] = True          # never emit an empty target
            mask = mask.long()
        else:
            mask = (paint == target_cls).long()                   # nearest binary = paint==cls
        return paint, mask

    def _crop_paint_mask(self, e, cls, rng, crop_mm):
        """Organ-centred T³ (paint_lab, mask) (both long) around class `cls` in mask entry `e`."""
        arr = np.squeeze(np.load(self.cs.dir / "masks" / e["file"], mmap_mode="r"))
        center = tuple(e["cents"][cls][:3])
        _, crop_lbl, out_sizes, pad_lo, _ = organ_crop_arrays(
            arr, arr, center, e["spacing"], image_size=(self.T,) * 3,
            crop_mm=crop_mm, jitter=self.jitter, rng=rng)
        paint, mask = self._resample_paint_mask(np.asarray(crop_lbl), out_sizes, cls)
        return (place_label(paint, out_sizes, pad_lo, self.T),     # (T,T,T) multiclass (paint)
                place_label(mask, out_sizes, pad_lo, self.T))      # (T,T,T) binary (supervision)

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
        return self.assemble(rng, nrng, crop_mm)

    def assemble(self, rng, nrng, crop_mm):
        """Build one in-context item from an already-chosen (rng, nrng) and physical spacing.

        Split out from __getitem__ so the v2 cohort provider (src/providers/synth_gmm.py)
        can drive it from the InContextDataset engine's per-item RNG, sharing this single
        cohort-sample + shared-GMM + crop/paint implementation. `rng` selects the cohort and
        crops; `nrng` draws the cohort-shared GMM and the paint noise."""
        cls, cohort = self.cs.sample_cohort(rng)

        # cohort-shared GMM draw (indexed by shared MAISI id 0..maxid); id 0 = air (fixed)
        n = self.maxid + 1
        mu = nrng.uniform(0.0, 255.0, size=n).astype(np.float32)
        sd = np.sqrt(nrng.uniform(0.0, self.var_max, size=n)).astype(np.float32)
        if self.bg_mode == "zero":
            mu[0] = 0.0; sd[0] = 0.0
        else:
            mu[0] = nrng.uniform(0.0, 15.0); sd[0] = 0.5 ** 0.5

        name = MAISI_IDX_TO_CLASS.get(cls, str(cls))

        # GPU-realize: ship native crops + geometry + the cohort GMM draw; occupancy
        # resample + paint happen on GPU (src/gpu_synth_realize.SynthRealizer).
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
            paint, mask = self._crop_paint_mask(e, cls, rng, crop_mm)
            img = torch.from_numpy(                                            # paint drives shade
                self._paint(paint.numpy().astype(np.int64), mu, sd, nrng))[None]  # (1,T,T,T)
            imgs.append(img.float())
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
    from data.maisi_classes import MAISI_CLASS_TO_IDX
    try:
        return MAISI_CLASS_TO_IDX[name]
    except KeyError:
        # Common trap: TotalSeg underscore names (e.g. 'kidney_right') reach the MAISI
        # synth bank, whose vocabulary uses spaces ('right kidney'). Fail with the likely
        # fix instead of a bare KeyError. See docs on cross-source class vocabularies.
        alt = name.replace("_", " ")
        hint = (f" — did you mean {alt!r}? (MAISI uses space-separated names; "
                f"'{name}' looks like a TotalSeg class)"
                if alt in MAISI_CLASS_TO_IDX else
                f" (valid MAISI names use spaces, e.g. 'right kidney'; "
                f"pass MAISI ids/names or 'all')")
        raise KeyError(f"{name!r} is not a MAISI class{hint}") from None
