"""GPU realization of synth-GMM in-context items (occupancy mask resample + SynthSeg paint).

The CPU dataset worker (SynthGmmMaisiDataset with gpu_realize=True) ships only the native
multiclass label crops + placement geometry + the cohort-shared GMM draw. The heavy work —
the occupancy area-pool (native -> grid, per present class) that costs ~15 s/item on CPU, and
the per-voxel paint — runs HERE on GPU in the train loop, producing the same batch dict the
model / GpuAugmentor consume (image/label/context_in/context_out). Slot it in just before
GpuAugmentor (the painted volumes then flow through the existing geo/intensity aug unchanged).

Semantics mirror SynthGmmMaisiDataset._resample_paint_mask + ._paint exactly, so the training
distribution matches the CPU path (modulo GPU-vs-CPU float ordering). NB paint (multiclass,
nearest, image-like) and the supervised mask (target-class binary, occupancy) are DECOUPLED —
mask_occupancy_thr grows small targets in the mask only, never in the paint.
"""
import torch
import torch.nn.functional as F

from src.synth_gmm_maisi_dataset import GMM_MEAN, GMM_STD


def synth_gpu_collate_fn(batch):
    """Collate for gpu_realize items. Native crops are variable-shape, so they are kept as a
    list-of-lists (B x (K+1) uint8 tensors); everything else stacks. No image/label yet — the
    SynthRealizer fills those on GPU downstream."""
    out = {
        "native_lbls": [b["native_lbls"] for b in batch],              # B lists of K+1 tensors
        "out_sizes": torch.stack([b["out_sizes"] for b in batch]),     # (B,K+1,3)
        "pad_lo": torch.stack([b["pad_lo"] for b in batch]),           # (B,K+1,3)
        "cls": torch.tensor([b["cls"] for b in batch], dtype=torch.long),  # (B,)
        "gmm_mu": torch.stack([b["gmm_mu"] for b in batch]),           # (B,n)
        "gmm_sd": torch.stack([b["gmm_sd"] for b in batch]),           # (B,n)
        "spacing": torch.stack([b["spacing"] for b in batch]),         # (B,3) mm
        "subjects": [b["subject"] for b in batch],
        "label_names": [b["label_name"] for b in batch],
        "context_subjects": [b["context_subjects"] for b in batch],
        "aug_mode": torch.stack([b["aug_mode"] for b in batch]),       # (B,) int64
    }
    return out


@torch.no_grad()
def _resample_member(native, out_size, T, pad_lo, cls, occ_thr, mask_downsample, mu, sd, gen,
                     paint_mask_aligned=False):
    """Native-res multiclass crop (D',H',W' long, on device) -> placed T^3 (image, mask).

    GPU port of SynthGmmMaisiDataset._resample_paint_mask, decoupled by role:
      image: paint the GMM Gaussian at NATIVE res (continuous field, like a real CT) and
             trilinear-resample (+ area anti-alias pre-filter) THAT — mirrors
             src/providers/totalseg.place_image exactly, instead of nearest-resampling the
             discrete label and painting after. Gives real partial-volume-style blending at
             organ boundaries instead of a nearest-sample cliff (docs/logs.md).
      mask:  target-class BINARY under mask_downsample. "occupancy" area-pools the target
             fraction and keeps voxels clearing occ_thr (low thr GROWS a small/thin target,
             matches totalseg resample_binary); "nearest" point-samples. Non-empty guard.

    paint_mask_aligned=True overwrites the image in mask=1 voxels with a fresh draw of the
    target class's own Gaussian (see docs/logs.md) — previously a CPU-only option; this was
    the gap where gpu_realize silently ignored data.gmm.paint_mask_aligned."""
    device = native.device
    size = tuple(int(s) for s in out_size)

    noise = torch.randn(native.shape, generator=gen, device=device)
    paint_native = mu[native] + sd[native] * noise                     # continuous, native res
    paint_native = (paint_native - GMM_MEAN) / GMM_STD                 # normalize BEFORE resample
    src = paint_native[None, None]
    pre = tuple(min(o, s) for o, s in zip(size, native.shape))
    if pre != tuple(native.shape):
        src = F.interpolate(src, size=pre, mode="area")                # antialias pre-filter
    img = (src if tuple(src.shape[2:]) == size else
           F.interpolate(src, size=size, mode="trilinear", align_corners=False))[0, 0]

    if mask_downsample == "occupancy":
        frac = F.interpolate((native == cls).float()[None, None], size=size, mode="area")[0, 0]
        mask = frac >= occ_thr
        if not bool(mask.any()) and bool((native == cls).any()):
            mask.view(-1)[int(frac.argmax())] = True                   # never emit empty target
        mask = mask.long()
    else:
        paint_lbl = F.interpolate(native.float()[None, None], size=size, mode="nearest")[0, 0].long()
        mask = (paint_lbl == cls).long()

    if paint_mask_aligned:
        # See docs/logs.md: force the supervised region to a fresh draw of the target's own
        # Gaussian, trading the physically-realistic blend for a clean interior signal.
        fresh = torch.randn(img.shape, generator=gen, device=device)
        target_val = (mu[cls] + sd[cls] * fresh - GMM_MEAN) / GMM_STD
        img = torch.where(mask.bool(), target_val, img)

    if all(s == T for s in size):
        return img[None], mask
    fi = torch.full((T, T, T), float(paint_native.min()), device=device)
    fm = torch.zeros(T, T, T, dtype=torch.long, device=device)
    sl = tuple(slice(int(p), int(p) + s) for p, s in zip(pad_lo, size))
    fi[sl] = img
    fm[sl] = mask
    return fi[None], fm


class SynthRealizer:
    """Turn a gpu_realize batch (native crops + geometry + GMM draw) into the standard
    image/label/context_in/context_out batch dict, on GPU. Call before GpuAugmentor."""

    def __init__(self, T, occ_thr=0.1, mask_downsample="occupancy", seed=0,
                paint_mask_aligned=False, between_ratio=None):
        self.T = int(T)
        self.occ_thr = float(occ_thr)
        self.mask_downsample = mask_downsample
        self.paint_mask_aligned = bool(paint_mask_aligned)
        # Real intra-cohort variance (docs/logs.md): a FIXED calibration table (not cohort-
        # random, unlike gmm_mu/gmm_sd), so it's passed once here rather than shipped per-item
        # -- see SynthGmmMaisiDataset.between_ratio / src.gpu_gmm_intensity.resolve_between_ratio.
        self.between_ratio = (torch.as_tensor(between_ratio, dtype=torch.float32)
                              if between_ratio is not None else None)
        self._seed = int(seed)
        self._step = 0

    @torch.no_grad()
    def __call__(self, batch, device):
        T = self.T
        natives = batch["native_lbls"]
        B, Kp1 = len(natives), len(natives[0])
        out_sizes, pad_lo, cls = batch["out_sizes"], batch["pad_lo"], batch["cls"]
        mu = batch["gmm_mu"].to(device)                                # (B,n) cohort-shared
        sd = batch["gmm_sd"].to(device)
        between_ratio = self.between_ratio.to(device) if self.between_ratio is not None else None
        gen = torch.Generator(device=device)
        gen.manual_seed(self._seed + self._step)
        self._step += 1

        imgs, masks = [], []
        for b in range(B):
            cb = int(cls[b])
            mimg, mmask = [], []
            for t in range(Kp1):
                native = natives[b][t].to(device, non_blocking=True).long()
                # Real intra-cohort variance: each MEMBER gets its own mu_e on top of the
                # cohort-shared mu[b] (sd[b] stays the cohort-shared texture scale). Disabled
                # -> mu_e is mu[b], no extra randomness consumed (matches CPU path exactly).
                if between_ratio is not None:
                    eps = torch.randn(mu[b].shape, generator=gen, device=device)
                    mu_e = mu[b] + between_ratio * sd[b] * eps
                else:
                    mu_e = mu[b]
                img, mask = _resample_member(native, out_sizes[b, t], T, pad_lo[b, t],
                                             cb, self.occ_thr, self.mask_downsample,
                                             mu_e, sd[b], gen,
                                             paint_mask_aligned=self.paint_mask_aligned)
                mimg.append(img)                                       # (1,T,T,T) trilinear-blended
                mmask.append(mask)                                     # (T,T,T) binary (occupancy)
            imgs.append(torch.stack(mimg))                             # (K+1,1,T,T,T)
            masks.append(torch.stack(mmask))                           # (K+1,T,T,T)
        img = torch.stack(imgs).float()                                # (B,K+1,1,T,T,T)
        msk = torch.stack(masks)                                       # (B,K+1,T,T,T)
        batch["image"] = img[:, 0]                                     # (B,1,T,T,T)
        batch["context_in"] = img[:, 1:]                               # (B,K,1,T,T,T)
        batch["label"] = msk[:, 0]                                     # (B,T,T,T)
        batch["context_out"] = msk[:, 1:]                              # (B,K,T,T,T)
        batch["spacing"] = batch["spacing"].to(device)
        batch["aug_mode"] = batch["aug_mode"].to(device)
        return batch
