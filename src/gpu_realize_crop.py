"""GPU realization of TotalSeg native crops for the cascade data path.

Turns a list of provider NativeCrop payloads (raw integer-decimated crop + crop
geometry) into the standard image/label/context_in/context_out batch dict, on
device: trilinear/area resample to out_sizes, CT-normalize, centre-pad to T^3;
occupancy/soft target-class mask with the resample_binary semantics. Slots in
right before GpuAugmentor, exactly like src/gpu_synth_realize.SynthRealizer.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from src.totalseg_dataloader_incontext import _area_pool_3d


def normalize_ct_gpu(t: torch.Tensor, spec) -> torch.Tensor:
    """clip(t, [clip_lo, clip_hi]) then (t - mean) / std -> float32 (mirrors normalize_ct)."""
    return ((t.float().clamp(spec.clip_lo, spec.clip_hi) - spec.mean) / spec.std)


def _regroup(flat, B, Kp1):
    """Flat length-B*(K+1) list -> B lists of K+1 (target first in each)."""
    return [list(flat[b * Kp1:(b + 1) * Kp1]) for b in range(B)]


@torch.no_grad()
def _realize_member(nc, T, mask_downsample, occ_thr, ct_spec, device):
    """One NativeCrop -> placed (image (1,T,T,T) f32, mask (T,T,T)).

    Image: area-prefilter to min(out, src) then trilinear resample to out_sizes,
    CT-normalize, centre-pad with the resampled member's own normalized min (air).
    Mask: target-class binary area-pooled to out_sizes under resample_binary
    semantics ("soft" partial-volume fraction with peak floor, "occupancy"
    fraction >= occ_thr with a never-empty guard), centre-padded with 0.
    """
    size = tuple(int(s) for s in nc.out_sizes)
    src = nc.image.to(device).float()[None, None]
    pre = tuple(min(o, s) for o, s in zip(size, src.shape[2:]))
    if pre != tuple(src.shape[2:]):
        src = F.interpolate(src, size=pre, mode="area")
    img = (src if tuple(src.shape[2:]) == size else
           F.interpolate(src, size=size, mode="trilinear", align_corners=False))
    img = normalize_ct_gpu(img[0, 0], ct_spec)                          # (d,h,w) f32

    binm = (nc.label.to(device).long() == int(nc.class_idx)).float()[None, None]
    frac = _area_pool_3d(binm, size)[0, 0].clamp(0.0, 1.0)
    if mask_downsample == "soft":
        peak = float(frac.amax())
        if bool(binm.any()) and peak < occ_thr:
            frac = torch.where(frac >= peak, torch.full_like(frac, occ_thr), frac)
        mask = frac                                                     # f32
    else:                                                              # occupancy
        m = frac >= occ_thr
        if not bool(m.any()) and bool(binm.any()):
            m.view(-1)[int(frac.argmax())] = True
        mask = m.long()

    if size == (T, T, T):
        return img[None], mask
    fi = torch.full((T, T, T), float(img.min()), device=device)
    fm = torch.zeros(T, T, T, dtype=mask.dtype, device=device)
    sl = tuple(slice(int(p), int(p) + s) for p, s in zip(nc.pad_lo, size))
    fi[sl] = img
    fm[sl] = mask
    return fi[None], fm


@torch.no_grad()
def realize_native_crops(members, *, T, mask_downsample, occ_thr, ct_spec, device):
    """B lists of K+1 NativeCrop (target first) -> standard batch dict on `device`.

    Returns image/context_in/label/context_out plus per-row spacing and crop_geom
    taken from each row's target member (members[b][0]).
    """
    B, Kp1 = len(members), len(members[0])
    imgs, masks = [], []
    for b in range(B):
        mi, mm = [], []
        for t in range(Kp1):
            i, m = _realize_member(members[b][t], T, mask_downsample, occ_thr, ct_spec, device)
            mi.append(i); mm.append(m)
        imgs.append(torch.stack(mi)); masks.append(torch.stack(mm))
    img = torch.stack(imgs).float()                                    # (B,K+1,1,T,T,T)
    msk = torch.stack(masks)                                           # (B,K+1,T,T,T)
    geom = torch.stack([members[b][0].crop_geom.to(device) for b in range(B)])
    sp = torch.stack([torch.full((3,), float(members[b][0].crop_spacing_mm)) for b in range(B)])
    return {"image": img[:, 0], "context_in": img[:, 1:],
            "label": msk[:, 0], "context_out": msk[:, 1:],
            "spacing": sp.to(device), "crop_geom": geom}


def native_crop_collate_fn(batch):
    """Collate InContextDataset native-crop items. Keeps `native_crop` as a B-list
    (variable-shape crops don't stack); stacks aug_mode; passes id lists through."""
    out = {
        "native_crop": [b["native_crop"] for b in batch],              # B lists of (K+1)
        "subjects": [b["subject"] for b in batch],
        "label_names": [b["label_name"] for b in batch],
        "context_subjects": [b["context_subjects"] for b in batch],
        "aug_mode": torch.stack([b.get("aug_mode", torch.tensor(0, dtype=torch.long))
                                 for b in batch]),
    }
    return out
