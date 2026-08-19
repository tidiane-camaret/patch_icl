"""
Is the GMM-synth path worth it? Compare per-item + multi-worker dataloading time of:
  (A) the REAL TotalSeg in-context dataloader (use_crop, 1.5mm, p_synth=0), and
  (B) a GMM-synth dataloader: sample K+1 MAISI-bank masks sharing a class, organ-crop
      each, paint with a fully-random per-label Gaussian (SynthSeg).

Both reuse the SAME crop machinery (organ_crop_arrays/place_label) so the delta is
purely REAL-CT-read+resize (A) vs GMM-paint (B). Paint is vectorised (per-label mu/sigma
LUT gather — the real impl, no python label loop).

  .venv_thor/bin/python experiments/3d/synth_task_generation/bench_dataloader_gmm.py \
    --n_masks 24 --items 60 --workers 8
"""
import argparse
import gzip
import json
import random
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage as ndi
from torch.utils.data import DataLoader, Dataset

from src.totalseg_dataloader_incontext import (
    TotalSegInContextDataset, organ_crop_arrays, place_label,
)

REPO = Path("/home/dpxuser/repos/NV-Generate-CTMR")
TSEG = Path("/nfs/data/nii/data1/Analysis/camaret___in_context_segmentation/"
            "ANALYSIS_20251122/data/totalseg")
BANK = Path("/tmp/gmm_bank")
ZIP_ROOT = "all_masks_flexible_size_and_spacing_4000"
T, CROP_MM, JITTER, K = 128, 1.5, 32, 4
TSEG_CLASSES = ["liver", "spleen", "kidney_left", "kidney_right", "stomach",
                "aorta", "pancreas", "gallbladder", "urinary_bladder", "esophagus"]


# ---------------------------------------------------------------- GMM bank
def build_bank(n_masks):
    """Extract n_masks spread across sources → uint8 .npy + per-label centroid index."""
    BANK.mkdir(parents=True, exist_ok=True)
    idx_path = BANK / "index.json"
    if idx_path.exists():
        idx = json.load(open(idx_path))
        if len(idx) >= n_masks:
            return idx[:n_masks]
    cand = json.load(open(REPO / "datasets/candidate_masks_flexible_size_and_spacing_4000.json"))
    zf = zipfile.ZipFile(REPO / "datasets/all_masks_flexible_size_and_spacing_4000.zip")
    members = set(zf.namelist())
    rng = np.random.default_rng(0)
    by_src = {}
    for e in cand:
        by_src.setdefault(e["pseudo_label_filename"].split("/")[1], []).append(e)
    srcs = list(by_src)
    picks = []
    while len(picks) < n_masks:
        s = srcs[len(picks) % len(srcs)]
        picks.append(by_src[s][rng.integers(len(by_src[s]))])
    index = []
    for i, e in enumerate(picks):
        member = f"{ZIP_ROOT}/" + e["pseudo_label_filename"].lstrip("./")
        if member not in members:
            continue
        import nibabel as nib
        arr = np.asarray(nib.Nifti1Image.from_bytes(gzip.decompress(zf.read(member))).dataobj)
        arr = np.squeeze(arr).astype(np.uint8)
        labs = [int(l) for l in np.unique(arr) if l != 0]
        coms = ndi.center_of_mass(np.ones_like(arr, np.uint8), arr, labs)
        cents = {str(l): [int(round(c)) for c in com] for l, com in zip(labs, coms)}
        name = f"mask_{i:03d}.npy"
        np.save(BANK / name, arr)
        index.append({"file": name, "spacing": e["spacing"], "cents": cents})
        print(f"  banked {name} src={e['pseudo_label_filename'].split('/')[1]} "
              f"dim={arr.shape} labels={len(labs)}", flush=True)
    json.dump(index, open(idx_path, "w"))
    return index


def paint_gmm(lab, rng, blur=(0.5, 1.6), bias=0.25, noise=0.03):
    """Fully-random SynthSeg, vectorised: per-label mu,sigma LUT → gather → blur/bias/noise."""
    n = int(lab.max()) + 1
    mu = rng.uniform(0.0, 1.0, size=n).astype(np.float32)
    sd = rng.uniform(0.0, 0.15, size=n).astype(np.float32)
    img = mu[lab] + sd[lab] * rng.standard_normal(lab.shape, dtype=np.float32)
    img = ndi.gaussian_filter(img, rng.uniform(*blur))
    bf = rng.normal(1.0, bias, size=(4, 4, 4)).astype(np.float32)
    img = img * ndi.zoom(bf, np.array(lab.shape) / 4.0, order=1)
    img = img + rng.normal(0.0, noise, size=lab.shape).astype(np.float32)
    lo, hi = img.min(), img.max()
    return (img - lo) / (hi - lo + 1e-6)


class GMMBankDataset(Dataset):
    def __init__(self, index, k=K, n_items=4000):
        self.index, self.k, self.n = index, k, n_items
        self.cls2masks = {}
        for c in set().union(*[set(e["cents"]) for e in index]):
            ms = [i for i, e in enumerate(index) if c in e["cents"]]
            if len(ms) >= k + 1:
                self.cls2masks[c] = ms
        self.classes = list(self.cls2masks)
        # per-item timing accumulators (main-process, workers=0 only)
        self.t_load = self.t_paint = 0.0

    def __len__(self):
        return self.n

    def _one(self, mid, cls, rng):
        e = self.index[mid]
        t0 = time.perf_counter()
        lbl_mm = np.squeeze(np.load(BANK / e["file"], mmap_mode="r"))
        center = tuple(e["cents"][cls][:3])
        sp = e["spacing"]
        crop_a, crop_lbl, out_sizes, pad_lo, _ = organ_crop_arrays(
            lbl_mm, lbl_mm, center, sp, image_size=(T, T, T),
            crop_mm=CROP_MM, jitter=JITTER, rng=random)
        lbl_small = F.interpolate(
            torch.from_numpy(crop_lbl.astype(np.float32))[None, None],
            size=tuple(out_sizes), mode="nearest")[0, 0].long()
        lbl_full = place_label(lbl_small, out_sizes, pad_lo, T)  # (T,T,T) multiclass
        t1 = time.perf_counter()
        img = paint_gmm(lbl_full.numpy().astype(np.int16), rng)
        t2 = time.perf_counter()
        self.t_load += t1 - t0
        self.t_paint += t2 - t1
        return (torch.from_numpy(img)[None],           # (1,T,T,T)
                (lbl_full == int(cls)).long())          # (T,T,T) binary target

    def __getitem__(self, idx):
        rng = np.random.default_rng()
        cls = random.choice(self.classes)
        mids = random.sample(self.cls2masks[cls], self.k + 1)
        vols = [self._one(m, cls, rng) for m in mids]
        img0, lbl0 = vols[0]
        return {
            "image": img0, "label": lbl0,
            "context_in": torch.stack([v[0] for v in vols[1:]]),
            "context_out": torch.stack([v[1] for v in vols[1:]]),
        }


# ---------------------------------------------------------------- timing
def time_workers0(ds, items):
    it = iter(ds)
    for _ in range(3):
        ds[0]
    t0 = time.perf_counter()
    for _ in range(items):
        _ = ds[random.randrange(len(ds))]
    return (time.perf_counter() - t0) / items * 1e3


def time_loader(ds, items, workers, bs=1):
    dl = DataLoader(ds, batch_size=bs, num_workers=workers, shuffle=True,
                    persistent_workers=False, collate_fn=None)
    it = iter(dl)
    next(it)  # warm workers
    t0, n = time.perf_counter(), 0
    for b in it:
        n += bs
        if n >= items:
            break
    dt = time.perf_counter() - t0
    return n / dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_masks", type=int, default=24)
    ap.add_argument("--items", type=int, default=60)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    print("=== building GMM mask bank ===", flush=True)
    index = build_bank(a.n_masks)
    gmm = GMMBankDataset(index)
    print(f"GMM bank: {len(index)} masks, {len(gmm.classes)} usable classes "
          f"(>= K+1={K + 1} masks): {sorted(int(c) for c in gmm.classes)}", flush=True)

    print("\n=== building REAL TotalSeg dataset ===", flush=True)
    real = TotalSegInContextDataset(
        root=TSEG, classes=TSEG_CLASSES, image_size=(T, T, T), split="train",
        context_size=K, aug_cfg=None, p_synth=0.0, class_balanced=True,
        use_crop=True, crop_spacing_mm=CROP_MM)

    print("\n=== workers=0 per-item wall time ===", flush=True)
    r0 = time_workers0(real, a.items)
    g0 = time_workers0(gmm, a.items)
    print(f"REAL: {r0:7.1f} ms/item", flush=True)
    print(f"GMM : {g0:7.1f} ms/item  (load {gmm.t_load / a.items * 1e3:.1f} + "
          f"paint {gmm.t_paint / a.items * 1e3:.1f} ms, ×{K + 1} vols)", flush=True)
    print(f"  → GMM is {r0 / g0:.2f}× the real per-item speed", flush=True)

    print(f"\n=== num_workers={a.workers} throughput ===", flush=True)
    rt = time_loader(real, a.items, a.workers)
    gt = time_loader(gmm, a.items, a.workers)
    print(f"REAL: {rt:6.1f} items/s", flush=True)
    print(f"GMM : {gt:6.1f} items/s  ({gt / rt:.2f}× real)", flush=True)


if __name__ == "__main__":
    main()
