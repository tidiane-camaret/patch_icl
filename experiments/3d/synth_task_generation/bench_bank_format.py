"""
Bench GMM mask-bank storage formats against the REAL access pattern (mmap organ-crop a
subregion, not whole-volume load). Answers: does npz/compression or local staging save
dataloading time / disk before we write the full 5164-mask bank?

Formats:
  npy_nfs    : current — .npy + mmap partial-read crop, over NFS
  npy_local  : same, staged on node-local /tmp (nvme)
  npz_nfs    : np.savez_compressed (zlib) full-array load + crop, over NFS
  npz_local  : same, local
Disk: per-mask bytes for npy vs npz_compressed.
"""
import pickle
import random
import shutil
import time
from pathlib import Path

import numpy as np

from src.totalseg_dataloader_incontext import organ_crop_arrays

BANK = Path("/nfs/data/nii/data1/Analysis/camaret___in_context_segmentation/"
            "ANALYSIS_20251122/data/gmm_bank")
LOCAL = Path("/tmp/gmm_bank_bench")
T, CROP_MM, N = 128, 1.5, 30


def crop(arr, e, cls):
    """Replicate SynthGmmMaisiDataset._crop_multiclass load+crop (no interpolate)."""
    arr = np.squeeze(arr)
    center = tuple(e["cents"][cls][:3])
    _, crop_lbl, *_ = organ_crop_arrays(arr, arr, center, e["spacing"],
                                         image_size=(T,) * 3, crop_mm=CROP_MM,
                                         jitter=0, rng=random.Random(0))
    return crop_lbl


def main():
    idx = pickle.load(open(BANK / "index.pkl", "rb"))
    entries = idx["entries"]
    rng = np.random.default_rng(0)
    sample = [entries[i] for i in rng.choice(len(entries), N, replace=False)]
    # each entry: pick a class it contains
    picks = [(e, int(next(iter(e["label_list"])))) for e in sample]

    # ---- stage local npy, build npz_compressed both local and on a temp NFS dir ----
    (LOCAL / "masks").mkdir(parents=True, exist_ok=True)
    nfs_npz = BANK / "_bench_npz"; nfs_npz.mkdir(exist_ok=True)
    npy_bytes = npz_bytes = 0
    for e, _ in picks:
        src = BANK / "masks" / e["file"]
        dst_npy = LOCAL / "masks" / e["file"]
        if not dst_npy.exists():
            shutil.copy(src, dst_npy)
        stem = Path(e["file"]).with_suffix(".npz").name
        dst_npz = LOCAL / "masks" / stem
        nfs_dst = nfs_npz / stem
        if not dst_npz.exists():
            a = np.load(dst_npy)
            np.savez_compressed(dst_npz, a=a)
        if not nfs_dst.exists():
            shutil.copy(dst_npz, nfs_dst)
        npy_bytes += dst_npy.stat().st_size
        npz_bytes += dst_npz.stat().st_size

    def run(label, loader):
        # warm + timed (drop first for cache); time load and crop separately
        tl = tc = 0.0
        for e, cls in picks:
            t0 = time.perf_counter(); arr = loader(e); t1 = time.perf_counter()
            _ = crop(arr, e, cls); t2 = time.perf_counter()
            tl += t1 - t0; tc += t2 - t1
        n = len(picks)
        print(f"{label:12s} load {tl/n*1e3:7.1f} ms  crop {tc/n*1e3:6.1f} ms  "
              f"total {(tl+tc)/n*1e3:7.1f} ms/mask")

    print(f"masks={N}  T={T}  crop={CROP_MM}mm\n")
    print(f"disk/mask  npy {npy_bytes/N/1e6:6.2f} MB   "
          f"npz_zlib {npz_bytes/N/1e6:6.2f} MB   "
          f"ratio {npy_bytes/max(npz_bytes,1):4.1f}x\n")

    stem = lambda e: Path(e["file"]).with_suffix(".npz").name
    run("npy_nfs",   lambda e: np.load(BANK / "masks" / e["file"], mmap_mode="r"))
    run("npy_local", lambda e: np.load(LOCAL / "masks" / e["file"], mmap_mode="r"))
    run("npz_nfs",   lambda e: np.load(nfs_npz / stem(e))["a"])
    run("npz_local", lambda e: np.load(LOCAL / "masks" / stem(e))["a"])

    shutil.rmtree(nfs_npz)  # clean temp NFS npz


if __name__ == "__main__":
    main()
