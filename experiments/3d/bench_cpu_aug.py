"""Benchmark CPU augmentation cost with and without GIN / IPA.

Times the per-item training augmentation path that the dataloader runs in each
worker (src/totalseg_dataloader_incontext.py:1206-1208):

    all_images, all_masks = apply_task_aug(all_images, all_masks, aug.task)
    for i in range(N):
        all_images[i] = apply_intensity_aug(all_images[i], aug.intensity)

GIN/IPA lives only in apply_intensity_aug, so we compare three intensity
settings at the shapes of experiment 42 (image_size 128^3, context_size 3 ->
N = K+1 = 4 volumes per task):

    off  : augmentations.intensity.gin.p = 0   (current default)
    gin  : gin.p = 1, mode = gin
    ipa  : gin.p = 1, mode = ipa               (config default mode)

The resolved aug config is the real one: nnunet base <- exp-42 overrides.

Single torch thread by default to mimic a per-worker slice of a saturated
DataLoader pool (nfs cluster: num_workers=16). Pass --threads 0 for torch
default.

Usage:
    python experiments/3d/bench_cpu_aug.py
    python experiments/3d/bench_cpu_aug.py --size 128 --n 4 --iters 30
"""
import argparse
import statistics
import sys
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.augmentations import apply_task_aug, apply_intensity_aug   # noqa: E402
from src.totalseg_dataset import CT_NORM_MIN, CT_NORM_MAX           # noqa: E402

NNUNET = ROOT / "configs/augmentations/nnunet.yaml"
EXP42 = ROOT / "configs/experiment/3d/experiment/42_reg_to_all.yaml"


def resolved_aug():
    """augmentations subtree: nnunet base merged with experiment 42 overrides."""
    base = OmegaConf.load(NNUNET).augmentations
    exp = OmegaConf.load(EXP42).get("augmentations", {}) or {}
    return OmegaConf.merge(base, exp)


def make_data(n, size, seed):
    g = torch.Generator().manual_seed(seed)
    span = CT_NORM_MAX - CT_NORM_MIN
    imgs = torch.rand(n, 1, *size, generator=g) * span + CT_NORM_MIN
    masks = (torch.rand(n, *size, generator=g) > 0.7).long()
    return imgs, masks


def run_item(aug, imgs, masks):
    """One task's aug: shared geometric + independent per-volume intensity."""
    imgs, masks = apply_task_aug(imgs.clone(), masks.clone(), aug.task)
    for i in range(imgs.shape[0]):
        imgs[i] = apply_intensity_aug(imgs[i], aug.intensity)
    return imgs, masks


def run_intensity_only(intensity_cfg, imgs):
    """Isolate intensity aug (per-volume) with no geometric task aug."""
    for i in range(imgs.shape[0]):
        imgs[i] = apply_intensity_aug(imgs[i].clone(), intensity_cfg)
    return imgs


def timeit(fn, iters, warmup, seed0, n, size):
    for w in range(warmup):
        fn(*make_data(n, size, seed0 + w))
    ts = []
    for it in range(iters):
        imgs, masks = make_data(n, size, seed0 + 1000 + it)
        t0 = time.perf_counter()
        fn(imgs, masks)
        ts.append((time.perf_counter() - t0) * 1e3)   # ms
    return ts


def stats(ts):
    return (statistics.mean(ts), statistics.pstdev(ts), statistics.median(ts))


def set_gin(aug, *, p, mode):
    aug.intensity.gin.p = p
    aug.intensity.gin.mode = mode
    return aug


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--n", type=int, default=4, help="N = K+1 volumes per task")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--threads", type=int, default=1, help="torch intra-op threads; 0 = default")
    args = ap.parse_args()

    if args.threads > 0:
        torch.set_num_threads(args.threads)
    size = (args.size,) * 3
    print(f"shapes: N={args.n} volumes of {size}, torch_threads="
          f"{torch.get_num_threads()}, iters={args.iters} (warmup {args.warmup})\n")

    # --- full per-item path: task aug + per-volume intensity -----------------
    configs = [
        ("off (gin.p=0)", dict(p=0.0, mode="gin")),
        ("gin  (p=1)", dict(p=1.0, mode="gin")),
        ("ipa  (p=1)", dict(p=1.0, mode="ipa")),
    ]
    print("== FULL per-item aug: apply_task_aug + apply_intensity_aug x N ==")
    print(f"{'setting':<16}{'mean ms':>10}{'std':>8}{'median':>9}{'vs off':>9}")
    base_mean = None
    for label, gin in configs:
        aug = set_gin(resolved_aug(), **gin)
        fn = (lambda a: lambda i, mk: run_item(a, i, mk))(aug)
        m, s, med = stats(timeit(fn, args.iters, args.warmup, 0, args.n, size))
        if base_mean is None:
            base_mean = m
        print(f"{label:<16}{m:>10.1f}{s:>8.1f}{med:>9.1f}{m - base_mean:>+9.1f}")

    # --- isolated: intensity aug with ONLY gin/ipa active --------------------
    print("\n== ISOLATED intensity cost (all other intensity ops p=0) ==")
    print(f"{'setting':<16}{'mean ms':>10}{'std':>8}{'median':>9}  (per N-volume task)")
    for label, gin in [("gin only", dict(p=1.0, mode="gin")),
                       ("ipa only", dict(p=1.0, mode="ipa"))]:
        aug = resolved_aug()
        icfg = OmegaConf.create(OmegaConf.to_container(aug.intensity, resolve=True))
        for key in ("brightness_contrast", "gamma", "gaussian_noise",
                    "gaussian_blur", "simulate_low_resolution"):
            if key in icfg and icfg[key] is not None and "p" in icfg[key]:
                icfg[key].p = 0.0
        icfg.gin.p = gin["p"]
        icfg.gin.mode = gin["mode"]

        def fn(imgs, masks, _icfg=icfg):
            return run_intensity_only(_icfg, imgs)

        m, s, med = stats(timeit(fn, args.iters, args.warmup, 0, args.n, size))
        print(f"{label:<16}{m:>10.1f}{s:>8.1f}{med:>9.1f}")


if __name__ == "__main__":
    main()
