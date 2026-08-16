"""Benchmark each augmentation on CPU (per-item functional) vs GPU (batched).

CPU path = what the DataLoader workers run today: `apply_task_aug` once per task
(shared geometric over its T=K+1 volumes) + `apply_intensity_aug` per volume
(src/augmentations.py). GPU path = the batched engine (src/gpu_augment.py):
`_geometric(group_size=T)` over the whole batch and `_batched_intensity` /
`_batched_gin_ipa` over all B*T volumes at once.

Each augmentation is ISOLATED by zeroing every probability in its cfg subtree
and forcing the one op to p=1, so the reported time is that op's cost (not the
p-gate skip). A final FULL row runs the real resolved config (nnunet <- exp-42).

CPU ms is the single-thread cost to augment the WHOLE batch (B*T volumes) — the
total per-worker work. A real DataLoader spreads this over `num_workers`
processes, so effective CPU wall-time ≈ CPU_total / num_workers. GPU ms is the
batched wall-time (one call, cuda-synchronized). `speedup = CPU_total / GPU`.

Usage:
    UV_PROJECT_ENVIRONMENT=.venv_thor uv run python experiments/3d/bench_aug_cpu_vs_gpu.py
    .venv_thor/bin/python experiments/3d/bench_aug_cpu_vs_gpu.py --size 128 --batch 4 --context 3
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

from src.augmentations import apply_task_aug, apply_intensity_aug, apply_per_image_aug  # noqa: E402
from src.gpu_augment import _geometric, _batched_intensity                 # noqa: E402
from src.totalseg_dataset import CT_NORM_MIN, CT_NORM_MAX                   # noqa: E402

NNUNET = ROOT / "configs/augmentations/nnunet.yaml"
EXP42 = ROOT / "configs/experiment/3d/experiment/42_reg_to_all.yaml"


def resolved_aug(use_exp42=False):
    """augmentations subtree. Defaults to the nnunet base — the canonical schema
    the CPU/GPU augmentation code implements. `--exp42` merges the experiment-42
    overrides, but note exp-42 is mid-migration to a different (albumentations-style)
    intensity schema whose keys the code does not consume, so the merge is a mongrel;
    use it only to time the geometric ops under exp-42 probabilities."""
    base = OmegaConf.load(NNUNET).augmentations
    if not use_exp42:
        return base
    exp = OmegaConf.load(EXP42).get("augmentations", {}) or {}
    return OmegaConf.merge(base, exp)


def _clone_cfg(node):
    return OmegaConf.create(OmegaConf.to_container(node, resolve=True))


def intensity_only(intensity, op):
    """Copy of the intensity cfg with every op p=0 except `op` (forced p=1).

    op is a key like 'brightness_contrast'/'gamma'/'gaussian_noise'/'gaussian_blur'/
    'simulate_low_resolution', or 'gin'/'ipa' (both drive the gin block).
    """
    icfg = _clone_cfg(intensity)
    for key in ("brightness_contrast", "gamma", "gaussian_noise",
                "gaussian_blur", "simulate_low_resolution", "sharpness", "gin"):
        if key in icfg and icfg[key] is not None and "p" in icfg[key]:
            icfg[key].p = 0.0
    if op in ("gin", "ipa"):
        icfg.gin.p = 1.0
        icfg.gin.mode = op
    else:
        icfg[op].p = 1.0
    return icfg


def task_only(task, op):
    """Copy of the task (geometric) cfg with only `op` enabled at p=1.

    op is 'flip' | 'affine' | 'elastic'.
    """
    tcfg = _clone_cfg(task)
    tcfg.flip.p_d = tcfg.flip.p_h = tcfg.flip.p_w = 0.0
    tcfg.affine.p = 0.0
    if "elastic" in tcfg and tcfg.elastic is not None:
        tcfg.elastic.p = 0.0
    if op == "flip":
        tcfg.flip.p_d = tcfg.flip.p_h = tcfg.flip.p_w = 1.0
    elif op == "affine":
        tcfg.affine.p = 1.0
    elif op == "elastic":
        tcfg.elastic.p = 1.0
    return tcfg


def make_batch(B, T, size, device, seed):
    """B tasks x T volumes. Returns (vols (B*T,1,D,H,W), masks (B*T,D,H,W))."""
    g = torch.Generator().manual_seed(seed)
    span = CT_NORM_MAX - CT_NORM_MIN
    vols = (torch.rand(B * T, 1, *size, generator=g) * span + CT_NORM_MIN)
    masks = (torch.rand(B * T, *size, generator=g) > 0.7).long()
    return vols.to(device), masks.to(device)


# ----------------------------- CPU runners --------------------------------
def cpu_intensity(icfg, vols):
    for i in range(vols.shape[0]):
        apply_intensity_aug(vols[i].clone(), icfg)


def cpu_task(tcfg, vols, masks, T):
    B = vols.shape[0] // T
    for b in range(B):
        sl = slice(b * T, (b + 1) * T)
        apply_task_aug(vols[sl].clone(), masks[sl].clone(), tcfg)


def cpu_perimage(tcfg, vols, masks):
    """Independent per-volume geometric (apply_per_image_aug), one volume at a time."""
    for i in range(vols.shape[0]):
        apply_per_image_aug(vols[i].clone(), masks[i].clone(), tcfg)


def cpu_full(aug, vols, masks, T):
    B = vols.shape[0] // T
    for b in range(B):
        sl = slice(b * T, (b + 1) * T)
        img, _ = apply_task_aug(vols[sl].clone(), masks[sl].clone(), aug.task)
        for i in range(img.shape[0]):
            apply_intensity_aug(img[i], aug.intensity)


# ----------------------------- GPU runners --------------------------------
def gpu_intensity(icfg, vols, gen):
    _batched_intensity(vols.clone(), icfg, gen)


def gpu_task(tcfg, vols, masks, T, gen):
    _geometric(vols.clone(), masks.clone(), group_size=T, cfg=tcfg, gen=gen)


def gpu_perimage(tcfg, vols, masks, gen):
    """Independent per-volume geometric on GPU: _geometric with group_size=1."""
    _geometric(vols.clone(), masks.clone(), group_size=1, cfg=tcfg, gen=gen)


def gpu_full(aug, vols, masks, T, gen):
    v, m = _geometric(vols.clone(), masks.clone(), group_size=T, cfg=aug.task, gen=gen)
    _batched_intensity(v, aug.intensity, gen)


# ------------------------------- timing -----------------------------------
def time_cpu(fn, iters, warmup):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1e3)
    return ts


def time_gpu(fn, iters, warmup, device):
    for _ in range(warmup):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    return ts


def stat(ts):
    return statistics.median(ts), statistics.pstdev(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--batch", type=int, default=4, help="B tasks per batch")
    ap.add_argument("--context", type=int, default=3, help="K -> T = K+1 volumes per task")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--threads", type=int, default=1, help="CPU torch threads (per-worker slice); 0=default")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--exp42", action="store_true",
                    help="merge experiment-42 overrides (mongrel intensity schema; geometric only)")
    args = ap.parse_args()

    if args.threads > 0:
        torch.set_num_threads(args.threads)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    T = args.context + 1
    size = (args.size,) * 3
    B = args.batch
    N = B * T

    aug = resolved_aug(use_exp42=args.exp42)
    vols, masks = make_batch(B, T, size, device, seed=0)
    vols_cpu, masks_cpu = vols.cpu(), masks.cpu()
    gen = torch.Generator(device=device).manual_seed(0)

    gpu_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
    print(f"config: {'nnunet <- exp-42' if args.exp42 else 'nnunet (canonical schema)'}")
    print(f"batch: B={B} tasks x T={T} vols = {N} volumes of {size}  |  device={device} ({gpu_name})")
    print(f"CPU torch_threads={torch.get_num_threads()}  |  iters={args.iters} (warmup {args.warmup})")
    print("CPU ms = single-thread total to aug the whole batch (real pool spreads over num_workers).\n")

    header = f"{'augmentation':<26}{'CPU ms':>12}{'GPU ms':>12}{'speedup':>10}"
    print(header); print("-" * len(header))

    def row(label, cpu_fn, gpu_fn):
        c_med, c_sd = stat(time_cpu(cpu_fn, args.iters, args.warmup))
        g_med, g_sd = stat(time_gpu(gpu_fn, args.iters, args.warmup, device))
        sp = c_med / g_med if g_med > 0 else float("nan")
        print(f"{label:<26}{c_med:>8.1f}±{c_sd:<3.0f}{g_med:>8.1f}±{g_sd:<3.0f}{sp:>9.1f}x")

    # geometric TASK: one shared transform across a task's T volumes (group_size=T)
    for op in ("flip", "affine", "elastic"):
        tcfg = task_only(aug.task, op)
        row(f"task/{op} (shared)",
            lambda t=tcfg: cpu_task(t, vols_cpu, masks_cpu, T),
            lambda t=tcfg: gpu_task(t, vols, masks, T, gen))

    # geometric PER-IMAGE: independent transform per volume (group_size=1). CPU runs
    # apply_per_image_aug one volume at a time; GPU batches all N with per-volume thetas.
    for op in ("flip", "affine", "elastic"):
        tcfg = task_only(aug.per_image, op)
        row(f"per_image/{op} (indep)",
            lambda t=tcfg: cpu_perimage(t, vols_cpu, masks_cpu),
            lambda t=tcfg: gpu_perimage(t, vols, masks, gen))

    # intensity (per-volume)
    for op in ("brightness_contrast", "gamma", "gaussian_noise",
               "gaussian_blur", "simulate_low_resolution", "gin", "ipa"):
        icfg = intensity_only(aug.intensity, op)
        row(f"intensity/{op}",
            lambda c=icfg: cpu_intensity(c, vols_cpu),
            lambda c=icfg: gpu_intensity(c, vols, gen))

    # full resolved pipeline (task + intensity, real probabilities)
    print("-" * len(header))
    row("FULL (resolved cfg)",
        lambda: cpu_full(aug, vols_cpu, masks_cpu, T),
        lambda: gpu_full(aug, vols, masks, T, gen))


if __name__ == "__main__":
    main()
