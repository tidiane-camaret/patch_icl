"""
Profile VRAM, RAM, and forward/backward time for in-context segmentation models.
Optionally benchmarks the real dataloader to surface I/O bottlenecks.

Usage
-----
    # model benchmark only (synthetic inputs)
    python scripts/benchmark.py

    # single model, override matrix
    python scripts/benchmark.py --model resenc_in_context --image_size 64 --batch_size 1 --context_size 3

    # forward-only (no grad) — useful for inference budgeting
    python scripts/benchmark.py --no_grad

    # dataloader benchmark (needs --root)
    python scripts/benchmark.py --bench_data --root /nfs/data/.../totalseg

    # both
    python scripts/benchmark.py --bench_data --root /nfs/data/.../totalseg --image_size 64 128
"""

import argparse
import gc
import io
import json
import sys
import time
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path
from itertools import product
from dataclasses import dataclass

import psutil
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.vit_in_context import ViTInContext3D
from src.models.resenc_in_context import ResEncInContext3D
from src.totalseg_dataloader_incontext import TotalSegInContextDataset, incontext_collate_fn
from src.augmentations import apply_task_aug, apply_intensity_aug

_CONFIG_PATH = Path(__file__).parent.parent / "configs" / "config.yaml"
_AUG_CONFIG_PATH = Path(__file__).parent.parent / "configs" / "augmentations.yaml"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class RunConfig:
    model:        str
    image_size:   int
    batch_size:   int
    context_size: int
    with_grad:    bool = True

    # ResEncInContext3D
    features_per_stage: tuple = (32, 64, 128, 256)
    rope_theta:         float = 100.0

    # ViTInContext3D
    patch_size:   int   = 8
    embed_dim:    int   = 256

    # shared
    depth_stage1: int   = 3
    depth_stage2: int   = 3
    num_heads:    int   = 8


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build(cfg: RunConfig) -> nn.Module:
    sz = (cfg.image_size,) * 3
    shared = dict(
        image_size=sz, in_channels=1, num_classes=2,
        depth_stage1=cfg.depth_stage1, depth_stage2=cfg.depth_stage2,
        num_heads=cfg.num_heads,
    )
    if cfg.model == "vit_in_context":
        return ViTInContext3D(**shared,
                             patch_size=(cfg.patch_size,) * 3,
                             embed_dim=cfg.embed_dim)
    if cfg.model == "resenc_in_context":
        return ResEncInContext3D(**shared,
                                features_per_stage=cfg.features_per_stage,
                                rope_theta=cfg.rope_theta)
    raise ValueError(cfg.model)


# ---------------------------------------------------------------------------
# Profiling
# ---------------------------------------------------------------------------

def _ram_mb() -> float:
    return psutil.Process().memory_info().rss / 1e6


def profile_one(cfg: RunConfig, device: torch.device, n_warmup: int = 1, n_runs: int = 3):
    is_cuda = device.type == "cuda"

    # --- build & move model -------------------------------------------------
    ram_before = _ram_mb()
    model = build(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    ram_model = _ram_mb() - ram_before

    # --- dummy inputs -------------------------------------------------------
    B, K = cfg.batch_size, cfg.context_size
    sz = (cfg.image_size,) * 3
    def make_inputs():
        return (
            torch.randn(B, 1, *sz,     device=device),
            torch.randn(B, K, 1, *sz,  device=device),
            torch.zeros(B, K, *sz,     device=device),
        )

    # --- warmup -------------------------------------------------------------
    for _ in range(n_warmup):
        tgt, ctx_i, ctx_m = make_inputs()
        if cfg.with_grad:
            logits = model(tgt, ctx_i, ctx_m)
            logits.mean().backward()
            model.zero_grad(set_to_none=True)
        else:
            with torch.no_grad():
                model(tgt, ctx_i, ctx_m)
        if is_cuda:
            torch.cuda.synchronize()

    # --- peak VRAM ----------------------------------------------------------
    if is_cuda:
        torch.cuda.reset_peak_memory_stats(device)

    elapsed = []
    for _ in range(n_runs):
        tgt, ctx_i, ctx_m = make_inputs()
        if is_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        if cfg.with_grad:
            logits = model(tgt, ctx_i, ctx_m)
            logits.mean().backward()
            model.zero_grad(set_to_none=True)
        else:
            with torch.no_grad():
                logits = model(tgt, ctx_i, ctx_m)

        if is_cuda:
            torch.cuda.synchronize()
        elapsed.append(time.perf_counter() - t0)

    vram_peak_mb = torch.cuda.max_memory_allocated(device) / 1e6 if is_cuda else float("nan")

    # --- cleanup ------------------------------------------------------------
    del model, logits, tgt, ctx_i, ctx_m
    gc.collect()
    if is_cuda:
        torch.cuda.empty_cache()

    return dict(
        n_params_M   = round(n_params, 1),
        vram_peak_MB = round(vram_peak_mb),
        ram_model_MB = round(ram_model),
        time_mean_ms = round(1000 * sum(elapsed) / len(elapsed)),
        time_min_ms  = round(1000 * min(elapsed)),
    )


# ---------------------------------------------------------------------------
# Dataloader profiling
# ---------------------------------------------------------------------------

_BENCH_CLASSES = ["liver", "lung_lower_lobe_right", "heart", "aorta", "spleen"]


def profile_loader(
    root: str,
    image_size: int,
    batch_size: int,
    context_size: int,
    num_workers: int = 4,
    max_subjects: int = 100,
    n_batches: int = 20,
    device: torch.device = torch.device("cpu"),
) -> dict:
    """
    Time real DataLoader batches over a small class subset.
    Reports per-batch load+transfer time and whether the fast pre-resized path is active.
    """
    size_str = f"{image_size}x{image_size}x{image_size}"
    root_path = Path(root)

    # Check fast-path availability on first subject
    first_subj = next(p for p in sorted(root_path.iterdir()) if p.is_dir())
    has_fast_path = (first_subj / f"ct_{size_str}.npy").exists()

    # Build dataset; suppress init prints
    with redirect_stdout(io.StringIO()):
        ds = TotalSegInContextDataset(
            root=root,
            classes=_BENCH_CLASSES,
            image_size=(image_size,) * 3,
            split="train",
            context_size=context_size,
            max_subjects=max_subjects,
        )

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=incontext_collate_fn,
        pin_memory=(device.type == "cuda"),
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    is_cuda = device.type == "cuda"
    times: list[float] = []
    it = iter(loader)

    # warmup: one batch (prefetch workers settle)
    batch = next(it)
    if is_cuda:
        batch["image"].to(device, non_blocking=True)

    for _, batch in zip(range(n_batches), it):
        if is_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        img = batch["image"].to(device, non_blocking=True)
        ctx = batch["context_in"].to(device, non_blocking=True)
        if is_cuda:
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    del loader, ds
    gc.collect()

    n = len(times)
    mean_ms = round(1000 * sum(times) / n) if n else 0
    min_ms  = round(1000 * min(times))     if n else 0
    # bytes per batch: (B * (1+K) * 1 * D³ * 4 bytes)
    vol_bytes = batch_size * (1 + context_size) * image_size**3 * 4
    throughput_mbs = round(vol_bytes / (sum(times) / n) / 1e6) if n else 0

    return dict(
        has_fast_path   = has_fast_path,
        n_batches_timed = n,
        mean_ms         = mean_ms,
        min_ms          = min_ms,
        throughput_MBs  = throughput_mbs,
    )


# ---------------------------------------------------------------------------
# Augmentation profiling
# ---------------------------------------------------------------------------

def profile_augmentations(aug_cfg, image_size: int, context_size: int, n_runs: int = 20) -> dict:
    """
    Time task + intensity augmentations on CPU synthetic tensors.
    Mirrors exactly what __getitem__ does: one (K+1)-volume item per call.
    """
    K1 = context_size + 1
    sz = (image_size,) * 3
    images = torch.randn(K1, 1, *sz)
    masks  = torch.zeros(K1, *sz, dtype=torch.long)

    times_task      = []
    times_intensity = []

    for _ in range(n_runs):
        imgs, msks = images.clone(), masks.clone()

        t0 = time.perf_counter()
        imgs, msks = apply_task_aug(imgs, msks, aug_cfg.task)
        times_task.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        for i in range(K1):
            imgs[i] = apply_intensity_aug(imgs[i], aug_cfg.intensity)
        times_intensity.append(time.perf_counter() - t0)

    mean_task      = sum(times_task) / n_runs
    mean_intensity = sum(times_intensity) / n_runs
    return {
        "task_aug_mean_ms":      round(1000 * mean_task),
        "intensity_aug_mean_ms": round(1000 * mean_intensity),
        "total_aug_mean_ms":     round(1000 * (mean_task + mean_intensity)),
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

COL_W = {
    "model": 22, "img": 5, "B": 3, "K": 3,
    "params": 8, "vram": 10, "ram": 8, "t_mean": 10, "t_min": 9,
}

def _header():
    h = (f"{'model':<22} {'img':>5} {'B':>3} {'K':>3} "
         f"{'params':>8} {'vram(MB)':>10} {'ram(MB)':>8} "
         f"{'t_mean(ms)':>10} {'t_min(ms)':>9}")
    print(h)
    print("-" * len(h))

def _row(cfg: RunConfig, res: dict):
    print(f"{cfg.model:<22} {cfg.image_size:>5} {cfg.batch_size:>3} {cfg.context_size:>3} "
          f"{res['n_params_M']:>7.1f}M "
          f"{res['vram_peak_MB']:>10} "
          f"{res['ram_model_MB']:>8} "
          f"{res['time_mean_ms']:>10} "
          f"{res['time_min_ms']:>9}")


def _loader_header():
    h = (f"{'img':>5} {'B':>3} {'K':>3} {'fast_path':>10} "
         f"{'mean(ms)':>10} {'min(ms)':>8} {'MB/s':>8}")
    print(h)
    print("-" * len(h))

def _loader_row(image_size, batch_size, context_size, res: dict):
    print(f"{image_size:>5} {batch_size:>3} {context_size:>3} "
          f"{'yes' if res['has_fast_path'] else 'NO':>10} "
          f"{res['mean_ms']:>10} "
          f"{res['min_ms']:>8} "
          f"{res['throughput_MBs']:>8}")


def _aug_header():
    h = (f"{'img':>5} {'K':>3}  "
         f"{'task_aug(ms)':>14} {'intensity_aug(ms)':>18} {'total(ms)':>11}")
    print(h)
    print("-" * len(h))

def _aug_row(image_size, context_size, res: dict):
    print(f"{image_size:>5} {context_size:>3}  "
          f"{res['task_aug_mean_ms']:>14} "
          f"{res['intensity_aug_mean_ms']:>18} "
          f"{res['total_aug_mean_ms']:>11}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    # Load project config for defaults — same source of truth as train.py
    yaml_cfg = OmegaConf.load(_CONFIG_PATH)

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # sweep axes — each accepts one or more values
    parser.add_argument("--model",        nargs="+",
                        default=[yaml_cfg.model.name],
                        choices=["vit_in_context", "resenc_in_context"])
    parser.add_argument("--image_size",   type=int, nargs="+",
                        default=[yaml_cfg.data.image_size[0]])
    parser.add_argument("--batch_size",   type=int, nargs="+",
                        default=[yaml_cfg.train.batch_size])
    parser.add_argument("--context_size", type=int, nargs="+",
                        default=[yaml_cfg.data.context_size])
    parser.add_argument("--no_grad",      action="store_true")
    parser.add_argument("--n_runs",       type=int, default=3)
    # dataloader benchmark
    parser.add_argument("--bench_data",   action="store_true",
                        help="benchmark real dataloader in addition to model")
    parser.add_argument("--root",         default=yaml_cfg.paths.totalseg,
                        help="TotalSegmentator dataset root for --bench_data")
    parser.add_argument("--loader_workers", type=int, default=yaml_cfg.train.workers)
    parser.add_argument("--loader_batches", type=int, default=20,
                        help="number of batches to time in dataloader benchmark")
    # augmentation benchmark
    parser.add_argument("--bench_aug", action="store_true",
                        help="benchmark CPU augmentation time per dataset item")
    parser.add_argument("--aug_runs",  type=int, default=20,
                        help="number of augmentation calls to average over")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    if device.type == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(device)}")
        total_vram = torch.cuda.get_device_properties(device).total_memory / 1e9
        print(f"VRAM   : {total_vram:.1f} GB")
    print(f"Config : {_CONFIG_PATH}")
    print(f"Mode   : {'forward only' if args.no_grad else 'forward + backward'}\n")

    models      = args.model
    image_sizes = args.image_size
    batch_sizes = args.batch_size
    ctx_sizes   = args.context_size

    results_dir = Path(yaml_cfg.paths.results)
    results_dir.mkdir(parents=True, exist_ok=True)

    # output[model][image_size] accumulates entries for the final JSON
    output: dict[str, dict[int, dict]] = {
        m: {s: {"dataloader": [], "augmentations": [], "model": []} for s in image_sizes}
        for m in models
    }

    # ------------------------------------------------------------------
    # Dataloader benchmark
    # ------------------------------------------------------------------
    if args.bench_data:
        print("=" * 55)
        print("Dataloader benchmark  (load + CPU→GPU transfer per batch)")
        print(f"Root    : {args.root}")
        print(f"Classes : {_BENCH_CLASSES}")
        print(f"Workers : {args.loader_workers}  |  batches timed: {args.loader_batches}")
        print("=" * 55)
        _loader_header()
        # dataloader results are model-agnostic; store once, copy to all models below
        loader_results: dict[int, list] = {s: [] for s in image_sizes}
        for s, b, k in product(image_sizes, batch_sizes, ctx_sizes):
            try:
                res = profile_loader(
                    root=args.root,
                    image_size=s,
                    batch_size=b,
                    context_size=k,
                    num_workers=args.loader_workers,
                    n_batches=args.loader_batches,
                    device=device,
                )
                _loader_row(s, b, k, res)
                loader_results[s].append({"batch_size": b, "context_size": k, **res})
            except Exception as e:
                print(f"{s:>5} {b:>3} {k:>3}  ERROR: {e}")
                loader_results[s].append({"batch_size": b, "context_size": k, "error": str(e)})
        print()

        for m in models:
            for s in image_sizes:
                output[m][s]["dataloader"] = loader_results[s]

    # ------------------------------------------------------------------
    # Augmentation benchmark
    # ------------------------------------------------------------------
    if args.bench_aug:
        aug_cfg = OmegaConf.load(_AUG_CONFIG_PATH).augmentations
        print("=" * 55)
        print("Augmentation benchmark  (CPU, one dataset item per call)")
        print(f"Runs per config: {args.aug_runs}")
        print("=" * 55)
        _aug_header()
        aug_results: dict[int, list] = {s: [] for s in image_sizes}
        for s, k in product(image_sizes, ctx_sizes):
            try:
                res = profile_augmentations(aug_cfg, s, k, n_runs=args.aug_runs)
                _aug_row(s, k, res)
                aug_results[s].append({"context_size": k, **res})
            except Exception as e:
                print(f"{s:>5} {k:>3}  ERROR: {e}")
                aug_results[s].append({"context_size": k, "error": str(e)})
        print()

        for m in models:
            for s in image_sizes:
                output[m][s]["augmentations"] = aug_results[s]

    # ------------------------------------------------------------------
    # Model benchmark — architecture params from config
    # ------------------------------------------------------------------
    m_cfg = yaml_cfg.model
    configs = [
        RunConfig(
            model=m, image_size=s, batch_size=b, context_size=k,
            with_grad=not args.no_grad,
            features_per_stage=tuple(m_cfg.features_per_stage),
            rope_theta=m_cfg.rope_theta,
            patch_size=m_cfg.patch_size[0],
            embed_dim=m_cfg.embed_dim,
            depth_stage1=m_cfg.depth_stage1,
            depth_stage2=m_cfg.depth_stage2,
            num_heads=m_cfg.num_heads,
        )
        for m, s, b, k in product(models, image_sizes, batch_sizes, ctx_sizes)
    ]

    print("=" * 90)
    print("Model benchmark  (synthetic inputs, forward + backward)")
    print("=" * 90)
    _header()
    for cfg in configs:
        try:
            res = profile_one(cfg, device, n_runs=args.n_runs)
            _row(cfg, res)
            output[cfg.model][cfg.image_size]["model"].append(
                {"batch_size": cfg.batch_size, "context_size": cfg.context_size, **res}
            )
        except torch.cuda.OutOfMemoryError:
            print(f"{cfg.model:<22} {cfg.image_size:>5} {cfg.batch_size:>3} {cfg.context_size:>3}  "
                  f"{'OOM':>43}")
            output[cfg.model][cfg.image_size]["model"].append(
                {"batch_size": cfg.batch_size, "context_size": cfg.context_size, "error": "OOM"}
            )
            if device.type == "cuda":
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"{cfg.model:<22} {cfg.image_size:>5} {cfg.batch_size:>3} {cfg.context_size:>3}  "
                  f"ERROR: {e}")
            output[cfg.model][cfg.image_size]["model"].append(
                {"batch_size": cfg.batch_size, "context_size": cfg.context_size, "error": str(e)}
            )

    # ------------------------------------------------------------------
    # Write JSON — one file per (model, image_size)
    # ------------------------------------------------------------------
    device_info = {"device": str(device)}
    if device.type == "cuda":
        device_info["gpu"] = torch.cuda.get_device_name(device)
        device_info["vram_total_GB"] = round(
            torch.cuda.get_device_properties(device).total_memory / 1e9, 1
        )

    arch_info = OmegaConf.to_container(yaml_cfg.model, resolve=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M")
    for m in models:
        for s in image_sizes:
            payload = {
                "device":  device_info,
                "arch":    arch_info,
                "with_grad": not args.no_grad,
                **output[m][s],
            }
            out_path = results_dir / f"benchmark_{m}_{s}_{ts}.json"
            out_path.write_text(json.dumps(payload, indent=2))
            print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
