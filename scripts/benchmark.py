"""
Profile VRAM, RAM, and forward/backward time for in-context segmentation models.

Usage
-----
    # default: compare both models at all configs in the matrix below
    python scripts/profile.py

    # single model, override matrix
    python scripts/profile.py --model resenc_in_context --image_size 64 --batch_size 1 --context_size 3

    # forward-only (no grad) — useful for inference budgeting
    python scripts/profile.py --no_grad
"""

import argparse
import gc
import sys
import time
from pathlib import Path
from itertools import product
from dataclasses import dataclass, field

import psutil
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.vit_in_context import ViTInContext3D
from src.models.resenc_in_context import ResEncInContext3D


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


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",        default=None, choices=["vit_in_context", "resenc_in_context"])
    parser.add_argument("--image_size",   type=int,   default=None)
    parser.add_argument("--batch_size",   type=int,   default=None)
    parser.add_argument("--context_size", type=int,   default=None)
    parser.add_argument("--no_grad",      action="store_true")
    parser.add_argument("--n_runs",       type=int,   default=3)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    if device.type == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(device)}")
        total_vram = torch.cuda.get_device_properties(device).total_memory / 1e9
        print(f"VRAM   : {total_vram:.1f} GB")
    print(f"Mode   : {'forward only' if args.no_grad else 'forward + backward'}\n")

    # Build sweep matrix (or single config from args)
    models      = [args.model]       if args.model        else ["vit_in_context", "resenc_in_context"]
    image_sizes = [args.image_size]  if args.image_size   else [64, 128]
    batch_sizes = [args.batch_size]  if args.batch_size   else [1, 2]
    ctx_sizes   = [args.context_size]if args.context_size else [3]

    configs = [
        RunConfig(model=m, image_size=s, batch_size=b, context_size=k, with_grad=not args.no_grad)
        for m, s, b, k in product(models, image_sizes, batch_sizes, ctx_sizes)
    ]

    _header()
    for cfg in configs:
        try:
            res = profile_one(cfg, device, n_runs=args.n_runs)
            _row(cfg, res)
        except torch.cuda.OutOfMemoryError:
            print(f"{cfg.model:<22} {cfg.image_size:>5} {cfg.batch_size:>3} {cfg.context_size:>3}  "
                  f"{'OOM':>43}")
            if device.type == "cuda":
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"{cfg.model:<22} {cfg.image_size:>5} {cfg.batch_size:>3} {cfg.context_size:>3}  "
                  f"ERROR: {e}")


if __name__ == "__main__":
    main()
