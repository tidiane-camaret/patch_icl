"""
VRAM and throughput sweep for MultilevelICL.

Varies one parameter at a time over the following axes:
  image_size, num_levels, n_patches_l1, context_size, model.dim, model.num_layers

For each config: builds the model, runs N_BATCHES training steps (fwd + bwd +
optimizer), records peak VRAM and wall time, then frees memory before the next
config.  The STU-Net encoder is built once and reused.

Results saved to results/benchmarks/vram_sweep/<timestamp>_{json,csv}.

Usage
-----
    python experiments/multilevel/benchmark_vram.py
    python experiments/multilevel/benchmark_vram.py --cluster meta
    python experiments/multilevel/benchmark_vram.py --n_batches 5 --cluster meta
    python experiments/multilevel/benchmark_vram.py --groups dim num_layers
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
import time
from datetime import datetime
from itertools import islice
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, RandomSampler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from data.totalseg_classes import resolve_classes
from experiments.multilevel.model import MultilevelICL
from experiments.multilevel.train import (
    encode_image_only,
    extract_features,
    process_batch,
)
from src.models.encoders.stunet import STUNetEncoder
from src.totalseg_dataloader_incontext import (
    TotalSegInContextDataset,
    incontext_collate_fn,
)


# ---------------------------------------------------------------------------
# Sweep definitions
# ---------------------------------------------------------------------------

# Each entry: "group" + "label" identify the run; remaining keys override the
# base config using dot-notation keys.  All groups hold other params constant
# at the values listed in BASELINE so results are directly comparable.

BASELINE: dict = {
    "data.image_size":   [128, 128, 128],
    "data.resolutions":  [[8, 8, 8], [16, 16, 16]],
    "data.n_patches_l1": 512,
    "data.context_size": 1,
    "model.dim":         256,
    "model.num_layers":  4,
}

SWEEP: list[dict] = [
    # ---- image_size: encoder activation cost only -------------------------
    # 1 level + cheap model to isolate encoder cost; 256³/512³ may OOM.
    {"group": "image_size", "label": "64³",
     "data.image_size": [64, 64, 64],
     "data.resolutions": [[8, 8, 8]], "model.dim": 128, "model.num_layers": 2},
    {"group": "image_size", "label": "128³",
     "data.image_size": [128, 128, 128],
     "data.resolutions": [[8, 8, 8]], "model.dim": 128, "model.num_layers": 2},
    {"group": "image_size", "label": "256³",
     "data.image_size": [256, 256, 256],
     "data.resolutions": [[8, 8, 8]], "model.dim": 128, "model.num_layers": 2},
    {"group": "image_size", "label": "512³",
     "data.image_size": [512, 512, 512],
     "data.resolutions": [[8, 8, 8]], "model.dim": 128, "model.num_layers": 2},

    # ---- num_levels: resolution ladder doubles at each step ---------------
    # Finest-level encoder features dominate VRAM at 64³+; 128³/256³ will OOM.
    {"group": "num_levels", "label": "1  [8³]",
     "data.resolutions": [[8, 8, 8]],
     "model.dim": 128, "model.num_layers": 2},
    {"group": "num_levels", "label": "2  [→16³]",
     "data.resolutions": [[8, 8, 8], [16, 16, 16]],
     "model.dim": 128, "model.num_layers": 2},
    {"group": "num_levels", "label": "3  [→32³]",
     "data.resolutions": [[8, 8, 8], [16, 16, 16], [32, 32, 32]],
     "model.dim": 128, "model.num_layers": 2},
    {"group": "num_levels", "label": "4  [→64³]",
     "data.resolutions": [[8, 8, 8], [16, 16, 16], [32, 32, 32], [64, 64, 64]],
     "model.dim": 128, "model.num_layers": 2},
    {"group": "num_levels", "label": "5  [→128³]",
     "data.resolutions": [[8, 8, 8], [16, 16, 16], [32, 32, 32], [64, 64, 64], [128, 128, 128]],
     "model.dim": 128, "model.num_layers": 2},
    {"group": "num_levels", "label": "6  [→256³]",
     "data.resolutions": [[8, 8, 8], [16, 16, 16], [32, 32, 32], [64, 64, 64], [128, 128, 128], [256, 256, 256]],
     "model.dim": 128, "model.num_layers": 2},

    # ---- n_patches_l1: sparse level sequence length -----------------------
    # Use 3 levels [8³→16³→32³] so finest grid has 32³=32768 positions,
    # accommodating NP up to 8192 without topk overflow.
    {"group": "n_patches_l1", "label": "128",
     "data.n_patches_l1": 128,
     "data.resolutions": [[8, 8, 8], [16, 16, 16], [32, 32, 32]], "model.dim": 128, "model.num_layers": 2},
    {"group": "n_patches_l1", "label": "256",
     "data.n_patches_l1": 256,
     "data.resolutions": [[8, 8, 8], [16, 16, 16], [32, 32, 32]], "model.dim": 128, "model.num_layers": 2},
    {"group": "n_patches_l1", "label": "512",
     "data.n_patches_l1": 512,
     "data.resolutions": [[8, 8, 8], [16, 16, 16], [32, 32, 32]], "model.dim": 128, "model.num_layers": 2},
    {"group": "n_patches_l1", "label": "1024",
     "data.n_patches_l1": 1024,
     "data.resolutions": [[8, 8, 8], [16, 16, 16], [32, 32, 32]], "model.dim": 128, "model.num_layers": 2},
    {"group": "n_patches_l1", "label": "2048",
     "data.n_patches_l1": 2048,
     "data.resolutions": [[8, 8, 8], [16, 16, 16], [32, 32, 32]], "model.dim": 128, "model.num_layers": 2},
    {"group": "n_patches_l1", "label": "4096",
     "data.n_patches_l1": 4096,
     "data.resolutions": [[8, 8, 8], [16, 16, 16], [32, 32, 32]], "model.dim": 128, "model.num_layers": 2},
    {"group": "n_patches_l1", "label": "8192",
     "data.n_patches_l1": 8192,
     "data.resolutions": [[8, 8, 8], [16, 16, 16], [32, 32, 32]], "model.dim": 128, "model.num_layers": 2},

    # ---- context_size: multiplier on all context costs (ctx self-attn = K²) ---
    {"group": "context_size", "label": "K=1",
     "data.context_size": 1,
     "data.resolutions": [[8, 8, 8], [16, 16, 16]], "model.dim": 128, "model.num_layers": 2},
    {"group": "context_size", "label": "K=2",
     "data.context_size": 2,
     "data.resolutions": [[8, 8, 8], [16, 16, 16]], "model.dim": 128, "model.num_layers": 2},
    {"group": "context_size", "label": "K=4",
     "data.context_size": 4,
     "data.resolutions": [[8, 8, 8], [16, 16, 16]], "model.dim": 128, "model.num_layers": 2},

    # ---- model.dim: hidden size (O(dim²) params, O(N×dim) activations) ---
    {"group": "dim", "label": "64",
     "model.dim": 64,
     "data.resolutions": [[8, 8, 8], [16, 16, 16]], "model.num_layers": 4},
    {"group": "dim", "label": "128",
     "model.dim": 128,
     "data.resolutions": [[8, 8, 8], [16, 16, 16]], "model.num_layers": 4},
    {"group": "dim", "label": "256",
     "model.dim": 256,
     "data.resolutions": [[8, 8, 8], [16, 16, 16]], "model.num_layers": 4},
    {"group": "dim", "label": "512",
     "model.dim": 512,
     "data.resolutions": [[8, 8, 8], [16, 16, 16]], "model.num_layers": 4},

    # ---- num_layers: strictly linear in time and VRAM --------------------
    {"group": "num_layers", "label": "2",
     "model.num_layers": 2,
     "model.dim": 256, "data.resolutions": [[8, 8, 8], [16, 16, 16]]},
    {"group": "num_layers", "label": "4",
     "model.num_layers": 4,
     "model.dim": 256, "data.resolutions": [[8, 8, 8], [16, 16, 16]]},
    {"group": "num_layers", "label": "8",
     "model.num_layers": 8,
     "model.dim": 256, "data.resolutions": [[8, 8, 8], [16, 16, 16]]},
    {"group": "num_layers", "label": "16",
     "model.num_layers": 16,
     "model.dim": 256, "data.resolutions": [[8, 8, 8], [16, 16, 16]]},

    # ---- reference: actual training config --------------------------------
    {"group": "reference", "label": "actual_config",
     "data.image_size":   [128, 128, 128],
     "data.resolutions":  [[8, 8, 8], [16, 16, 16], [32, 32, 32]],
     "data.n_patches_l1": 512,
     "data.context_size": 1,
     "model.dim":         256,
     "model.num_layers":  8},
]


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_base_cfg(cluster: str) -> OmegaConf:
    base   = OmegaConf.load(ROOT / "configs" / "config.yaml")
    cl_cfg = OmegaConf.load(ROOT / "configs" / "cluster" / f"{cluster}.yaml")
    ex_cfg = OmegaConf.load(ROOT / "configs" / "experiment" / "multilevel.yaml")
    return OmegaConf.merge(base, cl_cfg, ex_cfg)


def apply_overrides(base_cfg: OmegaConf, overrides: dict) -> OmegaConf:
    """Apply dot-notation overrides to a config (deep copy)."""
    cfg = OmegaConf.structured(OmegaConf.to_container(base_cfg, resolve=True))
    for key, value in overrides.items():
        OmegaConf.update(cfg, key, value, merge=True)
    return cfg


def build_sweep_cfg(base_cfg: OmegaConf, entry: dict) -> OmegaConf:
    """Merge BASELINE + entry overrides onto base_cfg."""
    overrides = {**BASELINE, **{k: v for k, v in entry.items()
                                 if k not in ("group", "label")}}
    return apply_overrides(base_cfg, overrides)


# ---------------------------------------------------------------------------
# Dataset cache  (keyed by (image_size, context_size))
# ---------------------------------------------------------------------------

_ds_cache: dict[tuple, TotalSegInContextDataset] = {}


def get_dataset(cfg: OmegaConf) -> TotalSegInContextDataset:
    key = (tuple(cfg.data.image_size), cfg.data.context_size)
    if key not in _ds_cache:
        classes = resolve_classes(["liver"], cfg.paths.totalseg)
        _ds_cache[key] = TotalSegInContextDataset(
            root=cfg.paths.totalseg,
            classes=classes,
            image_size=tuple(cfg.data.image_size),
            split="train",
            context_size=cfg.data.context_size,
            class_balanced=True,
            use_crop=cfg.data.use_crop,
            synth_method=None,
            p_synth=0.0,
            random_coloring=False,
        )
    return _ds_cache[key]


def make_loader(ds: TotalSegInContextDataset, n_items: int) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=1,
        sampler=RandomSampler(ds, replacement=True, num_samples=n_items),
        num_workers=2,
        pin_memory=True,
        collate_fn=incontext_collate_fn,
        drop_last=False,
    )


# ---------------------------------------------------------------------------
# Encoder (built once, reused)
# ---------------------------------------------------------------------------

_encoder: STUNetEncoder | None = None


def get_encoder(cfg: OmegaConf, device: torch.device) -> tuple[STUNetEncoder, int]:
    global _encoder
    if _encoder is None:
        print("Building encoder …", flush=True)
        _encoder = STUNetEncoder(
            in_channels=1,
            variant=cfg.model.stunet_variant,
            pretrained=cfg.model.stunet_pretrained,
            freeze_encoder=True,
        ).to(device).eval()
    return _encoder


def get_embed_dim(encoder: STUNetEncoder, cfg: OmegaConf, device: torch.device) -> int:
    num_levels = len(encoder.skip_channels) + 1
    with torch.inference_mode():
        dummy = torch.zeros(1, 1, *cfg.data.image_size, device=device)
        feats = encode_image_only(encoder, dummy)
        ds    = extract_features(feats, cfg.model.feature_level,
                                 tuple(cfg.data.resolutions[0]), num_levels)
    return ds.shape[1]


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def build_model(embed_dim: int, cfg: OmegaConf) -> MultilevelICL:
    resolutions = [tuple(r) for r in cfg.data.resolutions]
    level_cfgs = [
        {
            "grid_size":       res,
            "dim":             cfg.model.dim,
            "num_heads":       cfg.model.num_heads,
            "num_layers":      cfg.model.num_layers,
            "ff_factor":       cfg.model.ff_factor,
            "label_injection": cfg.model.label_injection,
            "output_head":     cfg.model.output_head,
            "pos_encoding":    cfg.model.pos_encoding
                               if (i == 0 or cfg.model.pos_encoding == "rope3d")
                               else "none",
            "input_norm":      cfg.model.input_norm,
            "dropout":         0.0,   # no dropout for benchmarking
            "ctx_self_attn":   cfg.model.ctx_self_attn,
            "log_n_scaling":   cfg.model.log_n_scaling,
            "log_n_base":      cfg.model.log_n_base,
            "soft_labels":     getattr(cfg.model, "soft_labels_train", True),
        }
        for i, res in enumerate(resolutions)
    ]
    mask_cnn_dim     = int(getattr(cfg.model, "mask_cnn_dim", 0) or 0)
    num_registers    = int(getattr(cfg.model, "num_registers", 0) or 0)
    append_zero_attn = bool(getattr(cfg.model, "append_zero_attn", False))
    return MultilevelICL(
        embed_dim=embed_dim,
        level_cfgs=level_cfgs,
        mask_cnn_dim=mask_cnn_dim,
        num_registers=num_registers,
        append_zero_attn=append_zero_attn,
    )


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------

def _is_oom(exc: BaseException) -> bool:
    return isinstance(exc, torch.cuda.OutOfMemoryError) or (
        isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()
    )


def _cuda_cleanup(device: torch.device) -> None:
    """Synchronize and free the CUDA cache. Safe to call after OOM."""
    if device.type == "cuda":
        try:
            torch.cuda.synchronize(device)
        except Exception:
            pass
        torch.cuda.empty_cache()
    gc.collect()


def measure_config(
    entry:     dict,
    base_cfg:  OmegaConf,
    encoder:   STUNetEncoder,
    device:    torch.device,
    n_batches: int,
    n_warmup:  int = 1,
) -> dict:
    cfg = build_sweep_cfg(base_cfg, entry)
    resolutions = [tuple(r) for r in cfg.data.resolutions]

    result = {
        "group":          entry["group"],
        "label":          entry["label"],
        "image_size":     list(cfg.data.image_size),
        "resolutions":    [list(r) for r in resolutions],
        "n_levels":       len(resolutions),
        "n_patches_l1":   cfg.data.n_patches_l1,
        "context_size":   cfg.data.context_size,
        "dim":            cfg.model.dim,
        "num_layers":     cfg.model.num_layers,
        "mask_cnn_dim":   int(getattr(cfg.model, "mask_cnn_dim", 0) or 0),
        "num_registers":  int(getattr(cfg.model, "num_registers", 0) or 0),
        "n_params_model": None,
        "embed_dim":      None,
        "status":         "ok",
        "peak_vram_gb":   None,
        "times_s":        [],
        "mean_time_s":    None,
        "min_time_s":     None,
        "max_time_s":     None,
    }

    # All GPU objects initialised to None so the finally block can safely
    # delete them regardless of where an exception was thrown.
    model = optimizer = loader = it = batch = None

    try:
        # get_embed_dim runs a dummy encoder forward — may OOM for large image_size
        embed_dim = get_embed_dim(encoder, cfg, device)
        result["embed_dim"] = embed_dim

        model          = build_model(embed_dim, cfg).to(device).train()
        optimizer      = torch.optim.Adam(model.parameters(), lr=1e-4)
        result["n_params_model"] = sum(p.numel() for p in model.parameters()
                                       if p.requires_grad)

        ds     = get_dataset(cfg)
        loader = make_loader(ds, n_warmup + n_batches)
        it     = iter(loader)
        amp    = device.type == "cuda"

        def _step(b: dict) -> None:
            _, loss, *_ = process_batch(encoder, model, b, cfg, device, amp=amp)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if device.type == "cuda":
                torch.cuda.synchronize(device)

        # Warmup (not measured)
        for _ in range(n_warmup):
            batch = next(it)
            _step(batch)
            batch = None  # release GPU tensors immediately

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        times: list[float] = []
        for _ in range(n_batches):
            batch = next(it)
            t0 = time.perf_counter()
            _step(batch)
            times.append(time.perf_counter() - t0)
            batch = None  # release GPU tensors immediately

        if device.type == "cuda":
            result["peak_vram_gb"] = round(
                torch.cuda.max_memory_allocated(device) / 1e9, 3
            )
        result["times_s"]     = [round(t, 4) for t in times]
        result["mean_time_s"] = round(sum(times) / len(times), 4)
        result["min_time_s"]  = round(min(times), 4)
        result["max_time_s"]  = round(max(times), 4)

    except Exception as exc:
        result["status"] = "OOM" if _is_oom(exc) else f"ERROR: {exc}"

    finally:
        # Delete in reverse dependency order before clearing the CUDA cache.
        del batch, it, loader, optimizer, model
        _cuda_cleanup(device)

    return result


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

_HDR = (
    f"{'group':>14}  {'label':<22}  {'lvls':>4}  {'NP':>4}  {'K':>2}  "
    f"{'dim':>3}  {'L':>2}  {'params':>7}  {'VRAM':>8}  {'mean_t':>7}  {'min_t':>7}  status"
)
_SEP = "-" * len(_HDR)


def _fmt_result(r: dict) -> str:
    vram   = f"{r['peak_vram_gb']:.3f} GB"        if r["peak_vram_gb"]   is not None else "—"
    mean_t = f"{r['mean_time_s']:.3f} s"          if r["mean_time_s"]    is not None else "—"
    min_t  = f"{r['min_time_s']:.3f} s"           if r["min_time_s"]     is not None else "—"
    params = f"{r['n_params_model'] // 1000}k"    if r["n_params_model"] is not None else "—"
    status = r["status"] if r["status"] != "ok" else ""
    return (
        f"{r['group']:>14}  {r['label']:<22}  {r['n_levels']:>4}  {r['n_patches_l1']:>4}  "
        f"{r['context_size']:>2}  {r['dim']:>3}  {r['num_layers']:>2}  {params:>7}  "
        f"{vram:>8}  {mean_t:>7}  {min_t:>7}  {status}"
    )


def print_result(r: dict) -> None:
    print("  " + _fmt_result(r))


def print_summary(results: list[dict]) -> None:
    """Print a grouped summary table with relative scaling annotations."""
    from collections import defaultdict
    print("\n" + "═" * (len(_HDR) + 2))
    print("SWEEP SUMMARY")
    print("═" * (len(_HDR) + 2))
    print("  " + _HDR)

    groups: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        groups[r["group"]].append(r)

    for group, rows in groups.items():
        print(_SEP)
        ok_rows = [r for r in rows if r["status"] == "ok"
                   and r["peak_vram_gb"] is not None and r["mean_time_s"] is not None]
        for r in rows:
            print("  " + _fmt_result(r))

        if len(ok_rows) >= 2:
            base_vram = ok_rows[0]["peak_vram_gb"]
            base_time = ok_rows[0]["mean_time_s"]
            parts = []
            for r in ok_rows[1:]:
                vr = r["peak_vram_gb"] / base_vram if base_vram else float("nan")
                tr = r["mean_time_s"]  / base_time if base_time else float("nan")
                parts.append(f"{r['label']}: VRAM×{vr:.2f} time×{tr:.2f}")
            print(f"  {'↳ scaling vs ' + ok_rows[0]['label'] + ':':<36}  " + "  |  ".join(parts))

    print("═" * (len(_HDR) + 2))


CSV_FIELDS = [
    "group", "label", "image_size", "n_levels", "n_patches_l1", "context_size",
    "dim", "num_layers", "mask_cnn_dim", "num_registers", "n_params_model",
    "embed_dim", "status", "peak_vram_gb", "mean_time_s", "min_time_s", "max_time_s",
]


def save_results(results: list[dict], out_dir: Path, tag: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / f"{tag}.json"
    json_path.write_text(json.dumps(results, indent=2))

    csv_path = out_dir / f"{tag}.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            row = dict(r)
            row["image_size"] = "×".join(str(x) for x in r["image_size"])
            writer.writerow(row)

    print(f"\nSaved → {json_path}")
    print(f"       → {csv_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cluster",   default="nfs")
    parser.add_argument("--n_batches", type=int, default=5,
                        help="Measured batches per config (after warmup)")
    parser.add_argument("--n_warmup",  type=int, default=1,
                        help="Warmup batches (not measured)")
    parser.add_argument("--groups",    nargs="*", default=None,
                        help="Run only these sweep groups (default: all)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    print(f"Device: {device}")
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        print(f"GPU: {props.name}  total VRAM: {props.total_memory / 1e9:.1f} GB\n")

    base_cfg = load_base_cfg(args.cluster)
    encoder  = get_encoder(base_cfg, device)

    entries = SWEEP
    if args.groups:
        entries = [e for e in SWEEP if e["group"] in args.groups]

    out_dir = Path(base_cfg.paths.results) / "benchmarks" / "vram_sweep"
    tag     = datetime.now().strftime("%Y%m%d_%H%M%S")

    results: list[dict] = []
    prev_group = None

    for i, entry in enumerate(entries):
        if entry["group"] != prev_group:
            print(f"\n── {entry['group']} ──")
            prev_group = entry["group"]

        print(f"  [{i+1}/{len(entries)}] {entry['label']} …", end="  ", flush=True)
        t_start = time.perf_counter()
        r = measure_config(entry, base_cfg, encoder, device, args.n_batches, args.n_warmup)
        elapsed = time.perf_counter() - t_start
        print(f"done ({elapsed:.1f}s)")
        print_result(r)

        results.append(r)
        save_results(results, out_dir, tag)   # incremental save

    print_summary(results)
    save_results(results, out_dir, tag)
    print("\nDone.")


if __name__ == "__main__":
    main()
