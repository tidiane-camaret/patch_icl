"""
VRAM and feature-storage benchmark for NNInteractiveEncoder.

Sweeps over mask_injection mode (ch1 / separate) and num_stages (encoder depth),
mirroring the per-stage breakdown in benchmark_encoder.py.

Metrics reported per config
---------------------------
  stage shapes   : spatial shape of each returned feature tensor
  feat_store_mb  : total size of the returned feature list (kept in GPU RAM
                   while the downstream attention module runs)
  peak_vram_mb   : peak GPU memory during the forward pass (transient activations
                   + stored features)
  time_ms        : mean ± std wall-clock time per image
  trainable_M    : number of trainable parameters (0 for ch1, mask encoder for separate)

Mask injection modes
--------------------
  ch1       — image in ch0, mask in ch1 of the 8-channel input.  No extra params.
  separate  — image in ch0 only; mask encoded by SAM-style 3-D CNN fused at bottleneck.
              Trainable params = mask encoder only (~0.5–4 M depending on num_stages).

num_stages and spatial stride
------------------------------
  num_stages=4  →  8×  stride, bottleneck [B, 256, D/8,  H/8,  W/8 ]
  num_stages=5  →  16× stride, bottleneck [B, 320, D/16, H/16, W/16]
  num_stages=6  →  32× stride, bottleneck [B, 320, D/32, H/32, W/32]

Usage
-----
    python experiments/encoders/nninteractive_benchmark.py \\
        --ckpt_dir /home/dpxuser/model_checkpoints/nnint/nnInteractive_v1.0

    python experiments/encoders/nninteractive_benchmark.py \\
        --image_sizes 64 128 192 \\
        --num_stages 4 5 6 \\
        --mask_injections ch1 separate \\
        --n_runs 10 --n_warmup 3
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.models.encoders.nninteractive import NNInteractiveEncoder  # noqa


# ---------------------------------------------------------------------------
# Core measurement
# ---------------------------------------------------------------------------

def measure_encoding(
    encoder:    NNInteractiveEncoder,
    image_size: int,
    batch_size: int,
    amp:        bool = True,
    n_warmup:   int  = 3,
    n_runs:     int  = 10,
    device:     torch.device = torch.device("cuda"),
) -> dict:
    """Forward pass through (imgs, masks) → list[Tensor].

    Returns dict with stage_shapes, stage_storage_mb, feat_store_mb,
    peak_vram_mb, time_ms_mean, time_ms_std, time_ms_per_img, status.
    """
    result: dict = {
        "stage_shapes":     None,
        "stage_storage_mb": None,
        "feat_store_mb":    None,
        "peak_vram_mb":     None,
        "time_ms_mean":     None,
        "time_ms_std":      None,
        "time_ms_per_img":  None,
        "status":           "ok",
    }

    imgs = masks = feats = None
    try:
        dtype = torch.float16 if amp else torch.float32
        sz    = image_size
        imgs  = torch.randn(batch_size, 1, sz, sz, sz, device=device, dtype=dtype)
        masks = torch.zeros(batch_size, 1, sz, sz, sz, device=device, dtype=dtype)

        def _fwd() -> list[torch.Tensor]:
            with torch.no_grad(), \
                 torch.autocast(device_type=device.type, enabled=amp):
                return encoder(imgs, masks)

        for _ in range(n_warmup):
            _fwd()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)

        feats = _fwd()

        if device.type == "cuda":
            torch.cuda.synchronize(device)
            result["peak_vram_mb"] = torch.cuda.max_memory_allocated(device) / 1e6

        result["stage_shapes"]     = [tuple(f.shape) for f in feats]
        result["stage_storage_mb"] = [f.nbytes / 1e6 for f in feats]
        result["feat_store_mb"]    = sum(result["stage_storage_mb"])

        times: list[float] = []
        for _ in range(n_runs):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            _fwd()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            times.append((time.perf_counter() - t0) * 1e3)

        mean_t = sum(times) / len(times)
        std_t  = (sum((t - mean_t) ** 2 for t in times) / len(times)) ** 0.5
        result["time_ms_mean"]    = mean_t
        result["time_ms_std"]     = std_t
        result["time_ms_per_img"] = mean_t / batch_size

    except torch.cuda.OutOfMemoryError:
        result["status"] = "OOM"
    except Exception as exc:
        result["status"] = f"ERROR: {exc}"
    finally:
        del imgs, masks, feats
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    return result


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _mb(x: float | None) -> str:
    if x is None:
        return "—"
    if x >= 1000:
        return f"{x / 1000:.2f} GB"
    if x >= 1:
        return f"{x:.1f} MB"
    return f"{x * 1000:.0f} KB"


def _ms(x: float | None) -> str:
    return "—" if x is None else f"{x:.1f} ms"


def print_stage_table(
    result:         dict,
    mask_injection: str,
    num_stages:     int,
    image_size:     int,
    batch_size:     int,
    amp:            bool,
    trainable_M:    float,
) -> None:
    prec  = "fp16" if amp else "fp32"
    title = (f"NNInteractive  {image_size}³  B={batch_size}  {prec}"
             f"  mask={mask_injection}  stages={num_stages}")
    print(f"\n{'─' * 76}")
    print(f"  {title}")
    print(f"{'─' * 76}")

    if result["status"] != "ok":
        print(f"  ✗  {result['status']}")
        return

    shapes = result["stage_shapes"]
    mbytes = result["stage_storage_mb"]
    total  = result["feat_store_mb"]
    n      = len(shapes)

    print(f"  {'stage':>5}  {'shape':>36}  {'storage':>10}  {'% total':>8}")
    print(f"  {'─'*5}  {'─'*36}  {'─'*10}  {'─'*8}")
    for i, (sh, mb) in enumerate(zip(shapes, mbytes)):
        label     = f"s{i}" if i < n - 1 else f"s{i} (bottleneck)"
        shape_str = "×".join(str(x) for x in sh)
        pct       = 100 * mb / total if total > 0 else 0
        print(f"  {label:>5}  {shape_str:>36}  {_mb(mb):>10}  {pct:>7.1f}%")
    print(f"  {'':>5}  {'total':>36}  {_mb(total):>10}  {'100.0%':>8}")

    print()
    bot_mb = mbytes[-1]
    print(f"  feature_level='all'   → {_mb(total):>10}  (all stages kept)")
    print(f"  feature_level='-1'    → {_mb(bot_mb):>10}  (bottleneck only,"
          f" ×{total / bot_mb:.0f}× smaller)")

    print()
    print(f"  Peak VRAM             → {_mb(result['peak_vram_mb'])}")
    overhead = (result["peak_vram_mb"] - total) if result["peak_vram_mb"] else None
    if overhead is not None:
        print(f"  Transient VRAM        → {_mb(overhead)}  (peak − feat_store)")
    print(f"  Encoding time         → {_ms(result['time_ms_mean'])} ± {_ms(result['time_ms_std'])}"
          f"  ({_ms(result['time_ms_per_img'])}/image)")
    print(f"  Trainable params      → {trainable_M:.2f} M"
          f"  ({'mask encoder only' if mask_injection == 'separate' else 'none — encoder frozen'})")


def print_sweep_summary(rows: list[dict]) -> None:
    if not rows:
        return

    hdr = (f"  {'mask':>8}  {'stages':>6}  {'img':>5}  {'B':>2}  {'prec':>4}  "
           f"{'feat(all)':>10}  {'feat(bot)':>10}  "
           f"{'peak_vram':>10}  {'t/img':>8}  {'trainable':>10}")
    w = len(hdr) - 2
    print(f"\n{'═' * w}")
    print("SWEEP SUMMARY")
    print(f"{'═' * w}")
    print(hdr)
    print(f"  {'─'*8}  {'─'*6}  {'─'*5}  {'─'*2}  {'─'*4}  "
          f"{'─'*10}  {'─'*10}  {'─'*10}  {'─'*8}  {'─'*10}")

    for r in rows:
        if r["status"] != "ok":
            print(f"  {r['mask_injection']:>8}  {r['num_stages']:>6}  "
                  f"{r['img_size']:>5}  {r['batch_size']:>2}  "
                  f"{r['prec']:>4}  ✗ {r['status'][:40]}")
            continue
        res      = r["result"]
        feat_all = res["feat_store_mb"]
        feat_bot = res["stage_storage_mb"][-1]
        peak     = res["peak_vram_mb"]
        t_img    = res["time_ms_per_img"]
        trainM   = r["trainable_M"]
        print(
            f"  {r['mask_injection']:>8}  {r['num_stages']:>6}  "
            f"{r['img_size']:>5}  {r['batch_size']:>2}  {r['prec']:>4}  "
            f"{_mb(feat_all):>10}  {_mb(feat_bot):>10}  "
            f"{_mb(peak):>10}  {_ms(t_img):>8}  "
            f"{trainM:>8.2f} M"
        )
    print(f"{'═' * w}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--ckpt_dir", default="/home/dpxuser/model_checkpoints/nnint/nnInteractive_v1.0",
        help="Path to the nnInteractive_v1.0 checkpoint folder",
    )
    parser.add_argument("--image_sizes",     nargs="+", type=int,  default=[64, 128])
    parser.add_argument("--batch_sizes",     nargs="+", type=int,  default=[1])
    parser.add_argument("--num_stages",      nargs="+", type=int,  default=[4, 5, 6],
                        help="Encoder depth (4 → 8× stride, 5 → 16×, 6 → 32×)")
    parser.add_argument("--mask_injections", nargs="+",            default=["ch1", "separate"],
                        choices=["ch1", "separate"])
    parser.add_argument("--no_amp",          action="store_true",
                        help="Run in fp32 instead of fp16 autocast")
    parser.add_argument("--n_runs",          type=int,  default=10)
    parser.add_argument("--n_warmup",        type=int,  default=3)
    parser.add_argument("--device",          default=None)
    args = parser.parse_args()

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device     = torch.device(device_str)
    amp        = not args.no_amp

    print(f"Device  : {device}")
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        print(f"GPU     : {props.name}  ({props.total_memory / 1e9:.1f} GB)")
    print(f"AMP     : {'fp16 autocast' if amp else 'fp32'}")
    print(f"Runs    : {args.n_warmup} warmup + {args.n_runs} measured")
    print(f"Ckpt    : {args.ckpt_dir}")

    all_rows: list[dict] = []

    for mask_injection in args.mask_injections:
        for num_stages in args.num_stages:
            tag = f"mask={mask_injection}  stages={num_stages}"
            print(f"\nBuilding NNInteractive [{tag}] …", end="  ", flush=True)
            try:
                enc = NNInteractiveEncoder(
                    ckpt_dir=args.ckpt_dir,
                    mask_injection=mask_injection,
                    freeze_encoder=True,
                    num_stages=num_stages,
                    device="cpu",
                ).to(device).eval()
            except Exception as exc:
                print(f"FAILED: {exc}")
                continue

            total_M     = sum(p.numel() for p in enc.parameters()) / 1e6
            trainable_M = sum(p.numel() for p in enc.parameters()
                              if p.requires_grad) / 1e6
            print(f"{total_M:.1f} M total  /  {trainable_M:.2f} M trainable")

            for img_size in args.image_sizes:
                for batch_size in args.batch_sizes:
                    result = measure_encoding(
                        encoder=enc,
                        image_size=img_size,
                        batch_size=batch_size,
                        amp=amp,
                        n_warmup=args.n_warmup,
                        n_runs=args.n_runs,
                        device=device,
                    )
                    print_stage_table(
                        result, mask_injection, num_stages,
                        img_size, batch_size, amp, trainable_M,
                    )
                    all_rows.append({
                        "mask_injection": mask_injection,
                        "num_stages":     num_stages,
                        "img_size":       f"{img_size}³",
                        "batch_size":     batch_size,
                        "prec":           "fp16" if amp else "fp32",
                        "trainable_M":    trainable_M,
                        "status":         result["status"],
                        "result":         result,
                    })

            del enc
            if device.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

    print_sweep_summary(all_rows)


if __name__ == "__main__":
    main()
