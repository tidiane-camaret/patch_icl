"""
Encoder cost benchmark for STUNetEncoder.

Measures two distinct GPU memory costs per (variant, image_size, batch_size):

  peak_vram_gb   : peak GPU memory during encode_image_only, including transient
                   conv activations that exist only during the forward pass.
  feat_store_gb  : total size of the returned feature tensor list — what stays
                   in GPU memory while the downstream attention module runs.

Also reports a per-stage breakdown (shape + MB) so it is clear which encoder
stage dominates storage, and the cost difference between keeping all stages
("all" mode used in multilevel training) vs. the bottleneck only.

Usage
-----
    python experiments/encoders/benchmark_encoder.py
    python experiments/encoders/benchmark_encoder.py --variants small base large
    python experiments/encoders/benchmark_encoder.py --image_sizes 64 128 256
    python experiments/encoders/benchmark_encoder.py --batch_sizes 1 4 8
    python experiments/encoders/benchmark_encoder.py --no_amp          # fp32 forward
    python experiments/encoders/benchmark_encoder.py --n_runs 20
    python experiments/encoders/benchmark_encoder.py --pretrained /path/to/base_statedict.pt
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

from src.models.encoders.stunet import STUNetEncoder, _VARIANTS  # noqa: direct module import (avoids resenc dep)


# ---------------------------------------------------------------------------
# Core encoding function (mirrors experiments/multilevel/train.py)
# ---------------------------------------------------------------------------

@torch.no_grad()
def encode_image_only(
    encoder: STUNetEncoder,
    imgs: torch.Tensor,
) -> list[torch.Tensor]:
    """imgs: (B, 1, D, H, W) → list of feature tensors [skip0, …, bottleneck]."""
    bottleneck, skips = encoder.image_encoder(imgs, num_stages=encoder._num_stages)
    return skips + [bottleneck]


# ---------------------------------------------------------------------------
# Analytical storage estimate (no GPU needed)
# ---------------------------------------------------------------------------

def theoretical_stage_shapes(
    variant: str,
    image_size: tuple[int, int, int],
    batch_size: int,
) -> list[tuple[int, ...]]:
    """Return the expected output shape for each encoder stage."""
    dims = _VARIANTS[variant]["dims"]   # 6 channel widths
    D, H, W = image_size
    shapes = []
    for stage_idx in range(6):
        stride = 2 ** stage_idx         # stage 0: no downsampling
        d = max(1, D // stride)
        h = max(1, H // stride)
        w = max(1, W // stride)
        shapes.append((batch_size, dims[stage_idx], d, h, w))
    return shapes


def theoretical_storage_mb(
    shapes: list[tuple[int, ...]],
    bytes_per_elem: int = 2,            # fp16
) -> list[float]:
    """Return storage in MB for each stage tensor."""
    mbs = []
    for shape in shapes:
        n = 1
        for s in shape:
            n *= s
        mbs.append(n * bytes_per_elem / 1e6)
    return mbs


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------

def measure_encoding(
    encoder:    STUNetEncoder,
    image_size: tuple[int, int, int],
    batch_size: int,
    amp:        bool = True,
    n_warmup:   int  = 3,
    n_runs:     int  = 10,
    device:     torch.device = torch.device("cuda"),
) -> dict:
    """Run encode_image_only and measure VRAM + time.

    Returns
    -------
    dict with keys:
        stage_shapes      : list of tensor shapes, one per encoder stage
        stage_storage_mb  : list of MB, one per stage
        feat_store_mb     : total MB of all returned feature tensors
        peak_vram_mb      : peak GPU MB during the measured forward pass
                            (includes transient conv activations)
        time_ms_mean      : mean wall time over n_runs (ms)
        time_ms_std       : std  wall time over n_runs (ms)
        time_ms_per_img   : mean time per image in the batch (ms)
        status            : "ok" | "OOM" | "ERROR: …"
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

    imgs: torch.Tensor | None = None
    feats: list | None        = None
    try:
        imgs = torch.randn(batch_size, 1, *image_size, device=device,
                           dtype=torch.float16 if amp else torch.float32)

        def _forward() -> list[torch.Tensor]:
            with torch.autocast(device_type=device.type, enabled=amp):
                return encode_image_only(encoder, imgs)

        # Warmup
        for _ in range(n_warmup):
            _forward()
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        # Reset peak stats, then measure one forward for VRAM
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        feats = _forward()

        if device.type == "cuda":
            torch.cuda.synchronize(device)
            result["peak_vram_mb"] = torch.cuda.max_memory_allocated(device) / 1e6

        # Feature storage: sum of actual tensor sizes
        stage_shapes     = [tuple(f.shape) for f in feats]
        stage_storage_mb = [f.nbytes / 1e6 for f in feats]
        result["stage_shapes"]     = stage_shapes
        result["stage_storage_mb"] = stage_storage_mb
        result["feat_store_mb"]    = sum(stage_storage_mb)

        # Wall-time sweep
        times: list[float] = []
        for _ in range(n_runs):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            _forward()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            times.append((time.perf_counter() - t0) * 1e3)  # ms

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
        del imgs, feats
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    return result


# ---------------------------------------------------------------------------
# Reporting helpers
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
    result:     dict,
    variant:    str,
    image_size: tuple,
    batch_size: int,
    amp:        bool,
) -> None:
    """Print a per-stage breakdown for one (variant, image_size, batch_size) config."""
    D, H, W = image_size
    title = (f"STUNet-{variant}  |  input {D}³  |  B={batch_size}"
             f"  |  {'fp16' if amp else 'fp32'}")
    print(f"\n{'─' * 72}")
    print(f"  {title}")
    print(f"{'─' * 72}")

    if result["status"] != "ok":
        print(f"  ✗  {result['status']}")
        return

    shapes = result["stage_shapes"]
    mbytes = result["stage_storage_mb"]
    total  = result["feat_store_mb"]

    # header
    print(f"  {'stage':>5}  {'shape':>30}  {'storage':>10}  {'% total':>8}")
    print(f"  {'─'*5}  {'─'*30}  {'─'*10}  {'─'*8}")
    for i, (sh, mb) in enumerate(zip(shapes, mbytes)):
        label = f"s{i}" if i < len(shapes) - 1 else f"s{i} (bottleneck)"
        shape_str = "×".join(str(x) for x in sh)
        pct = 100 * mb / total if total > 0 else 0
        print(f"  {label:>5}  {shape_str:>30}  {_mb(mb):>10}  {pct:>7.1f}%")
    print(f"  {'':>5}  {'total':>30}  {_mb(total):>10}  {'100.0%':>8}")

    # feature_level breakdown: "all" vs "bottleneck only"
    print()
    bottleneck_mb = mbytes[-1]
    print(f"  feature_level='all'        → {_mb(total):>10}  (all stages kept)")
    print(f"  feature_level='-1'         → {_mb(bottleneck_mb):>10}  (bottleneck only, ×{total/bottleneck_mb:.0f}× smaller)")

    # VRAM + time
    print()
    print(f"  Peak VRAM during forward   → {_mb(result['peak_vram_mb'])}")
    overhead = (result['peak_vram_mb'] - total) if result['peak_vram_mb'] else None
    if overhead is not None:
        print(f"  Transient activation VRAM  → {_mb(overhead)}  (peak − feat_store)")
    print(f"  Encoding time              → {_ms(result['time_ms_mean'])} ± {_ms(result['time_ms_std'])}"
          f"  ({_ms(result['time_ms_per_img'])}/image)")


def print_sweep_summary(rows: list[dict]) -> None:
    """Print a compact multi-row summary table across all configs."""
    ok = [r for r in rows if r["status"] == "ok"]
    if not ok:
        return

    hdr = (f"  {'variant':>7}  {'img':>5}  {'B':>2}  {'prec':>4}  "
           f"{'feat(all)':>10}  {'feat(bot)':>10}  "
           f"{'peak_vram':>10}  {'t/img':>8}")
    print(f"\n{'═' * (len(hdr) - 2)}")
    print("SWEEP SUMMARY")
    print(f"{'═' * (len(hdr) - 2)}")
    print(hdr)
    print(f"  {'─'*7}  {'─'*5}  {'─'*2}  {'─'*4}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*8}")

    for r in rows:
        if r["status"] != "ok":
            status = r["status"][:20]
            print(f"  {r['variant']:>7}  {r['img_size']:>5}  {r['batch_size']:>2}  "
                  f"{r['prec']:>4}  {'— ' + status}")
            continue
        feat_all = r["result"]["feat_store_mb"]
        feat_bot = r["result"]["stage_storage_mb"][-1]
        peak     = r["result"]["peak_vram_mb"]
        t_img    = r["result"]["time_ms_per_img"]
        prec     = "fp16" if r["amp"] else "fp32"
        print(f"  {r['variant']:>7}  {r['img_size']:>5}  {r['batch_size']:>2}  {prec:>4}  "
              f"{_mb(feat_all):>10}  {_mb(feat_bot):>10}  "
              f"{_mb(peak):>10}  {_ms(t_img):>8}")

    print(f"{'═' * (len(hdr) - 2)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variants",    nargs="+", default=["base"],
                        choices=list(_VARIANTS),
                        help="STU-Net variants to benchmark (default: base)")
    parser.add_argument("--image_sizes", nargs="+", type=int, default=[64, 128],
                        help="Isotropic input edge lengths in voxels (default: 64 128)")
    parser.add_argument("--batch_sizes", nargs="+", type=int, default=[1],
                        help="Batch sizes to benchmark (default: 1)")
    parser.add_argument("--no_amp",     action="store_true",
                        help="Run in fp32 instead of fp16 autocast")
    parser.add_argument("--n_runs",     type=int, default=10,
                        help="Timing repetitions after warmup (default: 10)")
    parser.add_argument("--n_warmup",   type=int, default=3,
                        help="Warmup iterations before timing (default: 3)")
    parser.add_argument("--pretrained", default=None,
                        help="Path to STU-Net pretrained weights (.pt). "
                             "Skipped if omitted; weights are random.")
    parser.add_argument("--device",     default=None,
                        help="Device string, e.g. 'cuda:0' (default: cuda if available)")
    args = parser.parse_args()

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device     = torch.device(device_str)
    amp        = not args.no_amp

    print(f"Device : {device}")
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        print(f"GPU    : {props.name}  ({props.total_memory / 1e9:.1f} GB total VRAM)")
    print(f"Amp    : {'fp16 autocast' if amp else 'fp32 (no amp)'}")
    print(f"Runs   : {args.n_warmup} warmup + {args.n_runs} measured\n")

    # One encoder per variant (reused across image_sizes and batch_sizes)
    encoders: dict[str, STUNetEncoder] = {}
    for variant in args.variants:
        print(f"Building STUNet-{variant} …", end="  ", flush=True)
        enc = STUNetEncoder(
            in_channels=1,
            variant=variant,
            pretrained=args.pretrained,
            freeze_encoder=True,
        ).to(device).eval()
        n_params = sum(p.numel() for p in enc.parameters())
        print(f"{n_params / 1e6:.1f} M params")
        encoders[variant] = enc

    all_rows: list[dict] = []

    for variant in args.variants:
        encoder = encoders[variant]
        for img_size in args.image_sizes:
            for batch_size in args.batch_sizes:
                size3d = (img_size, img_size, img_size)

                # Analytical estimate first (no GPU)
                th_shapes = theoretical_stage_shapes(variant, size3d, batch_size)
                th_mb     = theoretical_storage_mb(th_shapes, bytes_per_elem=2 if amp else 4)

                tag = f"STUNet-{variant}  {img_size}³  B={batch_size}"
                print(f"\n[{tag}]  theoretical total feat: {_mb(sum(th_mb))}", flush=True)

                result = measure_encoding(
                    encoder=encoder,
                    image_size=size3d,
                    batch_size=batch_size,
                    amp=amp,
                    n_warmup=args.n_warmup,
                    n_runs=args.n_runs,
                    device=device,
                )

                print_stage_table(result, variant, size3d, batch_size, amp)

                all_rows.append({
                    "variant":    variant,
                    "img_size":   f"{img_size}³",
                    "batch_size": batch_size,
                    "amp":        amp,
                    "prec":       "fp16" if amp else "fp32",
                    "status":     result["status"],
                    "result":     result,
                })

    print_sweep_summary(all_rows)


if __name__ == "__main__":
    main()
