"""Sweep encoders x input sizes -> CSV + scaling-curve PNGs (+ optional wandb).

Each (encoder, size) can be measured in a fresh subprocess (--isolate, the CLI default
on CUDA) so every point gets the full GPU and a clean dynamo/inductor state. This avoids
cross-encoder memory accumulation (reduce-overhead CUDA-graph pools aren't freed by
empty_cache -> spurious late-encoder OOMs) and per-batch recompile contamination.
"""
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "3d"))

from encoder_bench import registry as R                     # noqa: E402
from encoder_bench.profiling import profile_point           # noqa: E402
from encoder_bench.optimize import apply_optimization       # noqa: E402

_FIELDS = ["encoder", "family", "input_size", "params", "gflops",
           "fwd_bwd_ms", "train_vram_mb", "throughput_vol_s", "status"]
_PLOT_METRICS = ("fwd_bwd_ms", "train_vram_mb", "gflops")
_FAMILY_MARKER = {"cnn": "o", "transformer": "s", "mamba": "^"}
_ROW_TAG = "__ROW__"   # sentinel prefixing the JSON row on a subprocess's stdout


def _empty_row(name, size, status):
    r = {k: None for k in _FIELDS}
    fam = R.REGISTRY[name].family if name in R.REGISTRY else None
    r.update(encoder=name, family=fam, input_size=size, status=status)
    return r


def run_point(name, size, device, n_warmup=3, n_timed=10) -> dict:
    """Measure a single (encoder, size) in-process. Returns one result row."""
    spec = R.REGISTRY[name]
    if size % spec.size_multiple != 0:
        return profile_point(spec, size, device)
    try:
        mod, ctx = apply_optimization(spec.factory(), spec.opt_profile, device)
    except Exception as e:                          # ckpt/dep missing -> unavailable row
        print(f"  WARN {name}@{size} unavailable: {e}", file=sys.stderr)
        return _empty_row(name, size, f"unavailable:{type(e).__name__}")
    with ctx:
        return profile_point(spec, size, device, module=mod,
                             n_warmup=n_warmup, n_timed=n_timed)


def _run_point_isolated(name, size, device, n_warmup, n_timed) -> dict:
    """Measure one point in a fresh subprocess; parse its JSON row from stdout.

    A hard crash (segfault / uncatchable CUDA OOM) yields no row -> synthesize one so the
    sweep continues. Most OOMs are caught inside profile_point and returned as a real row.
    """
    cmd = [sys.executable, __file__, "--_one", name, str(size),
           "--device", device.type,
           "--n_warmup", str(n_warmup), "--n_timed", str(n_timed)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    for line in proc.stdout.splitlines():
        if line.startswith(_ROW_TAG):
            return json.loads(line[len(_ROW_TAG):])
    # No row emitted -> the child died. Classify OOM vs generic crash from its stderr.
    status = "oom" if "OutOfMemory" in proc.stderr or "out of memory" in proc.stderr \
             else f"error:subprocess{proc.returncode}"
    print(f"  WARN {name}@{size} subprocess produced no row (rc={proc.returncode}, "
          f"status={status})", file=sys.stderr)
    return _empty_row(name, size, status)


def sweep(encoders, input_sizes, device, out_dir, n_warmup=3, n_timed=10, isolate=False):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in encoders:
        for size in input_sizes:
            if isolate:
                row = _run_point_isolated(name, size, device, n_warmup, n_timed)
            else:
                row = run_point(name, size, device, n_warmup, n_timed)
            rows.append(row)
            print(f"  {name}@{size}: {row['status']}", flush=True)
    with open(out_dir / "encoder_bench.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=_FIELDS); w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in _FIELDS})
    return rows


def plot_curves(rows, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out_dir = Path(out_dir); paths = []
    encoders = sorted({r["encoder"] for r in rows})
    
    for metric in _PLOT_METRICS:
        fig, ax = plt.subplots(figsize=(6, 4))
        drew = False
        for enc in encoders:
            pts = sorted([r for r in rows if r["encoder"] == enc
                          and r.get(metric) is not None], key=lambda r: r["input_size"])
            if not pts:
                continue
            fam = pts[0]["family"]
            
            # Use the family dictionary for markers, and let Matplotlib handle colors
            ax.plot([p["input_size"] for p in pts], [p[metric] for p in pts],
                    marker=_FAMILY_MARKER.get(fam, "o"), label=enc)
            drew = True
            
        if not drew:
            plt.close(fig); continue
        
        ax.set_yscale("log"); ax.set_xlabel("input size (D=H=W)"); ax.set_ylabel(metric)
        ax.set_title(f"{metric} vs input size"); ax.legend(fontsize=7)
        p = out_dir / f"scaling_{metric}.png"; fig.savefig(p, dpi=120, bbox_inches="tight")
        plt.close(fig); paths.append(p)
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoders", nargs="*", default=None)
    ap.add_argument("--input_sizes", nargs="*", type=int, default=[32, 64, 96, 128])
    ap.add_argument("--out_dir", default="results/encoder_bench")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--wandb_project", default=None)
    ap.add_argument("--n_warmup", type=int, default=3)
    ap.add_argument("--n_timed", type=int, default=10)
    ap.add_argument("--no_isolate", action="store_true",
                    help="run all points in one process (default: subprocess-per-point on CUDA)")
    ap.add_argument("--_one", nargs=2, metavar=("ENCODER", "SIZE"), default=None,
                    help=argparse.SUPPRESS)   # internal: measure one point, emit JSON row
    args = ap.parse_args()
    device = torch.device(args.device)

    # Internal single-point mode: profile one (encoder, size) and print its row as JSON.
    if args._one is not None:
        name, size = args._one[0], int(args._one[1])
        row = run_point(name, size, device, args.n_warmup, args.n_timed)
        print(_ROW_TAG + json.dumps(row), flush=True)
        return

    encoders = args.encoders or R.list_encoders()
    unknown = [e for e in encoders if e not in R.REGISTRY]
    if unknown:
        raise SystemExit(f"unknown encoder(s) {unknown}; available: {R.list_encoders()}")
    isolate = device.type == "cuda" and not args.no_isolate
    rows = sweep(encoders, args.input_sizes, device, Path(args.out_dir),
                 n_warmup=args.n_warmup, n_timed=args.n_timed, isolate=isolate)
    pngs = plot_curves(rows, Path(args.out_dir))
    print(f"Done. {len(rows)} rows, {len(pngs)} plots -> {args.out_dir}")
    if args.wandb_project:
        import wandb
        wandb.init(project=args.wandb_project,
                   config={"encoders": encoders, "input_sizes": args.input_sizes})
        wandb.log({"encoder_bench/table": wandb.Table(
            columns=_FIELDS, data=[[r.get(k) for k in _FIELDS] for r in rows])})
        for p in pngs:
            wandb.log({f"encoder_bench/{p.stem}": wandb.Image(str(p))})
        wandb.finish()


if __name__ == "__main__":
    main()
