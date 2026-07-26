"""Sweep encoders x input sizes -> CSV + scaling-curve PNGs (+ optional wandb)."""
import argparse
import csv
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "3d"))

from encoder_bench import registry as R                     # noqa: E402
from encoder_bench.profile import profile_point             # noqa: E402
from encoder_bench.optimize import apply_optimization       # noqa: E402

_FIELDS = ["encoder", "family", "input_size", "params", "gflops",
           "fwd_bwd_ms", "train_vram_mb", "throughput_vol_s", "status"]
_PLOT_METRICS = ("fwd_bwd_ms", "train_vram_mb", "gflops")
_FAMILY_COLOR = {"cnn": "tab:blue", "transformer": "tab:red", "mamba": "tab:green"}


def sweep(encoders, input_sizes, device, out_dir, n_warmup=3, n_timed=10):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in encoders:
        spec = R.REGISTRY[name]
        for size in input_sizes:
            if size % spec.size_multiple != 0:
                rows.append(profile_point(spec, size, device)); continue
            try:
                mod, ctx = apply_optimization(spec.factory(), spec.opt_profile, device)
            except Exception as e:                      # ckpt/dep missing -> log + skip
                print(f"  WARN {name}@{size} unavailable: {e}", file=sys.stderr)
                r = {k: None for k in _FIELDS}
                r.update(encoder=name, family=spec.family, input_size=size,
                         status=f"unavailable:{type(e).__name__}")
                rows.append(r); continue
            with ctx:
                rows.append(profile_point(spec, size, device, module=mod,
                                          n_warmup=n_warmup, n_timed=n_timed))
            print(f"  {name}@{size}: {rows[-1]['status']}")
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
            ax.plot([p["input_size"] for p in pts], [p[metric] for p in pts],
                    marker="o", label=enc, color=_FAMILY_COLOR.get(fam)); drew = True
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
    args = ap.parse_args()
    encoders = args.encoders or R.list_encoders()
    device = torch.device(args.device)
    rows = sweep(encoders, args.input_sizes, device, Path(args.out_dir))
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
