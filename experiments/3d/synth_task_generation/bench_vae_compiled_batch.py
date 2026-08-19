"""
Follow-up: throughput of the COMPILED VAE decode (the 10x lever) at 96^3, batched.
Answers 'how many diverse crops/s can a background decode worker produce?'.
"""
import argparse, sys, time
from pathlib import Path
import numpy as np, torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gen_maisi_fast import build_args


def _time(fn, n, warmup):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(); ts = []
    for _ in range(n):
        torch.cuda.synchronize(); t = time.time(); fn(); torch.cuda.synchronize()
        ts.append((time.time() - t) * 1e3)
    return float(np.mean(ts)), float(np.std(ts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=Path("/home/dpxuser/repos/NV-Generate-CTMR"))
    ap.add_argument("--env_file", default="environment_rflow-ct.json")
    ap.add_argument("--infer_file", default="config_infer_wholebody.json")
    a = ap.parse_args()
    sys.path.insert(0, str(a.repo))
    from scripts.utils_infer import ReconModel, load_image_models

    dev = torch.device("cuda")
    args = build_args(a.repo, a.env_file, a.infer_file)
    args.autoencoder_def["num_splits"] = 1
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    ae, dm, cn, sf, _ = load_image_models(args, dev); del dm, cn; torch.cuda.empty_cache()
    recon = ReconModel(autoencoder=ae.half(), scale_factor=sf).to(dev).eval()
    crecon = torch.compile(recon, mode="default")

    def dec(z):
        with torch.no_grad():
            return crecon(z)

    for crop in (96, 128):
        lc = crop // 4
        print(f"\n=== compiled decode {crop}^3 (latent 4x{lc}^3), batched ===", flush=True)
        print(f"{'B':>4} {'ms/batch':>10} {'ms/item':>9} {'crops/s':>8} {'peakGB':>7}", flush=True)
        for B in (1, 2, 4, 8, 16):
            z = torch.randn(B, args.latent_channels, lc, lc, lc, device=dev).half()
            torch.cuda.reset_peak_memory_stats()
            try:
                m, _ = _time(lambda: dec(z), n=8, warmup=6)  # per-batch-shape recompile on first call
                peak = torch.cuda.max_memory_allocated() / 1024**3
                print(f"{B:>4} {m:>10.1f} {m/B:>9.1f} {1000*B/m:>8.1f} {peak:>7.1f}", flush=True)
            except torch.cuda.OutOfMemoryError:
                print(f"{B:>4}  OOM", flush=True); torch.cuda.empty_cache(); break


if __name__ == "__main__":
    main()
