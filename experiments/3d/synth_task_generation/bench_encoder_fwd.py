"""Forward-time comparison: MAISI VAE encoder vs frozen CoLiPri Primus encoder, on 128³
crops. Both at their native precision (VAE autocast-fp16, Primus bf16), warmup + CUDA sync.

  MONAI_DATA_DIRECTORY=/home/dpxuser/repos/NV-Generate-CTMR/temp_work_dir \
  .venv_thor/bin/python experiments/3d/synth_task_generation/bench_encoder_fwd.py \
    --repo /home/dpxuser/repos/NV-Generate-CTMR
(run from /home/dpxuser/dev/patch_icl)
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "experiments" / "3d"))
from gen_maisi_fast import build_args  # noqa: E402

COLIPRI = Path("/nfs/data/nii/data1/Analysis/camaret___in_context_segmentation/"
               "ANALYSIS_20251122/checkpoints/colipri/primus_colipri.json")


def timed(fn, n=15, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        torch.cuda.synchronize(); t = time.time()
        fn(); torch.cuda.synchronize()
        ts.append((time.time() - t) * 1e3)
    return float(np.mean(ts)), float(np.std(ts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=Path("/home/dpxuser/repos/NV-Generate-CTMR"))
    ap.add_argument("--env_file", default="environment_rflow-ct.json")
    ap.add_argument("--infer_file", default="config_infer_wholebody.json")
    a = ap.parse_args()
    dev = torch.device("cuda")

    from src.models.primus_encoder import PrimusEncoder
    primus = PrimusEncoder(str(COLIPRI), resolution=16, frozen=True, device="cuda",
                           native_grid=True, precision="bf16").eval()
    npar_p = sum(p.numel() for p in primus.primus.parameters())
    print(f"[load] Primus ready ({npar_p/1e6:.1f}M params)", flush=True)

    for mkey in [k for k in sys.modules if k == "scripts" or k.startswith("scripts.")]:
        del sys.modules[mkey]
    sys.path.insert(0, str(a.repo))
    from scripts.utils_infer import load_image_models
    args = build_args(a.repo, a.env_file, a.infer_file)
    args.autoencoder_def["num_splits"] = 1
    ae, dm, cn, sf, sched = load_image_models(args, dev); del dm, cn
    torch.cuda.empty_cache()
    npar_e = sum(p.numel() for p in ae.encoder.parameters()) if hasattr(ae, "encoder") else \
        sum(p.numel() for p in ae.parameters())
    print(f"[load] MAISI VAE ready (~{npar_e/1e6:.1f}M params, full AE)", flush=True)

    print(f"\n{'encoder':>16} {'B':>3} {'out shape':>18} {'ms/batch':>10} {'ms/crop':>9}", flush=True)
    for B in (1, 4):
        x = torch.randn(B, 1, 128, 128, 128, device=dev)

        def vae_enc():
            with torch.no_grad(), torch.amp.autocast("cuda"):
                return ae.encode(x)[0]
        with torch.no_grad(), torch.amp.autocast("cuda"):
            vo = tuple(ae.encode(x)[0].shape)
        m, s = timed(vae_enc)
        print(f"{'MAISI VAE enc':>16} {B:>3} {str(vo):>18} {m:>10.1f} {m/B:>9.1f}", flush=True)

        def prim_enc():
            with torch.no_grad():
                return primus(x)
        po = tuple(primus(x).shape)
        m, s = timed(prim_enc)
        print(f"{'Primus ViT':>16} {B:>3} {str(po):>18} {m:>10.1f} {m/B:>9.1f}", flush=True)


if __name__ == "__main__":
    main()
