"""
TRT-decoder spike: wrap the MAISI autoencoder DECODER with monai.trt_compile and bench
against the eager and torch.compile paths on 96³/128³ crops. MAISI's own production path
(config_trt.json: trt_compile(autoencoder, ..., submodule='decoder')).

  MONAI_DATA_DIRECTORY=/home/dpxuser/repos/NV-Generate-CTMR/temp_work_dir \
  .venv_thor/bin/python experiments/3d/synth_task_generation/bench_vae_trt.py \
    --repo /home/dpxuser/repos/NV-Generate-CTMR --crop 96
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gen_maisi_fast import build_args  # noqa: E402


def timed(fn, n=20, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        torch.cuda.synchronize(); t = time.time(); fn(); torch.cuda.synchronize()
        ts.append((time.time() - t) * 1e3)
    return float(np.mean(ts)), float(np.std(ts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=Path("/home/dpxuser/repos/NV-Generate-CTMR"))
    ap.add_argument("--env_file", default="environment_rflow-ct.json")
    ap.add_argument("--infer_file", default="config_infer_wholebody.json")
    ap.add_argument("--crop", type=int, default=96)
    ap.add_argument("--engine_dir", type=Path, default=Path("/tmp/maisi_trt_engines"))
    a = ap.parse_args()
    a.engine_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(a.repo))
    from monai.networks import trt_compile
    from scripts.utils_infer import ReconModel, load_image_models

    dev = torch.device("cuda")
    args = build_args(a.repo, a.env_file, a.infer_file)
    args.autoencoder_def["num_splits"] = 1
    lc = a.crop // 4
    z = torch.randn(1, args.latent_channels, lc, lc, lc, device=dev).half()

    def new_ae():
        ae, dm, cn, sf, _ = load_image_models(args, dev); del dm, cn
        torch.cuda.empty_cache()
        return ae.half().eval(), sf

    # ---- 1. eager (pure fp16) ----
    ae, sf = new_ae()
    recon = ReconModel(autoencoder=ae, scale_factor=sf).to(dev).eval()
    with torch.no_grad():
        m_eager, s = timed(lambda: recon(z))
    print(f"[eager   fp16] {a.crop}^3: {m_eager:7.1f} ± {s:.1f} ms", flush=True)

    # ---- 2. torch.compile ----
    crecon = torch.compile(ReconModel(autoencoder=ae, scale_factor=sf).to(dev).eval(), mode="default")
    with torch.no_grad():
        m_comp, s = timed(lambda: crecon(z), n=10, warmup=6)
    print(f"[compile fp16] {a.crop}^3: {m_comp:7.1f} ± {s:.1f} ms", flush=True)

    # ---- 3. TRT (submodule='decoder', fp16). Static shape for this crop size. ----
    ae2, sf2 = new_ae()
    plan = str(a.engine_dir / f"maisi_dec_{a.crop}")
    # static engine built from the first-seen input shape (this crop's latent); fp16.
    trt_args = {"precision": "fp16"}
    print(f"[trt] building engine (first forward, minutes)... plan={plan}", flush=True)
    t0 = time.time()
    trt_compile(ae2, plan, args=trt_args, submodule="decoder")
    recon_trt = ReconModel(autoencoder=ae2, scale_factor=sf2).to(dev).eval()
    with torch.no_grad():
        _ = recon_trt(z)  # triggers ONNX export + engine build
    torch.cuda.synchronize()
    print(f"[trt] engine ready in {time.time()-t0:.0f}s", flush=True)
    with torch.no_grad():
        m_trt, s = timed(lambda: recon_trt(z))
    print(f"[TRT     fp16] {a.crop}^3: {m_trt:7.1f} ± {s:.1f} ms", flush=True)

    print(f"\n=== decode {a.crop}^3 (latent 4x{lc}^3) ===", flush=True)
    print(f"  eager   : {m_eager:7.1f} ms  (1.0x)", flush=True)
    print(f"  compile : {m_comp:7.1f} ms  ({m_eager/m_comp:.1f}x)", flush=True)
    print(f"  TRT     : {m_trt:7.1f} ms  ({m_eager/m_trt:.1f}x vs eager, "
          f"{m_comp/m_trt:.1f}x vs compile)", flush=True)


if __name__ == "__main__":
    main()
