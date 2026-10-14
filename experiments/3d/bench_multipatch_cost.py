"""Pure compute/memory benchmark for PatchSet3D's forward pass vs (n_patches, image_size)
-- isolates the MODEL's own cost from data loading / native-GT Dice/NSD scoring (which
dominates wall-clock in the real eval sweep but is irrelevant to "how expensive is
attending over more/bigger windows"). Synthetic random tensors, no dataset/provider
involved, so this runs in seconds per point instead of minutes.

Usage (run from the REPO ROOT):
    python experiments/3d/bench_multipatch_cost.py eval.checkpoint=/path/to/best.pt
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import hydra
import torch

from eval import _build_model


@hydra.main(config_path="../../configs/experiment/3d", config_name="eval", version_base="1.3")
def main(cfg):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg.eval.model = "patchset3d"
    from omegaconf import OmegaConf
    OmegaConf.set_struct(cfg, False)

    K = 1                                   # context_size=1, matches the production config
    sizes = [128, 64, 32]
    n_values = [1, 2, 4, 8]
    reps = 5

    print(f"{'T':>5} {'n':>3} {'ms/fwd':>10} {'ms/window':>11} {'peak_mem_GB':>12}")
    for T in sizes:
        # The model's OWN fine-stage/decode-skip sizes are derived from cfg.data.image_size
        # at CONSTRUCTION time (train.py::build_model: "image_size": list(cfg.data.image_size)
        # -- the one arch field NOT restored from the checkpoint's stored arch), not from
        # whatever tensor shape is fed in at forward time. So a true "cost at T" measurement
        # needs the model rebuilt per T, matching what the real eval pipeline does when
        # data.image_size is overridden -- feeding a mismatched T into a model built for a
        # DIFFERENT T crashes in _decode_conv's skip-fusion (confirmed: that is a different,
        # genuine bug, not what this benchmark is measuring).
        cfg.data.image_size = [T, T, T]
        model = _build_model(cfg).to(dev).eval()
        for n in n_values:
            image = torch.randn(1, n, 1, T, T, T, device=dev)
            context_in = torch.randn(1, K * n, 1, T, T, T, device=dev)
            context_out = (torch.rand(1, K * n, T, T, T, device=dev) > 0.5).float()
            try:
                for _ in range(2):          # warmup (first call may lazily init/compile)
                    with torch.no_grad(), torch.autocast(
                        device_type=dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                        model(image, context_in=context_in, context_out=context_out,
                             mode="train", spacing=1.5)
                if dev.type == "cuda":
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                t0 = time.perf_counter()
                for _ in range(reps):
                    with torch.no_grad(), torch.autocast(
                        device_type=dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                        model(image, context_in=context_in, context_out=context_out,
                             mode="train", spacing=1.5)
                if dev.type == "cuda":
                    torch.cuda.synchronize()
                dt_ms = (time.perf_counter() - t0) / reps * 1000
                mem_gb = (torch.cuda.max_memory_allocated() / 1e9) if dev.type == "cuda" else float("nan")
                print(f"{T:>5} {n:>3} {dt_ms:>10.1f} {dt_ms / n:>11.1f} {mem_gb:>12.2f}")
            except torch.cuda.OutOfMemoryError:
                print(f"{T:>5} {n:>3} {'OOM':>10} {'-':>11} {'-':>12}")
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"{T:>5} {n:>3} ERROR: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
