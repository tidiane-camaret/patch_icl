"""Micro-benchmark to pick batch_size for the 4_loss_eps_per_lvl medsegbench run on thor (A6000).

Builds the REAL PatchSetCNN via train.build_model on the resolved Hydra config
(same overrides as the target command) and times a full train step (fwd + surrogate
loss + bwd + AdamW/Muon step) across batch sizes, reporting peak GPU mem + it/s.
No data loading — pure compute ceiling. Run with the thor venv.
"""
import sys, time, statistics
from pathlib import Path

import torch
from hydra import initialize_config_dir, compose

_ROOT = Path("/home/dpxuser/dev/patch_icl")
sys.path.insert(0, str(_ROOT / "experiments" / "2d"))
sys.path.insert(0, str(_ROOT))

from train import build_model, _autocast  # noqa: E402
from pfn_train import Muon  # noqa: E402

DEVICE = torch.device("cuda")
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")

CFG_DIR = str(_ROOT / "configs" / "experiment" / "2d")
OVERRIDES = ["arch.full_attn=true", "train.epochs=500", "train.warmup_epochs=20",
            "aug_preset=2d_strong", "data.source=medsegbench"]

with initialize_config_dir(config_dir=CFG_DIR, version_base=None):
    cfg = compose(config_name="4_loss_eps_per_lvl", overrides=OVERRIDES)

H = cfg.data.image_size
K = cfg.data.context_size
print(f"image_size={H} context_size={K} full_attn={cfg.arch.get('full_attn')} "
      f"resolutions={list(cfg.arch.resolutions)}")

model, name, meta = build_model(cfg)
model = model.to(DEVICE)
nparams = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"model={name} params={nparams/1e6:.2f}M")

# Same optimizer split as train.py
muon_p = [p for n, p in model.named_parameters() if p.requires_grad and p.ndim == 2 and "transformer" in n]
adam_p = [p for n, p in model.named_parameters() if p.requires_grad and not (p.ndim == 2 and "transformer" in n)]
opts = [torch.optim.AdamW(adam_p, lr=3e-4, weight_decay=0.01)]
if muon_p:
    opts.append(Muon(muon_p, lr=0.1*3e-4, momentum=0.96, weight_decay=0.1))


def step(B):
    img = torch.randn(B, 1, H, H, device=DEVICE)
    cin = torch.randn(B, K, 1, H, H, device=DEVICE)
    cout = (torch.rand(B, K, 1, H, H, device=DEVICE) > 0.7).float()
    for o in opts:
        o.zero_grad(set_to_none=True)
    with _autocast():
        out = model(img, context_in=cin, context_out=cout, mode="train")
        loss = out["final_logit"].float().pow(2).mean()
        if out.get("refine_logit") is not None:
            loss = loss + out["refine_logit"].float().pow(2).mean()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    for o in opts:
        o.step()


print(f"\n{'B':>5} {'peak_GB':>8} {'it/s':>8} {'img/s':>9}")
for B in [64, 96, 128, 160, 192, 224, 256, 320, 384]:
    try:
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        for _ in range(3):  # warmup
            step(B)
        torch.cuda.synchronize()
        t0 = time.perf_counter(); N = 8
        for _ in range(N):
            step(B)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / N
        peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"{B:>5} {peak:>8.2f} {1/dt:>8.2f} {B/dt:>9.1f}")
    except RuntimeError as e:
        print(f"{B:>5}   OOM  ({str(e)[:50]})")
        torch.cuda.empty_cache()
        break
