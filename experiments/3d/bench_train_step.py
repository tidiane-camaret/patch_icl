"""End-to-end training-step benchmark (real loader + real model), for judging whether
a config is data-bound or compute-bound. Complements bench_dataloading.py (host-side
loading only) and bench_arch.py (model compute on random inputs at a fixed hardcoded arch).

Drives the REAL train_loader (crop path reads native ct.npy/label.npy — no nii.gz) and the
REAL model built by train.build_model, then times the exact train_epoch call path. Reports,
per step: data-wait, GPU compute (fwd+bwd+opt), total, and end-to-end steps/s — plus a
compute-only loop (one batch reused) so you can read off the data-vs-compute split.

    .venv_blackwell/bin/python experiments/3d/bench_train_step.py \
        --workers 16 --steps 40 -- \
        model=patchset3d arch.encoder=primus \
        'arch.primus_sidecar=${paths.colipri}/primus_colipri.json' \
        arch.l=2 arch.resolution=24 data.image_size=[192,192,192] data.use_crop=true
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.append(str(ROOT / "experiments" / "2d"))

import torch
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra

from common import DEVICE, train_loader
import train as T
from grid_metrics import target_like


def _cfg(workers, overrides):
    GlobalHydra.instance().clear()
    ov = ["experiment=22_totalseg_train_test", f"train.workers={workers}",
          "wandb.project=null"] + list(overrides)
    with initialize(config_path="../../configs/experiment/3d", version_base="1.3"):
        return compose(config_name="train", overrides=ov)


def _to_dev(batch):
    return (batch["image"].to(DEVICE, non_blocking=True),
            batch["context_in"].to(DEVICE, non_blocking=True),
            batch["context_out"].to(DEVICE, non_blocking=True),
            batch["label"].to(DEVICE, non_blocking=True).float())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--warmup", type=int, default=6)
    ap.add_argument("overrides", nargs="*", help="extra hydra overrides (after --)")
    args = ap.parse_args()

    cfg = _cfg(args.workers, args.overrides)
    if DEVICE.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    model, name = T.build_model(cfg)
    is_patchset = name == "patchset3d"
    net = getattr(model, "model", model)
    net.to(DEVICE).train()
    loss_fn = T.build_loss(cfg)
    # Single AdamW on all trainable params — faithful enough for timing (Muon's extra
    # Newton–Schulz is a small fraction of fwd+bwd; noted so the compute number isn't
    # read as the literal Muon+LAWA step).
    opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=1e-4)
    n_params = sum(p.numel() for p in net.parameters() if p.requires_grad) / 1e6
    B = int(cfg.train.batch_size)
    print(f"model={name} enc={cfg.arch.get('encoder','conv') if 'arch' in cfg else '-'} "
          f"size={list(cfg.data.image_size)} use_crop={cfg.data.use_crop} K={cfg.data.context_size} "
          f"B={B} workers={args.workers} | trainable={n_params:.1f}M")

    def step(dev_batch):
        img, cin, cout, lbl = dev_batch
        opt.zero_grad(set_to_none=True)
        with T._autocast():
            if is_patchset:
                out = model(img, context_in=cin, context_out=cout, mode="train")
                logits = out["final_logit"].float()
                target = target_like(lbl.unsqueeze(1), logits)
            else:
                logits = model.train_forward(img, cin, cout)
                target = lbl.unsqueeze(1)
            loss = loss_fn(logits, target)
        loss.backward()
        opt.step()
        return loss

    loader = train_loader(cfg)
    it = iter(loader)

    # ---- End-to-end: real loader + real step, split into data-wait vs compute ----
    data_ms = comp_ms = 0.0
    last_dev = None
    for i in range(args.warmup + args.steps):
        t0 = time.perf_counter()
        batch = next(it)
        dev = _to_dev(batch)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        step(dev)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        t2 = time.perf_counter()
        if i >= args.warmup:
            data_ms += (t1 - t0) * 1e3
            comp_ms += (t2 - t1) * 1e3
        last_dev = dev
    n = args.steps
    data_ms /= n; comp_ms /= n
    total_ms = data_ms + comp_ms
    peak = torch.cuda.max_memory_allocated() / 1e9 if DEVICE.type == "cuda" else 0.0

    # ---- Compute-only: reuse one on-device batch (isolates GPU-bound step time) ----
    for _ in range(args.warmup):
        step(last_dev)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        step(last_dev)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    comp_only_ms = (time.perf_counter() - t0) * 1e3 / n

    epoch_items = int(cfg.data.get("max_ds_len_train", 0) or 0)
    print(f"\n{'':<16}{'data ms':>9}{'compute ms':>12}{'total ms':>10}{'steps/s':>9}")
    print(f"{'end-to-end':<16}{data_ms:>9.1f}{comp_ms:>12.1f}{total_ms:>10.1f}{1e3/total_ms:>9.2f}")
    print(f"{'compute-only':<16}{'-':>9}{comp_only_ms:>12.1f}{comp_only_ms:>10.1f}{1e3/comp_only_ms:>9.2f}")
    bound = "DATA-bound" if data_ms > comp_ms else "COMPUTE-bound"
    print(f"\npeak GPU mem: {peak:.2f} GB | {bound} "
          f"(data {data_ms/total_ms*100:.0f}% / compute {comp_ms/total_ms*100:.0f}% of wall)")
    if epoch_items:
        print(f"projected epoch ({epoch_items} items / B={B}): "
              f"{epoch_items/B * total_ms/1e3:.0f}s end-to-end, "
              f"{epoch_items/B * comp_only_ms/1e3:.0f}s if data-perfect")


if __name__ == "__main__":
    main()
