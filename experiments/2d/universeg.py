"""
Evaluate UniverSeg on MedSegBench (native resolution — no pre-resize).

Images are loaded and fed to the model at data.image_size. UniverSegBaseline
handles its own internal resize; with input_size=image_size the model sees
native resolution.

Logs per-sample/class/dataset Dice and FLOPs to wandb.

Usage:
    python experiments/2d/universeg.py
    python experiments/2d/universeg.py data.image_size=256 data.context_size=5
    python experiments/2d/universeg.py data.dataset=abdomenus
"""

import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import hydra
import numpy as np
import torch
import wandb
from omegaconf import DictConfig
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DEVICE, build_loader, hard_dice, log_summary


def measure_flops(model, image_size: int, context_size: int) -> int:
    from torch.utils.flop_counter import FlopCounterMode
    dummy_img     = torch.zeros(1, 1, image_size, image_size, device=DEVICE)
    dummy_ctx_in  = torch.zeros(1, context_size, 1, image_size, image_size, device=DEVICE)
    dummy_ctx_out = torch.zeros(1, context_size, 1, image_size, image_size, device=DEVICE)
    with FlopCounterMode(display=False) as counter:
        with torch.no_grad():
            model(dummy_img, context_in=dummy_ctx_in, context_out=dummy_ctx_out, mode="val")
    return counter.get_total_flops()


@hydra.main(config_path="../../configs/experiment/2d", config_name="config", version_base=None)
def main(cfg: DictConfig):
    random.seed(cfg.eval.seed)
    torch.manual_seed(cfg.eval.seed)
    if DEVICE.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    loader = build_loader(cfg)

    # ── model ─────────────────────────────────────────────────────────────────
    from src.models.universeg_baseline import UniverSegBaseline
    print(f"Loading UniverSeg (size={cfg.data.image_size})...")
    model = UniverSegBaseline(pretrained=True, input_size=cfg.data.image_size).to(DEVICE).eval()

    # ── flops ─────────────────────────────────────────────────────────────────
    flops = measure_flops(model, cfg.data.image_size, cfg.data.context_size)
    print(f"FLOPs per sample (S={cfg.data.context_size}, {cfg.data.image_size}²): {flops/1e9:.2f} GFLOPs")

    # ── wandb ─────────────────────────────────────────────────────────────────
    run_name = cfg.wandb.name or f"{cfg.model}_s{cfg.data.image_size}_k{cfg.data.context_size}"
    run = wandb.init(
        project=cfg.wandb.project,
        name=run_name,
        config={
            "model":        cfg.model,
            "image_size":   cfg.data.image_size,
            "context_size": cfg.data.context_size,
            "split":        cfg.data.split,
            "flops":        flops,
        },
    )
    wandb.log({"flops_giga": flops / 1e9})
    sample_table = wandb.Table(columns=["dataset", "sample_idx", "label", "dice_ds", "dice_native"])

    # ── eval loop ─────────────────────────────────────────────────────────────
    per_ds:    dict[str, list[float]] = defaultdict(list)
    per_label: dict[str, list[float]] = defaultdict(list)
    inference_times: list[float] = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="eval"):
            if batch is None:
                continue

            B           = len(batch["dataset"])
            image       = batch["image"].to(DEVICE, non_blocking=True)
            labels      = batch["label"]                                        # CPU
            context_in  = batch["context_in"].to(DEVICE, non_blocking=True)
            context_out = batch["context_out"].to(DEVICE, non_blocking=True)

            t0 = time.perf_counter()
            with torch.autocast(device_type=DEVICE.type, enabled=DEVICE.type == "cuda"):
                out = model(image, context_in=context_in, context_out=context_out, mode="val")
            preds = (out["final_logit"] > 0).float().cpu()
            inference_times.append((time.perf_counter() - t0) / B)

            for b in range(B):
                ds_name     = batch["dataset"][b]
                sample_idx  = int(batch["sample_idx"][b])
                label_value = int(batch["label_value"][b])
                d           = hard_dice(preds[b, 0], labels[b, 0])

                per_ds[ds_name].append(d)
                per_label[f"{ds_name}/label_{label_value}"].append(d)
                # universeg predicts at native resolution: dice_ds == dice_native
                sample_table.add_data(ds_name, sample_idx, label_value, d, d)

    # ── aggregate & log ───────────────────────────────────────────────────────
    mean_t = float(np.mean(inference_times)) if inference_times else float("nan")
    print(f"\n  avg inference: {mean_t * 1000:.1f} ms/item")

    summary = log_summary(per_ds, per_label, sample_table,
                          extra={"time/inference_ms": mean_t * 1000,
                                 "time/total_ms":     mean_t * 1000})
    wandb.log(summary)
    run.finish()


if __name__ == "__main__":
    main()
