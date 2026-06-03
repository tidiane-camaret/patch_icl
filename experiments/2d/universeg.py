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

import sys
from pathlib import Path

import hydra
import numpy as np
import torch
import wandb
from omegaconf import DictConfig
from collections import defaultdict
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, "/home/dpxuser/ic_segmentation")
sys.path.insert(0, "/home/dpxuser/repos/UniverSeg")

from src.datasets.medsegbench import MedSegBenchDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def dice(pred: torch.Tensor, gt: torch.Tensor) -> float:
    p, g = pred.flatten(), gt.flatten()
    return float(2 * (p * g).sum() / (p.sum() + g.sum() + 1e-8))


def measure_flops(model, image_size: int, context_size: int) -> int:
    from torch.utils.flop_counter import FlopCounterMode
    dummy_img     = torch.zeros(1, 1, image_size, image_size, device=DEVICE)
    dummy_ctx_in  = torch.zeros(1, context_size, 1, image_size, image_size, device=DEVICE)
    dummy_ctx_out = torch.zeros(1, context_size, 1, image_size, image_size, device=DEVICE)
    with FlopCounterMode(display=False) as counter:
        with torch.no_grad():
            model(dummy_img, context_in=dummy_ctx_in, context_out=dummy_ctx_out, mode="val")
    return counter.get_total_flops()


def collate(batch):
    batch = [b for b in batch if b["context_in"].shape[0] > 0]
    if not batch:
        return None
    return {
        "image":       torch.stack([b["image"]       for b in batch]),
        "label":       torch.stack([b["label"]       for b in batch]),
        "context_in":  torch.stack([b["context_in"]  for b in batch]),
        "context_out": torch.stack([b["context_out"] for b in batch]),
        "dataset":     [b["dataset"]     for b in batch],
        "sample_idx":  [b["sample_idx"]  for b in batch],
        "label_value": [b["label_value"] for b in batch],
    }


class TaggedDataset(torch.utils.data.Dataset):
    """Attaches (dataset, sample_idx, label_value) metadata to each item."""
    def __init__(self, inner):
        self.inner = inner

    def __len__(self):
        return len(self.inner)

    def __getitem__(self, idx):
        item = self.inner[idx]
        ds_name, sample_idx, label_value = self.inner.samples[idx]
        item["dataset"]     = ds_name
        item["sample_idx"]  = sample_idx
        item["label_value"] = label_value
        return item


@hydra.main(config_path="../../configs/experiment/2d", config_name="config", version_base=None)
def main(cfg: DictConfig):
    torch.manual_seed(cfg.eval.seed)

    # ── dataset ───────────────────────────────────────────────────────────────
    datasets = [cfg.data.dataset] if cfg.data.dataset else None
    ds = MedSegBenchDataset(
        split=cfg.data.split,
        context_size=cfg.data.context_size,
        image_size=cfg.data.image_size,
        datasets=datasets,
    )
    loader = DataLoader(
        TaggedDataset(ds),
        batch_size=cfg.eval.batch_size,
        shuffle=False,
        num_workers=cfg.eval.workers,
        collate_fn=collate,
    )

    # ── model ─────────────────────────────────────────────────────────────────
    from src.models.universeg_baseline import UniverSegBaseline
    print(f"Loading UniverSeg (native size={cfg.data.image_size})...")
    model = UniverSegBaseline(
        pretrained=True, input_size=cfg.data.image_size
    ).to(DEVICE).eval()

    # ── flops ─────────────────────────────────────────────────────────────────
    flops = measure_flops(model, cfg.data.image_size, cfg.data.context_size)
    print(f"FLOPs (S={cfg.data.context_size}, {cfg.data.image_size}²): {flops/1e9:.2f} GFLOPs")

    # ── wandb ─────────────────────────────────────────────────────────────────
    run_name = cfg.wandb.name or (
        f"{cfg.model}_s{cfg.data.image_size}_k{cfg.data.context_size}"
    )
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

    sample_table = wandb.Table(columns=["dataset", "sample_idx", "label", "dice"])

    # ── eval loop ─────────────────────────────────────────────────────────────
    per_ds:    dict[str, list[float]] = defaultdict(list)
    per_label: dict[str, list[float]] = defaultdict(list)

    with torch.no_grad():
        for batch in tqdm(loader, desc="eval"):
            if batch is None:
                continue

            image       = batch["image"].to(DEVICE)
            label       = batch["label"].to(DEVICE)
            context_in  = batch["context_in"].to(DEVICE)
            context_out = batch["context_out"].to(DEVICE)

            out   = model(image, context_in=context_in, context_out=context_out, mode="val")
            preds = (out["final_logit"] > 0).float()

            for i in range(len(batch["dataset"])):
                ds_name     = batch["dataset"][i]
                sample_idx  = int(batch["sample_idx"][i])
                label_value = int(batch["label_value"][i])
                d           = dice(preds[i, 0], label[i, 0])

                per_ds[ds_name].append(d)
                per_label[f"{ds_name}/label_{label_value}"].append(d)
                sample_table.add_data(ds_name, sample_idx, label_value, d)

    # ── aggregate & log ───────────────────────────────────────────────────────
    summary = {}

    print(f"\n{'Dataset':>25}  {'N':>5}  {'Dice':>6}")
    print("-" * 42)
    all_scores = []
    for name in sorted(per_ds):
        scores = per_ds[name]
        mean   = float(np.mean(scores))
        all_scores.extend(scores)
        summary[f"dice/dataset/{name}"] = mean
        print(f"{name:>25}  {len(scores):>5}  {mean:.4f}")
    print("-" * 42)
    overall = float(np.mean(all_scores))
    summary["dice/mean"] = overall
    print(f"{'MEAN':>25}  {len(all_scores):>5}  {overall:.4f}")

    for key, scores in per_label.items():
        summary[f"dice/class/{key}"] = float(np.mean(scores))

    summary["samples"] = sample_table
    wandb.log(summary)
    run.finish()


if __name__ == "__main__":
    main()
