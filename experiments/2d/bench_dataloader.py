"""
Benchmark dataloading throughput: MedSegBench vs BiomedParse.

MedSegBench loads pre-resized npz into RAM (cheap per-item, heavy init);
BiomedParse decodes 1024x1024 RGBA PNGs lazily and resizes on the fly. This
measures steady-state samples/sec for batch_size=32, workers=16 at several sizes.

    .venv311/bin/python experiments/2d/bench_dataloader.py
    .venv311/bin/python experiments/2d/bench_dataloader.py --sizes 128 256 --batches 30
"""

import argparse
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))   # repo root (for `src`)
sys.path.insert(0, str(HERE))              # experiments/2d (for `common`)

from common import collate, TaggedDataset          # noqa: E402
from src.datasets.medsegbench import MedSegBenchDataset  # noqa: E402
from src.datasets.biomedparse import BiomedParseDataset  # noqa: E402

# Subsets chosen to give enough samples for the benchmark without a huge init.
MEDSEG_DATASETS = ["isic2018", "busi", "kvasir", "drive", "pandental"]
BIOMED_DATASETS = ["ISIC", "REFUGE", "GlaS", "DRIVE", "BreastUS"]


def time_loader(name: str, ds, batch_size: int, workers: int, n_batches: int) -> None:
    n = min(len(ds), (n_batches + 2) * batch_size)
    if n < len(ds):
        ds = torch.utils.data.Subset(TaggedDataset(ds), list(range(n)))
    else:
        ds = TaggedDataset(ds)

    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False, num_workers=workers,
        collate_fn=collate, pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
    )

    seen = 0
    t0 = None
    for i, batch in enumerate(loader):
        if batch is None:
            continue
        if i == 1:                       # start timing after warmup batch (workers spun up)
            t0 = time.perf_counter()
        if i >= 1:
            seen += batch["image"].shape[0]
        if i >= n_batches:
            break
    dt = time.perf_counter() - t0 if t0 else float("nan")
    rate = seen / dt if dt and dt > 0 else float("nan")
    print(f"    {name:14s}  {seen:5d} samples in {dt:6.2f}s  ->  {rate:8.1f} samples/s")
    return rate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[128, 256, 512])
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--batches", type=int, default=30)
    ap.add_argument("--context-size", type=int, default=3)
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    print(f"batch_size={args.batch_size} workers={args.workers} "
          f"batches={args.batches} context_size={args.context_size}\n")

    for size in args.sizes:
        print(f"image_size={size}")
        msb = MedSegBenchDataset(split=args.split, context_size=args.context_size,
                                 image_size=size, datasets=MEDSEG_DATASETS)
        time_loader("MedSegBench", msb, args.batch_size, args.workers, args.batches)
        del msb

        bmp = BiomedParseDataset(split=args.split, context_size=args.context_size,
                                 image_size=size, datasets=BIOMED_DATASETS)
        time_loader("BiomedParse", bmp, args.batch_size, args.workers, args.batches)
        del bmp
        print()


if __name__ == "__main__":
    main()
