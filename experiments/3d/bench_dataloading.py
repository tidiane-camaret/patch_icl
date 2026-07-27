"""Dataloading throughput vs image size for experiment=22_totalseg_train_test.

Drives the REAL train_loader (common.train_loader) — same dataset, collate, workers,
pin_memory, prefetch as training — and times pure host-side loading (no model). Shows
how image_size affects items/sec and, crucially, the pre-resize cache cliff: only
ct_{64,128}.npy exist, so other sizes fall to the slow native-load + CPU-interpolate path.

    .venv_thor/bin/python experiments/3d/bench_dataloading.py \
        --sizes 64 96 128 160 --workers 8 16 --n_timed 60
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra

from common import train_loader, _source_root


def _cfg(size, workers):
    GlobalHydra.instance().clear()
    with initialize(config_path="../../configs/experiment/3d", version_base="1.3"):
        return compose(config_name="train", overrides=[
            "experiment=22_totalseg_train_test",
            f"data.image_size=[{size},{size},{size}]",
            f"train.workers={workers}",
        ])


def _cache_hit(cfg, size) -> bool:
    """Does the pre-resized fast-path file exist (else slow native+interpolate path)?"""
    _, root, _ = _source_root(cfg)
    subj = next(p.name for p in Path(root).iterdir() if p.name.startswith("s0"))
    return (Path(root) / subj / f"ct_{size}x{size}x{size}.npy").exists()


def _batch_mb(batch) -> float:
    tot = 0
    for v in batch.values():
        if torch.is_tensor(v):
            tot += v.element_size() * v.nelement()
    return tot / 1024 ** 2


def bench(size, workers, n_warmup, n_timed):
    cfg = _cfg(size, workers)
    hit = _cache_hit(cfg, size)
    loader = train_loader(cfg)
    it = iter(loader)
    # Warmup: absorb worker spin-up + prefetch fill.
    mb = 0.0
    for _ in range(n_warmup):
        next(it)
    t0 = time.perf_counter()
    seen, last_mb = 0, 0.0
    for _ in range(n_timed):
        b = next(it)
        seen += b["image"].shape[0]
        last_mb = _batch_mb(b)
    dt = time.perf_counter() - t0
    del it, loader
    return {"size": size, "workers": workers, "path": "cache" if hit else "SLOW",
            "items_s": seen / dt, "ms_item": 1e3 * dt / seen, "batch_mb": last_mb}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", nargs="+", type=int, default=[64, 96, 128, 160])
    ap.add_argument("--workers", nargs="+", type=int, default=[8, 16])
    ap.add_argument("--n_warmup", type=int, default=8)
    ap.add_argument("--n_timed", type=int, default=60)
    args = ap.parse_args()

    print(f"{'size':>5} {'workers':>7} {'path':>6} {'items/s':>9} {'ms/item':>9} {'MB/batch':>9}")
    rows = []
    for nw in args.workers:
        for s in args.sizes:
            r = bench(s, nw, args.n_warmup, args.n_timed)
            rows.append(r)
            print(f"{r['size']:>5} {r['workers']:>7} {r['path']:>6} "
                  f"{r['items_s']:>9.1f} {r['ms_item']:>9.1f} {r['batch_mb']:>9.2f}", flush=True)
    return rows


if __name__ == "__main__":
    main()
