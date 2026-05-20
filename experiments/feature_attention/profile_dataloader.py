"""
Debug DataLoader bottleneck: per-step timing inside __getitem__,
worker-count scaling, and memory growth over batches.

Three phases
------------
1. Sub-step breakdown (main process, no workers)
   Replicates _get_synth_item step-by-step:
     load_ct    — np.load(ct_128x128x128.npy) + astype(float32)
     load_synth — np.load(synth_128x128x128.npy) + mask boolean op
     to_torch   — torch.from_numpy / .clone for image + mask
     aug        — apply_synth_aug × (context_size + 1)
     total      — full _get_synth_item() call (ground truth)

2. Worker-count scaling
   Measures batch delivery time at workers = 0, 1, 4, 8, 18.
   Computes the theoretical serial time (N_ITEMS × mean_total / workers)
   vs observed, showing the overhead / parallelism efficiency.

3. Memory growth
   Runs N_EPOCHS × N_BATCHES batches with the real training loader config
   and tracks main-process RSS and sum of worker-process RSS after each batch.

Usage
-----
    python experiments/feature_attention/profile_dataloader.py
    python experiments/feature_attention/profile_dataloader.py n_items=50 n_batches=30
    python experiments/feature_attention/profile_dataloader.py cluster=meta n_items=20
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, RandomSampler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Strip profiler-specific CLI args before load_config reads sys.argv
# ---------------------------------------------------------------------------
_raw = sys.argv[1:]
N_ITEMS   = next((int(a.split("=")[1]) for a in _raw if a.startswith("n_items=")),   40)
N_BATCHES = next((int(a.split("=")[1]) for a in _raw if a.startswith("n_batches=")), 20)
N_EPOCHS  = next((int(a.split("=")[1]) for a in _raw if a.startswith("n_epochs=")),   3)
sys.argv  = [sys.argv[0]] + [
    a for a in _raw if not a.startswith(("n_items=", "n_batches=", "n_epochs="))
]

from experiments.feature_attention.train import load_config
from src.totalseg_dataloader_incontext import (
    TotalSegInContextDataset, incontext_collate_fn,
)
from src.augmentations import apply_synth_aug


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _t() -> float:
    return time.perf_counter()


class Acc:
    """Accumulate timings and print stats."""
    def __init__(self, name: str):
        self.name = name
        self.times: list[float] = []

    def add(self, dt: float):
        self.times.append(dt)

    @property
    def mean_ms(self) -> float:
        return 1e3 * float(np.mean(self.times)) if self.times else 0.0

    @property
    def p95_ms(self) -> float:
        return 1e3 * float(np.percentile(self.times, 95)) if self.times else 0.0

    @property
    def total_s(self) -> float:
        return float(np.sum(self.times))


def rss_gb(include_workers: bool = False) -> tuple[float, float]:
    """Return (main_rss_gb, worker_rss_gb)."""
    import psutil
    main = psutil.Process().memory_info().rss / 1e9
    if not include_workers:
        return main, 0.0
    children = psutil.Process().children(recursive=True)
    workers = sum(c.memory_info().rss for c in children if c.is_running()) / 1e9
    return main, workers


def print_table(rows: list[dict], cols: list[tuple[str, str, int]]):
    """cols: list of (key, header, width)."""
    header = "  ".join(f"{h:>{w}}" for _, h, w in cols)
    sep    = "  ".join("-" * w for _, _, w in cols)
    print(header)
    print(sep)
    for r in rows:
        print("  ".join(f"{r[k]:>{w}}" for k, _, w in cols))


# ---------------------------------------------------------------------------
# Phase 1: sub-step breakdown in the main process
# ---------------------------------------------------------------------------

def phase1_substep(ds: TotalSegInContextDataset, n_items: int):
    """Replicate _get_synth_item with per-step timing."""
    print(f"\n{'='*60}")
    print(f"Phase 1 — sub-step breakdown  ({n_items} items, main process)")

    size_str   = ds._size_str
    synth_fname = ds._synth_fname
    aug_cfg    = ds.aug_cfg
    K          = ds.context_size

    accs = {k: Acc(k) for k in ("load_ct", "load_synth", "to_torch", "aug", "total")}

    for i in range(n_items):
        # Pick a random subject + SV (same logic as _get_synth_item)
        subj   = ds._synth_subjects[torch.randint(len(ds._synth_subjects), (1,)).item()]
        sv_ids = ds._synth_sv_ids[subj]
        sv_idx = int(sv_ids[torch.randint(len(sv_ids), (1,)).item()])
        subj_dir = ds.root / subj

        t0_total = _t()

        # --- load_ct ---
        t0 = _t()
        ct_pre = subj_dir / f"ct_{size_str}.npy" if size_str else None
        if ct_pre is not None and ct_pre.exists():
            arr_ct = np.load(ct_pre, mmap_mode="r").astype(np.float32)
        else:
            raise RuntimeError("Pre-sized CT not found — adjust size_str")
        accs["load_ct"].add(_t() - t0)

        # --- load_synth + boolean mask ---
        t0 = _t()
        sized_synth = subj_dir / synth_fname.replace(".npy", f"_{size_str}.npy")
        sv_vol = np.load(sized_synth, mmap_mode="r")
        mask   = (sv_vol == sv_idx).astype(np.uint8)
        accs["load_synth"].add(_t() - t0)

        # --- to_torch ---
        t0 = _t()
        image_t = torch.from_numpy(arr_ct).unsqueeze(0)   # (1, D, H, W)
        mask_t  = torch.from_numpy(mask).long()            # (D, H, W)
        # clone so mmap pages can be released
        image_t = image_t.clone()
        mask_t  = mask_t.clone()
        accs["to_torch"].add(_t() - t0)

        # --- aug ---
        t0 = _t()
        if aug_cfg is not None and aug_cfg.enabled:
            for _ in range(K + 1):
                apply_synth_aug(image_t.clone(), mask_t.clone(), aug_cfg.synth)
        accs["aug"].add(_t() - t0)

        accs["total"].add(_t() - t0_total)

    # Print
    print(f"{'Step':<14}  {'Mean':>8}  {'P95':>8}  {'Total':>8}  {'Frac':>6}")
    print(f"{'-'*14}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*6}")
    total_total = accs["total"].total_s
    for name, acc in accs.items():
        frac = acc.total_s / total_total if name != "total" else 1.0
        print(f"  {name:<12}  {acc.mean_ms:>7.1f}ms  {acc.p95_ms:>7.1f}ms  "
              f"{acc.total_s:>7.2f}s  {frac:>5.1%}")


# ---------------------------------------------------------------------------
# Phase 2: worker-count scaling
# ---------------------------------------------------------------------------

def _make_loader(ds, batch_size: int, num_workers: int) -> DataLoader:
    sampler = RandomSampler(ds, replacement=False, num_samples=min(512, len(ds)))
    return DataLoader(
        ds,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=(num_workers > 0),
        persistent_workers=(num_workers > 0),
        prefetch_factor=(2 if num_workers > 0 else None),
        collate_fn=incontext_collate_fn,
        drop_last=True,
    )


def phase2_scaling(ds: TotalSegInContextDataset, batch_size: int,
                   n_batches: int, worker_counts: list[int]):
    print(f"\n{'='*60}")
    print(f"Phase 2 — worker-count scaling  "
          f"(batch_size={batch_size}, {n_batches} batches each)")

    rows = []
    for nw in worker_counts:
        loader = _make_loader(ds, batch_size, nw)
        times = []
        # warm-up: 2 batches (discarded)
        it = iter(loader)
        for _ in range(min(2, n_batches)):
            next(it)

        t0 = _t()
        for bi, _ in enumerate(loader):
            if bi >= n_batches:
                break
            times.append(_t() - t0)
            t0 = _t()

        del loader   # shut down workers before next iteration

        if not times:
            continue
        mean_ms = 1e3 * float(np.mean(times))
        p95_ms  = 1e3 * float(np.percentile(times, 95))
        throughput = batch_size / (float(np.mean(times)))
        rows.append({"workers": nw, "mean_ms": f"{mean_ms:.0f}ms",
                     "p95_ms": f"{p95_ms:.0f}ms",
                     "samples/s": f"{throughput:.1f}"})
        print(f"  workers={nw:<3}  mean={mean_ms:>7.0f}ms  "
              f"p95={p95_ms:>7.0f}ms  {throughput:.1f} samples/s")

    return rows


# ---------------------------------------------------------------------------
# Phase 3: memory growth over batches
# ---------------------------------------------------------------------------

def phase3_memory(ds: TotalSegInContextDataset, batch_size: int,
                  num_workers: int, n_batches: int, n_epochs: int):
    print(f"\n{'='*60}")
    print(f"Phase 3 — memory growth  "
          f"(workers={num_workers}, {n_epochs} epochs × {n_batches} batches)")

    loader = _make_loader(ds, batch_size, num_workers)

    print(f"  {'Epoch':>5}  {'Batch':>5}  {'MainRSS':>9}  {'WorkRSS':>9}  {'Total':>9}")
    print(f"  {'-'*5}  {'-'*5}  {'-'*9}  {'-'*9}  {'-'*9}")

    for ep in range(1, n_epochs + 1):
        for bi, _ in enumerate(loader):
            if bi >= n_batches:
                break
            if bi % max(1, n_batches // 5) == 0 or bi == n_batches - 1:
                main_gb, work_gb = rss_gb(include_workers=(num_workers > 0))
                print(f"  {ep:>5}  {bi:>5}  {main_gb:>8.2f}G  {work_gb:>8.2f}G  "
                      f"{main_gb + work_gb:>8.2f}G")

    del loader


# ---------------------------------------------------------------------------
# Phase 4: collation cost
# ---------------------------------------------------------------------------

def phase4_collate(ds: TotalSegInContextDataset, batch_size: int, n_iters: int):
    """Time incontext_collate_fn alone on pre-fetched items."""
    print(f"\n{'='*60}")
    print(f"Phase 4 — collation cost  ({n_iters} iterations, batch_size={batch_size})")

    # Pre-fetch batch_size items in the main process
    items = [ds[i % len(ds)] for i in range(batch_size)]

    times = []
    for _ in range(n_iters):
        t0 = _t()
        incontext_collate_fn(items)
        times.append(_t() - t0)

    mean_ms = 1e3 * float(np.mean(times))
    p95_ms  = 1e3 * float(np.percentile(times, 95))
    print(f"  mean={mean_ms:.1f}ms  p95={p95_ms:.1f}ms")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    cfg = load_config()
    torch.manual_seed(cfg.train.seed)

    aug_cfg = None
    if cfg.train.aug:
        aug_yaml = ROOT / "configs" / "augmentations" / f"{cfg.train.aug_preset}.yaml"
        aug_cfg  = OmegaConf.load(aug_yaml).augmentations

    train_classes = list(cfg.data.train_classes)

    print(f"Building dataset…")
    ds = TotalSegInContextDataset(
        root=cfg.paths.totalseg, classes=train_classes,
        image_size=tuple(cfg.data.image_size), split="train",
        context_size=cfg.data.context_size, max_subjects=None,
        class_balanced=cfg.data.class_balanced, aug_cfg=aug_cfg,
        use_crop=cfg.data.use_crop, synth_method=cfg.data.synth_method or None,
        synth_unions=cfg.data.synth_unions, p_synth=cfg.data.p_synth,
        random_coloring=cfg.data.random_coloring,
        num_labels_per_sample=cfg.data.num_labels_per_sample,
    )

    BS = cfg.train.batch_size
    NW = cfg.train.workers

    phase1_substep(ds, n_items=N_ITEMS)
    phase4_collate(ds, batch_size=BS, n_iters=20)
    phase2_scaling(ds, batch_size=BS, n_batches=N_BATCHES,
                   worker_counts=[0, 1, 4, 8, min(NW, 18)])
    phase3_memory(ds, batch_size=BS, num_workers=NW,
                  n_batches=N_BATCHES, n_epochs=N_EPOCHS)

    print(f"\n{'='*60}")
    print("Done.")


if __name__ == "__main__":
    main()
