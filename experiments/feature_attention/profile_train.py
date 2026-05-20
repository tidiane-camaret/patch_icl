"""
Profile training loop: per-phase timing, RAM, VRAM, and DataLoader worker memory.

Reproduces the full train.py setup (same config, same model, same DataLoader) but
runs only a few epochs with a capped batch count, then prints a breakdown table and
saves a CSV for deeper analysis.  Useful for identifying periodic slowdowns.

Phases timed per batch
  data      — waiting for DataLoader to yield the next batch (prefetch pipeline)
  encode    — STU-Net encoder forward (frozen, inference_mode)
  feat      — feature extraction + reshape to (B, N, C)
  labels    — mask downsample + label preparation
  model     — PatchICLAttention forward
  backward  — loss + backward + optimizer step

Epoch-level timings
  validation   — full val loop (encoder + model + metrics per item)
  fig_save     — matplotlib figure rendering + PNG writes

Memory snapshots (printed per epoch)
  VRAM alloc / reserved / peak
  CPU RAM (main process RSS)
  Workers RAM (sum RSS of DataLoader child processes)

Usage
-----
    python experiments/feature_attention/profile_train.py
    python experiments/feature_attention/profile_train.py n_epochs=3 n_batches=20
    python experiments/feature_attention/profile_train.py cluster=meta n_batches=10
"""

import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, RandomSampler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Strip profiler-specific CLI args before load_config reads sys.argv
# ---------------------------------------------------------------------------
_raw = sys.argv[1:]
N_EPOCHS  = next((int(a.split("=")[1]) for a in _raw if a.startswith("n_epochs=")),  3)
N_BATCHES = next((int(a.split("=")[1]) for a in _raw if a.startswith("n_batches=")), 20)
sys.argv = [sys.argv[0]] + [a for a in _raw if not a.startswith(("n_epochs=", "n_batches="))]

from experiments.feature_attention.train import (
    load_config,
    encode_image_only,
    extract_features,
    downsample_mask,
    save_val_figure,
    save_train_figures,
    validate,
)
from src.totalseg_dataloader_incontext import TotalSegInContextDataset, incontext_collate_fn
from src.models.encoders.stunet import STUNetEncoder
from experiments.feature_attention.model import PatchICLAttention


# ---------------------------------------------------------------------------
# Measurement helpers
# ---------------------------------------------------------------------------

def cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def ram_gb() -> float:
    import psutil
    return psutil.Process().memory_info().rss / 1e9


def worker_ram_gb() -> float:
    try:
        import psutil
        children = psutil.Process().children(recursive=True)
        return sum(c.memory_info().rss for c in children if c.is_running()) / 1e9
    except Exception:
        return 0.0


def vram_snapshot() -> dict:
    if not torch.cuda.is_available():
        return {"alloc": 0.0, "reserved": 0.0, "peak": 0.0}
    return {
        "alloc":    torch.cuda.memory_allocated()    / 1e9,
        "reserved": torch.cuda.memory_reserved()     / 1e9,
        "peak":     torch.cuda.max_memory_allocated() / 1e9,
    }


class Stopwatch:
    """Accumulate per-call timings for a named phase. CUDA-syncs before/after."""
    def __init__(self, name: str):
        self.name = name
        self.times: list[float] = []
        self._t0 = 0.0

    def start(self):
        cuda_sync()
        self._t0 = time.perf_counter()

    def stop(self):
        cuda_sync()
        self.times.append(time.perf_counter() - self._t0)

    @property
    def n(self) -> int:
        return len(self.times)

    @property
    def mean_ms(self) -> float:
        return 1e3 * float(np.mean(self.times)) if self.times else 0.0

    @property
    def total_s(self) -> float:
        return float(np.sum(self.times))

    @property
    def p50_ms(self) -> float:
        return 1e3 * float(np.median(self.times)) if self.times else 0.0

    @property
    def p95_ms(self) -> float:
        return 1e3 * float(np.percentile(self.times, 95)) if self.times else 0.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    cfg = load_config()
    results_dir = Path(cfg.paths.results) / "feature_attention"
    results_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.train.seed)


    device_str = cfg.train.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    out_size = tuple(cfg.data.output_size)
    amp      = device.type == "cuda"

    print(f"Profiling {N_EPOCHS} epoch(s) × {N_BATCHES} batch(es) | device={device}")
    print(f"  image_size={tuple(cfg.data.image_size)}  output_size={out_size}  "
          f"batch_size={cfg.train.batch_size}  workers={cfg.train.workers}")

    # ---- Augmentation config -----------------------------------------------
    aug_cfg = None
    if cfg.train.aug:
        aug_yaml = ROOT / "configs" / "augmentations" / f"{cfg.train.aug_preset}.yaml"
        aug_cfg  = OmegaConf.load(aug_yaml).augmentations

    train_classes = list(cfg.data.train_classes)
    val_classes   = list(cfg.data.val_classes) or train_classes

    # ---- Datasets + loader -------------------------------------------------
    ds_train = TotalSegInContextDataset(
        root=cfg.paths.totalseg, classes=train_classes,
        image_size=tuple(cfg.data.image_size), split="train",
        context_size=cfg.data.context_size, max_subjects=None,
        class_balanced=cfg.data.class_balanced, aug_cfg=aug_cfg,
        use_crop=cfg.data.use_crop, synth_method=cfg.data.synth_method or None,
        synth_unions=cfg.data.synth_unions, p_synth=cfg.data.p_synth,
        random_coloring=cfg.data.random_coloring,
        num_labels_per_sample=cfg.data.num_labels_per_sample,
    )
    ds_val = TotalSegInContextDataset(
        root=cfg.paths.totalseg, classes=val_classes,
        image_size=tuple(cfg.data.image_size), split="val",
        context_size=cfg.data.context_size, use_crop=cfg.data.use_crop,
    )

    n_train      = min(cfg.data.max_ds_len_train, len(ds_train))
    train_sampler = RandomSampler(ds_train, replacement=False, num_samples=n_train)
    train_loader  = DataLoader(
        ds_train, batch_size=cfg.train.batch_size, sampler=train_sampler,
        num_workers=cfg.train.workers, pin_memory=True,
        persistent_workers=cfg.train.workers > 0,
        prefetch_factor=2 if cfg.train.workers > 0 else None,
        collate_fn=incontext_collate_fn, drop_last=True,
    )
    print(f"  train: {n_train} samples, {len(train_loader)} batches/epoch")

    # ---- Encoder -----------------------------------------------------------
    encoder = STUNetEncoder(
        in_channels=1, variant=cfg.model.stunet_variant,
        pretrained=cfg.model.stunet_pretrained, freeze_encoder=True,
    ).to(device).eval()

    num_levels = len(encoder.skip_channels) + 1
    level      = cfg.model.feature_level

    with torch.inference_mode():
        dummy      = torch.zeros(1, 1, *cfg.data.image_size, device=device)
        dummy_feat = extract_features(encode_image_only(encoder, dummy), level, out_size, num_levels)
    embed_dim = dummy_feat.shape[1]
    print(f"  embed_dim={embed_dim}  dim={cfg.model.dim}  level={level}")

    # ---- Model -------------------------------------------------------------
    label_dim = 3 if cfg.data.random_coloring else 1
    model = PatchICLAttention(
        embed_dim=embed_dim, dim=cfg.model.dim, num_heads=cfg.model.num_heads,
        num_layers=cfg.model.num_layers, ff_factor=cfg.model.ff_factor,
        label_injection=cfg.model.label_injection, output_head=cfg.model.output_head,
        pos_encoding=cfg.model.pos_encoding, input_norm=cfg.model.input_norm,
        grid_size=out_size, dropout=cfg.model.dropout,
        ctx_self_attn=cfg.model.ctx_self_attn, log_n_scaling=cfg.model.log_n_scaling,
        log_n_base=cfg.model.log_n_base, label_dim=label_dim,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  PatchICLAttention params: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    scaler    = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    # ---- Temp fig dir for figure-save profiling ----------------------------
    fig_dir = Path("/tmp/patch_icl_profile_figs")
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ---- Records -----------------------------------------------------------
    batch_rows: list[dict] = []
    epoch_rows: list[dict] = []

    # ---- Profile loop ------------------------------------------------------
    for epoch in range(1, N_EPOCHS + 1):
        model.train()
        torch.cuda.reset_peak_memory_stats()

        clocks: dict[str, Stopwatch] = {
            k: Stopwatch(k) for k in ("data", "to_device", "encode", "feat", "labels", "model", "backward")
        }

        data_clock_start = time.perf_counter()

        for batch_idx, batch in enumerate(train_loader):
            if batch_idx >= N_BATCHES:
                break

            # data: time spent waiting for this batch to arrive (DataLoader + workers)
            cuda_sync()
            clocks["data"].times.append(time.perf_counter() - data_clock_start)

            # to_device: H2D transfer
            clocks["to_device"].start()
            images  = batch["image"].to(device, non_blocking=True)
            labels  = batch["label"].to(device, non_blocking=True)
            ctx_in  = batch["context_in"].to(device, non_blocking=True)
            ctx_out = batch["context_out"].to(device, non_blocking=True)
            B, K = ctx_in.shape[:2]
            clocks["to_device"].stop()

            # encode: frozen STU-Net encoder
            clocks["encode"].start()
            with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=amp):
                tgt_feats      = encode_image_only(encoder, images)
                ctx_feats_flat = encode_image_only(encoder, ctx_in.reshape(B * K, 1, *ctx_in.shape[3:]))
            clocks["encode"].stop()

            # feat: extract + reshape to (B, N, C)
            clocks["feat"].start()
            tgt_feat_ds      = extract_features(tgt_feats,      level, out_size, num_levels)
            ctx_feat_ds_flat = extract_features(ctx_feats_flat, level, out_size, num_levels)
            C = ctx_feat_ds_flat.shape[1]
            D_, H_, W_ = out_size
            N = D_ * H_ * W_
            ctx_feat_ds = ctx_feat_ds_flat.reshape(B, K, C, D_, H_, W_)
            tgt_feat = tgt_feat_ds.float().reshape(B, C, N).permute(0, 2, 1)
            ctx_feat = ctx_feat_ds.float().permute(0, 1, 3, 4, 5, 2).reshape(B, K * N, C)
            clocks["feat"].stop()

            # labels: mask downsample + binary label prep
            clocks["labels"].start()
            ctx_mask_ds = downsample_mask(
                ctx_out.reshape(B * K, *ctx_out.shape[2:]), out_size, cfg.data.mask_pool
            ).reshape(B, K, D_, H_, W_)
            tgt_mask_ds = downsample_mask(labels, out_size, cfg.data.mask_pool)
            ctx_lbls = (ctx_mask_ds.reshape(B, K * N) > 0).float()
            gt_loss  = (tgt_mask_ds.reshape(B, N) > 0).float()
            clocks["labels"].stop()

            # model: PatchICLAttention forward
            clocks["model"].start()
            with torch.autocast(device_type=device.type, enabled=amp):
                pred = model(tgt_feat, ctx_feat, ctx_lbls)
            clocks["model"].stop()

            # backward: loss + grad + optimizer
            clocks["backward"].start()
            loss = F.binary_cross_entropy(pred.float(), gt_loss)
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            clocks["backward"].stop()

            vm = vram_snapshot()
            batch_rows.append({
                "epoch": epoch, "batch": batch_idx,
                **{f"t_{k}_ms": 1e3 * clocks[k].times[-1] for k in clocks},
                "vram_alloc_gb": vm["alloc"],
            })

            data_clock_start = time.perf_counter()

        # ---- Validation ----------------------------------------------------
        t_val_start = time.perf_counter()
        cuda_sync()
        val_metrics, val_figs = validate(
            model, encoder, ds_val, level, out_size, num_levels,
            cfg.data.mask_pool, device, cfg.train.val_items_per_class,
            fig_dir=fig_dir, epoch=epoch,
        )
        cuda_sync()
        t_val = time.perf_counter() - t_val_start

        # ---- Figure saving -------------------------------------------------
        t_fig_start = time.perf_counter()
        train_vis_batch = {k: v[:2] for k, v in batch.items() if isinstance(v, torch.Tensor)}
        train_vis_batch["label_names"] = batch.get("label_names", ["s0", "s1"])[:2]
        train_vis_batch["context_in"]  = batch["context_in"][:2]
        train_vis_batch["context_out"] = batch["context_out"][:2]
        save_train_figures(train_vis_batch, pred.detach().float().cpu()[:2], out_size, fig_dir, epoch)
        t_fig = time.perf_counter() - t_fig_start

        import psutil
        vm_ep = vram_snapshot()
        cpu_ram   = ram_gb()
        work_ram  = worker_ram_gb()
        n_workers = len(psutil.Process().children(recursive=True))

        epoch_rows.append({
            "epoch":          epoch,
            "t_train_s":      sum(clocks[k].total_s for k in clocks),
            "t_val_s":        t_val,
            "t_fig_s":        t_fig,
            "vram_alloc_gb":  vm_ep["alloc"],
            "vram_peak_gb":   vm_ep["peak"],
            "cpu_ram_gb":     cpu_ram,
            "worker_ram_gb":  work_ram,
            "n_worker_procs": n_workers,
            "val_auroc":      val_metrics.get("val/auroc", float("nan")),
        })

        # ---- Per-epoch summary ---------------------------------------------
        batch_total = sum(clocks[k].total_s for k in clocks)
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}  ({N_BATCHES} batches profiled)")
        print(f"{'Phase':<12}  {'Mean':>8}  {'P95':>8}  {'Total':>8}  {'Frac':>6}")
        print(f"{'-'*12}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*6}")
        for name, sw in clocks.items():
            frac = sw.total_s / batch_total if batch_total > 0 else 0
            print(f"  {name:<10}  {sw.mean_ms:>7.1f}ms  {sw.p95_ms:>7.1f}ms  "
                  f"{sw.total_s:>7.2f}s  {frac:>5.1%}")
        print(f"  {'validation':<10}  {'':>8}  {'':>8}  {t_val:>7.2f}s")
        print(f"  {'fig_save':<10}  {'':>8}  {'':>8}  {t_fig:>7.2f}s")
        print(f"\nMemory")
        print(f"  VRAM  alloc={vm_ep['alloc']:.2f}GB  reserved={vm_ep['reserved']:.2f}GB  peak={vm_ep['peak']:.2f}GB")
        print(f"  CPU   main={cpu_ram:.2f}GB  workers={work_ram:.2f}GB  n_procs={n_workers}")

    # ---- Final cross-epoch summary -----------------------------------------
    print(f"\n{'='*60}")
    print("Cross-epoch summary")
    print(f"{'Epoch':<7}  {'Train':>8}  {'Val':>8}  {'FigSave':>8}  "
          f"{'VRAM_pk':>8}  {'CPUram':>7}  {'Wrkram':>7}")
    for r in epoch_rows:
        print(f"  {r['epoch']:<5}  {r['t_train_s']:>7.1f}s  {r['t_val_s']:>7.1f}s  "
              f"{r['t_fig_s']:>7.2f}s  {r['vram_peak_gb']:>7.2f}GB  "
              f"{r['cpu_ram_gb']:>6.2f}GB  {r['worker_ram_gb']:>6.2f}GB")

    # ---- Per-batch time series (detect slowdowns) --------------------------
    if len(batch_rows) > 1:
        print(f"\nPer-batch encode+model time (ms) — look for spikes:")
        for r in batch_rows:
            t = r["t_encode_ms"] + r["t_model_ms"]
            bar = "█" * int(t / 20)
            print(f"  ep{r['epoch']} b{r['batch']:02d}  {t:6.0f}ms  {bar}")

    # ---- Save CSV ----------------------------------------------------------
    csv_path = Path(results_dir) / "patch_icl_profile_batches.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=batch_rows[0].keys())
        writer.writeheader()
        writer.writerows(batch_rows)
    print(f"\nBatch CSV saved to {csv_path}")

    csv_path2 = Path(results_dir) / "patch_icl_profile_epochs.csv"
    with open(csv_path2, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=epoch_rows[0].keys())
        writer.writeheader()
        writer.writerows(epoch_rows)
    print(f"Epoch CSV saved to {csv_path2}")


if __name__ == "__main__":
    main()
