"""
Feature-similarity experiment on MedSegBench using the UniverSeg encoder.

Each image is encoded independently through the UniverSeg encoder blocks with a
zero dummy support (no cross-conv influence). Target patch features are then
classified by a TabPFN model fit on context patch features and their labels.

Procedure (mirrors experiments/feature_similarity/run.py for 3D):
  1. Encode target and each context image independently → per-scale feature maps
  2. Pick encoder level(s) and pool to output_size × output_size
  3. Downsample GT masks to the same output_size
  4. For each target: fit TabPFN on context (feature, label) patches → predict target

Usage:
    python experiments/2d/feature_sim.py
    python experiments/2d/feature_sim.py feature.level=-1 feature.output_size=8
    python experiments/2d/feature_sim.py data.dataset=abdomenus data.context_size=5
"""

import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import hydra
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from omegaconf import DictConfig
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DEVICE, build_loader, hard_dice, downsample_mask, log_summary


# ── UniverSeg image-only encoding ────────────────────────────────────────────

@torch.no_grad()
def encode_images(
    useg,
    images: torch.Tensor,
    masks: torch.Tensor | None = None,
) -> list[torch.Tensor]:
    """
    Extract per-scale features from the UniverSeg encoder.

    Support is always a dummy (zeros image channel), but the label channel of
    the support can optionally carry the GT mask. This lets the cross-conv fuse
    mask location into the features without the self-support trick.

    Args:
        useg:   inner UniverSeg nn.Module
        images: (B, 1, H, W) float [0, 1]
        masks:  (B, 1, H, W) binary, optional. When provided, used as the
                support label channel (second channel of dummy support).

    Returns:
        list of 4 tensors, each (B, 64, H/2^i, W/2^i)
    """
    B, _, H, W = images.shape
    target  = images.unsqueeze(1)                                # (B, 1, 1, H, W)
    dev = images.device
    label_ch = masks.to(dev) if masks is not None \
               else torch.zeros(B, 1, H, W, device=dev)
    dummy_s = torch.cat([
        torch.zeros(B, 1, 1, H, W, device=dev),                 # image channel = 0
        label_ch.unsqueeze(1),                                   # label channel
    ], dim=2)                                                    # (B, 1, 2, H, W)

    feats = []
    for i, block in enumerate(useg.enc_blocks):
        target, dummy_s = block(target, dummy_s)
        feats.append(target.squeeze(1))  # (B, C, H', W')
        if i < len(useg.enc_blocks) - 1:
            target  = F.max_pool2d(target .squeeze(1), 2).unsqueeze(1)
            dummy_s = F.max_pool2d(dummy_s.squeeze(1), 2).unsqueeze(1)

    return feats  # index 0 = highest res, -1 = bottleneck


# ── Feature helpers ───────────────────────────────────────────────────────────

def extract_features_batch(
    feats: list[torch.Tensor],
    level: str | int,
    output_size: int,
) -> torch.Tensor:
    """
    Pool encoder feature maps to output_size x output_size.

    feats: list of (N, C, H', W') tensors where N = B or B*K.
    level="all" concatenates all levels on the channel dim.
    level=int picks a single level (negative indexing supported).
    Returns (N, C', os, os).
    """
    size = (output_size, output_size)
    if str(level) == "all":
        maps = [F.adaptive_avg_pool2d(f.float(), size) for f in feats]
    else:
        idx = int(level) % len(feats)
        maps = [F.adaptive_avg_pool2d(feats[idx].float(), size)]
    return torch.cat(maps, dim=1)   # (N, C', os, os)


# ── TabPFN prediction ─────────────────────────────────────────────────────────

def balance_context(
    ctx_flat: torch.Tensor,
    labels_flat: torch.Tensor,
    bg_ratio: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Subsample background patches to bg_ratio × effective_fg_count."""
    n_eff_fg = max(1, round(labels_flat.sum().item()))
    order    = labels_flat.argsort(descending=True)
    keep_fg  = order[:n_eff_fg]
    bg_cands = order[n_eff_fg:]
    if len(bg_cands) == 0:
        return ctx_flat, labels_flat
    n_bg  = min(len(bg_cands), max(1, int(n_eff_fg * bg_ratio)))
    perm  = torch.randperm(len(bg_cands), device=bg_cands.device)[:n_bg]
    keep  = torch.cat([keep_fg, bg_cands[perm]]).sort().values
    return ctx_flat[keep], labels_flat[keep]


def predict_tabpfn(
    tgt_feat:  torch.Tensor,   # (C, H', W')
    ctx_feats: torch.Tensor,   # (K, C, H', W')
    ctx_masks: torch.Tensor,   # (K, H', W') float
    clf,
    balance_ratio: float | None = None,
) -> torch.Tensor:
    """
    Fit TabPFN on context (feature, label) patches; predict soft mask for target.

    Returns (H', W') float tensor in [0, 1].
    Falls back to context positive rate when all context labels are the same class.
    """
    C, H, W = tgt_feat.shape
    K = ctx_feats.shape[0]
    N = H * W

    ctx_flat   = ctx_feats.reshape(K, C, N).permute(0, 2, 1).reshape(K * N, C)
    ctx_labels = (ctx_masks.reshape(K * N) > 0).float()

    if balance_ratio is not None:
        ctx_flat, ctx_labels = balance_context(ctx_flat, ctx_labels, balance_ratio)

    X_ctx = ctx_flat.cpu().numpy()
    y_ctx = ctx_labels.cpu().numpy().astype(int)
    X_tgt = tgt_feat.reshape(C, N).T.cpu().numpy()

    if y_ctx.sum() == 0 or y_ctx.sum() == len(y_ctx):
        fill = float(y_ctx.mean())
        return torch.full((H, W), fill, dtype=torch.float32)

    # Per-feature z-score (matches TabPFN pretraining)
    mu  = X_ctx.mean(axis=0, keepdims=True)
    sig = X_ctx.std( axis=0, keepdims=True) + 1e-8
    X_ctx = (X_ctx - mu) / sig
    X_tgt = (X_tgt - mu) / sig

    clf.fit(X_ctx, y_ctx)
    proba = clf.predict_proba(X_tgt)    # (N, 2)
    return torch.from_numpy(proba[:, 1]).float().reshape(H, W)


def batch_tabpfn(
    tgt_feats:  torch.Tensor,   # (B, C, H', W')
    ctx_feats:  torch.Tensor,   # (B, K, C, H', W')
    ctx_masks:  torch.Tensor,   # (B, K, H', W') float — may be on CPU
    model,                      # clf.models_[0]
    n_estimators: int,
    n_classes: int = 2,
) -> torch.Tensor:
    """
    Batched TabPFN prediction: calls the underlying transformer once per
    estimator with all B samples stacked on the batch dimension.

    Returns (B, H', W') float tensor in [0, 1].
    Falls back to constant (context positive rate) per sample when all
    context labels are one class.
    """
    B, C, H, W = tgt_feats.shape
    K  = ctx_feats.shape[1]
    N  = H * W
    dev = next(model.parameters()).device

    # Reshape to (B, N_ctx/N_tgt, C)
    X_ctx = ctx_feats.reshape(B, K, C, N).permute(0, 1, 3, 2).reshape(B, K * N, C)
    y_ctx = (ctx_masks.reshape(B, K * N) > 0).float()
    X_tgt = tgt_feats.reshape(B, C, N).permute(0, 2, 1)  # (B, N, C)

    # Per-sample z-score using context stats
    mu  = X_ctx.mean(dim=1, keepdim=True)   # (B, 1, C)
    sig = X_ctx.std( dim=1, keepdim=True) + 1e-8
    X_ctx = (X_ctx - mu) / sig
    X_tgt = (X_tgt - mu) / sig

    # Identify samples with degenerate labels; will fill after
    label_sums = y_ctx.sum(dim=1)                          # (B,)
    degenerate = (label_sums == 0) | (label_sums == K * N)

    # Build (N_ctx + N_tgt, B, C) and (N_ctx, B) for model
    X_all = torch.cat([X_ctx, X_tgt], dim=1).permute(1, 0, 2).to(dev)  # (N_ctx+N_tgt, B, C)
    Y_all = y_ctx.permute(1, 0).to(dev)                                  # (N_ctx, B)

    # n_estimators calls with different column permutations → average probabilities
    proba_sum = None
    with torch.inference_mode():
        for _ in range(n_estimators):
            perm = torch.randperm(C, device=dev)
            out = model(X_all[:, :, perm], Y_all, only_return_standard_out=True)
            # out: (N_tgt, B, 160) — only test rows, first n_classes cols matter
            logits = out[-N:, :, :n_classes]
            p = torch.softmax(logits, dim=-1)
            proba_sum = p if proba_sum is None else proba_sum + p

    proba = proba_sum / n_estimators                           # (N, B, 2)
    preds = proba[..., 1].permute(1, 0).reshape(B, H, W)      # (B, H, W)

    # Overwrite degenerate samples with their context positive rate
    for b in range(B):
        if degenerate[b]:
            preds[b] = float(y_ctx[b].mean())

    return preds.cpu()


# ── Metrics ───────────────────────────────────────────────────────────────────

def dice_at_native(
    pred_ds: torch.Tensor, gt_native: torch.Tensor, native_size: int
) -> float:
    """Upsample patch-level prediction to native resolution and compute Dice."""
    pred_up = F.interpolate(
        pred_ds.unsqueeze(0).unsqueeze(0).float(),
        size=(native_size, native_size),
        mode="bilinear", align_corners=False,
    ).squeeze()
    return hard_dice(pred_up, gt_native)


# ── Visualisation ─────────────────────────────────────────────────────────────

def _overlay_ax(ax, image: np.ndarray, mask: np.ndarray, title: str) -> None:
    """Grayscale image with red mask overlay."""
    ax.imshow(image, cmap="gray", vmin=0, vmax=1)
    ax.imshow(mask,  cmap="Reds", alpha=0.45, vmin=0, vmax=1)
    ax.set_title(title, fontsize=8)
    ax.axis("off")


def _heatmap_ax(ax, arr: np.ndarray, title: str) -> None:
    ax.imshow(arr, cmap="hot", vmin=0, vmax=1, interpolation="nearest")
    ax.set_title(title, fontsize=8)
    ax.axis("off")


def save_figure(
    tgt_image:   np.ndarray,         # (H, W)  float [0,1]
    tgt_gt:      np.ndarray,         # (H, W)  binary
    tgt_gt_ds:   np.ndarray,         # (os, os)
    pred_ds:     np.ndarray,         # (os, os)  soft prediction
    ctx_images:  list[np.ndarray],   # K × (H, W)
    ctx_gts:     list[np.ndarray],   # K × (H, W)
    ctx_gts_ds:  list[np.ndarray],   # K × (os, os)
    out_path:    Path,
    title:       str = "",
) -> None:
    """
    Row 0: target+GT overlay | GT@output_size | prediction@output_size
    Row 1: ctx_k+GT overlay  | ctx GT@output_size   (repeated for each k)
    """
    K     = len(ctx_images)
    ncols = max(3, 2 * K)
    span  = ncols // 3

    fig, axes = plt.subplots(2, ncols, figsize=(3.2 * ncols, 6.5))
    fig.subplots_adjust(hspace=0.35, wspace=0.05)

    # Row 0
    _overlay_ax(axes[0, 0],          tgt_image, tgt_gt,   "Target + GT")
    _heatmap_ax(axes[0, span],       tgt_gt_ds,            f"GT ↓{tgt_gt_ds.shape[0]}")
    _heatmap_ax(axes[0, 2 * span],   pred_ds,              "Prediction")
    for col in range(1, span):
        axes[0, col].axis("off")
    for col in range(span + 1, 2 * span):
        axes[0, col].axis("off")
    for col in range(2 * span + 1, ncols):
        axes[0, col].axis("off")

    # Row 1: contexts
    for k in range(K):
        _overlay_ax(axes[1, 2 * k],     ctx_images[k], ctx_gts[k], f"Ctx {k} + GT")
        _heatmap_ax(axes[1, 2 * k + 1], ctx_gts_ds[k],             f"Ctx {k} GT ↓")
    for col in range(2 * K, ncols):
        axes[1, col].axis("off")

    fig.suptitle(title, fontsize=9)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

@hydra.main(config_path="../../configs/experiment/2d", config_name="feature_sim", version_base=None)
def main(cfg: DictConfig):
    random.seed(cfg.eval.seed)
    torch.manual_seed(cfg.eval.seed)
    if DEVICE.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    loader = build_loader(cfg)

    # ── model ─────────────────────────────────────────────────────────────────
    from src.models.universeg_baseline import UniverSegBaseline
    print("Loading UniverSeg encoder...")
    wrapper = UniverSegBaseline(pretrained=True, input_size=cfg.data.image_size)
    useg    = wrapper.model.to(DEVICE).eval()  # encoder only; decoder stays off GPU
    del wrapper
    torch.cuda.empty_cache()

    # ── flops (encoder only, 1 target + K context images) ─────────────────────
    from torch.utils.flop_counter import FlopCounterMode
    _n = 1 + cfg.data.context_size
    _dummy = torch.zeros(_n, 1, cfg.data.image_size, cfg.data.image_size, device=DEVICE)
    with FlopCounterMode(display=False) as _fc:
        with torch.no_grad():
            encode_images(useg, _dummy)
    flops = _fc.get_total_flops()
    print(f"Encoder FLOPs per sample (1+{cfg.data.context_size} images, {cfg.data.image_size}²): {flops/1e9:.2f} GFLOPs")
    del _dummy

    # ── TabPFN ────────────────────────────────────────────────────────────────
    from tabpfn import TabPFNClassifier
    _mem_mode: bool | str = {"auto": "auto", "true": True, "false": False}[
        str(cfg.tabpfn.memory_saving).lower()
    ]
    clf = TabPFNClassifier(
        n_estimators=cfg.tabpfn.n_estimators,
        device=str(DEVICE),
        ignore_pretraining_limits=True,
        memory_saving_mode=_mem_mode,
    )
    # Initialize model weights with a tiny dummy fit
    _C = 4 * 64 if str(cfg.feature.level) == "all" else 64  # rough feature dim
    _rng = np.random.default_rng(0)
    clf.fit(_rng.standard_normal((4, _C)).astype(np.float32), np.array([0, 0, 1, 1]))
    tabpfn_model  = clf.models_[0]
    tabpfn_n_est  = cfg.tabpfn.n_estimators
    print(f"TabPFN ready  n_estimators={tabpfn_n_est}  (batched inference)")

    # ── wandb ─────────────────────────────────────────────────────────────────
    level_tag = str(cfg.feature.level).replace("-", "m")
    run_name  = cfg.wandb.name or (
        f"{cfg.model}_lvl{level_tag}_os{cfg.feature.output_size}"
        f"_s{cfg.data.image_size}_k{cfg.data.context_size}"
    )
    run = wandb.init(
        project=cfg.wandb.project,
        name=run_name,
        config={
            "model":         cfg.model,
            "image_size":    cfg.data.image_size,
            "context_size":  cfg.data.context_size,
            "split":         cfg.data.split,
            "feature_level": str(cfg.feature.level),
            "output_size":   cfg.feature.output_size,
            "mask_pool":     cfg.feature.mask_pool,
            "n_estimators":  cfg.tabpfn.n_estimators,
            "balance_ratio": cfg.tabpfn.balance_ratio,
            "memory_saving": cfg.tabpfn.memory_saving,
            "flops_encoder": flops,
        },
    )
    wandb.log({"flops_giga": flops / 1e9})
    sample_table = wandb.Table(
        columns=["dataset", "sample_idx", "label", "dice_ds", "dice_native"]
    )

    out_dir = Path(cfg.eval.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── eval loop ─────────────────────────────────────────────────────────────
    per_ds:    dict[str, list[float]] = defaultdict(list)
    per_label: dict[str, list[float]] = defaultdict(list)
    encode_times: list[float] = []
    tabpfn_times: list[float] = []
    saved_figures: set[tuple[str, int]] = set()

    if cfg.tabpfn.balance_ratio is not None:
        print("WARNING: balance_ratio is set — batched TabPFN is disabled; inference will be slower.")

    with torch.no_grad():
        for batch in tqdm(loader, desc="eval"):
            if batch is None:
                continue

            B = len(batch["dataset"])
            images       = batch["image"].to(DEVICE, non_blocking=True)       # (B, 1, H, W)
            labels       = batch["label"]                                      # (B, 1, H, W) on CPU
            context_ins  = batch["context_in"].to(DEVICE, non_blocking=True)  # (B, K, 1, H, W)
            context_outs = batch["context_out"]                                # (B, K, 1, H, W) on CPU
            K = context_ins.shape[1]
            H, W = images.shape[-2], images.shape[-1]

            ctx_imgs_flat = context_ins.reshape(B * K, 1, H, W)  # (B*K, 1, H, W)

            t0 = time.perf_counter()

            # ── Single GPU encoding pass (fp16 autocast) ─────────────────────
            # When context_mask=False both targets and contexts use zero dummy
            # support → combine into one forward call (mirrors the 3D script).
            with torch.autocast(device_type=DEVICE.type, enabled=DEVICE.type == "cuda"):
                if not cfg.feature.context_mask:
                    all_feats = encode_images(useg, torch.cat([images, ctx_imgs_flat], dim=0))
                    tgt_feats_b = [f[:B]  for f in all_feats]   # list[(B,  C, H', W')]
                    ctx_feats_b = [f[B:]  for f in all_feats]   # list[(B*K,C, H', W')]
                else:
                    tgt_feats_b = encode_images(useg, images)
                    ctx_masks_enc = context_outs.reshape(B * K, 1, H, W).to(DEVICE, non_blocking=True)
                    ctx_feats_b = encode_images(useg, ctx_imgs_flat, masks=ctx_masks_enc)

            encode_times.append((time.perf_counter() - t0) / B)

            # ── Batched feature extraction (vectorized, fp32 for TabPFN) ─────
            os = cfg.feature.output_size
            tgt_feats_all = extract_features_batch(tgt_feats_b, cfg.feature.level, os)
            ctx_feats_raw = extract_features_batch(ctx_feats_b, cfg.feature.level, os)
            C_feat = tgt_feats_all.shape[1]
            ctx_feats_all = ctx_feats_raw.reshape(B, K, C_feat, os, os)

            ctx_outs_flat = context_outs.reshape(B * K, 1, H, W).float()
            _pool = F.adaptive_max_pool2d if cfg.feature.mask_pool == "max" \
                    else F.adaptive_avg_pool2d
            ctx_masks_all = _pool(ctx_outs_flat, (os, os)).squeeze(1).reshape(B, K, os, os)

            # ── Batched TabPFN (one model call per estimator) ─────────────────
            t1 = time.perf_counter()
            if cfg.tabpfn.balance_ratio is None:
                preds_all = batch_tabpfn(
                    tgt_feats_all, ctx_feats_all, ctx_masks_all,
                    tabpfn_model, tabpfn_n_est,
                )  # (B, os, os) on CPU
            else:
                # balance_ratio requires per-sample subsampling → falls back to serial
                preds_all = torch.stack([
                    predict_tabpfn(tgt_feats_all[b], ctx_feats_all[b], ctx_masks_all[b],
                                   clf, cfg.tabpfn.balance_ratio)
                    for b in range(B)
                ])  # (B, os, os)
            tabpfn_times.append((time.perf_counter() - t1) / B)

            for b in range(B):
                pred_ds     = preds_all[b]
                label       = labels[b, 0]
                ds_name     = batch["dataset"][b]
                sample_idx  = int(batch["sample_idx"][b])
                label_value = int(batch["label_value"][b])

                d_ds     = hard_dice(pred_ds, downsample_mask(label, cfg.feature.output_size))
                d_native = dice_at_native(pred_ds, label, cfg.data.image_size)

                per_ds[ds_name].append(d_native)
                per_label[f"{ds_name}/label_{label_value}"].append(d_native)
                sample_table.add_data(ds_name, sample_idx, label_value, d_ds, d_native)

                # ── figure (one per dataset/label) ───────────────────────────
                fig_key = (ds_name, label_value)
                if fig_key not in saved_figures:
                    saved_figures.add(fig_key)
                    gt_ds_np = downsample_mask(label, cfg.feature.output_size,
                                               cfg.feature.mask_pool).cpu().numpy()
                    ctx_gts_ds_np = [
                        downsample_mask(context_outs[b, k, 0], cfg.feature.output_size,
                                        cfg.feature.mask_pool).cpu().numpy()
                        for k in range(K)
                    ]
                    fig_path = out_dir / f"{ds_name}_l{label_value}.png"
                    save_figure(
                        tgt_image=images[b, 0].cpu().numpy(),
                        tgt_gt=label.cpu().numpy(),
                        tgt_gt_ds=gt_ds_np,
                        pred_ds=pred_ds.cpu().numpy(),
                        ctx_images=[context_ins[b, k, 0].cpu().numpy() for k in range(K)],
                        ctx_gts=  [context_outs[b, k, 0].cpu().numpy() for k in range(K)],
                        ctx_gts_ds=ctx_gts_ds_np,
                        out_path=fig_path,
                        title=f"{ds_name}  label={label_value}  sample={sample_idx}"
                              f"  dice_native={d_native:.3f}",
                    )
                    wandb.log({
                        f"figures/{ds_name}/label_{label_value}": wandb.Image(str(fig_path)),
                    })

    # ── aggregate & log ───────────────────────────────────────────────────────
    mean_enc = float(np.mean(encode_times)) if encode_times else float("nan")
    mean_pfn = float(np.mean(tabpfn_times)) if tabpfn_times else float("nan")
    print(f"\n  avg encode:  {mean_enc * 1000:.1f} ms/item")
    print(f"  avg tabpfn:  {mean_pfn * 1000:.1f} ms/item")
    print(f"  avg total:   {(mean_enc + mean_pfn) * 1000:.1f} ms/item")

    summary = log_summary(per_ds, per_label, sample_table, extra={
        "time/encode_ms": mean_enc * 1000,
        "time/tabpfn_ms": mean_pfn * 1000,
        "time/total_ms":  (mean_enc + mean_pfn) * 1000,
    })
    wandb.log(summary)
    run.finish()


if __name__ == "__main__":
    main()
