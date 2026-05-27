"""
NNInteractive encoder + multilevel PatchICLAttention on TotalSegmentator.

Identical pipeline to experiments/multilevel/train.py with two differences:

1. Encoder — NNInteractiveEncoder (pretrained, 90 M params, frozen) instead
   of STUNetEncoder.  Supports two mask injection modes:

     ch1      : context images encoded as [img, ctx_mask, 0, 0, 0, 0, 0, 0]
                (uses nnInteractive's native "current segmentation" channel).
                Target images encoded as [img, 0, 0, 0, 0, 0, 0, 0].
     separate : image in ch0 only; mask fused at bottleneck via _Mask3DEncoder.

2. Context encoding — context masks are passed to the encoder (not only to the
   attention module as labels).  This lets the encoder leverage its interactive
   segmentation pretraining to produce mask-conditioned features for context
   images while keeping target features unconditional.

Everything else — attention architecture (MultilevelICL), patch sampling,
coarse-to-fine cascade, label injection, visualisation — is unchanged.

Usage
-----
    python experiments/nninteractive/train.py
    python experiments/nninteractive/train.py model.nnint_mask_injection=separate
    python experiments/nninteractive/train.py model.nnint_num_stages=4
    python experiments/nninteractive/train.py train.run_name=debug data.max_ds_len_train=100
    python experiments/nninteractive/train.py cluster=meta
"""

import random
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.totalseg_dataloader_incontext import TotalSegInContextDataset, incontext_collate_fn
from src.models.encoders.nninteractive import NNInteractiveEncoder
from experiments.multilevel.model import MultilevelICL
from data.totalseg_classes import resolve_classes

# ---------------------------------------------------------------------------
# Re-use all stateless utilities from multilevel/train.py
# ---------------------------------------------------------------------------
from experiments.multilevel.train import (
    downsample_feat, extract_features, downsample_mask,
    _binary_entropy, _gumbel_topk,
    sample_target_patches, sample_context_patches,
    gather_patches, grid_coords_3d,
    _encode_ctx_labels,
    norm_dice_score, dice_score, soft_dice_score,
    _best_slice, _overlay, _heatmap, _patch_overlay,
    save_val_figure, _save_synth_train_figure,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config() -> OmegaConf:
    cli_overrides = [a for a in sys.argv[1:] if "=" in a]
    cli     = OmegaConf.from_dotlist(cli_overrides)
    cluster = OmegaConf.select(cli, "cluster") or "nfs"
    base    = OmegaConf.load(ROOT / "configs" / "config.yaml")
    cl_cfg  = OmegaConf.load(ROOT / "configs" / "cluster" / f"{cluster}.yaml")
    ex_cfg  = OmegaConf.load(ROOT / "configs" / "experiment" / "nninteractive.yaml")
    return OmegaConf.merge(base, cl_cfg, ex_cfg, cli)


# ---------------------------------------------------------------------------
# Encoding  (the only structural difference vs multilevel/train.py)
# ---------------------------------------------------------------------------

@torch.no_grad()
def encode_target(
    encoder: NNInteractiveEncoder,
    imgs: torch.Tensor,
) -> list[torch.Tensor]:
    """Encode target images — zero mask (no prior segmentation available).

    imgs: (B, 1, D, H, W)  →  list of feature tensors [s0, …, bottleneck]
    """
    masks = torch.zeros_like(imgs)
    return encoder(imgs, masks)


@torch.no_grad()
def encode_context(
    encoder: NNInteractiveEncoder,
    ctx_imgs: torch.Tensor,
    ctx_masks: torch.Tensor,
) -> list[torch.Tensor]:
    """Encode context images with their ground-truth masks.

    ctx_imgs  : (B*K, 1, D, H, W) — normalised CT
    ctx_masks : (B*K, 1, D, H, W) — binary GT mask at image resolution

    For ch1 mode  : mask goes into channel 1 of the 8-channel input, directly
                    conditioning encoder features on the segmentation target.
    For separate  : mask encoded by _Mask3DEncoder and fused at the bottleneck.
    """
    return encoder(ctx_imgs, ctx_masks)


# ---------------------------------------------------------------------------
# Forward pass for one batch
# ---------------------------------------------------------------------------

def process_batch(
    encoder: NNInteractiveEncoder,
    model:   MultilevelICL,
    batch:   dict,
    cfg:     OmegaConf,
    device:  torch.device,
    amp:     bool = False,
    compiled_attn = None,  # compiled PatchICLAttention (shared_weights=True only)
) -> tuple[list, torch.Tensor, list, list, list, list]:
    """Returns (preds, loss, level_losses, grid_preds, tgt_idxs, ctx_idxs)."""
    images  = batch["image"].to(device, non_blocking=True)
    labels  = batch["label"].to(device, non_blocking=True)
    ctx_in  = batch["context_in"].to(device, non_blocking=True)
    ctx_out = batch["context_out"].to(device, non_blocking=True)
    spacing = batch["spacing"].to(device, non_blocking=True)
    B, K = ctx_in.shape[:2]

    resolutions = [tuple(r) for r in cfg.data.resolutions]
    level       = cfg.model.feature_level
    num_levels  = len(encoder.skip_channels) + 1
    NP          = cfg.data.n_patches_l1
    temp        = cfg.data.sampling_temperature

    # ── Encode all images (no_grad; gradients flow only through the ICL model) ──
    ctx_imgs_flat  = ctx_in.reshape(B * K, 1, *ctx_in.shape[3:])
    # ctx_out: (B, K, D, H, W) → (B*K, 1, D, H, W) float
    ctx_masks_flat = ctx_out.reshape(B * K, *ctx_out.shape[2:]).unsqueeze(1).float()

    with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=amp):
        tgt_feats      = encode_target(encoder, images)
        ctx_feats_flat = encode_context(encoder, ctx_imgs_flat, ctx_masks_flat)

    preds, grid_preds, raw_losses, tgt_idxs, ctx_idxs = [], [], [], [], []
    C = None
    cascade_regs = None

    # Encode context masks once at finest resolution for the MaskCNN (if used)
    mask_cnn_vol = None
    if model.mask_cnn is not None:
        finest_res = resolutions[-1]
        mask_in = downsample_mask(
            ctx_out.reshape(B * K, *ctx_out.shape[2:]), finest_res, cfg.data.mask_pool
        ).unsqueeze(1)
        if not cfg.model.soft_labels_train:
            mask_in = (mask_in > 0).float()
        with torch.autocast(device_type=device.type, enabled=amp):
            emb = model.mask_cnn(mask_in)
        mask_cnn_vol = emb.transpose(1, 2).reshape(
            B * K, model.mask_cnn_dim, *finest_res
        )

    for i, res in enumerate(resolutions):
        N = res[0] * res[1] * res[2]
        tgt_feat_i  = extract_features(tgt_feats,      level, res, num_levels)
        ctx_feat_if = extract_features(ctx_feats_flat, level, res, num_levels)
        if C is None:
            C = tgt_feat_i.shape[1]
        ctx_feat_i = ctx_feat_if.reshape(B, K, C, *res)
        ctx_mask_i = downsample_mask(
            ctx_out.reshape(B * K, *ctx_out.shape[2:]), res, cfg.data.mask_pool
        ).reshape(B, K, *res)

        scale_mm = (cfg.data.image_size[0] / res[0]) * spacing.mean(dim=1)

        tgt_idx = ctx_idx = None
        if i == 0:
            # Dense forward over all N patches
            tgt_f   = tgt_feat_i.float().reshape(B, C, N).permute(0, 2, 1)
            ctx_f   = ctx_feat_i.float().permute(0, 1, 3, 4, 5, 2).reshape(B, K * N, C)
            ctx_lbl = _encode_ctx_labels(ctx_mask_i, None, cfg.model.soft_labels_train,
                                         mask_cnn_vol, res)
            gt      = downsample_mask(labels, res, cfg.data.mask_pool).reshape(B, N)

            coords   = grid_coords_3d(res, device)
            tgt_crds = coords.unsqueeze(0).expand(B, -1, -1)
            ctx_crds = coords.unsqueeze(0).expand(B, -1, -1).repeat(1, K, 1)

            with torch.autocast(device_type=device.type, enabled=amp):
                _attn = compiled_attn if compiled_attn is not None else model[i]
                if compiled_attn is not None:
                    torch.compiler.cudagraph_mark_step_begin()
                result = _attn(tgt_f, ctx_f, ctx_lbl,
                               tgt_coords=tgt_crds, ctx_coords=ctx_crds,
                               cascade_registers=cascade_regs, scale_mm=scale_mm)
            if isinstance(result, tuple):
                regs = result[1]
                cascade_regs = regs.detach() if cfg.model.detach_cascade_registers else regs
                if compiled_attn is not None:
                    cascade_regs = cascade_regs.clone()  # break CUDA graph cross-graph dependency
                pred = result[0].float()
            else:
                pred = result.float()
            grid_pred = pred

        else:
            # Sparse: Gumbel-TopK sampling guided by previous level
            prev_res = resolutions[i - 1]
            with torch.no_grad():
                prev_up = F.interpolate(
                    grid_preds[-1].detach().reshape(B, 1, *prev_res),
                    size=res, mode="trilinear", align_corners=False,
                ).reshape(B, N)

            gt_flat = downsample_mask(labels, res, cfg.data.mask_pool).reshape(B, N)
            tgt_idx = sample_target_patches(prev_up, gt_flat, NP, temp,
                                            cfg.data.target_sampling)
            ctx_idx = sample_context_patches(ctx_mask_i, NP, temp)

            tgt_flat = tgt_feat_i.float().reshape(B, C, N).permute(0, 2, 1)
            ctx_flat = ctx_feat_i.float().permute(0, 1, 3, 4, 5, 2).reshape(B, K, N, C)

            tgt_sparse = gather_patches(tgt_flat, tgt_idx)
            ctx_pieces = [gather_patches(ctx_flat[:, k], ctx_idx) for k in range(K)]
            ctx_sparse = torch.cat(ctx_pieces, dim=1)
            ctx_lbl    = _encode_ctx_labels(ctx_mask_i, ctx_idx, cfg.model.soft_labels_train,
                                            mask_cnn_vol, res)

            coords   = grid_coords_3d(res, device)
            tgt_crds = coords[tgt_idx.reshape(-1)].reshape(B, NP, 3)
            ctx_crds = coords[ctx_idx.reshape(-1)].reshape(B, NP, 3)
            ctx_crds = ctx_crds.unsqueeze(1).expand(-1, K, -1, -1).reshape(B, K * NP, 3)

            with torch.autocast(device_type=device.type, enabled=amp):
                _attn = compiled_attn if compiled_attn is not None else model[i]
                if compiled_attn is not None:
                    torch.compiler.cudagraph_mark_step_begin()
                result = _attn(tgt_sparse, ctx_sparse, ctx_lbl,
                               tgt_coords=tgt_crds, ctx_coords=ctx_crds,
                               cascade_registers=cascade_regs, scale_mm=scale_mm)
            if isinstance(result, tuple):
                regs = result[1]
                cascade_regs = regs.detach() if cfg.model.detach_cascade_registers else regs
                if compiled_attn is not None:
                    cascade_regs = cascade_regs.clone()  # break CUDA graph cross-graph dependency
                pred = result[0].float()
            else:
                pred = result.float()
            grid_pred = prev_up.clone()
            grid_pred.scatter_(1, tgt_idx, pred)
            gt = gt_flat.gather(1, tgt_idx)

        raw_losses.append(F.binary_cross_entropy(pred, gt))
        preds.append(pred)
        grid_preds.append(grid_pred)
        tgt_idxs.append(tgt_idx)
        ctx_idxs.append(ctx_idx)

    weights = list(cfg.train.loss_weights)
    while len(weights) < len(raw_losses):
        weights.append(1.0)
    loss = sum(w * l for w, l in zip(weights, raw_losses))
    return preds, loss, [l.item() for l in raw_losses], grid_preds, tgt_idxs, ctx_idxs


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(
    model:           MultilevelICL,
    encoder:         NNInteractiveEncoder,
    ds_val:          TotalSegInContextDataset,
    cfg:             OmegaConf,
    device:          torch.device,
    items_per_class: int,
    fig_dir:         Path | None = None,
    epoch:           int = 0,
    compiled_attn    = None,  # compiled PatchICLAttention (shared_weights=True only)
) -> tuple[dict, dict]:
    from collections import defaultdict
    from sklearn.metrics import roc_auc_score

    model.eval()
    amp = device.type == "cuda"
    if fig_dir is not None:
        fig_dir.mkdir(parents=True, exist_ok=True)

    resolutions = [tuple(r) for r in cfg.data.resolutions]
    n_levels    = len(resolutions)
    level       = cfg.model.feature_level
    num_levels  = len(encoder.skip_channels) + 1
    NP          = cfg.data.n_patches_l1
    temp        = cfg.data.sampling_temperature

    val_loader = DataLoader(
        ds_val,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.workers,
        collate_fn=incontext_collate_fn,
        pin_memory=True,
        persistent_workers=cfg.train.workers > 0,
        prefetch_factor=2 if cfg.train.workers > 0 else None,
        drop_last=False,
    )

    level_dices       = [[] for _ in range(n_levels)]
    level_fused_dices = [[] for _ in range(n_levels)]
    final_dices, final_soft_dices, final_norm_dices, final_aurocs, final_losses = [], [], [], [], []
    wandb_images: dict = {}
    cls_fig_saved: set = set()
    class_counts: dict = defaultdict(int)

    for batch in val_loader:
        label_names = batch["label_names"]
        images  = batch["image"].to(device, non_blocking=True)
        labels  = batch["label"].to(device, non_blocking=True)
        ctx_in  = batch["context_in"].to(device, non_blocking=True)
        ctx_out = batch["context_out"].to(device, non_blocking=True)
        spacing = batch["spacing"].to(device, non_blocking=True)
        B, K = ctx_in.shape[:2]

        ctx_imgs_flat  = ctx_in.reshape(B * K, 1, *ctx_in.shape[3:])
        ctx_masks_flat = ctx_out.reshape(B * K, *ctx_out.shape[2:]).unsqueeze(1).float()

        with torch.inference_mode():
            tgt_feats      = encode_target(encoder, images)
            ctx_feats_flat = encode_context(encoder, ctx_imgs_flat, ctx_masks_flat)

        preds_l, tgt_idxs_l, ctx_idxs_l, grid_preds_l = [], [], [], []
        C = None
        cascade_regs = None

        mask_cnn_vol = None
        if model.mask_cnn is not None:
            finest_res = resolutions[-1]
            mask_in = downsample_mask(
                ctx_out.reshape(B * K, *ctx_out.shape[2:]), finest_res, cfg.data.mask_pool
            ).unsqueeze(1)
            if not cfg.model.soft_labels_eval:
                mask_in = (mask_in > 0).float()
            with torch.autocast(device_type=device.type, enabled=amp):
                emb = model.mask_cnn(mask_in)
            mask_cnn_vol = emb.transpose(1, 2).reshape(
                B * K, model.mask_cnn_dim, *finest_res
            )

        for i, res in enumerate(resolutions):
            N = res[0] * res[1] * res[2]
            tgt_feat_i  = extract_features(tgt_feats,      level, res, num_levels)
            ctx_feat_if = extract_features(ctx_feats_flat, level, res, num_levels)
            if C is None:
                C = tgt_feat_i.shape[1]
            ctx_feat_i = ctx_feat_if.reshape(B, K, C, *res)
            ctx_mask_i = downsample_mask(
                ctx_out.reshape(B * K, *ctx_out.shape[2:]), res, cfg.data.mask_pool
            ).reshape(B, K, *res)

            scale_mm = (cfg.data.image_size[0] / res[0]) * spacing.mean(dim=1)

            if i == 0:
                tgt_f   = tgt_feat_i.float().reshape(B, C, N).permute(0, 2, 1)
                ctx_f   = ctx_feat_i.float().permute(0, 1, 3, 4, 5, 2).reshape(B, K * N, C)
                ctx_lbl = _encode_ctx_labels(ctx_mask_i, None, cfg.model.soft_labels_eval,
                                             mask_cnn_vol, res)

                coords   = grid_coords_3d(res, device)
                tgt_crds = coords.unsqueeze(0).expand(B, -1, -1)
                ctx_crds = coords.unsqueeze(0).expand(B, -1, -1).repeat(1, K, 1)

                with torch.autocast(device_type=device.type, enabled=amp):
                    _attn = compiled_attn if compiled_attn is not None else model[0]
                    if compiled_attn is not None:
                        torch.compiler.cudagraph_mark_step_begin()
                    result = _attn(tgt_f, ctx_f, ctx_lbl,
                                   tgt_coords=tgt_crds, ctx_coords=ctx_crds,
                                   cascade_registers=cascade_regs, scale_mm=scale_mm)
                if isinstance(result, tuple):
                    regs = result[1]
                    cascade_regs = regs.detach() if cfg.model.detach_cascade_registers else regs
                    if compiled_attn is not None:
                        cascade_regs = cascade_regs.clone()  # break CUDA graph cross-graph dependency
                    pred = result[0].float()
                else:
                    pred = result.float()
                tgt_idx = ctx_idx = None
                grid_pred = pred

            else:
                prev_res = resolutions[i - 1]
                prev_up  = F.interpolate(
                    grid_preds_l[-1].reshape(B, 1, *prev_res),
                    size=res, mode="trilinear", align_corners=False,
                ).reshape(B, N)

                gt_flat = downsample_mask(labels, res, cfg.data.mask_pool).reshape(B, N)
                tgt_idx = sample_target_patches(prev_up, gt_flat, NP, temp,
                                                cfg.data.target_sampling)
                ctx_idx = sample_context_patches(ctx_mask_i, NP, temp)

                tgt_flat_f = tgt_feat_i.float().reshape(B, C, N).permute(0, 2, 1)
                ctx_flat_f = ctx_feat_i.float().permute(0, 1, 3, 4, 5, 2).reshape(B, K, N, C)

                tgt_sparse = gather_patches(tgt_flat_f, tgt_idx)
                ctx_pieces = [gather_patches(ctx_flat_f[:, k], ctx_idx) for k in range(K)]
                ctx_sparse = torch.cat(ctx_pieces, dim=1)
                ctx_lbl    = _encode_ctx_labels(ctx_mask_i, ctx_idx, cfg.model.soft_labels_eval,
                                                mask_cnn_vol, res)

                coords   = grid_coords_3d(res, device)
                tgt_crds = coords[tgt_idx.reshape(-1)].reshape(B, NP, 3)
                ctx_crds = coords[ctx_idx.reshape(-1)].reshape(B, NP, 3)
                ctx_crds = ctx_crds.unsqueeze(1).expand(-1, K, -1, -1).reshape(B, K * NP, 3)

                with torch.autocast(device_type=device.type, enabled=amp):
                    _attn = compiled_attn if compiled_attn is not None else model[i]
                    if compiled_attn is not None:
                        torch.compiler.cudagraph_mark_step_begin()
                    result = _attn(tgt_sparse, ctx_sparse, ctx_lbl,
                                   tgt_coords=tgt_crds, ctx_coords=ctx_crds,
                                   cascade_registers=cascade_regs, scale_mm=scale_mm)
                if isinstance(result, tuple):
                    regs = result[1]
                    cascade_regs = regs.detach() if cfg.model.detach_cascade_registers else regs
                    if compiled_attn is not None:
                        cascade_regs = cascade_regs.clone()  # break CUDA graph cross-graph dependency
                    pred = result[0].float()
                else:
                    pred = result.float()
                grid_pred = prev_up.clone()
                grid_pred.scatter_(1, tgt_idx, pred)

            preds_l.append(pred)
            tgt_idxs_l.append(tgt_idx)
            ctx_idxs_l.append(ctx_idx)
            grid_preds_l.append(grid_pred)

        for b in range(B):
            cls = label_names[b]
            if class_counts[cls] >= items_per_class:
                continue
            label_b = labels[b:b + 1]

            for i, res in enumerate(resolutions):
                N_i      = res[0] * res[1] * res[2]
                gt_i     = downsample_mask(label_b, res, cfg.data.mask_pool).reshape(N_i)
                gt_i_bin = (gt_i > 0).float()
                if tgt_idxs_l[i] is None:
                    di = dice_score(preds_l[i][b], gt_i_bin)
                else:
                    gt_at_tgt = gt_i[tgt_idxs_l[i][b]]
                    di = dice_score(preds_l[i][b], (gt_at_tgt > 0).float())
                if di == di:
                    level_dices[i].append(di)
                df = dice_score(grid_preds_l[i][b], gt_i_bin)
                if df == df:
                    level_fused_dices[i].append(df)

            final_res = resolutions[-1]
            N_final   = final_res[0] * final_res[1] * final_res[2]
            gt_final  = downsample_mask(label_b, final_res, cfg.data.mask_pool).reshape(N_final)
            gt_final_b = (gt_final > 0).float()
            pf         = grid_preds_l[-1][b]

            final_losses.append(F.binary_cross_entropy(pf, gt_final_b).item())
            d = dice_score(pf, gt_final_b)
            if d == d:
                final_dices.append(d)
            sd = soft_dice_score(pf, gt_final_b)
            if sd == sd:
                final_soft_dices.append(sd)
            nd = norm_dice_score(pf, gt_final_b)
            if nd == nd:
                final_norm_dices.append(nd)
            pf_np = pf.cpu().numpy()
            gf_np = gt_final_b.cpu().numpy().astype(int)
            if 0 < gf_np.sum() < len(gf_np):
                try:
                    final_aurocs.append(__import__("sklearn.metrics", fromlist=["roc_auc_score"]).roc_auc_score(gf_np, pf_np))
                except Exception:
                    pass

            if cls not in cls_fig_saved and fig_dir is not None:
                tgt_np    = batch["image"][b].squeeze().cpu().numpy()
                tgt_gt_np = batch["label"][b].cpu().numpy()
                ctx_np    = batch["context_in"][b, 0].squeeze(0).cpu().numpy()
                ctx_gt_np = batch["context_out"][b, 0].cpu().numpy()
                levels_data = []
                for i, res in enumerate(resolutions):
                    D_, H_, W_ = res
                    N_i        = D_ * H_ * W_
                    gt_ds_np   = downsample_mask(label_b, res, cfg.data.mask_pool).squeeze().cpu().numpy().reshape(D_, H_, W_)
                    fused_np   = grid_preds_l[i][b].cpu().numpy().reshape(D_, H_, W_)
                    if tgt_idxs_l[i] is None:
                        pred_np = preds_l[i][b].cpu().numpy().reshape(D_, H_, W_)
                    else:
                        sparse = np.zeros(N_i, dtype=np.float32)
                        sparse[tgt_idxs_l[i][b].cpu().numpy()] = preds_l[i][b].cpu().numpy()
                        pred_np = sparse.reshape(D_, H_, W_)
                    levels_data.append({
                        "res": res, "gt_ds": gt_ds_np, "pred": pred_np,
                        "pred_fused": fused_np,
                        "tgt_idx": tgt_idxs_l[i][b].cpu().numpy() if tgt_idxs_l[i] is not None else None,
                        "ctx_idx": ctx_idxs_l[i][b].cpu().numpy() if ctx_idxs_l[i] is not None else None,
                    })
                title_parts = [f"[ep {epoch}] {cls}"]
                for i in range(n_levels):
                    v = level_fused_dices[i][-1] if level_fused_dices[i] else float("nan")
                    title_parts.append(f"fused_l{i}={v:.3f}")
                fig_path = fig_dir / f"epoch{epoch:03d}_{cls}.png"
                save_val_figure(
                    tgt_image=tgt_np, tgt_gt=tgt_gt_np,
                    ctx_image=ctx_np, ctx_gt=ctx_gt_np,
                    levels=levels_data, out_path=fig_path,
                    title="  ".join(title_parts),
                )
                wandb_images[f"val/pred/{cls}"] = fig_path
                cls_fig_saved.add(cls)

            class_counts[cls] += 1

    model.train()
    metrics: dict = {}
    for i in range(n_levels):
        metrics[f"val/dice_l{i}"]       = float(np.nanmean(level_dices[i]))       if level_dices[i]       else float("nan")
        metrics[f"val/dice_fused_l{i}"] = float(np.nanmean(level_fused_dices[i])) if level_fused_dices[i] else float("nan")
    metrics["val/dice"]      = float(np.nanmean(final_dices))      if final_dices      else float("nan")
    metrics["val/soft_dice"] = float(np.nanmean(final_soft_dices)) if final_soft_dices else float("nan")
    metrics["val/norm_dice"] = float(np.nanmean(final_norm_dices)) if final_norm_dices else float("nan")
    metrics["val/auroc"]     = float(np.nanmean(final_aurocs))     if final_aurocs     else float("nan")
    metrics["val/loss"]      = float(np.mean(final_losses))        if final_losses     else float("nan")
    return metrics, wandb_images


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    cfg = load_config()
    torch.manual_seed(cfg.train.seed)

    device_str = cfg.train.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    out_dir = Path(cfg.paths.results) / "nninteractive"
    out_dir.mkdir(parents=True, exist_ok=True)

    aug_cfg = None
    if cfg.train.aug:
        aug_yaml = ROOT / "configs" / "augmentations" / f"{cfg.train.aug_preset}.yaml"
        aug_cfg  = OmegaConf.load(aug_yaml).augmentations

    data_root = cfg.paths.totalsegmri if getattr(cfg.data, "dataset", "totalseg") == "totalsegmri" \
                else cfg.paths.totalseg
    is_mri    = getattr(cfg.data, "dataset", "totalseg") == "totalsegmri"

    train_classes = resolve_classes(cfg.data.train_classes, data_root, is_mri=is_mri)
    val_classes   = resolve_classes(cfg.data.val_classes, data_root, is_mri=is_mri) if cfg.data.val_classes else []
    val_classes   = val_classes or train_classes

    ds_train = TotalSegInContextDataset(
        root=data_root,
        classes=train_classes,
        image_size=tuple(cfg.data.image_size),
        split="train",
        context_size=cfg.data.context_size,
        class_balanced=cfg.data.class_balanced,
        aug_cfg=aug_cfg,
        use_crop=cfg.data.use_crop,
        synth_method=cfg.data.synth_method or None,
        n_synth_merge_min=cfg.data.n_synth_merge_min,
        n_synth_merge_max=cfg.data.n_synth_merge_max,
        p_synth=cfg.data.p_synth,
        random_coloring=False,
        num_labels_per_sample=cfg.data.num_labels_per_sample,
    )
    ds_val = TotalSegInContextDataset(
        root=data_root,
        classes=val_classes,
        image_size=tuple(cfg.data.image_size),
        split="val",
        context_size=cfg.data.context_size,
        use_crop=cfg.data.use_crop,
    )

    n_train = min(cfg.data.max_ds_len_train, len(ds_train))
    train_loader = DataLoader(
        ds_train,
        batch_size=cfg.train.batch_size,
        sampler=RandomSampler(ds_train, replacement=False, num_samples=n_train),
        num_workers=cfg.train.workers,
        pin_memory=True,
        persistent_workers=cfg.train.workers > 0,
        prefetch_factor=2 if cfg.train.workers > 0 else None,
        collate_fn=incontext_collate_fn,
        drop_last=True,
    )
    print(f"Train: {n_train} samples  |  {len(train_loader)} batches/epoch")

    # ── NNInteractive encoder (frozen) ──────────────────────────────────────
    ckpt_dir        = str(cfg.model.nnint_ckpt)
    mask_injection  = str(cfg.model.nnint_mask_injection)
    num_stages      = int(cfg.model.nnint_num_stages)

    encoder = NNInteractiveEncoder(
        ckpt_dir=ckpt_dir,
        mask_injection=mask_injection,
        freeze_encoder=True,
        num_stages=num_stages,
        device="cpu",
    ).to(device).eval()

    total_enc_params    = sum(p.numel() for p in encoder.parameters()) / 1e6
    trainable_enc_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad) / 1e6
    print(f"NNInteractiveEncoder  mask={mask_injection}  stages={num_stages}  "
          f"{total_enc_params:.1f} M total / {trainable_enc_params:.2f} M trainable")

    num_levels = len(encoder.skip_channels) + 1
    level      = cfg.model.feature_level

    # ── Determine embed_dim from dummy forward ───────────────────────────────
    with torch.inference_mode():
        dummy_img  = torch.zeros(1, 1, *cfg.data.image_size, device=device)
        dummy_msk  = torch.zeros(1, 1, *cfg.data.image_size, device=device)
        dummy_feat = encoder(dummy_img, dummy_msk)
        dummy_ds   = extract_features(dummy_feat, level, tuple(cfg.data.resolutions[0]), num_levels)
    embed_dim = dummy_ds.shape[1]
    print(f"embed_dim={embed_dim}  |  level={level}  |  resolutions={list(cfg.data.resolutions)}")

    # ── MultilevelICL (same as multilevel/train.py) ──────────────────────────
    resolutions = [tuple(r) for r in cfg.data.resolutions]
    level_cfgs  = [
        {
            "grid_size":       res,
            "dim":             cfg.model.dim,
            "num_heads":       cfg.model.num_heads,
            "num_layers":      cfg.model.num_layers,
            "ff_factor":       cfg.model.ff_factor,
            "label_injection": cfg.model.label_injection,
            "output_head":     cfg.model.output_head,
            "pos_encoding":    cfg.model.pos_encoding
                               if (i == 0 or cfg.model.pos_encoding == "rope3d")
                               else "none",
            "input_norm":      cfg.model.input_norm,
            "dropout":         cfg.model.dropout,
            "ctx_self_attn":   cfg.model.ctx_self_attn,
            "log_n_scaling":   cfg.model.log_n_scaling,
            "log_n_base":      cfg.model.log_n_base,
            "soft_labels":     cfg.model.soft_labels_train,
        }
        for i, res in enumerate(resolutions)
    ]

    model = MultilevelICL(
        embed_dim=embed_dim,
        level_cfgs=level_cfgs,
        mask_cnn_dim=int(getattr(cfg.model, "mask_cnn_dim", 0) or 0),
        num_registers=int(getattr(cfg.model, "num_registers", 0) or 0),
        append_zero_attn=bool(getattr(cfg.model, "append_zero_attn", False)),
        shared_weights=bool(getattr(cfg.model, "shared_weights", False)),
        use_scale_embed=bool(getattr(cfg.model, "use_scale_embed", False)),
        use_role_embed=bool(getattr(cfg.model, "use_role_embed", False)),
        max_context_size=int(getattr(cfg.model, "max_context_size", 8)),
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"MultilevelICL  params: {n_params:,}  ({len(model)} levels)")

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    scaler    = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    amp       = device.type == "cuda"

    if cfg.train.checkpoint:
        ckpt = torch.load(cfg.train.checkpoint, map_location=device)
        missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
        if missing:
            print(f"  New parameters (random init): {missing}")
        print(f"Loaded checkpoint: {cfg.train.checkpoint}  (epoch {ckpt['epoch']})")

    # ── Compile attention module (optional; reduce-overhead ≈ 6× speedup) ────
    compiled_attn = None
    if getattr(cfg.train, "compile_model", False) and device.type == "cuda":
        if model.shared_weights:
            print("torch.compile (reduce-overhead) shared attention module ...", flush=True)
            compiled_attn = torch.compile(model.shared_level, mode="reduce-overhead")
        else:
            print("Note: compile_model=True requires shared_weights=True — skipping.")

    # ── W&B ─────────────────────────────────────────────────────────────────
    use_wandb = bool(cfg.train.wandb_project) and str(cfg.train.wandb_project).lower() != "null"
    if use_wandb:
        import wandb
        wandb.init(
            project=cfg.train.wandb_project,
            name=cfg.train.run_name or None,
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        wandb.config.update({
            "embed_dim": embed_dim, "n_params": n_params,
            "nnint_mask_injection": mask_injection,
            "nnint_num_stages": num_stages,
        })

    date_str = datetime.today().strftime("%Y-%m-%d")
    run_name  = (wandb.run.name if use_wandb else None) or cfg.train.run_name or "run"
    run_dir   = out_dir / f"{date_str}_{run_name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    fig_dir   = run_dir / "figures"

    # ── Training ─────────────────────────────────────────────────────────────
    best_dice   = -1.0
    nd_interval = cfg.train.nd_interval
    n_levels    = len(resolutions)

    for epoch in range(1, cfg.train.epochs + 1):
        model.train()
        epoch_loss         = 0.0
        epoch_level_losses = [0.0] * n_levels
        epoch_nd, epoch_dice = 0.0, 0.0
        n_batches, n_nd, n_dice = 0, 0, 0
        last_nd, last_dice = float("nan"), float("nan")
        t0 = time.perf_counter()

        synth_train_fig_path = None
        real_train_fig_path  = None
        bar = tqdm(train_loader, desc=f"Epoch {epoch:3d}/{cfg.train.epochs}", unit="batch", leave=False)

        for batch in bar:
            preds, loss, level_losses, grid_preds_b, tgt_idxs_b, ctx_idxs_b = process_batch(
                encoder, model, batch, cfg, device, amp=amp, compiled_attn=compiled_attn,
            )
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            n_batches  += 1
            epoch_loss += loss.item()
            for j, lv in enumerate(level_losses):
                epoch_level_losses[j] += lv

            if n_batches % nd_interval == 0:
                with torch.no_grad():
                    res_0 = tuple(cfg.data.resolutions[0])
                    N0    = res_0[0] * res_0[1] * res_0[2]
                    gt_0  = downsample_mask(
                        batch["label"].to(device), res_0, cfg.data.mask_pool
                    ).reshape(preds[0].shape[0], N0)
                    gt_bin = (gt_0 > 0).float()
                    nd = norm_dice_score(preds[0].detach(), gt_bin)
                    dc = dice_score(preds[0].detach(), gt_bin)
                if nd == nd:
                    epoch_nd   += nd; n_nd   += 1; last_nd   = nd
                if dc == dc:
                    epoch_dice += dc; n_dice += 1; last_dice = dc

            if fig_dir is not None:
                for b_idx, lname in enumerate(batch["label_names"]):
                    if synth_train_fig_path is None and lname.startswith("sv_"):
                        synth_train_fig_path = _save_synth_train_figure(
                            batch, preds, grid_preds_b, tgt_idxs_b, ctx_idxs_b,
                            b_idx, epoch, fig_dir / "train_synth", resolutions, cfg,
                        )
                    elif real_train_fig_path is None and not lname.startswith("sv_"):
                        real_train_fig_path = _save_synth_train_figure(
                            batch, preds, grid_preds_b, tgt_idxs_b, ctx_idxs_b,
                            b_idx, epoch, fig_dir / "train_real", resolutions, cfg,
                        )
                    if synth_train_fig_path is not None and real_train_fig_path is not None:
                        break

            bar.set_postfix(
                loss=f"{epoch_loss / n_batches:.4f}",
                **{f"l{j}": f"{epoch_level_losses[j] / n_batches:.4f}" for j in range(n_levels)},
                dice=f"{last_dice:.3f}",
                nd=f"{last_nd:.3f}",
            )

        bar.close()
        elapsed  = time.perf_counter() - t0
        avg_loss = epoch_loss / max(n_batches, 1)
        level_str = "  ".join(
            f"l{j}={epoch_level_losses[j] / max(n_batches, 1):.4f}" for j in range(n_levels)
        )
        print(f"Epoch {epoch:3d}/{cfg.train.epochs}  loss={avg_loss:.4f}  {level_str}  "
              f"dice_l0={epoch_dice / max(n_dice, 1):.3f}  {elapsed:.0f}s")

        val_metrics, val_figs = validate(
            model, encoder, ds_val, cfg, device,
            cfg.train.val_items_per_class,
            fig_dir=fig_dir, epoch=epoch, compiled_attn=compiled_attn,
        )
        dice_str = "  ".join(
            f"{k.split('/')[-1]}={v:.3f}"
            for k, v in val_metrics.items()
            if "dice" in k or k == "val/auroc"
        )
        print(f"  val {dice_str}")

        best_key = f"val/dice_fused_l{n_levels - 1}"
        if val_metrics[best_key] > best_dice:
            best_dice = val_metrics[best_key]
            torch.save({
                "epoch":    epoch,
                "model":    model.state_dict(),
                "config":   OmegaConf.to_container(cfg, resolve=True),
                "val_dice": best_dice,
            }, run_dir / "best.pt")
            print(f"  saved best  {best_key.split('/')[-1]}={best_dice:.3f}")

        if use_wandb:
            import wandb
            all_figs = {k: wandb.Image(str(v)) for k, v in val_figs.items()}
            if synth_train_fig_path is not None:
                all_figs["train/pred_synth"] = wandb.Image(str(synth_train_fig_path))
            if real_train_fig_path is not None:
                all_figs["train/pred_real"] = wandb.Image(str(real_train_fig_path))
            wandb.log({
                "train/loss": avg_loss,
                **{f"train/loss_l{j}": epoch_level_losses[j] / max(n_batches, 1) for j in range(n_levels)},
                "train/dice_l0": epoch_dice / max(n_dice, 1),
                "epoch": epoch, **val_metrics, **all_figs,
            })

    if use_wandb:
        import wandb
        wandb.finish()
    print(f"\nBest val dice_fused_l1: {best_dice:.3f}  |  checkpoint: {run_dir}/best.pt")


if __name__ == "__main__":
    main()
