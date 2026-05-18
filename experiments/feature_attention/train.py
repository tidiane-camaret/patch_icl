"""
Train PatchICLAttention on TotalSegmentator train split.

The STU-Net encoder is frozen; only the attention module is trained.
Features are extracted on-the-fly for each batch via a DataLoader with
multiple workers, enabling true batching through both the encoder and the
attention model.

Usage
-----
    python experiments/feature_attention/train.py
    python experiments/feature_attention/train.py --num_layers 4 --label_injection additive
    python experiments/feature_attention/train.py --output_head retrieval --pos_encoding sinusoidal
"""

import argparse
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.totalseg_dataloader_incontext import TotalSegInContextDataset, incontext_collate_fn
from src.models.encoders.stunet import STUNetEncoder
from experiments.feature_attention.model import PatchICLAttention

TOTALSEG_ROOT = (
    "/nfs/data/nii/data1/Analysis/camaret___in_context_segmentation"
    "/ANALYSIS_20251122/data/totalseg"
)
DEFAULT_PRETRAINED = (
    "/nfs/data/nii/data1/Analysis/camaret___in_context_segmentation"
    "/ANALYSIS_20251122/results/patch_icl/checkpoints/stunet/base_statedict.pt"
)


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    _default_classes_train = [
        "liver", "lung_lower_lobe_right", "lung_lower_lobe_left",
        "lung_upper_lobe_right", "lung_upper_lobe_left", "lung_middle_lobe_right",
        "small_bowel", "heart", "autochthon_left", "autochthon_right",
        "hip_left", "hip_right", "aorta", "brain", "skull",
    ]
    _default_classes_val = [
    "colon",
    "stomach",
    "spleen",
    "gluteus_maximus_right",
    "gluteus_maximus_left",
    "gluteus_medius_left",
    "gluteus_medius_right",
    "iliopsoas_left",
    "iliopsoas_right",
    "costal_cartilages",
    "urinary_bladder",
    "femur_left",
    "femur_right",
    "sacrum",
    "kidney_left"
]
    # Data
    p.add_argument("--data_root",      default=TOTALSEG_ROOT)
    p.add_argument("--train_classes",  nargs="+", default=_default_classes_train)
    p.add_argument("--val_classes",    nargs="+", default=_default_classes_val)
    p.add_argument("--context_size",   type=int, default=1)
    p.add_argument("--image_size",     type=int, nargs=3, default=[128, 128, 128])
    p.add_argument("--output_size",    type=int, nargs=3, default=[8, 8, 8])
    p.add_argument("--feature_level",  type=str, default="all",
                   help="Encoder level: int index or 'all'.")
    p.add_argument("--mask_pool",      default="max", choices=["max", "avg"])
    # Encoder
    p.add_argument("--stunet_variant",    default="base",
                   choices=["small", "base", "large", "huge"])
    p.add_argument("--stunet_pretrained", default=DEFAULT_PRETRAINED)
    # Model architecture
    p.add_argument("--num_heads",       type=int, default=8)
    p.add_argument("--num_layers",      type=int, default=4)
    p.add_argument("--ff_factor",       type=int, default=2)
    p.add_argument("--label_injection", default="additive",
                   choices=["additive", "concat", "gate", "none"])
    p.add_argument("--output_head",     default="linear",
                   choices=["linear", "mlp", "retrieval"])
    p.add_argument("--pos_encoding",    default="sinusoidal",
                   choices=["none", "sinusoidal", "learned"])
    p.add_argument("--input_norm",      default="rmsnorm",
                   choices=["none", "rmsnorm", "l2"])
    p.add_argument("--dropout",         type=float, default=0.0)
    p.add_argument("--no_ctx_self_attn", action="store_true",
                   help="Disable context self-attention (enabled by default).")
    p.add_argument("--no_log_n_scaling", action="store_true",
                   help="Disable log-n query scaling (enabled by default).")
    p.add_argument("--log_n_base",      type=int, default=512,
                   help="Reference context size for log-n scaling (default: 1×8³=512).")
    # Training
    p.add_argument("--epochs",              type=int,   default=20)
    p.add_argument("--batch_size",          type=int,   default=8)
    p.add_argument("--lr",                  type=float, default=1e-4)
    p.add_argument("--weight_decay",        type=float, default=1e-5)
    p.add_argument("--workers",             type=int,   default=14,
                   help="DataLoader num_workers.")
    p.add_argument("--max_ds_len_train",    type=int,   default=2000,
                   help="Max samples per epoch (reshuffled each epoch).")
    p.add_argument("--val_items_per_class", type=int,   default=5)
    p.add_argument("--nd_interval",         type=int,   default=50,
                   help="Compute train norm_dice every N batches (reduces GPU syncs).")
    p.add_argument("--device",  default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed",    type=int, default=42)
    # Augmentation
    p.add_argument("--no_aug", action="store_true",
                   help="Disable training augmentation (default: enabled).")
    p.add_argument("--aug_preset", default="nnunet",
                   choices=["multiverseg", "nnunet"],
                   help="Augmentation config to load from configs/augmentations/.")
    # Checkpoint
    p.add_argument("--checkpoint", default=None,
                   help="Path to a saved checkpoint (.pt) to load weights from before training.")
    # Output
    p.add_argument("--out_dir",       default="experiments/feature_attention/checkpoints")
    p.add_argument("--wandb_project", default="patch_icl_3d_exps",
                   help="Set to 'null' to disable W&B.")
    p.add_argument("--run_name",      default=None)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Feature extraction (batched)
# ---------------------------------------------------------------------------

@torch.no_grad()
def encode_image_only(encoder: STUNetEncoder, imgs: torch.Tensor) -> list[torch.Tensor]:
    """imgs: (B, 1, D, H, W) → list of (B, C_i, d_i, h_i, w_i)."""
    bottleneck, skips = encoder.image_encoder(imgs, num_stages=encoder._num_stages)
    return skips + [bottleneck]


def downsample_feat(feat: torch.Tensor, size: tuple) -> torch.Tensor:
    """(B, C, d, h, w) → (B, C, D', H', W')."""
    return F.interpolate(feat, size=size, mode="trilinear", align_corners=False)


def extract_features(feats: list[torch.Tensor], level: str, out_size: tuple, num_levels: int) -> torch.Tensor:
    """Returns (B, C, D', H', W')."""
    if level == "all":
        return torch.cat([downsample_feat(f, out_size) for f in feats], dim=1)
    return downsample_feat(feats[int(level) % num_levels], out_size)


def downsample_mask(mask: torch.Tensor, size: tuple, mode: str = "max") -> torch.Tensor:
    """mask: (B, D, H, W) → (B, D', H', W')."""
    x = mask.float().unsqueeze(1)   # (B, 1, D, H, W)
    if mode == "max":
        return F.adaptive_max_pool3d(x, output_size=size).squeeze(1)
    return F.adaptive_avg_pool3d(x, output_size=size).squeeze(1)


def norm_dice_score(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """Min-max normalise pred then compute soft dice. pred, gt: arbitrary shape."""
    p = pred - pred.min()
    pmax = p.max()
    if pmax < 1e-8:
        return float("nan")
    p = p / pmax
    return (2 * (p * gt).sum() / (p.sum() + gt.sum() + 1e-6)).item()


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

def process_batch(
    encoder:    STUNetEncoder,
    model:      PatchICLAttention,
    batch:      dict,
    level:      str,
    out_size:   tuple,
    num_levels: int,
    mask_pool:  str,
    device:     torch.device,
    amp:        bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (pred, gt) both shaped (B, N) on device."""
    images  = batch["image"].to(device, non_blocking=True)        # (B, 1, D, H, W)
    labels  = batch["label"].to(device, non_blocking=True)        # (B, D, H, W)
    ctx_in  = batch["context_in"].to(device, non_blocking=True)   # (B, K, 1, D, H, W)
    ctx_out = batch["context_out"].to(device, non_blocking=True)  # (B, K, D, H, W)
    B, K = ctx_in.shape[:2]

    # Encoder is frozen — use inference_mode + autocast for maximum throughput
    with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=amp):
        tgt_feats = encode_image_only(encoder, images)
        ctx_imgs_flat  = ctx_in.reshape(B * K, 1, *ctx_in.shape[3:])
        ctx_feats_flat = encode_image_only(encoder, ctx_imgs_flat)

    # (B, C, D', H', W')
    tgt_feat_ds      = extract_features(tgt_feats,      level, out_size, num_levels)
    ctx_feat_ds_flat = extract_features(ctx_feats_flat, level, out_size, num_levels)
    C = ctx_feat_ds_flat.shape[1]
    D_, H_, W_ = out_size
    N = D_ * H_ * W_
    ctx_feat_ds = ctx_feat_ds_flat.reshape(B, K, C, D_, H_, W_)

    tgt_mask_ds = downsample_mask(labels, out_size, mask_pool)
    ctx_mask_ds = downsample_mask(
        ctx_out.reshape(B * K, *ctx_out.shape[2:]), out_size, mask_pool
    ).reshape(B, K, D_, H_, W_)

    tgt_feat = tgt_feat_ds.float().reshape(B, C, N).permute(0, 2, 1)              # (B, N, C)
    ctx_feat = ctx_feat_ds.float().permute(0, 1, 3, 4, 5, 2).reshape(B, K*N, C)  # (B, K*N, C)
    ctx_lbls = (ctx_mask_ds.reshape(B, K * N) > 0).float()
    gt       = (tgt_mask_ds.reshape(B, N) > 0).float()

    with torch.autocast(device_type=device.type, enabled=amp):
        pred = model(tgt_feat, ctx_feat, ctx_lbls)   # (B, N)
    return pred.float(), gt


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def _best_slice(mask: np.ndarray) -> int:
    counts = mask.sum(axis=(1, 2))
    return int(counts.argmax()) if counts.max() > 0 else mask.shape[0] // 2


def _overlay(ax, image: np.ndarray, mask: np.ndarray, idx: int, title: str) -> None:
    sl = image[idx]
    sl_norm = (sl - sl.min()) / (sl.max() - sl.min() + 1e-6)
    ax.imshow(sl_norm, cmap="gray")
    ax.imshow(mask[idx], cmap="Reds", alpha=0.45, vmin=0, vmax=1)
    ax.set_title(title, fontsize=8)
    ax.axis("off")


def _heatmap(ax, vol: np.ndarray, idx: int, title: str) -> None:
    ax.imshow(vol[idx], cmap="hot", vmin=0, vmax=1, interpolation="nearest")
    ax.set_title(title, fontsize=8)
    ax.axis("off")


def save_val_figure(
    tgt_image:   np.ndarray,          # (D, H, W)
    tgt_gt:      np.ndarray,          # (D, H, W)  full-res
    tgt_gt_ds:   np.ndarray,          # (D', H', W')
    pred:        np.ndarray,          # (D', H', W')
    ctx_images:  list[np.ndarray],    # K × (D, H, W)
    ctx_gts:     list[np.ndarray],    # K × (D, H, W)
    ctx_gts_ds:  list[np.ndarray],    # K × (D', H', W')
    out_path:    Path,
    title:       str = "",
) -> None:
    K = len(ctx_images)
    ncols = max(3, 2 + K)
    fig, axes = plt.subplots(2, ncols, figsize=(3.2 * ncols, 6.5))

    tgt_z    = _best_slice(tgt_gt)
    tgt_z_ds = tgt_z * tgt_gt_ds.shape[0] // tgt_gt.shape[0]

    _overlay(axes[0, 0], tgt_image, tgt_gt,    tgt_z,    "Target + GT")
    _heatmap(axes[0, 1], tgt_gt_ds,            tgt_z_ds, f"GT ↓")
    _heatmap(axes[0, 2], pred,                 tgt_z_ds, "Prediction")
    for col in range(3, ncols):
        axes[0, col].set_visible(False)

    for k in range(K):
        ctx_z    = _best_slice(ctx_gts[k])
        ctx_z_ds = ctx_z * ctx_gts_ds[k].shape[0] // ctx_gts[k].shape[0]
        if 2 * k < ncols:
            _overlay(axes[1, 2 * k],     ctx_images[k], ctx_gts[k],    ctx_z,    f"Ctx {k} + GT")
        if 2 * k + 1 < ncols:
            _heatmap(axes[1, 2 * k + 1], ctx_gts_ds[k],               ctx_z_ds, f"Ctx {k} GT ↓")
    for col in range(2 * K, ncols):
        axes[1, col].set_visible(False)

    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(
    model:       PatchICLAttention,
    encoder:     STUNetEncoder,
    ds_val:      TotalSegInContextDataset,
    level:       str,
    out_size:    tuple,
    num_levels:  int,
    mask_pool:   str,
    device:      torch.device,
    items_per_class: int,
    fig_dir:     Path | None = None,
    epoch:       int = 0,
) -> tuple[dict, dict]:
    """Returns (metrics, wandb_images) where wandb_images maps key → wandb.Image."""
    from collections import defaultdict
    from sklearn.metrics import roc_auc_score

    model.eval()
    if fig_dir is not None:
        fig_dir.mkdir(parents=True, exist_ok=True)

    cls_to_indices: dict[str, list[int]] = defaultdict(list)
    for i, (_, cls) in enumerate(ds_val.samples):
        cls_to_indices[cls].append(i)

    aurocs, norm_dices = [], []
    wandb_images: dict = {}

    for cls in ds_val.classes:
        collected = 0
        cls_fig_saved = False
        for idx in cls_to_indices[cls]:
            if collected >= items_per_class:
                break
            try:
                item = ds_val[idx]
            except Exception:
                continue

            image   = item["image"].unsqueeze(0).to(device)
            label   = item["label"].unsqueeze(0).to(device)
            ctx_in  = item["context_in"].unsqueeze(0).to(device)
            ctx_out = item["context_out"].unsqueeze(0).to(device)
            K = ctx_in.shape[1]

            tgt_feats      = encode_image_only(encoder, image)
            ctx_imgs_flat  = ctx_in.reshape(K, 1, *ctx_in.shape[3:])
            ctx_feats_flat = encode_image_only(encoder, ctx_imgs_flat)

            tgt_feat_ds      = extract_features(tgt_feats,      level, out_size, num_levels)
            ctx_feat_ds_flat = extract_features(ctx_feats_flat, level, out_size, num_levels)
            C        = ctx_feat_ds_flat.shape[1]
            D_, H_, W_ = out_size
            N        = D_ * H_ * W_
            ctx_feat_ds = ctx_feat_ds_flat.reshape(1, K, C, D_, H_, W_)

            tgt_mask_ds = downsample_mask(label, out_size, mask_pool)
            ctx_mask_ds = downsample_mask(
                ctx_out.reshape(K, *ctx_out.shape[2:]), out_size, mask_pool
            ).reshape(1, K, D_, H_, W_)

            tgt_feat = tgt_feat_ds.reshape(1, C, N).permute(0, 2, 1)
            ctx_feat = ctx_feat_ds.permute(0, 1, 3, 4, 5, 2).reshape(1, K * N, C)
            ctx_lbls = (ctx_mask_ds.reshape(1, K * N) > 0).float()
            gt       = (tgt_mask_ds.reshape(1, N) > 0).float()

            pred = model(tgt_feat, ctx_feat, ctx_lbls).squeeze(0)   # (N,)
            gt   = gt.squeeze(0)                                     # (N,)

            pred_np = pred.cpu().numpy()
            gt_np   = (gt.cpu().numpy() > 0).astype(int)
            if 0 < gt_np.sum() < len(gt_np):
                aurocs.append(roc_auc_score(gt_np, pred_np))

            p = pred - pred.min()
            pmax = p.max()
            nd = float("nan")
            if pmax > 1e-8:
                p = p / pmax
                nd = (2 * (p * gt).sum() / (p.sum() + gt.sum() + 1e-6)).item()
                norm_dices.append(nd)

            # Save one figure per class (first valid item)
            if not cls_fig_saved and fig_dir is not None:
                pred_vol = pred.cpu().numpy().reshape(D_, H_, W_)
                fig_path = fig_dir / f"epoch{epoch:03d}_{cls}.png"
                title = f"[ep {epoch}] {cls}  norm_dice={nd:.3f}"
                save_val_figure(
                    tgt_image  = item["image"].squeeze().cpu().numpy(),
                    tgt_gt     = item["label"].cpu().numpy(),
                    tgt_gt_ds  = tgt_mask_ds.squeeze().cpu().numpy(),
                    pred       = pred_vol,
                    ctx_images = [ctx_in[0, k].squeeze(0).cpu().numpy() for k in range(K)],
                    ctx_gts    = [ctx_out[0, k].cpu().numpy() for k in range(K)],
                    ctx_gts_ds = [ctx_mask_ds[0, k].cpu().numpy() for k in range(K)],
                    out_path   = fig_path,
                    title      = title,
                )
                wandb_images[f"val/pred/{cls}"] = fig_path
                cls_fig_saved = True

            collected += 1

    model.train()
    metrics = {
        "val/auroc":     float(np.nanmean(aurocs))     if aurocs     else float("nan"),
        "val/norm_dice": float(np.nanmean(norm_dices)) if norm_dices else float("nan"),
    }
    return metrics, wandb_images


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")  # enable TF32 on Ampere+ GPUs

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_size = tuple(args.output_size)

    # ---- Augmentation config -----------------------------------------------
    aug_cfg = None
    if not args.no_aug:
        aug_yaml = ROOT / "configs" / "augmentations" / f"{args.aug_preset}.yaml"
        aug_cfg  = OmegaConf.load(aug_yaml).augmentations   # has .enabled, .task, .intensity, .synth

    train_classes = args.train_classes
    val_classes   = args.val_classes or args.train_classes

    # ---- Datasets ----------------------------------------------------------
    ds_train = TotalSegInContextDataset(
        root=args.data_root,
        classes=train_classes,
        image_size=tuple(args.image_size),
        split="train",
        context_size=args.context_size,
        max_subjects=None,
        class_balanced=True,
        aug_cfg=aug_cfg,
        use_crop=True,
    )
    ds_val = TotalSegInContextDataset(
        root=args.data_root,
        classes=val_classes,
        image_size=tuple(args.image_size),
        split="val",
        context_size=args.context_size,
        use_crop=True
    )

    from torch.utils.data import RandomSampler
    n_train = min(args.max_ds_len_train, len(ds_train))
    train_sampler = RandomSampler(ds_train, replacement=False, num_samples=n_train)
    train_loader = DataLoader(
        ds_train,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
        prefetch_factor=2 if args.workers > 0 else None,
        collate_fn=incontext_collate_fn,
        drop_last=True,
    )
    print(f"Train: {n_train} samples  |  {len(train_loader)} batches/epoch  |  batch_size={args.batch_size}")

    # ---- Encoder (frozen) -------------------------------------------------
    encoder = STUNetEncoder(
        in_channels=1,
        variant=args.stunet_variant,
        pretrained=args.stunet_pretrained,
        freeze_encoder=True,
    ).to(device).eval()

    num_levels = len(encoder.skip_channels) + 1
    level = args.feature_level

    # Determine embed_dim from a dummy forward (before compiling)
    with torch.inference_mode():
        dummy = torch.zeros(1, 1, *args.image_size, device=device)
        dummy_feats = encode_image_only(encoder, dummy)
        dummy_feat_ds = extract_features(dummy_feats, level, out_size, num_levels)
    embed_dim = dummy_feat_ds.shape[1]
    print(f"Encoder embed_dim: {embed_dim}  |  grid: {out_size}  |  level: {level}")

    # Compile image encoder (frozen, inference-only path)
    print("Compiling encoder...", flush=True)
    encoder.image_encoder = torch.compile(encoder.image_encoder)

    # ---- Model -------------------------------------------------------------
    model = PatchICLAttention(
        embed_dim       = embed_dim,
        num_heads       = args.num_heads,
        num_layers      = args.num_layers,
        ff_factor       = args.ff_factor,
        label_injection = args.label_injection,
        output_head     = args.output_head,
        pos_encoding    = args.pos_encoding,
        input_norm      = args.input_norm,
        grid_size       = out_size,
        dropout         = args.dropout,
        ctx_self_attn   = not args.no_ctx_self_attn,
        log_n_scaling   = not args.no_log_n_scaling,
        log_n_base      = args.log_n_base,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"PatchICLAttention  params: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler    = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    amp       = device.type == "cuda"

    best_auroc = -1.0
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model"])
        best_auroc = ckpt.get("val_auroc", -1.0)
        print(f"Loaded checkpoint: {args.checkpoint}  "
              f"(epoch {ckpt['epoch']}, val_auroc={best_auroc:.3f})")

    # Compile attention model (keep uncompiled reference for checkpointing)
    print("Compiling model...", flush=True)
    model_module = model
    model = torch.compile(model)

    # ---- W&B ---------------------------------------------------------------
    use_wandb = args.wandb_project and args.wandb_project.lower() != "null"
    if use_wandb:
        import wandb
        wandb.init(
            project=args.wandb_project,
            name=args.run_name,
            config={k: v for k, v in vars(args).items()
                    if k not in ("wandb_project", "run_name")},
        )
        wandb.config.update({"embed_dim": embed_dim, "n_params": n_params,
                             "aug": not args.no_aug})

    fig_dir = out_dir / "figures"

    # ---- Training ----------------------------------------------------------
    best_auroc = -1.0
    nd_interval = args.nd_interval
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss, epoch_nd, n_batches, n_nd = 0.0, 0.0, 0, 0
        last_nd = float("nan")
        t0 = time.perf_counter()

        bar = tqdm(train_loader, desc=f"Epoch {epoch:3d}/{args.epochs}", unit="batch", leave=False)
        for batch in bar:
            pred, gt = process_batch(
                encoder, model, batch, level, out_size, num_levels, args.mask_pool, device, amp=amp
            )
            loss = F.binary_cross_entropy(pred, gt)
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            n_batches  += 1
            epoch_loss += loss.item()   # one sync per batch (unavoidable for loss display)

            # norm_dice syncs the GPU — only do it every nd_interval batches
            if n_batches % nd_interval == 0:
                with torch.no_grad():
                    nd = norm_dice_score(pred.detach(), gt)
                if nd == nd:   # skip NaN
                    epoch_nd += nd
                    n_nd     += 1
                    last_nd   = nd

            bar.set_postfix(loss=f"{epoch_loss / n_batches:.4f}",
                            nd=f"{last_nd:.3f}")

        bar.close()
        avg_loss = epoch_loss / max(n_batches, 1)
        avg_nd   = epoch_nd   / max(n_nd, 1)
        elapsed  = time.perf_counter() - t0
        print(f"Epoch {epoch:3d}/{args.epochs}  loss={avg_loss:.4f}  "
              f"norm_dice={avg_nd:.3f}  batches={n_batches}  {elapsed:.0f}s")

        # Validation
        val_metrics, val_figs = validate(
            model, encoder, ds_val, level, out_size, num_levels,
            args.mask_pool, device, args.val_items_per_class,
            fig_dir=fig_dir, epoch=epoch,
        )
        print(f"  val auroc={val_metrics['val/auroc']:.3f}  "
              f"norm_dice={val_metrics['val/norm_dice']:.3f}")

        # Save best checkpoint
        if val_metrics["val/auroc"] > best_auroc:
            best_auroc = val_metrics["val/auroc"]
            ckpt = {
                "epoch":   epoch,
                "model":   model_module.state_dict(),
                "config": {
                    "embed_dim":       embed_dim,
                    "num_heads":       args.num_heads,
                    "num_layers":      args.num_layers,
                    "ff_factor":       args.ff_factor,
                    "label_injection": args.label_injection,
                    "output_head":     args.output_head,
                    "pos_encoding":    args.pos_encoding,
                    "input_norm":      args.input_norm,
                    "grid_size":       list(out_size),
                    "dropout":         args.dropout,
                },
                "feature_level": level,
                "val_auroc":     best_auroc,
            }
            torch.save(ckpt, out_dir / "best.pt")
            print(f"  saved best checkpoint  auroc={best_auroc:.3f}")

        if use_wandb:
            import wandb
            wandb_figs = {k: wandb.Image(str(v)) for k, v in val_figs.items()}
            wandb.log({"train/loss": avg_loss, "train/norm_dice": avg_nd,
                       "epoch": epoch, **val_metrics, **wandb_figs})

    if use_wandb:
        import wandb
        wandb.finish()
    print(f"\nBest val AUROC: {best_auroc:.3f}  |  checkpoint: {out_dir}/best.pt")


if __name__ == "__main__":
    main()
