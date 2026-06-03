"""
Evaluate a trained PatchICLAttention checkpoint on the TotalSegmentator val split.

Mirrors experiments/feature_similarity/run.py — same metrics, same visualization.

Usage
-----
    python experiments/feature_attention/run.py --checkpoint experiments/feature_attention/checkpoints/best.pt
    python experiments/feature_attention/run.py --checkpoint best.pt --samples_per_class 10
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.totalseg_dataloader_incontext import TotalSegInContextDataset
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
    p.add_argument("--checkpoint", required=True,
                   help="Path to a PatchICLAttention .pt checkpoint.")
    p.add_argument("--data_root", default=TOTALSEG_ROOT)
    p.add_argument("--classes", nargs="+", default=[
        "liver", "lung_lower_lobe_right", "lung_lower_lobe_left",
        "lung_upper_lobe_right", "lung_upper_lobe_left", "lung_middle_lobe_right",
        "small_bowel", "heart", "autochthon_left", "autochthon_right",
        "hip_left", "hip_right", "aorta", "brain", "skull",
    ])
    p.add_argument("--context_size",     type=int, default=3)
    p.add_argument("--image_size",       type=int, nargs=3, default=[128, 128, 128])
    p.add_argument("--samples_per_class",type=int, default=5)
    p.add_argument("--mask_pool",        default="max", choices=["max", "avg"])
    p.add_argument("--stunet_variant",   default="base",
                   choices=["small", "base", "large", "huge"])
    p.add_argument("--stunet_pretrained",default=DEFAULT_PRETRAINED)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed",   type=int, default=42)
    p.add_argument("--out_dir",default="experiments/feature_attention/outputs")
    p.add_argument("--wandb_project", default="patch_icl_3d_exps",
                   help="Set to 'null' to disable W&B.")
    p.add_argument("--run_name", default=None)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Feature extraction helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def encode_image_only(encoder: STUNetEncoder, imgs: torch.Tensor) -> list[torch.Tensor]:
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
    x = mask.float().unsqueeze(1)
    if mode == "max":
        return F.adaptive_max_pool3d(x, output_size=size).squeeze(1)
    return F.adaptive_avg_pool3d(x, output_size=size).squeeze(1)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def soft_dice_score(pred: torch.Tensor, gt: torch.Tensor) -> float:
    num = 2 * (pred * gt).sum()
    den = pred.sum() + gt.sum() + 1e-6
    return (num / den).item()


def norm_dice_score(pred: torch.Tensor, gt: torch.Tensor) -> float:
    p = pred - pred.min()
    pmax = p.max()
    if pmax < 1e-8:
        return 0.0
    p = p / pmax
    num = 2 * (p * gt).sum()
    den = p.sum() + gt.sum() + 1e-6
    return (num / den).item()


def auroc_score(pred: torch.Tensor, gt: torch.Tensor) -> float:
    from sklearn.metrics import roc_auc_score
    gt_np   = (gt.cpu().numpy().ravel() > 0).astype(int)
    pred_np = pred.cpu().numpy().ravel()
    if gt_np.sum() == 0 or gt_np.sum() == len(gt_np):
        return float("nan")
    return roc_auc_score(gt_np, pred_np)


def auprc_score(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """Area under the Precision-Recall curve — more sensitive than AUROC for small organs."""
    from sklearn.metrics import average_precision_score
    gt_np   = (gt.cpu().numpy().ravel() > 0).astype(int)
    pred_np = pred.cpu().numpy().ravel()
    if gt_np.sum() == 0 or gt_np.sum() == len(gt_np):
        return float("nan")
    return float(average_precision_score(gt_np, pred_np))


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def _best_slice(mask: np.ndarray) -> int:
    counts = mask.sum(axis=(1, 2))
    return int(counts.argmax()) if counts.max() > 0 else mask.shape[0] // 2


def _ds_idx(full_idx: int, full_depth: int, ds_depth: int) -> int:
    return full_idx * ds_depth // full_depth


def _overlay(ax, image: np.ndarray, mask: np.ndarray, idx: int, title: str) -> None:
    sl = image[idx]
    sl_norm = (sl - sl.min()) / (sl.max() - sl.min() + 1e-6)
    ax.imshow(sl_norm, cmap="gray")
    ax.imshow(mask[idx], cmap="Reds", alpha=0.45, vmin=0, vmax=1)
    ax.set_title(f"{title} [z={idx}]", fontsize=8)
    ax.axis("off")


def _heatmap(ax, vol: np.ndarray, idx: int, title: str) -> None:
    ax.imshow(vol[idx], cmap="hot", vmin=0, vmax=1, interpolation="nearest")
    ax.set_title(title, fontsize=8)
    ax.axis("off")


def save_slice_figure(
    tgt_image: np.ndarray, tgt_gt: np.ndarray, tgt_gt_ds: np.ndarray,
    pred: np.ndarray,
    ctx_images: list[np.ndarray], ctx_gts: list[np.ndarray], ctx_gts_ds: list[np.ndarray],
    out_path: Path, title: str = "",
) -> None:
    from matplotlib.gridspec import GridSpec
    K = len(ctx_images)
    ds_size = tgt_gt_ds.shape[1]

    tgt_full_idx = _best_slice(tgt_gt)
    tgt_ds_idx   = _ds_idx(tgt_full_idx, tgt_gt.shape[0], tgt_gt_ds.shape[0])
    ctx_full_idx = [_best_slice(ctx_gts[k]) for k in range(K)]
    ctx_ds_idx   = [_ds_idx(ctx_full_idx[k], ctx_gts[k].shape[0], ctx_gts_ds[k].shape[0])
                    for k in range(K)]

    ncols = max(3, 2 * K)
    span  = ncols // 3
    fig = plt.figure(figsize=(3.2 * ncols, 6.5))
    gs  = GridSpec(2, ncols, figure=fig, hspace=0.35, wspace=0.05)

    _overlay(fig.add_subplot(gs[0, 0:span]),    tgt_image, tgt_gt, tgt_full_idx, "Target + GT")
    _heatmap(fig.add_subplot(gs[0, span:2*span]),tgt_gt_ds, tgt_ds_idx, f"Target GT ↓{ds_size}³")
    _heatmap(fig.add_subplot(gs[0, 2*span:]),    pred, tgt_ds_idx, "Predicted (attention)")

    for k in range(K):
        _overlay(fig.add_subplot(gs[1, 2*k]),   ctx_images[k], ctx_gts[k], ctx_full_idx[k], f"Ctx {k} + GT")
        _heatmap(fig.add_subplot(gs[1, 2*k+1]), ctx_gts_ds[k], ctx_ds_idx[k], f"Ctx {k} GT ↓{ds_size}³")
    for col in range(2 * K, ncols):
        fig.add_subplot(gs[1, col]).set_visible(False)

    fig.suptitle(title, fontsize=9)
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load checkpoint --------------------------------------------------
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg  = ckpt["config"]
    level    = ckpt.get("feature_level", "-1")
    out_size = tuple(cfg["grid_size"])
    print(f"Loaded checkpoint: val_auroc={ckpt.get('val_auroc', '?'):.3f}  "
          f"epoch={ckpt.get('epoch', '?')}")
    print(f"  label_injection={cfg['label_injection']}  output_head={cfg['output_head']}  "
          f"pos_encoding={cfg['pos_encoding']}  input_norm={cfg['input_norm']}  "
          f"num_layers={cfg['num_layers']}")

    model = PatchICLAttention(**cfg).to(device).eval()
    model.load_state_dict(ckpt["model"])

    # ---- Dataset ----------------------------------------------------------
    ds = TotalSegInContextDataset(
        root=args.data_root,
        classes=args.classes,
        image_size=tuple(args.image_size),
        split="val",
        context_size=args.context_size,
    )

    # ---- Encoder (frozen) -------------------------------------------------
    encoder = STUNetEncoder(
        in_channels=1,
        variant=args.stunet_variant,
        pretrained=args.stunet_pretrained,
        freeze_encoder=True,
    ).to(device).eval()
    num_levels = len(encoder.skip_channels) + 1

    # ---- W&B ---------------------------------------------------------------
    use_wandb = args.wandb_project and args.wandb_project.lower() != "null"
    if use_wandb:
        import wandb
        wandb.init(
            project=args.wandb_project,
            name=args.run_name,
            config={**cfg, "feature_level": level, "mask_pool": args.mask_pool,
                    "checkpoint": args.checkpoint},
        )

    # ---- Build per-class sample index ------------------------------------
    from collections import defaultdict
    cls_to_indices: dict[str, list[int]] = defaultdict(list)
    for i, (_, cls) in enumerate(ds.samples):
        cls_to_indices[cls].append(i)

    # ---- Evaluation loop -------------------------------------------------
    results: list[dict] = []
    fig_idx = 0

    for cls in ds.classes:
        indices   = cls_to_indices[cls]
        collected = 0

        for sample_idx in indices:
            if collected >= args.samples_per_class:
                break
            try:
                item = ds[sample_idx]
            except Exception:
                continue

            subj    = item["subject"]
            image   = item["image"].unsqueeze(0).to(device)
            label   = item["label"].to(device)
            ctx_in  = item["context_in"].to(device)
            ctx_out = item["context_out"].to(device)
            K = ctx_in.shape[0]

            with torch.no_grad():
                tgt_feats      = encode_image_only(encoder, image)           # image: (1,1,D,H,W)
                ctx_imgs_flat  = ctx_in.reshape(K, 1, *ctx_in.shape[3:])
                ctx_feats_flat = encode_image_only(encoder, ctx_imgs_flat)

            tgt_feat_ds      = extract_features(tgt_feats, level, out_size, num_levels)  # (1,C,D',H',W')
            ctx_feat_ds_flat = extract_features(ctx_feats_flat, level, out_size, num_levels)  # (K,C,D',H',W')
            C = tgt_feat_ds.shape[1]
            D_, H_, W_ = out_size
            N = D_ * H_ * W_

            tgt_mask_ds = downsample_mask(label.unsqueeze(0), out_size, args.mask_pool)  # (1,D',H',W')
            ctx_mask_ds = downsample_mask(ctx_out, out_size, args.mask_pool)              # (K,D',H',W')

            ctx_feat_ds = ctx_feat_ds_flat.unsqueeze(0)               # (1,K,C,D',H',W')
            tgt_feat = tgt_feat_ds.reshape(1, C, N).permute(0, 2, 1)  # (1,N,C)
            ctx_feat = ctx_feat_ds.permute(0, 1, 3, 4, 5, 2).reshape(1, K * N, C)  # (1,K*N,C)
            ctx_lbls = (ctx_mask_ds.reshape(1, K * N) > 0).float()    # (1,K*N)

            t0 = time.perf_counter()
            with torch.no_grad():
                pred = model(tgt_feat, ctx_feat, ctx_lbls).squeeze(0)  # (N,)
            inference_time = time.perf_counter() - t0

            gt_ds   = (tgt_mask_ds.squeeze(0) > 0).float()             # (D',H',W')
            pred_3d = pred.reshape(D_, H_, W_)
            gt_3d   = gt_ds

            d_soft = soft_dice_score(pred_3d, gt_3d)
            d_norm = norm_dice_score(pred_3d, gt_3d)
            auc    = auroc_score(pred_3d, gt_3d)
            aprc   = auprc_score(pred_3d, gt_3d)
            pred_np = pred_3d.cpu().numpy()

            print(f"[{cls:<30s}] subj={subj}  "
                  f"soft_dice={d_soft:.3f}  norm_dice={d_norm:.3f}  auroc={auc:.3f}  auprc={aprc:.3f}  "
                  f"pred[{pred_np.min():.3f}…{pred_np.max():.3f}]")
            results.append({"soft_dice": d_soft, "norm_dice": d_norm, "auroc": auc, "auprc": aprc,
                            "class": cls, "inference_time": inference_time})

            fig_path = out_dir / f"{fig_idx:03d}_{cls}_{subj}.png"
            save_slice_figure(
                tgt_image=item["image"].squeeze(0).cpu().numpy(),
                tgt_gt=item["label"].cpu().numpy(),
                tgt_gt_ds=gt_3d.cpu().numpy(),
                pred=pred_np,
                ctx_images=[ctx_in[k].squeeze(0).cpu().numpy() for k in range(K)],
                ctx_gts=[item["context_out"][k].cpu().numpy() for k in range(K)],
                ctx_gts_ds=[ctx_mask_ds[k].cpu().numpy() for k in range(K)],
                out_path=fig_path,
                title=f"{cls} | {subj} | norm_dice={d_norm:.3f}  auroc={auc:.3f}",
            )
            if use_wandb:
                import wandb
                wandb.log({
                    "sample/soft_dice":     d_soft,
                    "sample/norm_dice":     d_norm,
                    "sample/auroc":         auc,
                    "sample/auprc":         aprc,
                    "sample/inference_time":inference_time,
                    "sample/class":         cls,
                    "sample/figure":        wandb.Image(str(fig_path)),
                })
            collected += 1
            fig_idx  += 1

        if collected == 0:
            print(f"[{cls:<30s}] no valid samples")

    # ---- Summary -----------------------------------------------------------
    if results:
        per_cls: dict[str, list[dict]] = defaultdict(list)
        for r in results:
            per_cls[r["class"]].append(r)

        print(f"\n{'─'*80}")
        print(f"  {'class':<30s} {'soft_dice':>10} {'norm_dice':>10} {'auroc':>8} {'auprc':>8}  n")
        print(f"  {'─'*76}")
        for cls, rs in per_cls.items():
            print(f"  {cls:<30s} "
                  f"{np.nanmean([r['soft_dice'] for r in rs]):>10.3f} "
                  f"{np.nanmean([r['norm_dice'] for r in rs]):>10.3f} "
                  f"{np.nanmean([r['auroc']     for r in rs]):>8.3f} "
                  f"{np.nanmean([r['auprc']     for r in rs]):>8.3f}  {len(rs)}")
        print(f"  {'─'*76}")
        overall_soft = np.nanmean([r['soft_dice']      for r in results])
        overall_norm = np.nanmean([r['norm_dice']      for r in results])
        overall_auc  = np.nanmean([r['auroc']          for r in results])
        overall_aprc = np.nanmean([r['auprc']          for r in results])
        overall_time = np.mean(  [r['inference_time']  for r in results])
        print(f"  {'overall':<30s} {overall_soft:>10.3f} {overall_norm:>10.3f} {overall_auc:>8.3f} {overall_aprc:>8.3f}")
        print(f"\n  avg inference time : {overall_time*1000:.1f} ms/item")
        print(f"  Figures : {out_dir}/")

        if use_wandb:
            import wandb
            summary: dict = {
                "overall/soft_dice":      overall_soft,
                "overall/norm_dice":      overall_norm,
                "overall/auroc":          overall_auc,
                "overall/auprc":          overall_aprc,
                "overall/inference_time": overall_time,
            }
            for cls, rs in per_cls.items():
                summary[f"class/{cls}/soft_dice"] = np.nanmean([r['soft_dice'] for r in rs])
                summary[f"class/{cls}/norm_dice"]  = np.nanmean([r['norm_dice'] for r in rs])
                summary[f"class/{cls}/auroc"]      = np.nanmean([r['auroc']     for r in rs])
                summary[f"class/{cls}/auprc"]      = np.nanmean([r['auprc']     for r in rs])
            wandb.log(summary)
            wandb.finish()


if __name__ == "__main__":
    main()
