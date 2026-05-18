"""
Feature-similarity coarse segmentation experiment.

For each target patch at a chosen STU-Net feature level, predict the mask as
the softmax-weighted average of context labels, where weights come from cosine
similarity between the target patch features and every context patch.

Usage
-----
    python experiments/feature_similarity/run.py
    python experiments/feature_similarity/run.py --feature_level 3 --temperature 0.07
    python experiments/feature_similarity/run.py --stunet_pretrained /path/to/checkpoint.model
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

TABPFN_SRC = "/nfs/norasys/notebooks/camaret/repos/TabPFN/src"
if TABPFN_SRC not in sys.path:
    sys.path.insert(0, TABPFN_SRC)

from src.totalseg_dataloader_incontext import TotalSegInContextDataset
from src.models.encoders.stunet import STUNetEncoder

TOTALSEG_ROOT = (
    "/nfs/data/nii/data1/Analysis/camaret___in_context_segmentation"
    "/ANALYSIS_20251122/data/totalseg"
)


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_root", default=TOTALSEG_ROOT)
    p.add_argument("--classes", nargs="+", default=["liver","lung_lower_lobe_right","lung_lower_lobe_left","lung_upper_lobe_right","lung_upper_lobe_left","lung_middle_lobe_right","small_bowel","heart","autochthon_left","autochthon_right","hip_left","hip_right","aorta","brain", "skull"])
    p.add_argument("--context_size", type=int, default=1)
    p.add_argument("--image_size", type=int, nargs=3, default=[128, 128, 128])
    p.add_argument("--max_subjects", type=int, default=None,
                   help="Limit total subjects loaded (None = all). Keep None so "
                        "rare classes have enough context candidates.")
    p.add_argument("--samples_per_class", type=int, default=5,
                   help="How many valid target samples to evaluate per class.")
    p.add_argument("--mask_pool", default="max", choices=["max", "avg"],
                   help="How to downsample GT masks: 'max' (patch=1 if any voxel labeled) "
                        "or 'avg' (patch = fraction of labeled voxels).")
    p.add_argument("--feature_level", type=str, default="-1",
                   help="Encoder level to use: int index (-1 = bottleneck) or 'all' "
                        "to concatenate all levels along the channel dim.")
    p.add_argument("--output_size", type=int, nargs=3, default=[16, 16, 16],
                   help="Spatial resolution for similarity computation.")
    p.add_argument("--temperature", type=float, default=0.05,
                   help="Softmax temperature for similarity weights.")
    p.add_argument("--method", default="cosine", choices=["cosine", "tabpfn"],
                   help="Prediction method: cosine-similarity retrieval or TabPFN classifier.")
    p.add_argument("--stunet_variant", default="base",
                   choices=["small", "base", "large", "huge"])
    p.add_argument("--stunet_pretrained", default="/nfs/data/nii/data1/Analysis/camaret___in_context_segmentation/ANALYSIS_20251122/results/patch_icl/checkpoints/stunet/base_statedict.pt",
                   help="Path to a STU-Net checkpoint (.model or .pt).")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", default="experiments/feature_similarity/outputs")
    p.add_argument("--wandb_project", default="patch_icl_3d_exps",
                   help="W&B project name. Set to 'null' to disable logging.")
    p.add_argument("--run_name", default=None,
                   help="W&B run name (auto-generated if None).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Core prediction
# ---------------------------------------------------------------------------

@torch.no_grad()
def encode_image_only(
    encoder: STUNetEncoder,
    imgs: torch.Tensor,
) -> list[torch.Tensor]:
    """Run only the image encoder, skipping the mask branch entirely.

    Returns [s0, …, s_{n-2}, bottleneck] — same layout as the full forward,
    but the bottleneck is pure image features with no mask fusion.
    """
    bottleneck, skips = encoder.image_encoder(imgs, num_stages=encoder._num_stages)
    return skips + [bottleneck]


def predict_similarity(
    tgt_feat: torch.Tensor,    # (C, D, H, W)
    ctx_feats: torch.Tensor,   # (K, C, D, H, W)
    ctx_masks: torch.Tensor,   # (K, D, H, W)  float in {0, 1}
    temperature: float,
) -> torch.Tensor:
    """
    Predict a soft mask for the target by cosine-similarity retrieval.

    For each target spatial position, compute cosine similarity against every
    context position (across all K context images), apply softmax, and take
    the weighted sum of context mask values.

    Returns
    -------
    pred : (D, H, W) float tensor in [0, 1].
    """
    C, D, H, W = tgt_feat.shape
    K = ctx_feats.shape[0]
    N = D * H * W

    tgt_flat = tgt_feat.reshape(C, N).T                          # (N, C)
    ctx_flat = ctx_feats.reshape(K, C, N).permute(0, 2, 1).reshape(K * N, C)  # (K*N, C)
    ctx_labels = ctx_masks.reshape(K * N).float()                # (K*N,)

    tgt_norm = F.normalize(tgt_flat, dim=-1)   # (N, C)
    ctx_norm = F.normalize(ctx_flat, dim=-1)   # (K*N, C)
    sim = tgt_norm @ ctx_norm.T                # (N, K*N)

    weights = F.softmax(sim / temperature, dim=-1)  # (N, K*N)
    pred = (weights @ ctx_labels).reshape(D, H, W)  # (D, H, W)
    return pred


def predict_tabpfn(
    tgt_feat: torch.Tensor,    # (C, D, H, W)
    ctx_feats: torch.Tensor,   # (K, C, D, H, W)
    ctx_masks: torch.Tensor,   # (K, D, H, W)  float
) -> torch.Tensor:
    """
    Predict a soft mask for the target using TabPFN in-context classification.

    Each context patch (position × context image) is a training sample with
    features = encoder embedding and label = 0/1 (any labeled voxel → 1).
    TabPFN fits a non-linear classifier from these examples in a single forward
    pass, then returns class-1 probabilities for every target patch.

    Falls back to the context positive rate if all context labels are the same
    class (TabPFN requires at least one sample of each class).

    Returns
    -------
    pred : (D, H, W) float tensor in [0, 1].
    """
    from tabpfn import TabPFNClassifier

    C, D, H, W = tgt_feat.shape
    K = ctx_feats.shape[0]
    N = D * H * W

    X_ctx = ctx_feats.reshape(K, C, N).permute(0, 2, 1).reshape(K * N, C).cpu().numpy()
    y_ctx = (ctx_masks.reshape(K * N) > 0).cpu().numpy().astype(int)
    X_tgt = tgt_feat.reshape(C, N).T.cpu().numpy()

    if y_ctx.sum() == 0 or y_ctx.sum() == len(y_ctx):
        fill = float(y_ctx.mean())
        return torch.full((D, H, W), fill, dtype=torch.float32, device=tgt_feat.device)

    clf = TabPFNClassifier(ignore_pretraining_limits=True)
    clf.fit(X_ctx, y_ctx)
    proba = clf.predict_proba(X_tgt)   # (N, 2)
    return torch.from_numpy(proba[:, 1]).float().reshape(D, H, W).to(tgt_feat.device)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def downsample_feat(feat: torch.Tensor, size: tuple[int, int, int]) -> torch.Tensor:
    """feat: (1, C, d, h, w) → (C, D, H, W)."""
    return F.interpolate(feat, size=size, mode="trilinear", align_corners=False).squeeze(0)


def extract_features(
    feats: list[torch.Tensor],
    level: str,
    out_size: tuple[int, int, int],
    num_levels: int,
) -> torch.Tensor:
    """Downsample and return features at the chosen level(s).

    level="all"  → all levels interpolated to out_size and concatenated on C dim.
    level="-1"   → single level (int index, negative indexing supported).
    Returns (C, D', H', W').
    """
    if level == "all":
        return torch.cat([downsample_feat(f, out_size) for f in feats], dim=0)
    return downsample_feat(feats[int(level) % num_levels], out_size)


def downsample_mask(mask: torch.Tensor, size: tuple[int, int, int], mode: str = "max") -> torch.Tensor:
    """mask: (D, H, W) → (D', H', W').
    mode='max': patch is 1 if any voxel in it is labeled (conservative).
    mode='avg': patch value is the fraction of labeled voxels (soft label).
    """
    x = mask.float().unsqueeze(0).unsqueeze(0)
    if mode == "max":
        return F.adaptive_max_pool3d(x, output_size=size).squeeze()
    return F.adaptive_avg_pool3d(x, output_size=size).squeeze()


def soft_dice_score(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """Soft dice — biased by prediction scale; use alongside auroc."""
    num = 2 * (pred * gt).sum()
    den = pred.sum() + gt.sum() + 1e-6
    return (num / den).item()


def norm_dice_score(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """Soft dice after min-max normalising pred to [0,1].
    Decouples ranking ability from absolute prediction scale."""
    p = pred - pred.min()
    pmax = p.max()
    if pmax < 1e-8:
        return 0.0
    p = p / pmax
    num = 2 * (p * gt).sum()
    den = p.sum() + gt.sum() + 1e-6
    return (num / den).item()


def auroc_score(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """Area under the ROC curve: 1.0 = perfect ranking, 0.5 = random.
    Scale-invariant — tests only whether GT patches rank above background."""
    from sklearn.metrics import roc_auc_score
    gt_np   = (gt.cpu().numpy().ravel() > 0).astype(int)
    pred_np = pred.cpu().numpy().ravel()
    if gt_np.sum() == 0 or gt_np.sum() == len(gt_np):
        return float("nan")
    return roc_auc_score(gt_np, pred_np)


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def _best_slice(mask: np.ndarray) -> int:
    """Slice index (axis 0) with the most labeled voxels; central slice if empty."""
    counts = mask.sum(axis=(1, 2))
    return int(counts.argmax()) if counts.max() > 0 else mask.shape[0] // 2


def _ds_idx(full_idx: int, full_depth: int, ds_depth: int) -> int:
    """Map a full-res slice index to the corresponding downsampled slice."""
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
    tgt_image: np.ndarray,          # (D, H, W)  full-res target CT
    tgt_gt: np.ndarray,             # (D, H, W)  full-res target GT
    tgt_gt_ds: np.ndarray,          # (D', H', W')  downsampled target GT
    pred: np.ndarray,               # (D', H', W')  soft prediction
    ctx_images: list[np.ndarray],   # K × (D, H, W)  full-res context CTs
    ctx_gts: list[np.ndarray],      # K × (D, H, W)  full-res context GTs
    ctx_gts_ds: list[np.ndarray],   # K × (D', H', W')  downsampled context GTs
    out_path: Path,
    title: str = "",
) -> None:
    """
    Row 0: tgt+gt overlay | downsampled tgt gt | predicted soft mask
    Row 1: for each context k — ctx_k+gt overlay | downsampled ctx_k gt
    """
    from matplotlib.gridspec import GridSpec

    K = len(ctx_images)
    ds_size = tgt_gt_ds.shape[1]

    # Pre-compute best slice indices so overlay and heatmap show the same plane
    tgt_full_idx = _best_slice(tgt_gt)
    tgt_ds_idx   = _ds_idx(tgt_full_idx, tgt_gt.shape[0], tgt_gt_ds.shape[0])
    ctx_full_idx = [_best_slice(ctx_gts[k])   for k in range(K)]
    ctx_ds_idx   = [_ds_idx(ctx_full_idx[k], ctx_gts[k].shape[0], ctx_gts_ds[k].shape[0])
                    for k in range(K)]

    ncols = max(3, 2 * K)
    span  = ncols // 3

    fig = plt.figure(figsize=(3.2 * ncols, 6.5))
    gs = GridSpec(2, ncols, figure=fig, hspace=0.35, wspace=0.05)

    # --- Row 0 ---
    _overlay(fig.add_subplot(gs[0, 0:span]),
             tgt_image, tgt_gt, tgt_full_idx, "Target + GT")
    _heatmap(fig.add_subplot(gs[0, span:2*span]),
             tgt_gt_ds, tgt_ds_idx, f"Target GT ↓{ds_size}³")
    _heatmap(fig.add_subplot(gs[0, 2*span:]),
             pred, tgt_ds_idx, "Predicted (similarity)")

    # --- Row 1: one pair per context ---
    for k in range(K):
        _overlay(fig.add_subplot(gs[1, 2*k]),
                 ctx_images[k], ctx_gts[k], ctx_full_idx[k], f"Ctx {k} + GT")
        _heatmap(fig.add_subplot(gs[1, 2*k + 1]),
                 ctx_gts_ds[k], ctx_ds_idx[k], f"Ctx {k} GT ↓{ds_size}³")

    # Hide unused axes in row 1 if 2*K < ncols
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
    out_size = tuple(args.output_size)

    # ---- Dataset -----------------------------------------------------------
    ds = TotalSegInContextDataset(
        root=args.data_root,
        classes=args.classes,
        image_size=tuple(args.image_size),
        split="val",
        context_size=args.context_size,
        max_subjects=args.max_subjects,
    )

    # ---- Encoder -----------------------------------------------------------
    encoder = STUNetEncoder(
        in_channels=1,
        variant=args.stunet_variant,
        pretrained=args.stunet_pretrained,
        freeze_encoder=True,
    ).to(device).eval()

    num_levels = len(encoder.skip_channels) + 1  # skips + bottleneck
    level = args.feature_level  # "all" or int string; resolved in extract_features
    level_desc = "all" if level == "all" else f"index {int(level) % num_levels}"
    print(f"Feature level: {level_desc} of {num_levels} | "
          f"output_size={out_size} | temperature={args.temperature}\n")

    # ---- W&B ---------------------------------------------------------------
    use_wandb = args.wandb_project and args.wandb_project.lower() != "null"
    if use_wandb:
        import wandb
        wandb.init(
            project=args.wandb_project,
            name=args.run_name,
            config={
                "method":           args.method,
                "feature_level":    args.feature_level,
                "output_size":      args.output_size,
                "temperature":      args.temperature,
                "mask_pool":        args.mask_pool,
                "context_size":     args.context_size,
                "image_size":       args.image_size,
                "stunet_variant":   args.stunet_variant,
                "stunet_pretrained": str(args.stunet_pretrained),
                "samples_per_class": args.samples_per_class,
                "classes":          args.classes,
            },
        )

    # ---- Build per-class sample index --------------------------------------
    from collections import defaultdict
    cls_to_indices: dict[str, list[int]] = defaultdict(list)
    for i, (subj, cls) in enumerate(ds.samples):
        cls_to_indices[cls].append(i)

    # ---- Evaluation loop (per class) ---------------------------------------
    results: list[dict] = []
    fig_idx = 0

    for cls in ds.classes:
        indices = cls_to_indices[cls]
        collected = 0

        for sample_idx in indices:
            if collected >= args.samples_per_class:
                break

            try:
                item = ds[sample_idx]
            except Exception:
                continue  # no context candidates for this subject — skip

            subj    = item["subject"]
            image   = item["image"].unsqueeze(0).to(device)           # (1, 1, D, H, W)
            label   = item["label"].to(device)                        # (D, H, W) int64
            ctx_in  = item["context_in"].to(device)                   # (K, 1, D, H, W)
            ctx_out = item["context_out"].to(device)                   # (K, D, H, W) int64
            K = ctx_in.shape[0]

            # Encode images only — mask branch skipped so features are comparable
            tgt_feats = encode_image_only(encoder, image)
            ctx_all_feats = [
                encode_image_only(encoder, ctx_in[k : k + 1]) for k in range(K)
            ]

            # Extract and downsample to out_size (concat across C if level="all")
            tgt_feat_ds = extract_features(tgt_feats, level, out_size, num_levels)
            ctx_feat_ds = torch.stack([
                extract_features(f, level, out_size, num_levels) for f in ctx_all_feats
            ])                                                                     # (K, C, D', H', W')

            # Downsample GT masks
            tgt_mask_ds = downsample_mask(label, out_size, args.mask_pool)
            ctx_mask_ds = torch.stack([
                downsample_mask(ctx_out[k], out_size, args.mask_pool) for k in range(K)
            ])

            t0 = time.perf_counter()
            if args.method == "tabpfn":
                pred = predict_tabpfn(tgt_feat_ds, ctx_feat_ds, ctx_mask_ds)
            else:
                pred = predict_similarity(tgt_feat_ds, ctx_feat_ds, ctx_mask_ds, args.temperature)
            inference_time = time.perf_counter() - t0
            gt_ds   = tgt_mask_ds.float()
            d_soft  = soft_dice_score(pred, gt_ds)
            d_norm  = norm_dice_score(pred, gt_ds)
            auc     = auroc_score(pred, gt_ds)
            pred_np = pred.cpu().numpy()

            print(f"[{cls:<30s}] subj={subj}  "
                  f"soft_dice={d_soft:.3f}  norm_dice={d_norm:.3f}  auroc={auc:.3f}  "
                  f"pred[{pred_np.min():.3f}…{pred_np.max():.3f}]")
            results.append({"soft_dice": d_soft, "norm_dice": d_norm, "auroc": auc, "class": cls,
                            "inference_time": inference_time})

            fig_path = out_dir / f"{fig_idx:03d}_{cls}_{subj}.png"
            save_slice_figure(
                tgt_image=item["image"].squeeze(0).cpu().numpy(),
                tgt_gt=label.cpu().numpy(),
                tgt_gt_ds=gt_ds.cpu().numpy(),
                pred=pred_np,
                ctx_images=[ctx_in[k].squeeze(0).cpu().numpy() for k in range(K)],
                ctx_gts=[ctx_out[k].cpu().numpy() for k in range(K)],
                ctx_gts_ds=[ctx_mask_ds[k].cpu().numpy() for k in range(K)],
                out_path=fig_path,
                title=f"{cls} | {subj} | norm_dice={d_norm:.3f}  auroc={auc:.3f}",
            )
            if use_wandb:
                wandb.log({
                    "sample/soft_dice":     d_soft,
                    "sample/norm_dice":     d_norm,
                    "sample/auroc":         auc,
                    "sample/inference_time": inference_time,
                    "sample/class":         cls,
                    "sample/subject":       subj,
                    "sample/figure":        wandb.Image(str(fig_path)),
                })
            collected += 1
            fig_idx  += 1

        if collected == 0:
            print(f"[{cls:<30s}] no valid samples (not enough context subjects)")

    # ---- Summary -----------------------------------------------------------
    if results:
        per_cls: dict[str, list[dict]] = defaultdict(list)
        for r in results:
            per_cls[r["class"]].append(r)
        print(f"\n{'─'*70}")
        print(f"  {'class':<30s} {'soft_dice':>10} {'norm_dice':>10} {'auroc':>8}  n")
        print(f"  {'─'*66}")
        for cls, rs in per_cls.items():
            print(f"  {cls:<30s} "
                  f"{np.nanmean([r['soft_dice'] for r in rs]):>10.3f} "
                  f"{np.nanmean([r['norm_dice'] for r in rs]):>10.3f} "
                  f"{np.nanmean([r['auroc']     for r in rs]):>8.3f}  {len(rs)}")
        print(f"  {'─'*66}")
        overall_soft = np.nanmean([r['soft_dice'] for r in results])
        overall_norm = np.nanmean([r['norm_dice'] for r in results])
        overall_auc  = np.nanmean([r['auroc']     for r in results])
        print(f"  {'overall':<30s} "
              f"{overall_soft:>10.3f} "
              f"{overall_norm:>10.3f} "
              f"{overall_auc:>8.3f}")
        print(f"\n  Figures : {out_dir}/")

        overall_time = np.mean([r['inference_time'] for r in results])
        print(f"  avg inference time : {overall_time*1000:.1f} ms/item")

        if use_wandb:
            summary: dict = {
                "overall/soft_dice":      overall_soft,
                "overall/norm_dice":      overall_norm,
                "overall/auroc":          overall_auc,
                "overall/inference_time": overall_time,
            }
            for cls, rs in per_cls.items():
                summary[f"class/{cls}/soft_dice"] = np.nanmean([r['soft_dice'] for r in rs])
                summary[f"class/{cls}/norm_dice"]  = np.nanmean([r['norm_dice'] for r in rs])
                summary[f"class/{cls}/auroc"]      = np.nanmean([r['auroc']     for r in rs])
            wandb.log(summary)
            wandb.finish()


if __name__ == "__main__":
    main()
