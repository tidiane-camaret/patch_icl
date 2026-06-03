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
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from torch.utils.data import DataLoader, Subset
from src.totalseg_dataloader_incontext import TotalSegInContextDataset, incontext_collate_fn
from src.models.encoders.stunet import STUNetEncoder
from src.models.encoders.nninteractive import NNInteractiveEncoder
from src.models.encoders.threedino import ThreeDINOEncoder
from src.models.encoders.vocomni import VoComniEncoder
from src.models.encoders.vocomni_nnunet import VoComniNNUNetEncoder
from data.totalseg_classes import resolve_classes

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
    p.add_argument("--classes", nargs="+", default=["benchmark"],
                   help="Class list or a named preset: 'benchmark', 'not_benchmark', "
                        "or any split name in label_stats.csv.")#["liver","lung_lower_lobe_right","lung_lower_lobe_left","lung_upper_lobe_right","lung_upper_lobe_left","lung_middle_lobe_right","small_bowel","heart","autochthon_left","autochthon_right","hip_left","hip_right","aorta","brain", "skull"])
    p.add_argument("--context_size", type=int, default=1)
    p.add_argument("--image_size", type=int, nargs=3, default=[192, 192, 192])
    p.add_argument("--max_subjects", type=int, default=None,
                   help="Limit total subjects loaded (None = all). Keep None so "
                        "rare classes have enough context candidates.")
    p.add_argument("--samples_per_class", type=int, default=10,
                   help="How many valid target samples to evaluate per class.")
    p.add_argument("--mask_pool", default="avg", choices=["max", "avg"],
                   help="How to downsample GT masks: 'max' (patch=1 if any voxel labeled) "
                        "or 'avg' (patch = fraction of labeled voxels).")
    p.add_argument("--feature_level", type=str, nargs="+", default=["-1"],
                   help="Encoder level(s) to use. Pass one or more int indices "
                        "(-1 = bottleneck), or the single token 'all' to use all levels. "
                        "Multiple indices are concatenated on the channel dim, e.g. "
                        "--feature_level -1 -2 -3")
    p.add_argument("--output_size", type=int, nargs=3, default=[8, 8, 8],
                   help="Spatial resolution for similarity computation.")
    p.add_argument("--temperature", type=float, default=0.05,
                   help="Softmax temperature for similarity weights.")
    p.add_argument("--method", default="cosine", choices=["cosine", "tabpfn", "prototype"],
                   help="Prediction method: cosine-similarity retrieval, TabPFN classifier, "
                        "or prototype (masked-avg-pool foreground/background prototypes).")
    p.add_argument("--encoder", default="stunet",
                   choices=["stunet", "nninteractive", "threedino", "vocomni", "vocomni_nnunet"],
                   help="Encoder backbone.")
    # STU-Net options
    p.add_argument("--stunet_variant", default="base",
                   choices=["small", "base", "large", "huge"])
    p.add_argument("--stunet_pretrained", default="/nfs/data/nii/data1/Analysis/camaret___in_context_segmentation/ANALYSIS_20251122/results/patch_icl/checkpoints/stunet/base_statedict.pt",
                   help="Path to a STU-Net checkpoint (.model or .pt).")
    # NNInteractive options
    p.add_argument("--nnint_ckpt", default="/home/dpxuser/model_checkpoints/nnint/nnInteractive_v1.0",
                   help="Path to NNInteractive checkpoint directory.")
    p.add_argument("--nnint_mask_injection", default="none", choices=["ch1", "separate", "none"],
                   help="How context masks are injected into the NNInteractive encoder.")
    p.add_argument("--nnint_num_stages", type=int, default=5,
                   help="Number of encoder stages (5 → 16× stride, 320-ch bottleneck at 128³).")
    # 3DINO options
    p.add_argument("--dino_ckpt",
                   default="/home/dpxuser/model_checkpoints/3DINO/3dino_vit_weights.pth",
                   help="Path to 3DINO ViT-Large-3D weights (.pth).")
    p.add_argument("--dino_n_blocks", type=int, default=4,
                   help="How many block-group outputs to use (1–4). All at D//16 resolution.")
    # VoComni options
    p.add_argument("--vocomni_ckpt", default="/home/dpxuser/model_checkpoints/voco/VoComni_B.pt",
                   help="Path to VoCo/VoComni .pt checkpoint (VoComni_B/L/H.pt). "
                        "Omit to use random weights.")
    p.add_argument("--vocomni_feature_size", type=int, default=48, choices=[48, 96, 192],
                   help="SwinUNETR base embedding dim: 48=Base, 96=Large, 192=Huge.")
    p.add_argument("--vocomni_compile", action="store_true", default=True,
                   help="Wrap SwinUNETR with torch.compile (default True; use --vocomni_compile=False to disable).")
    p.add_argument("--no_vocomni_compile", dest="vocomni_compile", action="store_false")
    # VoComni nnUNet options
    p.add_argument("--vocomni_nnunet_ckpt",
                   default="/home/dpxuser/model_checkpoints/voco/VoComni_nnunet.pt",
                   help="Path to VoComni_nnunet.pt PlainConvUNet checkpoint.")
    p.add_argument("--vocomni_nnunet_compile", action="store_true", default=True,
                   help="Wrap PlainConvEncoder with torch.compile (default True).")
    p.add_argument("--no_vocomni_nnunet_compile", dest="vocomni_nnunet_compile", action="store_false")
    # Dataset options
    p.add_argument("--use_crop", action="store_true",
                   help="Load native-res crops centred on the organ instead of pre-resized volumes.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", default="results/feature_similarity/outputs")
    p.add_argument("--batch_size", type=int, default=8,
                   help="Number of samples to encode in one GPU forward pass.")
    p.add_argument("--num_workers", type=int, default=20,
                   help="DataLoader worker processes for data loading.")
    p.add_argument("--tabpfn_n_estimators", type=int, default=6,
                   help="TabPFN ensemble size. 1=fastest; 4-8 for better quality.")
    p.add_argument("--tabpfn_memory_saving", default="auto",
                   choices=["auto", "true", "false"],
                   help="TabPFN memory_saving_mode. 'auto' chunks attention on small GPUs "
                        "(causes ~50%% VRAM); 'false' uses full VRAM per call.")
    p.add_argument("--balance_ratio", type=float, default=None,
                   help="If set, subsample background context patches to this multiple of "
                        "the foreground count (e.g. 3.0 keeps 3 bg per fg patch). "
                        "None disables balancing.")
    p.add_argument("--wandb_project", default="patch_icl_feature_similarity",
                   help="W&B project name. Set to 'null' to disable logging.")
    p.add_argument("--run_name", default=None,
                   help="W&B run name (auto-generated if None).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Core prediction
# ---------------------------------------------------------------------------

@torch.inference_mode()
def encode_image_only(
    encoder: STUNetEncoder,
    imgs: torch.Tensor,
) -> list[torch.Tensor]:
    """STUNet image-only encoding (skips mask branch).

    Returns [s0, …, s_{n-2}, bottleneck] ordered high-res → low-res.
    """
    bottleneck, skips = encoder.image_encoder(imgs, num_stages=encoder._num_stages)
    return skips + [bottleneck]


@torch.inference_mode()
def encode_image_generic(
    encoder: nn.Module,
    imgs: torch.Tensor,
) -> list[torch.Tensor]:
    """Image-only encoding for encoders whose forward(imgs, masks=None) ignores masks."""
    return encoder(imgs, None)


@torch.inference_mode()
def encode_target_nnint(
    encoder: NNInteractiveEncoder,
    imgs: torch.Tensor,
) -> list[torch.Tensor]:
    """Encode target with a zero mask (no prior segmentation available)."""
    return encoder(imgs, torch.zeros_like(imgs))


@torch.inference_mode()
def encode_context_nnint(
    encoder: NNInteractiveEncoder,
    ctx_imgs: torch.Tensor,   # (1, 1, D, H, W)
    ctx_masks: torch.Tensor,  # (1, 1, D, H, W) float
) -> list[torch.Tensor]:
    """Encode context image conditioned on its GT mask."""
    return encoder(ctx_imgs, ctx_masks)


def balance_context(
    ctx_flat: torch.Tensor,    # (K*N, C)
    labels_flat: torch.Tensor, # (K*N,)  float in [0, 1]
    bg_ratio: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Subsample background patches to bg_ratio × effective_foreground_count.

    Uses soft label values directly — no hard threshold.
    effective_fg = round(sum(labels)), i.e. the expected number of fully-labeled
    foreground patches implied by the soft labels. The top effective_fg patches
    by label value are treated as foreground; the rest are randomly subsampled
    to bg_ratio * effective_fg patches.

    This works correctly for small organs with avg-pooled labels where
    foreground patches may have values like 0.02–0.15.
    """
    n_eff_fg = max(1, round(labels_flat.sum().item()))
    order = labels_flat.argsort(descending=True)   # high label → foreground
    keep_fg  = order[:n_eff_fg]
    bg_cands = order[n_eff_fg:]
    if len(bg_cands) == 0:
        print(f"  [balance] fg={n_eff_fg}  bg=0 (all patches are fg)")
        return ctx_flat, labels_flat
    n_bg = min(len(bg_cands), max(1, int(n_eff_fg * bg_ratio)))
    perm = torch.randperm(len(bg_cands), device=bg_cands.device)[:n_bg]
    keep = torch.cat([keep_fg, bg_cands[perm]]).sort().values
    print(f"  [balance] fg={n_eff_fg}  bg={n_bg}  total={n_eff_fg + n_bg}"
          f"  (of {len(labels_flat)} patches, ratio={bg_ratio})")
    return ctx_flat[keep], labels_flat[keep]


def predict_similarity(
    tgt_feat: torch.Tensor,    # (C, D, H, W)
    ctx_feats: torch.Tensor,   # (K, C, D, H, W)
    ctx_masks: torch.Tensor,   # (K, D, H, W)  float in {0, 1}
    temperature: float,
    balance_ratio: float | None = None,
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

    if balance_ratio is not None:
        ctx_flat, ctx_labels = balance_context(ctx_flat, ctx_labels, balance_ratio)

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
    clf,                       # pre-instantiated TabPFNClassifier (reused across calls)
    balance_ratio: float | None = None,
) -> torch.Tensor:
    """Predict a soft mask using TabPFN in-context classification.

    Each context patch is a training sample (features = encoder embedding,
    label = 0/1). TabPFN fits a classifier in one forward pass, then returns
    class-1 probabilities for every target patch.

    Falls back to context positive rate when all context labels are the same class.
    """
    C, D, H, W = tgt_feat.shape
    K = ctx_feats.shape[0]
    N = D * H * W

    ctx_flat = ctx_feats.reshape(K, C, N).permute(0, 2, 1).reshape(K * N, C)
    ctx_labels = (ctx_masks.reshape(K * N) > 0).float()

    if balance_ratio is not None:
        ctx_flat, ctx_labels = balance_context(ctx_flat, ctx_labels, balance_ratio)

    X_ctx = ctx_flat.cpu().numpy()
    y_ctx = ctx_labels.cpu().numpy().astype(int)
    X_tgt = tgt_feat.reshape(C, N).T.cpu().numpy()

    if y_ctx.sum() == 0 or y_ctx.sum() == len(y_ctx):
        fill = float(y_ctx.mean())
        return torch.full((D, H, W), fill, dtype=torch.float32, device=tgt_feat.device)

    # Per-feature z-score so TabPFN receives standardized input (matches pretraining)
    mu  = X_ctx.mean(axis=0, keepdims=True)
    sig = X_ctx.std(axis=0, keepdims=True) + 1e-8
    X_ctx = (X_ctx - mu) / sig
    X_tgt = (X_tgt - mu) / sig

    clf.fit(X_ctx, y_ctx)
    proba = clf.predict_proba(X_tgt)   # (N, 2)
    return torch.from_numpy(proba[:, 1]).float().reshape(D, H, W).to(tgt_feat.device)


def predict_prototype(
    tgt_feat: torch.Tensor,    # (C, D, H, W)
    ctx_feats: torch.Tensor,   # (K, C, D, H, W)
    ctx_masks: torch.Tensor,   # (K, D, H, W)  float in [0, 1]
    temperature: float,
    balance_ratio: float | None = None,
) -> torch.Tensor:
    """Predict a soft mask using masked-average-pooling prototypes.

    Computes one foreground prototype P+ and one background prototype P- by
    averaging context features weighted by the context mask and its inverse.
    Each target position is scored by cos_sim(x, P+) - cos_sim(x, P-), passed
    through a sigmoid scaled by temperature.

    Advantages over per-position cosine retrieval:
      - O(C) instead of O(K*N*C) — much faster with large K or output grids.
      - Noise from individual ambiguous patches is averaged out.
      - Principled few-shot learning baseline (PANet / CANet lineage).

    Returns
    -------
    pred : (D, H, W) float tensor in [0, 1].
    """
    C = tgt_feat.shape[0]

    fg_w = ctx_masks.reshape(-1)                   # (K*N,) soft weights
    bg_w = (1.0 - ctx_masks).reshape(-1)
    ctx_flat = ctx_feats.reshape(-1, C)            # (K*N, C)

    if balance_ratio is not None:
        ctx_flat, fg_w_bal = balance_context(ctx_flat, fg_w, balance_ratio)
        bg_w = 1.0 - fg_w_bal
        fg_w = fg_w_bal

    fg_sum = fg_w.sum().clamp(min=1e-6)
    bg_sum = bg_w.sum().clamp(min=1e-6)
    P_pos = (ctx_flat * fg_w.unsqueeze(1)).sum(dim=0) / fg_sum   # (C,)
    P_neg = (ctx_flat * bg_w.unsqueeze(1)).sum(dim=0) / bg_sum   # (C,)

    P_pos = F.normalize(P_pos, dim=0)
    P_neg = F.normalize(P_neg, dim=0)

    N = tgt_feat.shape[1] * tgt_feat.shape[2] * tgt_feat.shape[3]
    tgt_flat = F.normalize(tgt_feat.reshape(C, N).T, dim=-1)     # (N, C)

    score = (tgt_flat @ P_pos - tgt_flat @ P_neg) / temperature  # (N,)
    return torch.sigmoid(score).reshape(*tgt_feat.shape[1:])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def downsample_feat(feat: torch.Tensor, size: tuple[int, int, int]) -> torch.Tensor:
    """feat: (1, C, d, h, w) → (C, D, H, W) in float32.

    Uses avg-pool for downsampling (preserves activation energy) and
    trilinear interpolation for upsampling only.
    """
    x = feat.float()   # ensure fp32 before spatial ops
    d, h, w = x.shape[2:]
    if d >= size[0] and h >= size[1] and w >= size[2]:
        return F.adaptive_avg_pool3d(x, output_size=size).squeeze(0)
    return F.interpolate(x, size=size, mode="trilinear", align_corners=False).squeeze(0)


def extract_features(
    feats: list[torch.Tensor],
    levels: list[str],
    out_size: tuple[int, int, int],
    num_levels: int,
) -> torch.Tensor:
    """Downsample and return features at the chosen level(s).

    levels=["all"]      → all levels concatenated on C dim.
    levels=["-1"]       → single bottleneck level.
    levels=["-1","-2"]  → two levels concatenated.
    Returns (C, D', H', W') float32.
    """
    if levels == ["all"]:
        return torch.cat([downsample_feat(f, out_size) for f in feats], dim=0)
    return torch.cat(
        [downsample_feat(feats[int(l) % num_levels], out_size) for l in levels], dim=0
    )


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


def auprc_score(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """Area under the Precision-Recall curve.
    More sensitive than AUROC for small organs (class imbalance): measures
    whether high-scoring voxels are truly foreground, not just ranked above bg."""
    from sklearn.metrics import average_precision_score
    gt_np   = (gt.cpu().numpy().ravel() > 0).astype(int)
    pred_np = pred.cpu().numpy().ravel()
    if gt_np.sum() == 0 or gt_np.sum() == len(gt_np):
        return float("nan")
    return float(average_precision_score(gt_np, pred_np))


def hard_dice_score(pred: torch.Tensor, gt: torch.Tensor, threshold: float = 0.5) -> float:
    """Binary Dice after thresholding pred at `threshold`."""
    p = (pred >= threshold).float()
    g = (gt   >  0        ).float()
    num = 2 * (p * g).sum()
    den = p.sum() + g.sum()
    if den < 1e-6:
        return float("nan")
    return (num / den).item()


def spearman_score(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """Spearman rank correlation ρ ∈ [-1, 1].
    Fully invariant to any monotone rescaling of pred — answers
    'does the spatial ranking match GT regardless of amplitude?'"""
    from scipy.stats import spearmanr
    if gt.float().std() < 1e-8:
        return float("nan")
    return float(spearmanr(pred.cpu().numpy().ravel(), gt.cpu().numpy().ravel()).statistic)


def ncc_score(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """Normalized cross-correlation ∈ [-1, 1].
    Invariant to linear rescaling and offset of pred — measures
    whether pred and GT co-vary spatially in a linear sense."""
    p = pred.float().ravel()
    g = gt.float().ravel()
    sp, sg = p.std(), g.std()
    if sp < 1e-8 or sg < 1e-8:
        return float("nan")
    return float(((p - p.mean()) * (g - g.mean())).mean() / (sp * sg))


def recall_at_k(pred: torch.Tensor, gt: torch.Tensor, k_frac: float) -> float:
    """Fraction of GT foreground captured by the top-k_frac predicted patches.
    Directly models zone-of-interest selection: 'if I keep the top K% patches
    by predicted score, how much of the GT organ do I cover?'"""
    gt_np   = (gt.cpu().numpy().ravel() > 0).astype(int)
    pred_np = pred.cpu().numpy().ravel()
    n_fg = int(gt_np.sum())
    if n_fg == 0:
        return float("nan")
    k = max(1, int(k_frac * len(pred_np)))
    top_idx = pred_np.argsort()[::-1][:k]
    return float(gt_np[top_idx].sum() / n_fg)


def js_divergence(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """Jensen-Shannon divergence ∈ [0, 1] between spatial probability distributions.
    Both pred and GT are L1-normalised to sum to 1 before comparison — measures
    whether probability mass is placed in the same spatial region."""
    from scipy.spatial.distance import jensenshannon
    p = pred.float().cpu().numpy().ravel()
    g = gt.float().cpu().numpy().ravel()
    p_sum, g_sum = p.sum(), g.sum()
    if p_sum < 1e-8 or g_sum < 1e-8:
        return float("nan")
    return float(jensenshannon(p / p_sum, g / g_sum, base=2))


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
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_size = tuple(args.output_size)

    # ---- Dataset -----------------------------------------------------------
    # args.classes may be a single-token list like ["benchmark"] — resolve to actual names
    classes = resolve_classes(
        args.classes[0] if len(args.classes) == 1 else args.classes,
        totalseg_root=args.data_root,
    )
    ds = TotalSegInContextDataset(
        root=args.data_root,
        classes=classes,
        image_size=tuple(args.image_size),
        split="val",
        context_size=args.context_size,
        max_subjects=args.max_subjects,
        use_crop=args.use_crop,
    )

    # ---- Encoder -----------------------------------------------------------
    if args.encoder == "stunet":
        encoder = STUNetEncoder(
            in_channels=1,
            variant=args.stunet_variant,
            pretrained=args.stunet_pretrained,
            freeze_encoder=True,
        ).to(device).eval()
    elif args.encoder == "nninteractive":
        encoder = NNInteractiveEncoder(
            ckpt_dir=args.nnint_ckpt,
            mask_injection=args.nnint_mask_injection,
            freeze_encoder=True,
            num_stages=args.nnint_num_stages,
            device="cpu",
        ).to(device).eval()
    elif args.encoder == "threedino":
        encoder = ThreeDINOEncoder(
            ckpt_path=args.dino_ckpt,
            n_last_blocks=args.dino_n_blocks,
            freeze_encoder=True,
            compile_model=False,
            device="cpu",
        ).to(device).eval()
    elif args.encoder == "vocomni":
        encoder = VoComniEncoder(
            ckpt_path=args.vocomni_ckpt,
            feature_size=args.vocomni_feature_size,
            freeze_encoder=True,
            compile_model=args.vocomni_compile,
        ).to(device).eval()
    else:  # vocomni_nnunet
        encoder = VoComniNNUNetEncoder(
            ckpt_path=args.vocomni_nnunet_ckpt,
            freeze_encoder=True,
            compile_model=args.vocomni_nnunet_compile,
        ).to(device).eval()

    num_levels = len(encoder.skip_channels) + 1  # skips + bottleneck
    levels = args.feature_level  # list of str; e.g. ["-1"] or ["all"] or ["-1", "-2"]
    if levels == ["all"]:
        level_desc = "all"
    else:
        level_desc = "+".join(str(int(l) % num_levels) for l in levels)
    print(f"Encoder: {args.encoder} | "
          f"Feature level: {level_desc} (of 0–{num_levels - 1}) | "
          f"output_size={out_size} | temperature={args.temperature}\n")

    # ---- W&B ---------------------------------------------------------------
    use_wandb = args.wandb_project and args.wandb_project.lower() != "null"
    if use_wandb:
        import wandb
        _ckpt_tag = ""
        if args.encoder == "vocomni":
            _ckpt_tag = "_" + Path(args.vocomni_ckpt).stem
        auto_name = (
            f"{args.encoder}{_ckpt_tag}_{args.image_size[0]}_{args.output_size[0]}"
            f"_lvl{level_desc}_K{args.context_size}_{args.method}"
        )
        wandb.init(
            project=args.wandb_project,
            name=args.run_name or auto_name,
            config={
                "method":            args.method,
                "encoder":           args.encoder,
                "feature_level":     level_desc,
                "output_size":       args.output_size,
                "temperature":       args.temperature,
                "mask_pool":         args.mask_pool,
                "context_size":      args.context_size,
                "image_size":        args.image_size,
                "use_crop":          args.use_crop,
                "stunet_variant":    args.stunet_variant,
                "stunet_pretrained": str(args.stunet_pretrained),
                "nnint_ckpt":        str(args.nnint_ckpt),
                "nnint_mask_injection": args.nnint_mask_injection,
                "nnint_num_stages":  args.nnint_num_stages,
                "vocomni_ckpt":      str(args.vocomni_ckpt),
                "vocomni_feature_size": args.vocomni_feature_size,
                "vocomni_nnunet_ckpt": str(args.vocomni_nnunet_ckpt),
                "samples_per_class": args.samples_per_class,
                "classes":           args.classes,
            },
        )

    # ---- TabPFN classifier (instantiated once, reused across all samples) ----
    tabpfn_clf = None
    if args.method == "tabpfn":
        from tabpfn import TabPFNClassifier
        _mem_mode: bool | str = {"auto": "auto", "true": True, "false": False}[
            args.tabpfn_memory_saving
        ]
        tabpfn_clf = TabPFNClassifier(
            n_estimators=args.tabpfn_n_estimators,
            device=args.device,
            ignore_pretraining_limits=True,
            memory_saving_mode=_mem_mode,
        )
        print(
            f"TabPFN ready  n_estimators={args.tabpfn_n_estimators}  "
            f"memory_saving_mode={_mem_mode}  device={args.device}\n"
        )

    # ---- Build eval DataLoader (pre-select samples_per_class per class) -----
    _per_cls: dict[str, int] = {}
    eval_indices: list[int] = []
    for i, (_, cls) in enumerate(ds.samples):
        cnt = _per_cls.get(cls, 0)
        if cnt < args.samples_per_class:
            eval_indices.append(i)
            _per_cls[cls] = cnt + 1
    print(f"Evaluating {len(eval_indices)} samples | "
          f"{len(_per_cls)} classes | batch_size={args.batch_size}")

    eval_loader = DataLoader(
        Subset(ds, eval_indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=incontext_collate_fn,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    # ---- Evaluation loop ---------------------------------------------------
    amp     = device.type == "cuda"
    results: list[dict] = []
    fig_idx = 0

    for batch in eval_loader:
        images  = batch["image"].to(device, non_blocking=True)        # (B, 1, D, H, W)
        labels  = batch["label"].to(device, non_blocking=True)         # (B, D, H, W)
        ctx_in  = batch["context_in"].to(device, non_blocking=True)    # (B, K, 1, D, H, W)
        ctx_out = batch["context_out"].to(device, non_blocking=True)   # (B, K, D, H, W)
        subjects    = batch["subjects"]
        label_names = batch["label_names"]
        B, K = ctx_in.shape[:2]

        ctx_imgs_flat  = ctx_in.reshape(B * K, 1, *ctx_in.shape[3:])
        ctx_masks_flat = ctx_out.reshape(B * K, *ctx_out.shape[2:]).unsqueeze(1).float()

        # ── Batch encode: one pass for all B targets, one for all B*K contexts ──
        with torch.autocast(device_type=device.type, enabled=amp):
            if args.encoder == "stunet":
                # Targets and contexts use the same image-only path — combine into one call
                all_imgs   = torch.cat([images, ctx_imgs_flat], dim=0)  # (B + B*K, 1, D, H, W)
                all_feats  = encode_image_only(encoder, all_imgs)
                tgt_feats_b = [f[:B]  for f in all_feats]               # (B, C, d, h, w)
                ctx_feats_b = [f[B:]  for f in all_feats]               # (B*K, C, d, h, w)
            elif args.encoder in ("threedino", "vocomni", "vocomni_nnunet"):
                # Image-only encoders: combine targets + contexts into one batch call.
                all_imgs    = torch.cat([images, ctx_imgs_flat], dim=0)
                all_feats   = encode_image_generic(encoder, all_imgs)
                tgt_feats_b = [f[:B] for f in all_feats]
                ctx_feats_b = [f[B:] for f in all_feats]
            else:  # nninteractive
                tgt_feats_b = encode_target_nnint(encoder, images)
                ctx_feats_b = encode_context_nnint(encoder, ctx_imgs_flat, ctx_masks_flat)

        # ── Per-sample prediction ──────────────────────────────────────────────
        for b in range(B):
            subj = subjects[b]
            cls  = label_names[b]

            tgt_feats    = [f[b : b + 1]                   for f in tgt_feats_b]
            ctx_all_feats = [
                [f[b * K + k : b * K + k + 1] for f in ctx_feats_b]
                for k in range(K)
            ]

            tgt_feat_ds = extract_features(tgt_feats, levels, out_size, num_levels).float()
            ctx_feat_ds = torch.stack([
                extract_features(f, levels, out_size, num_levels) for f in ctx_all_feats
            ]).float()

            label_b   = labels[b]
            ctx_out_b = ctx_out[b]
            tgt_mask_ds = downsample_mask(label_b, out_size, args.mask_pool)
            ctx_mask_ds = torch.stack([
                downsample_mask(ctx_out_b[k], out_size, args.mask_pool) for k in range(K)
            ])

            t0 = time.perf_counter()
            if args.method == "tabpfn":
                pred = predict_tabpfn(tgt_feat_ds, ctx_feat_ds, ctx_mask_ds, tabpfn_clf,
                                      balance_ratio=args.balance_ratio)
            elif args.method == "prototype":
                pred = predict_prototype(tgt_feat_ds, ctx_feat_ds, ctx_mask_ds, args.temperature,
                                         balance_ratio=args.balance_ratio)
            else:
                pred = predict_similarity(tgt_feat_ds, ctx_feat_ds, ctx_mask_ds, args.temperature,
                                          balance_ratio=args.balance_ratio)
            inference_time = time.perf_counter() - t0

            gt_ds   = tgt_mask_ds.float()
            d_soft  = soft_dice_score(pred, gt_ds)
            d_norm  = norm_dice_score(pred, gt_ds)
            auc     = auroc_score(pred, gt_ds)
            aprc    = auprc_score(pred, gt_ds)
            hd      = hard_dice_score(pred, gt_ds)
            rho     = spearman_score(pred, gt_ds)
            ncc     = ncc_score(pred, gt_ds)
            r05     = recall_at_k(pred, gt_ds, 0.05)
            r10     = recall_at_k(pred, gt_ds, 0.10)
            r20     = recall_at_k(pred, gt_ds, 0.20)
            js      = js_divergence(pred, gt_ds)

            pred_fullres = F.interpolate(
                pred.reshape(1, 1, *out_size),
                size=tuple(args.image_size), mode="trilinear", align_corners=False,
            ).squeeze()
            hd_full = hard_dice_score(pred_fullres, label_b.float())

            pred_np = pred.cpu().numpy()

            print(f"[{cls:<30s}] subj={subj}  "
                  f"hd={hd:.3f}  hd_full={hd_full:.3f}  auroc={auc:.3f}  auprc={aprc:.3f}  "
                  f"spearman={rho:.3f}  ncc={ncc:.3f}  "
                  f"r@5={r05:.3f}  r@10={r10:.3f}  r@20={r20:.3f}  js={js:.3f}  "
                  f"pred[{pred_np.min():.3f}…{pred_np.max():.3f}]")
            results.append({
                "soft_dice": d_soft, "norm_dice": d_norm,
                "hard_dice": hd, "hard_dice_full": hd_full,
                "auroc": auc, "auprc": aprc, "spearman": rho, "ncc": ncc,
                "recall_05": r05, "recall_10": r10, "recall_20": r20,
                "js_div": js,
                "class": cls, "subject": subj, "inference_time": inference_time,
            })

            fig_path = out_dir / f"{fig_idx:03d}_{cls}_{subj}.png"
            save_slice_figure(
                tgt_image=batch["image"][b].squeeze(0).cpu().numpy(),
                tgt_gt=label_b.cpu().numpy(),
                tgt_gt_ds=gt_ds.cpu().numpy(),
                pred=pred_np,
                ctx_images=[ctx_in[b, k].squeeze(0).cpu().numpy() for k in range(K)],
                ctx_gts=[ctx_out_b[k].cpu().numpy() for k in range(K)],
                ctx_gts_ds=[ctx_mask_ds[k].cpu().numpy() for k in range(K)],
                out_path=fig_path,
                title=f"{cls} | {subj} | norm_dice={d_norm:.3f}  auroc={auc:.3f}",
            )
            if use_wandb:
                wandb.log({
                    "sample/soft_dice":      d_soft,
                    "sample/norm_dice":      d_norm,
                    "sample/hard_dice":      hd,
                    "sample/hard_dice_full": hd_full,
                    "sample/auroc":          auc,
                    "sample/auprc":          aprc,
                    "sample/spearman":       rho,
                    "sample/ncc":            ncc,
                    "sample/recall_05":      r05,
                    "sample/recall_10":      r10,
                    "sample/recall_20":      r20,
                    "sample/js_div":         js,
                    "sample/inference_time": inference_time,
                    "sample/class":          cls,
                    "sample/subject":        subj,
                    "sample/figure":         wandb.Image(str(fig_path)),
                })
            fig_idx += 1

    # ---- Summary -----------------------------------------------------------
    if results:
        per_cls: dict[str, list[dict]] = defaultdict(list)
        for r in results:
            per_cls[r["class"]].append(r)
        def _m(key):
            return np.nanmean([r[key] for r in results])
        def _mc(key, rs):
            return np.nanmean([r[key] for r in rs])

        print(f"\n{'─'*120}")
        print(f"  {'class':<30s} {'hd':>6} {'hd_f':>6} {'auroc':>7} {'auprc':>7} {'spear':>7} {'ncc':>7} "
              f"{'r@5':>6} {'r@10':>6} {'r@20':>6} {'js↓':>6}  n")
        print(f"  {'─'*116}")
        for cls, rs in per_cls.items():
            print(f"  {cls:<30s} "
                  f"{_mc('hard_dice',      rs):>6.3f} "
                  f"{_mc('hard_dice_full', rs):>6.3f} "
                  f"{_mc('auroc',          rs):>7.3f} "
                  f"{_mc('auprc',          rs):>7.3f} "
                  f"{_mc('spearman',       rs):>7.3f} "
                  f"{_mc('ncc',            rs):>7.3f} "
                  f"{_mc('recall_05',      rs):>6.3f} "
                  f"{_mc('recall_10',      rs):>6.3f} "
                  f"{_mc('recall_20',      rs):>6.3f} "
                  f"{_mc('js_div',         rs):>6.3f}  {len(rs)}")
        print(f"  {'─'*116}")
        print(f"  {'overall':<30s} "
              f"{_m('hard_dice'):>6.3f} "
              f"{_m('hard_dice_full'):>6.3f} "
              f"{_m('auroc'):>7.3f} "
              f"{_m('auprc'):>7.3f} "
              f"{_m('spearman'):>7.3f} "
              f"{_m('ncc'):>7.3f} "
              f"{_m('recall_05'):>6.3f} "
              f"{_m('recall_10'):>6.3f} "
              f"{_m('recall_20'):>6.3f} "
              f"{_m('js_div'):>6.3f}")
        print(f"\n  Figures : {out_dir}/")
        print(f"  avg inference time : {_m('inference_time')*1000:.1f} ms/item")

        if use_wandb:
            summary: dict = {
                "overall/hard_dice":      _m("hard_dice"),
                "overall/hard_dice_full": _m("hard_dice_full"),
                "overall/auroc":          _m("auroc"),
                "overall/auprc":          _m("auprc"),
                "overall/spearman":       _m("spearman"),
                "overall/ncc":            _m("ncc"),
                "overall/recall_05":      _m("recall_05"),
                "overall/recall_10":      _m("recall_10"),
                "overall/recall_20":      _m("recall_20"),
                "overall/js_div":         _m("js_div"),
                "overall/soft_dice":      _m("soft_dice"),
                "overall/norm_dice":      _m("norm_dice"),
                "overall/inference_time": _m("inference_time"),
            }
            for cls, rs in per_cls.items():
                for key in ("hard_dice", "hard_dice_full", "auroc", "auprc", "spearman", "ncc",
                            "recall_05", "recall_10", "recall_20", "js_div"):
                    summary[f"class/{cls}/{key}"] = _mc(key, rs)

            _metric_cols = [
                "hard_dice", "hard_dice_full", "auroc", "auprc", "spearman",
                "ncc", "recall_05", "recall_10", "recall_20", "js_div",
            ]

            # Per-subject table (one row per evaluated sample)
            subj_table = wandb.Table(columns=["class", "subject"] + _metric_cols)
            for r in results:
                subj_table.add_data(
                    r["class"], r.get("subject", ""),
                    *[r[k] for k in _metric_cols],
                )

            # Per-class table (mean metrics per class + overall row)
            cls_table = wandb.Table(columns=["class", "n"] + _metric_cols)
            for cls, rs in per_cls.items():
                cls_table.add_data(cls, len(rs), *[_mc(k, rs) for k in _metric_cols])
            cls_table.add_data("__overall__", len(results), *[_m(k) for k in _metric_cols])

            summary["tables/per_subject"] = subj_table
            summary["tables/per_class"]   = cls_table
            wandb.log(summary)
            wandb.finish()


if __name__ == "__main__":
    main()
