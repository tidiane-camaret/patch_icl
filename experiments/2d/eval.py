"""
Unified 2D evaluation script for MedSegBench.

Dispatches on cfg.model:
  universeg             → full UniverSeg forward (encoder + cross-conv + decoder)
  universeg_featuresim  → UniverSeg encoder features classified by TabPFN
  pfn_seg_2d            → trained ImagePFN checkpoint (arch read from the .pt)
  patchset_pfn          → multilevel coarse→fine chain (frozen stage-1 ImagePFN +
                          frozen UniverSeg encoder + trained PatchSetPFN hops); the
                          final hop's native-resolution composite is the prediction.

Usage:
    python experiments/2d/eval.py                                     # universeg
    python experiments/2d/eval.py --config-name feature_sim           # feature_sim
    python experiments/2d/eval.py data.dataset=abdomenus data.context_size=5
    python experiments/2d/eval.py --config-name feature_sim feature.level=-1 feature.output_size=8
    python experiments/2d/eval.py model=pfn_seg_2d eval.checkpoint=results/2d/pfn_seg/<run>/best.pt
    python experiments/2d/eval.py model=patchset_pfn eval.checkpoint=results/2d/<run>/best.pt \
        eval.stage1_checkpoint=results/2d/pfn_seg_universeg/<run>/best.pt
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

# For the pfn_seg backend, patch_icl's `src` package must win over ic_segmentation's
# shadowing copy (common.py puts the latter on sys.path). Cache patch_icl's src
# package before common imports from it — mirrors pfn_seg.py. Skipped for the
# universeg backends, which rely on ic_segmentation's src.models.universeg_baseline.
if any(("pfn_seg" in _a) or ("patchset_pfn" in _a) for _a in sys.argv):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    import src.datasets.medsegbench  # noqa: F401  (caches patch_icl's src)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DEVICE, build_loader, hard_dice, soft_dice, downsample_mask, log_summary


# ── UniverSeg encoder (feature_sim backend) ───────────────────────────────────

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
        masks:  (B, 1, H, W) binary, optional

    Returns:
        list of 4 tensors, each (B, 64, H/2^i, W/2^i)
    """
    B, _, H, W = images.shape
    target  = images.unsqueeze(1)                                 # (B, 1, 1, H, W)
    dev = images.device
    label_ch = masks.to(dev) if masks is not None \
               else torch.zeros(B, 1, H, W, device=dev)
    dummy_s = torch.cat([
        torch.zeros(B, 1, 1, H, W, device=dev),
        label_ch.unsqueeze(1),
    ], dim=2)                                                     # (B, 1, 2, H, W)

    feats = []
    for i, block in enumerate(useg.enc_blocks):
        target, dummy_s = block(target, dummy_s)
        feats.append(target[:, 0])                                # (B, C, H', W')
        if i < len(useg.enc_blocks) - 1:
            target  = F.max_pool2d(target[:, 0], 2).unsqueeze(1)
            dummy_s = F.max_pool2d(dummy_s[:, 0], 2).unsqueeze(1)

    return feats  # index 0 = highest res, -1 = bottleneck


def extract_features_batch(
    feats: list[torch.Tensor],
    level: str | int,
    output_size: int,
) -> torch.Tensor:
    """
    Pool encoder feature maps to output_size × output_size.

    feats: list of (N, C, H', W') tensors.
    level="all" concatenates all levels on the channel dim.
    Returns (N, C', os, os).
    """
    size = (output_size, output_size)
    if str(level) == "all":
        maps = [F.adaptive_avg_pool2d(f.float(), size) for f in feats]
    else:
        idx = int(level) % len(feats)
        maps = [F.adaptive_avg_pool2d(feats[idx].float(), size)]
    return torch.cat(maps, dim=1)   # (N, C', os, os)


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
    n_bg = min(len(bg_cands), max(1, int(n_eff_fg * bg_ratio)))
    perm = torch.randperm(len(bg_cands), device=bg_cands.device)[:n_bg]
    keep = torch.cat([keep_fg, bg_cands[perm]]).sort().values
    return ctx_flat[keep], labels_flat[keep]


def predict_tabpfn(
    tgt_feat:  torch.Tensor,   # (C, H', W')
    ctx_feats: torch.Tensor,   # (K, C, H', W')
    ctx_masks: torch.Tensor,   # (K, H', W') float
    clf,
    balance_ratio: float | None = None,
) -> torch.Tensor:
    """Fit TabPFN on context patches; predict soft mask for target. Returns (H', W')."""
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
        return torch.full((H, W), float(y_ctx.mean()), dtype=torch.float32)

    mu  = X_ctx.mean(axis=0, keepdims=True)
    sig = X_ctx.std( axis=0, keepdims=True) + 1e-8
    X_ctx = (X_ctx - mu) / sig
    X_tgt = (X_tgt - mu) / sig

    clf.fit(X_ctx, y_ctx)
    proba = clf.predict_proba(X_tgt)
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
    Batched TabPFN: one transformer forward per estimator, all B samples stacked.
    Returns (B, H', W') float on CPU.
    """
    B, C, H, W = tgt_feats.shape
    K  = ctx_feats.shape[1]
    N  = H * W
    dev = next(model.parameters()).device

    X_ctx = ctx_feats.reshape(B, K, C, N).permute(0, 1, 3, 2).reshape(B, K * N, C)
    y_ctx = (ctx_masks.reshape(B, K * N) > 0).float()
    X_tgt = tgt_feats.reshape(B, C, N).permute(0, 2, 1)  # (B, N, C)

    mu  = X_ctx.mean(dim=1, keepdim=True)
    sig = X_ctx.std( dim=1, keepdim=True) + 1e-8
    X_ctx = (X_ctx - mu) / sig
    X_tgt = (X_tgt - mu) / sig

    label_sums = y_ctx.sum(dim=1)
    degenerate = (label_sums == 0) | (label_sums == K * N)
    nd_idx = (~degenerate).nonzero(as_tuple=True)[0]

    preds = torch.zeros(B, H, W)
    for b in degenerate.nonzero(as_tuple=True)[0]:
        preds[b] = float(y_ctx[b].mean())

    if len(nd_idx) > 0:
        X_all = torch.cat([X_ctx[nd_idx], X_tgt[nd_idx]], dim=1).permute(1, 0, 2).to(dev)
        Y_all = y_ctx[nd_idx].permute(1, 0).to(dev)

        proba_sum = None
        with torch.inference_mode():
            for _ in range(n_estimators):
                perm = torch.randperm(C, device=dev)
                out  = model(X_all[:, :, perm], Y_all, only_return_standard_out=True)
                # out: (N_tgt, B_nd, 160) — only test rows, first n_classes cols matter
                logits = out[-N:, :, :n_classes]
                p = torch.softmax(logits, dim=-1)
                proba_sum = p if proba_sum is None else proba_sum + p

        proba = proba_sum / n_estimators                                    # (N, B_nd, 2)
        preds[nd_idx] = proba[..., 1].permute(1, 0).reshape(len(nd_idx), H, W).cpu()

    return preds


def dice_at_native(pred_ds: torch.Tensor, gt_native: torch.Tensor, native_size: int) -> float:
    """Upsample patch-level prediction to native resolution and compute Dice."""
    pred_up = F.interpolate(
        pred_ds.unsqueeze(0).unsqueeze(0).float(),
        size=(native_size, native_size),
        mode="bilinear", align_corners=False,
    ).squeeze()
    return hard_dice(pred_up, gt_native)


def load_stage1(path: str):
    """Load a frozen stage-1 ImagePFN from its checkpoint (arch read from the .pt).

    Mirrors experiments/2d/multilevel/train.py:load_stage1 so the multilevel chain
    eval reconstructs the exact same coarse model. Frozen, eval mode.
    """
    from src.models.pfn_seg_2d import ImagePFN
    from src.models.pretrained_encoders import build_image_encoder
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    arch, img_size = ckpt["arch"], ckpt["image_size"]
    resolution = arch.get("resolution", img_size // arch["patch_size"] if "patch_size" in arch else None)
    input_patch_size = arch.get("input_patch_size", img_size // resolution)
    image_encoder, feature_dim = build_image_encoder(arch, DEVICE)
    model = ImagePFN(resolution=resolution, image_size=img_size,
                     input_patch_size=input_patch_size,
                     image_encoder=image_encoder, feature_dim=feature_dim,
                     e=arch["e"], h=arch["h"], l=arch["l"], a=arch["a"],
                     thinking_rows=arch["thinking_rows"],
                     residual_decay=arch["residual_decay"]).to(DEVICE)
    state = {k.removeprefix("_orig_mod."): v for k, v in ckpt["model"].items()}
    model.load_state_dict(state)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    print(f"Stage-1 loaded: resolution={resolution}, encoder={arch.get('image_encoder','patch')}")
    return model


# ── Visualisation (feature_sim backend) ──────────────────────────────────────

def _overlay_ax(ax, image: np.ndarray, mask: np.ndarray, title: str) -> None:
    ax.imshow(image, cmap="gray", vmin=0, vmax=1)
    ax.imshow(mask,  cmap="Reds", alpha=0.45, vmin=0, vmax=1)
    ax.set_title(title, fontsize=8)
    ax.axis("off")


def _heatmap_ax(ax, arr: np.ndarray, title: str) -> None:
    ax.imshow(arr, cmap="hot", vmin=0, vmax=1, interpolation="nearest")
    ax.set_title(title, fontsize=8)
    ax.axis("off")


def save_figure(
    tgt_image:  np.ndarray,
    tgt_gt:     np.ndarray,
    tgt_gt_ds:  np.ndarray,
    pred_ds:    np.ndarray,
    ctx_images: list[np.ndarray],
    ctx_gts:    list[np.ndarray],
    ctx_gts_ds: list[np.ndarray],
    out_path:   Path,
    title:      str = "",
) -> None:
    """Row 0: target+GT | GT@output_size | pred.  Row 1: context images+GTs."""
    K     = len(ctx_images)
    ncols = max(3, 2 * K)
    span  = ncols // 3

    fig, axes = plt.subplots(2, ncols, figsize=(3.2 * ncols, 6.5))
    fig.subplots_adjust(hspace=0.35, wspace=0.05)

    _overlay_ax(axes[0, 0],        tgt_image, tgt_gt, "Target + GT")
    _heatmap_ax(axes[0, span],     tgt_gt_ds,          f"GT ↓{tgt_gt_ds.shape[0]}")
    _heatmap_ax(axes[0, 2 * span], pred_ds,            "Prediction")
    for col in (list(range(1, span))
                + list(range(span + 1, 2 * span))
                + list(range(2 * span + 1, ncols))):
        axes[0, col].axis("off")

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

@hydra.main(config_path="../../configs/experiment/2d", config_name="base", version_base=None)
def main(cfg: DictConfig):
    random.seed(cfg.eval.seed)
    torch.manual_seed(cfg.eval.seed)
    if DEVICE.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    is_feat_sim   = cfg.model == "universeg_featuresim"
    is_pfn_seg    = cfg.model == "pfn_seg_2d"
    is_multilevel = cfg.model == "patchset_pfn"

    # pfn_seg / patchset_pfn: arch + image_size live in the checkpoint. The model is
    # built at the checkpoint's `model_size` (token grids are baked in); the loader
    # serves images at `encode_size`. By default the two are equal. Setting
    # eval.encode_size feeds a *different* resolution while keeping the grids fixed
    # ("Strategy A"): only the frozen conv encoder sees the new size, then pools into
    # the same grids; the stage-1 patch size is rescaled and the output is upsampled
    # to native for scoring. encode_size must be divisible by the stage-1 resolution.
    pfn_ckpt = None
    model_size = None
    if is_pfn_seg or is_multilevel:
        from omegaconf import open_dict
        pfn_ckpt = torch.load(cfg.eval.checkpoint, map_location="cpu", weights_only=False)
        model_size  = pfn_ckpt["image_size"]
        encode_size = cfg.eval.get("encode_size", None) or model_size
        if encode_size != model_size:
            print(f"Strategy-A eval: model grids fixed at {model_size}px, but encoding "
                  f"images at {encode_size}px (conv encoder runs at the new size; output "
                  f"upsampled to {encode_size} for scoring).")
        with open_dict(cfg):
            cfg.data.image_size = encode_size
            if is_multilevel and "context_size" in pfn_ckpt:
                cfg.data.context_size = pfn_ckpt["context_size"]

    loader = build_loader(cfg)

    from torch.utils.flop_counter import FlopCounterMode

    # ── model setup ───────────────────────────────────────────────────────────
    if is_feat_sim:
        from src.models.universeg_baseline import UniverSegBaseline
        print("Loading UniverSeg encoder...")
        wrapper = UniverSegBaseline(pretrained=True, input_size=cfg.data.image_size)
        useg    = wrapper.model.to(DEVICE).eval()
        del wrapper
        torch.cuda.empty_cache()

        _n = 1 + cfg.data.context_size
        _dummy = torch.zeros(_n, 1, cfg.data.image_size, cfg.data.image_size, device=DEVICE)
        with FlopCounterMode(display=False) as _fc:
            with torch.no_grad():
                encode_images(useg, _dummy)
        flops = _fc.get_total_flops()
        del _dummy
        print(f"Encoder FLOPs (1+{cfg.data.context_size} imgs, {cfg.data.image_size}²): {flops/1e9:.2f} GFLOPs")

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
        _C = 4 * 64 if str(cfg.feature.level) == "all" else 64
        _rng = np.random.default_rng(0)
        clf.fit(_rng.standard_normal((4, _C)).astype(np.float32), np.array([0, 0, 1, 1]))
        tabpfn_model = clf.models_[0]
        tabpfn_n_est = cfg.tabpfn.n_estimators
        print(f"TabPFN ready  n_estimators={tabpfn_n_est}  (batched inference)")
        if cfg.tabpfn.balance_ratio is not None:
            print("WARNING: balance_ratio is set — batched TabPFN is disabled; inference will be slower.")

        level_tag = str(cfg.feature.level).replace("-", "m")
        run_name  = cfg.wandb.name or (
            f"{cfg.model}_lvl{level_tag}_os{cfg.feature.output_size}"
            f"_s{cfg.data.image_size}_k{cfg.data.context_size}"
        )
        run_cfg = {
            "model":         cfg.model,
            "source":        cfg.data.get("source", "medsegbench"),
            # encoder_input_size = served pixel resolution; model_size = the resolution
            # the model's grids were built at. Equal here (no Strategy-A split for this
            # backend), but kept as distinct keys so runs are comparable across backends.
            "encoder_input_size": cfg.data.image_size,
            "model_size":         cfg.data.image_size,
            "context_size":  cfg.data.context_size,
            "split":         cfg.data.split,
            "feature_level": str(cfg.feature.level),
            "output_size":   cfg.feature.output_size,
            "mask_pool":     cfg.feature.mask_pool,
            "n_estimators":  cfg.tabpfn.n_estimators,
            "balance_ratio": cfg.tabpfn.balance_ratio,
            "memory_saving": cfg.tabpfn.memory_saving,
            "flops_encoder": flops,
        }

    elif is_pfn_seg:
        # pfn_seg_2d lives in patch_icl's src, but common.py puts ic_segmentation
        # (which has its own shadowing src/) ahead on sys.path, so a plain
        # `import src.models.pfn_seg_2d` resolves to the wrong package. The module
        # only depends on torch, so load it directly by file path.
        import importlib.util
        _pfn_path = Path(__file__).resolve().parents[2] / "src" / "models" / "pfn_seg_2d.py"
        _spec = importlib.util.spec_from_file_location("pfn_seg_2d", _pfn_path)
        _pfn_mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_pfn_mod)
        ImagePFN = _pfn_mod.ImagePFN
        arch = pfn_ckpt["arch"]
        img_size = pfn_ckpt["image_size"]
        # New checkpoints store `resolution` (+ `input_patch_size`); old ones store
        # `patch_size`. Derive the new params from the old field for back-compat.
        resolution = arch.get("resolution", img_size // arch["patch_size"]
                              if "patch_size" in arch else None)
        input_patch_size = arch.get("input_patch_size", img_size // resolution)
        # Rebuild the frozen image encoder if the checkpoint used one (injected,
        # mirroring training). Encoder weights are also in the state_dict, so the
        # subsequent strict load just overwrites these fresh (identical) weights.
        from src.models.pretrained_encoders import build_image_encoder
        image_encoder, feature_dim = build_image_encoder(arch, DEVICE)
        print(f"Loading ImagePFN from {cfg.eval.checkpoint} "
              f"(size={img_size}, resolution={resolution}, Q={input_patch_size}, "
              f"encoder={arch.get('image_encoder', 'patch')})...")
        model = ImagePFN(
            resolution       = resolution,
            image_size       = img_size,
            input_patch_size = input_patch_size,
            image_encoder    = image_encoder,
            feature_dim      = feature_dim,
            e              = arch["e"],
            h              = arch["h"],
            l              = arch["l"],
            a              = arch["a"],
            thinking_rows  = arch["thinking_rows"],
            residual_decay = arch["residual_decay"],
        ).to(DEVICE)
        state = {k.removeprefix("_orig_mod."): v for k, v in pfn_ckpt["model"].items()}
        model.load_state_dict(state)
        model.eval()

        # Strategy A: feed encode_size pixels while keeping the resolution×resolution
        # token grid. Rescale the effective patch size so Hp = H//P stays == resolution
        # (otherwise the patch count != self.N and the forward reshape crashes). Only
        # the conv encoder sees the new size; output (Hp grid) is upsampled in the loop.
        if cfg.data.image_size != model_size:
            assert cfg.data.image_size % resolution == 0, \
                f"encode_size={cfg.data.image_size} must be divisible by resolution {resolution}"
            model.patch_size = cfg.data.image_size // resolution

        _n = cfg.data.context_size
        _imgs = torch.zeros(1, _n + 1, 1, cfg.data.image_size, cfg.data.image_size, device=DEVICE)
        _msks = torch.zeros_like(_imgs)
        with FlopCounterMode(display=False) as _fc:
            with torch.no_grad():
                model(_imgs, _msks, sep=_n)
        flops = _fc.get_total_flops()
        del _imgs, _msks
        print(f"FLOPs (K={cfg.data.context_size}, {cfg.data.image_size}²): {flops/1e9:.2f} GFLOPs")

        # Name encodes the model grid; append the encoder input size only when it
        # differs (Strategy A), e.g. pfn_seg_2d_s128e256_k3.
        size_tag = (f"s{model_size}" if cfg.data.image_size == model_size
                    else f"s{model_size}e{cfg.data.image_size}")
        run_name = cfg.wandb.name or f"{cfg.model}_{size_tag}_k{cfg.data.context_size}"
        run_cfg  = {
            "model":        cfg.model,
            "source":       cfg.data.get("source", "medsegbench"),
            "encoder_input_size": cfg.data.image_size,  # served pixels (== model_size unless Strategy A)
            "model_size":         model_size,            # resolution the token grid was built at
            "context_size": cfg.data.context_size,
            "split":        cfg.data.split,
            "checkpoint":   str(cfg.eval.checkpoint),
            "arch":         dict(arch),
            "flops":        flops,
        }

    elif is_multilevel:
        # Multilevel coarse→fine chain: frozen stage-1 ImagePFN + frozen UniverSeg
        # encoder + trained PatchSetPFN hops. arch + sample (resolutions/budgets)
        # come from the checkpoint; inject them into cfg so run_chain/_level_cfg read
        # the exact training-time config. The prediction is the final hop's
        # native-resolution composite (== training's dice_r{native}/mean).
        import torch.nn as nn
        from omegaconf import OmegaConf
        from src.models.patchset_pfn import PatchSetPFN
        from src.models.pretrained_encoders import build_image_encoder
        sys.path.insert(0, str(Path(__file__).resolve().parent / "multilevel"))
        from pipeline import run_chain

        arch = pfn_ckpt["arch"]
        with open_dict(cfg):
            cfg.arch   = OmegaConf.create(dict(arch))
            cfg.sample = OmegaConf.create(dict(pfn_ckpt["sample"]))

        # Frozen stage-1: path stored in the checkpoint (new runs) else from config.
        stage1_path = pfn_ckpt.get("stage1_checkpoint") or cfg.eval.get("stage1_checkpoint", None)
        if not stage1_path:
            raise ValueError(
                "patchset_pfn eval needs the frozen stage-1 ImagePFN checkpoint, but the "
                "best.pt does not record one. Pass eval.stage1_checkpoint=<path/to/stage1/best.pt>.")
        stage1  = load_stage1(stage1_path)
        # Chain encoder matches training: defaults to UniverSeg unless arch.image_encoder set.
        encoder, feature_dim = build_image_encoder(
            {"image_encoder": cfg.arch.get("image_encoder", "universeg"),
             "feature_level": cfg.arch.feature_level,
             "encoder_resize_to_input": cfg.arch.get("encoder_resize_to_input", False),
             "encoder_imagenet_norm": cfg.arch.get("encoder_imagenet_norm", True),
             "encoder_reduce": cfg.arch.get("encoder_reduce", "none"),
             "encoder_stage_l2norm": cfg.arch.get("encoder_stage_l2norm", False)}, DEVICE)
        # A PCA reduction is loaded from the disk cache written at training time
        # (the chain encoder's buffers are not in the patchset_pfn checkpoint).
        if getattr(encoder, "needs_pca_fit", False):
            def _img_iter():
                for batch in loader:
                    if batch is None:
                        continue
                    img = batch["image"].to(DEVICE)
                    ctx = batch["context_in"].to(DEVICE)
                    yield torch.cat([ctx.flatten(0, 1), img], dim=0)
            encoder.ensure_pca(_img_iter(), fit_out_size=int(cfg.sample.resolutions[1]))
        stage1_dim  = stage1.thinking.tokens.shape[-1] if cfg.arch.use_stage1_thinking else None

        resolutions = list(cfg.sample.resolutions)
        stage1_res = int(round(stage1.N ** 0.5))
        assert resolutions[0] == stage1_res, \
            f"resolutions[0]={resolutions[0]} must equal stage-1 res {stage1_res}"
        # Token grids (and the baked mask-tile size) follow model_size, NOT the served
        # image size — so the chain is unchanged when encode_size differs (Strategy A).
        model = nn.ModuleList([
            PatchSetPFN(feature_dim=feature_dim, e=cfg.arch.e, h=cfg.arch.h, l=cfg.arch.l,
                        a=cfg.arch.a, thinking_rows=cfg.arch.thinking_rows,
                        residual_decay=cfg.arch.residual_decay, fourier_bands=cfg.arch.fourier_bands,
                        mask_prior=cfg.arch.mask_prior,
                        mask_patch_size=model_size // grid,
                        stage1_dim=(stage1_dim if L == 0 else cfg.arch.e),
                        query_self_attn=cfg.arch.query_self_attn).to(DEVICE)
            for L, grid in enumerate(resolutions[1:])])
        # `replace` (not `removeprefix`): a compiled ModuleList buries _orig_mod. mid-key.
        sd = {k.replace("_orig_mod.", ""): v for k, v in pfn_ckpt["model"].items()}
        model.load_state_dict(sd)
        for m in model:
            m.eval()
        print(f"Loaded PatchSetPFN chain: {len(model)} hops, ladder={resolutions}")

        # Strategy A: rescale the frozen stage-1's effective patch size so its coarse
        # seed survives the larger input (Hp = H//P stays == stage-1 resolution, so the
        # token count == self.N). Only the conv encoder then sees encode_size.
        if cfg.data.image_size != model_size:
            assert cfg.data.image_size % stage1_res == 0, \
                f"encode_size={cfg.data.image_size} must be divisible by stage-1 res {stage1_res}"
            stage1.patch_size = cfg.data.image_size // stage1_res

        # FLOPs: full chain (stage-1 + encoder + all hops) on a dummy batch.
        K = cfg.data.context_size
        Himg = cfg.data.image_size
        _db = {"image":       torch.zeros(1, 1, Himg, Himg, device=DEVICE),
               "context_in":  torch.zeros(1, K, 1, Himg, Himg, device=DEVICE),
               "context_out": torch.zeros(1, K, 1, Himg, Himg, device=DEVICE),
               "label":       torch.zeros(1, 1, Himg, Himg, device=DEVICE)}
        with FlopCounterMode(display=False) as _fc:
            with torch.no_grad():
                run_chain(_db, stage1, encoder, model, cfg, cfg.sample.eval,
                          stochastic=False, device=DEVICE)
        flops = _fc.get_total_flops()
        del _db
        print(f"FLOPs (K={K}, {Himg}²): {flops/1e9:.2f} GFLOPs")

        # Name encodes the model grid; append the encoder input size only when it
        # differs (Strategy A), e.g. patchset_pfn_s128e256_k3.
        size_tag = f"s{model_size}" if Himg == model_size else f"s{model_size}e{Himg}"
        run_name = cfg.wandb.name or f"{cfg.model}_{size_tag}_k{K}"
        run_cfg  = {
            "model":        cfg.model,
            "source":       cfg.data.get("source", "medsegbench"),
            "encoder_input_size": Himg,         # served pixels (== model_size unless Strategy A)
            "model_size":         model_size,   # final-hop grid the chain was built at
            "context_size": K,
            "split":        cfg.data.split,
            "checkpoint":   str(cfg.eval.checkpoint),
            "stage1_checkpoint": str(stage1_path),
            "arch":         dict(arch),
            "sample":       dict(pfn_ckpt["sample"]),
            "flops":        flops,
        }

    else:
        from src.models.universeg_baseline import UniverSegBaseline
        print(f"Loading UniverSeg (size={cfg.data.image_size})...")
        model = UniverSegBaseline(pretrained=True, input_size=cfg.data.image_size).to(DEVICE).eval()

        _img = torch.zeros(1, 1, cfg.data.image_size, cfg.data.image_size, device=DEVICE)
        _ctx_in  = torch.zeros(1, cfg.data.context_size, 1, cfg.data.image_size, cfg.data.image_size, device=DEVICE)
        _ctx_out = torch.zeros(1, cfg.data.context_size, 1, cfg.data.image_size, cfg.data.image_size, device=DEVICE)
        with FlopCounterMode(display=False) as _fc:
            with torch.no_grad():
                model(_img, context_in=_ctx_in, context_out=_ctx_out, mode="val")
        flops = _fc.get_total_flops()
        del _img, _ctx_in, _ctx_out
        print(f"FLOPs (S={cfg.data.context_size}, {cfg.data.image_size}²): {flops/1e9:.2f} GFLOPs")

        run_name = cfg.wandb.name or f"{cfg.model}_s{cfg.data.image_size}_k{cfg.data.context_size}"
        run_cfg  = {
            "model":        cfg.model,
            "source":       cfg.data.get("source", "medsegbench"),
            # UniverSeg is rebuilt at the served size, so model_size == encoder_input_size.
            "encoder_input_size": cfg.data.image_size,
            "model_size":         cfg.data.image_size,
            "context_size": cfg.data.context_size,
            "split":        cfg.data.split,
            "flops":        flops,
        }

    # ── wandb ─────────────────────────────────────────────────────────────────
    run = wandb.init(project=cfg.wandb.project, name=run_name, config=run_cfg)
    wandb.log({"flops_giga": flops / 1e9})
    sample_table = wandb.Table(columns=["dataset", "sample_idx", "label",
                                         "dice_ds", "dice_native", "dice_ds_soft"])
    out_dir = Path(cfg.eval.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── eval loop ─────────────────────────────────────────────────────────────
    per_ds:    dict[str, list[float]] = defaultdict(list)
    per_label: dict[str, list[float]] = defaultdict(list)
    per_ds_ds:    dict[str, list[float]] = defaultdict(list)  # downsampled (low-res) hard dice
    per_label_ds: dict[str, list[float]] = defaultdict(list)
    per_ds_ds_soft:    dict[str, list[float]] = defaultdict(list)  # low-res soft (shape) dice
    per_label_ds_soft: dict[str, list[float]] = defaultdict(list)
    encode_times:    list[float] = []
    tabpfn_times:    list[float] = []
    inference_times: list[float] = []
    saved_figures:   set[tuple[str, int]] = set()
    # per-patch error analysis (pfn_seg only, opt-in via eval.patch_csv)
    patch_csv = cfg.eval.get("patch_csv", None) if is_pfn_seg else None
    patch_rows: list[tuple] | None = [] if patch_csv else None

    with torch.no_grad():
        for batch in tqdm(loader, desc="eval"):
            if batch is None:
                continue

            B = len(batch["dataset"])
            images      = batch["image"].to(DEVICE, non_blocking=True)       # (B, 1, H, W)
            labels      = batch["label"]                                      # (B, 1, H, W) CPU
            context_in  = batch["context_in"].to(DEVICE, non_blocking=True)  # (B, K, 1, H, W)
            context_out = batch["context_out"]                                # (B, K, 1, H, W) CPU
            K = context_in.shape[1]
            H, W = images.shape[-2], images.shape[-1]

            if is_feat_sim:
                ctx_imgs_flat = context_in.reshape(B * K, 1, H, W)

                t0 = time.perf_counter()
                with torch.autocast(device_type=DEVICE.type, enabled=DEVICE.type == "cuda"):
                    if not cfg.feature.context_mask:
                        all_feats   = encode_images(useg, torch.cat([images, ctx_imgs_flat], dim=0))
                        tgt_feats_b = [f[:B] for f in all_feats]
                        ctx_feats_b = [f[B:] for f in all_feats]
                    else:
                        tgt_feats_b   = encode_images(useg, images)
                        ctx_masks_enc = context_out.reshape(B * K, 1, H, W).to(DEVICE, non_blocking=True)
                        ctx_feats_b   = encode_images(useg, ctx_imgs_flat, masks=ctx_masks_enc)
                encode_times.append((time.perf_counter() - t0) / B)

                os_ = cfg.feature.output_size
                tgt_feats_all = extract_features_batch(tgt_feats_b, cfg.feature.level, os_)
                ctx_feats_raw = extract_features_batch(ctx_feats_b, cfg.feature.level, os_)
                C_feat = tgt_feats_all.shape[1]
                ctx_feats_all = ctx_feats_raw.reshape(B, K, C_feat, os_, os_)

                _pool = F.adaptive_max_pool2d if cfg.feature.mask_pool == "max" \
                        else F.adaptive_avg_pool2d
                ctx_masks_all = _pool(
                    context_out.reshape(B * K, 1, H, W).float(), (os_, os_)
                ).squeeze(1).reshape(B, K, os_, os_)

                t1 = time.perf_counter()
                if cfg.tabpfn.balance_ratio is None:
                    preds_all = batch_tabpfn(tgt_feats_all, ctx_feats_all, ctx_masks_all,
                                             tabpfn_model, tabpfn_n_est)
                else:
                    preds_all = torch.stack([
                        predict_tabpfn(tgt_feats_all[b], ctx_feats_all[b], ctx_masks_all[b],
                                       clf, cfg.tabpfn.balance_ratio)
                        for b in range(B)
                    ])
                tabpfn_times.append((time.perf_counter() - t1) / B)

            elif is_pfn_seg:
                # Stack context + query: (B, K+1, 1, H, W); query mask is zeros
                # (model fills it with the context-mask mean internally).
                all_images = torch.cat([context_in, images.unsqueeze(1)], dim=1)
                all_masks  = torch.cat([
                    context_out.to(DEVICE, non_blocking=True),
                    torch.zeros_like(images.unsqueeze(1)),
                ], dim=1)

                t0 = time.perf_counter()
                with torch.autocast(device_type=DEVICE.type, dtype=torch.bfloat16,
                                    enabled=DEVICE.type == "cuda"):
                    logits = model(all_images, all_masks, sep=K)   # (B, Hp, Hp)
                Hp = logits.shape[-1]
                preds_lowres = torch.sigmoid(logits.float()).cpu()  # (B, Hp, Hp)
                if Hp != H:
                    preds = F.interpolate(preds_lowres.unsqueeze(1), size=(H, W),
                                          mode="bilinear", align_corners=False).squeeze(1)
                else:
                    preds = preds_lowres                            # (B, H, W)
                inference_times.append((time.perf_counter() - t0) / B)

            elif is_multilevel:
                # Full coarse→fine chain; final hop's composite is at native res.
                t0 = time.perf_counter()
                with torch.autocast(device_type=DEVICE.type, dtype=torch.bfloat16,
                                    enabled=DEVICE.type == "cuda"):
                    outputs, _ = run_chain(batch, stage1, encoder, model, cfg,
                                           cfg.sample.eval,
                                           stochastic=not cfg.sample.eval_deterministic,
                                           device=DEVICE)
                Hg = resolutions[-1]                                 # final-hop grid (model_size)
                preds = outputs[-1]["refined_grid"].reshape(B, Hg, Hg)
                if Hg != H:   # Strategy A: model grid < served/native size → upsample
                    preds = F.interpolate(preds.unsqueeze(1), size=(H, W),
                                          mode="bilinear", align_corners=False).squeeze(1)
                preds = preds.float().cpu()
                inference_times.append((time.perf_counter() - t0) / B)

            else:
                t0 = time.perf_counter()
                with torch.autocast(device_type=DEVICE.type, enabled=DEVICE.type == "cuda"):
                    out = model(images, context_in=context_in,
                                context_out=context_out.to(DEVICE, non_blocking=True), mode="val")
                preds = (out["final_logit"] > 0).float().cpu()   # (B, 1, H, W)
                inference_times.append((time.perf_counter() - t0) / B)

            for b in range(B):
                ds_name     = batch["dataset"][b]
                sample_idx  = int(batch["sample_idx"][b])
                label_value = int(batch["label_value"][b])
                label       = labels[b, 0]

                if is_feat_sim:
                    pred_ds  = preds_all[b]
                    d_ds     = hard_dice(pred_ds, downsample_mask(label, os_))
                    d_native = dice_at_native(pred_ds, label, cfg.data.image_size)
                    # shape: continuous low-res pred vs soft (avg-pooled, un-binarized) GT
                    d_ds_soft = soft_dice(pred_ds, downsample_mask(label, os_))

                    fig_key = (ds_name, label_value)
                    if fig_key not in saved_figures:
                        saved_figures.add(fig_key)
                        fig_path = out_dir / f"{ds_name}_l{label_value}.png"
                        save_figure(
                            tgt_image  = images[b, 0].cpu().numpy(),
                            tgt_gt     = label.cpu().numpy(),
                            tgt_gt_ds  = downsample_mask(label, os_, cfg.feature.mask_pool).cpu().numpy(),
                            pred_ds    = pred_ds.cpu().numpy(),
                            ctx_images = [context_in[b, k, 0].cpu().numpy() for k in range(K)],
                            ctx_gts    = [context_out[b, k, 0].cpu().numpy() for k in range(K)],
                            ctx_gts_ds = [
                                downsample_mask(context_out[b, k, 0], os_, cfg.feature.mask_pool).cpu().numpy()
                                for k in range(K)
                            ],
                            out_path = fig_path,
                            title    = (f"{ds_name}  label={label_value}  sample={sample_idx}"
                                        f"  dice_native={d_native:.3f}"),
                        )
                        wandb.log({f"figures/{ds_name}/label_{label_value}": wandb.Image(str(fig_path))})
                elif is_pfn_seg:
                    d_native = hard_dice(preds[b], label)
                    # Binarize the avg-pooled GT at >= 0.5 (majority vote) so the
                    # low-res target isn't OR-dilated by partially-covered cells.
                    d_ds     = hard_dice(preds_lowres[b], (downsample_mask(label, Hp) >= 0.5).float())
                    # shape: continuous low-res sigmoid map vs soft (un-binarized) GT
                    d_ds_soft = soft_dice(preds_lowres[b], downsample_mask(label, Hp))

                    if patch_rows is not None:
                        # one row per low-res patch: pred, soft GT, signed error,
                        # plus per-sample gt_size and mean target↔context Dice.
                        pred_p = preds_lowres[b].numpy()                  # (Hp, Hp) sigmoid
                        gt_p   = downsample_mask(label, Hp).numpy()       # (Hp, Hp) soft frac
                        err_p  = pred_p - gt_p
                        gt_size = float((label > 0).sum())                # native fg pixels
                        ctx_d = [hard_dice(label, context_out[b, k, 0]) for k in range(K)]
                        ctx_dice = float(np.nanmean(ctx_d)) if ctx_d else float("nan")
                        for i in range(Hp):
                            for j in range(Hp):
                                patch_rows.append((
                                    ds_name, label_value, sample_idx, i, j,
                                    float(pred_p[i, j]), float(gt_p[i, j]),
                                    float(err_p[i, j]), gt_size, ctx_dice,
                                ))
                elif is_multilevel:
                    # Final hop composite is already native-res; same metric as the
                    # other models. (Low-res variants not tracked — final dice only.)
                    d_ds = d_native = hard_dice(preds[b], label)
                    d_ds_soft = soft_dice(preds[b], label)
                else:
                    d_ds = d_native = hard_dice(preds[b, 0], label)
                    # universeg has no low-res map; binary preds → equals hard dice
                    d_ds_soft = soft_dice(preds[b, 0], label)

                per_ds[ds_name].append(d_native)
                per_label[f"{ds_name}/label_{label_value}"].append(d_native)
                per_ds_ds[ds_name].append(d_ds)
                per_label_ds[f"{ds_name}/label_{label_value}"].append(d_ds)
                per_ds_ds_soft[ds_name].append(d_ds_soft)
                per_label_ds_soft[f"{ds_name}/label_{label_value}"].append(d_ds_soft)
                sample_table.add_data(ds_name, sample_idx, label_value, d_ds, d_native, d_ds_soft)

    # ── aggregate & log ───────────────────────────────────────────────────────
    if is_feat_sim:
        mean_enc = float(np.mean(encode_times)) if encode_times else float("nan")
        mean_pfn = float(np.mean(tabpfn_times)) if tabpfn_times else float("nan")
        print(f"\n  avg encode:  {mean_enc * 1000:.1f} ms/item")
        print(f"  avg tabpfn:  {mean_pfn * 1000:.1f} ms/item")
        print(f"  avg total:   {(mean_enc + mean_pfn) * 1000:.1f} ms/item")
        extra = {
            "time/encode_ms": mean_enc * 1000,
            "time/tabpfn_ms": mean_pfn * 1000,
            "time/total_ms":  (mean_enc + mean_pfn) * 1000,
        }
    else:
        mean_t = float(np.mean(inference_times)) if inference_times else float("nan")
        print(f"\n  avg inference: {mean_t * 1000:.1f} ms/item")
        extra = {
            "time/inference_ms": mean_t * 1000,
            "time/total_ms":     mean_t * 1000,
        }

    summary = log_summary(per_ds, per_label, sample_table, extra=extra)
    summary.update(log_summary(per_ds_ds, per_label_ds,
                               prefix="dice_ds", metric_label="downsampled"))
    summary.update(log_summary(per_ds_ds_soft, per_label_ds_soft,
                               prefix="dice_ds_soft", metric_label="low-res soft/shape"))
    wandb.log(summary)

    if patch_rows is not None:
        import csv
        csv_path = Path(patch_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["dataset", "label_value", "sample_idx", "patch_i", "patch_j",
                        "pred", "gt", "error", "gt_size", "ctx_dice"])
            w.writerows(patch_rows)
        print(f"Wrote {len(patch_rows)} patch records to {csv_path}")

    run.finish()


if __name__ == "__main__":
    main()
