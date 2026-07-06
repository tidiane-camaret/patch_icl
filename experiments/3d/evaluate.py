"""
Shared 3D in-context eval loop — the single source of truth used by
experiments/3d/eval.py (now) and experiments/3d/train.py's val step (later).

`validate(model, loader, cls)` runs one single-class loader through
`model.predict()` and returns a per-class summary row + per-case records, plus
optional qualitative figures. Mirrors experiments/2d/evaluate.py's role.

Ported from scripts/eval.py so the 3D experiments harness is self-contained;
scripts/eval.py stays as the legacy CLI benchmark.
"""

import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode


# ---------------------------------------------------------------------------
# Metrics + figures
# ---------------------------------------------------------------------------

def dice_binary(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Smooth Dice between two binary tensors of any shape."""
    pred, target = pred.bool(), target.bool()
    inter = (pred & target).sum().item()
    union = pred.sum().item() + target.sum().item()
    return (2 * inter + 1) / (union + 1)


def _best_slice(mask: np.ndarray) -> int:
    counts = mask.sum(axis=(1, 2))
    return int(counts.argmax()) if counts.max() > 0 else mask.shape[0] // 2


def save_eval_figure(target_img, gt, pred, ctx_img, ctx_gt, out_path: Path, title: str = "") -> None:
    """Save a 4-panel figure: context | target | GT overlay | pred overlay."""
    def _norm(sl):
        mn, mx = sl.min(), sl.max()
        return (sl - mn) / (mx - mn + 1e-6)

    z, z_ctx = _best_slice(gt), _best_slice(ctx_gt)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), gridspec_kw={"wspace": 0.04})
    axes[0].imshow(_norm(ctx_img[z_ctx]), cmap="gray")
    axes[0].imshow(ctx_gt[z_ctx].astype(float), cmap="Reds", alpha=0.45, vmin=0, vmax=1)
    axes[0].set_title("context", fontsize=8)
    axes[1].imshow(_norm(target_img[z]), cmap="gray")
    axes[1].set_title("target", fontsize=8)
    axes[2].imshow(_norm(target_img[z]), cmap="gray")
    axes[2].imshow(gt[z].astype(float), cmap="Reds", alpha=0.45, vmin=0, vmax=1)
    axes[2].set_title("GT", fontsize=8)
    axes[3].imshow(_norm(target_img[z]), cmap="gray")
    axes[3].imshow(pred[z].astype(float), cmap="Blues", alpha=0.45, vmin=0, vmax=1)
    axes[3].set_title("pred", fontsize=8)
    for ax in axes:
        ax.axis("off")
    fig.suptitle(title, fontsize=9)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def measure_flops(model, image_size: tuple, K: int, device: torch.device) -> float:
    """GFLOPs for one predict() call with a single-sample dummy input (0.0 on failure)."""
    D, H, W = image_size
    dummy_target  = torch.zeros(1, 1, D, H, W, device=device)
    dummy_ctx_img = torch.zeros(1, K, 1, D, H, W, device=device)
    dummy_ctx_msk = torch.zeros(1, K, D, H, W, dtype=torch.long, device=device)
    try:
        with FlopCounterMode(display=False) as fc:
            model.predict(dummy_target, dummy_ctx_img, dummy_ctx_msk)
        return fc.get_total_flops() / 1e9
    except Exception as exc:  # noqa: BLE001
        print(f"    [FLOPs] Could not count: {exc}")
        return 0.0


# ---------------------------------------------------------------------------
# Per-class eval loop
# ---------------------------------------------------------------------------

def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def validate(model, loader, cls: str, *, fig_dir: Path | None = None) -> tuple[dict, list[dict]]:
    """Run inference over one single-class loader.

    Returns (summary_row, cases): summary_row aggregates mean/std Dice + mean time;
    cases is a list of {class, subject, dice, time_ms}. Saves one figure per class
    to fig_dir (first batch) when provided. Uses model.predict().
    """
    cases: list[dict] = []
    fig_saved = False

    for batch in loader:
        target_img    = batch["image"]
        context_imgs  = batch["context_in"]
        context_masks = batch["context_out"]
        label         = batch["label"]
        subjects      = batch.get("subjects", [None] * target_img.shape[0])

        _sync()
        t0 = time.perf_counter()
        pred = model.predict(target_img, context_imgs, context_masks)
        _sync()
        elapsed_ms = (time.perf_counter() - t0) * 1000

        pred, label = pred.cpu(), label.cpu()

        if fig_dir is not None and not fig_saved:
            subj = subjects[0] or "s0"
            save_eval_figure(
                target_img=target_img[0].squeeze(0).numpy(),
                gt=label[0].numpy(),
                pred=pred[0].numpy(),
                ctx_img=context_imgs[0, 0].squeeze(0).numpy(),
                ctx_gt=context_masks[0, 0].numpy(),
                out_path=fig_dir / f"{cls}_{subj}.png",
                title=f"{cls}  {subj}  dice={dice_binary(pred[0], label[0]):.3f}",
            )
            fig_saved = True

        for i in range(pred.shape[0]):
            cases.append({
                "class":   cls,
                "subject": subjects[i],
                "dice":    round(dice_binary(pred[i], label[i]), 4),
                "time_ms": round(elapsed_ms / pred.shape[0], 1),
            })

    dice_scores = [c["dice"] for c in cases]
    times = [c["time_ms"] for c in cases]
    n = len(dice_scores)
    mean_dice = sum(dice_scores) / n if n else 0.0
    std_dice  = (sum((d - mean_dice) ** 2 for d in dice_scores) / n) ** 0.5 if n > 1 else 0.0
    mean_ms   = sum(times) / len(times) if times else 0.0

    summary = {
        "class":        cls,
        "n_samples":    n,
        "mean_dice":    round(mean_dice, 4),
        "std_dice":     round(std_dice, 4),
        "mean_time_ms": round(mean_ms, 1),
    }
    return summary, cases


def evaluate_classes(model, cfg, classes, *, split=None, fig_dir: Path | None = None):
    """Run `validate` over each class via common.make_loader; return (rows, cases).

    Shared by experiments/3d/eval.py (benchmark) and train.py's val step. Rows that
    error carry an "error" key instead of metrics. split defaults to cfg.eval.split.
    """
    from common import make_loader  # local import: common/evaluate are sibling modules

    split = split or cfg.eval.split
    rows, all_cases = [], []
    for cls in classes:
        try:
            loader = make_loader(cfg, cls, split=split)
            row, cases = validate(model, loader, cls, fig_dir=fig_dir)
            rows.append(row)
            all_cases.extend(cases)
        except Exception as exc:  # noqa: BLE001
            rows.append({"class": cls, "error": str(exc)})
    return rows, all_cases
