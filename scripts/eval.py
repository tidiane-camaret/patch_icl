"""
Quality benchmark for in-context 3D segmentation models.

Evaluates one or more models on TotalSegmentator's test split over a curated
organ class list, reporting per-class Dice, mean inference time, and GFLOPs.

Usage
-----
# Quick smoke test (1 class, 5 subjects, K=1)
python scripts/eval.py --models medverse --K 1 --classes liver --n_subjects 5

# Compare native model against Medverse (requires trained checkpoint)
python scripts/eval.py \\
    --models native_resenc medverse \\
    --ckpt_resenc results/checkpoints/resenc_in_context_best.pt \\
    --K 3

# Full benchmark over all default classes
python scripts/eval.py --models medverse native_resenc --K 3 --n_subjects 50
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode
from tqdm import tqdm

# Import wandb BEFORE inserting ROOT into sys.path to avoid the local wandb/
# output directory shadowing the installed package.
import wandb  # noqa: E402

# Make sure src/ is importable when running from repo root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.benchmark_classes import BENCHMARK_CLASSES, MRI_ALL_CLASSES, MRI_BENCHMARK_CLASSES
from data.totalseg_classes import ALL_CLASSES
from src.benchmark_models import load_model
from src.totalseg_dataloader_incontext import TotalSegInContextDataset, incontext_collate_fn
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def dice_binary(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Smooth Dice between two binary int64 tensors of any shape."""
    pred   = pred.bool()
    target = target.bool()
    inter  = (pred & target).sum().item()
    union  = pred.sum().item() + target.sum().item()
    return (2 * inter + 1) / (union + 1)


def _best_slice(mask: np.ndarray) -> int:
    counts = mask.sum(axis=(1, 2))
    return int(counts.argmax()) if counts.max() > 0 else mask.shape[0] // 2


def save_eval_figure(
    target_img: np.ndarray,  # (D, H, W)
    gt:         np.ndarray,  # (D, H, W) binary
    pred:       np.ndarray,  # (D, H, W) binary
    ctx_img:    np.ndarray,  # (D, H, W) first context image
    ctx_gt:     np.ndarray,  # (D, H, W) first context mask
    out_path:   Path,
    title:      str = "",
) -> None:
    """Save a 4-panel figure: context | target | GT overlay | pred overlay."""
    def _norm(sl):
        mn, mx = sl.min(), sl.max()
        return (sl - mn) / (mx - mn + 1e-6)

    z     = _best_slice(gt)
    z_ctx = _best_slice(ctx_gt)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4),
                             gridspec_kw={"wspace": 0.04})
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


def make_loader(cfg, cls: str, context_size: int, n_subjects: int, use_crop: bool = True,
                batch_size: int = 1, num_workers: int = 4) -> DataLoader:
    ds = TotalSegInContextDataset(
        root=cfg.totalseg_root,
        classes=[cls],
        image_size=tuple(cfg.image_size),
        split="test",
        context_size=context_size,
        max_subjects=n_subjects,
        aug_cfg=None,        # no augmentation at eval time
        synth_method=None,   # no synthetic data
        p_synth=0.0,
        class_balanced=False,
        use_crop=use_crop,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=incontext_collate_fn,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )


# ---------------------------------------------------------------------------
# FLOPs measurement
# ---------------------------------------------------------------------------

def measure_flops(model, image_size: tuple, K: int, device: torch.device) -> float:
    """
    Return GFLOPs for one predict() call with a single-sample dummy input.
    Uses FlopCounterMode (PyTorch ≥2.1) which hooks into ATen ops.
    Returns 0.0 on failure (e.g. dynamic control flow that defeats static counting).
    """
    D, H, W = image_size
    dummy_target  = torch.zeros(1, 1, D, H, W, device=device)
    dummy_ctx_img = torch.zeros(1, K, 1, D, H, W, device=device)
    dummy_ctx_msk = torch.zeros(1, K, D, H, W, dtype=torch.long, device=device)
    try:
        with FlopCounterMode(display=False) as fc:
            model.predict(dummy_target, dummy_ctx_img, dummy_ctx_msk)
        return fc.get_total_flops() / 1e9
    except Exception as e:  # noqa: BLE001
        print(f"    [FLOPs] Could not count: {e}")
        return 0.0


# ---------------------------------------------------------------------------
# Per-model evaluation
# ---------------------------------------------------------------------------

def _sync():
    """GPU barrier so time.perf_counter() sees wall-clock GPU time."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def evaluate_model(model, loader, cls: str,
                   fig_dir: Path | None = None) -> tuple[dict, list[dict]]:
    """
    Run inference over a dataloader.

    Returns:
        summary  : per-class aggregated stats
        cases    : list of per-case dicts {class, subject, dice, time_ms}
    """
    cases = []
    fig_saved = False

    for batch in loader:
        target_img    = batch["image"]          # (B, 1, D, H, W)
        context_imgs  = batch["context_in"]     # (B, K, 1, D, H, W)
        context_masks = batch["context_out"]    # (B, K, D, H, W)
        label         = batch["label"]          # (B, D, H, W)
        subjects      = batch.get("subjects", [None] * target_img.shape[0])

        _sync()
        t0 = time.perf_counter()
        pred = model.predict(target_img, context_imgs, context_masks)
        _sync()
        elapsed_ms = (time.perf_counter() - t0) * 1000

        pred  = pred.cpu()
        label = label.cpu()

        if fig_dir is not None and not fig_saved:
            subj = subjects[0] or "s0"
            dice_val = dice_binary(pred[0], label[0])
            save_eval_figure(
                target_img=target_img[0].squeeze(0).numpy(),
                gt=label[0].numpy(),
                pred=pred[0].numpy(),
                ctx_img=context_imgs[0, 0].squeeze(0).numpy(),
                ctx_gt=context_masks[0, 0].numpy(),
                out_path=fig_dir / f"{cls}_{subj}.png",
                title=f"{cls}  {subj}  dice={dice_val:.3f}",
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
    time_ms_list = [c["time_ms"] for c in cases]
    n = len(dice_scores)
    mean_dice = sum(dice_scores) / n if n else 0.0
    std_dice  = (sum((d - mean_dice) ** 2 for d in dice_scores) / n) ** 0.5 if n > 1 else 0.0
    mean_ms   = sum(time_ms_list) / len(time_ms_list) if time_ms_list else 0.0

    summary = {
        "class":        cls,
        "n_samples":    n,
        "mean_dice":    round(mean_dice, 4),
        "std_dice":     round(std_dice, 4),
        "mean_time_ms": round(mean_ms, 1),
    }
    return summary, cases


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="In-context segmentation quality benchmark")

    p.add_argument("--models", nargs="+", required=True,
                   choices=["native_vit", "native_resenc", "medverse", "multilevel", "nninteractive"],
                   help="Models to evaluate")
    p.add_argument("--K", type=int, default=1,
                   help="Number of context (image, mask) pairs")
    p.add_argument("--image_size", type=int, nargs="+", default=[128, 128, 128],
                   help="Spatial size of volumes (must match training size)")
    p.add_argument("--classes", nargs="*", default=None,
                   help="Organ classes to evaluate (default: BENCHMARK_CLASSES)")
    p.add_argument("--all_classes", action="store_true",
                   help="Evaluate on all 121 classes (117 CT + 4 MRI-only); overrides --classes")
    p.add_argument("--n_subjects", type=int, default=50,
                   help="Max test subjects per class")
    p.add_argument("--dataset", choices=["totalseg", "totalsegmri"], default="totalseg",
                   help="Dataset to evaluate on (default: totalseg)")
    p.add_argument("--totalseg_root", type=str, default=None,
                   help="Dataset root (auto-detected from cluster config if absent)")

    # Native model checkpoints
    p.add_argument("--ckpt_vit",        type=str, default="results/checkpoints/vit_in_context_best.pt")
    p.add_argument("--ckpt_resenc",     type=str, default="results/checkpoints/resenc_in_context_best.pt")
    p.add_argument("--ckpt_multilevel", type=str, default=None,
                   help="Path to MultilevelICL best.pt checkpoint")
    p.add_argument("--ckpt_nninteractive", type=str, default=None,
                   help="Path to NNInteractive experiment best.pt checkpoint")
    p.add_argument("--nnint_ckpt", type=str, default=None,
                   help="Override the NNInteractive encoder checkpoint path stored in the experiment config")

    # Medverse options
    p.add_argument("--medverse_ckpt", type=str, default=None,
                   help="Path to Medverse.ckpt (default: repo default)")
    p.add_argument("--medverse_sw_roi", type=int, nargs=3, default=None,
                   help="Sliding-window ROI size for Medverse (default: image_size)")

    p.add_argument("--output", type=str, default=None,
                   help="Output CSV path (default: results/eval_<timestamp>.csv)")
    p.add_argument("--device", type=str, default=None,
                   help="Torch device (default: cuda if available)")
    p.add_argument("--wandb_project", type=str, default="patch_icl_3d_eval",
                   help="W&B project name")
    p.add_argument("--no_wandb", action="store_true",
                   help="Disable W&B logging")
    p.add_argument("--use_crop", action=argparse.BooleanOptionalAction, default=True,
                   help="Enable crop augmentation in the dataloader (default: True)")
    p.add_argument("--batch_size", type=int, default=8,
                   help="Dataloader batch size (default: 1)")
    p.add_argument("--num_workers", type=int, default=20,
                   help="Dataloader num_workers (default: 4)")
    p.add_argument("--save_figs", action=argparse.BooleanOptionalAction, default=True,
                   help="Save one segmentation figure per class (default: True)")

    return p.parse_args()


def resolve_totalseg_root(arg_root: str, dataset: str = "totalseg") -> str:
    """Try cluster config files to find the dataset root.

    Uses an exact-key regex so 'totalseg:' and 'totalsegmri:' are never confused.
    """
    if arg_root:
        return arg_root
    import re
    # Match only the exact key, not a prefix of another key.
    pattern = rf"(?<!\w){re.escape(dataset)}\s*:\s*(.+)"
    for cfg_path in [
        ROOT / "configs" / "cluster" / "nfs.yaml",
        ROOT / "configs" / "cluster" / "meta.yaml",
    ]:
        if cfg_path.exists():
            text = cfg_path.read_text()
            m = re.search(pattern, text)
            if m:
                return m.group(1).strip()
    raise RuntimeError(
        f"Cannot auto-detect root for dataset '{dataset}'. Pass --totalseg_root explicitly."
    )


class _Cfg:
    """Minimal config container passed to make_loader."""
    def __init__(self, totalseg_root, image_size):
        self.totalseg_root = totalseg_root
        self.image_size    = image_size


def main():
    args = parse_args()

    device = torch.device(args.device) if args.device else \
             torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    default_classes = MRI_BENCHMARK_CLASSES if args.dataset == "totalsegmri" else BENCHMARK_CLASSES
    all_classes     = MRI_ALL_CLASSES if args.dataset == "totalsegmri" else ALL_CLASSES
    classes = all_classes if args.all_classes else (args.classes or default_classes)
    image_size = tuple(args.image_size)
    totalseg_root = resolve_totalseg_root(args.totalseg_root, dataset=args.dataset)
    print(f"Dataset       : {args.dataset}")
    print(f"Data root     : {totalseg_root}")
    print(f"Classes ({len(classes)}): {', '.join(classes)}")
    print(f"Context K={args.K}  |  image_size={image_size}  |  n_subjects≤{args.n_subjects}\n")

    cfg = _Cfg(totalseg_root=totalseg_root, image_size=image_size)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir   = ROOT / "results/eval"
    out_dir.mkdir(exist_ok=True)
    fig_base  = out_dir / f"figures_{timestamp}" if args.save_figs else None

    results = {}  # model_name → list[dict]

    for model_name in args.models:
        print(f"{'='*60}")
        print(f"  Model: {model_name}")
        print(f"{'='*60}")

        # Instantiate model
        model_kwargs = {}
        if model_name == "native_vit":
            model_kwargs["ckpt_path"] = args.ckpt_vit
        elif model_name == "native_resenc":
            model_kwargs["ckpt_path"] = args.ckpt_resenc
        elif model_name == "medverse":
            if args.medverse_ckpt:
                model_kwargs["ckpt_path"] = args.medverse_ckpt
            if args.medverse_sw_roi:
                model_kwargs["sw_roi_size"] = tuple(args.medverse_sw_roi)
        elif model_name == "multilevel":
            if args.ckpt_multilevel is None:
                raise ValueError("--ckpt_multilevel is required for the multilevel model")
            model_kwargs["ckpt_path"] = args.ckpt_multilevel
        elif model_name == "nninteractive":
            if args.ckpt_nninteractive is None:
                raise ValueError("--ckpt_nninteractive is required for the nninteractive model")
            model_kwargs["ckpt_path"] = args.ckpt_nninteractive
            if args.nnint_ckpt:
                model_kwargs["nnint_ckpt"] = args.nnint_ckpt

        model = load_model(model_name, image_size=image_size, device=device, **model_kwargs)

        print(f"  Measuring FLOPs (K={args.K}, size={image_size})...")
        gflops = measure_flops(model, image_size, args.K, device)
        print(f"  GFLOPs: {gflops:.2f}")

        # W&B: one run per model
        use_wandb = not args.no_wandb
        if use_wandb:
            run = wandb.init(
                project=args.wandb_project,
                name=model_name,
                config=dict(
                    model=model_name,
                    dataset=args.dataset,
                    totalseg_root=totalseg_root,
                    K=args.K,
                    image_size=list(image_size),
                    n_subjects=args.n_subjects,
                    classes=classes,
                    batch_size=args.batch_size,
                    gflops=round(gflops, 2),
                ),
                reinit="finish_previous",
            )
            case_table = wandb.Table(columns=["class", "subject", "dice", "time_ms"])

        fig_dir = fig_base / model_name if fig_base is not None else None

        model_results = []
        all_cases = []
        for cls in tqdm(classes, desc=model_name):
            try:
                loader = make_loader(cfg, cls, context_size=args.K, n_subjects=args.n_subjects,
                                    use_crop=args.use_crop, batch_size=args.batch_size,
                                    num_workers=args.num_workers)
                row, cases = evaluate_model(model, loader, cls, fig_dir=fig_dir)
                row["gflops"] = round(gflops, 2)
                model_results.append(row)
                all_cases.extend(cases)

                tqdm.write(
                    f"  {cls:<35s}  dice={row['mean_dice']:.3f} ± {row['std_dice']:.3f}"
                    f"  {row['mean_time_ms']:.0f}ms/sample  n={row['n_samples']}"
                )

                if use_wandb:
                    wandb.log({
                        f"class/{cls}/mean_dice":    row["mean_dice"],
                        f"class/{cls}/std_dice":     row["std_dice"],
                        f"class/{cls}/mean_time_ms": row["mean_time_ms"],
                    })
                    for c in cases:
                        case_table.add_data(c["class"], c["subject"], c["dice"], c["time_ms"])

            except Exception as e:  # noqa: BLE001
                tqdm.write(f"  {cls:<35s}  ERROR: {e}")
                model_results.append({"class": cls, "error": str(e)})

        valid = [r for r in model_results if "mean_dice" in r]
        if valid:
            mean_overall = sum(r["mean_dice"] for r in valid) / len(valid)
            mean_ms_overall = sum(r["mean_time_ms"] for r in valid) / len(valid)
            print(
                f"\n  Mean Dice: {mean_overall:.4f}  |  "
                f"Mean time: {mean_ms_overall:.1f} ms/sample  |  "
                f"GFLOPs: {gflops:.2f}\n"
            )
            if use_wandb:
                wandb.log({
                    "mean_dice":    round(mean_overall, 4),
                    "mean_time_ms": round(mean_ms_overall, 1),
                    "gflops":       round(gflops, 2),
                    "cases":        case_table,
                })

        if use_wandb:
            wandb.finish()

        results[model_name] = model_results

    # -----------------------------------------------------------------------
    # Save outputs
    # -----------------------------------------------------------------------
    # JSON (full detail)
    json_path = out_dir / f"eval_{timestamp}.json"
    json_path.write_text(json.dumps(results, indent=2))
    print(f"Saved JSON → {json_path}")

    # CSV (summary table: model × class)
    csv_path = Path(args.output) if args.output else out_dir / f"eval_{timestamp}.csv"
    rows = ["model,class,mean_dice,std_dice,mean_time_ms,gflops,n_samples"]
    for model_name, model_results in results.items():
        for r in model_results:
            if "mean_dice" in r:
                rows.append(
                    f"{model_name},{r['class']},{r['mean_dice']},{r['std_dice']},"
                    f"{r.get('mean_time_ms', '')},"
                    f"{r.get('gflops', '')},"
                    f"{r['n_samples']}"
                )
    csv_path.write_text("\n".join(rows) + "\n")
    print(f"Saved CSV  → {csv_path}")


if __name__ == "__main__":
    main()
