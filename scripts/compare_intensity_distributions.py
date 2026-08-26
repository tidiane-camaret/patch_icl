#!/usr/bin/env python3
"""Compare intensity distributions of synth_gmm vs totalseg after augmentation pipeline.

Usage:
  python scripts/compare_intensity_distributions.py
  python scripts/compare_intensity_distributions.py --n_samples 50
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
import torch


def load_config():
    """Load the 58_organs_synth_gmm config with full augmentations."""
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    config_dir = str(Path(__file__).resolve().parents[1] / "configs" / "experiment" / "3d")
    initialize_config_dir(config_dir=config_dir, version_base=None)
    cfg = compose(config_name="train", overrides=["experiment=58_organs_synth_gmm"])
    return cfg


def get_datasets(cfg, n_samples=30):
    """Create both synth_gmm and totalseg datasets with augmentations."""
    pass  # imports done in collect_intensities

    # synth_gmm dataset
    cfg_synth = OmegaConf.to_container(cfg, resolve=True)
    cfg_synth = OmegaConf.create(cfg_synth)
    cfg_synth.data.source = "synth_gmm_maisi"

    # totalseg dataset
    cfg_totalseg = OmegaConf.to_container(cfg, resolve=True)
    cfg_totalseg = OmegaConf.create(cfg_totalseg)
    cfg_totalseg.data.source = "totalseg"

    return cfg_synth, cfg_totalseg


def collect_intensities(cfg, source_name, n_samples=30, paint_mask_aligned=False):
    """Collect intensity statistics from a dataset."""
    from src.incontext_dataset_v2 import InContextDataset

    # Temporarily modify config for this source
    cfg = OmegaConf.to_container(cfg, resolve=True)
    cfg = OmegaConf.create(cfg)

    # Handle image_size - could be int or list
    img_size = cfg.data.image_size
    try:
        # Try as iterable first (ListConfig, list, tuple)
        image_size = tuple(int(x) for x in img_size)
    except TypeError:
        # Fall back to scalar
        image_size = (int(img_size),) * 3

    if source_name == "synth_gmm_maisi":
        from src.synth_gmm_maisi_dataset import SynthGmmMaisiDataset
        from src.providers.synth_gmm import SynthGmmProvider
        from data.totalseg_classes import resolve_classes

        train_classes = resolve_classes(cfg.data.train_classes)
        synth_ds = SynthGmmMaisiDataset(
            bank_dir=cfg.paths.gmm_bank,
            image_size=image_size,
            context_size=cfg.data.context_size,
            crop_spacing_mm=cfg.data.crop_spacing_mm,
            classes=train_classes,
            length=n_samples * 2,
            eval_seed=42,
            paint_mask_aligned=paint_mask_aligned,
        )
        ds = InContextDataset(
            SynthGmmProvider(synth_ds),
            context_size=cfg.data.context_size,
            aug_cfg=cfg.augmentations,
            crop_spacing_mm=cfg.data.crop_spacing_mm,
            eval_seed=None,  # fresh entropy for aug
        )
    else:  # totalseg
        from src.providers.totalseg import TotalSegProvider
        from data.totalseg_classes import resolve_classes

        train_classes = resolve_classes(cfg.data.train_classes)
        provider = TotalSegProvider(
            root=Path(cfg.paths.totalseg),
            split="train",
            image_size=image_size,
            crop_spacing_mm=cfg.data.crop_spacing_mm,
            classes=train_classes,
            max_subjects=n_samples,
        )
        ds = InContextDataset(
            provider,
            context_size=cfg.data.context_size,
            aug_cfg=cfg.augmentations,
            crop_spacing_mm=cfg.data.crop_spacing_mm,
            eval_seed=None,
        )

    all_vals = []
    mask_vals = []
    bg_vals = []
    intra_mask_stds = []

    print(f"Collecting {n_samples} samples from {source_name}...")
    for i in range(min(n_samples, len(ds))):
        try:
            item = ds[i]
            img = item["image"].numpy().squeeze()  # (T,T,T)
            mask = item["label"].numpy()           # (T,T,T)

            # All values
            all_vals.extend(img.flatten().tolist())

            # Masked values
            fg = img[mask > 0]
            bg = img[mask == 0]
            if len(fg) > 10:
                mask_vals.extend(fg.tolist())
                intra_mask_stds.append(np.std(fg))
            if len(bg) > 10:
                bg_vals.extend(bg.tolist())

        except Exception as e:
            print(f"  Skip {i}: {e}")
            continue

    return {
        "all": np.array(all_vals),
        "mask": np.array(mask_vals),
        "bg": np.array(bg_vals),
        "intra_mask_stds": np.array(intra_mask_stds),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=30)
    parser.add_argument("--out", type=str, default="results/intensity_comparison.png")
    parser.add_argument("--paint_mask_aligned", action="store_true",
                        help="Enable paint_mask_aligned for synth_gmm")
    args = parser.parse_args()

    cfg = load_config()
    print("Config loaded. Augmentations:")
    print(OmegaConf.to_yaml(cfg.augmentations))

    # Collect from both sources
    synth_data = collect_intensities(cfg, "synth_gmm_maisi", args.n_samples,
                                     paint_mask_aligned=args.paint_mask_aligned)
    totalseg_data = collect_intensities(cfg, "totalseg", args.n_samples)

    # Plot comparison
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Row 1: Histograms
    bins = np.linspace(-3, 4, 100)

    # All values
    ax = axes[0, 0]
    ax.hist(synth_data["all"], bins=bins, alpha=0.5, density=True, label="synth_gmm")
    ax.hist(totalseg_data["all"], bins=bins, alpha=0.5, density=True, label="totalseg")
    ax.set_xlabel("Intensity (normalized)")
    ax.set_ylabel("Density")
    ax.set_title("All voxels")
    ax.legend()
    ax.set_xlim(-3, 4)

    # Foreground (mask) values
    ax = axes[0, 1]
    ax.hist(synth_data["mask"], bins=bins, alpha=0.5, density=True, label="synth_gmm")
    ax.hist(totalseg_data["mask"], bins=bins, alpha=0.5, density=True, label="totalseg")
    ax.set_xlabel("Intensity (normalized)")
    ax.set_ylabel("Density")
    ax.set_title("Foreground (mask=1) voxels")
    ax.legend()
    ax.set_xlim(-3, 4)

    # Background values
    ax = axes[0, 2]
    ax.hist(synth_data["bg"], bins=bins, alpha=0.5, density=True, label="synth_gmm")
    ax.hist(totalseg_data["bg"], bins=bins, alpha=0.5, density=True, label="totalseg")
    ax.set_xlabel("Intensity (normalized)")
    ax.set_ylabel("Density")
    ax.set_title("Background (mask=0) voxels")
    ax.legend()
    ax.set_xlim(-3, 4)

    # Row 2: Statistics
    # Intra-mask std distribution
    ax = axes[1, 0]
    std_bins = np.linspace(0, 1.5, 50)
    ax.hist(synth_data["intra_mask_stds"], bins=std_bins, alpha=0.5, density=True, label="synth_gmm")
    ax.hist(totalseg_data["intra_mask_stds"], bins=std_bins, alpha=0.5, density=True, label="totalseg")
    ax.set_xlabel("Intra-mask σ")
    ax.set_ylabel("Density")
    ax.set_title("Intra-mask variance distribution")
    ax.legend()
    ax.axvline(np.median(synth_data["intra_mask_stds"]), color="C0", linestyle="--", alpha=0.7)
    ax.axvline(np.median(totalseg_data["intra_mask_stds"]), color="C1", linestyle="--", alpha=0.7)

    # Summary stats text
    ax = axes[1, 1]
    ax.axis("off")
    stats_text = f"""
    Summary Statistics (after augmentation)
    =======================================

    SYNTH_GMM:
      All voxels:  μ={np.mean(synth_data['all']):.3f}, σ={np.std(synth_data['all']):.3f}
      Foreground:  μ={np.mean(synth_data['mask']):.3f}, σ={np.std(synth_data['mask']):.3f}
      Background:  μ={np.mean(synth_data['bg']):.3f}, σ={np.std(synth_data['bg']):.3f}
      Intra-mask σ: median={np.median(synth_data['intra_mask_stds']):.3f},
                    P5-P95=[{np.percentile(synth_data['intra_mask_stds'], 5):.3f}, {np.percentile(synth_data['intra_mask_stds'], 95):.3f}]

    TOTALSEG:
      All voxels:  μ={np.mean(totalseg_data['all']):.3f}, σ={np.std(totalseg_data['all']):.3f}
      Foreground:  μ={np.mean(totalseg_data['mask']):.3f}, σ={np.std(totalseg_data['mask']):.3f}
      Background:  μ={np.mean(totalseg_data['bg']):.3f}, σ={np.std(totalseg_data['bg']):.3f}
      Intra-mask σ: median={np.median(totalseg_data['intra_mask_stds']):.3f},
                    P5-P95=[{np.percentile(totalseg_data['intra_mask_stds'], 5):.3f}, {np.percentile(totalseg_data['intra_mask_stds'], 95):.3f}]
    """
    ax.text(0.1, 0.5, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='center', fontfamily='monospace')

    # Percentile comparison
    ax = axes[1, 2]
    percentiles = [1, 5, 25, 50, 75, 95, 99]
    synth_pct = [np.percentile(synth_data["all"], p) for p in percentiles]
    totalseg_pct = [np.percentile(totalseg_data["all"], p) for p in percentiles]

    x = np.arange(len(percentiles))
    width = 0.35
    ax.bar(x - width/2, synth_pct, width, label="synth_gmm", alpha=0.7)
    ax.bar(x + width/2, totalseg_pct, width, label="totalseg", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([f"P{p}" for p in percentiles])
    ax.set_ylabel("Intensity")
    ax.set_title("Percentile comparison")
    ax.legend()
    ax.axhline(0, color="gray", linestyle="--", alpha=0.3)

    plt.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=150)
    print(f"\nSaved → {args.out}")

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"\nIntra-mask σ comparison:")
    print(f"  synth_gmm: {np.median(synth_data['intra_mask_stds']):.3f} (median)")
    print(f"  totalseg:  {np.median(totalseg_data['intra_mask_stds']):.3f} (median)")
    print(f"  Ratio:     {np.median(synth_data['intra_mask_stds']) / np.median(totalseg_data['intra_mask_stds']):.2f}x")


if __name__ == "__main__":
    main()
