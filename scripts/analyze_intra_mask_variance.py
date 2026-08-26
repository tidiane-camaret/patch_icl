#!/usr/bin/env python3
"""Estimate real intra-mask variance from TotalSegmentator test set.

Uses the v2 dataset infrastructure with experiment config params.
Compares to synth_gmm's var_max=5.0 (max σ ≈ 2.2).
"""
import argparse
import numpy as np
import torch
from collections import defaultdict
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.providers.totalseg import TotalSegProvider
from src.incontext_dataset_v2 import LoadRequest
import random


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--totalseg", type=str,
                        default="/nfs/data/nii/data1/Analysis/camaret___in_context_segmentation/ANALYSIS_20251122/data/totalseg")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--crop_spacing_mm", type=float, default=3.0)
    parser.add_argument("--classes", type=str, nargs="*", default=None,
                        help="Subset of classes to analyze (default: ts_organs)")
    parser.add_argument("--max_subjects", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Resolve classes
    if args.classes is None:
        from data.totalseg_classes import resolve_classes
        classes = resolve_classes(["ts_organs", "-kidney_cyst_left", "-kidney_cyst_right"])
    else:
        classes = args.classes

    print(f"Analyzing {len(classes)} classes from {args.split} split")
    print(f"Image size: {args.image_size}³, crop spacing: {args.crop_spacing_mm} mm")

    # Create provider
    provider = TotalSegProvider(
        root=Path(args.totalseg),
        split=args.split,
        image_size=(args.image_size,) * 3,
        crop_spacing_mm=args.crop_spacing_mm,
        classes=classes,
        max_subjects=args.max_subjects,
    )

    rng = random.Random(args.seed)

    # Collect per-class stats
    class_vars = defaultdict(list)   # class -> list of per-mask variances
    class_stds = defaultdict(list)   # class -> list of per-mask stds
    class_means = defaultdict(list)  # class -> list of per-mask means
    class_sizes = defaultdict(list)  # class -> list of mask voxel counts

    total_samples = 0
    for cls in provider.classes:
        subjects = provider.subjects_for(cls)
        if not subjects:
            continue

        for subj in subjects[:args.max_subjects]:
            try:
                req = LoadRequest(rng=rng, crop_spacing_mm=args.crop_spacing_mm)
                result = provider.load(subj, cls, req)

                img = result.image.numpy().squeeze()  # (T,T,T)
                mask = result.label.numpy()           # (T,T,T)

                # Get intensities within mask
                vals = img[mask > 0]
                if len(vals) < 10:
                    continue

                var = float(np.var(vals))
                std = float(np.std(vals))
                mean = float(np.mean(vals))

                class_vars[cls].append(var)
                class_stds[cls].append(std)
                class_means[cls].append(mean)
                class_sizes[cls].append(len(vals))
                total_samples += 1

            except Exception as e:
                print(f"  Skip {subj}/{cls}: {e}")
                continue

    print(f"\nCollected {total_samples} samples across {len(class_vars)} classes\n")

    # Summary stats
    print("=" * 80)
    print(f"{'Class':<30} {'N':>4} {'Mean σ':>8} {'Med σ':>8} {'P95 σ':>8} {'Max σ':>8} {'Voxels':>8}")
    print("=" * 80)

    all_stds = []
    all_vars = []
    for cls in sorted(class_vars.keys()):
        stds = class_stds[cls]
        sizes = class_sizes[cls]
        all_stds.extend(stds)
        all_vars.extend(class_vars[cls])

        print(f"{cls:<30} {len(stds):>4} "
              f"{np.mean(stds):>8.3f} {np.median(stds):>8.3f} "
              f"{np.percentile(stds, 95):>8.3f} {np.max(stds):>8.3f} "
              f"{int(np.median(sizes)):>8}")

    print("=" * 80)
    print(f"\nGlobal stats (across all {len(all_stds)} samples):")
    print(f"  Mean intra-mask σ: {np.mean(all_stds):.3f}")
    print(f"  Median intra-mask σ: {np.median(all_stds):.3f}")
    print(f"  P5-P95 range: [{np.percentile(all_stds, 5):.3f}, {np.percentile(all_stds, 95):.3f}]")
    print(f"  Min-Max: [{np.min(all_stds):.3f}, {np.max(all_stds):.3f}]")

    print(f"\nFor comparison, synth_gmm uses:")
    print(f"  var_max = 5.0 → max σ = √5 ≈ 2.24")
    print(f"  σ ~ Uniform(0, 2.24)")
    print(f"  Mean σ ≈ 1.12")

    # Variance stats
    print(f"\nIntra-mask VARIANCE stats:")
    print(f"  Mean: {np.mean(all_vars):.3f}")
    print(f"  Median: {np.median(all_vars):.3f}")
    print(f"  P95: {np.percentile(all_vars, 95):.3f}")
    print(f"  Max: {np.max(all_vars):.3f}")


if __name__ == "__main__":
    main()
