"""Recompute TotalSegMRI's mri_stats.json at different foreground percentile bounds,
WITHOUT touching mri_raw.npy / mri.npy / any sized cache — those are untouched by this.

mri_raw.npy already holds raw, unclipped MRI intensities (see convert_to_npy.py); only
mri_stats.json's per-subject {clip_lo, clip_hi, mean, std} (consumed at load time by
normalize_mri / resolve_ct_norm) need to change. This re-reads each subject's existing
mri_raw.npy (no .nii.gz decode, no resample) and calls the same src.totalseg_dataset.mri_stats
used by convert_to_npy.py's --mri-percentile-lo/--mri-percentile-hi.

Usage:
  python scripts/recompute_mri_stats.py --data <totalsegmri_root> --pct-lo 2 --pct-hi 98
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.totalseg_dataset import mri_stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="totalsegmri root (contains s0000/, ...)")
    ap.add_argument("--pct-lo", type=float, default=2.0)
    ap.add_argument("--pct-hi", type=float, default=98.0)
    ap.add_argument("--out", default=None, help="output path (default: <data>/mri_stats.json)")
    args = ap.parse_args()

    root = Path(args.data)
    out_path = Path(args.out) if args.out else root / "mri_stats.json"

    subj_dirs = sorted(p for p in root.iterdir() if p.is_dir() and (p / "mri_raw.npy").exists())
    print(f"{len(subj_dirs)} subjects with mri_raw.npy under {root}")

    stats = {}
    for i, subj_dir in enumerate(subj_dirs):
        raw = np.load(subj_dir / "mri_raw.npy").astype(np.float32)
        stats[subj_dir.name] = mri_stats(raw, pct_lo=args.pct_lo, pct_hi=args.pct_hi)
        if (i + 1) % 50 == 0 or i + 1 == len(subj_dirs):
            print(f"  {i + 1}/{len(subj_dirs)}")

    if out_path.exists():
        backup = out_path.with_suffix(".json.bak")
        backup.write_text(out_path.read_text())
        print(f"backed up existing {out_path} -> {backup}")

    with open(out_path, "w") as f:
        json.dump(stats, f)
    print(f"wrote {out_path} ({len(stats)} subjects, pct_lo={args.pct_lo}, pct_hi={args.pct_hi})")


if __name__ == "__main__":
    main()
