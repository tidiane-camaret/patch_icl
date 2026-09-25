"""Recompute a native-grid MRI source's per-subject intensity stats at different foreground
percentile bounds, WITHOUT touching the raw npy / any sized cache — those are untouched by this.

The raw file already holds raw, unclipped MRI intensities (see convert_to_npy.py); only the
stats sidecar's per-subject {clip_lo, clip_hi, mean, std} (consumed at load time by
normalize_mri / resolve_ct_norm) need to change. This re-reads each subject's existing raw npy
(no .nii.gz decode, no resample) and calls the same src.totalseg_dataset.mri_stats used by
convert_to_npy.py's --mri-percentile-lo/--mri-percentile-hi.

Default filenames match TotalSegMRI (mri_raw.npy / mri_stats.json); pass --raw-name ct_raw.npy
--stats-name ct_stats.json for the native-grid OOD sources (isles22, shifts_ms,
msd_hippocampus, msd_prostate, atlas_v2, gnc_kidney — see src/providers/native_grid.py), which
use that naming convention even though the modality is MRI.

Usage:
  python scripts/recompute_mri_stats.py --data <totalsegmri_root> --pct-lo 2 --pct-hi 98
  python scripts/recompute_mri_stats.py --data <isles22_npy_root> --pct-lo 2 --pct-hi 98 \
    --raw-name ct_raw.npy --stats-name ct_stats.json
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
    ap.add_argument("--raw-name", default="mri_raw.npy",
                     help="per-subject raw npy filename (default: mri_raw.npy; use ct_raw.npy "
                          "for the native-grid OOD sources)")
    ap.add_argument("--stats-name", default="mri_stats.json",
                     help="output stats filename (default: mri_stats.json; use ct_stats.json "
                          "for the native-grid OOD sources)")
    ap.add_argument("--out", default=None, help="output path (default: <data>/<stats-name>)")
    args = ap.parse_args()

    root = Path(args.data)
    out_path = Path(args.out) if args.out else root / args.stats_name

    subj_dirs = sorted(p for p in root.iterdir() if p.is_dir() and (p / args.raw_name).exists())
    print(f"{len(subj_dirs)} subjects with {args.raw_name} under {root}")

    stats = {}
    for i, subj_dir in enumerate(subj_dirs):
        raw = np.load(subj_dir / args.raw_name).astype(np.float32)
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
