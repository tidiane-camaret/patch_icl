"""One-time rename: totalsegmri's files were mistakenly written with ct-prefixed names
(a from-CT copy-paste in convert_to_npy.py), fixed 2026-09-22 across the codebase. This
applies the matching on-disk rename via os.rename (same filesystem -> atomic, no copy).

Root-level: ct_stats.json[.bak] -> mri_stats.json[.bak]
Per-subject: ct.npy -> mri.npy, ct_raw.npy -> mri_raw.npy, ct_{size}.npy -> mri_{size}.npy,
             ct_raw_{pitch}mm.npy -> mri_raw_{pitch}mm.npy

Usage:
  python scripts/rename_mri_files.py --data <totalsegmri_root> [--dry-run]
"""
import argparse
import re
from pathlib import Path

_SUBJ_PATTERNS = [
    (re.compile(r"^ct\.npy$"), "mri.npy"),
    (re.compile(r"^ct_raw\.npy$"), "mri_raw.npy"),
    (re.compile(r"^ct_raw_(?P<pitch>[\d.]+mm)\.npy$"), "mri_raw_{pitch}.npy"),
    (re.compile(r"^ct_(?P<size>\d+x\d+x\d+)\.npy$"), "mri_{size}.npy"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="totalsegmri root (contains s0000/, ...)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    root = Path(args.data)

    n = 0
    for root_name in ("ct_stats.json", "ct_stats.json.bak"):
        src = root / root_name
        if src.exists():
            dst = root / root_name.replace("ct_stats", "mri_stats")
            print(f"{src} -> {dst}")
            if not args.dry_run:
                src.rename(dst)
            n += 1

    subj_dirs = sorted(p for p in root.iterdir() if p.is_dir())
    for subj_dir in subj_dirs:
        for f in list(subj_dir.iterdir()):
            for pat, template in _SUBJ_PATTERNS:
                m = pat.match(f.name)
                if m:
                    new_name = template.format(**m.groupdict())
                    dst = subj_dir / new_name
                    if not args.dry_run:
                        f.rename(dst)
                    n += 1
                    break
        if n % 200 < 4:  # coarse progress ping
            pass

    print(f"{'[dry-run] ' if args.dry_run else ''}renamed {n} files under {root}")


if __name__ == "__main__":
    main()
