"""Convert ATLAS v2.0 (flat images/masks + train.jsonl) -> per-subject .npy at NATIVE spacing.
Mirrors scripts/convert_isles22.py: native grid only, resampling deferred to the dataloader
(src/providers/atlas_v2.py).

PROVENANCE CAVEAT (see docs/datasets/atlas_v2.md #1): the official ATLAS v2.0 release is
DUA-gated (ICPSR restricted-use, or NITRC/INDI's "preprocessed" copy behind a reviewed
Google-Form application) -- this converter reads from an UNOFFICIAL HuggingFace re-upload
(jayzzzzz0134/atlas-stroke) that ships no README/license metadata, very likely an unauthorized
redistribution of consent-controlled patient data. Downloaded and integrated on explicit user
direction after this distinction was flagged. Not equivalent to every other converter in this
repo, which all read from a genuinely open official/archival source.

Unlike ISLES22/Shifts-MS/Hippocampus, this source needs NO orientation fix and NO per-subject
geometry handling: every one of the 655 subjects shares the exact same 197x233x189 @ 1mm/RAS
grid (verified at characterization time) -- this mirror ships an already-template-registered
version, not raw per-site acquisitions. `nib.as_closest_canonical` is kept as a no-op safety
net rather than a real fix, matching the pattern used for MSD Hippocampus.

Layout written under --out:
    sub-r001s001/ct_raw.npy   (D,H,W) float32, RAS, native (1mm) grid, T1 intensity
    sub-r001s001/label.npy    (D,H,W) uint8, 0=bg, 1=stroke_lesion
    spacings.json              {subj: {spacing, shape, affine}}
    ct_stats.json              {subj: {clip_lo, clip_hi, mean, std}}  (mri_stats)

Usage:
    python scripts/convert_atlas_v2.py --workers 16
"""
import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.totalseg_dataset import mri_stats  # noqa: E402

ATLAS_CLASSES = ["stroke_lesion"]

DEFAULT_SRC = ("/nfs/data/nii/data1/Analysis/camaret___in_context_segmentation/"
               "ANALYSIS_20251122/data/atlas_v2")
DEFAULT_OUT = ("/nfs/data/nii/data1/Analysis/camaret___in_context_segmentation/"
               "ANALYSIS_20251122/data/atlas_v2/npy")


def convert_one(subj: str, img_path: Path, msk_path: Path, out_dir: Path,
                overwrite: bool) -> tuple[str, dict | None, dict | None, str | None]:
    """Convert one case. Returns (subj, spacings_meta, ct_stats_entry, error)."""
    try:
        subj_dir = out_dir / subj
        img_out, lbl_out = subj_dir / "ct_raw.npy", subj_dir / "label.npy"

        img = nib.as_closest_canonical(nib.load(str(img_path)))
        msk = nib.as_closest_canonical(nib.load(str(msk_path)))
        affine = np.asarray(img.affine, dtype=np.float64)
        spacing = [float(x) for x in nib.affines.voxel_sizes(affine)[:3]]
        meta = {"spacing": spacing, "shape": [int(x) for x in img.shape[:3]],
                "affine": affine.tolist()}

        if img.shape[:3] != msk.shape[:3]:
            return subj, None, None, f"shape mismatch img{img.shape[:3]} msk{msk.shape[:3]}"
        if not np.allclose(affine, np.asarray(msk.affine), atol=1e-3):
            return subj, None, None, "image/mask affine mismatch"

        raw = np.ascontiguousarray(img.get_fdata(dtype=np.float32))
        stats = mri_stats(raw)                              # whole-volume, BEFORE overwrite check

        if not overwrite and img_out.exists() and lbl_out.exists():
            return subj, meta, stats, None

        lab = np.ascontiguousarray(msk.get_fdata(dtype=np.float32))
        if not np.array_equal(lab, np.round(lab)):
            return subj, None, None, "mask is not integral-valued"
        lab = np.round(lab).astype(np.int32)
        if lab.min() < 0 or lab.max() > 1:
            return subj, None, None, f"mask range [{lab.min()}, {lab.max()}] outside 0..1"

        subj_dir.mkdir(parents=True, exist_ok=True)
        np.save(img_out, raw.astype(np.float32))
        np.save(lbl_out, lab.astype(np.uint8))
        meta["classes_present"] = sorted(int(v) for v in np.unique(lab) if v != 0)
        return subj, meta, stats, None
    except Exception as exc:  # noqa: BLE001
        return subj, None, None, f"{type(exc).__name__}: {exc}"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=DEFAULT_SRC, help="atlas_v2 root (images/, masks/, train.jsonl)")
    ap.add_argument("--out", default=DEFAULT_OUT, help="npy root to write")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    src, out = Path(args.src), Path(args.out)
    manifest = src / "train.jsonl"
    if not manifest.exists():
        raise SystemExit(f"expected {manifest}")

    cases = []
    with open(manifest) as f:
        for line in f:
            rec = json.loads(line)
            subj = rec["patient_id"]
            img_path = src / rec["image"]
            msk_path = src / rec["mask"]
            if not img_path.exists() or not msk_path.exists():
                print(f"  [skip] {subj}: missing image or mask on disk")
                continue
            cases.append((subj, img_path, msk_path))
    print(f"{len(cases)} cases  {src} -> {out}")
    out.mkdir(parents=True, exist_ok=True)

    spacings, ct_stats, errors = {}, {}, []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(convert_one, s, i, m, out, args.overwrite) for s, i, m in cases]
        for n, fut in enumerate(as_completed(futs), 1):
            subj, meta, stats, err = fut.result()
            if err:
                errors.append(f"{subj}: {err}")
                print(f"  [FAIL] {subj}: {err}")
            else:
                spacings[subj] = meta
                ct_stats[subj] = stats
            if n % 50 == 0:
                print(f"  {n}/{len(cases)}")

    with open(out / "spacings.json", "w") as f:
        json.dump({k: spacings[k] for k in sorted(spacings)}, f, indent=1)
    with open(out / "ct_stats.json", "w") as f:
        json.dump({k: ct_stats[k] for k in sorted(ct_stats)}, f, indent=1)

    n_empty = sum(1 for m in spacings.values() if not m.get("classes_present"))
    sp = np.array([m["spacing"] for m in spacings.values()])
    sh = np.array([m["shape"] for m in spacings.values()])
    print(f"\nwrote {len(spacings)} cases ({n_empty} with an empty lesion mask), "
          f"{len(errors)} failed")
    if len(sp):
        print(f"  spacing: min {sp.min(0).round(2)} max {sp.max(0).round(2)}")
        print(f"  shape: min {sh.min(0)} median {np.median(sh, 0)} max {sh.max(0)}")
        print(f"  -> {out/'spacings.json'}, {out/'ct_stats.json'}")
    for e in errors:
        print(f"  {e}")


if __name__ == "__main__":
    main()
