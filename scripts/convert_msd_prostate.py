"""Convert MSD Task05_Prostate (nnU-Net layout, 4D .nii.gz) -> per-CHANNEL .npy at NATIVE
spacing. Mirrors scripts/convert_isles22.py / convert_msd_hippocampus.py: native grid only,
resampling deferred to the dataloader (src/providers/msd_prostate.py).

The design decision this dataset needed (docs/datasets/msd_prostate.md §7): `dataset.json`
declares `"tensorImageSize": "4D"`, each `imagesTr/*.nii.gz` is a single file stacking
T2-weighted (channel 0) and ADC (channel 1) on a 4th axis -- every provider in the harness is
single-channel. Rather than build multi-channel input support, **each channel becomes its own
single-channel (subject, class) task**: `prostate_00_t2` and `prostate_00_adc` are two separate
converted "subjects" that BOTH point at the same PZ/TZ label mask. This doubles the eval pool
(32 -> 64 virtual subjects) instead of shrinking it to one modality, and sidesteps the
multi-channel-input design question entirely -- T2 and ADC have wildly different intensity
units anyway (this converter computes independent per-channel `mri_stats`, same as every other
native MRI source), so treating them as genuinely different "subjects" is not a stretch.

One caveat worth knowing before reading Dice numbers off this source: a real minority of cases
(6/32, verified) have a non-negligible affine SHEAR (up to ~42% of the axis norm on Y/Z --
gantry-tilted acquisition, common in pelvic MRI to dodge hip-prosthesis artifacts). Like every
converter in this harness, this one takes `nib.affines.voxel_sizes` (column-norm) as the
per-axis spacing and treats the grid as if axis-aligned for the crop/resample step downstream
-- there is no true oblique/shear-aware resampling anywhere in this pipeline. For the sheared
minority this introduces a genuine (if modest, sub-voxel-to-few-mm at this FOV) geometric skew
between what the header claims and what a naive axis-aligned crop extracts. Flagged, not fixed
-- fixing it would mean adding shear-aware resampling to the whole harness, out of scope here.

Layout written under --out:
    prostate_00_t2/ct_raw.npy    (D,H,W) float32, RAS(-ish, see caveat above), native grid, T2
    prostate_00_t2/label.npy     (D,H,W) uint8, 0=bg, 1=PZ, 2=TZ
    prostate_00_adc/ct_raw.npy   (D,H,W) float32, native grid, ADC (same label as _t2)
    prostate_00_adc/label.npy
    spacings.json                 {subj: {spacing, shape, affine}}
    ct_stats.json                 {subj: {clip_lo, clip_hi, mean, std}}  (mri_stats, per channel)

Usage:
    python scripts/convert_msd_prostate.py --workers 16
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

CHANNELS = {"t2": 0, "adc": 1}   # dataset.json: {"0": "T2", "1": "ADC"}

DEFAULT_SRC = ("/nfs/data/nii/data1/Analysis/camaret___in_context_segmentation/"
               "ANALYSIS_20251122/data/msd_prostate/Task05_Prostate")
DEFAULT_OUT = ("/nfs/data/nii/data1/Analysis/camaret___in_context_segmentation/"
               "ANALYSIS_20251122/data/msd_prostate/npy")


def convert_one(base: str, img_path: Path, lbl_path: Path, out_dir: Path,
                overwrite: bool) -> tuple[str, dict | None, str | None]:
    """Convert one nnU-Net case into TWO channel-subjects. Returns (base, {subj: (meta, stats)}
    or None, error)."""
    try:
        img4d = nib.load(str(img_path))
        data4d = np.asarray(img4d.dataobj)
        if data4d.ndim != 4 or data4d.shape[-1] != 2:
            return base, None, f"expected 4D (X,Y,Z,2), got {data4d.shape}"
        affine = np.asarray(img4d.affine, dtype=np.float64)

        msk = nib.as_closest_canonical(nib.load(str(lbl_path)))
        lab = np.ascontiguousarray(msk.get_fdata(dtype=np.float32))
        if not np.array_equal(lab, np.round(lab)):
            return base, None, "mask is not integral-valued"
        lab = np.round(lab).astype(np.int32)
        if lab.min() < 0 or lab.max() > 2:
            return base, None, f"mask range [{lab.min()}, {lab.max()}] outside 0..2"
        lab = lab.astype(np.uint8)

        results = {}
        for ch_name, ch_idx in CHANNELS.items():
            subj = f"{base}_{ch_name}"
            ch_img = nib.as_closest_canonical(nib.Nifti1Image(data4d[..., ch_idx], affine))
            ch_affine = np.asarray(ch_img.affine, dtype=np.float64)
            if ch_img.shape[:3] != msk.shape[:3]:
                return base, None, (f"{subj}: shape mismatch img{ch_img.shape[:3]} "
                                    f"msk{msk.shape[:3]}")
            if not np.allclose(ch_affine, np.asarray(msk.affine), atol=1e-3):
                return base, None, f"{subj}: image/mask affine mismatch"

            spacing = [float(x) for x in nib.affines.voxel_sizes(ch_affine)[:3]]
            meta = {"spacing": spacing, "shape": [int(x) for x in ch_img.shape[:3]],
                    "affine": ch_affine.tolist()}
            raw = np.ascontiguousarray(ch_img.get_fdata(dtype=np.float32))
            stats = mri_stats(raw)

            subj_dir = out_dir / subj
            img_out, lbl_out = subj_dir / "ct_raw.npy", subj_dir / "label.npy"
            if not overwrite and img_out.exists() and lbl_out.exists():
                meta["classes_present"] = sorted(int(v) for v in np.unique(lab) if v != 0)
                results[subj] = (meta, stats)
                continue

            subj_dir.mkdir(parents=True, exist_ok=True)
            np.save(img_out, raw.astype(np.float32))
            np.save(lbl_out, lab)
            meta["classes_present"] = sorted(int(v) for v in np.unique(lab) if v != 0)
            results[subj] = (meta, stats)
        return base, results, None
    except Exception as exc:  # noqa: BLE001
        return base, None, f"{type(exc).__name__}: {exc}"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=DEFAULT_SRC, help="Task05_Prostate root")
    ap.add_argument("--out", default=DEFAULT_OUT, help="npy root to write")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    src, out = Path(args.src), Path(args.out)
    images_dir, labels_dir = src / "imagesTr", src / "labelsTr"
    if not images_dir.is_dir() or not labels_dir.is_dir():
        raise SystemExit(f"expected {images_dir} and {labels_dir}")

    cases = []
    for img_path in sorted(images_dir.glob("prostate_*.nii.gz")):
        base = img_path.name.removesuffix(".nii.gz")
        lbl_path = labels_dir / img_path.name
        if not lbl_path.exists():
            print(f"  [skip] {base}: no matching label")
            continue
        cases.append((base, img_path, lbl_path))
    print(f"{len(cases)} cases (-> {len(cases) * len(CHANNELS)} channel-subjects)  "
          f"{src} -> {out}")
    out.mkdir(parents=True, exist_ok=True)

    spacings, ct_stats, errors = {}, {}, []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(convert_one, b, i, m, out, args.overwrite) for b, i, m in cases]
        for n, fut in enumerate(as_completed(futs), 1):
            base, results, err = fut.result()
            if err:
                errors.append(f"{base}: {err}")
                print(f"  [FAIL] {base}: {err}")
            else:
                for subj, (meta, stats) in results.items():
                    spacings[subj] = meta
                    ct_stats[subj] = stats
            if n % 10 == 0:
                print(f"  {n}/{len(cases)}")

    with open(out / "spacings.json", "w") as f:
        json.dump({k: spacings[k] for k in sorted(spacings)}, f, indent=1)
    with open(out / "ct_stats.json", "w") as f:
        json.dump({k: ct_stats[k] for k in sorted(ct_stats)}, f, indent=1)

    print(f"\nwrote {len(spacings)} channel-subjects from {len(cases) - len(errors)} cases, "
          f"{len(errors)} failed")
    if spacings:
        sp = np.array([m["spacing"] for m in spacings.values()])
        sh = np.array([m["shape"] for m in spacings.values()])
        print(f"  spacing: min {sp.min(0).round(2)} max {sp.max(0).round(2)}")
        print(f"  shape: min {sh.min(0)} median {np.median(sh, 0)} max {sh.max(0)}")
        print(f"  -> {out/'spacings.json'}, {out/'ct_stats.json'}")
    for e in errors:
        print(f"  {e}")


if __name__ == "__main__":
    main()
