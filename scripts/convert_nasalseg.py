"""Convert NasalSeg (.nrrd) -> per-subject .npy at NATIVE anisotropic spacing, RAS.

Mirrors scripts/convert_flare22.py: native grid only, every resampling decision
deferred to the dataloader (src/providers/nasalseg.py).

Three things this dataset needs that FLARE22 did not (see docs/datasets/nasalseg.md):

1. DE-DUPLICATION. 130 files hold only 107 unique cases — 19 groups of byte-identical
   image AND label. For in-context eval a duplicate drawn as its twin's context is an
   exact clone of the target (leakage toward the trivial-matching ceiling) that the
   `SELF` check misses, because the case ids differ. Groups are detected here by hashing
   the raw arrays, not hardcoded; the first id of each group is kept.

2. LPS -> RAS. All 130 files are left-posterior-superior, the mirror of what
   `nib.as_closest_canonical` yields in our other converters. Without the flip the model
   sees mirrored heads. Done by reversing axes 0 and 1 and rebuilding the affine.

3. GEOMETRY FROM THE IMAGE HEADER. 12 `_seg.nrrd` headers are junk (9 with a negative
   z-direction and a non-overlapping origin, 3 with z-spacing 1.0 vs the image's 1.5)
   while the arrays are index-aligned — verified by contrast polarity (labels mark AIR;
   as-is the mask is 93-99% below -300 HU, z-flipped only 45-74%). So only shape equality
   is asserted, and all geometry comes from the image.

Layout written under --out:
    P001/ct_raw.npy   (D,H,W) int16, RAS, native grid
    P001/label.npy    (D,H,W) uint8, 0=bg, 1..5
    spacings.json     {subj: {spacing, shape, affine, duplicates_of_this}}
    duplicates.json   {kept_subj: [dropped ids]}

Usage:
    python scripts/convert_nasalseg.py --workers 16
"""
import argparse
import hashlib
import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import nrrd
import numpy as np

# Label index -> name. Verified geometrically (the files carry no label metadata):
# 1/2 stand 12-15 mm clear of the mid-sagittal plane, 3/4 touch it (septum-adjacent),
# 5 is midline + posterior + inferior. See docs/datasets/nasalseg.md.
NASALSEG_CLASSES = [
    "maxillary_sinus_right", "maxillary_sinus_left",
    "nasal_cavity_right", "nasal_cavity_left", "nasopharynx",
]

DEFAULT_SRC = ("/nfs/data/nii/data1/Analysis/camaret___in_context_segmentation/"
               "ANALYSIS_20251122/data/nasalseg")
DEFAULT_OUT = ("/nfs/data/nii/data1/Analysis/camaret___in_context_segmentation/"
               "ANALYSIS_20251122/data/nasalseg/npy")


def _hash(subj: str, src: Path) -> tuple[str, str, str]:
    """(subj, image-array hash, label-array hash) for duplicate detection."""
    img, _ = nrrd.read(str(src / "images" / f"{subj}_img.nrrd"))
    seg, _ = nrrd.read(str(src / "labels" / f"{subj}_seg.nrrd"))
    return (subj, hashlib.sha1(np.ascontiguousarray(img).tobytes()).hexdigest(),
            hashlib.sha1(np.ascontiguousarray(seg).tobytes()).hexdigest())


def convert_one(subj: str, src: Path, out_dir: Path,
                overwrite: bool) -> tuple[str, dict | None, str | None]:
    """Convert one case to RAS .npy. Returns (subj, meta, error)."""
    try:
        img, ih = nrrd.read(str(src / "images" / f"{subj}_img.nrrd"))
        seg, _sh = nrrd.read(str(src / "labels" / f"{subj}_seg.nrrd"))

        # Geometry from the IMAGE header only — 12 seg headers are junk (see module docstring).
        if img.shape != seg.shape:
            return subj, None, f"shape mismatch img{img.shape} seg{seg.shape}"
        if str(ih.get("space")) != "left-posterior-superior":
            return subj, None, f"unexpected space frame {ih.get('space')!r}"
        sd = np.asarray(ih["space directions"], dtype=np.float64)
        if not np.allclose(sd, np.diag(np.diag(sd))):
            return subj, None, "non-diagonal space directions (needs a full resample)"
        d = np.diag(sd).astype(np.float64)
        if not np.all(d > 0):
            return subj, None, f"non-positive image spacing {d.tolist()}"
        origin = np.asarray(ih["space origin"], dtype=np.float64)

        # LPS -> RAS: negate x,y. Keep the diagonal positive by reversing axes 0 and 1,
        # which shifts the origin to what was the far corner on those axes.
        img = np.ascontiguousarray(img[::-1, ::-1, :])
        seg = np.ascontiguousarray(seg[::-1, ::-1, :])
        n = np.asarray(img.shape, dtype=np.float64)
        ras_origin = np.array([-origin[0] - d[0] * (n[0] - 1),
                               -origin[1] - d[1] * (n[1] - 1),
                                origin[2]])
        affine = np.eye(4)
        affine[:3, :3] = np.diag(d)
        affine[:3, 3] = ras_origin

        meta = {"spacing": [float(x) for x in d],
                "shape": [int(x) for x in img.shape],
                "affine": affine.tolist()}

        ct_out, lbl_out = out_dir / subj / "ct_raw.npy", out_dir / subj / "label.npy"
        if not overwrite and ct_out.exists() and lbl_out.exists():
            return subj, meta, None

        raw = img.astype(np.float64)
        if not np.array_equal(raw, np.round(raw)):
            return subj, None, "CT is not integral-valued; int16 would be lossy"
        if raw.min() < -32768 or raw.max() > 32767:
            return subj, None, f"CT range [{raw.min()}, {raw.max()}] exceeds int16"
        lab = seg.astype(np.int32)
        if lab.min() < 0 or lab.max() > len(NASALSEG_CLASSES):
            return subj, None, f"label range [{lab.min()}, {lab.max()}] outside 0..5"

        (out_dir / subj).mkdir(parents=True, exist_ok=True)
        np.save(ct_out, np.round(raw).astype(np.int16))
        np.save(lbl_out, lab.astype(np.uint8))
        meta["hu_range"] = [float(raw.min()), float(raw.max())]
        meta["classes_present"] = sorted(int(v) for v in np.unique(lab) if v != 0)
        # Sanity: labels mark AIR cavities. A mirrored/misaligned mask would not be air.
        meta["frac_air_under_label"] = float((raw[lab > 0] < -300).mean()) if (lab > 0).any() else 0.0
        return subj, meta, None
    except Exception as exc:  # noqa: BLE001
        return subj, None, f"{type(exc).__name__}: {exc}"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=DEFAULT_SRC, help="dir containing images/ and labels/")
    ap.add_argument("--out", default=DEFAULT_OUT, help="npy root to write")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--keep-duplicates", action="store_true",
                    help="convert all files instead of one per unique volume (NOT for eval)")
    args = ap.parse_args()

    src, out = Path(args.src), Path(args.out)
    if not (src / "images").is_dir() or not (src / "labels").is_dir():
        raise SystemExit(f"expected {src/'images'} and {src/'labels'}")
    subs = sorted(p.name[: -len("_img.nrrd")] for p in (src / "images").glob("*_img.nrrd"))
    print(f"{len(subs)} files under {src}")

    # --- de-duplicate on the raw arrays -------------------------------------
    print("  hashing arrays to find duplicates...")
    by_key = defaultdict(list)
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for fut in as_completed([ex.submit(_hash, s, src) for s in subs]):
            s, hi, hs = fut.result()
            by_key[(hi, hs)].append(s)
    groups = {sorted(v)[0]: sorted(v)[1:] for v in by_key.values()}
    dup_groups = {k: v for k, v in groups.items() if v}
    keep = sorted(groups) if not args.keep_duplicates else subs
    print(f"  {len(subs)} files -> {len(groups)} unique volumes "
          f"({len(dup_groups)} duplicate groups, {sum(len(v) for v in dup_groups.values())} dropped)")
    for k in sorted(dup_groups):
        print(f"     {k} == {', '.join(dup_groups[k])}")

    out.mkdir(parents=True, exist_ok=True)
    metas, errors = {}, []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(convert_one, s, src, out, args.overwrite) for s in keep]
        for n, fut in enumerate(as_completed(futs), 1):
            subj, meta, err = fut.result()
            if err:
                errors.append(f"{subj}: {err}")
                print(f"  [FAIL] {subj}: {err}")
            else:
                meta["duplicates_of_this"] = groups.get(subj, [])
                metas[subj] = meta
            if n % 25 == 0:
                print(f"  {n}/{len(keep)}")

    with open(out / "spacings.json", "w") as f:
        json.dump({k: metas[k] for k in sorted(metas)}, f, indent=1)
    with open(out / "duplicates.json", "w") as f:
        json.dump({k: dup_groups[k] for k in sorted(dup_groups)}, f, indent=1)

    print(f"\nwrote {len(metas)} cases, {len(errors)} failed")
    if metas:
        air = np.array([m["frac_air_under_label"] for m in metas.values()])
        print(f"  air fraction under label: min {air.min():.1%} med {np.median(air):.1%} "
              f"(low values would mean a misaligned/mirrored mask)")
        sp = np.array([m["spacing"] for m in metas.values()])
        print(f"  in-plane {sp[:, 0].min():.3f}-{sp[:, 0].max():.3f} mm  "
              f"z {sorted(set(np.round(sp[:, 2], 3)))} mm")
        print(f"  -> {out/'spacings.json'}, {out/'duplicates.json'}")
    for e in errors:
        print(f"  {e}")


if __name__ == "__main__":
    main()
