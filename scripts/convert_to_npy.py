"""
Convert TotalSegmentator / TotalSegMRI subjects from .nii.gz to .npy for fast data loading.

Per subject written (always):
  ct.npy     — float16, normalised, native resolution (D,H,W)
  label.npy  — uint8, merged label volume using ALL_CLASSES ordering (0=bg), native resolution

With --size D H W (e.g. --size 128 128 128), also writes:
  ct_DxHxW.npy     — float16, isotropic resize (longest axis → D) + Gaussian AA + zero-pad to cube
  label_DxHxW.npy  — uint8, same isotropic resize with nearest-neighbour interpolation

Modality handling (--modality):
  ct  (default) : reads ct.nii.gz; clips HU to [CT_CLIP_MIN, CT_CLIP_MAX]; global z-score.
  mri           : reads mri.nii.gz; clips to [0, per-volume 99.5th percentile of foreground];
                  per-volume z-score (foreground mean/std).  Output still named ct.npy so the
                  dataloader needs no changes.

Usage
-----
  python scripts/convert_to_npy.py [--data DIR] [--workers N] [--overwrite]
  python scripts/convert_to_npy.py --size 128 128 128
  python scripts/convert_to_npy.py --data /nfs/data/nii/data1/Analysis/camaret___in_context_segmentation/ANALYSIS_20251122/data/totalsegmri --modality mri --size 128 128 128 --overwrite
"""

import argparse
import csv
import json
import multiprocessing as mp
import os
import sys
import time
import traceback
from pathlib import Path

import nibabel as nib
import numpy as np
import scipy.ndimage as ndi
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.totalseg_classes import ALL_CLASSES
from data.totalseg_total_map import remap_ts_total
from src.totalseg_dataset import (
    CT_CLIP_MIN, CT_CLIP_MAX, CT_MEAN, CT_STD,
    normalize_ct, mri_stats, normalize_mri,
)

ROOT = Path(__file__).resolve().parents[1]

COHORT_JSON = ROOT / "experiments/3d/universal_coords/coords_paths_chemotox.json"

# label channels each source emits (written as {name}.npy; "label" is the primary mask)
SOURCE_LABELS = {"totalseg": ["label"], "chemotox": ["label", "bc"]}


def _iso_resize(vol: np.ndarray, target: tuple, order: int = 1, aa: bool = True,
                spacing: tuple | None = None) -> np.ndarray:
    """Isotropic resize: scale longest physical axis to target[0], zero-pad shorter axes.

    With aa=True (and order > 0), applies a per-axis Gaussian blur before
    downsampling (σ = 0.5*(s/n − 1)) to suppress aliasing.  Upsampled axes skip
    the blur.  Uses order=0 for label volumes (nearest-neighbour, no AA needed).

    spacing: voxel size in mm (D, H, W). When provided, scaling is based on
    physical extent (shape × spacing) rather than voxel count alone — critical
    for anisotropic 2-D multi-slice MRI where slice thickness >> in-plane spacing.
    """
    T = target[0]
    if spacing is not None:
        # Scale so the longest physical dimension fits in T voxels; preserve aspect ratio.
        phys = tuple(s * sp for s, sp in zip(vol.shape, spacing))
        new_shape = tuple(min(T, max(1, round(T * p / max(phys)))) for p in phys)
    else:
        scale = T / max(vol.shape)
        new_shape = tuple(min(T, max(1, round(s * scale))) for s in vol.shape)
    if aa and order > 0:
        sigma = [max(0.0, 0.5 * (s / n - 1)) for s, n in zip(vol.shape, new_shape)]
        if any(s > 0.1 for s in sigma):
            vol = ndi.gaussian_filter(vol, sigma=sigma)
    zoom = tuple(n / s for n, s in zip(new_shape, vol.shape))
    resized = ndi.zoom(vol, zoom, order=order)
    out = np.zeros(target, dtype=vol.dtype)
    pad = [(T - s) // 2 for s in new_shape]
    sl = tuple(slice(p, p + s) for p, s in zip(pad, new_shape))
    out[sl] = resized
    return out


def _resample_to_spacing(vol: np.ndarray, native_sp, target_sp: float,
                         order: int = 1) -> np.ndarray:
    """Resample `vol` from native voxel spacing (mm, per axis) to `target_sp` mm
    isotropic. order=1 (trilinear) for images, order=0 (nearest) for label maps.
    out_shape[i] = round(shape[i] * native_sp[i] / target_sp)."""
    zoom = [float(ns) / float(target_sp) for ns in native_sp]
    out = ndi.zoom(vol, zoom, order=order)
    return out.astype(vol.dtype, copy=False)


def _default_data_dir() -> str:
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
        cfg = compose(config_name="config")
    return cfg.paths.totalseg

_CLASS_TO_IDX = {cls: i + 1 for i, cls in enumerate(ALL_CLASSES)}  # 1-indexed

# Verify constants match the expected fingerprint values at import time (fast sanity check).
assert CT_CLIP_MIN == -1007.0 and CT_CLIP_MAX == 1573.0, "unexpected CT clip constants"


# Normalisation delegates to the shared helpers in src.totalseg_dataset so the values
# written here are byte-identical to what the raw-CT loader path produces on the fly.
_normalise_ct = normalize_ct


def enumerate_subjects(source: str, data, out, limit=None) -> list[dict]:
    """Return a list of per-subject task dicts (subj_id, out_dir, inputs)."""
    tasks: list[dict] = []
    if source == "totalseg":
        for s in sorted(p for p in Path(data).iterdir() if p.is_dir()):
            tasks.append({"subj_id": s.name, "out_dir": str(Path(out) / s.name),
                          "inputs": {"subj_dir": str(s)}})
    elif source == "chemotox":
        json_path = data if data else COHORT_JSON
        cohort = json.load(open(json_path))
        for key, rec in cohort.items():
            subj_id = key.replace("#", "_")
            tasks.append({"subj_id": subj_id, "out_dir": str(Path(out) / subj_id),
                          "inputs": {"img": rec["img"], "totalseg": rec["totalseg"],
                                     "bclabels": rec["bclabels"]}})
    else:
        raise ValueError(f"unknown source {source!r}")
    if limit is not None:
        tasks = tasks[:limit]
    return tasks


def load_raw(task: dict):
    """(raw_ct f32, native_spacing [3], {label_name: array}) for a chemotox subject.

    All three volumes share one native grid, so no canonicalization is needed — read
    raw dataobj and take spacing from the img affine. (The totalseg source does its own
    CT+segmentations reading inside _convert_totalseg to stay byte-identical.)"""
    assert task["source"] == "chemotox", "load_raw serves the chemotox source only"
    p = task["inputs"]
    img = nib.load(p["img"])
    raw = np.asanyarray(img.dataobj).astype(np.float32)
    sp = [abs(float(x)) for x in nib.affines.voxel_sizes(img.affine)[:3]]
    ts = np.asanyarray(nib.load(p["totalseg"]).dataobj)
    label = remap_ts_total(ts)
    bc = np.asanyarray(nib.load(p["bclabels"]).dataobj)[..., 0].astype(np.uint8)
    return raw, sp, {"label": label, "bc": bc}


def _convert_totalseg(task: dict) -> tuple[str, str, list | None, list | None, dict | None]:
    """Convert one TotalSegmentator subject.

    Returns (subject_id, status, native_spacing, native_shape, mri_stats).
    native_spacing and native_shape are None when the NIfTI was not read
    (skipped subjects or sized-only runs where the native files already exist).
    mri_stats is the per-volume MRI normalisation stats dict (for ct_stats.json) when
    store_raw + modality=mri, else None.
    """
    subj_dir = Path(task["inputs"]["subj_dir"])
    overwrite = task["overwrite"]
    size = task["size"]
    modality = task["modality"]
    store_raw = task["store_raw"]
    subj = subj_dir.name

    ct_out    = subj_dir / "ct.npy"
    ct_raw_out = subj_dir / "ct_raw.npy"
    label_out = subj_dir / "label.npy"

    size_str    = f"{size[0]}x{size[1]}x{size[2]}" if size else None
    ct_sized    = subj_dir / f"ct_{size_str}.npy"  if size else None
    label_sized = subj_dir / f"label_{size_str}.npy" if size else None

    need_native = overwrite or not (ct_out.exists() and label_out.exists())
    need_raw    = store_raw and (overwrite or not ct_raw_out.exists())
    need_sized  = size is not None and (
        overwrite or not (ct_sized.exists() and label_sized.exists())
    )

    if not need_native and not need_raw and not need_sized:
        return subj, "skip", None, None, None

    try:
        vol = label = None
        native_spacing = native_shape = None
        stats = None

        if need_native or need_raw:
            img_fname = "mri.nii.gz" if modality == "mri" else "ct.nii.gz"
            img_path  = subj_dir / img_fname
            ct_img    = nib.as_closest_canonical(nib.load(str(img_path)))
            native_spacing = [float(x) for x in nib.affines.voxel_sizes(ct_img.affine)[:3]]
            raw = ct_img.get_fdata(dtype=np.float32)   # raw intensities (HU for CT)
            native_shape = list(raw.shape)

            if modality == "mri":
                stats = mri_stats(raw)                  # whole-volume stats (sidecar)
                if need_raw:
                    # MRI has no canonical integer range; keep raw as float16.
                    np.save(ct_raw_out, raw.astype(np.float16))
                vol = normalize_mri(raw, stats)
            else:
                if need_raw:
                    # CT HU are integers -> int16 is lossless and same 2 B/voxel as float16.
                    np.save(ct_raw_out, np.clip(np.round(raw), -32768, 32767).astype(np.int16))
                vol = _normalise_ct(raw)

        if need_native:
            seg_dir = subj_dir / "segmentations"
            label = np.zeros(vol.shape, dtype=np.uint8)
            for cls, idx in _CLASS_TO_IDX.items():
                mask_path = seg_dir / f"{cls}.nii.gz"
                if not mask_path.exists():
                    continue
                mask = nib.as_closest_canonical(nib.load(str(mask_path))).get_fdata(dtype=np.float32) > 0
                label[mask] = idx

            np.save(ct_out, vol.astype(np.float16))
            np.save(label_out, label)

        if need_sized:
            if vol is None:
                vol   = np.load(ct_out,    mmap_mode="r").astype(np.float32)
                label = np.load(label_out, mmap_mode="r")
            if native_shape is None:
                native_shape = list(vol.shape)
            if native_spacing is None:
                # Read spacing from the NIfTI header (header-only, no data load)
                img_fname = "mri.nii.gz" if modality == "mri" else "ct.nii.gz"
                img_path  = subj_dir / img_fname
                if img_path.exists():
                    canonical = nib.as_closest_canonical(nib.load(str(img_path)))
                    native_spacing = [float(x) for x in nib.affines.voxel_sizes(canonical.affine)[:3]]

            sp = tuple(native_spacing) if native_spacing else None
            np.save(ct_sized,    _iso_resize(vol,   size, order=1, aa=True,  spacing=sp).astype(np.float16))
            np.save(label_sized, _iso_resize(label, size, order=0, aa=False, spacing=sp))

    except Exception:
        return subj, traceback.format_exc(), None, None, None

    return subj, "ok", native_spacing, native_shape, stats


def _convert_chemotox(task: dict):
    """Convert one chemotox subject to the out tree. Returns (subj_id, status, sp, shape, None)."""
    subj_id = task["subj_id"]
    out_dir = Path(task["out_dir"]); out_dir.mkdir(parents=True, exist_ok=True)
    overwrite = task["overwrite"]; size = task["size"]; target_sp = task["target_spacing"]
    label_names = SOURCE_LABELS["chemotox"]
    ct_out = out_dir / "ct.npy"
    label_outs = {n: out_dir / f"{n}.npy" for n in label_names}
    if (ct_out.exists() and all(p.exists() for p in label_outs.values())
            and not overwrite and size is None):
        return subj_id, "skip", None, None, None
    try:
        raw, native_sp, labels = load_raw(task)
        vol = _normalise_ct(raw)
        if target_sp is not None:
            vol = _resample_to_spacing(vol, native_sp, target_sp, order=1)
            labels = {n: _resample_to_spacing(a, native_sp, target_sp, order=0)
                      for n, a in labels.items()}
            out_sp = [float(target_sp)] * 3
        else:
            out_sp = native_sp
        out_shape = list(vol.shape)
        np.save(ct_out, vol.astype(np.float16))
        for n, a in labels.items():
            np.save(label_outs[n], a.astype(np.uint8))
        if size is not None:  # optional fixed-cube sized variants (primary label only)
            size_str = f"{size[0]}x{size[1]}x{size[2]}"
            sp = tuple(out_sp)
            np.save(out_dir / f"ct_{size_str}.npy",
                    _iso_resize(vol.astype(np.float32), size, order=1, aa=True, spacing=sp).astype(np.float16))
            np.save(out_dir / f"label_{size_str}.npy",
                    _iso_resize(labels["label"], size, order=0, aa=False, spacing=sp))
    except Exception:
        return subj_id, traceback.format_exc(), None, None, None
    return subj_id, "ok", out_sp, out_shape, None


def convert_subject(task: dict):
    if task["source"] == "totalseg":
        return _convert_totalseg(task)
    return _convert_chemotox(task)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None,
                        help="dataset root; defaults to paths.totalseg in configs/config.yaml")
    parser.add_argument("--workers", type=int, default=min(20, os.cpu_count()),
                        help="parallel worker processes (default: min(20, cpu_count))")
    parser.add_argument("--overwrite", action="store_true",
                        help="reconvert even if .npy files already exist")
    parser.add_argument("--size", nargs=3, type=int, metavar=("D", "H", "W"),
                        default=None,
                        help="also write pre-resized ct_DxHxW.npy and label_DxHxW.npy")
    parser.add_argument("--modality", choices=["ct", "mri"], default="ct",
                        help="ct (default): reads ct.nii.gz with HU normalisation; "
                             "mri: reads mri.nii.gz with per-volume percentile z-score")
    parser.add_argument("--store-raw", action="store_true",
                        help="also write native ct_raw.npy (raw intensities: int16 HU for CT, "
                             "float16 for MRI) so the loader can normalise on the fly. For MRI "
                             "also writes per-volume stats to ct_stats.json.")
    parser.add_argument("--source", choices=["totalseg", "chemotox"], default="totalseg",
                        help="dataset source: totalseg (dir tree, default) or chemotox (JSON of paths)")
    parser.add_argument("--out", default=None,
                        help="output root; defaults to --data (in-place for totalseg)")
    parser.add_argument("--target-spacing", type=float, default=None, dest="target_spacing",
                        help="resample the native outputs to this mm-isotropic spacing "
                             "(default: keep full native)")
    parser.add_argument("--limit", type=int, default=None,
                        help="convert only the first N subjects (smoke test)")
    args = parser.parse_args()

    data_dir = args.data
    if data_dir is None:
        data_dir = str(COHORT_JSON) if args.source == "chemotox" else _default_data_dir()
    out_root = Path(args.out) if args.out else Path(data_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    subjects = enumerate_subjects(args.source, data_dir, out_root, args.limit)
    total = len(subjects)
    size = tuple(args.size) if args.size else None
    for t in subjects:
        t.update(overwrite=args.overwrite, size=size, target_spacing=args.target_spacing,
                 source=args.source, modality=args.modality, store_raw=args.store_raw)
    print(f"source={args.source} | {total} subjects | out={out_root} | "
          f"target_spacing={args.target_spacing} | size={size}")

    spacings_path = out_root / "spacings.json"
    if spacings_path.exists():
        with open(spacings_path) as f:
            spacings = json.load(f)
    else:
        spacings = {}
    stats_path = out_root / "ct_stats.json"
    if stats_path.exists():
        with open(stats_path) as f:
            ct_stats = json.load(f)
    else:
        ct_stats = {}

    done = ok = skipped = errors = 0
    t0 = time.time()
    with mp.Pool(processes=args.workers) as pool:
        for subj, status, sp, shape, subj_stats in pool.imap_unordered(
            convert_subject, subjects, chunksize=1
        ):
            done += 1
            if status == "ok":
                ok += 1
                if sp is not None and shape is not None:
                    spacings[subj] = {"spacing": sp, "shape": shape}
                if subj_stats is not None:
                    ct_stats[subj] = subj_stats
            elif status == "skip":
                skipped += 1
            else:
                errors += 1; print(f"\n[ERROR] {subj}:\n{status}")
            elapsed = time.time() - t0; rate = done / elapsed if elapsed else 0
            print(f"\r  {done}/{total} ok={ok} skip={skipped} err={errors} "
                  f"{rate:.1f} subj/s", end="", flush=True)

    if spacings:
        with open(spacings_path, "w") as f:
            json.dump(spacings, f)
        print(f"\nSpacings -> {spacings_path} ({len(spacings)})")
    if ct_stats:
        with open(stats_path, "w") as f:
            json.dump(ct_stats, f)
    # meta.csv for sources with no native split (chemotox): all subjects -> test
    if args.source == "chemotox":
        with open(out_root / "meta.csv", "w", newline="") as f:
            w = csv.writer(f, delimiter=";"); w.writerow(["image_id", "split"])
            for s in sorted(spacings): w.writerow([s, "test"])
        print(f"meta.csv -> {out_root / 'meta.csv'} ({len(spacings)} test)")
    print(f"\nDone in {(time.time()-t0)/60:.1f} min — ok={ok} skip={skipped} err={errors}")


if __name__ == "__main__":
    mp.set_start_method("forkserver", force=True)
    main()
