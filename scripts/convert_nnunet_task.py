"""
Convert a TotalSegmentator subtask training dataset (nnU-Net raw format) into the
per-subject .npy layout consumed by TotalSegInContextDataset.

The subtask datasets published on Zenodo / GitHub releases (hip_implant,
pleural_pericard_effusion, liver_lesions, lung_nodules, ...) are self-contained
nnU-Net raw datasets with their OWN re-anonymised subject IDs — disjoint from the
main `total` cohort (s0000...). This script ingests one such dataset as a new root
so its class(es) can be used as in-context targets (eval-only by default).

Source layout (extract the Zenodo/release .zip first):
    <src>/dataset.json
    <src>/imagesTr/<sid>_0000.nii.gz     # CT
    <src>/labelsTr/<sid>.nii.gz          # integer label map

Output layout (one dir per subject, matching convert_to_npy.py):
    <out>/<sid>/ct.npy                   # float16, HU-normalised, native
    <out>/<sid>/label.npy                # uint8, ALL_CLASSES index encoding
    <out>/<sid>/ct_DxHxW.npy             # (with --size) pre-resized CT
    <out>/<sid>/label_DxHxW.npy          # (with --size) pre-resized label
    <out>/meta.csv                       # image_id;split  (all split=test by default)
    <out>/spacings.json                  # {sid: {spacing, shape}}

Label mapping: each non-background entry in dataset.json is written into label.npy
under its index in ALL_CLASSES (data/totalseg_classes.py), so the dataloader's
_ALL_CLASSES_IDX resolves it with no code change. Override the name mapping with
--map "src_name=our_class" when the dataset.json name differs from ALL_CLASSES.

Usage
-----
  # 1. download + unzip the Zenodo record, e.g. Dataset260_hip_implant
  # 2. convert (all subjects -> eval/test split), also emit 64^3 fast-path files
  python scripts/convert_nnunet_task.py \
      --src /path/Dataset260_hip_implant \
      --out ${paths.totalseg}/../totalseg_hip_implant \
      --size 64 64 64
"""

import argparse
import json
import multiprocessing as mp
import sys
import time
import traceback
from pathlib import Path

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data.totalseg_classes import ALL_CLASSES
from scripts.convert_to_npy import _iso_resize, _normalise_ct

_CLASS_TO_IDX = {cls: i + 1 for i, cls in enumerate(ALL_CLASSES)}  # 1-indexed


def _read_label_map(src: Path, overrides: dict[str, str]) -> dict[int, int]:
    """Build {source_label_value -> ALL_CLASSES index} from dataset.json.

    dataset.json labels are {name: value}. Each name is mapped to our class of the
    same name unless renamed via --map. Names absent from ALL_CLASSES are skipped
    with a warning (so you can convert a subset of a multi-class task).
    """
    with open(src / "dataset.json") as f:
        labels = json.load(f)["labels"]
    src_to_our: dict[int, int] = {}
    for name, value in labels.items():
        if int(value) == 0:  # background
            continue
        our_name = overrides.get(name, name)
        our_idx = _CLASS_TO_IDX.get(our_name)
        if our_idx is None:
            print(f"  [warn] '{name}' -> '{our_name}' not in ALL_CLASSES; skipping "
                  f"(add it to data/totalseg_classes.py to include).")
            continue
        src_to_our[int(value)] = our_idx
    if not src_to_our:
        raise SystemExit("No source labels mapped to ALL_CLASSES — nothing to write.")
    print(f"  label map (src_value -> our_idx): {src_to_our}")
    return src_to_our


def convert_subject(task: tuple) -> tuple[str, str, list | None, list | None]:
    """Convert one subject; returns (sid, status, native_spacing, native_shape)."""
    sid, img_path, lbl_path, out_dir, size, label_map, overwrite = task
    img_path, lbl_path, out_dir = Path(img_path), Path(lbl_path), Path(out_dir)

    ct_out, label_out = out_dir / "ct.npy", out_dir / "label.npy"
    size_str = f"{size[0]}x{size[1]}x{size[2]}" if size else None
    ct_sized = out_dir / f"ct_{size_str}.npy" if size else None
    label_sized = out_dir / f"label_{size_str}.npy" if size else None

    if not overwrite and ct_out.exists() and label_out.exists() and (
        size is None or (ct_sized.exists() and label_sized.exists())
    ):
        return sid, "skip", None, None

    try:
        out_dir.mkdir(parents=True, exist_ok=True)

        ct_img = nib.as_closest_canonical(nib.load(str(img_path)))
        native_spacing = [float(x) for x in nib.affines.voxel_sizes(ct_img.affine)[:3]]
        vol = _normalise_ct(ct_img.get_fdata(dtype=np.float32))
        native_shape = list(vol.shape)

        seg = nib.as_closest_canonical(nib.load(str(lbl_path))).get_fdata(dtype=np.float32)
        seg = np.rint(seg).astype(np.int32)
        label = np.zeros(vol.shape, dtype=np.uint8)
        for src_val, our_idx in label_map.items():
            label[seg == src_val] = our_idx

        np.save(ct_out, vol.astype(np.float16))
        np.save(label_out, label)

        if size:
            sp = tuple(native_spacing)
            np.save(ct_sized, _iso_resize(vol, size, order=1, aa=True, spacing=sp).astype(np.float16))
            np.save(label_sized, _iso_resize(label, size, order=0, aa=False, spacing=sp))
    except Exception:
        return sid, traceback.format_exc(), None, None

    return sid, "ok", native_spacing, native_shape


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="extracted nnU-Net dataset dir (imagesTr/labelsTr/dataset.json)")
    p.add_argument("--out", required=True, help="output dataset root (created)")
    p.add_argument("--size", nargs=3, type=int, metavar=("D", "H", "W"), default=None,
                   help="also write pre-resized ct_DxHxW.npy / label_DxHxW.npy")
    p.add_argument("--map", action="append", default=[], metavar="src_name=our_class",
                   help="rename a dataset.json label to an ALL_CLASSES name (repeatable)")
    p.add_argument("--split", default="test", help="split written to meta.csv for all subjects")
    p.add_argument("--workers", type=int, default=min(16, mp.cpu_count()))
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    src, out = Path(args.src), Path(args.out)
    size = tuple(args.size) if args.size else None
    overrides = dict(kv.split("=", 1) for kv in args.map)
    label_map = _read_label_map(src, overrides)

    images = sorted((src / "imagesTr").glob("*_0000.nii.gz"))
    tasks = []
    for img in images:
        sid = img.name[: -len("_0000.nii.gz")]
        lbl = src / "labelsTr" / f"{sid}.nii.gz"
        if not lbl.exists():
            print(f"  [warn] no label for {sid}; skipping")
            continue
        tasks.append((sid, str(img), str(lbl), str(out / sid), size, label_map, args.overwrite))

    print(f"Converting {len(tasks)} subjects  |  workers={args.workers}  |  "
          f"size={size or 'native only'}  |  split={args.split}")
    out.mkdir(parents=True, exist_ok=True)

    spacings: dict = {}
    if (out / "spacings.json").exists():
        spacings = json.loads((out / "spacings.json").read_text())

    done = ok = skipped = errors = 0
    rows: list[str] = []
    t0 = time.time()
    with mp.Pool(processes=args.workers) as pool:
        for sid, status, sp, shape in pool.imap_unordered(convert_subject, tasks, chunksize=1):
            done += 1
            if status in ("ok", "skip"):
                rows.append(f"{sid};{args.split}")
            if status == "ok":
                ok += 1
                if sp is not None:
                    spacings[sid] = {"spacing": sp, "shape": shape}
            elif status == "skip":
                skipped += 1
            else:
                errors += 1
                print(f"\n[ERROR] {sid}:\n{status}")
            print(f"\r  {done}/{len(tasks)}  ok={ok}  skip={skipped}  err={errors}", end="", flush=True)

    (out / "meta.csv").write_text("image_id;split\n" + "\n".join(rows) + "\n")
    (out / "spacings.json").write_text(json.dumps(spacings))
    print(f"\nDone in {(time.time()-t0)/60:.1f} min  —  ok={ok}  skip={skipped}  err={errors}")
    print(f"meta.csv + spacings.json written to {out}")


if __name__ == "__main__":
    mp.set_start_method("forkserver", force=True)
    main()
