"""
Pre-resize BiomedParseData PNGs to memmap-able uint8 arrays for fast dataloading.

BiomedParseData ships every image/mask as a 1024x1024 RGBA PNG. Decoding one down
to a small grayscale tensor costs ~33 ms (pure CPU: zlib inflate + RGBA->luma),
which makes `BiomedParseDataset` decode-bound: even at 32 DataLoader workers it
tops out ~100 img/s, and each __getitem__ does up to 2*(K+1) decodes. This script
decodes every PNG ONCE and stores it at the training resolution, so the dataset can
memmap rows instead (~0.04 ms/img reload, ~760x faster; see docs/logs.md).

Why .npy stacks (not a single .npz like scripts/totalseg2d/to_npz.py): the corpus
is ~20 GB at 128px and NpzFile cannot be memory-mapped, so a monolithic npz would
force each persistent worker to hold its own in-RAM copy. Standalone uint8 .npy
stacks are memmap'd, so all workers share one OS-page-cached copy (COW-safe).

Output (per dataset key, e.g. ACDC, amos22/CT, MSD/Task01_BrainTumour):
    <out>/<split>/<ds_key>/images_{S}.npy   uint8 (N_img,  S, S)   row = image_idx
    <out>/<split>/<ds_key>/masks_{S}.npy    uint8 (N_mask, S, S)   {0,1} foreground
    <out>/<split>/<ds_key>/index_{S}.npz    image_paths, mask_paths (rel to data_root)

Row order matches BiomedParseDataset's own discovery (same sorted-glob first-seen
ordering), and the index stores PNG paths relative to data_root so the dataset
fast-path can map its existing absolute paths -> rows without re-deriving anything.

Resize semantics replicate the dataset exactly: image -> L, BILINEAR, /255; mask ->
L, NEAREST, (>0). Binarization happens after the nearest resize, as at load time.

Usage
-----
python scripts/datasets/biomedparse/to_npz.py --size 128            # all datasets, both splits
python scripts/datasets/biomedparse/to_npz.py --size 128 --datasets ACDC BreastUS --workers 32
python scripts/datasets/biomedparse/to_npz.py --size 256 --split test --overwrite
"""

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.datasets.biomedparse import (  # noqa: E402
    DATA_ROOT, _ABSENT_MASK, _SPLIT_DIRS, _discover_sources, _parse_mask_stem,
)


# ── PNG -> uint8 row (top-level so it is picklable by the process pool) ─────────

def _decode(args):
    """Decode one PNG to a uint8 (S, S) row. is_mask -> NEAREST + binarize."""
    path, size, is_mask = args
    im = Image.open(path).convert("L")
    if is_mask:
        if im.size != (size, size):
            im = im.resize((size, size), Image.NEAREST)
        return (np.asarray(im) > 0).astype(np.uint8)
    if im.size != (size, size):
        im = im.resize((size, size), Image.BILINEAR)
    return np.asarray(im, dtype=np.uint8)


# ── Per-dataset discovery: reproduce BiomedParseDataset's path/row ordering ─────

def _collect(img_dir, mask_dir):
    """Return (image_paths, mask_paths) in the dataset's exact discovery order.

    image_paths: unique image PNGs, first-seen order -> row == dataset image_idx.
    mask_paths : one per kept mask file (absent.png + orphan masks dropped), in
                 sorted mask-glob order -> row == the mask's slot in the dataset.
    """
    import glob

    image_paths, mask_paths = [], []
    seen_img = {}
    for mask_path in sorted(glob.glob(os.path.join(mask_dir, "*.png"))):
        if os.path.basename(mask_path) == _ABSENT_MASK:
            continue
        stem = os.path.splitext(os.path.basename(mask_path))[0]
        image_stem, *_ = _parse_mask_stem(stem)
        img_path = os.path.join(img_dir, image_stem + ".png")
        if not os.path.exists(img_path):
            continue  # orphan mask
        if image_stem not in seen_img:
            seen_img[image_stem] = len(image_paths)
            image_paths.append(img_path)
        mask_paths.append(mask_path)
    return image_paths, mask_paths


def _build_stack(paths, size, is_mask, ex):
    """Decode `paths` -> uint8 (N, size, size), filled in order via the pool."""
    out = np.empty((len(paths), size, size), dtype=np.uint8)
    chunk = max(1, len(paths) // 256)
    for row, arr in enumerate(ex.map(_decode, ((p, size, is_mask) for p in paths),
                                     chunksize=chunk)):
        out[row] = arr
    return out


def _process_source(ds_key, img_dir, mask_dir, split, size, data_root, out_root,
                    ex, overwrite):
    out_dir = Path(out_root) / split / ds_key
    img_npy = out_dir / f"images_{size}.npy"
    msk_npy = out_dir / f"masks_{size}.npy"
    idx_npz = out_dir / f"index_{size}.npz"
    if not overwrite and img_npy.exists() and msk_npy.exists() and idx_npz.exists():
        print(f"  [skip] {ds_key} (exists)")
        return 0, 0

    image_paths, mask_paths = _collect(img_dir, mask_dir)
    if not mask_paths:
        print(f"  [warn] {ds_key}: no usable masks")
        return 0, 0

    t = time.perf_counter()
    images = _build_stack(image_paths, size, False, ex)
    masks = _build_stack(mask_paths, size, True, ex)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(img_npy, images)
    np.save(msk_npy, masks)
    # Store paths relative to data_root so the dataset maps its abs paths -> rows.
    np.savez(idx_npz,
             image_paths=np.array([os.path.relpath(p, data_root) for p in image_paths]),
             mask_paths=np.array([os.path.relpath(p, data_root) for p in mask_paths]),
             size=np.int32(size))
    mb = (images.nbytes + masks.nbytes) / 1e6
    print(f"  [ok] {ds_key}: {len(image_paths)} imgs, {len(mask_paths)} masks  "
          f"({mb:.0f} MB, {time.perf_counter() - t:.1f}s)")
    return len(image_paths), len(mask_paths)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--size", type=int, default=128, help="Output square resolution (pixels)")
    p.add_argument("--split", choices=list(_SPLIT_DIRS) + ["both"], default="both",
                   help="Which split(s) to convert")
    p.add_argument("--datasets", nargs="*", default=None,
                   help="Restrict to these top-level dataset folders (default: all)")
    p.add_argument("--data_root", default=DATA_ROOT, help="BiomedParseData root")
    p.add_argument("--out", default=None,
                   help="Output root (default <data_root>/_npy)")
    p.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 8))
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    out_root = args.out or os.path.join(args.data_root, "_npy")
    splits = list(_SPLIT_DIRS) if args.split == "both" else [args.split]
    print(f"Data root : {args.data_root}")
    print(f"Output    : {out_root}  (size={args.size}, workers={args.workers})")

    t0 = time.perf_counter()
    tot_img = tot_msk = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for split in splits:
            img_dir_name, mask_dir_name = _SPLIT_DIRS[split]
            sources = _discover_sources(args.data_root, img_dir_name, mask_dir_name,
                                        args.datasets)
            print(f"\n[{split}] {len(sources)} dataset sources")
            for ds_key, img_dir, mask_dir in sources:
                ni, nm = _process_source(ds_key, img_dir, mask_dir, split, args.size,
                                         args.data_root, out_root, ex, args.overwrite)
                tot_img += ni
                tot_msk += nm

    print(f"\nDone: {tot_img} images + {tot_msk} masks in {time.perf_counter() - t0:.0f}s "
          f"-> {out_root}")


if __name__ == "__main__":
    main()
