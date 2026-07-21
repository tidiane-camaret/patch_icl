"""Precompute per-class organ tile caches for omniSynth 3D.

For a split, crop every organ once from the pre-resized label_{D}x{H}x{W}.npy +
ct_{D}x{H}x{W}.npy and write <out>/T{D}/{split}/class_{lv}.pkl (fp16 [2,T,T,T]
tiles) + index.pkl ({lv: class_name}). Built once; TotalSegObjectBank reads these
small files at train time (no full-volume reads in the hot path).

--root / --out default to the Hydra config's paths.totalseg and
paths.totalseg/omni_tiles (matching configs/experiment/3d/dataset/omnisynth3d.yaml's
synth3d.tiles_root). Any trailing Hydra overrides are forwarded to that compose,
so `cluster=meta` selects the right per-cluster path:

  python scripts/synth3d/build_totalseg_tiles.py --split train
  python scripts/synth3d/build_totalseg_tiles.py --split val cluster=meta
  python scripts/synth3d/build_totalseg_tiles.py --split train \
      --classes liver spleen aorta        # put Hydra overrides BEFORE --classes
"""

import argparse
import csv
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from data.totalseg_classes import ALL_CLASSES
from src.datasets.omniSynth.bank_common3d import crop_to_tile_3d


def subjects_for_split(root, split):
    root = Path(root)
    subs = sorted(p.name for p in root.iterdir() if p.is_dir())
    if split is None:
        return subs
    valid = set()
    with open(root / "meta.csv", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f, delimiter=";"):
            if row["split"].strip() == split:
                valid.add(row["image_id"].strip())
    return [s for s in subs if s in valid]


def build_tiles_for_split(root, out_root, size, split, *, max_renditions=200,
                          min_vox=8, size_scale=1.0, classes=None):
    root, out_root = Path(root), Path(out_root)
    D, H, W = size
    if not (D == H == W):
        raise ValueError(
            f"omniSynth 3D requires a cubic canvas (D==H==W); got size={tuple(size)}")
    suffix = f"{D}x{H}x{W}"
    src_size = max(size)                       # canvas-relative sizing reference
    allowed_lv = (None if not classes
                  else {ALL_CLASSES.index(c) + 1 for c in classes})

    subs = subjects_for_split(root, split)
    per_class: dict[int, dict] = {}            # lv -> {"name", "tiles"}
    for subj in subs:
        lab_p = root / subj / f"label_{suffix}.npy"
        ct_p = root / subj / f"ct_{suffix}.npy"
        if not lab_p.exists() or not ct_p.exists():
            continue
        lab = np.load(lab_p)
        ct = np.clip(np.load(ct_p).astype(np.float32), 0, None)
        ct = ct / (ct.max() + 1e-6)            # -> [0,1] for the intensity channel
        for lv in np.unique(lab):
            lv = int(lv)
            if lv == 0 or lv > len(ALL_CLASSES):
                continue
            if allowed_lv is not None and lv not in allowed_lv:
                continue
            entry = per_class.setdefault(
                lv, {"name": ALL_CLASSES[lv - 1], "tiles": []})
            if len(entry["tiles"]) >= max_renditions:
                continue
            tile = crop_to_tile_3d(ct, lab == lv, min_vox,
                                   source_size=src_size, image_size=src_size,
                                   size_scale=size_scale)
            if tile is not None:
                entry["tiles"].append(tile)

    split_dir = out_root / f"T{D}" / split
    split_dir.mkdir(parents=True, exist_ok=True)
    index = {}
    for lv, entry in per_class.items():
        if not entry["tiles"]:
            continue
        (split_dir / f"class_{lv}.pkl").write_bytes(pickle.dumps(entry))
        index[lv] = entry["name"]
    (split_dir / "index.pkl").write_bytes(pickle.dumps(index))
    print(f"[{split}] wrote {len(index)} classes -> {split_dir}", flush=True)
    return split_dir


def _resolve_root_out(root, out, hydra_overrides):
    """Fill root/out from the Hydra config when unset: root <- paths.totalseg,
    out <- <root>/omni_tiles. The config is composed only if a default is needed
    (so an explicit --root never requires a resolvable cluster config)."""
    if root is None:
        from hydra import compose, initialize_config_dir
        cfg_dir = Path(__file__).resolve().parents[2] / "configs"
        with initialize_config_dir(config_dir=str(cfg_dir), version_base="1.3"):
            cfg = compose(config_name="config", overrides=hydra_overrides)
        root = cfg.paths.totalseg
    if out is None:
        out = str(Path(root) / "omni_tiles")
    return root, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None,
                    help="TotalSeg root (default: paths.totalseg from the Hydra config)")
    ap.add_argument("--out", default=None,
                    help="output root (default: <root>/omni_tiles)")
    ap.add_argument("--size", type=int, nargs=3, default=[128, 128, 128])
    ap.add_argument("--split", default="train")
    ap.add_argument("--max-renditions", type=int, default=200)
    ap.add_argument("--min-vox", type=int, default=8)
    ap.add_argument("--size-scale", type=float, default=1.0)
    ap.add_argument("--classes", nargs="*", default=None)
    a, hydra_overrides = ap.parse_known_args()
    root, out = _resolve_root_out(a.root, a.out, hydra_overrides)
    build_tiles_for_split(root, out, tuple(a.size), a.split,
                          max_renditions=a.max_renditions, min_vox=a.min_vox,
                          size_scale=a.size_scale, classes=a.classes)


if __name__ == "__main__":
    main()
