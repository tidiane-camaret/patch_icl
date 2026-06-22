"""
BiomedParseData in-context dataset (prototype).

Wraps the official BiomedParseData on-disk layout for in-context 2D segmentation,
mirroring `src.datasets.medsegbench.MedSegBenchDataset` so it slots into the 2D
eval pipeline (`common.TaggedDataset` / `collate` expect `self.samples` to be a
list of `(dataset, sample_idx, label_value)` 3-tuples).

On-disk layout (as extracted from `microsoft/BiomedParseData`): each dataset is
double-nested `<root>/<DATASET>/<DATASET>/`, and a few datasets insert one extra
sublevel (modality / task / class) before the split dirs:

    <root>/<DATASET>/<DATASET>/[<SUBLEVEL>/]
        train/        [IMAGE-NAME]_[MODALITY]_[SITE].png            (1024x1024)
        train_mask/   [IMAGE-NAME]_[MODALITY]_[SITE]_[TARGET].png   (binary, !=0 = fg)
        test/         ...
        test_mask/    ...

    e.g.  ACDC/ACDC/test_mask                         (regular)
          amos22/amos22/CT/test_mask                  (modality sublevel)
          MSD/MSD/Task01_BrainTumour/test_mask        (task sublevel)
          Radiography/Radiography/COVID/test_mask     (class sublevel)

We discover split dirs at any depth under each top-level dataset folder; the
collapsed sub-path (`amos22/CT`, `MSD/Task01_BrainTumour`, `ACDC`) is the dataset
key, so context is never mixed across sublevels.

One mask file = one (image, modality, site, target) triple = one eval unit. A
single image may have several masks (one per target). `[TARGET]` encodes spaces
as '+' (and never contains '_'); we restore spaces on read. A literal
`absent.png` mask is a "target not present" sentinel and is skipped.

Pixels are loaded lazily from disk (PNGs are large), with a small LRU image cache.

Diversity tags (`dataset_of`, `modality_of`, `target_of`, `cell_of`) expose the
(dataset x modality x target) grid so a macro-averaging eval can weight cells
equally — the whole point of moving off MedSegBench's flat per-sample
micro-average.
"""

import glob
import os
import random
from collections import defaultdict
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.datasets.cow_index import SampleIndex, build_candidate_index, sample_context

DATA_ROOT = "/nfs/data/nii/data1/Analysis/camaret___in_context_segmentation/ANALYSIS_20251122/data/biomedparse"

_SPLIT_DIRS = {"train": ("train", "train_mask"), "test": ("test", "test_mask")}

# Mask filename that marks "this target is not present in the image" — not a real mask.
_ABSENT_MASK = "absent.png"

# Max directory depth (below data_root) at which a split dir may live. Covers
# regular (<DS>/<DS>) and the single-sublevel irregulars (<DS>/<DS>/<SUB>).
_MAX_SPLIT_DEPTH = 4


def _collapse_key(rel_path: str) -> str:
    """
    Turn a split dir's parent rel-path into a friendly dataset key.

    The on-disk tree repeats the dataset name (`ACDC/ACDC`); collapse that
    redundant first segment but keep any real sublevel:
        'ACDC/ACDC'                  -> 'ACDC'
        'amos22/amos22/CT'           -> 'amos22/CT'
        'MSD/MSD/Task01_BrainTumour' -> 'MSD/Task01_BrainTumour'
    """
    parts = rel_path.split(os.sep)
    if len(parts) >= 2 and parts[0] == parts[1]:
        parts = parts[:1] + parts[2:]
    return "/".join(parts)


def _discover_sources(
    data_root: str, img_dir_name: str, mask_dir_name: str,
    datasets: Optional[List[str]],
) -> List[Tuple[str, str, str]]:
    """
    Find every (ds_key, img_dir, mask_dir) split-pair under data_root, at any
    nesting depth. `datasets`, if given, filters by top-level folder name.
    """
    seen = set()
    sources: List[Tuple[str, str, str]] = []
    for depth in range(1, _MAX_SPLIT_DEPTH + 1):
        pattern = os.path.join(data_root, *(["*"] * depth), mask_dir_name)
        for mask_dir in sorted(glob.glob(pattern)):
            parent = os.path.dirname(mask_dir)
            img_dir = os.path.join(parent, img_dir_name)
            if not os.path.isdir(img_dir) or parent in seen:
                continue
            rel = os.path.relpath(parent, data_root)
            top = rel.split(os.sep)[0]
            if datasets is not None and top not in datasets:
                continue
            seen.add(parent)
            sources.append((_collapse_key(rel), img_dir, mask_dir))
    return sorted(sources)


def _parse_mask_stem(mask_stem: str) -> Tuple[str, str, str, str]:
    """
    Split a mask filename stem into (image_stem, modality, site, target).

    Mask stem = `[IMAGE-NAME]_[MODALITY]_[SITE]_[TARGET]`; the image it belongs to
    is `[IMAGE-NAME]_[MODALITY]_[SITE]`. TARGET has '+' for spaces and no '_', so
    it is exactly the final underscore-delimited token. MODALITY and SITE are the
    two tokens before it (best-effort; absent → "unknown").
    """
    target_raw = mask_stem.rsplit("_", 1)[-1]
    image_stem = mask_stem[: -(len(target_raw) + 1)]  # drop "_<target>"
    target = target_raw.replace("+", " ")

    parts = image_stem.split("_")
    site = parts[-1] if len(parts) >= 2 else "unknown"
    modality = parts[-2] if len(parts) >= 3 else "unknown"
    return image_stem, modality, site, target


def _read_image_tensor(path: str, size: int) -> torch.Tensor:
    """
    Decode a PNG to a grayscale [1, size, size] float tensor in [0, 1].

    Grayscale + resize happen inside PIL (C), which is ~1.5x faster than decoding
    the full 1024x1024 RGBA array and doing the channel-mean + interpolate in torch
    (decode itself is the irreducible cost). `convert("L")` is the ITU-R 601 luma.
    """
    im = Image.open(path).convert("L")
    if im.size != (size, size):
        im = im.resize((size, size), Image.BILINEAR)
    arr = np.asarray(im, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0).clamp_(0, 1)


def _read_mask_tensor(path: str, size: int) -> torch.Tensor:
    """Decode a binary mask PNG (any !=0 = fg) to [1, size, size] float {0, 1}."""
    im = Image.open(path).convert("L")
    if im.size != (size, size):
        im = im.resize((size, size), Image.NEAREST)
    arr = (np.asarray(im) > 0).astype(np.float32)
    return torch.from_numpy(arr).unsqueeze(0)


class BiomedParseDataset(Dataset):
    """
    In-context BiomedParseData dataset.

    Args:
        split: 'train' or 'test'
        context_size: number of context (image, mask) pairs per sample
        image_size: spatial resolution to resize to (default 128, matches 2D eval)
        data_root: path to the `biomedparse_datasets` directory
        datasets: list of dataset folder names to load; None loads all under root
        cache_size: per-process LRU capacity for resized tensors. None (default)
            picks a size-aware budget (~64 MB/worker) so 512px runs don't OOM.
    """

    def __init__(
        self,
        split: str = "test",
        context_size: int = 3,
        image_size: int = 128,
        data_root: str = DATA_ROOT,
        datasets: Optional[List[str]] = None,
        cache_size: Optional[int] = None,
        deterministic: Optional[bool] = None,
    ):
        if split not in _SPLIT_DIRS:
            raise ValueError(f"split must be one of {list(_SPLIT_DIRS)}, got {split!r}")
        if cache_size is None:
            # ~64 MB/worker: scales the entry count down as resolution grows.
            cache_size = max(64, (64 << 20) // (image_size * image_size * 4))
        self.split = split
        self.context_size = context_size
        self.image_size = image_size
        # Non-train splits get reproducible context (seeded from idx) so eval Dice
        # isn't perturbed by context-sampling variance across runs/epochs.
        self.deterministic = (split != "train") if deterministic is None else deterministic
        self.data_root = data_root
        img_dir_name, mask_dir_name = _SPLIT_DIRS[split]
        sources = _discover_sources(data_root, img_dir_name, mask_dir_name, datasets)

        # COW-safe sample index, built from parallel int lists below (see cow_index).
        self._ds_names: List[str] = []
        _ds_ids: List[int] = []
        _img_idxs: List[int] = []
        _tgt_ints: List[int] = []
        # Side tables for lazy loading + context lookup.
        self.image_paths: Dict[str, List[str]] = {}                      # ds -> [path]
        self.mask_path: Dict[Tuple[str, int, int], str] = {}            # (ds, img_idx, tgt) -> path
        self.target_to_int: Dict[str, Dict[str, int]] = {}             # ds -> {target_str: int}
        self.int_to_target: Dict[str, Dict[int, str]] = {}            # ds -> {int: target_str}
        self.meta: Dict[Tuple[str, int], Tuple[str, str]] = {}        # (ds, img_idx) -> (modality, site)

        # Cache the small *resized* tensors (not the 1024x1024 source): context
        # images are reused heavily across samples, so a hit skips the PNG decode
        # entirely. Keyed by path; image_size is fixed for the instance.
        sz = image_size
        self._img_cache = lru_cache(maxsize=cache_size)(
            lambda path: _read_image_tensor(path, sz))
        self._mask_cache = lru_cache(maxsize=cache_size)(
            lambda path: _read_mask_tensor(path, sz))

        print(f"Loading BiomedParseData (size={image_size}, split={split})...")
        for ds, img_dir, mask_dir in sources:
            ds_id = len(self._ds_names)
            self._ds_names.append(ds)
            image_idx_of: Dict[str, int] = {}   # image_stem -> image_idx (per dataset)
            paths: List[str] = []
            t2i = self.target_to_int.setdefault(ds, {})
            i2t = self.int_to_target.setdefault(ds, {})
            n_before = len(_ds_ids)

            for mask_path in sorted(glob.glob(os.path.join(mask_dir, "*.png"))):
                if os.path.basename(mask_path) == _ABSENT_MASK:
                    continue  # "target not present" sentinel
                stem = os.path.splitext(os.path.basename(mask_path))[0]
                image_stem, modality, site, target = _parse_mask_stem(stem)

                img_path = os.path.join(img_dir, image_stem + ".png")
                if not os.path.exists(img_path):
                    continue  # mask with no matching image — skip

                if image_stem not in image_idx_of:
                    image_idx_of[image_stem] = len(paths)
                    paths.append(img_path)
                img_idx = image_idx_of[image_stem]
                self.meta[(ds, img_idx)] = (modality, site)

                if target not in t2i:
                    tgt_int = len(t2i) + 1  # 1-based (0 reserved for background)
                    t2i[target] = tgt_int
                    i2t[tgt_int] = target
                tgt_int = t2i[target]

                self.mask_path[(ds, img_idx, tgt_int)] = mask_path
                _ds_ids.append(ds_id)
                _img_idxs.append(img_idx)
                _tgt_ints.append(tgt_int)

            self.image_paths[ds] = paths
            print(f"  [ok] {ds}: {len(paths)} images, {len(t2i)} targets, "
                  f"{len(_ds_ids) - n_before} samples")

        # COW-safe index + per-(ds, target) candidate arrays (replaces group_index).
        self.samples = SampleIndex(_ds_ids, _img_idxs, _tgt_ints, self._ds_names)
        self._cand = build_candidate_index(_ds_ids, _img_idxs, _tgt_ints)
        print(f"Total: {len(self.samples)} samples from {len(self.image_paths)} datasets")

    # ── diversity tags (for macro-averaging eval) ──────────────────────────────

    def dataset_of(self, idx: int) -> str:
        si = self.samples
        return si.ds_names[int(si.ds_ids[idx])]

    def modality_of(self, idx: int) -> str:
        si = self.samples
        ds = si.ds_names[int(si.ds_ids[idx])]
        return self.meta[(ds, int(si.img_idxs[idx]))][0]

    def target_of(self, idx: int) -> str:
        si = self.samples
        ds = si.ds_names[int(si.ds_ids[idx])]
        return self.int_to_target[ds][int(si.label_values[idx])]

    def cell_of(self, idx: int) -> Tuple[str, str, str]:
        """
        (dataset, modality, target) grid cell — the unit a balanced eval should
        macro-average over. Keying on dataset (not just modality) keeps each
        source separate, so e.g. amos22 CT-liver and another CT dataset's liver
        are distinct cells rather than pooled.
        """
        return self.dataset_of(idx), self.modality_of(idx), self.target_of(idx)

    # ── Dataset protocol ───────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def _img(self, ds: str, img_idx: int) -> torch.Tensor:
        return self._img_cache(self.image_paths[ds][img_idx])

    def _mask(self, ds: str, img_idx: int, tgt: int) -> torch.Tensor:
        return self._mask_cache(self.mask_path[(ds, img_idx, tgt)])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        si = self.samples
        did = int(si.ds_ids[idx])
        ds = si.ds_names[did]
        img_idx = int(si.img_idxs[idx])
        tgt = int(si.label_values[idx])

        # Context: other images sharing the same (dataset, target).
        rng = random.Random(idx) if self.deterministic else None
        cand = self._cand.get((did, tgt))
        ctx_idxs = sample_context(cand, img_idx, self.context_size, rng) if cand is not None \
            else np.empty(0, dtype=np.int32)

        if len(ctx_idxs):
            context_in = torch.stack([self._img(ds, int(i)) for i in ctx_idxs])
            context_out = torch.stack([self._mask(ds, int(i), tgt) for i in ctx_idxs])
        else:
            z = torch.zeros(0, 1, self.image_size, self.image_size)
            context_in = context_out = z

        return {
            "image": self._img(ds, img_idx),
            "label": self._mask(ds, img_idx, tgt),
            "context_in": context_in,    # [K, 1, H, W]
            "context_out": context_out,  # [K, 1, H, W]
        }


# ── self-test on a synthetic fixture (runs without the real download) ──────────

def _make_synthetic_fixture(root: str, seed: int = 0) -> None:
    """
    Write a tiny tree mirroring the real on-disk shape: double-nested
    `<root>/<DS>/<DS>/...`, including one dataset with a modality sublevel and an
    `absent.png` sentinel, plus a stray mask with no matching image.
    """
    rng = np.random.default_rng(seed)
    # (dataset, sub, modality, site, [targets], n_images)
    specs = [
        ("CXR_demo", "", "X-Ray", "chest", ["lung", "chest+tube"], 5),
        ("Derm_demo", "", "Dermoscopy", "skin", ["lesion"], 4),
        ("Abd_demo", "CT", "CT", "abdomen", ["liver", "left+kidney"], 3),
    ]
    for ds, sub, modality, site, targets, n_imgs in specs:
        base = os.path.join(root, ds, ds, sub) if sub else os.path.join(root, ds, ds)
        img_dir = os.path.join(base, "test")
        mask_dir = os.path.join(base, "test_mask")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(mask_dir, exist_ok=True)
        for n in range(n_imgs):
            stem = f"img{n:03d}_{modality}_{site}"
            Image.fromarray(rng.integers(0, 256, (64, 64), dtype=np.uint8)).save(
                os.path.join(img_dir, stem + ".png"))
            for tgt in targets:
                m = (rng.random((64, 64)) > 0.7).astype(np.uint8) * 255
                Image.fromarray(m).save(os.path.join(mask_dir, f"{stem}_{tgt}.png"))
        # sentinel + an orphan mask (no matching image) — both must be ignored
        Image.fromarray(np.zeros((64, 64), dtype=np.uint8)).save(
            os.path.join(mask_dir, "absent.png"))
        m = (rng.random((64, 64)) > 0.7).astype(np.uint8) * 255
        Image.fromarray(m).save(os.path.join(mask_dir, f"img999_{modality}_{site}_{targets[0]}.png"))


def _self_test() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        root = os.path.join(tmp, "biomedparse_datasets")
        _make_synthetic_fixture(root)

        K, S = 3, 96
        ds = BiomedParseDataset(split="test", context_size=K, image_size=S,
                                data_root=root, cache_size=64)

        # 5x2 (CXR) + 4x1 (Derm) + 3x2 (Abd) = 20; absent.png + orphan masks excluded
        assert len(ds) == 20, f"expected 20 samples, got {len(ds)}"
        assert set(ds.image_paths) == {"CXR_demo", "Derm_demo", "Abd_demo/CT"}, \
            f"unexpected dataset keys: {sorted(ds.image_paths)}"  # double-nest + sublevel collapse
        assert all(os.path.basename(p) != "absent.png" for p in ds.mask_path.values())

        item = ds[0]
        assert item["image"].shape == (1, S, S), item["image"].shape
        assert item["label"].shape == (1, S, S)
        assert item["context_in"].shape == (K, 1, S, S), item["context_in"].shape
        assert item["context_out"].shape == (K, 1, S, S)
        assert item["image"].dtype == torch.float32
        assert 0.0 <= float(item["image"].min()) and float(item["image"].max()) <= 1.0
        assert set(torch.unique(item["label"]).tolist()) <= {0.0, 1.0}

        # Context shares the same (dataset, target) cell as the target.
        cell = ds.cell_of(0)
        assert isinstance(cell, tuple) and len(cell) == 3
        si = ds.samples
        did0, img0, tgt0 = int(si.ds_ids[0]), int(si.img_idxs[0]), int(si.label_values[0])
        cand = ds._cand[(did0, tgt0)]                      # COW-safe candidate array
        assert (cand == img0).any()                        # target's own image is in its cell

        # Grid coverage: distinct (dataset, modality, target) cells present.
        cells = {ds.cell_of(i) for i in range(len(ds))}
        assert ("CXR_demo", "X-Ray", "lung") in cells
        assert ("CXR_demo", "X-Ray", "chest tube") in cells  # '+' restored to space
        assert ("Derm_demo", "Dermoscopy", "lesion") in cells
        assert ("Abd_demo/CT", "CT", "left kidney") in cells  # sublevel dataset, '+' restored

        print(f"\nself-test OK — {len(ds)} samples, {len(cells)} (dataset,modality,target) cells: "
              f"{sorted(cells)}")


if __name__ == "__main__":
    _self_test()
