"""In-context EVAL dataset over the extra TotalSegmentator `more_labels` classes.

Reuses TotalSegInContextDataset for context sampling, eval-seed determinism, the
single-label __getitem__ path, and the collate contract. Overrides only:

  * class identity  — classes are task-qualified keys "{task}/{name}" from
                      more_labels_classes.json (329 unique names collide across the
                      37 tasks, so the bare name is not unique); subject->classes
                      comes from more_labels_subject_classes.json, not a label.npy scan.
  * loading (_load) — fast path: CT from ct.nii.gz, reproducing convert_to_npy's
                      normalise + iso_resize so it aligns pixel-for-pixel with the
                      pre-resized more_labels/{task}_{size}.npy masks; binary mask =
                      task array == local_id.
                      crop path (use_crop=True): organ-centred native crop of fixed
                      physical extent (T*crop_spacing_mm) resampled to T³, so eval runs
                      at a chosen mm/voxel. Uses the native ct.npy + spacings.json written
                      by generate_crop_assets.py and the task mask more_labels/{task}.npy.

Eval-only: synth / augmentation / multi-label are asserted off (use_crop is supported).
"""
import hashlib
import json
import os
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np
import torch

from scripts.convert_to_npy import _iso_resize, _normalise_ct
from src.totalseg_dataloader_incontext import TotalSegInContextDataset


def _ml_centroids_for_subject(args: tuple) -> tuple[str, dict | None]:
    """Per-{task}/{name} organ centroids for one subject, in native task-mask voxel
    space (module-level for pickling). `task_to_keys` maps task -> {local_id: key}."""
    root, subj, task_to_keys = args
    mdir = Path(root) / subj / "more_labels"
    try:
        result: dict[str, tuple[int, int, int]] = {}
        grids = None
        for task, id_to_key in task_to_keys.items():
            f = mdir / f"{task}.npy"
            if not f.exists():
                continue
            arr = np.load(f, mmap_mode="r")
            if grids is None:  # every task file of a subject shares the native grid
                D, H, W = arr.shape
                grids = (np.arange(D, dtype=np.float32)[:, None, None],
                         np.arange(H, dtype=np.float32)[None, :, None],
                         np.arange(W, dtype=np.float32)[None, None, :])
            d_g, h_g, w_g = grids
            present = {int(x) for x in np.unique(arr)}
            for lid, key in id_to_key.items():
                if lid == 0 or lid not in present:
                    continue
                mask = (arr == lid)
                n = int(mask.sum())
                if n == 0:
                    continue
                result[key] = (int((d_g * mask).sum() / n),
                               int((h_g * mask).sum() / n),
                               int((w_g * mask).sum() / n))
        return subj, result
    except Exception:
        return subj, None


class TotalSegMoreLabelsDataset(TotalSegInContextDataset):
    def __init__(
        self,
        root: str | Path,
        classes: list[str],
        image_size: Optional[tuple[int, int, int]] = (64, 64, 64),
        split: Optional[str] = None,
        context_size: int = 3,
        max_subjects: Optional[int] = None,
        eval_seed: int = 0,
        use_crop: bool = False,
        crop_spacing_mm: float = 1.5,
        crop_jitter: Optional[int] = None,
    ):
        root = Path(root)
        # Read the global index BEFORE super().__init__: the overridden
        # _load_or_build_cache (called inside super) needs _gid_to_key.
        with open(root / "more_labels_classes.json") as f:
            index = json.load(f)
        self._resolve: dict[str, tuple[str, int]] = {}
        self._gid_to_key: dict[int, str] = {}
        for c in index["classes"]:
            key = f"{c['task']}/{c['name']}"
            self._resolve[key] = (c["task"], int(c["local_id"]))
            self._gid_to_key[int(c["global_id"])] = key
        with open(root / "more_labels_subject_classes.json") as f:
            self._subject_gids: dict[str, list[int]] = json.load(f)
        self._ct_cache: dict[str, torch.Tensor] = {}

        super().__init__(
            root=root,
            classes=classes,
            image_size=image_size,
            split=split,
            context_size=context_size,
            max_subjects=max_subjects,
            aug_cfg=None,
            synth_method=None,
            p_synth=0.0,
            class_balanced=False,
            use_crop=use_crop,
            crop_spacing_mm=crop_spacing_mm,
            crop_jitter=crop_jitter,
            num_labels_per_sample=1,
            eval_seed=eval_seed,
        )

    # --- overrides -----------------------------------------------------------
    def _get_subjects(self, split, meta_csv, max_subjects) -> list[str]:
        """No meta.csv in this tree; the 25 subjects are all 'test'. List dirs that
        actually carry a more_labels/ folder (ignores the two root JSON files)."""
        assert split in (None, "test"), \
            f"TotalSegMoreLabelsDataset is eval-only (split={split!r})"
        subs = sorted(p.name for p in self.root.iterdir()
                      if p.is_dir() and (p / "more_labels").is_dir())
        if max_subjects is not None:
            subs = subs[:max_subjects]
        return subs

    def _load_or_build_cache(self) -> dict[str, frozenset]:
        """subject -> frozenset("{task}/{name}") straight from the JSON — no label.npy
        scan, no .scan_cache pickle."""
        return {
            subj: frozenset(self._gid_to_key[g] for g in gids if g in self._gid_to_key)
            for subj, gids in self._subject_gids.items()
        }

    def _load_ct_resized(self, subj: str) -> torch.Tensor:
        """(1, D, H, W) f32 CT, resized to match the main tree's ct_{size}.npy. Cached
        per subject (25 subjects, ~26 MB/worker) so contexts don't re-decode the NIfTI."""
        t = self._ct_cache.get(subj)
        if t is not None:
            return t
        subj_dir = self.root / subj
        pre = (subj_dir / f"ct_{self._size_str}.npy") if self._size_str else None
        if pre is not None and pre.exists():
            t = torch.from_numpy(np.load(pre, mmap_mode="r").astype(np.float32)).unsqueeze(0)
        else:
            img = nib.as_closest_canonical(nib.load(str(subj_dir / "ct.nii.gz")))
            sp = tuple(float(x) for x in nib.affines.voxel_sizes(img.affine)[:3])
            vol = _normalise_ct(img.get_fdata(dtype=np.float32))
            if self.image_size is not None:
                vol = _iso_resize(vol, self.image_size, order=1, aa=True, spacing=sp)
            # This on-the-fly path returns float32; convert_to_npy stores ct_{size}.npy as
            # float16, so alignment checks use atol≈1e-2 to account for rounding.
            t = torch.from_numpy(np.ascontiguousarray(vol, dtype=np.float32)).unsqueeze(0)
        self._ct_cache[subj] = t
        return t

    def _load_or_build_bbox_cache(self) -> dict[str, dict[str, tuple[int, int, int]]]:
        """{subject: {"{task}/{name}": (d,h,w)}} organ centroids in native task-mask
        voxel space, for the use_crop path. Built once from the native task masks
        (parallel), pickled, keyed by the subject-list hash. Replaces the base cache,
        which reads label.npy / the 117-class index this tree doesn't have."""
        all_subjects = sorted(p.name for p in self.root.iterdir()
                              if p.is_dir() and (p / "more_labels").is_dir())
        hkey = hashlib.sha256(("ml_centroid|" + "|".join(all_subjects)).encode()).hexdigest()[:12]
        cache_path = self.root / f".ml_centroid_cache_{hkey}.pkl"
        if cache_path.exists():
            with open(cache_path, "rb") as f:
                cache = pickle.load(f)
            print(f"Loaded more_labels centroid cache ({len(cache)} subjects) "
                  f"from {cache_path.name}", flush=True)
            return cache

        task_to_keys: dict[str, dict[int, str]] = {}
        for key, (task, lid) in self._resolve.items():
            task_to_keys.setdefault(task, {})[lid] = key

        n_workers = min(16, os.cpu_count() or 1)
        print(f"Building more_labels centroid cache for {len(all_subjects)} subjects "
              f"({n_workers} workers)...", flush=True)
        cache: dict[str, dict[str, tuple[int, int, int]]] = {}
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futs = {ex.submit(_ml_centroids_for_subject, (str(self.root), s, task_to_keys)): s
                    for s in all_subjects}
            for fut in as_completed(futs):
                subj, result = fut.result()
                if result is not None:
                    cache[subj] = result
                else:
                    print(f"  Skipping centroids for {futs[fut]}", flush=True)
        with open(cache_path, "wb") as f:
            pickle.dump(cache, f)
        print(f"more_labels centroid cache saved ({len(cache)} subjects).", flush=True)
        return cache

    def _load_crop(self, subj: str, cls: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Organ-centred native crop of fixed physical extent (T*crop_spacing_mm),
        resampled to T³ -> isotropic crop_spacing_mm/voxel. Mirrors the base _load_crop
        but crops the task mask (== local_id) + the native ct.npy, with centroids from
        the more_labels centroid cache. Reuses the base slice/place helpers."""
        subj_dir = self.root / subj
        T = self.image_size[0]
        task, local_id = self._resolve[cls]

        label_mm = np.load(subj_dir / "more_labels" / f"{task}.npy", mmap_mode="r")
        D, H, W = label_mm.shape

        center = self._bbox_cache.get(subj, {}).get(cls)
        center = center if center is not None else (D // 2, H // 2, W // 2)

        sp = self._get_spacing(subj).tolist()   # native mm/voxel (3,)
        crop_ct, crop_lbl, out_sizes, pad_lo = self._organ_crop_arrays(
            subj_dir, label_mm, center, sp)

        image_t = self._place_image(crop_ct, out_sizes, pad_lo)
        label_t = self._place_label(
            self._resample_binary(crop_lbl == local_id, tuple(out_sizes)),
            out_sizes, pad_lo)
        return image_t, label_t

    def _load(self, subj: str, cls: str) -> tuple[torch.Tensor, torch.Tensor]:
        if self.use_crop:
            return self._load_crop(subj, cls)
        image_t = self._load_ct_resized(subj).clone()
        task, local_id = self._resolve[cls]
        mdir = self.root / subj / "more_labels"
        sized = (mdir / f"{task}_{self._size_str}.npy") if self._size_str else None
        if sized is not None and sized.exists():
            arr = np.load(sized, mmap_mode="r")[:]
        else:
            native = np.load(mdir / f"{task}.npy", mmap_mode="r")[:]
            if self.image_size is not None:
                # Compute voxel spacing from CT header to align the fallback mask resize
                # with the fast path (convert_to_npy.py uses true spacing for iso_resize).
                img = nib.as_closest_canonical(nib.load(str(self.root / subj / "ct.nii.gz")))
                sp = tuple(float(x) for x in nib.affines.voxel_sizes(img.affine)[:3])
                arr = _iso_resize(native, self.image_size, order=0, aa=False, spacing=sp)
            else:
                arr = native
        label_t = torch.from_numpy((arr == local_id).astype(np.int64))
        return image_t, label_t
