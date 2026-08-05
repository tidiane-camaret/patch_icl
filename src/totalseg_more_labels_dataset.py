"""In-context EVAL dataset over the extra TotalSegmentator `more_labels` classes.

Reuses TotalSegInContextDataset for context sampling, eval-seed determinism, the
single-label __getitem__ path, and the collate contract. Overrides only:

  * class identity  — classes are task-qualified keys "{task}/{name}" from
                      more_labels_classes.json (329 unique names collide across the
                      37 tasks, so the bare name is not unique); subject->classes
                      comes from more_labels_subject_classes.json, not a label.npy scan.
  * loading (_load) — CT from ct.nii.gz, reproducing convert_to_npy's normalise +
                      iso_resize so it aligns pixel-for-pixel with the pre-resized
                      more_labels/{task}_{size}.npy masks; binary mask = task array
                      == local_id.

Eval-only: use_crop / synth / augmentation / multi-label are asserted off.
"""
import json
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np
import torch

from scripts.convert_to_npy import _iso_resize, _normalise_ct
from src.totalseg_dataloader_incontext import TotalSegInContextDataset


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
            use_crop=False,
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
            t = torch.from_numpy(np.ascontiguousarray(vol, dtype=np.float32)).unsqueeze(0)
        self._ct_cache[subj] = t
        return t

    def _load(self, subj: str, cls: str) -> tuple[torch.Tensor, torch.Tensor]:
        image_t = self._load_ct_resized(subj).clone()
        task, local_id = self._resolve[cls]
        mdir = self.root / subj / "more_labels"
        sized = (mdir / f"{task}_{self._size_str}.npy") if self._size_str else None
        if sized is not None and sized.exists():
            arr = np.load(sized, mmap_mode="r")[:]
        else:
            native = np.load(mdir / f"{task}.npy", mmap_mode="r")[:]
            arr = (_iso_resize(native, self.image_size, order=0, aa=False)
                   if self.image_size is not None else native)
        label_t = torch.from_numpy((arr == local_id).astype(np.int64))
        return image_t, label_t
