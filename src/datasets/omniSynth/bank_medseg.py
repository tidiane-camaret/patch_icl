"""MedSegObjectBank: real MedSegBench objects as an omniSynth object source.

Mirrors OmniglotBank's interface (task_ids / get / alphabet) so it drops into the
same render_scene + placement + target_mode machinery. Differences from glyphs:

  - a "class" is a (dataset, label_value) pair (analog of alphabet/character);
    alphabet(cid) returns "<dataset>/label_<lv>".
  - a rendition is one image's WHOLE binary mask for that label (all connected
    components kept — so multi-component objects stay intact), cropped to its bbox
    and resized into a cell tile, together with the intensity patch under it.
  - a rendition is a [2, tile, tile] float32 array: channel 0 = intensity (0..1,
    zeroed outside the mask so only the object's texture is pasted), channel 1 =
    binary mask {0,1}. Glyph renditions are 2D bitmaps (image == mask); render.py
    handles both via _split.

Splitting is by MedSegBench's own image splits: the "train" bank reads each
dataset's train images, the "val" bank its val images (there is no test set — val
doubles as test). train_datasets / val_datasets select which datasets feed each
split ([] = all). Built once per split and process-shared (forked workers inherit).
"""

import glob
import os

import numpy as np

from src.datasets.medsegbench import _load_images_and_labels
from .bank_common import crop_to_tile
from .config import OmniMedSegConfig

_BANK_CACHE: dict = {}


def get_or_build_medseg_bank(cfg: OmniMedSegConfig, cell_size: int,
                             cell_margin: float = 0.1, split: str = "train",
                             image_size: int = None) -> "MedSegObjectBank":
    key = (repr(cfg), int(cell_size), float(cell_margin), str(split), int(image_size or 0))
    if key not in _BANK_CACHE:
        _BANK_CACHE[key] = MedSegObjectBank(cfg, cell_size, cell_margin, split, image_size)
    return _BANK_CACHE[key]


class MedSegObjectBank:
    def __init__(self, cfg: OmniMedSegConfig, cell_size: int, cell_margin: float = 0.1,
                 split: str = "train", image_size: int = None):
        self.cell_size = int(cell_size)
        self.cell_margin = float(cell_margin)
        self.split = split
        # canvas-mode sizing references (source px -> canvas px). image_size defaults to
        # source_size, i.e. objects keep their original pixel size on the canvas.
        self.source_size = int(cfg.source_size)
        self.image_size = int(image_size) if image_size else int(cfg.source_size)
        self._sizing = dict(cell_size=self.cell_size, cell_margin=self.cell_margin,
                            source_size=self.source_size, image_size=self.image_size,
                            size_mode=cfg.size_mode, size_scale=float(cfg.size_scale))
        self._renditions: dict[int, list[np.ndarray]] = {}
        self._name: dict[int, str] = {}

        # This split's datasets ([] = all) and the MedSegBench image split to read
        # objects from (val==test, so both non-train splits read the val images).
        ds_list = cfg.train_datasets if split == "train" else cfg.val_datasets
        img_split = "train" if split == "train" else "val"
        if ds_list:
            names = [str(n) for n in ds_list]
        else:
            names = sorted(os.path.basename(p).replace(f"_{cfg.source_size}.npz", "")
                           for p in glob.glob(os.path.join(
                               cfg.data_root, f"*_{cfg.source_size}.npz")))

        # One class per (dataset, label_value) present in this split's images.
        self._pool: list[int] = []
        next_id = 0
        for name in names:
            path = os.path.join(cfg.data_root, f"{name}_{cfg.source_size}.npz")
            if not os.path.exists(path):
                continue
            try:
                images, labels = _load_images_and_labels(np.load(path), img_split)
            except KeyError:
                continue                                  # split absent for this dataset
            for lv in np.unique(labels):
                if lv == 0:
                    continue
                rends = self._extract(images, labels, int(lv),
                                      cfg.max_renditions_per_class, cfg.min_mask_px)
                if not rends:
                    continue
                self._name[next_id] = f"{name}/label_{int(lv)}"
                self._renditions[next_id] = rends
                self._pool.append(next_id)
                next_id += 1

    def _extract(self, images, labels, lv, cap, min_px):
        """One rendition per image containing label `lv`: whole mask bbox-cropped +
        intensity under it, tiled by bank_common. Capped at `cap` renditions."""
        rends = []
        for i in range(len(labels)):
            if len(rends) >= cap:
                break
            tile = crop_to_tile(images[i].astype(np.float32) / 255.0,
                                labels[i] == lv, min_px, **self._sizing)
            if tile is not None:
                rends.append(tile)
        return rends

    def task_ids(self, split: str = None) -> list[int]:
        # The bank is scoped to one split (train, or val used for val+test), so it
        # returns that split's classes regardless of the argument.
        return list(self._pool)

    def get(self, class_id: int) -> list[np.ndarray]:
        return self._renditions[class_id]

    def alphabet(self, class_id: int) -> str:
        return self._name[class_id]
