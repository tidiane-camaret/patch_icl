"""BiomedParseObjectBank: real BiomedParse objects as an omniSynth object source.

Same interface + rendition format as MedSegObjectBank (task_ids / get / alphabet;
[2, tile, tile] intensity+mask tiles via bank_common), so it drops into the same
render machinery. Differences from medseg are only in loading:

  - a "class" is a (dataset, target) pair, e.g. "ACDC/myocardium" — the target is
    parsed from the mask filename (BiomedParse encodes it after the last '_', with
    '+' for spaces). alphabet(cid) returns "<dataset>/<target>".
  - objects come from the pre-resized store written for the 2D pipeline:
    <root>/<split>/<ds_key>/{images,masks}_{size}.npy + index_{size}.npz, where each
    mask row is one image's WHOLE binary mask for one target (multi-component kept).
    The mask's image is resolved via its filename stem -> the store's image rows.

Split maps to BiomedParse's own store splits: "train" -> train store, "val"/"test"
-> test store (no val set; val doubles as test). train_datasets / val_datasets filter
which datasets feed each split ([] = all). Built once per split, process-shared.
"""

import os
from collections import defaultdict

import numpy as np

from src.datasets.biomedparse import (_discover_stores, _index_from_npz,
                                       _parse_mask_stem)
from .bank_common import crop_to_tile
from .config import OmniMedSegConfig

_BANK_CACHE: dict = {}


def get_or_build_biomedparse_bank(cfg: OmniMedSegConfig, cell_size: int,
                                  cell_margin: float = 0.1, split: str = "train",
                                  image_size: int = None) -> "BiomedParseObjectBank":
    key = (repr(cfg), int(cell_size), float(cell_margin), str(split), int(image_size or 0))
    if key not in _BANK_CACHE:
        _BANK_CACHE[key] = BiomedParseObjectBank(cfg, cell_size, cell_margin, split, image_size)
    return _BANK_CACHE[key]


class BiomedParseObjectBank:
    def __init__(self, cfg: OmniMedSegConfig, cell_size: int, cell_margin: float = 0.1,
                 split: str = "train", image_size: int = None):
        self.split = split
        src = int(cfg.source_size)
        self._sizing = dict(cell_size=int(cell_size), cell_margin=float(cell_margin),
                            source_size=src, image_size=int(image_size) if image_size else src,
                            size_mode=cfg.size_mode, size_scale=float(cfg.size_scale))
        self._renditions: dict[int, list[np.ndarray]] = {}
        self._name: dict[int, str] = {}
        self._pool: list[int] = []

        # This split's datasets ([] = all) and the BiomedParse store split to read from
        # (val==test: both non-train splits use the test store).
        store_split = "train" if split == "train" else "test"
        ds_list = cfg.train_datasets if split == "train" else cfg.val_datasets
        stores = _discover_stores(cfg.biomedparse_root, store_split, src,
                                  [str(d) for d in ds_list] or None)

        next_id = 0
        for ds_key, idx_npz in stores:
            d = os.path.dirname(idx_npz)
            images = np.load(os.path.join(d, f"images_{src}.npy"), mmap_mode="r")
            masks = np.load(os.path.join(d, f"masks_{src}.npy"), mmap_mode="r")
            image_paths, mask_paths = _index_from_npz(idx_npz, cfg.biomedparse_root)
            stem_to_row = {os.path.splitext(os.path.basename(p))[0]: i
                           for i, p in enumerate(image_paths)}

            # Group mask rows by target -> one class per (dataset, target).
            by_target: dict[str, list[tuple[int, int]]] = defaultdict(list)
            for mrow, mp in enumerate(mask_paths):
                stem = os.path.splitext(os.path.basename(mp))[0]
                image_stem, _, _, target = _parse_mask_stem(stem)
                irow = stem_to_row.get(image_stem)
                if irow is not None:
                    by_target[target].append((mrow, irow))

            for target, items in by_target.items():
                rends = self._extract(images, masks, items,
                                      cfg.max_renditions_per_class, cfg.min_mask_px)
                if not rends:
                    continue
                self._name[next_id] = f"{ds_key}/{target}"
                self._renditions[next_id] = rends
                self._pool.append(next_id)
                next_id += 1

    def _extract(self, images, masks, items, cap, min_px):
        """One rendition per (mask_row, image_row): whole mask bbox-cropped + intensity
        under it, tiled by bank_common. Capped at `cap` kept renditions."""
        rends = []
        for mrow, irow in items:
            if len(rends) >= cap:
                break
            tile = crop_to_tile(images[irow].astype(np.float32) / 255.0,
                                masks[mrow] > 0, min_px, **self._sizing)
            if tile is not None:
                rends.append(tile)
        return rends

    def task_ids(self, split: str = None) -> list[int]:
        return list(self._pool)

    def get(self, class_id: int) -> list[np.ndarray]:
        return self._renditions[class_id]

    def alphabet(self, class_id: int) -> str:
        return self._name[class_id]
