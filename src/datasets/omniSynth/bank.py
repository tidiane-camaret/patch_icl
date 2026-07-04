"""OmniglotBank: reads character PNGs directly from the Omniglot zips and caches
cell-sized binary bitmaps. Built once and process-shared so forked DataLoader
workers inherit it (the in-memory analog of controlSynth's GeometryBank cache).

Splits follow the Omniglot convention: background alphabets -> train; evaluation
alphabets -> val/test (partitioned by val_test_split, seeded on master_seed).
class_id is a global int across both zips. Renditions are inverted (ink->1),
resized to an inner box with a hardcoded 0.1 margin (cell_margin config field not yet
wired in V1) and centered into a cell_size tile.
"""

import io
import os
import re
import zipfile

import numpy as np
from PIL import Image

from .config import OmniDiversityConfig

_BANK_CACHE: dict = {}

# zip entries look like: images_background/Greek/character05/0123_07.png
_ENTRY = re.compile(r"^[^/]+/([^/]+)/(character\d+)/([^/]+\.png)$")


def get_or_build_bank(diversity: OmniDiversityConfig, cell_size: int,
                      cell_margin: float = 0.1) -> "OmniglotBank":
    key = (repr(diversity), int(cell_size), float(cell_margin))
    if key not in _BANK_CACHE:
        _BANK_CACHE[key] = OmniglotBank(diversity, cell_size, cell_margin)
    return _BANK_CACHE[key]


class OmniglotBank:
    def __init__(self, diversity: OmniDiversityConfig, cell_size: int,
                 cell_margin: float = 0.1):
        self.cell_size = int(cell_size)
        self.cell_margin = float(cell_margin)
        self._renditions: dict[int, list[np.ndarray]] = {}
        self._alphabet: dict[int, str] = {}
        self._pools: dict[str, list[int]] = {"train": [], "val": [], "test": []}

        root = diversity.omniglot_root
        train_zip = os.path.join(root, diversity.train_zip)
        eval_zip = os.path.join(root, diversity.eval_zip)
        next_id = 0

        # Train pool: every (alphabet, character) in the background zip.
        next_id = self._ingest(train_zip, target_pools=["train"], start_id=next_id)
        # Eval pool: ingest all eval classes, then split into val/test.
        eval_ids_start = next_id
        next_id = self._ingest(eval_zip, target_pools=None, start_id=next_id)
        eval_ids = list(range(eval_ids_start, next_id))
        rng = np.random.default_rng(diversity.master_seed)
        perm = rng.permutation(len(eval_ids))
        n_val = int(round(len(eval_ids) * diversity.val_test_split))
        val_set = {eval_ids[i] for i in perm[:n_val]}
        for cid in eval_ids:
            self._pools["val" if cid in val_set else "test"].append(cid)
        self._pools["val"].sort()
        self._pools["test"].sort()

    def _ingest(self, zip_path, target_pools, start_id):
        """Read a zip, group PNGs by (alphabet, character), assign class_ids.

        target_pools: list of pool names to append class_ids to (e.g. ["train"]),
        or None to leave pool assignment to the caller (eval val/test split)."""
        next_id = start_id
        groups: dict[tuple, list[bytes]] = {}
        with zipfile.ZipFile(zip_path) as zf:
            for name in zf.namelist():
                m = _ENTRY.match(name)
                if not m:
                    continue
                alphabet, character, _png = m.groups()
                groups.setdefault((alphabet, character), []).append(zf.read(name))
        for (alphabet, character) in sorted(groups):
            cid = next_id
            next_id += 1
            self._alphabet[cid] = alphabet
            self._renditions[cid] = [self._to_bitmap(b) for b in groups[(alphabet, character)]]
            if target_pools:
                for p in target_pools:
                    self._pools[p].append(cid)
        return next_id

    def _to_bitmap(self, png_bytes: bytes) -> np.ndarray:
        """PNG bytes -> [tile,tile] uint8 in {0,1}, inverted, centered in its tile.

        The glyph is resized to an `inner`-sized box where `inner = (1 - 2*margin)*cell`:
        margin>0 shrinks the glyph inside a cell-sized tile (the tile stays `cell` so a
        margin>=0 tile drops into one grid cell exactly); margin<0 makes `inner > cell`, so
        the tile is `inner` and the glyph deliberately overflows its cell (render pastes it
        onto the canvas with union blending, giving larger, mutually overlapping characters)."""
        img = Image.open(io.BytesIO(png_bytes)).convert("L")
        arr = np.asarray(img)
        fg = (arr < 128)                       # ink (black) -> foreground
        cell = self.cell_size
        inner = max(1, int(round(cell * (1.0 - 2.0 * self.cell_margin))))
        resized = np.asarray(
            Image.fromarray((fg * 255).astype(np.uint8)).resize((inner, inner), Image.BILINEAR)
        )
        bm_inner = (resized >= 128).astype(np.uint8)
        tile = max(cell, inner)                # >= cell keeps margin>=0 exactly cell-sized
        out = np.zeros((tile, tile), dtype=np.uint8)
        off = (tile - inner) // 2
        out[off:off + inner, off:off + inner] = bm_inner
        return out

    def task_ids(self, split: str) -> list[int]:
        if split not in self._pools:
            raise ValueError(f"unknown split {split!r} (train | val | test)")
        return list(self._pools[split])

    def get(self, class_id: int) -> list[np.ndarray]:
        return self._renditions[class_id]

    def alphabet(self, class_id: int) -> str:
        return self._alphabet[class_id]
