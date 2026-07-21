"""TotalSegObjectBank: real TotalSegmentator organs as an omniSynth object source.

Mirrors MedSegObjectBank's interface (task_ids / get / alphabet) so it drops into
the render_scene + target_mode machinery. It reads the precomputed per-class tile
caches written by scripts/synth3d/build_totalseg_tiles.py — no full-volume reads at
train time. A class is a TotalSeg organ label value; a rendition is one subject's
organ as a [2, T, T, T] fp16 tile. Each class file is loaded once and LRU-cached."""

import pickle
from collections import OrderedDict
from pathlib import Path

_BANK_CACHE: dict = {}


def get_or_build_totalseg_bank(tiles_root, size, split="train", classes=(),
                               lru_classes=64):
    key = (str(tiles_root), int(size[0]), str(split), tuple(classes), int(lru_classes))
    if key not in _BANK_CACHE:
        _BANK_CACHE[key] = TotalSegObjectBank(tiles_root, size, split, classes,
                                              lru_classes)
    return _BANK_CACHE[key]


class TotalSegObjectBank:
    def __init__(self, tiles_root, size, split="train", classes=(), lru_classes=64):
        self.split_dir = Path(tiles_root) / f"T{int(size[0])}" / split
        index_path = self.split_dir / "index.pkl"
        if not index_path.exists():
            raise FileNotFoundError(f"no tile cache at {index_path} — run "
                                    "scripts/synth3d/build_totalseg_tiles.py first")
        self._index: dict[int, str] = pickle.loads(index_path.read_bytes())
        if classes:
            wanted = set(classes)
            self._index = {lv: n for lv, n in self._index.items() if n in wanted}
        if not self._index:
            raise ValueError(f"empty class pool for split {split!r} at {self.split_dir}")
        self._pool = sorted(self._index)
        self._lru_classes = int(lru_classes)
        self._loaded: "OrderedDict[int, list]" = OrderedDict()

    def task_ids(self, split=None) -> list[int]:
        return list(self._pool)

    def get(self, class_id: int) -> list:
        cid = int(class_id)
        if cid not in self._index:
            raise KeyError(f"class_id {cid} not in bank (known: {self._pool})")
        if cid in self._loaded:
            self._loaded.move_to_end(cid)
            return self._loaded[cid]
        data = pickle.loads((self.split_dir / f"class_{cid}.pkl").read_bytes())
        tiles = data["tiles"]
        self._loaded[cid] = tiles
        if len(self._loaded) > self._lru_classes:
            self._loaded.popitem(last=False)
        return tiles

    def alphabet(self, class_id: int) -> str:
        cid = int(class_id)
        if cid not in self._index:
            raise KeyError(f"class_id {cid} not in bank (known: {self._pool})")
        return self._index[cid]
