"""BackgroundBank: a pool of real full images (medseg / biomedparse) used as the
omniSynth canvas when scene.background="image". Independent of the object source —
e.g. omniglot glyphs or biomedparse objects can be composited over medseg image
backgrounds. Roots + source_size come from the medseg config block (shared install
paths); bg_source / bg_datasets / bg_max_images come from the scene config.

Images are pooled per split (train->train, val/test->val for medseg / test for
biomedparse) up to bg_max_images (spread across datasets), and one is sampled +
resized to the canvas on demand. Built once per (config, split, size) and
process-shared (forked workers inherit).
"""

import glob
import os

import numpy as np
from PIL import Image

from src.datasets.biomedparse import _discover_stores
from src.datasets.medsegbench import _load_images_and_labels

_BG_CACHE: dict = {}


def get_or_build_background_bank(medseg_cfg, scene, image_size, split):
    key = (repr(medseg_cfg), scene.bg_source, tuple(scene.bg_datasets),
           int(scene.bg_max_images), int(image_size), str(split))
    if key not in _BG_CACHE:
        _BG_CACHE[key] = BackgroundBank(medseg_cfg, scene, image_size, split)
    return _BG_CACHE[key]


class BackgroundBank:
    def __init__(self, medseg_cfg, scene, image_size, split):
        self.image_size = int(image_size)
        src = scene.bg_source
        size = int(medseg_cfg.source_size)
        datasets = [str(d) for d in scene.bg_datasets] or None
        max_images = int(scene.bg_max_images)
        img_split = "train" if split == "train" else ("val" if src == "medseg" else "test")

        providers = list(self._providers(medseg_cfg, src, size, img_split, datasets))
        if not providers:
            raise ValueError(f"no background images for bg_source={src!r} split={split!r}")
        # Spread the budget across datasets so one big dataset can't dominate the pool.
        per_ds = max(1, max_images // len(providers))
        self._items = []                          # (images_array, row) — array may be a memmap
        for _name, images in providers:
            take = min(len(images), per_ds, max_images - len(self._items))
            if take <= 0:
                break
            arr = images[:take]
            if not isinstance(images, np.memmap):    # medseg: copy the slice, drop the rest
                arr = np.ascontiguousarray(arr)
            self._items.extend((arr, r) for r in range(take))
        if not self._items:
            raise ValueError("empty background pool")

    @staticmethod
    def _providers(cfg, src, size, img_split, datasets):
        """Yield (name, images[N,H,W] uint8) for each dataset in this split."""
        if src == "medseg":
            paths = ([os.path.join(cfg.data_root, f"{n}_{size}.npz") for n in datasets]
                     if datasets else
                     sorted(glob.glob(os.path.join(cfg.data_root, f"*_{size}.npz"))))
            for p in paths:
                if not os.path.exists(p):
                    continue
                name = os.path.basename(p).replace(f"_{size}.npz", "")
                try:
                    images, _ = _load_images_and_labels(np.load(p), img_split)
                except KeyError:
                    continue
                yield name, images
        elif src == "biomedparse":
            for ds_key, idx_npz in _discover_stores(cfg.biomedparse_root, img_split, size, datasets):
                d = os.path.dirname(idx_npz)
                yield ds_key, np.load(os.path.join(d, f"images_{size}.npy"), mmap_mode="r")
        else:
            raise ValueError(f"unknown bg_source {src!r} (medseg | biomedparse)")

    def sample(self, rng) -> np.ndarray:
        """A random background image as [image_size, image_size] float32 in [0, 1]."""
        arr, r = self._items[rng.integers(len(self._items))]
        im = np.asarray(arr[r], dtype=np.float32) / 255.0
        if im.shape != (self.image_size, self.image_size):
            im = np.asarray(Image.fromarray((im * 255).astype(np.uint8))
                            .resize((self.image_size, self.image_size), Image.BILINEAR)
                            ).astype(np.float32) / 255.0
        return im
