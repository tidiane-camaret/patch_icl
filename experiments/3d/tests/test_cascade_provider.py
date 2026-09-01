"""Task 1: LoadRequest.jitter field + provider jitter resolution."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import random

from src.incontext_dataset_v2 import LoadRequest
from src.providers.totalseg import _resolve_jitter


def _req(jitter=None):
    return LoadRequest(rng=random.Random(0), crop_spacing_mm=1.5, jitter=jitter)


def test_loadrequest_jitter_defaults_none():
    assert _req().jitter is None


def test_loadrequest_jitter_set():
    assert _req(jitter=0).jitter == 0
    assert _req(jitter=7).jitter == 7


def test_resolve_jitter_prefers_request():
    assert _resolve_jitter(_req(jitter=0), default=12) == 0
    assert _resolve_jitter(_req(jitter=3), default=12) == 3


def test_resolve_jitter_falls_back_to_default():
    assert _resolve_jitter(_req(jitter=None), default=12) == 12


# --- Task 2: NativeCrop + TotalSegProvider.load_native_crop -------------------
import numpy as np
import torch

from src.providers.totalseg import NativeCrop


def _tiny_provider(tmp_path, spacing=1.5, T=8):
    """A TotalSegProvider over a 2-subject fake root with ram_cache on."""
    from src.providers.totalseg import TotalSegProvider
    from src.totalseg_dataset import _ALL_CLASSES_IDX

    for s in ("s0", "s1"):
        d = tmp_path / s
        d.mkdir()
        # smooth ramp so decimation error is bounded
        v = np.linspace(-500, 500, 20 * 20 * 20, dtype=np.float32).reshape(20, 20, 20)
        np.save(d / "ct_raw.npy", v.astype(np.float16))
        lbl = np.zeros((20, 20, 20), dtype=np.uint8)
        lbl[8:12, 8:12, 8:12] = _ALL_CLASSES_IDX["liver"]     # merged-label index 44
        np.save(d / "label.npy", lbl)
    (tmp_path / "meta.csv").write_text("image_id;split\ns0;train\ns1;train\n")
    # spacings.json so native_spacing resolves to 1.5
    (tmp_path / "spacings.json").write_text(
        '{"s0":{"spacing":[1.5,1.5,1.5],"shape":[20,20,20]},'
        ' "s1":{"spacing":[1.5,1.5,1.5],"shape":[20,20,20]}}')
    return TotalSegProvider(
        root=str(tmp_path), classes=["liver"], image_size=(T, T, T), split="train",
        crop_spacing_mm=spacing, crop_jitter=0, mask_downsample="soft",
        mask_occupancy_thr=0.5, ram_cache=True)


def test_load_native_crop_geom_matches_crop_and_place(tmp_path):
    import random

    from src.incontext_dataset_v2 import LoadRequest
    from src.totalseg_dataloader_incontext import organ_crop_arrays
    from src.totalseg_dataset import _ALL_CLASSES_IDX

    prov = _tiny_provider(tmp_path, spacing=3.0, T=8)
    center = (10, 10, 10)
    nc = prov.load_native_crop(
        "s0", "liver",
        LoadRequest(rng=random.Random(0), crop_spacing_mm=3.0, center=center, jitter=0))
    # reference geom from the pure helper on the same inputs
    lbl = np.load(tmp_path / "s0" / "label.npy")
    _, _, out_sizes, pad_lo, geom = organ_crop_arrays(
        lbl, lbl, center, [1.5, 1.5, 1.5], image_size=(8, 8, 8),
        crop_mm=3.0, jitter=0, rng=random.Random(0))
    assert torch.equal(nc.crop_geom, geom)
    assert nc.out_sizes == list(out_sizes) and nc.pad_lo == list(pad_lo)
    assert nc.decim == (2, 2, 2)                        # 3.0 / 1.5, crop_sizes // out_sizes
    # decimated crop still >= out_sizes on every axis (GPU never upsamples)
    assert all(s >= o for s, o in zip(nc.image.shape, nc.out_sizes))
    assert nc.class_idx == _ALL_CLASSES_IDX["liver"]


def test_load_native_crop_consumes_rng_once(tmp_path):
    import random

    from src.incontext_dataset_v2 import LoadRequest
    from src.totalseg_dataloader_incontext import organ_crop_arrays

    prov = _tiny_provider(tmp_path, spacing=1.5, T=8)
    center = (10, 10, 10)
    r1 = random.Random(0)
    prov.load_native_crop(
        "s0", "liver",
        LoadRequest(rng=r1, crop_spacing_mm=1.5, center=center, jitter=3))
    # mirror the single organ_crop_arrays call load_native_crop makes internally
    lbl = np.load(tmp_path / "s0" / "label.npy")
    r2 = random.Random(0)
    organ_crop_arrays(
        lbl, lbl, center, [1.5, 1.5, 1.5], image_size=(8, 8, 8),
        crop_mm=1.5, jitter=3, rng=r2)
    # r1 advanced by exactly that one call and no further
    assert r1.random() == r2.random()
