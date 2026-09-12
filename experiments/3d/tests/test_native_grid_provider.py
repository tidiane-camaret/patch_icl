"""NativeGridProvider (FLARE22/NasalSeg base): .load_native_crop for the GPU-realize cascade
path (data.gpu_realize_crop), and the req.jitter fix shared by .load and .load_native_crop.

Mirrors test_cascade_provider.py's TotalSegProvider coverage -- NativeGridProvider.load_native_crop
is a mechanical port of TotalSegProvider.load_native_crop onto self._centroids/self._meta.
"""
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from src.incontext_dataset_v2 import LoadRequest
from src.providers.native_grid import NativeGridProvider


class _FakeGridProvider(NativeGridProvider):
    """Minimal NativeGridProvider subclass (same shape as NasalSegProvider/Flare22Provider)."""
    SOURCE = "fake_native"
    ALL_CLASSES = ["blob"]
    CLASS_IDX = {"blob": 1}


def _tiny_provider(tmp_path, spacing_mm=3.0, native_spacing=(1.5, 1.5, 1.5), T=8, crop_jitter=0):
    for s in ("s0", "s1"):
        d = tmp_path / s
        d.mkdir()
        v = np.linspace(-500, 500, 20 * 20 * 20, dtype=np.float32).reshape(20, 20, 20)
        np.save(d / "ct_raw.npy", v.astype(np.int16))
        lbl = np.zeros((20, 20, 20), dtype=np.uint8)
        lbl[8:12, 8:12, 8:12] = 1
        np.save(d / "label.npy", lbl)
    meta = {s: {"spacing": list(native_spacing), "shape": [20, 20, 20],
                "affine": np.eye(4).tolist()} for s in ("s0", "s1")}
    (tmp_path / "spacings.json").write_text(json.dumps(meta))
    return _FakeGridProvider(
        root=str(tmp_path), classes=["blob"], image_size=(T, T, T),
        crop_spacing_mm=spacing_mm, crop_jitter=crop_jitter,
        mask_downsample="soft", mask_occupancy_thr=0.5)


def test_load_native_crop_geom_matches_organ_crop_arrays(tmp_path):
    from src.totalseg_dataloader_incontext import organ_crop_arrays

    prov = _tiny_provider(tmp_path, spacing_mm=3.0)
    center = (10, 10, 10)
    nc = prov.load_native_crop(
        "s0", "blob",
        LoadRequest(rng=random.Random(0), crop_spacing_mm=3.0, center=center, jitter=0))
    lbl = np.load(tmp_path / "s0" / "label.npy")
    _, _, out_sizes, pad_lo, geom = organ_crop_arrays(
        lbl, lbl, center, [1.5, 1.5, 1.5], image_size=(8, 8, 8),
        crop_mm=3.0, jitter=0, rng=random.Random(0))
    assert torch.equal(nc.crop_geom, geom)
    assert nc.out_sizes == list(out_sizes) and nc.pad_lo == list(pad_lo)
    assert nc.class_idx == 1
    assert all(s >= o for s, o in zip(nc.image.shape, nc.out_sizes))


def test_load_native_crop_honors_request_jitter(tmp_path):
    """req.jitter overrides the provider's construction-time crop_jitter. Before the fix,
    NativeGridProvider.load_native_crop would ignore req.jitter=5 and act as if it were the
    provider's own crop_jitter=0 -- so its result would NOT match a jitter=5 reference call."""
    from src.totalseg_dataloader_incontext import organ_crop_arrays

    prov = _tiny_provider(tmp_path, spacing_mm=1.5, crop_jitter=0)
    center = (10, 10, 10)
    nc = prov.load_native_crop(
        "s0", "blob",
        LoadRequest(rng=random.Random(0), crop_spacing_mm=1.5, center=center, jitter=5))
    lbl = np.load(tmp_path / "s0" / "label.npy")
    _, _, _, _, geom_j5 = organ_crop_arrays(
        lbl, lbl, center, [1.5, 1.5, 1.5], image_size=(8, 8, 8),
        crop_mm=1.5, jitter=5, rng=random.Random(0))
    assert torch.equal(nc.crop_geom, geom_j5)


def test_load_honors_request_jitter_too(tmp_path):
    """Same fix, exercised through the plain .load() path (the CPU cascade re-crop path used
    when data.gpu_realize_crop=false)."""
    from src.totalseg_dataloader_incontext import organ_crop_arrays

    prov = _tiny_provider(tmp_path, spacing_mm=1.5, crop_jitter=0)
    center = (10, 10, 10)
    r = prov.load(
        "s0", "blob",
        LoadRequest(rng=random.Random(0), crop_spacing_mm=1.5, center=center, jitter=5))
    lbl = np.load(tmp_path / "s0" / "label.npy")
    _, _, _, _, geom_j5 = organ_crop_arrays(
        lbl, lbl, center, [1.5, 1.5, 1.5], image_size=(8, 8, 8),
        crop_mm=1.5, jitter=5, rng=random.Random(0))
    assert torch.equal(r.crop_geom, geom_j5)


def test_load_and_load_native_crop_agree_on_geometry(tmp_path):
    """The CPU (.load) and GPU-realize (.load_native_crop) paths must place the SAME crop --
    only the resample/normalize timing differs."""
    prov = _tiny_provider(tmp_path, spacing_mm=3.0)
    center = (10, 10, 10)
    r = prov.load(
        "s0", "blob", LoadRequest(rng=random.Random(0), crop_spacing_mm=3.0, center=center, jitter=0))
    nc = prov.load_native_crop(
        "s0", "blob", LoadRequest(rng=random.Random(0), crop_spacing_mm=3.0, center=center, jitter=0))
    assert torch.equal(r.crop_geom, nc.crop_geom)


def test_load_native_crop_random_fg_center_lands_inside_the_blob(tmp_path):
    """center=None + center_mode='random_fg' (via the shared _resolve_center helper) must land
    inside the planted [8:12) blob, not just at its fixed centroid (10,10,10 by construction)."""
    prov = _tiny_provider(tmp_path, spacing_mm=1.5, crop_jitter=0)
    req = LoadRequest(rng=random.Random(1), crop_spacing_mm=1.5, jitter=0, center_mode="random_fg")
    nc = prov.load_native_crop("s0", "blob", req)
    starts, crop_sizes = nc.crop_geom[0].tolist(), nc.crop_geom[1].tolist()
    center = tuple(s + cs // 2 for s, cs in zip(starts, crop_sizes))
    assert all(8 <= c < 12 for c in center), \
        f"random_fg center {center} is outside the planted blob [8,12)"
