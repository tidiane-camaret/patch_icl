"""NativeGridProvider (FLARE22/NasalSeg/ISLES22 base): .load_native_crop for the GPU-realize
cascade path (data.gpu_realize_crop), the req.jitter fix shared by .load and
.load_native_crop, and the MODALITY="mri" per-subject normalization branch (ISLES22).

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
import pytest
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


class _FakeMriProvider(NativeGridProvider):
    """MODALITY='mri' variant (mirrors Isles22Provider) -- exercises the per-subject
    mri_stats/normalize_mri path instead of the fixed global CT frame."""
    SOURCE = "fake_native_mri"
    ALL_CLASSES = ["blob"]
    CLASS_IDX = {"blob": 1}
    MODALITY = "mri"


def _tiny_mri_provider(tmp_path, spacing_mm=3.0, native_spacing=(1.5, 1.5, 1.5), T=8,
                       write_ct_stats=True):
    import json as _json
    for s in ("s0", "s1"):
        d = tmp_path / s
        d.mkdir()
        v = np.linspace(0, 1000, 20 * 20 * 20, dtype=np.float32).reshape(20, 20, 20)
        np.save(d / "ct_raw.npy", v.astype(np.float32))
        lbl = np.zeros((20, 20, 20), dtype=np.uint8)
        lbl[8:12, 8:12, 8:12] = 1
        np.save(d / "label.npy", lbl)
    meta = {s: {"spacing": list(native_spacing), "shape": [20, 20, 20],
                "affine": np.eye(4).tolist()} for s in ("s0", "s1")}
    (tmp_path / "spacings.json").write_text(_json.dumps(meta))
    if write_ct_stats:
        # deliberately distinct per-subject stats, so a stats mixup is detectable
        stats = {"s0": {"clip_lo": 50.0, "clip_hi": 900.0, "mean": 400.0, "std": 100.0},
                 "s1": {"clip_lo": 10.0, "clip_hi": 300.0, "mean": 150.0, "std": 50.0}}
        (tmp_path / "ct_stats.json").write_text(_json.dumps(stats))
    return _FakeMriProvider(
        root=str(tmp_path), classes=["blob"], image_size=(T, T, T),
        crop_spacing_mm=spacing_mm, crop_jitter=0,
        mask_downsample="soft", mask_occupancy_thr=0.5)


def test_mri_modality_missing_ct_stats_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="ct_stats.json"):
        _tiny_mri_provider(tmp_path, write_ct_stats=False)


def test_mri_modality_load_uses_per_subject_stats(tmp_path):
    from src.totalseg_dataset import normalize_mri

    prov = _tiny_mri_provider(tmp_path, spacing_mm=1.5)
    center = (10, 10, 10)
    for subj, stats in (("s0", {"clip_lo": 50.0, "clip_hi": 900.0, "mean": 400.0, "std": 100.0}),
                        ("s1", {"clip_lo": 10.0, "clip_hi": 300.0, "mean": 150.0, "std": 50.0})):
        r = prov.load(subj, "blob",
                      LoadRequest(rng=random.Random(0), crop_spacing_mm=1.5, center=center, jitter=0))
        raw = np.load(tmp_path / subj / "ct_raw.npy")
        expect = normalize_mri(np.ascontiguousarray(raw[6:14, 6:14, 6:14]), stats)
        # place_image resamples/pads -- just check the normalized range matches the stats
        # frame (a mixed-up stats dict would clip/scale to a visibly different range).
        lo_expect, hi_expect = (stats["clip_lo"] - stats["mean"]) / stats["std"], \
                               (stats["clip_hi"] - stats["mean"]) / stats["std"]
        assert r.image.min().item() >= min(lo_expect, hi_expect) - 1e-3
        assert r.image.max().item() <= max(lo_expect, hi_expect) + 1e-3


def test_mri_modality_load_native_crop_norm_matches_ct_stats(tmp_path):
    prov = _tiny_mri_provider(tmp_path, spacing_mm=1.5)
    nc = prov.load_native_crop(
        "s0", "blob",
        LoadRequest(rng=random.Random(0), crop_spacing_mm=1.5, center=(10, 10, 10), jitter=0))
    assert nc.norm.clip_lo == 50.0 and nc.norm.clip_hi == 900.0
    assert nc.norm.mean == 400.0 and nc.norm.std == 100.0


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
