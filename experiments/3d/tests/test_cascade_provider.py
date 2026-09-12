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


def test_load_native_crop_label_is_partial_volume_fraction(tmp_path):
    """The payload label is a per-class FRACTION built pre-decimation, not a point sample."""
    import random

    from src.incontext_dataset_v2 import LoadRequest

    prov = _tiny_provider(tmp_path, spacing=3.0, T=8)     # decim=(2,2,2)
    nc = prov.load_native_crop(
        "s0", "liver",
        LoadRequest(rng=random.Random(0), crop_spacing_mm=3.0, center=(10, 10, 10), jitter=0))
    assert nc.decim == (2, 2, 2)
    assert nc.has_fg is True
    assert nc.label_frac.shape == tuple(nc.out_sizes)      # already at out_sizes
    f = nc.label_frac.float()
    assert 0.0 <= float(f.min()) and float(f.max()) <= 1.0
    # a 4^3 native cube under a 2x pool covers 8 whole cells -> total fraction mass = 8
    assert abs(float(f.sum()) - 4 ** 3 / 8) < 1e-3
    # image is HU-clipped to the ct_spec window BEFORE the decimation average
    assert float(nc.image.float().max()) <= prov.ct_spec.clip_hi + 1e-3


def test_provider_is_not_pickled_with_its_ram_cache(tmp_path):
    """forkserver/spawn eval workers pickle the dataset -> the provider. The RAM cache
    (tens of GB in a real run) must never ride along; workers only call load()."""
    import pickle

    prov = _tiny_provider(tmp_path, spacing=1.5, T=8)
    assert prov._ram is not None and len(prov._ram) >= 1     # cache really is populated
    blob = pickle.dumps(prov)
    assert len(blob) < 100_000, f"pickled provider is {len(blob)} bytes (cache leaked in?)"
    assert pickle.loads(blob)._ram is None
    assert prov._ram is not None                             # the live provider keeps it


# --- data.cascade_center_mode="random_fg" -------------------------------------
from src.providers.totalseg import _resolve_center


def test_resolve_center_prefers_explicit_center():
    req = LoadRequest(rng=random.Random(0), crop_spacing_mm=1.5,
                      center=(1, 2, 3), center_mode="random_fg")
    lbl = np.ones((4, 4, 4), dtype=np.uint8)
    assert _resolve_center(req, lbl, class_idx=1, fallback=(9, 9, 9)) == (1, 2, 3)


def test_resolve_center_com_mode_uses_fallback():
    req = LoadRequest(rng=random.Random(0), crop_spacing_mm=1.5, center_mode="com")
    lbl = np.ones((4, 4, 4), dtype=np.uint8)
    assert _resolve_center(req, lbl, class_idx=1, fallback=(9, 9, 9)) == (9, 9, 9)


def test_resolve_center_random_fg_lands_on_a_class_voxel():
    lbl = np.zeros((6, 6, 6), dtype=np.uint8)
    lbl[2, 3, 4] = 1                                  # single voxel of class 1
    req = LoadRequest(rng=random.Random(0), crop_spacing_mm=1.5, center_mode="random_fg")
    assert _resolve_center(req, lbl, class_idx=1, fallback=(0, 0, 0)) == (2, 3, 4)


def test_resolve_center_random_fg_falls_back_when_class_absent():
    lbl = np.zeros((6, 6, 6), dtype=np.uint8)          # no voxel == class 1
    req = LoadRequest(rng=random.Random(0), crop_spacing_mm=1.5, center_mode="random_fg")
    assert _resolve_center(req, lbl, class_idx=1, fallback=(3, 3, 3)) == (3, 3, 3)
    # unknown class (-1, e.g. _ALL_CLASSES_IDX miss) also falls back rather than scanning
    assert _resolve_center(req, lbl, class_idx=-1, fallback=(3, 3, 3)) == (3, 3, 3)


def test_resolve_center_random_fg_is_seeded_by_req_rng():
    lbl = np.zeros((6, 6, 6), dtype=np.uint8)
    lbl[1, 1, 1] = 1; lbl[4, 4, 4] = 1
    req = LoadRequest(rng=random.Random(0), crop_spacing_mm=1.5, center_mode="random_fg")
    a = _resolve_center(req, lbl, class_idx=1, fallback=(0, 0, 0))
    req2 = LoadRequest(rng=random.Random(0), crop_spacing_mm=1.5, center_mode="random_fg")
    b = _resolve_center(req2, lbl, class_idx=1, fallback=(0, 0, 0))
    assert a == b and a in {(1, 1, 1), (4, 4, 4)}


def test_load_native_crop_random_fg_center_lands_inside_the_liver_block(tmp_path):
    """End-to-end through TotalSegProvider.load_native_crop: center=None + center_mode=
    'random_fg' must pick a voxel actually inside the liver block (8:12 on every axis in
    the 20^3 fixture), unlike the fixed bbox-COM default which always returns its centroid."""
    from src.totalseg_dataset import _ALL_CLASSES_IDX

    prov = _tiny_provider(tmp_path, spacing=1.5, T=8)
    req = LoadRequest(rng=random.Random(1), crop_spacing_mm=1.5, jitter=0,
                      center_mode="random_fg")
    nc = prov.load_native_crop("s0", "liver", req)
    # starts (crop_geom row 0) recovers the resolved centre for a jitter=0, T==crop_sizes
    # request: ideal = center - crop_sizes//2 == starts, so center == starts + crop_sizes//2.
    starts, crop_sizes = nc.crop_geom[0].tolist(), nc.crop_geom[1].tolist()
    center = tuple(s + cs // 2 for s, cs in zip(starts, crop_sizes))
    assert all(8 <= c < 12 for c in center), \
        f"random_fg center {center} is outside the planted liver block [8,12)"

    # the "com" default keeps returning the bbox centroid (9,9,9) for the same request shape
    req_com = LoadRequest(rng=random.Random(1), crop_spacing_mm=1.5, jitter=0, center_mode="com")
    nc_com = prov.load_native_crop("s0", "liver", req_com)
    starts_c, crop_c = nc_com.crop_geom[0].tolist(), nc_com.crop_geom[1].tolist()
    center_com = tuple(s + cs // 2 for s, cs in zip(starts_c, crop_c))
    assert center_com == (9, 9, 9)
