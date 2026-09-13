"""SynthGmmProvider cascade center_mode tests (data.cascade_center_mode="random_fg").

Uses a tiny synthetic 2-mask bank (real CohortSampler/SynthGmmMaisiDataset, no real
gmm_bank needed) so these run standalone. `organ_crop_arrays` is wrapped (not replaced)
via mock.patch(wraps=...) purely to observe the `center` argument each call resolves to.
"""
import pickle
import random
from unittest.mock import patch

import numpy as np

from src.incontext_dataset_v2 import LoadRequest
from src.providers.synth_gmm import SynthGmmProvider
from src.synth_gmm_maisi_dataset import SynthGmmMaisiDataset
from src.totalseg_dataloader_incontext import organ_crop_arrays as _real_organ_crop_arrays

CLS = 5          # arbitrary MAISI-like id, doesn't need to be in the real vocab
DIM = 32         # native mask side (voxels)
T = 16           # model grid side
FALLBACK_CENT = [20, 20, 20]   # deliberately far from the fg cube below, so "com" vs
                                # "random_fg" are trivially distinguishable


def _make_bank(tmp_path):
    """One mask with an 8-voxel fg cube (not at FALLBACK_CENT) for class CLS."""
    masks_dir = tmp_path / "masks"
    masks_dir.mkdir()
    entries, size_vecs = [], []
    for i in range(2):  # SynthGmmMaisiDataset(context_size=1) needs >= k+1=2 masks/class
        arr = np.zeros((DIM, DIM, DIM), dtype=np.uint8)
        arr[2:4, 2:4, 2:4] = CLS
        fname = f"m{i:05d}.npy"
        np.save(masks_dir / fname, arr)
        counts = np.bincount(arr.ravel(), minlength=256).astype(np.float64)
        size_vecs.append((counts / counts[1:].sum()).astype(np.float32))
        entries.append({"file": fname, "spacing": [3.0, 3.0, 3.0], "dim": [DIM, DIM, DIM],
                         "label_list": [CLS], "span": (1, 1), "cents": {CLS: FALLBACK_CENT}})
    index = {"maxid": 256, "spacing": 3.0, "entries": entries, "size_mat": np.stack(size_vecs)}
    with open(tmp_path / "index.pkl", "wb") as f:
        pickle.dump(index, f)
    return tmp_path


def _make_provider(tmp_path):
    bank_dir = _make_bank(tmp_path)
    ds = SynthGmmMaisiDataset(bank_dir, image_size=(T, T, T), context_size=1,
                               crop_spacing_mm=3.0, classes=[CLS], maxid=256)
    return SynthGmmProvider(ds, cascade=True)


def _center_used(provider, subject, cls, req):
    """Run load_native_crop, return the native-voxel `center` organ_crop_arrays resolved to."""
    with patch("src.providers.synth_gmm.organ_crop_arrays",
               wraps=_real_organ_crop_arrays) as m:
        provider.load_native_crop(subject, cls, req)
    return m.call_args.args[2]


def test_center_mode_com_uses_the_precomputed_centroid(tmp_path):
    provider = _make_provider(tmp_path)
    subject = "m00000.npy|1|0"
    req = LoadRequest(rng=random.Random(0), crop_spacing_mm=3.0, center=None, center_mode="com")
    assert _center_used(provider, subject, str(CLS), req) == tuple(FALLBACK_CENT)


def test_center_mode_random_fg_samples_a_masked_voxel_deterministically(tmp_path):
    provider = _make_provider(tmp_path)
    subject = "m00000.npy|1|0"
    arr = np.load(tmp_path / "masks" / "m00000.npy")
    coords = np.argwhere(arr == CLS)

    for seed in range(5):
        req = LoadRequest(rng=random.Random(seed), crop_spacing_mm=3.0,
                          center=None, center_mode="random_fg")
        center = _center_used(provider, subject, str(CLS), req)
        # matches _resolve_center's own draw exactly (same algorithm, replayed independently)
        expected = tuple(int(v) for v in coords[random.Random(seed).randrange(coords.shape[0])])
        assert center == expected
        assert center != tuple(FALLBACK_CENT)  # proves it did NOT fall back to the centroid


def test_center_mode_random_fg_respects_an_explicit_center(tmp_path):
    """An explicit predicted center (cascade level>=1 on a non-empty prediction) always wins,
    regardless of center_mode -- matches TotalSegProvider's _resolve_center contract."""
    provider = _make_provider(tmp_path)
    subject = "m00000.npy|1|0"
    explicit = (5, 6, 7)
    req = LoadRequest(rng=random.Random(0), crop_spacing_mm=3.0,
                      center=explicit, center_mode="random_fg")
    assert _center_used(provider, subject, str(CLS), req) == explicit


def _native_shape_painted(provider, subject, cls, req):
    """Run load_native_crop, return the crop_lbl.shape actually handed to
    _resample_paint_mask (i.e. after any gpu_realize_max_native cap)."""
    from src.synth_gmm_maisi_dataset import SynthGmmMaisiDataset
    real = SynthGmmMaisiDataset._resample_paint_mask
    with patch.object(SynthGmmMaisiDataset, "_resample_paint_mask",
                      autospec=True, side_effect=real) as m:
        provider.load_native_crop(subject, cls, req)
    return tuple(m.call_args.args[1].shape)  # args: (self, crop_lbl, out_sizes, ...)


def test_build_nc_caps_native_crop_before_painting(tmp_path):
    """DIM (32) > gpu_realize_max_native (8) -> _build_nc must pre-downsample the native
    crop_lbl to <=8/axis before the expensive paint, mirroring _native_crop's cap."""
    bank_dir = _make_bank(tmp_path)
    ds = SynthGmmMaisiDataset(bank_dir, image_size=(T, T, T), context_size=1,
                              crop_spacing_mm=3.0, classes=[CLS], maxid=256,
                              gpu_realize_max_native=8)
    provider = SynthGmmProvider(ds, cascade=True)
    subject = "m00000.npy|1|0"
    req = LoadRequest(rng=random.Random(0), crop_spacing_mm=3.0, center=None, center_mode="com")
    shape = _native_shape_painted(provider, subject, str(CLS), req)
    assert max(shape) <= 8, shape


def test_build_nc_uncapped_when_gpu_realize_max_native_is_zero(tmp_path):
    """gpu_realize_max_native=0 (falsy) -> unchanged behavior, native crop_lbl untouched.

    organ_crop_arrays itself already shrinks DIM (32) down to the physical crop size
    (T*crop_mm/spacing = 16*3/3 = 16) before _build_nc ever sees it, so the uncapped
    shape here is 16, not DIM -- this pins that geometry so the cap test above (cap=8)
    is verified against a real >cap native shape rather than an already-small one."""
    bank_dir = _make_bank(tmp_path)
    ds = SynthGmmMaisiDataset(bank_dir, image_size=(T, T, T), context_size=1,
                              crop_spacing_mm=3.0, classes=[CLS], maxid=256,
                              gpu_realize_max_native=0)
    provider = SynthGmmProvider(ds, cascade=True)
    subject = "m00000.npy|1|0"
    req = LoadRequest(rng=random.Random(0), crop_spacing_mm=3.0, center=None, center_mode="com")
    shape = _native_shape_painted(provider, subject, str(CLS), req)
    assert shape == (16, 16, 16), shape


def test_native_crop_gpu_realize_path_also_caps_before_materializing(tmp_path):
    """SynthGmmMaisiDataset._native_crop (the standalone data.source=synth_gmm_maisi +
    gpu_realize=True path) shares the same stride-before-materialize cap fix as
    SynthGmmProvider._build_nc -- same bank fixture, direct call."""
    bank_dir = _make_bank(tmp_path)
    ds = SynthGmmMaisiDataset(bank_dir, image_size=(T, T, T), context_size=1,
                              crop_spacing_mm=3.0, classes=[CLS], maxid=256,
                              gpu_realize=True, gpu_realize_max_native=8)
    e = ds.cs.entries[0]
    native, out_sizes, pad_lo = ds._native_crop(e, CLS, random.Random(0), 3.0)
    assert max(native.shape) <= 8, tuple(native.shape)
