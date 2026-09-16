"""SynthGmmProvider cascade center_mode tests (data.cascade_center_mode="random_fg").

Uses a tiny synthetic 2-mask bank (real CohortSampler/SynthGmmMaisiDataset, no real
gmm_bank needed) so these run standalone. `organ_crop_arrays` is wrapped (not replaced)
via mock.patch(wraps=...) purely to observe the `center` argument each call resolves to.
"""
import pickle
import random
from unittest.mock import patch

import numpy as np
import pytest

from data.maisi_classes import SHAPE_ID_TO_FAMILY
from src.incontext_dataset_v2 import LoadRequest
from src.providers.synth_gmm import (
    HeterogeneitySpec, SynthGmmProvider, TextureSpec, _fractal_value_noise, _heterogeneity_map,
)
from src.shapes3d.spec import ShapeCohortSpec
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
    """Run load_native_crop, return the shape of the NATIVE (possibly capped) crop
    actually shipped in the returned NativeCrop.image -- _build_nc ships the native
    painted crop directly now (docs/logs.md 2026-09-15), rather than resampling it to
    the T grid itself, so `nc.image.shape` IS the post-cap native shape."""
    nc = provider.load_native_crop(subject, cls, req)
    return tuple(nc.image.shape)


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


def test_build_nc_uses_precomputed_fg_samples_without_scanning_arr(tmp_path, monkeypatch):
    """entry carries fg_samples (add_fg_samples_to_bank.py output) -> random_fg draws from
    it in O(1); proven here by breaking np.argwhere (the live-scan fallback) and confirming
    the resolved center still lands on one of the stored samples, not FALLBACK_CENT."""
    bank_dir = _make_bank(tmp_path)
    with open(bank_dir / "index.pkl", "rb") as f:
        index = pickle.load(f)
    fg_samples = np.array([[2, 2, 2], [3, 3, 3]])
    for e in index["entries"]:
        e["fg_samples"] = {CLS: fg_samples}
    with open(bank_dir / "index.pkl", "wb") as f:
        pickle.dump(index, f)

    ds = SynthGmmMaisiDataset(bank_dir, image_size=(T, T, T), context_size=1,
                              crop_spacing_mm=3.0, classes=[CLS], maxid=256)
    provider = SynthGmmProvider(ds, cascade=True)
    subject = "m00000.npy|1|0"

    def _boom(*a, **k):
        raise AssertionError("np.argwhere must not run when fg_samples is stored")
    monkeypatch.setattr("src.providers.totalseg.np.argwhere", _boom)

    for seed in range(5):
        req = LoadRequest(rng=random.Random(seed), crop_spacing_mm=3.0,
                          center=None, center_mode="random_fg")
        center = _center_used(provider, subject, str(CLS), req)
        assert center in {(2, 2, 2), (3, 3, 3)}, center
        assert center != tuple(FALLBACK_CENT)


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


def _make_shape_provider(tmp_path, p_shape=1.0, shape_spec=None):
    bank_dir = _make_bank(tmp_path)
    ds = SynthGmmMaisiDataset(bank_dir, image_size=(T, T, T), context_size=1,
                              crop_spacing_mm=3.0, classes=[CLS], maxid=256)
    return SynthGmmProvider(ds, cascade=True, p_shape=p_shape,
                            shape_spec=shape_spec or ShapeCohortSpec())


# Wide variant for the crop-jitter reproduction test below. _make_bank's DIM=32 gives
# crop_sizes=16 < DIM=32 (not literally clamped), but its FALLBACK_CENT=(20,20,20) sits
# close enough to the array edge that the jitter window's high end (ideal+jitter=16)
# lands EXACTLY on smax=dim-crop_size=16 -- a coincidental boundary, not a generous
# interior. DIM_WIDE=96 with a centered fallback removes that coincidence outright
# (smax=80), so the crop genuinely has room to move on every axis, confirmed directly
# (20 distinct crop starts observed across 20 req.rng seeds during development).
_WIDE_DIM = 96
_WIDE_CENTER = [48, 48, 48]


def _make_wide_bank(tmp_path):
    masks_dir = tmp_path / "masks"
    masks_dir.mkdir()
    entries, size_vecs = [], []
    for i in range(2):
        arr = np.zeros((_WIDE_DIM, _WIDE_DIM, _WIDE_DIM), dtype=np.uint8)
        arr[2:4, 2:4, 2:4] = CLS
        fname = f"m{i:05d}.npy"
        np.save(masks_dir / fname, arr)
        counts = np.bincount(arr.ravel(), minlength=256).astype(np.float64)
        size_vecs.append((counts / counts[1:].sum()).astype(np.float32))
        entries.append({"file": fname, "spacing": [3.0, 3.0, 3.0], "dim": [_WIDE_DIM] * 3,
                         "label_list": [CLS], "span": (1, 1), "cents": {CLS: _WIDE_CENTER}})
    index = {"maxid": 256, "spacing": 3.0, "entries": entries, "size_mat": np.stack(size_vecs)}
    with open(tmp_path / "index.pkl", "wb") as f:
        pickle.dump(index, f)
    return tmp_path


def _make_wide_shape_provider(tmp_path, p_shape=1.0, shape_spec=None, crop_jitter=None):
    bank_dir = _make_wide_bank(tmp_path)
    ds = SynthGmmMaisiDataset(bank_dir, image_size=(T, T, T), context_size=1,
                              crop_spacing_mm=3.0, classes=[CLS], maxid=256,
                              crop_jitter=crop_jitter)
    return SynthGmmProvider(ds, cascade=True, p_shape=p_shape,
                            shape_spec=shape_spec or ShapeCohortSpec())


def test_build_nc_handles_a_shape_stamped_into_a_full_array_readonly_crop(tmp_path):
    """Critical regression: when the resolved crop spans the FULL native mask array
    (crop_sizes == the mask's own native size on every axis -- here forced by a large
    crop_spacing_mm relative to the tiny DIM=32 fixture mask), organ_crop_arrays
    returns the array UNSLICED. That's already C-contiguous uint8, so
    np.ascontiguousarray returns a read-only VIEW sharing the same buffer (no copy) --
    confirmed directly: np.ascontiguousarray(full_slice) shares memory with the
    original mmap_mode="r" array and raises `ValueError: assignment destination is
    read-only` on write. This is independent of gpu_realize_max_native (the cap can
    only shrink a crop that's already bigger than it; it can never fire on a crop that
    already exactly fits the native array), so it is NOT exercised by the other
    tests in this file, which all use a small crop_spacing_mm that produces a
    genuinely-sliced (and thus already-copied) crop_lbl."""
    provider = _make_shape_provider(tmp_path, p_shape=1.0)
    rng = random.Random(0)
    # phys_ref = T*crop_mm = 16*12 = 192mm; target_sizes = round(192/spacing=3) = 64
    # >= DIM=32 -> organ_crop_arrays' crop_sizes clamp to DIM -> full, unsliced crop.
    task = provider.assemble_task(rng, crop_spacing_mm=12.0)
    for nc in task["native_crop"]:
        assert nc.class_idx in SHAPE_ID_TO_FAMILY


def test_assemble_task_shape_mode_returns_a_shape_pseudo_class(tmp_path):
    provider = _make_shape_provider(tmp_path, p_shape=1.0)
    rng = random.Random(0)
    task = provider.assemble_task(rng, crop_spacing_mm=3.0)
    ncs = task["native_crop"]
    assert len(ncs) == 2                      # context_size=1 -> target + 1 context
    for nc in ncs:
        assert nc.class_idx in SHAPE_ID_TO_FAMILY
        assert nc.has_fg                      # the shape was actually stamped and survived paint
    assert task["label_name"].startswith("shape_")


def test_assemble_task_p_shape_zero_never_returns_a_shape_class(tmp_path):
    provider = _make_shape_provider(tmp_path, p_shape=0.0)
    rng = random.Random(0)
    for _ in range(10):
        task = provider.assemble_task(rng, crop_spacing_mm=3.0)
        for nc in task["native_crop"]:
            assert nc.class_idx not in SHAPE_ID_TO_FAMILY


def test_load_native_crop_reproduces_the_same_shape_at_the_same_crop_spacing(tmp_path):
    """Re-crop consistency at a FIXED crop_spacing_mm: re-deriving from the subject
    string alone must reproduce the identical shape mask AND painted image.

    Uses crop_jitter=0 on BOTH sides: `assemble_task`'s own level-0 build applies
    self.ds.jitter same as any other call (jitter is a training-time augmentation, not
    something load_native_crop can retroactively cancel out), so a nonzero jitter would
    make even a real (non-shape) class's crop window land at a DIFFERENT native start
    on each call regardless of shape mode -- an apples-to-oranges comparison, not a
    shape regression. Pinning jitter=0 isolates the comparison to what this test
    actually means to prove: member_nrng-driven shape/paint consistency (keyed off
    gmm_seed/member_idx), independent of the crop window's own placement.

    (Reproducing across DIFFERENT req.rng jitter draws -- i.e. reconstructing a crop
    whose WINDOW itself is only known up to a random jitter offset -- was never a
    documented guarantee for any class, shape or real; it also isn't how any real
    cascade re-crop calls load_native_crop, which always supplies an explicit predicted
    `center` and thus jitter=0 anyway, see _build_nc's "no jitter for cascade recrops"
    comment. See test_load_native_crop_supports_a_different_crop_spacing_mm_without_raising
    below for the jittered/center=None path -- it only asserts no crash.)

    Physical consistency ACROSS DIFFERENT crop_spacing_mm -- the harder claim -- is
    covered separately: precisely, at the pure-function level, by
    src/shapes3d/test_instantiate.py::test_rasterize_shape_in_crop_is_physically_consistent_across_crop_resolutions."""
    provider = _make_wide_shape_provider(tmp_path, p_shape=1.0, crop_jitter=0)
    rng = random.Random(0)
    task = provider.assemble_task(rng, crop_spacing_mm=3.0)
    original = task["native_crop"][0]

    req = LoadRequest(rng=random.Random(0), crop_spacing_mm=3.0, center=None,
                      center_mode="com")
    rebuilt = provider.load_native_crop(task["subject"], task["label_name"], req)

    assert rebuilt.class_idx == original.class_idx
    np.testing.assert_array_equal(rebuilt.label_frac.numpy(), original.label_frac.numpy())
    np.testing.assert_array_equal(rebuilt.image.numpy(), original.image.numpy())


def test_shape_mode_subject_string_carries_the_host_class_as_a_fourth_field(tmp_path):
    provider = _make_shape_provider(tmp_path, p_shape=1.0)
    rng = random.Random(0)
    task = provider.assemble_task(rng, crop_spacing_mm=3.0)
    assert task["subject"].count("|") == 3       # filename|gmm_seed|member_idx|host<id>
    assert task["subject"].split("|")[3].startswith("host")


def test_between_ratio_zero_still_paints_a_nonempty_shape_in_every_member(tmp_path):
    """shape/size/position_between_ratio=0 pins the DRAWN PARAMETERS identical across
    the cohort (already exercised precisely, at the pure-function level, by
    src/shapes3d/test_instantiate.py::test_between_ratio_zero_reuses_the_cohort_value_exactly).
    Here we only need the provider-level plumbing check: with those params pinned, every
    member still ends up with a real, non-empty painted shape (has_fg) -- i.e. the pinned
    values survive _build_nc's crop/rasterize/paint pipeline instead of e.g. silently
    landing outside every member's own crop bounds."""
    spec = ShapeCohortSpec(shape_between_ratio=0.0, size_between_ratio=0.0,
                          position_between_ratio=0.0)
    provider = _make_shape_provider(tmp_path, p_shape=1.0, shape_spec=spec)
    rng = random.Random(1)
    task = provider.assemble_task(rng, crop_spacing_mm=3.0)
    for nc in task["native_crop"]:
        assert nc.has_fg


def test_shape_mode_intensity_between_ratio_does_not_mutate_shared_mu(tmp_path):
    """Regression: when dataset.between_ratio is None (the default, as here -- no
    sd_between_ratio passed to SynthGmmMaisiDataset), _build_nc's `mu_e = mu` line
    does NOT copy. The shape branch's `mu_e[shape_id] = ...` write (under
    intensity_between_ratio) must copy first, or member 0's per-member intensity draw
    would corrupt the cohort-shared `mu` array in place before members 1..K (which all
    receive that same object from assemble_task's list comprehension) ever read it as
    their baseline -- defeating the isolation intensity_between_ratio exists to
    provide. Proven directly: snapshot the `mu` array immediately as `_draw_gmm`
    returns it (before any _build_nc call can touch it), then compare that same
    object's contents after assemble_task (and all its member _build_nc calls) has
    run -- any in-place write shows up as a mismatch."""
    spec = ShapeCohortSpec(intensity_between_ratio=0.7)
    provider = _make_shape_provider(tmp_path, p_shape=1.0, shape_spec=spec)
    rng = random.Random(0)

    captured = {}
    real_draw_gmm = provider._draw_gmm

    def _spy(gmm_seed):
        mu, sd = real_draw_gmm(gmm_seed)
        captured["mu_ref"] = mu
        captured["mu_snapshot"] = mu.copy()
        return mu, sd

    with patch.object(provider, "_draw_gmm", side_effect=_spy):
        provider.assemble_task(rng, crop_spacing_mm=3.0)

    assert "mu_ref" in captured, "shape mode did not fire -- fixture/seed assumption broken"
    np.testing.assert_array_equal(captured["mu_ref"], captured["mu_snapshot"])


def test_load_native_crop_supports_a_different_crop_spacing_mm_without_raising(tmp_path):
    """World-space fix regression: a shape-mode subject used to be REJECTED
    (NotImplementedError) whenever a cascade re-crop supplied a predicted (non-None)
    center, because shape size/position were crop-relative and could not be
    consistently re-derived at a different FOV. Now that size/position are anchored to
    the host's own centroid in physical (mm) units (see synth_gmm.py's shape branch and
    rasterize_shape_in_crop), both a predicted center AND a different crop_spacing_mm
    (like a different cascade level) must work without raising. Precise physical-size/
    position consistency across resolutions is proven at the pure-function level in
    src/shapes3d/test_instantiate.py::test_rasterize_shape_in_crop_is_physically_consistent_across_crop_resolutions;
    this test only proves the provider-level plumbing (native-origin lookup via
    crop_geom, cap-stride, spacing conversion) doesn't crash end-to-end."""
    provider = _make_wide_shape_provider(tmp_path, p_shape=1.0)
    rng = random.Random(0)
    task = provider.assemble_task(rng, crop_spacing_mm=6.0)
    assert task["subject"].split("|")[3].startswith("host")  # confirms shape mode fired

    for center, spacing in [(None, 3.0), (tuple(_WIDE_CENTER), 1.5)]:
        req = LoadRequest(rng=random.Random(0), crop_spacing_mm=spacing, center=center,
                          center_mode="com")
        rebuilt = provider.load_native_crop(task["subject"], task["label_name"], req)
        assert rebuilt.class_idx in SHAPE_ID_TO_FAMILY


def test_build_nc_ships_native_unresampled_crop(tmp_path):
    """docs/logs.md 2026-09-15: _build_nc no longer resamples to the T grid itself --
    the shipped NativeCrop.image/label_frac are at NATIVE (post-cap) resolution, with
    real out_sizes/pad_lo from organ_crop_arrays (not a hardcoded [T,T,T]/[0,0,0]),
    decim=step, so the already-active gpu_realize_crop step does the real resample."""
    provider = _make_provider(tmp_path)
    subject = "m00000.npy|1|0"
    req = LoadRequest(rng=random.Random(0), crop_spacing_mm=3.0, center=None, center_mode="com")
    nc = provider.load_native_crop(subject, str(CLS), req)
    assert nc.image.shape == nc.label_frac.shape
    assert nc.decim == (1, 1, 1)          # no cap fired at this fixture's small DIM
    assert list(nc.out_sizes) != [0, 0, 0] and list(nc.pad_lo) is not None


def test_build_nc_ships_paint_mask_aligned_target_gaussian_when_enabled(tmp_path):
    bank_dir = _make_bank(tmp_path)
    ds = SynthGmmMaisiDataset(bank_dir, image_size=(T, T, T), context_size=1,
                              crop_spacing_mm=3.0, classes=[CLS], maxid=256,
                              paint_mask_aligned=True)
    provider = SynthGmmProvider(ds, cascade=True)
    rng = random.Random(0)
    task = provider.assemble_task(rng, crop_spacing_mm=3.0)
    nc = task["native_crop"][0]
    mu, sd = provider._draw_gmm(int(task["subject"].split("|")[1]))
    assert nc.paint_mask_aligned is True
    assert nc.target_mu == pytest.approx(float(mu[CLS]))
    assert nc.target_sd == pytest.approx(float(sd[CLS]))


def test_build_nc_omits_paint_mask_aligned_target_by_default(tmp_path):
    provider = _make_provider(tmp_path)   # paint_mask_aligned defaults to False
    rng = random.Random(0)
    task = provider.assemble_task(rng, crop_spacing_mm=3.0)
    nc = task["native_crop"][0]
    assert nc.paint_mask_aligned is False
    assert nc.target_mu is None and nc.target_sd is None


# --- _fractal_value_noise / TextureSpec (multi-octave correlated paint noise) ---
# See docs/datasets/eval_expansion_status.md investigation: real tissue has lag-1
# voxel autocorrelation ~0.6, the old per-voxel i.i.d. GMM noise has ~0. TextureSpec's
# default (n_octaves=1) must reduce to exactly that old i.i.d. behavior -- opt-in only.

def test_fractal_value_noise_default_is_byte_identical_to_plain_standard_normal():
    shape = (6, 7, 5)
    expected = np.random.default_rng(42).standard_normal(shape).astype(np.float32)
    actual = _fractal_value_noise(shape, np.random.default_rng(42), (1.0, 1.0, 1.0), TextureSpec())
    np.testing.assert_array_equal(actual, expected)


def test_fractal_value_noise_multi_octave_is_zero_mean_unit_variance():
    shape = (40, 40, 40)
    spec = TextureSpec(n_octaves=4, persistence=0.5, base_scale_mm=15.0)
    field = _fractal_value_noise(shape, np.random.default_rng(0), (1.0, 1.0, 1.0), spec)
    assert abs(float(field.mean())) < 0.1
    assert float(field.std()) == pytest.approx(1.0, abs=0.05)


def test_fractal_value_noise_multi_octave_is_spatially_correlated_unlike_plain_noise():
    shape = (40, 40, 40)
    spec = TextureSpec(n_octaves=4, persistence=0.5, base_scale_mm=15.0)
    field = _fractal_value_noise(shape, np.random.default_rng(1), (1.0, 1.0, 1.0), spec)
    plain = np.random.default_rng(1).standard_normal(shape).astype(np.float32)

    def _lag1_autocorr(a):
        a0, a1 = a[:-1], a[1:]
        return np.corrcoef(a0.ravel(), a1.ravel())[0, 1]

    assert _lag1_autocorr(field) > 0.3
    assert abs(_lag1_autocorr(plain)) < 0.05


def test_fractal_value_noise_is_deterministic_given_the_same_seed():
    shape = (20, 20, 20)
    spec = TextureSpec(n_octaves=3, persistence=0.6, base_scale_mm=10.0)
    a = _fractal_value_noise(shape, np.random.default_rng(7), (2.0, 2.0, 2.0), spec)
    b = _fractal_value_noise(shape, np.random.default_rng(7), (2.0, 2.0, 2.0), spec)
    np.testing.assert_array_equal(a, b)


def test_provider_texture_spec_multi_octave_changes_painted_image_not_geometry(tmp_path):
    """Same seed/geometry either way (texture only changes HOW noise is drawn, not the
    RNG calls that resolve crop window/center) -- label_frac identical, image differs.

    background_mode="uniform" (sd[0] > 0) makes this robust even though this fixture's
    "com" crop window is centered far from the tiny fg cube (see FALLBACK_CENT comment
    above) and so is mostly/entirely background class 0 -- with the default "zero" bg
    mode sd[0]==0 and the noise generator's choice would never show up in the image."""
    bank_dir = _make_bank(tmp_path)
    ds = SynthGmmMaisiDataset(bank_dir, image_size=(T, T, T), context_size=1,
                              crop_spacing_mm=3.0, classes=[CLS], maxid=256,
                              background_mode="uniform")
    provider_plain = SynthGmmProvider(ds, cascade=True)
    provider_textured = SynthGmmProvider(
        ds, cascade=True,
        texture_spec=TextureSpec(n_octaves=4, persistence=0.5, base_scale_mm=6.0))

    task_plain = provider_plain.assemble_task(random.Random(0), crop_spacing_mm=3.0)
    task_textured = provider_textured.assemble_task(random.Random(0), crop_spacing_mm=3.0)
    nc_plain, nc_textured = task_plain["native_crop"][0], task_textured["native_crop"][0]

    np.testing.assert_array_equal(nc_plain.label_frac.numpy(), nc_textured.label_frac.numpy())
    assert not np.allclose(nc_plain.image.numpy(), nc_textured.image.numpy())


# --- HeterogeneitySpec / _heterogeneity_map (core/rim multi-region target paint) ---
# See docs/logs.md 2026-09-16: SyntheticTumors/DiffTumor-style threshold-noise blob map,
# used to blend a target mask's mu between a "rim" (the class's own mu_e) and a fresh
# "core" draw, layered UNDER TextureSpec's per-voxel noise (unaffected by this feature).

def _het_spec(**kw):
    return HeterogeneitySpec(**kw)


def test_heterogeneity_map_values_are_in_unit_range_within_mask():
    mask = np.ones((30, 30, 30), dtype=bool)
    m = _heterogeneity_map(mask, np.random.default_rng(0), _het_spec())
    assert m.shape == mask.shape
    assert m[mask].min() >= 0.0 and m[mask].max() <= 1.0


def test_heterogeneity_map_respects_core_fraction_roughly():
    mask = np.ones((40, 40, 40), dtype=bool)
    spec = _het_spec(core_fraction=0.3, sigma1_range=(2.0, 2.0), sigma2_range=(0.5, 0.5))
    m = _heterogeneity_map(mask, np.random.default_rng(1), spec)
    core_frac_observed = float((m[mask] > 0.5).mean())
    assert 0.15 < core_frac_observed < 0.45   # loose band -- blur softens the hard cut


def test_heterogeneity_map_is_deterministic_given_the_same_seed():
    mask = np.ones((20, 20, 20), dtype=bool)
    spec = _het_spec()
    a = _heterogeneity_map(mask, np.random.default_rng(3), spec)
    b = _heterogeneity_map(mask, np.random.default_rng(3), spec)
    np.testing.assert_array_equal(a, b)


def test_heterogeneity_map_handles_an_empty_mask_without_raising():
    mask = np.zeros((10, 10, 10), dtype=bool)
    m = _heterogeneity_map(mask, np.random.default_rng(0), _het_spec())
    assert m.shape == mask.shape
    assert not m.any()


def _make_full_bank(tmp_path):
    """Bank where the ENTIRE native array is class CLS (not a small cube) -- so any
    resolved crop is fully foreground, needed to observe within-mask heterogeneity."""
    masks_dir = tmp_path / "masks"
    masks_dir.mkdir()
    entries, size_vecs = [], []
    for i in range(2):
        arr = np.full((DIM, DIM, DIM), CLS, dtype=np.uint8)
        fname = f"m{i:05d}.npy"
        np.save(masks_dir / fname, arr)
        counts = np.bincount(arr.ravel(), minlength=256).astype(np.float64)
        size_vecs.append((counts / counts[1:].sum()).astype(np.float32))
        entries.append({"file": fname, "spacing": [3.0, 3.0, 3.0], "dim": [DIM, DIM, DIM],
                         "label_list": [CLS], "span": (1, 1),
                         "cents": {CLS: [DIM // 2, DIM // 2, DIM // 2]}})
    index = {"maxid": 256, "spacing": 3.0, "entries": entries, "size_mat": np.stack(size_vecs)}
    with open(tmp_path / "index.pkl", "wb") as f:
        pickle.dump(index, f)
    return tmp_path


def _make_full_provider(tmp_path, p_heterogeneity=0.0, heterogeneity_spec=None):
    bank_dir = _make_full_bank(tmp_path)
    ds = SynthGmmMaisiDataset(bank_dir, image_size=(T, T, T), context_size=1,
                              crop_spacing_mm=3.0, classes=[CLS], maxid=256)
    return SynthGmmProvider(ds, cascade=True, p_heterogeneity=p_heterogeneity,
                            heterogeneity_spec=heterogeneity_spec)


def test_assemble_task_p_heterogeneity_zero_never_marks_subject_het(tmp_path):
    provider = _make_full_provider(tmp_path, p_heterogeneity=0.0)
    rng = random.Random(0)
    for _ in range(10):
        task = provider.assemble_task(rng, crop_spacing_mm=3.0)
        assert "het" not in task["subject"].split("|")
        for s in task["context_subjects"]:
            assert "het" not in s.split("|")


def test_assemble_task_p_heterogeneity_one_marks_subject_het(tmp_path):
    provider = _make_full_provider(tmp_path, p_heterogeneity=1.0)
    rng = random.Random(0)
    task = provider.assemble_task(rng, crop_spacing_mm=3.0)
    assert "het" in task["subject"].split("|")
    for s in task["context_subjects"]:
        assert "het" in s.split("|")


def test_build_nc_heterogeneous_target_paint_is_not_flat_within_mask(tmp_path):
    """p_heterogeneity=1.0 -> the painted image must show real spread within the fully-
    foreground mask (beyond what plain noise alone gives), i.e. the core/rim mu blend
    actually fires and changes the local mean, not just per-voxel noise."""
    bank_dir = _make_full_bank(tmp_path)
    ds = SynthGmmMaisiDataset(bank_dir, image_size=(T, T, T), context_size=1,
                              crop_spacing_mm=3.0, classes=[CLS], maxid=256, var_max=0.01)
    provider_flat = SynthGmmProvider(ds, cascade=True, p_heterogeneity=0.0)
    provider_het = SynthGmmProvider(ds, cascade=True, p_heterogeneity=1.0,
                                    heterogeneity_spec=HeterogeneitySpec(core_offset_ratio=3.0))

    task_flat = provider_flat.assemble_task(random.Random(0), crop_spacing_mm=3.0)
    task_het = provider_het.assemble_task(random.Random(0), crop_spacing_mm=3.0)
    img_flat = task_flat["native_crop"][0].image.numpy()
    img_het = task_het["native_crop"][0].image.numpy()

    assert img_flat.std() < 1e-3          # var_max~0 -> flat target is ~uniform
    assert img_het.std() > img_flat.std() * 5   # heterogeneous target has real structure


def test_load_native_crop_reproduces_the_same_heterogeneous_paint_at_the_same_spacing(tmp_path):
    provider = _make_full_provider(tmp_path, p_heterogeneity=1.0)
    rng = random.Random(0)
    ds = provider.ds
    ds.jitter = 0
    task = provider.assemble_task(rng, crop_spacing_mm=3.0)
    original = task["native_crop"][0]

    req = LoadRequest(rng=random.Random(0), crop_spacing_mm=3.0, center=None, center_mode="com")
    rebuilt = provider.load_native_crop(task["subject"], str(CLS), req)

    np.testing.assert_array_equal(rebuilt.image.numpy(), original.image.numpy())


def test_het_and_host_markers_both_parse_regardless_of_order(tmp_path):
    """Shape mode's |host<id> field and heterogeneity's |het field are independent and
    optional -- load_native_crop must handle a subject carrying both."""
    bank_dir = _make_full_bank(tmp_path)
    ds = SynthGmmMaisiDataset(bank_dir, image_size=(T, T, T), context_size=1,
                              crop_spacing_mm=3.0, classes=[CLS], maxid=256, crop_jitter=0)
    provider = SynthGmmProvider(ds, cascade=True, p_shape=1.0, p_heterogeneity=1.0,
                                shape_spec=ShapeCohortSpec())
    rng = random.Random(0)
    task = provider.assemble_task(rng, crop_spacing_mm=12.0)
    subject = task["subject"]
    assert "het" in subject.split("|")
    assert any(p.startswith("host") for p in subject.split("|"))

    req = LoadRequest(rng=random.Random(0), crop_spacing_mm=12.0, center=None, center_mode="com")
    rebuilt = provider.load_native_crop(subject, task["label_name"], req)
    assert rebuilt.class_idx in SHAPE_ID_TO_FAMILY
