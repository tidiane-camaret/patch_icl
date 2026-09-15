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
from src.providers.synth_gmm import SynthGmmProvider
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


def _make_wide_shape_provider(tmp_path, p_shape=1.0, shape_spec=None):
    bank_dir = _make_wide_bank(tmp_path)
    ds = SynthGmmMaisiDataset(bank_dir, image_size=(T, T, T), context_size=1,
                              crop_spacing_mm=3.0, classes=[CLS], maxid=256)
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
    """Single-level re-crop consistency ONLY: re-deriving from the subject string alone,
    at the SAME crop_spacing_mm assemble_task used, must reproduce the identical shape
    mask AND painted image. This does NOT test (and shape mode does not guarantee)
    reproducing the same shape across cascade levels that use a DIFFERENT
    crop_spacing_mm -- size_frac and position_uvw are resolved relative to the crop grid
    they're rasterized into, not in world/mm space, so a shape's absolute size/position
    can drift across levels with different physical FOVs. See ShapeCohortSpec's
    docstring and rasterize_shape_in_crop's docstring for the limitation.

    Uses _make_wide_shape_provider (DIM=96), not the module's default _make_shape_provider
    (DIM=32): with DIM=32, the crop-jitter window's high end coincidentally sits exactly at
    smax (dim - crop_size), so this test used to pass even with jitter/RNG-stream alignment
    completely broken -- reconstructing with req.rng seeded 0, 1, 2, 99 all produced
    bit-identical crops regardless, and only `.label_frac` was compared, which never
    touches the member_nrng-seeded paint path. DIM=96 removes that coincidence (crop
    genuinely lands at a different native start per req.rng seed -- verified directly:
    20/20 distinct starts across 20 seeds), and the assertions below additionally check
    that the crop start DOES move and that `.image` (which exercises `_resample_paint_mask`'s
    member_nrng stream, unlike `.label_frac`) still reproduces exactly regardless of where
    the (crop-relative) shape and paint land -- proving reproduction is driven by
    (gmm_seed, member_idx), not by hitting the same native crop window by chance."""
    provider = _make_wide_shape_provider(tmp_path, p_shape=1.0)
    rng = random.Random(0)
    task = provider.assemble_task(rng, crop_spacing_mm=3.0)
    original = task["native_crop"][0]

    req = LoadRequest(rng=random.Random(0), crop_spacing_mm=3.0, center=None,
                      center_mode="com")
    rebuilt = provider.load_native_crop(task["subject"], task["label_name"], req)

    assert rebuilt.class_idx == original.class_idx
    np.testing.assert_array_equal(rebuilt.label_frac.numpy(), original.label_frac.numpy())
    np.testing.assert_array_equal(rebuilt.image.numpy(), original.image.numpy())

    # Prove the crop-jitter mechanism this test is meant to exercise actually has room to
    # move (not a no-op): reconstructing at several different req.rng seeds must land on
    # more than one distinct native crop start.
    starts = {tuple(provider.load_native_crop(
        task["subject"], task["label_name"],
        LoadRequest(rng=random.Random(seed), crop_spacing_mm=3.0, center=None,
                    center_mode="com")).crop_geom[0].tolist())
        for seed in range(8)}
    assert len(starts) > 1, "crop start never moved across req.rng seeds -- jitter is a no-op"


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


def test_load_native_crop_rejects_a_predicted_center_for_a_shape_mode_subject(tmp_path):
    """Guard: shape size_frac/position_uvw are crop-relative, not world-relative (see
    docs/superpowers/specs/2026-09-14-cohort-consistent-synthetic-shapes-design.md sec 2
    and docs/logs.md's Known Limitation). A cascade re-crop with a predicted center
    (req.center is not None) would re-derive the shape into a DIFFERENT physical FOV than
    level 0 showed the model, silently supervising a different object. That must raise
    loudly instead of returning a wrong-but-plausible NativeCrop."""
    provider = _make_shape_provider(tmp_path, p_shape=1.0)
    rng = random.Random(0)
    task = provider.assemble_task(rng, crop_spacing_mm=3.0)
    assert task["subject"].split("|")[3].startswith("host")  # confirms shape mode fired

    req = LoadRequest(rng=random.Random(0), crop_spacing_mm=3.0, center=(1, 2, 3),
                      center_mode="com")
    with pytest.raises(NotImplementedError):
        provider.load_native_crop(task["subject"], task["label_name"], req)


def test_load_native_crop_allows_a_shape_mode_subject_when_center_is_none(tmp_path):
    """Complement of the guard test above: center=None (same-level reconstruction, the
    only case any current caller -- the reproduction test or a same-level realize -- ever
    passes) must NOT raise."""
    provider = _make_shape_provider(tmp_path, p_shape=1.0)
    rng = random.Random(0)
    task = provider.assemble_task(rng, crop_spacing_mm=3.0)
    assert task["subject"].split("|")[3].startswith("host")

    req = LoadRequest(rng=random.Random(0), crop_spacing_mm=3.0, center=None,
                      center_mode="com")
    rebuilt = provider.load_native_crop(task["subject"], task["label_name"], req)
    assert rebuilt.class_idx in SHAPE_ID_TO_FAMILY
