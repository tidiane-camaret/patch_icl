"""Parity of realize_native_crops vs crop_and_place (plan Task 3)."""
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.incontext_dataset_v2 import LoadRequest
from src.providers.totalseg import crop_and_place, build_native_crop, NativeCrop
from src.totalseg_dataloader_incontext import organ_crop_arrays, _area_pool_3d
from src.totalseg_dataset import resolve_ct_norm, normalize_ct
from src.gpu_realize_crop import realize_native_crops, native_crop_collate_fn, _regroup
import torch.nn.functional as F


def _smooth_vol(D=24):
    a = np.linspace(-800, 400, D, dtype=np.float32)
    g = a[:, None, None] + a[None, :, None] * 0.3 + a[None, None, :] * 0.6
    return g.astype(np.float16)


def _native_crop_from(image_np, label_np, cls_idx, center, T, spacing, spec=None):
    """Same payload TotalSegProvider.load_native_crop builds (shared build_native_crop)."""
    crop_ct, crop_lbl, out_sizes, pad_lo, geom = organ_crop_arrays(
        image_np, label_np, center, [1.5, 1.5, 1.5], image_size=(T, T, T),
        crop_mm=spacing, jitter=0, rng=random.Random(0))
    return build_native_crop(crop_ct, crop_lbl, cls_idx, out_sizes, pad_lo, geom,
                             crop_spacing_mm=spacing,
                             ct_spec=(spec if spec is not None else resolve_ct_norm(None)))


def _fake_nc(T, class_idx=7, fg=None, spacing=3.0):
    """Minimal hand-built NativeCrop for shape/plumbing tests (no provider I/O)."""
    geom = torch.tensor([[0, 0, 0], [T, T, T], [T, T, T], [0, 0, 0]], dtype=torch.long)
    frac = torch.zeros(T, T, T, dtype=torch.float16)
    if fg is not None:
        frac[fg] = 1.0
    return NativeCrop(image=torch.zeros(T, T, T, dtype=torch.float16), label_frac=frac,
                      class_idx=class_idx, has_fg=bool(frac.any()),
                      out_sizes=[T, T, T], pad_lo=[0, 0, 0], crop_geom=geom,
                      crop_spacing_mm=spacing, decim=(1, 1, 1))


def _reference(image_np, label_np, cls_idx, center, T, spacing, md, thr, spec,
               antialias=False):
    return crop_and_place(
        image_np, label_np, cls_idx, center, T, crop_spacing_mm=spacing,
        native_spacing=(1.5, 1.5, 1.5), jitter=0, rng=random.Random(0),
        mask_downsample=md, occ_thr=thr, antialias=antialias,
        normalize_fn=lambda a: normalize_ct(a, spec))


def test_image_and_geom_parity_no_pad_spacing_1p5():
    spec = resolve_ct_norm(None)
    img = _smooth_vol(24); lbl = np.zeros((24, 24, 24), np.uint8); lbl[8:16, 8:16, 8:16] = 3
    T, s, center = 8, 1.5, (12, 12, 12)
    nc = _native_crop_from(img, lbl, 3, center, T, s)
    out = realize_native_crops([[nc]], T=T, mask_downsample="soft", occ_thr=0.5,
                               ct_spec=spec, device="cpu")
    img_ref, lbl_ref, geom_ref = _reference(img, lbl, 3, center, T, s, "soft", 0.5, spec)
    assert torch.equal(out["crop_geom"][0], geom_ref)
    assert (out["image"][0] - img_ref).abs().max() < 2e-2
    assert (out["label"][0] - lbl_ref.float()).abs().max() < 1e-4


def test_image_parity_decimated_spacing_3():
    spec = resolve_ct_norm(None)
    img = _smooth_vol(40); lbl = np.zeros((40, 40, 40), np.uint8); lbl[14:26, 14:26, 14:26] = 3
    T, s, center = 8, 3.0, (20, 20, 20)
    nc = _native_crop_from(img, lbl, 3, center, T, s)
    assert nc.decim == (2, 2, 2)
    out = realize_native_crops([[nc]], T=T, mask_downsample="occupancy", occ_thr=0.5,
                               ct_spec=spec, device="cpu")
    img_ref, lbl_ref, _ = _reference(img, lbl, 3, center, T, s, "occupancy", 0.5, spec)
    assert (out["image"][0] - img_ref).abs().max() < 2e-2
    inter = (out["label"][0].bool() & lbl_ref.bool()).sum()
    dice = (2 * inter / (out["label"][0].bool().sum() + lbl_ref.bool().sum())).item()
    assert dice == 1.0


def test_occupancy_never_empty():
    spec = resolve_ct_norm(None)
    img = _smooth_vol(24); lbl = np.zeros((24, 24, 24), np.uint8); lbl[12, 12, 12] = 3   # 1 voxel
    nc = _native_crop_from(img, lbl, 3, (12, 12, 12), 8, 1.5)
    out = realize_native_crops([[nc]], T=8, mask_downsample="occupancy", occ_thr=0.5,
                               ct_spec=spec, device="cpu")
    assert int(out["label"][0].sum()) >= 1


def test_regroup_and_collate():
    # grouping is by each task's own row index, not by a B*(K+1) stride
    assert _regroup(list(range(6)), [0, 0, 0, 1, 1, 1], 2) == [[0, 1, 2], [3, 4, 5]]
    # a ragged K (row 0 has 2 members, row 1 has 4) still groups correctly
    assert _regroup(list(range(6)), [0, 0, 1, 1, 1, 1], 2) == [[0, 1], [2, 3, 4, 5]]
    spec = resolve_ct_norm(None)
    img = _smooth_vol(24); lbl = np.zeros((24, 24, 24), np.uint8); lbl[10:14, 10:14, 10:14] = 3
    nc = _native_crop_from(img, lbl, 3, (12, 12, 12), 8, 1.5)
    items = [{"native_crop": [nc, nc], "subject": "s0",
              "context_subjects": ["c0"], "label_name": "liver",
              "aug_mode": torch.tensor(0)}]
    b = native_crop_collate_fn(items)
    assert isinstance(b["native_crop"], list) and len(b["native_crop"][0]) == 2
    assert b["subjects"] == ["s0"] and b["label_names"] == ["liver"]


def _smooth_box(shape):
    """Smooth separable ramp of arbitrary (D,H,W) shape (fp16), like _smooth_vol but not cubic."""
    zs = [np.linspace(-800, 400, s, dtype=np.float32) for s in shape]
    g = (zs[0][:, None, None] + zs[1][None, :, None] * 0.3 + zs[2][None, None, :] * 0.6)
    return g.astype(np.float16)


def test_pad_branch_parity():
    # Thin D axis (size 6) so organ_crop_arrays returns out_sizes[0]=6 < T with pad_lo[0]=1:
    # exercises the centre-pad branch of _realize_member (float(img.min()) air fill + placement).
    spec = resolve_ct_norm(None)
    img = _smooth_box((6, 40, 40))
    lbl = np.zeros((6, 40, 40), np.uint8); lbl[1:5, 18:22, 18:22] = 3
    T, s, center = 8, 1.5, (3, 20, 20)
    nc = _native_crop_from(img, lbl, 3, center, T, s)
    assert nc.decim == (1, 1, 1)
    assert nc.out_sizes[0] < T and nc.pad_lo[0] > 0
    out = realize_native_crops([[nc]], T=T, mask_downsample="soft", occ_thr=0.5,
                               ct_spec=spec, device="cpu")
    img_ref, lbl_ref, geom_ref = _reference(img, lbl, 3, center, T, s, "soft", 0.5, spec)

    assert out["image"].shape == (1, 1, T, T, T)
    assert torch.equal(out["crop_geom"][0], geom_ref)

    p0 = nc.pad_lo[0]; o0 = nc.out_sizes[0]
    got = out["image"][0, 0]                                    # (T,T,T)
    air = float(got.min())
    lo_pad, hi_pad = got[:p0], got[p0 + o0:]
    assert lo_pad.numel() > 0 and hi_pad.numel() > 0
    assert (lo_pad == air).all() and (hi_pad == air).all()     # uniform air fill
    # in-box region matches the reference crop_and_place
    assert (got[p0:p0 + o0] - img_ref[0, p0:p0 + o0]).abs().max() < 2e-2
    # the float(img.min()) air rule: pad value == reference pad value (both = resampled-crop min)
    assert abs(air - float(img_ref[0, :p0].min())) < 1e-4
    assert (img_ref[0, :p0] == float(img_ref[0, :p0].min())).all()
    # soft mask parity
    assert (out["label"][0] - lbl_ref.float()).abs().max() < 1e-4


def test_batch_and_context_shapes():
    # B=2, K=2: exercises B>1 stacking and the target/context split (img[:,0] vs img[:,1:]).
    spec = resolve_ct_norm(None)
    img = _smooth_vol(24)
    lbl_t = np.zeros((24, 24, 24), np.uint8); lbl_t[10:14, 10:14, 10:14] = 3   # target blob
    lbl_c = np.zeros((24, 24, 24), np.uint8); lbl_c[8:12, 8:12, 8:12] = 3      # context blob
    T, s, center = 8, 1.5, (12, 12, 12)
    nc_t = _native_crop_from(img, lbl_t, 3, center, T, s)
    nc_c = _native_crop_from(img, lbl_c, 3, center, T, s)
    members = [[nc_t, nc_c, nc_c], [nc_t, nc_c, nc_c]]

    out = realize_native_crops(members, T=T, mask_downsample="occupancy", occ_thr=0.1,
                               ct_spec=spec, device="cpu")
    assert out["image"].shape == (2, 1, 8, 8, 8)
    assert out["context_in"].shape == (2, 2, 1, 8, 8, 8)
    assert out["label"].shape == (2, 8, 8, 8)
    assert out["context_out"].shape == (2, 2, 8, 8, 8)
    assert out["spacing"].shape == (2, 3)
    assert out["crop_geom"].shape == (2, 4, 3)
    assert out["label"].dtype == torch.int64
    assert out["context_out"].dtype == torch.int64

    soft = realize_native_crops(members, T=T, mask_downsample="soft", occ_thr=0.1,
                                ct_spec=spec, device="cpu")
    assert soft["label"].dtype == torch.float32

    # target blob at rel [2:6]^3, context blob at rel [0:4]^3 — split must not be transposed.
    tgt = out["label"][0]
    ctx = out["context_out"][0, 0]
    assert int(tgt.sum()) == 64 and int(tgt[2:6, 2:6, 2:6].sum()) == 64
    assert int(ctx.sum()) == 64 and int(ctx[0:4, 0:4, 0:4].sum()) == 64
    assert int(tgt[0, 0, 0]) == 0 and int(ctx[0, 0, 0]) == 1
    assert torch.equal(out["label"][0], out["label"][1])       # identical rows stack cleanly


def test_engine_emits_native_crop_payload(tmp_path):
    import random
    from src.incontext_dataset_v2 import InContextDataset, LoadRequest
    from src.providers.totalseg import NativeCrop

    T = 8
    geom = torch.tensor([[0, 0, 0], [T, T, T], [T, T, T], [0, 0, 0]], dtype=torch.long)

    class P:
        classes = ["liver"]
        def subjects_for(self, cls): return ["a", "b", "c", "d"]
        def load_native_crop(self, subject, cls, req):
            return NativeCrop(image=torch.zeros(T, T, T, dtype=torch.float16),
                              label_frac=torch.zeros(T, T, T, dtype=torch.float16),
                              class_idx=3, has_fg=False,
                              out_sizes=[T, T, T], pad_lo=[0, 0, 0],
                              crop_geom=geom, crop_spacing_mm=req.crop_spacing_mm,
                              decim=(1, 1, 1))

    ds = InContextDataset(P(), context_size=3, class_balanced=False,
                          crop_spacing_mm=3.0, gpu_realize_crop=True)
    ds.samples = [("a", "liver")]
    item = ds[0]
    assert "image" not in item and "native_crop" in item
    assert len(item["native_crop"]) == 4                # target + 3 contexts
    assert all(isinstance(x, NativeCrop) for x in item["native_crop"])
    assert item["subject"] == "a" and item["label_name"] == "liver"
    assert set(item["context_subjects"]) <= {"b", "c", "d"}
    assert len(item["context_subjects"]) == 3
    assert int(item["aug_mode"]) == 0


# ---------------------------------------------------------------------------
# Hard parity fixture (review finding I2): the smooth-ramp / even-aligned-cube
# cases above are degenerate -- box and trilinear agree exactly, nothing exceeds
# the clip window, and cubes survive 2x subsampling. This one has additive noise,
# a supra-clip_hi bone shell, and an OFF-grid rotated ellipsoid label, so it
# catches (a) a point-sampled instead of partial-volume mask and (b) a
# CT-normalize applied after (rather than before) the decimation/resample.
# ---------------------------------------------------------------------------

def _hard_volume(D=48, seed=0):
    """Smooth base + additive noise + a spherical shell far ABOVE clip_hi (2500 HU)."""
    rng = np.random.default_rng(seed)
    ax = np.linspace(-600, 300, D, dtype=np.float32)
    base = ax[:, None, None] + 0.4 * ax[None, :, None] + 0.7 * ax[None, None, :]
    base = base + rng.normal(0, 120, size=(D, D, D)).astype(np.float32)
    zz, yy, xx = np.meshgrid(*[np.arange(D, dtype=np.float32)] * 3, indexing="ij")
    r = np.sqrt((zz - 23.4) ** 2 + (yy - 24.6) ** 2 + (xx - 22.7) ** 2)
    base[(r > 9.3) & (r < 12.7)] = 2500.0                  # > clip_hi
    return base.astype(np.float16)


def _off_grid_ellipsoid(D, center, radii):
    """Rotated, non-integer-centred ellipsoid -- no axis is voxel- or even-aligned."""
    zz, yy, xx = np.meshgrid(*[np.arange(D, dtype=np.float32)] * 3, indexing="ij")
    p = np.stack([zz - center[0], yy - center[1], xx - center[2]], 0)
    cz, sz = np.cos(0.37), np.sin(0.37)
    cy, sy = np.cos(0.61), np.sin(0.61)
    M = (np.array([[1, 0, 0], [0, cy, -sy], [0, sy, cy]], np.float32)
         @ np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], np.float32))
    q = np.tensordot(M, p, axes=(1, 0))
    return sum((q[i] / radii[i]) ** 2 for i in range(3)) <= 1.0


def test_hard_parity_noise_supraclip_offgrid_ellipsoid_decim2():
    """decim=2 (3 mm from a 1.5 mm grid): full parity vs crop_and_place."""
    spec = resolve_ct_norm(None)
    D, T, s = 48, 16, 3.0
    img = _hard_volume(D)
    lbl = np.zeros((D, D, D), np.uint8)
    lbl[_off_grid_ellipsoid(D, (23.4, 24.6, 22.7), (5.3, 3.7, 4.9))] = 3
    center = (D // 2, D // 2, D // 2)
    nc = _native_crop_from(img, lbl, 3, center, T, s, spec)
    assert nc.decim == (2, 2, 2) and nc.has_fg

    soft = realize_native_crops([[nc]], T=T, mask_downsample="soft", occ_thr=0.5,
                                ct_spec=spec, device="cpu")
    ref_i, ref_soft, ref_g = _reference(img, lbl, 3, center, T, s, "soft", 0.5, spec)
    d = (soft["image"][0] - ref_i).abs()
    assert torch.equal(soft["crop_geom"][0], ref_g)
    assert d.max() < 2e-2, f"image max|d|={d.max():.4f}"
    assert d.mean() < 2e-3, f"image mean|d|={d.mean():.5f}"
    dm = (soft["label"][0] - ref_soft.float()).abs()
    assert dm.max() < 1e-4, f"soft mask max|d|={dm.max():.5f}"

    occ = realize_native_crops([[nc]], T=T, mask_downsample="occupancy", occ_thr=0.5,
                               ct_spec=spec, device="cpu")
    _, ref_occ, _ = _reference(img, lbl, 3, center, T, s, "occupancy", 0.5, spec)
    a, b = occ["label"][0].bool(), ref_occ.bool()
    assert int(b.sum()) > 0
    dice = (2 * (a & b).sum() / (a.sum() + b.sum())).item()
    assert dice == 1.0, f"occupancy dice={dice:.4f} (ours {int(a.sum())} vs ref {int(b.sum())})"


def test_hard_parity_small_label_survives_decim4():
    """decim=4 (6 mm) + a few-native-voxel label: a strided subsample would drop it."""
    spec = resolve_ct_norm(None)
    D, T, s = 48, 16, 6.0
    img = _hard_volume(D)
    lbl = np.zeros((D, D, D), np.uint8)
    lbl[_off_grid_ellipsoid(D, (23.4, 24.6, 22.7), (1.6, 1.3, 1.9))] = 3
    assert 0 < int((lbl == 3).sum()) < 30            # genuinely sub-cell at 6 mm
    center = (D // 2, D // 2, D // 2)
    nc = _native_crop_from(img, lbl, 3, center, T, s, spec)
    assert nc.decim == (4, 4, 4) and nc.has_fg
    assert nc.pad_lo[0] > 0                          # also exercises the centre-pad branch

    occ = realize_native_crops([[nc]], T=T, mask_downsample="occupancy", occ_thr=0.5,
                               ct_spec=spec, device="cpu")
    _, ref_occ, ref_g = _reference(img, lbl, 3, center, T, s, "occupancy", 0.5, spec)
    assert int(occ["label"][0].sum()) > 0, "small class vanished under decimation"
    a, b = occ["label"][0].bool(), ref_occ.bool()
    dice = (2 * (a & b).sum() / (a.sum() + b.sum())).item()
    assert dice == 1.0, f"occupancy dice={dice:.4f}"
    assert torch.equal(occ["crop_geom"][0], ref_g)

    soft = realize_native_crops([[nc]], T=T, mask_downsample="soft", occ_thr=0.5,
                                ct_spec=spec, device="cpu")
    _, ref_soft, _ = _reference(img, lbl, 3, center, T, s, "soft", 0.5, spec)
    assert float(soft["label"][0].sum()) > 0
    dm = (soft["label"][0] - ref_soft.float()).abs()
    assert dm.max() < 1e-4, f"soft mask max|d|={dm.max():.5f}"

    # Image parity at decim=4 is against the ANTIALIASED reference. `place_image`'s
    # default (antialias=False) trilinear point-samples: at scale 4 it weights only the
    # middle 2 of every 4 voxels per axis, so on a noisy volume it aliases. The realize
    # path area-pools (avg over all 4), which is what antialias=True does. The two agree
    # exactly at decim<=2 (trilinear x0.5 IS the 2^3 box average) -- see the decim=2 test.
    di = (soft["image"][0]
          - _reference(img, lbl, 3, center, T, s, "soft", 0.5, spec, antialias=True)[0]).abs()
    assert di.max() < 2e-2, f"image max|d|={di.max():.4f}"
    # The residual is dominated by the air-pad constant: place_image fills with the
    # FULL-RES normalized crop min, the realize path with the DECIMATED crop's min
    # (~2 HU higher once 4^3 voxels are averaged). ~4e-3 in normalized units.
    assert di.mean() < 5e-3, f"image mean|d|={di.mean():.5f}"
