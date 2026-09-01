"""Parity of realize_native_crops vs crop_and_place (plan Task 3)."""
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.incontext_dataset_v2 import LoadRequest
from src.providers.totalseg import crop_and_place, NativeCrop
from src.totalseg_dataloader_incontext import organ_crop_arrays, _area_pool_3d
from src.totalseg_dataset import resolve_ct_norm, normalize_ct
from src.gpu_realize_crop import realize_native_crops, native_crop_collate_fn, _regroup
import torch.nn.functional as F


def _smooth_vol(D=24):
    a = np.linspace(-800, 400, D, dtype=np.float32)
    g = a[:, None, None] + a[None, :, None] * 0.3 + a[None, None, :] * 0.6
    return g.astype(np.float16)


def _native_crop_from(image_np, label_np, cls_idx, center, T, spacing):
    crop_ct, crop_lbl, out_sizes, pad_lo, geom = organ_crop_arrays(
        image_np, label_np, center, [1.5, 1.5, 1.5], image_size=(T, T, T),
        crop_mm=spacing, jitter=0, rng=random.Random(0))
    crop_sizes = geom[1].tolist()
    decim = tuple(max(1, int(cs) // max(1, int(o))) for cs, o in zip(crop_sizes, out_sizes))
    it = torch.from_numpy(np.ascontiguousarray(crop_ct))
    lt = torch.from_numpy(np.ascontiguousarray(crop_lbl))
    if any(d > 1 for d in decim):
        it = F.avg_pool3d(it.float()[None, None], decim, decim)[0, 0].half()
        lt = lt[:: decim[0], :: decim[1], :: decim[2]].contiguous()
    return NativeCrop(image=it, label=lt, class_idx=cls_idx, out_sizes=list(out_sizes),
                      pad_lo=list(pad_lo), crop_geom=geom, crop_spacing_mm=spacing, decim=decim)


def _reference(image_np, label_np, cls_idx, center, T, spacing, md, thr, spec):
    return crop_and_place(
        image_np, label_np, cls_idx, center, T, crop_spacing_mm=spacing,
        native_spacing=(1.5, 1.5, 1.5), jitter=0, rng=random.Random(0),
        mask_downsample=md, occ_thr=thr,
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
    assert _regroup(list(range(6)), 2, 3) == [[0, 1, 2], [3, 4, 5]]
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
