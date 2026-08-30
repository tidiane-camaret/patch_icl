"""Tasks 4-5,7: cascade.py — invert_geo_center, run_cascade, _cascade_loss."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # experiments/3d siblings

import numpy as np
import torch
from omegaconf import OmegaConf

from cascade import invert_geo_center
from evaluate import _predicted_native_center, _grid_centroid


def _geom(starts=(10, 20, 30), crop=(8, 8, 8), out=(8, 8, 8), pad=(0, 0, 0)):
    return torch.tensor([list(starts), list(crop), list(out), list(pad)], dtype=torch.long)


def _prob_blob(T=8, c=(4, 4, 4)):
    p = np.zeros((T, T, T), dtype=np.float32)
    p[c[0], c[1], c[2]] = 1.0
    return p


def test_identity_matches_predicted_native_center():
    T = 8
    prob = _prob_blob(T, c=(5, 3, 6))
    geom = _geom()
    cen = _grid_centroid(prob)                       # np array (d,h,w)
    got = invert_geo_center(cen, None, torch.zeros(3, dtype=torch.bool), geom, T)
    want = _predicted_native_center(torch.from_numpy(prob), geom)
    assert got == want


def test_empty_centroid_returns_none():
    assert invert_geo_center(None, None, torch.zeros(3, dtype=torch.bool), _geom(), 8) is None


def test_flip_mirrors_the_centroid():
    T = 8
    geom = _geom(starts=(0, 0, 0), crop=(8, 8, 8), out=(8, 8, 8), pad=(0, 0, 0))
    cen = np.array([2.0, 1.0, 7.0])                  # d,h,w
    flips = torch.tensor([True, False, True])        # flip d and w
    got = invert_geo_center(cen, None, flips, geom, T)
    # unflip: d -> (T-1)-2 = 5, w -> (T-1)-7 = 0 ; native == pre-aug grid here (identity geom)
    assert got == (5, 1, 0)


def test_grid_shift_maps_through():
    # A constant grid that maps every output voxel to the volume centre in normalized coords
    # (0,0,0) -> pre-aug voxel ((T-1)/2). Identity geom -> native == (T-1)/2 per axis.
    T = 8
    grid_row = torch.zeros(T, T, T, 3)               # all (x,y,z) = 0 -> centre
    geom = _geom(starts=(0, 0, 0), crop=(8, 8, 8), out=(8, 8, 8), pad=(0, 0, 0))
    got = invert_geo_center(np.array([1.0, 2.0, 3.0]), grid_row,
                            torch.zeros(3, dtype=torch.bool), geom, T)
    mid = round((T - 1) / 2)
    assert got == (mid, mid, mid)


def test_directional_grid_maps_x_channel_to_w():
    """Verify that grid's x-channel (normalized) maps to the w output axis, not d.

    align_corners=False denormalization: voxel = ((norm + 1) * T - 1) / 2.
    With grid[..., 0] = 0.5 (x-channel), centroid (3,3,3), identity geom, T=8:
    - w (x-channel 0.5): ((0.5+1)*8-1)/2 = 5.5 → rounds to 6
    - d,h (z,y-channels 0): ((0+1)*8-1)/2 = 3.5 → round to 4
    Assertion: w > d and w > h (x pulls w toward +x, not d).
    """
    T = 8
    grid_row = torch.zeros(T, T, T, 3)
    grid_row[..., 0] = 0.5  # x-channel = +0.5, y and z = 0
    geom = _geom(starts=(0, 0, 0), crop=(8, 8, 8), out=(8, 8, 8), pad=(0, 0, 0))
    centroid = np.array([3.0, 3.0, 3.0])

    d, h, w = invert_geo_center(centroid, grid_row, torch.zeros(3, dtype=torch.bool), geom, T)

    # With align_corners=False: ((norm + 1) * T - 1) / 2
    expected_w = round(((0.5 + 1.0) * T - 1.0) / 2.0)
    expected_d = round(((0.0 + 1.0) * T - 1.0) / 2.0)
    expected_h = round(((0.0 + 1.0) * T - 1.0) / 2.0)

    assert w == expected_w, f"w={w}, expected {expected_w}"
    assert d == expected_d, f"d={d}, expected {expected_d}"
    assert h == expected_h, f"h={h}, expected {expected_h}"
    # Key assertion: w should differ from d and h in the expected direction
    assert w > d, f"w ({w}) should be greater than d ({d}) due to +0.5 grid offset"
    assert w > h, f"w ({w}) should be greater than h ({h}) due to +0.5 grid offset"


def test_flip_then_grid_inversion_order():
    """Discriminates correct order (grid-lookup-then-unflip) from wrong (unflip-then-lookup).

    grid[..., 2] = 0.5 displaces along d (z-channel). With flip on d:
    Correct: interp grid at g_aug (2,3,3) -> d in flipped vol = ((0.5+1)*8-1)/2 = 5.5 ->
    unflip d -> (8-1)-5.5 = 1.5 -> round 2 -> native (2,4,4).
    Wrong order would give round(5.5)=6 (no 2nd unflip).
    """
    T = 8
    grid_row = torch.zeros(T, T, T, 3)
    grid_row[..., 2] = 0.5                                   # z-channel -> d output axis
    geom = _geom(starts=(0, 0, 0), crop=(8, 8, 8), out=(8, 8, 8), pad=(0, 0, 0))
    flips = torch.tensor([True, False, False])               # flip d only
    got = invert_geo_center(np.array([2.0, 3.0, 3.0]), grid_row, flips, geom, T)
    assert got == (2, 4, 4)


# ---------------------------------------------------------------------------
# Task 5: run_cascade + _cascade_loss
# ---------------------------------------------------------------------------
from dataclasses import dataclass as _dc

from cascade import run_cascade, CascadeResult, _cascade_loss
from src.incontext_dataset_v2 import LoadResult, LoadRequest


class _FakeModel(torch.nn.Module):
    """Returns a fixed low-res logit with all mass at one grid cell; records call spacings."""
    spacing_aware = False

    def __init__(self, G=4, hot=(1, 1, 1)):
        super().__init__()
        self.G, self.hot, self.seen_spacing = G, hot, []
        self.p = torch.nn.Parameter(torch.zeros(1))

    def forward(self, image, context_in, context_out, mode="train", spacing=None):
        self.seen_spacing.append(spacing)
        B = image.shape[0]
        lg = torch.full((B, 1, self.G, self.G, self.G), -10.0, device=image.device)
        lg[:, :, self.hot[0], self.hot[1], self.hot[2]] = 10.0
        lg = lg + self.p.to(image.device)                 # keep autograd alive
        return {"final_logit": lg}


class _FakeProvider:
    """Records every load() call; returns synthetic T^3 crops."""
    def __init__(self, T=8):
        self.T, self.calls = T, []

    def load(self, subject, cls, req: LoadRequest):
        self.calls.append({"subject": subject, "cls": cls, "center": req.center,
                           "spacing": req.crop_spacing_mm, "jitter": req.jitter})
        T = self.T
        geom = torch.tensor([[0, 0, 0], [T, T, T], [T, T, T], [0, 0, 0]], dtype=torch.long)
        return LoadResult(image=torch.zeros(1, T, T, T), label=torch.zeros(T, T, T),
                          spacing=torch.full((3,), float(req.crop_spacing_mm)), crop_geom=geom)


def _v2_batch(B=2, K=3, T=8):
    geom = torch.tensor([[0, 0, 0], [T, T, T], [T, T, T], [0, 0, 0]], dtype=torch.long)
    return {
        "image": torch.zeros(B, 1, T, T, T),
        "label": torch.zeros(B, T, T, T),
        "context_in": torch.zeros(B, K, 1, T, T, T),
        "context_out": torch.zeros(B, K, T, T, T),
        "subjects": [f"s{b}" for b in range(B)],
        "context_subjects": [[f"c{b}_{k}" for k in range(K)] for b in range(B)],
        "label_names": ["liver"] * B,
        "crop_geom": geom.unsqueeze(0).repeat(B, 1, 1),
        "aug_mode": torch.zeros(B, dtype=torch.long),
    }


def test_run_cascade_two_levels_no_aug():
    B, T, G = 2, 8, 4
    model, prov = _FakeModel(G=G, hot=(1, 1, 1)), _FakeProvider(T=T)
    res = run_cascade(model, prov, _v2_batch(B=B, T=T), augmentor=None,
                      spacings=[3.0, 1.5], device=torch.device("cpu"),
                      training=True, step=0, seed=0, jitter=0)
    assert isinstance(res, CascadeResult)
    assert len(res.logits) == 2 and len(res.targets) == 2
    assert res.centers[0] == [None, None]
    # level-1 target loads: center == inverted COM (identity geom + hot cell 1/G -> native),
    # contexts loaded with center=None, spacing == 1.5, jitter == 0
    tgt_calls = [c for c in prov.calls if c["center"] is not None]
    ctx_calls = [c for c in prov.calls if c["center"] is None]
    assert len(tgt_calls) == B and all(c["spacing"] == 1.5 for c in tgt_calls)
    assert all(c["jitter"] == 0 for c in prov.calls)
    assert len(ctx_calls) == B * 3
    assert model.seen_spacing == [None] * (2)  # spacing_aware False -> None both levels
    # loss aggregation helper
    lf = lambda logit, target: (logit.float().mean() - target.float().mean()) ** 2
    total, per = _cascade_loss(res, lf, [1.0, 2.0])
    assert total.requires_grad and len(per) == 2


def test_run_cascade_empty_prob_falls_back_to_gt_centroid():
    B, T = 2, 8
    model = _FakeModel(G=4)
    # force an all-background logit -> empty prob -> center None
    model.forward = lambda image, context_in, context_out, mode="train", spacing=None: {
        "final_logit": torch.full((image.shape[0], 1, 4, 4, 4), -30.0) + model.p}
    prov = _FakeProvider(T=T)
    res = run_cascade(model, prov, _v2_batch(B=B, T=T), augmentor=None,
                      spacings=[3.0, 1.5], device=torch.device("cpu"),
                      training=True, step=1, seed=0)
    assert res.centers[1] == [None, None]
    assert res.empty_frac == 1.0
    assert all(c["center"] is None for c in prov.calls)  # every level-1 load GT-centred


def test_train_epoch_cascade_smoke(monkeypatch):
    """train_epoch runs the cascade branch end-to-end on fakes: 2 optimiser steps, finite loss,
    per-level metric keys present."""
    import types
    import train as train_mod

    B, T, G = 2, 8, 4
    model = _FakeModel(G=G, hot=(1, 1, 1))
    prov = _FakeProvider(T=T)

    class _Loader:
        dataset = types.SimpleNamespace(provider=prov)
        def __iter__(self): return iter([_v2_batch(B=B, T=T), _v2_batch(B=B, T=T)])
        def __len__(self): return 2

    opt = torch.optim.SGD(model.parameters(), lr=0.0)

    class _Sched:
        def step(self, *a): pass

    cfg = OmegaConf.create({
        "model": "patchset3d",
        "data": {"cascade_spacings": [3.0, 1.5], "cascade_crop_jitter": 0,
                 "crop_spacing_mm": 3.0},
        "train": {"seed": 0, "cascade_loss_weights": [1.0, 1.0]},
    })
    loss_fn = lambda logit, target: torch.nn.functional.mse_loss(
        torch.sigmoid(logit.float()), target.float())

    mean_loss, mean_dice, mean_soft, grid = train_mod.train_epoch(
        model, _Loader(), [opt], _Sched(), step_per_batch=True, loss_fn=loss_fn,
        cfg=cfg, epoch=0, is_patchset=True, gpu_aug=None)

    assert np.isfinite(mean_loss)
    assert "loss_r3" in grid and "loss_r1.5" in grid
    assert "dice_r3" in grid and "dice_r1.5" in grid
    assert "cascade_empty_frac" in grid


# ---------------------------------------------------------------------------
# Task 8: evaluate_cascade
# ---------------------------------------------------------------------------
from cascade import evaluate_cascade


def _named_batch(names, B=2, T=8):
    b = _v2_batch(B=B, T=T)
    b["label_names"] = list(names)
    return b


def test_evaluate_cascade_shapes_and_nan_safe_macro(tmp_path, monkeypatch):
    """evaluate_cascade returns (rows, cases) shaped like evaluate_classes; every case carries
    dice + per-spacing dice_r cols; a class absent from _ALL_CLASSES_IDX (all-NaN dice) yields
    an `error` row, not a NaN mean_dice that would poison the checkpoint macro."""
    import common
    from src.totalseg_dataloader_incontext import _ALL_CLASSES_IDX

    B, T = 2, 8
    idx = _ALL_CLASSES_IDX["liver"]
    lbl = np.zeros((T, T, T), dtype=np.uint8)
    lbl[1:4, 1:4, 1:4] = idx
    for b in range(B):
        (tmp_path / f"s{b}").mkdir()
        np.save(tmp_path / f"s{b}" / "label.npy", lbl)

    monkeypatch.setattr(common, "_source_root",
                        lambda cfg: (None, str(tmp_path), False), raising=False)

    model, prov = _FakeModel(G=4, hot=(1, 1, 1)), _FakeProvider(T=T)

    class _Loader:
        dataset = __import__("types").SimpleNamespace(provider=prov)
        def __iter__(self):
            return iter([_named_batch(["liver"] * B, B=B, T=T),
                         _named_batch(["not_a_real_class"] * B, B=B, T=T)])
        def __len__(self): return 2

    cfg = OmegaConf.create({"data": {"cascade_spacings": [3.0, 1.5]}})
    rows, cases = evaluate_cascade(model, cfg, ["liver", "not_a_real_class"],
                                   loader=_Loader(), seed=0, is_prob=False)

    assert len(cases) == 2 * B
    for c in cases:
        assert set(("class", "subject", "dice", "dice_r3", "dice_r1.5")) <= set(c)

    by_cls = {r["class"]: r for r in rows}
    assert "mean_dice" in by_cls["liver"] and np.isfinite(by_cls["liver"]["mean_dice"])
    assert "mean_dice" not in by_cls["not_a_real_class"]
    assert by_cls["not_a_real_class"]["error"] == "no valid samples"
