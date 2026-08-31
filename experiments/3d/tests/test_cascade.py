"""Tasks 4-5,7: cascade.py — invert_geo_center, run_cascade, _cascade_loss."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # experiments/3d siblings

import numpy as np
import pytest
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

import threading

from cascade import run_cascade, CascadeResult, _cascade_loss, _recrop_level
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


class _CentroidEchoModel(torch.nn.Module):
    """Returns the (augmented) target image straight through as its logit, so
    _centroid_from_logit recovers the AUGMENTED target blob centroid. This makes
    run_cascade's capture->invert seam a closed loop: plant a blob, let the real
    GpuAugmentor flip+warp it, and check the inverted native centre lands back on
    the planted voxel."""
    spacing_aware = False

    def __init__(self):
        super().__init__()
        self.p = torch.nn.Parameter(torch.zeros(1))

    def forward(self, image, context_in, context_out, mode="train", spacing=None):
        return {"final_logit": image + self.p.to(image.device)}   # (B,1,T,T,T)


def _echo_aug_cfg():
    return OmegaConf.create({
        "enabled": True, "gpu": True,
        "task": {
            "flip": {"p_d": 1.0, "p_h": 1.0, "p_w": 1.0},           # force every flip
            "affine": {"p": 1.0, "max_angle_deg": 20.0, "scale_min": 0.9,
                       "scale_max": 1.1, "max_translate": 0.1},
            "deform": {"p": 1.0, "control_points": 4, "max_disp": 0.1, "num_steps": 4},
            "elastic": {"p": 0.0, "alpha": 0.1, "grid_scale": 8},
            "mask_interp": "bilinear",
        },
        "intensity": {
            "brightness_contrast": {"p": 0.0, "brightness": 0.0,
                                    "contrast_range": [0.9, 1.1]},
        },
    })


@pytest.mark.parametrize("device", [
    "cpu",
    pytest.param("cuda", marks=pytest.mark.skipif(
        not torch.cuda.is_available(), reason="needs CUDA")),
])
def test_run_cascade_with_real_augmentor_recovers_planted_centroid(device):
    """C1 + capture/invert-seam regression guard: run_cascade with a REAL GpuAugmentor
    (all flips + affine + deform on). The device="cuda" case is the exact C1 repro —
    the captured grid lives on CUDA while invert_geo_center builds its query tensor."""
    from src.gpu_augment import GpuAugmentor

    B, K, T = 2, 3, 8
    dev = torch.device(device)
    aug = GpuAugmentor(_echo_aug_cfg())
    batch = _v2_batch(B=B, K=K, T=T)
    planted = (4, 4, 3)                                   # near-centre so the warp stays <2 vox
    d, h, w = planted
    for b in range(B):
        batch["image"][b, 0, d - 1:d + 2, h - 1:h + 2, w - 1:w + 2] = 1.0   # 3^3 blob
        batch["label"][b, d - 1:d + 2, h - 1:h + 2, w - 1:w + 2] = 1.0
    for k in ("image", "label", "context_in", "context_out"):
        batch[k] = batch[k].to(dev)

    model = _CentroidEchoModel().to(dev)
    res = run_cascade(model, _FakeProvider(T=T), batch, augmentor=aug,
                      spacings=[3.0, 1.5], device=dev, training=True,
                      step=0, seed=0, is_prob=True)

    assert isinstance(res, CascadeResult)
    assert len(res.centers) == 2 and len(res.centers[1]) == B
    for b in range(B):
        nc = res.centers[1][b]
        assert nc is not None, f"b={b}: COM inversion hit the empty fallback"
        err = max(abs(nc[a] - planted[a]) for a in range(3))
        assert err <= 2, f"b={b}: inverted native centre {nc} vs planted {planted} (err {err})"


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


class _PriorSpyModel(torch.nn.Module):
    """Records the query_prior tensor it receives per level; otherwise a fixed hot-cell logit."""
    spacing_aware = False

    def __init__(self, G=4, hot=(1, 1, 1)):
        super().__init__()
        self.G, self.hot, self.priors = G, hot, []
        self.p = torch.nn.Parameter(torch.zeros(1))

    def forward(self, image, context_in, context_out, mode="train", spacing=None,
                query_prior=None):
        self.priors.append(None if query_prior is None else query_prior.detach().clone())
        if query_prior is not None:
            assert float(query_prior.min()) >= 0.0 and float(query_prior.max()) <= 1.0
        B = image.shape[0]
        lg = torch.full((B, 1, self.G, self.G, self.G), -10.0, device=image.device)
        lg[:, :, self.hot[0], self.hot[1], self.hot[2]] = 10.0
        return {"final_logit": lg + self.p.to(image.device)}


class _LabelProvider(_FakeProvider):
    """_FakeProvider but returns a label with a planted 2x2x2 blob (identity crop_geom)."""
    def load(self, subject, cls, req):
        r = super().load(subject, cls, req)
        lbl = torch.zeros_like(r.label)
        lbl[1:3, 1:3, 1:3] = 1
        return LoadResult(image=r.image, label=lbl, spacing=r.spacing, crop_geom=r.crop_geom)


def _labeled_batch(B=2, K=3, T=8):
    b = _v2_batch(B=B, K=K, T=T)
    b["label"][:, 5:7, 5:7, 5:7] = 1                      # level-0 GT blob (distinct location)
    return b


@pytest.mark.parametrize("mode", ["pred", "gt_coarse", "gt_fine"])
def test_run_cascade_query_prior_modes(mode):
    B, T = 2, 8
    model = _PriorSpyModel(G=4)
    res = run_cascade(model, _LabelProvider(T=T), _labeled_batch(B=B, T=T), augmentor=None,
                      spacings=[3.0, 1.5], device=torch.device("cpu"),
                      training=True, step=0, seed=0, query_prior=mode)
    assert model.priors[0] is None                        # level 0 never gets a prior
    pr = model.priors[1]
    assert pr.shape == (B, 1, T, T, T) and 0.0 <= float(pr.min()) and float(pr.max()) <= 1.0
    assert all(torch.isfinite(l).all() for l in res.logits)
    if mode == "gt_fine":                                 # level-1 own GT, no warp
        want = _LabelProvider(T=T).load("s", "liver", LoadRequest(rng=None,
                                        crop_spacing_mm=1.5)).label.float()
        assert torch.equal(pr[0, 0], want)
    if mode == "gt_coarse":                               # level-0 GT, identity warp (bilinear)
        assert torch.allclose(pr[:, 0], _labeled_batch(B=B, T=T)["label"].float(), atol=1e-4)


def test_run_cascade_query_prior_hard_thresholds():
    B, T = 2, 8
    model = _PriorSpyModel(G=4)
    run_cascade(model, _LabelProvider(T=T), _labeled_batch(B=B, T=T), augmentor=None,
                spacings=[3.0, 1.5], device=torch.device("cpu"), training=True, step=0,
                seed=0, query_prior="pred", query_prior_hard=True)
    assert set(torch.unique(model.priors[1]).tolist()) <= {0.0, 1.0}


@pytest.mark.parametrize("qp", [False, None, "none"])
def test_run_cascade_query_prior_off(qp):
    B, T = 2, 8
    model = _PriorSpyModel(G=4)
    run_cascade(model, _FakeProvider(T=T), _v2_batch(B=B, T=T), augmentor=None,
                spacings=[3.0, 1.5], device=torch.device("cpu"),
                training=True, step=0, seed=0, query_prior=qp)
    assert model.priors == [None, None]


def test_run_cascade_query_prior_true_is_pred():
    B, T = 2, 8
    model = _PriorSpyModel(G=4)
    run_cascade(model, _LabelProvider(T=T), _labeled_batch(B=B, T=T), augmentor=None,
                spacings=[3.0, 1.5], device=torch.device("cpu"),
                training=True, step=0, seed=0, query_prior=True)
    assert model.priors[0] is None and model.priors[1].shape == (B, 1, T, T, T)


def test_run_cascade_query_prior_bad_mode_raises():
    with pytest.raises(ValueError):
        run_cascade(_PriorSpyModel(G=4), _FakeProvider(T=8), _v2_batch(B=2, T=8),
                    augmentor=None, spacings=[3.0, 1.5], device=torch.device("cpu"),
                    training=True, step=0, seed=0, query_prior="oracle")


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


class _KeyedProvider:
    """load() returns a crop whose fill value is a deterministic function of the subject
    string, so a mis-ordered threaded assembly is caught by a tensor comparison."""
    def __init__(self, T=8):
        self.T = T

    @staticmethod
    def _sig(subject):
        return float(int(__import__("hashlib").sha1(subject.encode()).hexdigest()[:6], 16) % 997)

    def load(self, subject, cls, req: LoadRequest):
        T = self.T
        v = self._sig(subject)
        geom = torch.tensor([[0, 0, 0], [T, T, T], [T, T, T], [0, 0, 0]], dtype=torch.long)
        return LoadResult(image=torch.full((1, T, T, T), v),
                          label=torch.full((T, T, T), v),
                          spacing=torch.full((3,), float(req.crop_spacing_mm)), crop_geom=geom)


class _BarrierProvider:
    """load() blocks on a shared barrier: it only returns once `barrier.parties` loads are
    in flight at the same time. The serial path can never reach that count -> timeout."""
    def __init__(self, T, barrier):
        self.T, self.barrier = T, barrier

    def load(self, subject, cls, req: LoadRequest):
        self.barrier.wait()
        T = self.T
        geom = torch.tensor([[0, 0, 0], [T, T, T], [T, T, T], [0, 0, 0]], dtype=torch.long)
        return LoadResult(image=torch.zeros(1, T, T, T), label=torch.zeros(T, T, T),
                          spacing=torch.full((3,), float(req.crop_spacing_mm)), crop_geom=geom)


def test_recrop_level_parallel_output_matches_serial():
    """recrop_workers>1 assembles a byte-identical batch to the serial path: per-b order,
    per-k context order and target/context roles are all preserved."""
    B, K, T = 3, 2, 8
    batch = _v2_batch(B=B, K=K, T=T)
    centers = [(4, 4, 4)] * B
    serial = _recrop_level(_KeyedProvider(T=T), batch, centers, 1.5,
                           step=2, seed=0, level=1, jitter=0, recrop_workers=1)
    parallel = _recrop_level(_KeyedProvider(T=T), batch, centers, 1.5,
                             step=2, seed=0, level=1, jitter=0, recrop_workers=8)
    for k in ("image", "label", "context_in", "context_out"):
        assert torch.equal(serial[k], parallel[k]), k
    assert serial["subjects"] == parallel["subjects"]
    assert serial["context_subjects"] == parallel["context_subjects"]


def test_recrop_level_runs_loads_concurrently():
    """recrop_workers fans the B*(K+1) provider loads out concurrently: a provider that
    blocks until every load is in flight completes only when the fan-out is parallel."""
    B, K, T = 2, 1, 8
    n_calls = B * (K + 1)
    prov = _BarrierProvider(T=T, barrier=threading.Barrier(n_calls, timeout=5))
    batch = _v2_batch(B=B, K=K, T=T)
    out = _recrop_level(prov, batch, [(4, 4, 4)] * B, 1.5,
                        step=0, seed=0, level=1, jitter=0, recrop_workers=n_calls)
    assert out["image"].shape[0] == B


def test_run_cascade_forwards_recrop_workers():
    """run_cascade threads its recrop_workers through to _recrop_level for every level>0."""
    B, K, T = 2, 1, 8
    n_calls = B * (K + 1)
    prov = _BarrierProvider(T=T, barrier=threading.Barrier(n_calls, timeout=5))
    batch = _v2_batch(B=B, K=K, T=T)
    res = run_cascade(_FakeModel(G=4, hot=(1, 1, 1)), prov, batch, augmentor=None,
                      spacings=[3.0, 1.5], device=torch.device("cpu"), training=True,
                      step=0, seed=0, jitter=0, recrop_workers=n_calls)
    assert len(res.logits) == 2


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


def test_train_epoch_reads_cascade_recrop_workers():
    """train_epoch's cascade branch forwards data.cascade_recrop_workers to run_cascade:
    a barrier provider that needs all B*(K+1) loads concurrent completes the epoch."""
    import types
    import train as train_mod

    B, K, T = 2, 1, 8
    n_calls = B * (K + 1)
    model = _FakeModel(G=4, hot=(1, 1, 1))
    prov = _BarrierProvider(T=T, barrier=threading.Barrier(n_calls, timeout=5))

    class _Loader:
        dataset = types.SimpleNamespace(provider=prov)
        def __iter__(self): return iter([_v2_batch(B=B, K=K, T=T)])
        def __len__(self): return 1

    class _Sched:
        def step(self, *a): pass

    cfg = OmegaConf.create({
        "model": "patchset3d",
        "data": {"cascade_spacings": [3.0, 1.5], "cascade_crop_jitter": 0,
                 "crop_spacing_mm": 3.0, "cascade_recrop_workers": n_calls},
        "train": {"seed": 0, "cascade_loss_weights": [1.0, 1.0]},
    })
    loss_fn = lambda logit, target: torch.nn.functional.mse_loss(
        torch.sigmoid(logit.float()), target.float())

    mean_loss, *_ = train_mod.train_epoch(
        model, _Loader(), [torch.optim.SGD(model.parameters(), lr=0.0)], _Sched(),
        step_per_batch=True, loss_fn=loss_fn, cfg=cfg, epoch=0, is_patchset=True, gpu_aug=None)
    assert np.isfinite(mean_loss)


# ---------------------------------------------------------------------------
# Task 8: evaluate_cascade
# ---------------------------------------------------------------------------
from cascade import evaluate_cascade


def _named_batch(names, B=2, T=8, K=3):
    b = _v2_batch(B=B, K=K, T=T)
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


def test_run_cascade_figure_arrays_opt_in():
    """want_figure_arrays=True -> figure_levels is a per-level list of {img,gt,ctx_img,ctx_gt}
    (B,T,T,T) post-aug arrays; default keeps it None."""
    B, T = 2, 8
    model, prov = _FakeModel(G=4, hot=(1, 1, 1)), _FakeProvider(T=T)
    off = run_cascade(model, prov, _v2_batch(B=B, T=T), augmentor=None,
                      spacings=[3.0, 1.5], device="cpu", training=False, step=0, seed=0)
    assert off.figure_levels is None

    res = run_cascade(model, prov, _v2_batch(B=B, T=T), augmentor=None,
                      spacings=[3.0, 1.5], device="cpu", training=False, step=0, seed=0,
                      want_figure_arrays=True)
    assert isinstance(res.figure_levels, list) and len(res.figure_levels) == 2
    for fl in res.figure_levels:
        assert set(fl) == {"img", "gt", "ctx_img", "ctx_gt"}
        for v in fl.values():
            assert v.shape == (B, T, T, T)


def test_evaluate_cascade_saves_cascade_figures(tmp_path, monkeypatch):
    """cascade_figures=True + fig_dir -> one <cls>_<s0>to<s1>mm.png per requested class under
    fig_dir/cascade/ (reusing evaluate.save_cascade_figure)."""
    pytest.importorskip("matplotlib")
    import common
    from src.totalseg_dataloader_incontext import _ALL_CLASSES_IDX

    B, T = 2, 8
    lbl = np.zeros((T, T, T), dtype=np.uint8)
    lbl[1:4, 1:4, 1:4] = _ALL_CLASSES_IDX["liver"]
    for b in range(B):
        (tmp_path / f"s{b}").mkdir()
        np.save(tmp_path / f"s{b}" / "label.npy", lbl)
    monkeypatch.setattr(common, "_source_root",
                        lambda cfg: (None, str(tmp_path), False), raising=False)

    class _Loader:
        dataset = __import__("types").SimpleNamespace(provider=_FakeProvider(T=T))
        def __iter__(self): return iter([_named_batch(["liver"] * B, B=B, T=T)])
        def __len__(self): return 1

    cfg = OmegaConf.create({"data": {"cascade_spacings": [6.0, 3.0]}})
    fig_dir = tmp_path / "figures"
    evaluate_cascade(_FakeModel(G=4, hot=(1, 1, 1)), cfg, ["liver"], loader=_Loader(),
                     seed=0, is_prob=False, fig_dir=fig_dir, cascade_figures=True)
    saved = list((fig_dir / "cascade").glob("*.png"))
    assert [p.name for p in saved] == ["liver_6to3mm.png"]
    assert saved[0].stat().st_size > 0


def test_evaluate_cascade_reads_cascade_recrop_workers(tmp_path, monkeypatch):
    """evaluate_cascade forwards data.cascade_recrop_workers to run_cascade."""
    import common
    from src.totalseg_dataloader_incontext import _ALL_CLASSES_IDX

    B, K, T = 2, 1, 8
    n_calls = B * (K + 1)
    lbl = np.zeros((T, T, T), dtype=np.uint8)
    lbl[1:4, 1:4, 1:4] = _ALL_CLASSES_IDX["liver"]
    for b in range(B):
        (tmp_path / f"s{b}").mkdir()
        np.save(tmp_path / f"s{b}" / "label.npy", lbl)
    monkeypatch.setattr(common, "_source_root",
                        lambda cfg: (None, str(tmp_path), False), raising=False)

    prov = _BarrierProvider(T=T, barrier=threading.Barrier(n_calls, timeout=5))

    class _Loader:
        dataset = __import__("types").SimpleNamespace(provider=prov)
        def __iter__(self): return iter([_named_batch(["liver"] * B, B=B, T=T, K=K)])
        def __len__(self): return 1

    cfg = OmegaConf.create({"data": {"cascade_spacings": [3.0, 1.5],
                                     "cascade_recrop_workers": n_calls}})
    rows, cases = evaluate_cascade(_FakeModel(G=4, hot=(1, 1, 1)), cfg, ["liver"],
                                   loader=_Loader(), seed=0, is_prob=False)
    assert len(cases) == B
