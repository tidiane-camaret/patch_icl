"""train.medverse_bounded_head: the adapter appends a sigmoid to Medverse's head, so
train_forward returns a real logit and the loss takes the standard with_logits path
(no [0,1] clamp, no out-of-bounds anchor). See docs/logs.md 2026-09-08.

Covers the three seams the flag threads through:
  1. model_output_is_prob(cfg) flips to False when the flag is set.
  2. build_loss(cfg) then returns bce_with_logits + soft_dice — finite and O(magnitude)
     for large logits, vs the is_prob path whose OOB anchor explodes on out-of-[0,1] output.
  3. MedverseModel._bounded_forward wraps LightningModel.forward with a sigmoid and restores.
"""
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train import build_loss, model_output_is_prob  # noqa: E402
from src.benchmark_models.medverse import MedverseModel  # noqa: E402


def _cfg(model="medverse", bounded=False, loss="bce_dice"):
    return OmegaConf.create(
        {"model": model, "train": {"loss": loss, "dice_weight": 1.0,
                                   "oob_weight": 10.0, "medverse_bounded_head": bounded}})


def test_output_is_prob_flips_with_flag():
    assert model_output_is_prob(_cfg(bounded=False)) is True
    assert model_output_is_prob(_cfg(bounded=True)) is False
    assert model_output_is_prob(_cfg(model="patchset3d")) is False


def test_bounded_loss_is_finite_and_unanchored():
    tgt = torch.zeros(1, 1, 8, 8, 8)
    out_of_range = torch.full((1, 1, 8, 8, 8), 15.0)   # a logit of +15 == a raw output of 15

    bounded = build_loss(_cfg(bounded=True))(out_of_range, tgt)      # bce_with_logits path
    unbounded = build_loss(_cfg(bounded=False))(out_of_range, tgt)   # is_prob + OOB anchor

    assert torch.isfinite(bounded)
    # with_logits(15, 0) ~= 15, + soft_dice ~= 1  ->  O(magnitude), NOT O(magnitude^2)
    assert bounded < 20.0
    # the is_prob path's oob_w * mean((15-1)**2) = 10 * 196 dominates -> ~2 orders larger
    assert unbounded > 100 * bounded

    # in-range output: both paths agree closely (no anchor, no clamp effect)
    good = torch.full((1, 1, 8, 8, 8), 0.02)
    assert abs(build_loss(_cfg(bounded=True))(torch.logit(good), tgt).item()
               - build_loss(_cfg(bounded=False))(good, tgt).item()) < 0.15


def test_bounded_forward_wraps_and_restores():
    m = MedverseModel.__new__(MedverseModel)          # skip __init__ (no repo / weights)
    m.bounded_head = True

    class _Stub:
        def forward(self, x):
            return x
    m.model = _Stub()
    raw = _Stub.forward

    x = torch.tensor([-2.0, 0.0, 3.0])
    assert torch.equal(m.model.forward(x), x)         # unwrapped
    with m._bounded_forward():
        assert torch.allclose(m.model.forward(x), torch.sigmoid(x))
    assert torch.equal(m.model.forward(x), x)         # restored
    assert m.model.forward.__func__ is raw

    m.bounded_head = False                            # flag off -> context is a no-op
    with m._bounded_forward():
        assert torch.equal(m.model.forward(x), x)
