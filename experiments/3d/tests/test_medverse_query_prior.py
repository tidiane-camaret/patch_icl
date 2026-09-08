"""MedverseModel.train_forward(query_prior=...) routes the prior through Medverse's
NA-ICL image_context channel: image_context_in = the (normalized) target image,
image_context_out = the prior, both shaped (B, 1, 1, D, H, W). None -> both omitted
(prior-free, the eval / released regime). See experiments/3d/query_prior.py + docs/logs.md.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.benchmark_models.medverse import MedverseModel  # noqa: E402


class _Net:
    """Minimal stand-in for LightningModel: records the forward kwargs."""
    def normalize_3d_volume(self, x):
        return x                                   # identity keeps the assert simple
    def forward(self, target_in, *, context_in, context_out,
                image_context_in=None, image_context_out=None, l=1):
        self.seen = dict(target_in=target_in, context_in=context_in, context_out=context_out,
                         image_context_in=image_context_in, image_context_out=image_context_out, l=l)
        return torch.zeros_like(target_in)


def _model():
    m = MedverseModel.__new__(MedverseModel)       # skip __init__ (no repo / weights / GPU)
    m.device = torch.device("cpu")
    m.forward_l_arg = 1
    m.bounded_head = True
    m.model = _Net()
    return m


def test_prior_none_is_prior_free():
    m = _model()
    B, D = 2, 8
    m.train_forward(torch.rand(B, 1, D, D, D), torch.rand(B, 1, 1, D, D, D),
                    torch.zeros(B, 1, D, D, D))
    assert m.model.seen["image_context_in"] is None
    assert m.model.seen["image_context_out"] is None


def test_prior_rides_image_context_channel():
    m = _model()
    B, D = 2, 8
    tgt = torch.rand(B, 1, D, D, D)
    qp = torch.rand(B, 1, D, D, D)
    m.train_forward(tgt, torch.rand(B, 1, 1, D, D, D), torch.zeros(B, 1, D, D, D),
                    query_prior=qp)
    ic_in, ic_out = m.model.seen["image_context_in"], m.model.seen["image_context_out"]
    assert ic_in.shape == (B, 1, 1, D, D, D) and ic_out.shape == (B, 1, 1, D, D, D)
    assert torch.equal(ic_in[:, 0], tgt)                       # normalized target image (identity norm)
    assert torch.equal(ic_out[:, 0], qp.clamp(0, 1))          # the prior, clamped


def test_prior_is_clamped_to_unit_interval():
    m = _model()
    B, D = 1, 4
    qp = torch.full((B, 1, D, D, D), 1.7)                     # a sloppy upstream prior
    m.train_forward(torch.rand(B, 1, D, D, D), torch.rand(B, 1, 1, D, D, D),
                    torch.zeros(B, 1, D, D, D), query_prior=qp)
    assert m.model.seen["image_context_out"].max() <= 1.0
