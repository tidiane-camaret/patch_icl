"""draw_cascade_ladder + CascadeSpacingBatchSampler (data.cascade_train random ladder)."""
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from torch.utils.data import SequentialSampler

from common import CascadeSpacingBatchSampler, draw_cascade_ladder, resolve_cascade_train


# --- resolve_cascade_train ----------------------------------------------------

def test_resolve_none_when_absent():
    assert resolve_cascade_train({}) is None
    assert resolve_cascade_train({"cascade_train": None}) is None


def test_resolve_ok():
    assert resolve_cascade_train(
        {"cascade_train": {"levels": 2, "spacing_range": [1.5, 6]}}) == (2, 1.5, 6.0)


def test_resolve_rejects():
    with pytest.raises(ValueError):
        resolve_cascade_train({"cascade_train": {"levels": 2, "spacing_range": [6, 1.5]}})
    with pytest.raises(ValueError):
        resolve_cascade_train({"cascade_train": {"levels": 1, "spacing_range": [1.5, 6]}})
    with pytest.raises(ValueError):
        resolve_cascade_train({"cascade_train": {"levels": 2}})


# --- draw_cascade_ladder ----------------------------------------------------

@pytest.mark.parametrize("n", [2, 3, 4])
def test_ladder_descending_and_anchored(n):
    rng = random.Random(0)
    for _ in range(200):
        s0 = math.exp(rng.uniform(math.log(3.0), math.log(6.0)))
        lad = draw_cascade_ladder(s0, n, 1.5, rng)
        assert len(lad) == n
        assert lad[0] == s0
        assert all(a > b for a, b in zip(lad, lad[1:])), lad          # strictly finer
        assert all(a / b >= 1.15 - 1e-9 for a, b in zip(lad, lad[1:]))  # min-ratio gap
        assert lad[-1] > 0


def test_ladder_reproducible():
    a = draw_cascade_ladder(5.0, 3, 1.5, random.Random("x_1_2_cascade_sp"))
    b = draw_cascade_ladder(5.0, 3, 1.5, random.Random("x_1_2_cascade_sp"))
    assert a == b


def test_ladder_single_level_is_noop():
    assert draw_cascade_ladder(4.0, 1, 1.5, random.Random(0)) == [4.0]


# --- CascadeSpacingBatchSampler -------------------------------------------------

def test_sampler_batches_and_coarse_band():
    base = SequentialSampler(range(20))
    bs = CascadeSpacingBatchSampler(base, batch_size=4, spacing_range=[1.5, 6.0],
                                    n_levels=2, seed=0)
    assert len(bs) == 5
    batches = list(bs)
    assert len(batches) == 5
    geo_mid = math.sqrt(1.5 * 6.0)                       # (n-1)/n = 1/2 up the log-range
    assert bs._s0_lo == pytest.approx(geo_mid)
    for b in batches:
        assert len(b) == 4
        s0s = {round(s, 9) for _, s in b}
        assert len(s0s) == 1                             # one spacing per batch
        s0 = next(iter(s0s))
        assert geo_mid - 1e-6 <= s0 <= 6.0 + 1e-6        # drawn in the coarse band
        # every finer level still fits below s0
        assert draw_cascade_ladder(s0, 2, 1.5, random.Random(0))[-1] < s0


def test_sampler_drop_last():
    base = SequentialSampler(range(22))
    bs = CascadeSpacingBatchSampler(base, batch_size=4, spacing_range=[1.5, 6.0],
                                    n_levels=2, drop_last=True, seed=1)
    assert len(bs) == 5
    assert [len(b) for b in bs] == [4, 4, 4, 4, 4]


def test_sampler_s0_lo_rises_with_levels():
    base = SequentialSampler(range(4))
    lo2 = CascadeSpacingBatchSampler(base, 4, [1.5, 6.0], n_levels=2)._s0_lo
    lo4 = CascadeSpacingBatchSampler(base, 4, [1.5, 6.0], n_levels=4)._s0_lo
    assert lo4 > lo2                                     # more finer levels -> coarser s0 floor
