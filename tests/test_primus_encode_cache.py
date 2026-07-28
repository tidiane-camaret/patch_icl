"""Tests for the frozen-PrimusEncoder eval encode cache (CPU-only, no Primus/GPU).

The cache exists so that a frozen encoder — whose output for a fixed eval crop is
invariant across epochs — encodes each distinct input volume once and reuses it
everywhere (within a batch, across the double val forward, and across epochs).
"""
import torch

from src.models.primus_encoder import _EncodeCache, _cached_encode


# --- _EncodeCache: an LRU, CPU-backed feature store --------------------------

def test_cache_miss_returns_none():
    c = _EncodeCache(max_entries=4)
    assert c.get("k") is None


def test_put_then_get_returns_equal_cpu_tensor():
    c = _EncodeCache(max_entries=4)
    t = torch.randn(3, 4)
    c.put("k", t)
    got = c.get("k")
    assert got is not None
    assert got.device.type == "cpu"        # stored off-GPU to spare VRAM
    assert torch.equal(got, t)


def test_lru_evicts_least_recently_used():
    c = _EncodeCache(max_entries=2)
    c.put("a", torch.zeros(1))
    c.put("b", torch.ones(1))
    c.get("a")                              # touch 'a' -> 'b' now the LRU
    c.put("c", torch.full((1,), 2.0))       # evict 'b'
    assert c.get("b") is None
    assert c.get("a") is not None
    assert c.get("c") is not None
    assert len(c) == 2


# --- _cached_encode: per-row miss-batching + assembly ------------------------

def _counting_encoder():
    """Returns (encode_fn, calls) where encode_fn(x)->x*10 and calls records
    the number of ROWS actually encoded (cache misses)."""
    calls = {"rows": 0, "batches": 0}
    def encode_fn(x):                        # (m,1) -> (m,1)
        calls["rows"] += x.shape[0]
        calls["batches"] += 1
        return x * 10.0
    return encode_fn, calls


def _key(row):
    return round(float(row.reshape(-1)[0]), 6)


def test_distinct_rows_encoded_once_each():
    c = _EncodeCache(max_entries=16)
    enc, calls = _counting_encoder()
    x = torch.tensor([[1.0], [2.0], [3.0]])
    out = _cached_encode(enc, x, _key, c)
    assert torch.equal(out, x * 10.0)
    assert calls["rows"] == 3


def test_repeated_rows_reuse_and_assemble_in_order():
    c = _EncodeCache(max_entries=16)
    enc, calls = _counting_encoder()
    x = torch.tensor([[1.0], [2.0], [1.0]])  # row 0 and 2 identical
    out = _cached_encode(enc, x, _key, c)
    assert torch.equal(out, torch.tensor([[10.0], [20.0], [10.0]]))
    assert calls["rows"] == 2                # only 1.0 and 2.0 encoded
    assert out[0].equal(out[2])


def test_second_call_reuses_cache_no_new_encodes():
    """Cross-epoch: the same inputs on a later call encode nothing new."""
    c = _EncodeCache(max_entries=16)
    enc, calls = _counting_encoder()
    x = torch.tensor([[1.0], [2.0]])
    _cached_encode(enc, x, _key, c)
    assert calls["rows"] == 2
    out2 = _cached_encode(enc, x, _key, c)   # later epoch, same crops
    assert calls["rows"] == 2                # unchanged -> zero re-encode
    assert torch.equal(out2, x * 10.0)


def test_output_on_input_device():
    c = _EncodeCache(max_entries=16)
    enc, _ = _counting_encoder()
    x = torch.tensor([[1.0], [2.0]])
    out = _cached_encode(enc, x, _key, c)
    assert out.device == x.device


# --- PrimusEncoder.forward routing (CPU, no real Primus) ---------------------

import torch.nn as nn
from src.models.primus_encoder import PrimusEncoder


class _FakePrimusEncoder(PrimusEncoder):
    """PrimusEncoder with the ViT stubbed out — exercises forward()'s cache gate
    without loading Primus or a GPU."""
    def __init__(self, frozen):
        nn.Module.__init__(self)               # gives .training (default True)
        self.primus = nn.Linear(1, 1)          # a parameter for device lookup
        self.frozen = frozen
        self.resolution = 4
        self._cache = _EncodeCache(64)
        self.rows = 0

    def _encode_batch(self, x):
        self.rows += x.shape[0]
        return x * 10.0                        # shape-preserving stub


def _vols():
    a = torch.full((1, 1, 4, 4, 4), 1.0)
    b = torch.full((1, 1, 4, 4, 4), 2.0)
    return torch.cat([a, b], dim=0)            # (2,1,4,4,4), distinct rows


def test_forward_frozen_eval_caches_across_calls():
    m = _FakePrimusEncoder(frozen=True).eval()
    x = _vols()
    m(x); m(x)                                 # second call = later epoch
    assert m.rows == 2                          # each volume encoded once, ever


def test_forward_train_mode_bypasses_cache():
    m = _FakePrimusEncoder(frozen=True).train()
    x = _vols()
    m(x); m(x)
    assert m.rows == 4                          # aug regime -> recompute every call


def test_forward_trainable_bypasses_cache():
    m = _FakePrimusEncoder(frozen=False).eval()
    x = _vols()
    m(x); m(x)
    assert m.rows == 4                          # grad needed -> no caching
