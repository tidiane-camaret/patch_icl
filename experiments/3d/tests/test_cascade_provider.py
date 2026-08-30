"""Task 1: LoadRequest.jitter field + provider jitter resolution."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import random

from src.incontext_dataset_v2 import LoadRequest
from src.providers.totalseg import _resolve_jitter


def _req(jitter=None):
    return LoadRequest(rng=random.Random(0), crop_spacing_mm=1.5, jitter=jitter)


def test_loadrequest_jitter_defaults_none():
    assert _req().jitter is None


def test_loadrequest_jitter_set():
    assert _req(jitter=0).jitter == 0
    assert _req(jitter=7).jitter == 7


def test_resolve_jitter_prefers_request():
    assert _resolve_jitter(_req(jitter=0), default=12) == 0
    assert _resolve_jitter(_req(jitter=3), default=12) == 3


def test_resolve_jitter_falls_back_to_default():
    assert _resolve_jitter(_req(jitter=None), default=12) == 12
