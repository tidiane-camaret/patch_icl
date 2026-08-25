# tests/test_incontext_v2_cohort.py
"""Engine cohort-hook path (bank-independent): a provider exposing `assemble_task`
bypasses the independent target+context sampling. Covers mode detection, length,
faithful delegation, aug gating, and gpu_realize passthrough."""
import types
import random

import torch

from src.incontext_dataset_v2 import InContextDataset

_T, _K = 8, 3


class _FakeCohort:
    """CPU-paint cohort provider: shade tied to the engine RNG, shared target+contexts."""
    epoch_length = 7
    classes = ["liver", "spleen"]

    def subjects_for(self, cls):
        return []

    def assemble_task(self, rng, crop_spacing_mm):
        shade = rng.getrandbits(8) / 255.0
        return {
            "image": torch.full((1, _T, _T, _T), shade),
            "label": torch.zeros(_T, _T, _T, dtype=torch.long),
            "context_in": torch.full((_K, 1, _T, _T, _T), shade),
            "context_out": torch.zeros(_K, _T, _T, _T, dtype=torch.long),
            "spacing": torch.full((3,), float(crop_spacing_mm)),
            "subject": "m0", "label_name": "liver",
            "context_subjects": ["c"] * _K,
            "aug_mode": torch.tensor(0, dtype=torch.long),
        }


class _FakeGpuRealize(_FakeCohort):
    """gpu_realize cohort provider: native payload, no 'image' to augment."""
    def assemble_task(self, rng, crop_spacing_mm):
        return {"native_lbls": [torch.zeros(4, 4, 4, dtype=torch.uint8)] * (_K + 1),
                "cls": 5, "gmm_mu": torch.zeros(201)}


def test_cohort_mode_len_and_schema():
    ds = InContextDataset(_FakeCohort(), context_size=_K, aug_cfg=None, crop_spacing_mm=1.5)
    assert ds.cohort_mode and len(ds) == 7 and ds.samples == []
    it = ds[0]
    assert it["image"].shape == (1, _T, _T, _T)
    assert it["context_in"].shape == (_K, 1, _T, _T, _T)
    assert it["context_out"].shape == (_K, _T, _T, _T)
    # shared-appearance coupling survives the engine (target shade == every context shade)
    assert torch.allclose(it["image"][0, 0, 0, 0], it["context_in"][0, 0, 0, 0, 0])


def test_cohort_delegates_faithfully_with_eval_seed():
    prov = _FakeCohort()
    ds = InContextDataset(prov, context_size=_K, aug_cfg=None, crop_spacing_mm=1.5, eval_seed=0)
    # engine reconstructs rng = Random(hash((eval_seed, idx))); item must match a direct call
    ref = prov.assemble_task(random.Random(hash((0, 0))), 1.5)
    assert torch.equal(ds[0]["image"], ref["image"])
    assert torch.equal(ds[0]["image"], ds[0]["image"])            # deterministic


def test_cohort_spacing_tuple_reaches_hook():
    ds = InContextDataset(_FakeCohort(), context_size=_K, aug_cfg=None, crop_spacing_mm=1.5)
    assert torch.allclose(ds[(2, 2.5)]["spacing"], torch.full((3,), 2.5))


def test_gpu_realize_payload_passes_through_aug():
    # aug enabled, but a native payload has no 'image' -> must pass through untouched
    ds = InContextDataset(_FakeGpuRealize(), context_size=_K,
                          aug_cfg=types.SimpleNamespace(enabled=True))
    it = ds[0]
    assert "image" not in it and len(it["native_lbls"]) == _K + 1 and it["cls"] == 5
