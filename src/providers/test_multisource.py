"""Unit tests for MultiSourceProvider (pure logic, fake sub-providers)."""
import random

import pytest
import torch

from src.incontext_dataset_v2 import LoadResult
from src.providers.multisource import MultiSourceProvider


class _FakeSub:
    """Minimal sub-provider: fixed class->subjects map, zero-tensor loads."""

    def __init__(self, modality, class_to_subjects):
        self.modality = modality
        self.classes = list(class_to_subjects)
        self._c2s = {c: list(s) for c, s in class_to_subjects.items()}
        self.loaded = []  # (subject, cls) log for assertions

    def subjects_for(self, cls):
        return self._c2s.get(cls, [])

    def load(self, subject, cls, req):
        self.loaded.append((subject, cls))
        return LoadResult(
            image=torch.zeros(1, 4, 4, 4),
            label=torch.zeros(4, 4, 4, dtype=torch.long),
            spacing=torch.full((3,), float(req.crop_spacing_mm)),
            crop_geom=torch.zeros(4, 3, dtype=torch.long),
            modality=self.modality,
        )


def _mk(regime_p=(1 / 3, 1 / 3, 1 / 3), context_size=1):
    ct = _FakeSub("ct", {"a": ["ca0", "ca1", "ca2"],
                          "b": ["cb0", "cb1", "cb2", "cb3"],
                          "c": ["cc0", "cc1", "cc2"]})
    mri = _FakeSub("mri", {"b": ["mb0", "mb1", "mb2"],
                            "c": ["mc0", "mc1", "mc2"],
                            "d": ["md0", "md1", "md2"]})
    prov = MultiSourceProvider({"ct": ct, "mri": mri},
                               context_size=context_size, regime_p=regime_p,
                               epoch_length=99)
    return prov, ct, mri


def test_class_union_and_availability():
    prov, _, _ = _mk()
    assert prov.classes == ["a", "b", "c", "d"]
    assert prov._avail["a"] == ["ct"]
    assert prov._avail["b"] == ["ct", "mri"]
    assert prov._avail["c"] == ["ct", "mri"]
    assert prov._avail["d"] == ["mri"]
    assert prov.epoch_length == 99


def test_item_dict_shape_k2():
    prov, _, _ = _mk(context_size=2)
    rng = random.Random(0)
    it = prov.assemble_task(rng, 3.0)
    assert set(it) >= {"image", "label", "context_in", "context_out", "spacing",
                       "crop_geom", "subject", "context_subjects", "label_name",
                       "modality", "aug_mode", "meta"}
    assert it["context_in"].shape == (2, 1, 4, 4, 4)
    assert it["context_out"].shape == (2, 4, 4, 4)
    assert len(it["context_subjects"]) == 2
    assert it["label_name"] in prov.classes
    assert it["meta"]["regime"] in ("ct", "mri", "cross")
    assert int(it["aug_mode"]) == 0


def test_regime_frequencies():
    prov, _, _ = _mk(regime_p=(0.5, 0.3, 0.2))
    rng = random.Random(0)
    tally = {"ct": 0, "mri": 0, "cross": 0}
    n = 6000
    for _ in range(n):
        tally[prov.assemble_task(rng, 3.0)["meta"]["regime"]] += 1
    assert abs(tally["ct"] / n - 0.5) < 0.03
    assert abs(tally["mri"] / n - 0.3) < 0.03
    assert abs(tally["cross"] / n - 0.2) < 0.03


def test_cross_is_cross_modality_when_both_available():
    prov, _, _ = _mk()
    rng = random.Random(1)
    seen = 0
    for _ in range(2000):
        it = prov.assemble_task(rng, 3.0)
        if it["meta"]["regime"] == "cross" and it["label_name"] in ("b", "c"):
            seen += 1
            assert it["meta"]["tgt_mod"] != it["meta"]["ctx_mod"]
            assert it["modality"] in ("ct", "mri")
    assert seen > 50  # sanity: the branch was actually exercised


def test_ct_only_class_never_produces_mri():
    prov, _, mri = _mk()
    rng = random.Random(2)
    for _ in range(500):
        it = prov.assemble_task(rng, 3.0)
        if it["label_name"] == "a":
            assert it["modality"] == "ct"
            assert it["meta"]["ctx_mod"] == "ct"
            assert all(s.startswith("ca") for s in it["context_subjects"])
            assert it["subject"].startswith("ca")


def test_pure_regime_falls_back_for_missing_modality():
    # regime forced to 'mri' (index 1) always; class 'a' has no MRI -> falls back to ct.
    prov, _, _ = _mk(regime_p=(0.0, 1.0, 0.0))
    rng = random.Random(3)
    for _ in range(200):
        it = prov.assemble_task(rng, 3.0)
        if it["label_name"] == "a":
            assert it["meta"]["regime"] == "mri"       # regime label is unchanged
            assert it["modality"] == "ct"              # resolved slot modality fell back
            assert it["meta"]["ctx_mod"] == "ct"


def test_determinism_same_seed_same_item():
    p1, _, _ = _mk()
    p2, _, _ = _mk()
    r1, r2 = random.Random(123), random.Random(123)
    for _ in range(50):
        a = p1.assemble_task(r1, 2.5)
        b = p2.assemble_task(r2, 2.5)
        assert a["subject"] == b["subject"]
        assert a["context_subjects"] == b["context_subjects"]
        assert a["meta"] == b["meta"]
        assert torch.equal(a["context_in"], b["context_in"])


def test_short_pool_warns_and_repeats():
    ct = _FakeSub("ct", {"x": ["only0"]})
    mri = _FakeSub("mri", {"x": ["mx0", "mx1"]})
    prov = MultiSourceProvider({"ct": ct, "mri": mri}, context_size=1,
                               regime_p=(1.0, 0.0, 0.0), epoch_length=10)
    rng = random.Random(0)
    with pytest.warns(UserWarning, match="repeating"):
        it = prov.assemble_task(rng, 3.0)
    assert it["subject"] == "only0"
    assert it["context_subjects"] == ["only0"]


def test_rejects_wrong_subprovider_count():
    ct = _FakeSub("ct", {"a": ["ca0"]})
    with pytest.raises(ValueError, match="exactly 2"):
        MultiSourceProvider({"ct": ct}, context_size=1, epoch_length=1)
