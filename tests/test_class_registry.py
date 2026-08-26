"""Tests for data.class_registry."""

import pytest
from data.class_registry import (
    normalize, normalize_lenient, get, to_maisi_idx, to_totalseg_idx,
    from_maisi_idx, from_totalseg_idx, maisi_to_totalseg_idx, totalseg_to_maisi_idx,
    CLASS_REGISTRY, all_with_maisi, all_with_totalseg,
)


class TestNormalize:
    """Test normalize() function."""

    def test_totalseg_passthrough(self):
        assert normalize("kidney_left") == "kidney_left"
        assert normalize("rib_left_6") == "rib_left_6"
        assert normalize("vertebrae_T6") == "vertebrae_T6"

    def test_maisi_lr_flip(self):
        assert normalize("left kidney") == "kidney_left"
        assert normalize("right kidney") == "kidney_right"
        assert normalize("left lung upper lobe") == "lung_upper_lobe_left"

    def test_maisi_rib_format(self):
        assert normalize("left rib 6") == "rib_left_6"
        assert normalize("right rib 9") == "rib_right_9"
        assert normalize("left rib 12") == "rib_left_12"

    def test_case_insensitive(self):
        assert normalize("LEFT KIDNEY") == "kidney_left"
        assert normalize("Left Rib 6") == "rib_left_6"

    def test_synonym(self):
        assert normalize("bladder") == "urinary_bladder"
        assert normalize("urinary_bladder") == "urinary_bladder"

    def test_unknown_raises(self):
        with pytest.raises(KeyError):
            normalize("nonexistent_organ")


class TestIndexLookups:
    """Test index conversion functions."""

    def test_to_maisi_idx(self):
        assert to_maisi_idx("kidney_left") == 14
        assert to_maisi_idx("left kidney") == 14  # MAISI format
        assert to_maisi_idx("liver") == 1
        assert to_maisi_idx("rib_left_6") == 68

    def test_to_totalseg_idx(self):
        assert to_totalseg_idx("kidney_left") == 42
        assert to_totalseg_idx("liver") == 44
        assert to_totalseg_idx("rib_left_6") == 59

    def test_from_maisi_idx(self):
        assert from_maisi_idx(14) == "kidney_left"
        assert from_maisi_idx(1) == "liver"
        assert from_maisi_idx(68) == "rib_left_6"

    def test_from_totalseg_idx(self):
        assert from_totalseg_idx(42) == "kidney_left"
        assert from_totalseg_idx(44) == "liver"
        assert from_totalseg_idx(59) == "rib_left_6"


class TestCrossVocabConversion:
    """Test cross-vocabulary index conversion."""

    def test_maisi_to_totalseg(self):
        assert maisi_to_totalseg_idx(14) == 42  # kidney_left
        assert maisi_to_totalseg_idx(1) == 44   # liver
        assert maisi_to_totalseg_idx(23) is None  # lung_tumor (MAISI-only)

    def test_totalseg_to_maisi(self):
        assert totalseg_to_maisi_idx(42) == 14  # kidney_left
        assert totalseg_to_maisi_idx(44) == 1   # liver
        assert totalseg_to_maisi_idx(118) is None  # lung_left (TS-only)


class TestRegistryCompleteness:
    """Verify registry matches original vocabularies."""

    def test_all_maisi_classes_present(self):
        from data.maisi_classes import MAISI_IDX_TO_CLASS
        for maisi_idx in MAISI_IDX_TO_CLASS:
            canon = from_maisi_idx(maisi_idx)
            assert canon in CLASS_REGISTRY

    def test_all_totalseg_classes_present(self):
        from data.totalseg_classes import ALL_CLASSES
        for i, ts_name in enumerate(ALL_CLASSES[:117], 1):
            canon = from_totalseg_idx(i)
            assert canon == ts_name

    def test_registry_stats(self):
        assert len(CLASS_REGISTRY) == 130
        assert len(all_with_maisi()) == 125
        assert len(all_with_totalseg()) == 122
