from data.maisi_classes import MAISI_CLASS_TO_IDX, MAISI_IDX_TO_CLASS, SHAPE_ID_TO_FAMILY


def test_shape_pseudo_classes_are_registered_and_round_trip():
    expected = {195: "shape_blob", 196: "shape_splatter", 197: "shape_disk",
               198: "shape_cylinder"}
    for idx, name in expected.items():
        assert MAISI_IDX_TO_CLASS[idx] == name
        assert MAISI_CLASS_TO_IDX[name] == idx


def test_shape_id_to_family_matches_the_bare_family_names():
    assert SHAPE_ID_TO_FAMILY == {195: "blob", 196: "splatter", 197: "disk",
                                  198: "cylinder"}


def test_shape_ids_stay_within_the_default_maxid_bound():
    """mu/sd arrays are sized maxid+1 (default maxid=200,
    src/synth_gmm_maisi_dataset.py:35) -- a shape id above maxid would index out of
    bounds when SynthGmmProvider indexes mu[crop_lbl]."""
    assert max(SHAPE_ID_TO_FAMILY) <= 200
