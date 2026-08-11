import json, os
import numpy as np
import pytest
from data.totalseg_total_map import (
    TOTALSEG_V2_TOTAL, build_ts_to_project_lut, remap_ts_total,
)
from data.totalseg_classes import ALL_CLASSES

_CLASS_TO_IDX = {c: i + 1 for i, c in enumerate(ALL_CLASSES)}

def test_map_has_117_known_anchors():
    assert len(TOTALSEG_V2_TOTAL) == 117
    # TS ids are 1-indexed positions in the ordered list
    assert TOTALSEG_V2_TOTAL[0] == "spleen"       # ts id 1
    assert TOTALSEG_V2_TOTAL[4] == "liver"        # ts id 5
    assert TOTALSEG_V2_TOTAL[50] == "heart"       # ts id 51
    assert TOTALSEG_V2_TOTAL[51] == "aorta"       # ts id 52

def test_all_ts_names_exist_in_project():
    assert all(n in _CLASS_TO_IDX for n in TOTALSEG_V2_TOTAL)

def test_remap_translates_by_name():
    arr = np.array([[0, 5, 52, 51]], dtype=np.int16)  # bg, liver, aorta, heart (TS ids)
    out = remap_ts_total(arr)
    assert out.dtype == np.uint8
    assert out[0, 0] == 0
    assert out[0, 1] == _CLASS_TO_IDX["liver"]
    assert out[0, 2] == _CLASS_TO_IDX["aorta"]
    assert out[0, 3] == _CLASS_TO_IDX["heart"]

def test_matches_cohort_stats_file_if_present():
    # The cohort's per-subject stats json is keyed by name in TS id order; if reachable,
    # assert our embedded list matches it exactly (transcription guard).
    p = ("/nfs/data/nii/data1/jungm___ChemoTox/10116066/20220316122148/"
         "ML/total_seg_total_stats_recomp.json")
    if not os.path.exists(p):
        pytest.skip("cohort stats file not reachable")
    names = list(json.load(open(p)).keys())
    assert names == TOTALSEG_V2_TOTAL
