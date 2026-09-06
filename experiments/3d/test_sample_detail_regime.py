"""_sample_detail renders a multisource regime meta into the detail column."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate import _sample_detail


def test_regime_meta_renders():
    assert _sample_detail({"regime": "cross", "tgt_mod": "ct", "ctx_mod": "mri"}) == "cross ct<-mri"
    assert _sample_detail({"regime": "ct", "tgt_mod": "ct", "ctx_mod": "ct"}) == "ct ct<-ct"


def test_fallback_marker():
    # a "cross" draw that collapsed to same-modality gets a [fb] suffix
    assert _sample_detail({"regime": "cross", "tgt_mod": "ct", "ctx_mod": "ct",
                           "fallback": True}) == "cross ct<-ct [fb]"
    # genuine cross (fallback False/absent) is unmarked
    assert _sample_detail({"regime": "cross", "tgt_mod": "ct", "ctx_mod": "mri",
                           "fallback": False}) == "cross ct<-mri"


def test_non_regime_meta_unchanged():
    assert _sample_detail(None) == ""
    assert _sample_detail({}) == ""
    assert _sample_detail({"class_id": 3, "target_mode": "x", "sample_index": 1}) == \
        "mode=x class=3 sub=1"
