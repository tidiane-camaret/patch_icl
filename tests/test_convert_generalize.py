import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.convert_to_npy import _resample_to_spacing

def test_resample_halves_shape_when_target_double_native():
    vol = np.random.rand(20, 20, 20).astype(np.float32)
    out = _resample_to_spacing(vol, native_sp=[1.0, 1.0, 1.0], target_sp=2.0, order=1)
    assert out.shape == (10, 10, 10)

def test_resample_anisotropic_native():
    vol = np.zeros((20, 20, 10), dtype=np.uint8)
    out = _resample_to_spacing(vol, native_sp=[1.5, 1.5, 3.0], target_sp=1.5, order=0)
    # x,y already 1.5 -> unchanged; z at 3.0 -> doubles to 20
    assert out.shape == (20, 20, 20)
    assert out.dtype == np.uint8
