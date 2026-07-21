"""Import shim: `experiments/3d` cannot be imported as a package (dir name starts
with a digit), so expose build_dataset by path for tests and callers that need it."""
import importlib.util
import sys
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "_threed_common", Path(__file__).resolve().parent / "3d" / "common.py")
_mod = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("_threed_common", _mod)
_spec.loader.exec_module(_mod)

build_dataset = _mod.build_dataset
