# tests/experiments_common_shim.py — expose experiments/3d/common.py under a stable name
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "3d"))
from common import build_dataset, make_eval_loader  # noqa: E402,F401
