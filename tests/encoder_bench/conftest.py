import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))                                  # repo root (for `src`)
sys.path.insert(0, str(ROOT / "experiments" / "3d"))           # for `encoder_bench`
