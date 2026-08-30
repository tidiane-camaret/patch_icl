"""Task 9: experiment=59_organs_cascade_from_scratch resolves and passes the guard."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hydra import compose, initialize_config_dir

from common import _assert_cascade_supported

CFG_DIR = str(ROOT / "configs" / "experiment" / "3d")


def test_exp59_resolves_and_passes_guard():
    with initialize_config_dir(config_dir=CFG_DIR, version_base="1.3"):
        cfg = compose(config_name="train",
                      overrides=["experiment=59_organs_cascade_from_scratch"])
    assert list(cfg.data.cascade_spacings) == [3, 1.5]
    assert float(cfg.data.crop_spacing_mm) == 3.0
    assert list(cfg.train.cascade_loss_weights) == [1.0, 1.0]
    assert cfg.data.get("train_spacing_range") is None
    _assert_cascade_supported(cfg)          # no raise
