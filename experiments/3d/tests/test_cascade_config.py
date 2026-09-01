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


def test_exp59_ram_cache_and_gpu_realize_resolve():
    with initialize_config_dir(config_dir=CFG_DIR, version_base="1.3"):
        cfg = compose(config_name="train",
                      overrides=["experiment=59_organs_cascade_from_scratch"])
    assert cfg.data.ram_cache is True
    assert cfg.data.gpu_realize_crop is True


def _captured_provider_kwargs(monkeypatch, cfg, split):
    """build_dataset(cfg, split) with TotalSegProvider / InContextDataset stubbed out,
    returning the kwargs build_dataset actually passes to the provider + dataset."""
    import src.providers.totalseg as ts_mod
    import src.incontext_dataset_v2 as ds_mod
    seen = {}

    class _P:
        classes = []
        def __init__(self, **kw):
            seen["provider"] = kw
        def subjects_for(self, cls):
            return []

    class _D:
        def __init__(self, provider, **kw):
            seen["dataset"] = kw

    monkeypatch.setattr(ts_mod, "TotalSegProvider", _P)
    monkeypatch.setattr(ds_mod, "InContextDataset", _D)
    from common import build_dataset
    build_dataset(cfg, split)
    return seen


def _mini_cfg(**data):
    from omegaconf import OmegaConf
    base = {"model": "patchset3d",
            "paths": {"totalseg": "/nonexistent"},
            "data": {"loader_v2": True, "source": "totalseg", "image_size": [8, 8, 8],
                     "context_size": 1, "crop_spacing_mm": 3, "train_classes": ["liver"],
                     "val_classes": ["liver"], "cascade_spacings": [3, 1.5]},
            "eval": {"seed": 0}, "augmentations": {"enabled": False}}
    cfg = OmegaConf.create(base)
    cfg.data.merge_with(OmegaConf.create(data))
    return cfg


def test_ram_cache_default_follows_resolved_gpu_realize(monkeypatch):
    """ram_cache is only read by load_native_crop, so its default must track the
    RESOLVED gpu_realize_crop -- never `cascade_spacings` alone (which would preload
    ~35 GB for a gpu_realize_crop=false run and for every non-train dataset)."""
    seen = _captured_provider_kwargs(monkeypatch, _mini_cfg(), "train")
    assert seen["provider"]["ram_cache"] is True
    assert seen["dataset"]["gpu_realize_crop"] is True

    seen = _captured_provider_kwargs(monkeypatch, _mini_cfg(gpu_realize_crop=False), "train")
    assert seen["provider"]["ram_cache"] is False        # nothing would read it
    assert seen["dataset"]["gpu_realize_crop"] is False

    seen = _captured_provider_kwargs(monkeypatch, _mini_cfg(), "val")
    assert seen["provider"]["ram_cache"] is False        # val emits no native crops

    seen = _captured_provider_kwargs(monkeypatch,
                                     _mini_cfg(gpu_realize_crop=False, ram_cache=True), "train")
    assert seen["provider"]["ram_cache"] is True         # explicit override still wins
