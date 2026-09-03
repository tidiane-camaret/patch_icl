"""The `modality` field rides LoadResult -> item dict -> batch, unused downstream."""
import torch

from src.incontext_dataset_v2 import LoadResult
from src.totalseg_dataloader_incontext import incontext_collate_fn


def test_loadresult_modality_defaults_ct():
    r = LoadResult(image=torch.zeros(1, 4, 4, 4), label=torch.zeros(4, 4, 4),
                   spacing=torch.ones(3), crop_geom=torch.zeros(4, 3, dtype=torch.long))
    assert r.modality == "ct"


def test_loadresult_modality_settable():
    r = LoadResult(image=torch.zeros(1, 4, 4, 4), label=torch.zeros(4, 4, 4),
                   spacing=torch.ones(3), crop_geom=torch.zeros(4, 3, dtype=torch.long),
                   modality="mri")
    assert r.modality == "mri"


def _item(modality):
    return {
        "image": torch.zeros(1, 4, 4, 4),
        "label": torch.zeros(4, 4, 4, dtype=torch.long),
        "context_in": torch.zeros(1, 1, 4, 4, 4),
        "context_out": torch.zeros(1, 4, 4, 4, dtype=torch.long),
        "subject": "s0", "context_subjects": ["s1"], "label_name": "liver",
        "spacing": torch.ones(3), "aug_mode": torch.tensor(0, dtype=torch.long),
        "crop_geom": torch.zeros(4, 3, dtype=torch.long),
        "modality": modality,
    }


def test_collate_emits_modality_list():
    batch = incontext_collate_fn([_item("ct"), _item("mri")])
    assert batch["modality"] == ["ct", "mri"]


def test_collate_omits_modality_when_absent():
    it = _item("ct")
    del it["modality"]
    batch = incontext_collate_fn([it, it])
    assert "modality" not in batch
