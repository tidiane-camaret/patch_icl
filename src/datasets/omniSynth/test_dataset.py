import sys; sys.path.insert(0, ".")
import torch
from src.datasets.omniSynth.config import OmniSceneConfig, OmniSamplingConfig
from src.datasets.omniSynth.dataset import OmniSynthICLDataset

K, IMG = 3, 64


def _ds(split, **scene_kw):
    return OmniSynthICLDataset(
        split=split, context_size=K, image_size=IMG,
        scene=OmniSceneConfig(**scene_kw),
        sampling=OmniSamplingConfig(eval_subjects_per_task=2),
    )


def test_item_shapes_and_keys():
    ds = _ds("train")
    item = ds[0]
    assert set(item) == {"image", "label", "context_in", "context_out", "meta"}
    assert item["image"].shape == (1, IMG, IMG)
    assert item["label"].shape == (1, IMG, IMG)
    assert item["context_in"].shape == (K, 1, IMG, IMG)
    assert item["context_out"].shape == (K, 1, IMG, IMG)


def test_label_matches_target_cells_only():
    ds = _ds("val", grid=4, k_min=2, k_max=2)   # pin 4x4 (cell=16) — this test's block math
    item = ds[0]
    img = item["image"][0]
    lbl = item["label"][0]
    assert lbl.dtype == torch.float32
    assert lbl.max() <= 1.0 and lbl.min() >= 0.0
    cells_with_mask = 0
    for cell in range(16):
        r, c = divmod(cell, 4)
        ib = img[r * 16:(r + 1) * 16, c * 16:(c + 1) * 16]
        mb = lbl[r * 16:(r + 1) * 16, c * 16:(c + 1) * 16]
        if mb.sum() > 0:
            cells_with_mask += 1
            assert torch.equal(mb, ib)          # mask = the character ink in that cell
            assert mb.sum() < 16 * 16           # ink, not the whole-cell block
    assert cells_with_mask == 2                 # exactly k=2 target cells masked


def test_image_size_divisible_guard():
    try:
        OmniSynthICLDataset(split="train", context_size=K, image_size=63,
                            scene=OmniSceneConfig(grid=4))
        assert False, "expected ValueError for non-divisible image_size"
    except ValueError:
        pass


def test_val_deterministic_train_not():
    v1, v2 = _ds("val"), _ds("val")
    a, b = v1[0], v2[0]
    assert torch.equal(a["image"], b["image"]) and torch.equal(a["label"], b["label"])
    assert torch.equal(a["context_in"], b["context_in"])
    # train: two reads almost surely differ
    t = _ds("train")
    assert not torch.equal(t[0]["image"], t[1]["image"])


def test_samples_contract():
    tr = _ds("train")
    assert tr.samples[0][0] == "omniglot/train"
    va = _ds("val")
    assert len(va.samples) == len(va)
    name, idx, lab = va.samples[0]
    assert name.startswith("omniglot/") and lab == 1


def test_identical_mode_repeats_target_cells():
    # identical: every target cell is the same bitmap -> all target cells byte-identical
    ds = _ds("val", grid=4, target_mode="identical", k_min=3, k_max=3)   # pin 4x4 (cell=16)
    item = ds[0]
    img = item["image"][0]
    cells = []
    for cell in range(16):
        r, c = divmod(cell, 4)
        block = img[r * 16:(r + 1) * 16, c * 16:(c + 1) * 16]
        if item["label"][0, r * 16:(r + 1) * 16, c * 16:(c + 1) * 16].sum() > 0:
            cells.append(block)
    assert len(cells) == 3
    for blk in cells[1:]:
        assert torch.equal(blk, cells[0])


def test_mix_mode_samples_all_three():
    # target_mode="mix" -> each item draws one of identical|aug|class; over many
    # train items all three appear, and meta records the resolved mode (a valid one).
    ds = _ds("train", target_mode="mix")
    seen = set()
    for i in range(60):
        m = ds[i]["meta"]["target_mode"]
        assert m in ("identical", "aug", "class")
        seen.add(m)
    assert seen == {"identical", "aug", "class"}, f"expected all 3 modes, got {seen}"


if __name__ == "__main__":
    test_item_shapes_and_keys()
    test_label_matches_target_cells_only()
    test_image_size_divisible_guard()
    test_val_deterministic_train_not()
    test_samples_contract()
    test_identical_mode_repeats_target_cells()
    test_mix_mode_samples_all_three()
    print("ALL DATASET TESTS PASSED")
