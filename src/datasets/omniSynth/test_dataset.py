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


def _iou(a, b):
    a = a > 0.5; b = b > 0.5
    inter = (a & b).sum().item(); union = (a | b).sum().item()
    return inter / union if union else 1.0


def test_copy_injects_one_aligned_slot():
    # p_copy=1, n_copy=1 (train): exactly one context slot is a near-copy of the
    # query (high mask-IoU), and it is the strongest match; meta flags it.
    ds = _ds("train", p_copy=1.0, n_copy=1)
    item = ds[0]
    assert item["meta"]["is_copy"] is True
    slot = item["meta"]["copy_slot"]
    assert 0 <= slot < K
    ious = [_iou(item["context_out"][j, 0], item["label"][0]) for j in range(K)]
    assert ious[slot] > 0.5, f"copy slot IoU too low: {ious}"
    assert ious[slot] == max(ious), f"copy slot not the strongest match: {ious}"


def test_copy_multi_slot():
    # p_copy=1, n_copy=2: two distinct slots are near-copies of the query.
    ds = _ds("train", p_copy=1.0, n_copy=2)
    item = ds[0]
    assert item["meta"]["is_copy"] is True
    ious = [_iou(item["context_out"][j, 0], item["label"][0]) for j in range(K)]
    n_high = sum(v > 0.5 for v in ious)
    assert n_high >= 2, f"expected >=2 copy slots, ious={ious}"


def test_eval_never_copies():
    # deterministic split must ignore p_copy entirely.
    ds = _ds("val", p_copy=1.0, n_copy=2)
    for i in range(min(4, len(ds))):
        assert ds[i]["meta"]["is_copy"] is False


def test_copy_disabled_when_pcopy_zero():
    ds = _ds("train", p_copy=0.0)
    item = ds[0]
    assert item["meta"]["is_copy"] is False
    assert item["meta"]["copy_slot"] == -1


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
    test_copy_injects_one_aligned_slot()
    test_copy_multi_slot()
    test_eval_never_copies()
    test_copy_disabled_when_pcopy_zero()
    test_item_shapes_and_keys()
    test_label_matches_target_cells_only()
    test_image_size_divisible_guard()
    test_val_deterministic_train_not()
    test_samples_contract()
    test_identical_mode_repeats_target_cells()
    test_mix_mode_samples_all_three()
    print("ALL DATASET TESTS PASSED")
