import sys; sys.path.insert(0, ".")
import torch
from src.gpu_augment import _stack_task, _unstack_task


def _fake_batch(B=2, K=3, D=6, H=6, W=6):
    return {
        "image":       torch.randn(B, 1, D, H, W),
        "label":       torch.randint(0, 2, (B, D, H, W)),
        "context_in":  torch.randn(B, K, 1, D, H, W),
        "context_out": torch.randint(0, 2, (B, K, D, H, W)),
        "aug_mode":    torch.zeros(B, dtype=torch.long),
    }


def test_stack_unstack_roundtrip():
    b = _fake_batch()
    ref = {k: v.clone() for k, v in b.items()}
    vols, masks, B, T = _stack_task(b)
    assert vols.shape == (B * T, 1, 6, 6, 6)
    assert masks.shape == (B * T, 6, 6, 6)
    assert masks.dtype == torch.long
    # target of task 0 is vols[0]; first context of task 0 is vols[1]
    assert torch.equal(vols[0, 0], ref["image"][0, 0])
    assert torch.equal(vols[1, 0], ref["context_in"][0, 0, 0])
    _unstack_task(vols, masks, B, T, b)
    for k in ("image", "label", "context_in", "context_out"):
        assert torch.equal(b[k], ref[k])
