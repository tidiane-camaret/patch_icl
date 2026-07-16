import sys; sys.path.insert(0, ".")
import torch
import torch.nn.functional as F
from src.models.scatter_sampling import gather_grid


def _refine_target(lbl, refine_idx, Rf):
    B = lbl.shape[0]
    gt_Rf = F.adaptive_avg_pool2d(lbl, (Rf, Rf)).reshape(B, Rf * Rf)
    return gather_grid(gt_Rf, refine_idx)


def test_refine_target_shape_and_range():
    B, M, Rf = 2, 20, 16
    lbl = (torch.rand(B, 1, 32, 32) > 0.5).float()
    idx = torch.stack([torch.randperm(Rf * Rf)[:M] for _ in range(B)])
    t = _refine_target(lbl, idx, Rf)
    assert t.shape == (B, M)
    assert t.min() >= 0.0 and t.max() <= 1.0


def test_refine_loss_finite():
    B, M, Rf = 2, 20, 16
    lbl = (torch.rand(B, 1, 32, 32) > 0.5).float()
    idx = torch.stack([torch.randperm(Rf * Rf)[:M] for _ in range(B)])
    rlogit = torch.randn(B, M, requires_grad=True)
    rtarget = _refine_target(lbl, idx, Rf)
    bce = F.binary_cross_entropy_with_logits(rlogit, rtarget)
    assert torch.isfinite(bce)
    bce.backward()
    assert rlogit.grad is not None


def test_refine_dice_term_finite():
    import sys; sys.path.insert(0, "experiments/2d")
    from pfn_train import soft_dice_loss
    B, M, Rf = 2, 20, 16
    lbl = (torch.rand(B, 1, 32, 32) > 0.5).float()
    idx = torch.stack([torch.randperm(Rf * Rf)[:M] for _ in range(B)])
    rlogit = torch.randn(B, M)
    rtarget = _refine_target(lbl, idx, Rf)
    rdice = soft_dice_loss(torch.sigmoid(rlogit), rtarget)
    assert torch.isfinite(rdice)
