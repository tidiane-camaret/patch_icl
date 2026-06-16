import sys; sys.path.insert(0, "."); sys.path.insert(0, "experiments/2d/multilevel")
import torch
from pipeline import build_patch_batch

class StubStage1:
    """Returns res-16 logits; here a fixed gradient so |pred-0.5| ranking is well-defined.
    Mirrors ImagePFN.forward(return_thinking=...): returns (logits, thinking) when asked."""
    def __call__(self, images, masks, sep, return_thinking=False):
        B = images.shape[0]
        # logits ramp across the 16x16 grid → varied sigmoid values
        row = torch.linspace(-4, 4, 16)
        grid = row.view(1, 16, 1).expand(B, 16, 16).clone()
        if return_thinking:
            return grid, torch.randn(B, 8, 64)   # (B, n_think, e1)
        return grid
    def eval(self): return self

class StubEncoder:
    """Returns (B*T, C, R, R) features. forward(images, out_size)."""
    feature_dim = 5
    def __call__(self, images, out_size):
        N = images.shape[0]
        return torch.randn(N, self.feature_dim, out_size, out_size)
    def eval(self): return self

class Cfg:
    class sample:
        grid_res = 32; n_total = 256; tau = 0.30; n_fg_core = 64
        blur_sigma = 1.0; floor = 0.005; temperature = 1.0
    class arch:   mask_prior = "scalar"

def test_build_patch_batch_shapes():
    B, K, H = 2, 3, 128
    batch = {
        "image":       torch.rand(B, 1, H, H),
        "label":       (torch.rand(B, 1, H, H) > 0.5).float(),
        "context_in":  torch.rand(B, K, 1, H, H),
        "context_out": (torch.rand(B, K, 1, H, H) > 0.5).float(),
    }
    out = build_patch_batch(batch, StubStage1(), StubEncoder(), Cfg, torch.device("cpu"))
    M = 256
    assert out["qry_feat"].shape  == (B, M, 5)
    assert out["sup_feat"].shape  == (B, K * M, 5)
    assert out["qry_ij"].shape    == (B, M, 2)
    assert out["sup_ij"].shape    == (B, K * M, 2)
    assert out["qry_gt"].shape    == (B, M)
    assert out["qry_coarse"].shape == (B, M)
    assert out["qry_prior"].shape == (B, M)
    # uncertain flag = boundary core: boolean, variable count, and the selected query
    # cells it marks have coarse value within tau of 0.5.
    assert out["qry_is_uncertain"].shape == (B, M)
    assert out["qry_is_uncertain"].dtype == torch.bool
    d = (out["qry_coarse"] - 0.5).abs()
    assert torch.all(d[out["qry_is_uncertain"]] < Cfg.sample.tau)
    # support labels/coords in valid ranges
    assert out["sup_label"].min() >= 0 and out["sup_label"].max() <= 1
    assert out["qry_ij"].max() < 32
    # stage-1 thinking memory passed through
    assert out["stage1_think"].shape == (B, 8, 64)
    # full-image metric tensors + flat query indices
    N = 32 * 32
    assert out["qry_idx"].shape == (B, M)
    assert out["coarse_full"].shape == (B, N)
    assert out["gt_full"].shape == (B, N)
    # flat query indices match the (i,j) coords: idx == i*R + j
    assert torch.equal(out["qry_idx"], out["qry_ij"][..., 0] * 32 + out["qry_ij"][..., 1])

class CfgPatch(Cfg):
    class arch:   mask_prior = "patch"

def test_build_patch_batch_patch_mode():
    B, K, H, R = 2, 3, 128, 32
    p = H // R                                  # auto p = 4 → p² = 16
    batch = {
        "image":       torch.rand(B, 1, H, H),
        "label":       (torch.rand(B, 1, H, H) > 0.5).float(),
        "context_in":  torch.rand(B, K, 1, H, H),
        "context_out": (torch.rand(B, K, 1, H, H) > 0.5).float(),
    }
    out = build_patch_batch(batch, StubStage1(), StubEncoder(), CfgPatch, torch.device("cpu"))
    M = 256
    # mask-token is now a p×p tile per patch on both sides
    assert out["sup_label"].shape == (B, K * M, p * p)
    assert out["qry_prior"].shape == (B, M, p * p)
    # support tiles come from the real binary GT → values in {0,1}
    assert set(torch.unique(out["sup_label"]).tolist()) <= {0.0, 1.0}
    # query coarse scalar (metrics baseline) is unchanged
    assert out["qry_coarse"].shape == (B, M)

if __name__ == "__main__":
    test_build_patch_batch_shapes()
    test_build_patch_batch_patch_mode()
    print("ALL PIPELINE TESTS PASSED")
