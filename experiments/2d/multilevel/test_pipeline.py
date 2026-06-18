import sys; sys.path.insert(0, "."); sys.path.insert(0, "experiments/2d/multilevel")
import torch

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


def test_refine_level_shapes_and_composite():
    import torch
    import sys; sys.path.insert(0, "experiments/2d/multilevel")
    from pipeline import refine_level
    from src.models.patchset_pfn import PatchSetPFN
    from types import SimpleNamespace
    torch.manual_seed(0)
    dev = torch.device("cpu")
    B, K, R, Cf, e, nth = 2, 2, 8, 8, 16, 4
    N, T = R * R, K + 1
    H = 32                                   # native image size for this toy (p = H//R = 4)
    batch = {
        "image":       torch.rand(B, 1, H, H),
        "label":       (torch.rand(B, 1, H, H) > 0.5).float(),
        "context_in":  torch.rand(B, K, 1, H, H),
        "context_out": (torch.rand(B, K, 1, H, H) > 0.5).float(),
    }
    feats = torch.randn(B, T, N, Cf)
    coarse_grid = torch.rand(B, N)
    model = PatchSetPFN(feature_dim=Cf, e=e, h=32, l=2, a=2, thinking_rows=nth,
                        mask_prior="scalar", mask_patch_size=H // R, stage1_dim=e,
                        query_self_attn=True)
    s = SimpleNamespace(n_total=10, n_fg_core=2, n_fg_core_ctx=4, tau=0.3,
                        blur_sigma=1.0, floor=0.01, temperature=1.0, mask_prior="scalar")
    prev_think = torch.randn(B, nth, e)
    out = refine_level(model, batch, feats, coarse_grid, prev_think, R, s,
                       "prev_pred", True, dev)
    assert out["logits"].shape == (B, s.n_total)
    assert out["refined_grid"].shape == (B, N)
    assert out["this_think"].shape == (B, nth, e)
    assert out["qidx"].shape == (B, s.n_total)
    sel = torch.zeros(B, N, dtype=torch.bool).scatter_(1, out["qidx"], True)
    assert torch.allclose(out["refined_grid"][~sel], coarse_grid[~sel])  # unsampled = input


def test_composite_predictions():
    import torch
    import sys; sys.path.insert(0, "experiments/2d/multilevel")
    from pipeline import composite_predictions
    B, N, M = 2, 16, 5
    coarse = torch.rand(B, N)
    qidx = torch.stack([torch.randperm(N)[:M] for _ in range(B)])
    vals = torch.rand(B, M)
    out = composite_predictions(coarse, qidx, vals)
    assert out.shape == (B, N)
    for b in range(B):
        sel = set(qidx[b].tolist())
        for j in range(N):
            if j in sel:
                pos = (qidx[b] == j).nonzero()[0, 0]
                assert torch.allclose(out[b, j], vals[b, pos])     # overwritten
            else:
                assert torch.allclose(out[b, j], coarse[b, j])     # untouched
    assert out is not coarse                                       # no in-place mutation


def test_run_chain_detaches_and_shapes():
    import torch
    import sys; sys.path.insert(0, "experiments/2d/multilevel")
    from pipeline import run_chain
    from src.models.patchset_pfn import PatchSetPFN
    from omegaconf import OmegaConf
    torch.manual_seed(0)
    dev = torch.device("cpu")
    B, K, H, Cf, e, nth = 2, 2, 32, 8, 16, 4
    R0, ladder = 16, [16, 32]                          # single hop → grid 32
    batch = {"image": torch.rand(B, 1, H, H),
             "label": (torch.rand(B, 1, H, H) > 0.5).float(),
             "context_in": torch.rand(B, K, 1, H, H),
             "context_out": (torch.rand(B, K, 1, H, H) > 0.5).float()}

    class StubStage1:        # mimics ImagePFN: stage1(imgs, masks, sep=K, return_thinking=True)
        def __call__(self_, all_images, all_masks, sep, return_thinking=False):
            b = all_images.shape[0]
            logits = torch.rand(b, R0, R0)              # res-16 logits
            think = torch.randn(b, nth, e)
            return (logits, think) if return_thinking else logits
    def stub_encoder(images, grid):                    # encode_grid calls encoder(imgs, grid)
        bT = images.shape[0]
        return torch.randn(bT, Cf, grid, grid)

    cfg = OmegaConf.create({"sample": {
        "resolutions": ladder, "n_total": [10], "n_fg_core": [2], "n_fg_core_ctx": [4],
        "tau": 0.3, "blur_sigma": 1.0, "floor": 0.01, "temperature": 1.0},
        "arch": {"mask_prior": "scalar"}, "data": {"image_size": H}})
    models = torch.nn.ModuleList([
        PatchSetPFN(feature_dim=Cf, e=e, h=32, l=2, a=2, thinking_rows=nth,
                    mask_prior="scalar", mask_patch_size=H // 32, stage1_dim=e,
                    query_self_attn=True)])
    outputs, coarse_lr = run_chain(batch, StubStage1(), stub_encoder, models, cfg,
                                   "prev_pred", True, dev)
    assert len(outputs) == 1
    assert outputs[0]["refined_grid"].shape == (B, 32 * 32)
    assert coarse_lr.shape == (B, R0, R0)


if __name__ == "__main__":
    test_composite_predictions()
    test_refine_level_shapes_and_composite()
    test_run_chain_detaches_and_shapes()
    print("ALL PIPELINE TESTS PASSED")
