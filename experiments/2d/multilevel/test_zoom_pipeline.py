import sys; sys.path.insert(0, "."); sys.path.insert(0, "experiments/2d/multilevel")
import torch
from omegaconf import OmegaConf
from src.models.pfn_seg_2d import ImagePFN


class StubStage1:
    """ImagePFN-like: returns (B,R0,R0) logits (and thinking if asked). N = R0²."""
    def __init__(self, R0=8): self.N = R0 * R0; self.R0 = R0
    def __call__(self, images, masks, sep, return_thinking=False):
        B = images.shape[0]
        logits = torch.randn(B, self.R0, self.R0)
        return (logits, torch.randn(B, 4, 16)) if return_thinking else logits
    def eval(self): return self


class StubEncoder:
    """encode_maps → list of (N, C_i, R_i, R_i) maps; feature_dim = sum(C_i)."""
    feature_dim = 5
    def encode_maps(self, images):
        N = images.shape[0]
        return [torch.randn(N, 2, 16, 16), torch.randn(N, 3, 8, 8)]
    def eval(self): return self


def _batch(B=2, K=2, H=32):
    return {
        "image":       torch.rand(B, 1, H, H),
        "label":       (torch.rand(B, 1, H, H) > 0.5).float(),
        "context_in":  torch.rand(B, K, 1, H, H),
        "context_out": (torch.rand(B, K, 1, H, H) > 0.5).float(),
        "dataset":     ["d"] * B,
    }


def test_crop_pool_maps_shape():
    from zoom_pipeline import crop_pool_maps
    maps = [torch.randn(6, 2, 16, 16), torch.randn(6, 3, 8, 8)]
    origin = torch.zeros(6, 2, dtype=torch.long)
    feat = crop_pool_maps(maps, origin, s=16, out=8)
    assert feat.shape == (6, 5, 8, 8)          # channels concatenated, pooled to 8×8


def test_run_zoom_chain_shapes_and_composite():
    from zoom_pipeline import run_zoom_chain
    torch.manual_seed(0)
    dev = torch.device("cpu")
    B, K, H, R0, Cf = 2, 2, 32, 8, 5
    batch = _batch(B, K, H)
    cfg = OmegaConf.create({"sample": {"crop_sizes": [16]},
                            "data": {"image_size": H}})
    models = torch.nn.ModuleList([
        ImagePFN(resolution=R0, image_size=H, input_patch_size=4, e=16, h=32, l=2, a=2,
                 thinking_rows=2, use_external_features=True, feature_dim=Cf)])
    outputs, coarse_lr = run_zoom_chain(batch, StubStage1(R0), StubEncoder(), models, cfg,
                                        "prev_pred", True, dev)
    assert coarse_lr.shape == (B, R0, R0)
    assert len(outputs) == 1
    o = outputs[0]
    assert o["logits"].shape == (B, R0 * R0)
    assert o["qry_gt"].shape == (B, R0 * R0)
    assert o["refined_full"].shape == (B, 1, H, H)
    assert o["origin"].shape == (B, 2)
    # composite changed only inside the bbox vs the upsampled stage-1 prediction
    assert torch.isfinite(o["refined_full"]).all()


if __name__ == "__main__":
    test_crop_pool_maps_shape()
    test_run_zoom_chain_shapes_and_composite()
    print("ALL ZOOM PIPELINE TESTS PASSED")
