import sys; sys.path.insert(0, ".")
import torch
from src.models.pfn_seg_2d import ImagePFN


def test_default_behavior_unchanged():
    torch.manual_seed(0)
    m = ImagePFN(resolution=8, image_size=32, input_patch_size=4, e=16, h=32, l=2, a=2,
                 thinking_rows=2)
    B, T, H = 2, 3, 32
    imgs = torch.rand(B, T, 1, H, H); msk = (torch.rand(B, T, 1, H, H) > 0.5).float()
    out = m(imgs, msk, sep=2)
    assert out.shape == (B, 8, 8)


def test_external_features_mode():
    torch.manual_seed(0)
    Cf = 5
    m = ImagePFN(resolution=8, image_size=32, input_patch_size=4, e=16, h=32, l=2, a=2,
                 thinking_rows=2, use_external_features=True, feature_dim=Cf)
    assert m.image_encoder is None
    assert m.image_embed.in_features == Cf
    B, T, H, N = 2, 3, 32, 64
    feats = torch.randn(B, T, N, Cf)
    msk = (torch.rand(B, T, 1, H, H) > 0.5).float()
    out = m(None, msk, sep=2, image_feats=feats, seed_query_mask=True)
    assert out.shape == (B, 8, 8)


def test_seed_query_mask_changes_output():
    # With seeding ON, the query mask we pass should influence the prediction; with it OFF
    # the query mask is overwritten by the context-mean, so the passed query mask is ignored.
    torch.manual_seed(0)
    Cf = 5
    m = ImagePFN(resolution=8, image_size=32, input_patch_size=4, e=16, h=32, l=2, a=2,
                 thinking_rows=2, use_external_features=True, feature_dim=Cf)
    B, T, H, N = 1, 3, 32, 64
    feats = torch.randn(B, T, N, Cf)
    msk = (torch.rand(B, T, 1, H, H) > 0.5).float()
    msk_a = msk.clone(); msk_a[:, -1] = 0.0      # query mask all-zero
    msk_b = msk.clone(); msk_b[:, -1] = 1.0      # query mask all-one
    with torch.no_grad():
        seed_a = m(None, msk_a, sep=2, image_feats=feats, seed_query_mask=True)
        seed_b = m(None, msk_b, sep=2, image_feats=feats, seed_query_mask=True)
        off_a  = m(None, msk_a, sep=2, image_feats=feats, seed_query_mask=False)
        off_b  = m(None, msk_b, sep=2, image_feats=feats, seed_query_mask=False)
    assert not torch.allclose(seed_a, seed_b)    # seeding: query mask matters
    assert torch.allclose(off_a, off_b)          # off: query mask overwritten → identical


if __name__ == "__main__":
    test_default_behavior_unchanged()
    test_external_features_mode()
    test_seed_query_mask_changes_output()
    print("ALL IMAGEPFN MODE TESTS PASSED")
