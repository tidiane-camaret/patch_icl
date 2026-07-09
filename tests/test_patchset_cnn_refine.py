import sys; sys.path.insert(0, ".")
import torch
import torch.nn.functional as F
from src.models.patchset_cnn import PatchSetCNN


def _model(refine, H=32, R=8):
    torch.manual_seed(0)
    return PatchSetCNN(image_size=H, resolution=R, enc_dims=[16], e=32, h=64, l=1, a=2,
                       thinking_rows=1, refine=refine, refine_crop=16)


def _batch(B=2, K=2, H=32):
    torch.manual_seed(1)
    return (torch.rand(B, 1, H, H),
            torch.rand(B, K, 1, H, H),
            (torch.rand(B, K, 1, H, H) > 0.5).float())


def test_refine_false_unchanged_shape():
    m = _model(refine=False)
    img, cin, cout = _batch()
    out = m(img, context_in=cin, context_out=cout)["final_logit"]
    assert out.shape == (2, 1, 8, 8)                         # native R×R
    # forward == _segment when refine is off
    seg = m._segment(img, cin, cout)
    assert torch.equal(out, seg)


def test_refine_true_native_shape_and_finite():
    m = _model(refine=True)
    img, cin, cout = _batch()
    out = m(img, context_in=cin, context_out=cout)["final_logit"]
    assert out.shape == (2, 1, 32, 32)                       # native H×W fused
    assert torch.isfinite(out).all()


def test_refine_grad_reaches_encoder_and_decoder():
    m = _model(refine=True)
    img, cin, cout = _batch()
    lbl = (torch.rand(2, 1, 32, 32) > 0.5).float()
    out = m(img, context_in=cin, context_out=cout)["final_logit"]
    loss = F.binary_cross_entropy_with_logits(out, lbl)
    loss.backward()
    assert m.decoder[0].weight.grad is not None
    assert m.encoder.stem[0].weight.grad is not None         # coarse+refine both use encoder
