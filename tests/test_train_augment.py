import sys; sys.path.insert(0, ".")
sys.path.insert(0, "experiments/2d")
import torch
from omegaconf import OmegaConf
from train import _augment_batch
from src.models.patchset_cnn import PatchSetCNN


def _geom_only_cfg():
    # rotate always on; every intensity/task op off -> query stays byte-identical.
    # augment() accesses ic.brightness/contrast/gamma/noise directly, so include them
    # with p=0 to satisfy the schema without touching anything.
    return OmegaConf.create({
        "enabled": True,
        "geometric": {"hflip_p": 0.0, "vflip_p": 0.0,
                      "rotate": {"p": 1.0, "max_angle_deg": 45.0}},
        "intensity": {
            "brightness": {"p": 0.0, "max_delta": 0.1},
            "contrast":   {"p": 0.0, "range": [0.8, 1.2]},
            "gamma":      {"p": 0.0, "range": [0.75, 1.33]},
            "noise":      {"p": 0.0, "std": 0.04},
        },
    })


def test_augment_batch_leaves_query_untouched_and_changes_context():
    torch.manual_seed(0)
    B, K, H = 2, 2, 16
    img = torch.rand(B, 1, H, H)
    cin = torch.rand(B, K, 1, H, H)
    cout = (torch.rand(B, K, 1, H, H) > 0.5).float()
    out_img, out_cin, out_cout = _augment_batch(img.clone(), cin.clone(), cout.clone(),
                                                _geom_only_cfg())
    # query (index K) is never geometrically transformed and no intensity op is active
    assert torch.equal(out_img, img)
    # contexts were rotated
    assert not torch.equal(out_cin, cin)


def test_augment_batch_shapes():
    B, K, H = 2, 3, 16
    img = torch.rand(B, 1, H, H)
    cin = torch.rand(B, K, 1, H, H)
    cout = (torch.rand(B, K, 1, H, H) > 0.5).float()
    out_img, out_cin, out_cout = _augment_batch(img, cin, cout, _geom_only_cfg())
    assert out_img.shape == (B, 1, H, H)
    assert out_cin.shape == (B, K, 1, H, H)
    assert out_cout.shape == (B, K, 1, H, H)


def test_augmented_batch_trains_one_step():
    torch.manual_seed(0)
    B, K, H = 2, 2, 32
    model = PatchSetCNN(image_size=H, resolution=8, enc_dims=[16], e=32, h=64, l=1, a=2,
                        thinking_rows=1, resolutions=[8, 16])
    img = torch.rand(B, 1, H, H)
    cin = torch.rand(B, K, 1, H, H)
    cout = (torch.rand(B, K, 1, H, H) > 0.5).float()
    aug = OmegaConf.create({
        "enabled": True,
        "geometric": {"hflip_p": 0.5, "vflip_p": 0.5, "rotate": {"p": 0.5, "max_angle_deg": 20.0}},
        "intensity": {
            "brightness": {"p": 0.5, "max_delta": 0.15},
            "contrast":   {"p": 0.0, "range": [0.8, 1.2]},
            "gamma":      {"p": 0.0, "range": [0.75, 1.33]},
            "noise":      {"p": 0.0, "std": 0.04},
        },
    })
    img, cin, cout = _augment_batch(img, cin, cout, aug)
    out = model(img, context_in=cin, context_out=cout, mode="train")
    loss = out["final_logit"].mean() + out["refine_logit"].mean()
    loss.backward()
    assert torch.isfinite(loss)


def test_train_base_augment_defaults_false():
    c = OmegaConf.load("configs/experiment/2d/train_base.yaml")
    assert c.get("augment", False) is False


def test_omnisynth_refine_opts_in():
    c = OmegaConf.load("configs/experiment/2d/2_omnisynth_medseg_refine.yaml")
    assert c.get("augment", False) is True
