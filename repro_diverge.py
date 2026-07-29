"""Verify the fix: a loss with RESTORING PRESSURE on the raw output keeps logits bounded.
Candidates all run the real optimizer loop from best.pt; report logits|max| trajectory
(the divergent quantity) + a hard-dice proxy so we see learning isn't sacrificed.
Pick candidate via env LOSSMODE in {broken, sl1_dice_raw, oob_dice}."""
import os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments" / "3d"))
import torch
from hydra import initialize_config_dir, compose
import train as T

MODE = os.environ.get("LOSSMODE", "sl1_dice_raw")
CKPT = "/nfs/data/nii/data1/Analysis/camaret___in_context_segmentation/ANALYSIS_20251122/results/patch_icl/3d_train/2026-07-29_dark-capybara-204/best.pt"
DEV = T.DEVICE
with initialize_config_dir(config_dir=str(ROOT / "configs" / "experiment" / "3d"), version_base="1.3"):
    cfg = compose(config_name="train", overrides=["experiment=31_medverse_colipri_task", f"train.checkpoint={CKPT}"])
torch.manual_seed(cfg.train.seed)
model, _ = T.build_model(cfg); net = getattr(model, "model", model)
model.load_finetuned(torch.load(CKPT, map_location=DEV, weights_only=False)["model"])
opt = T.build_optimizer(cfg, (p for p in net.parameters() if p.requires_grad))
clip = float(cfg.train.get("grad_clip", 1e9)); net.train()
sl1 = T.SmoothL3L1()

def st(x, lo, hi):
    return x + (x.clamp(lo, hi) - x).detach()

import torch.nn.functional as F

def _bce(out, target):  # autocast-unsafe -> fp32 with autocast disabled (as in train.py)
    with torch.autocast(device_type=DEV.type, enabled=False):
        return F.binary_cross_entropy(out.clamp(1e-6, 1 - 1e-6), target.float())

def loss_fn(logits, target):
    out = logits.float()
    dice = T._soft_dice(st(out, 0.0, 1.0), target)
    if MODE == "broken":                                   # current code: clamp blinds the loss
        return _bce(out, target) + dice
    if MODE == "sl1_dice_raw":                             # SmoothL3L1 on RAW output = restoring pressure
        return sl1(out, target) + dice
    if MODE == "oob_dice":                                 # explicit out-of-bounds penalty + dice
        oob = ((out - out.clamp(0.0, 1.0)) ** 2).mean()
        return dice + 10.0 * oob
    if MODE == "bce_dice_oob":                             # SHIPPABLE: keep bce_dice, add the anchor
        oob = ((out - out.clamp(0.0, 1.0)) ** 2).mean()
        return _bce(out, target) + dice + 10.0 * oob

def hard_dice(out, target):
    p = (out.clamp(0, 1) >= 0.5).float()
    return (2 * (p * target).sum() + 1) / (p.sum() + target.sum() + 1)

loader = T.train_loader(cfg)
print(f"LOSSMODE={MODE}")
dsum = dn = 0.0
for step, batch in enumerate(loader):
    if step >= 700:
        break
    lbl = batch["label"].to(DEV).float().unsqueeze(1)
    opt.zero_grad(set_to_none=True)
    with T._autocast():
        logits = model.train_forward(batch["image"], batch["context_in"], batch["context_out"])
        loss = loss_fn(logits, lbl)
    if not torch.isfinite(loss):
        print(f"*** non-finite loss @ step {step}"); break
    loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), clip)
    opt.step()
    dsum += hard_dice(logits.float(), lbl).item(); dn += 1
    if step % 50 == 0:
        print(f"step {step:4d} | loss={loss.item():+.4f} | logits|max|={logits.float().abs().max().item():.3g} "
              f"| dice(run)={dsum/dn:.3f} | fg={lbl.mean():.4f}")
