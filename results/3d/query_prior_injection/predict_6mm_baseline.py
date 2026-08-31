"""Spike: load the 3_6 checkpoint, predict a few val cases at 6 mm (no aug), plot pred vs GT.

Throwaway. Mirrors experiments/3d/eval.py::_build_model (patchset3d branch) for the model load
and common.make_eval_loader for a constant-spacing=6 eval loader.
"""
import os
import sys
from pathlib import Path

ROOT = Path("/home/dpxuser/dev/patch_icl")
os.chdir(ROOT)
os.environ["PWD"] = str(ROOT)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "3d"))

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

CKPT = ("/nfs/data/nii/data1/Analysis/camaret___in_context_segmentation/"
        "ANALYSIS_20251122/results/patch_icl/3d_train/2026-08-31_66_train_spacing_range_3_6/best.pt")
SPACING = 6.0
N_CASES = 10
OUT = Path(__file__).resolve().parent / "baseline_6mm_no_prior"
OUT.mkdir(parents=True, exist_ok=True)

with initialize_config_dir(config_dir=str(ROOT / "configs/experiment/3d"), version_base="1.3"):
    cfg = compose(config_name="train", overrides=[
        "experiment=57_organs_encoder_from_scratch",
        "data.context_size=1",
        "data.crop_spacing_mm=6",
        "data.train_spacing_range=[3,6]",
    ])

from common import DEVICE, make_eval_loader                       # noqa: E402
from train import build_model, _resolve_classes_for               # noqa: E402
from evaluate import save_eval_figure, dice_binary                # noqa: E402

print(f"device={DEVICE}")
ckpt = torch.load(CKPT, map_location=DEVICE, weights_only=False)
if ckpt.get("arch"):
    cfg.model = "patchset3d"
    cfg.arch = OmegaConf.create(ckpt["arch"])
print("arch:", {k: cfg.arch[k] for k in ("encoder", "e", "l", "a", "resolution",
      "mask_patch_size", "mask_patch_decode_size", "feat_norm", "encoder_frozen")})
print("train data:", {k: ckpt["data"].get(k) for k in ("crop_spacing_mm", "context_size",
      "train_spacing_range", "image_size")})

cfg.eval.workers = 0        # in-process loader (avoid forkserver bootstrap in a plain script)
model, _ = build_model(cfg)
model = model.to(DEVICE)
sd = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()}
missing, unexpected = model.load_state_dict(sd, strict=False)
print(f"load_state_dict: {len(missing)} missing, {len(unexpected)} unexpected")
if missing:
    print("  missing[:8]:", missing[:8])
if unexpected:
    print("  unexpected[:8]:", unexpected[:8])
model.eval()

_, root, _ = __import__("common")._source_root(cfg)
classes = _resolve_classes_for(cfg, "val_classes")
print(f"{len(classes)} val classes; loader spacing={SPACING} mm, image_size={cfg.data.image_size}")
loader = make_eval_loader(cfg, classes, split="val", spacing=SPACING)

done = 0
seen = set()          # one case per distinct class for a cross-class view
for batch in loader:
    ti, ci, cm, lab = (batch["image"], batch["context_in"],
                       batch["context_out"], batch["label"])
    names = batch["label_names"]
    subs = batch.get("subjects", [None] * ti.shape[0])
    sp_kw = ({"spacing": float(batch["spacing"][0, 0])}
             if getattr(model, "spacing_aware", False) and "spacing" in batch else {})
    with torch.no_grad():
        pred = model.predict(ti.to(DEVICE), ci.to(DEVICE), cm.to(DEVICE), **sp_kw).cpu()
    for i in range(pred.shape[0]):
        if names[i] in seen:
            continue
        seen.add(names[i])
        d = dice_binary(pred[i], lab[i])
        gt_vox = int(lab[i].sum())
        pr_vox = int(pred[i].sum())
        fn = OUT / f"{done:02d}_{names[i]}_{subs[i]}.png"
        save_eval_figure(
            target_img=ti[i, 0].numpy(), gt=lab[i].numpy(), pred=pred[i].numpy(),
            ctx_img=ci[i, 0, 0].numpy(), ctx_gt=cm[i, 0].numpy(), out_path=fn,
            title=f"{names[i]}  {subs[i]}  @{SPACING:g}mm   "
                  f"dice={d:.3f}  gt={gt_vox}  pred={pr_vox}")
        print(f"  [{done:02d}] {names[i]:<28s} {subs[i]}  dice={d:.3f}  "
              f"gt_vox={gt_vox}  pred_vox={pr_vox}  -> {fn.name}")
        done += 1
        if done >= N_CASES:
            break
    if done >= N_CASES:
        break

print(f"\nsaved {done} figures to {OUT}")
