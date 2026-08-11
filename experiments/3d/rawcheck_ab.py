"""Paired A/B: current float16-normalized store vs a lossless raw-HU store normalized
in the loader. Runs one PatchSet3D checkpoint over the eval loader twice on IDENTICAL
deterministic crops (eval_seed), differing only in the CT source:

  baseline : loader reads the stored float16 z-scored ct.npy (current pipeline)
  raw      : loader reads int16 raw-HU ct_raw (/tmp/rawcheck/raw/<subj>.npy) and applies
             the SAME clip+z-score in the loader (the proposed "store raw, normalize in
             loader" pipeline) — lossless (no float16 rounding; storage pre-clip removed)

Reports per-sample Dice under both and their agreement, so the signal-quality impact of
the pipeline change on the trained checkpoint is measured directly (paired, low variance).

  python experiments/3d/rawcheck_ab.py <checkpoint_path> [n_subjects]
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, open_dict

from common import DEVICE, make_eval_loader
from data.totalseg_classes import resolve_classes
import src.totalseg_dataloader_incontext as dl
from src.totalseg_dataset import CT_CLIP_MIN, CT_CLIP_MAX, CT_MEAN, CT_STD

RAW_DIR = Path("/tmp/rawcheck/raw")
_ORIG_OCA = dl.TotalSegInContextDataset._organ_crop_arrays


def _raw_organ_crop_arrays(self, subj_dir, label_mm, center, sp):
    """Copy of _organ_crop_arrays that loads int16 raw-HU ct_raw and normalizes the crop
    (clip+z-score) — i.e. the 'store raw, normalize in loader' path. Geometry identical."""
    T = self.image_size[0]
    cd, ch, cw = center
    D, H, W = label_mm.shape
    phys_ref = T * self._crop_mm
    target_sizes = [max(1, round(phys_ref / spi)) for spi in sp]
    crop_sizes = [min(dim, t) for t, dim in zip(target_sizes, (D, H, W))]
    j = self.crop_jitter
    starts = []
    for c, s, cs in zip((cd, ch, cw), (D, H, W), crop_sizes):
        ideal = c - cs // 2
        lo = max(0, ideal - j)
        hi = max(lo, min(max(0, s - cs), ideal + j))
        starts.append(self._cur_rng.randint(lo, hi))
    d0, h0, w0 = starts
    raw_path = RAW_DIR / f"{subj_dir.name}.npy"
    ct_mm = np.load(raw_path, mmap_mode="r")            # int16 raw HU, same geometry as ct.npy
    crop_ct = ct_mm[d0:d0+crop_sizes[0], h0:h0+crop_sizes[1], w0:w0+crop_sizes[2]]
    # normalize in the loader (lossless: no float16, storage pre-clip identical to convert)
    crop_ct = np.clip(crop_ct.astype(np.float32), CT_CLIP_MIN, CT_CLIP_MAX)
    crop_ct = (crop_ct - CT_MEAN) / CT_STD
    crop_lbl = label_mm[d0:d0+crop_sizes[0], h0:h0+crop_sizes[1], w0:w0+crop_sizes[2]]
    out_sizes = [max(1, min(T, round(cs / t * T))) for cs, t in zip(crop_sizes, target_sizes)]
    pad_lo = [(T - o) // 2 for o in out_sizes]
    return crop_ct, crop_lbl, out_sizes, pad_lo


def build(ckpt_path, n_subjects):
    with initialize_config_dir(config_dir=str(ROOT / "configs/experiment/3d"), version_base="1.3"):
        cfg = compose(config_name="eval", overrides=[
            "eval.model=patchset3d", "eval.split=test", "data.val_classes=benchmark",
            f"eval.checkpoint={ckpt_path}", f"eval.n_subjects={n_subjects}",
            "wandb.project=null", "eval.save_figures=false",
            "eval.workers=0",   # in-process loading so the raw monkeypatch definitely applies
        ])
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    from train import build_model
    with open_dict(cfg):
        cfg.model = "patchset3d"
        cfg.arch = OmegaConf.create(ckpt["arch"])
    model, _ = build_model(cfg)
    model = model.to(DEVICE)
    sd = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()}
    model.load_state_dict(sd)
    model.eval()
    _, root, is_mri = __import__("common")._source_root(cfg)
    classes = resolve_classes(cfg.data.val_classes, root, is_mri=is_mri)
    return cfg, model, classes


@torch.no_grad()
def run_pass(model, loader, spacing_aware):
    """Return dict subject|class -> per-sample Dice (vs GT) and the raw pred masks."""
    dices, preds = [], []
    for batch in loader:
        img = batch["image"].to(DEVICE)
        cin = batch["context_in"].to(DEVICE)
        cout = batch["context_out"].to(DEVICE)
        # spacing-aware frozen encoder wants ONE scalar per batch (shared RoPE table),
        # matching evaluate.py: float(batch["spacing"][0,0]).
        sp = float(batch["spacing"][0, 0]) if (spacing_aware and "spacing" in batch) else None
        pred = model.predict(img, cin, cout, spacing=sp)          # (B,D,H,W) {0,1}
        gt = (batch["label"].to(DEVICE) > 0).float()
        inter = (pred * gt).flatten(1).sum(1)
        denom = pred.flatten(1).sum(1) + gt.flatten(1).sum(1)
        d = torch.where(denom > 0, 2 * inter / denom, torch.ones_like(denom))
        dices.append(d.cpu())
        preds.append(pred.cpu().bool())
    return torch.cat(dices), preds


def main():
    ckpt = sys.argv[1]
    n_sub = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    cfg, model, classes = build(ckpt, n_sub)
    spacing_aware = bool(getattr(model, "spacing_aware", False))
    print(f"ckpt={ckpt}\n  encoder={cfg.arch.encoder}  spacing_aware={spacing_aware}  "
          f"n_subjects={n_sub}  classes={len(classes)}")

    # --- input-level sanity: prove the raw path changes the encoder input -----------
    ds = make_eval_loader(cfg, classes, split="test").dataset
    img_base = ds[0]["image"].clone()
    dl.TotalSegInContextDataset._organ_crop_arrays = _raw_organ_crop_arrays
    img_raw = ds[0]["image"].clone()
    dl.TotalSegInContextDataset._organ_crop_arrays = _ORIG_OCA
    diff = (img_raw - img_base).abs()
    print(f"  [sanity] baseline vs raw input crop: identical={torch.equal(img_raw, img_base)}  "
          f"max|Δ|={diff.max():.5f} (norm units) ~ {diff.max()*CT_STD:.3f} HU  "
          f"mean|Δ|={diff.mean():.6f}")

    torch.manual_seed(0)
    loader_b = make_eval_loader(cfg, classes, split="test")
    dice_b, preds_b = run_pass(model, loader_b, spacing_aware)

    dl.TotalSegInContextDataset._organ_crop_arrays = _raw_organ_crop_arrays
    try:
        torch.manual_seed(0)
        loader_r = make_eval_loader(cfg, classes, split="test")
        dice_r, preds_r = run_pass(model, loader_r, spacing_aware)
    finally:
        dl.TotalSegInContextDataset._organ_crop_arrays = _ORIG_OCA

    n = min(len(dice_b), len(dice_r))
    dice_b, dice_r = dice_b[:n], dice_r[:n]
    ddelta = (dice_r - dice_b)
    # voxel agreement between the two predictions (paired, per sample)
    agrees = []
    for pb, pr in zip(preds_b, preds_r):
        m = min(pb.shape[0], pr.shape[0])
        for i in range(m):
            a, b = pb[i], pr[i]
            agrees.append(float((a == b).float().mean()))
    agrees = np.array(agrees[:n])
    print("\n================  RAW-vs-FLOAT16 PAIRED RESULT  ================")
    print(f"  samples                : {n}")
    print(f"  mean Dice  baseline(f16): {dice_b.mean():.5f}")
    print(f"  mean Dice  raw(lossless): {dice_r.mean():.5f}")
    print(f"  mean Dice delta (raw-f16): {ddelta.mean():+.6f}  "
          f"(std {ddelta.std():.6f}, max|Δ| {ddelta.abs().max():.6f})")
    print(f"  pred voxel agreement    : mean {agrees.mean()*100:.4f}%  "
          f"min {agrees.min()*100:.4f}%")
    print(f"  # samples with any pred change: {(ddelta.abs() > 0).sum().item()} / {n}")
    print("================================================================")


if __name__ == "__main__":
    main()
