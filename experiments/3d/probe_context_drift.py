"""
Context-drift probe: how fast does Dice fall when the context is the target's OWN image+mask
but perturbed by a KNOWN geometric transform (translation / rotation / scale / elastic)?

Bridges the self-context ceiling (identical context -> Dice ~0.93) toward the real cross-subject
task, where context and target differ by pose/shape. We keep the TARGET fixed, warp only the
CONTEXT (image+mask together, so it stays a valid pair), and sweep the perturbation magnitude
one factor at a time. Magnitude 0 == self-context, so it must reproduce the ceiling (a sanity
check). The slope = the model's tolerance to target/context misalignment.

Loads a trained PatchSet3D checkpoint (default: the self-context run). NB that checkpoint was
trained at ZERO drift, so its drift-tolerance reflects a copy-specialised model.

    python experiments/3d/probe_context_drift.py experiment=40_colipri_large_head \
      train.checkpoint=/.../40_selfctx_ceiling/best.pt probe.n_samples=120
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import hydra
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.append(str(ROOT / "experiments" / "2d"))

from data.totalseg_classes import resolve_classes
from common import DEVICE, _source_root, make_eval_loader
from train import build_model
from src.augmentations import _make_affine_theta, _apply_grid

DEG = 3.141592653589793 / 180.0


def _warp(img, mask, *, rx=0.0, ry=0.0, rz=0.0, scale=1.0, t=(0.0, 0.0, 0.0), elastic=0.0,
          sigma=12.0, seed=0):
    """Warp one context volume (img (1,D,H,W), mask (D,H,W)) on DEVICE.

    Affine (rot rad / scale / normalised translate) then optional elastic (peak `elastic`
    voxels, matching augmentations.apply_synth_aug's convention). Image bilinear, mask nearest.
    """
    D, H, W = img.shape[-3:]
    theta = _make_affine_theta(rx, ry, rz, scale, t[0], t[1], t[2]).to(img.device)
    grid = F.affine_grid(theta, (1, 1, D, H, W), align_corners=False)
    if elastic > 0:
        g = torch.Generator(device="cpu").manual_seed(seed)
        sd, sh, sw = (max(2, round(D / sigma)), max(2, round(H / sigma)), max(2, round(W / sigma)))
        disp = F.interpolate(torch.randn(1, 3, sd, sh, sw, generator=g), size=(D, H, W),
                             mode="trilinear", align_corners=False).squeeze(0).to(img.device)
        disp = disp / disp.abs().amax().clamp(min=1e-6) * elastic
        scale_n = torch.tensor([2.0 / D, 2.0 / H, 2.0 / W], device=img.device).view(3, 1, 1, 1)
        grid = (grid + (disp * scale_n).permute(1, 2, 3, 0).unsqueeze(0)).clamp(-1, 1)
    wi, wm = _apply_grid(img.unsqueeze(0), mask.unsqueeze(0), grid)
    return wi.squeeze(0), wm.squeeze(0)


def _dice(pred, gt):
    p, g = (pred > 0.5).float(), (gt > 0.5).float()
    inter = (p * g).sum().item()
    return (2 * inter + 1) / (p.sum().item() + g.sum().item() + 1)


# One-factor-at-a-time magnitude grids. Translation/elastic in VOXELS (× spacing = mm);
# rotation in degrees (single axis); scale as a multiplicative factor.
SWEEPS = {
    "translate_vox": [0, 2, 4, 8, 16, 24],
    "rotate_deg":    [0, 5, 10, 20, 30, 45],
    "scale":         [1.0, 0.95, 0.90, 0.85, 1.05, 1.10, 1.15],
    "elastic_vox":   [0, 2, 4, 8, 12, 16],
}


def _params(axis, mag, D):
    """Map (axis, magnitude) -> _warp kwargs. Translation along the first spatial axis."""
    if axis == "translate_vox":
        return {"t": (2.0 * mag / D, 0.0, 0.0)}          # normalised: 2*vox/dim
    if axis == "rotate_deg":
        return {"rz": mag * DEG}
    if axis == "scale":
        return {"scale": float(mag)}
    if axis == "elastic_vox":
        return {"elastic": float(mag)}
    raise ValueError(axis)


@hydra.main(config_path="../../configs/experiment/3d", config_name="train", version_base="1.3")
@torch.no_grad()
def main(cfg: DictConfig) -> None:
    ckpt_path = cfg.train.get("checkpoint")
    assert ckpt_path and ckpt_path not in ("orig_weights", "random"), "set train.checkpoint=<path>"
    model, name = build_model(cfg)
    net = getattr(model, "model", model)
    net.to(DEVICE).eval()
    ck = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    sd = ck["model"] if "model" in ck else ck
    net.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in sd.items()})
    print(f"Loaded {ckpt_path}  (best_val_dice={ck.get('best_val_dice')}, epoch={ck.get('epoch')})")

    _, root, is_mri = _source_root(cfg)
    val_classes = resolve_classes(cfg.data.val_classes, root, is_mri=is_mri)
    loader = make_eval_loader(cfg, val_classes, split="val")
    n_target = int(cfg.get("probe", {}).get("n_samples", 120))

    rows, seen = [], 0
    agg = defaultdict(lambda: {"s": 0.0, "n": 0})   # (axis, mag) -> Dice
    pbar = tqdm(total=n_target, desc="drift probe")
    for batch in loader:
        B = batch["image"].shape[0]
        sp = batch.get("spacing")
        for b in range(B):
            if seen >= n_target:
                break
            img = batch["image"][b].to(DEVICE)          # (1,D,H,W)
            mask = batch["label"][b].to(DEVICE).float()  # (D,H,W)
            D = img.shape[-1]
            spacing = float(sp[b][0]) if sp is not None else None
            cls, subj = batch["label_names"][b], batch.get("subjects", [None] * B)[b]
            tgt_size = int((mask > 0.5).sum().item())
            ti = img.unsqueeze(0)                        # (1,1,D,H,W) target img
            for axis, mags in SWEEPS.items():
                for mag in mags:
                    ci, cm = _warp(img, mask, seed=seen, **_params(axis, mag, D))
                    pred = net.predict(ti, ci.view(1, 1, 1, *ci.shape[-3:]),
                                       cm.view(1, 1, *cm.shape[-3:]), spacing=spacing)
                    dv = _dice(pred.squeeze(), mask)
                    rows.append({"class": cls, "subject": subj, "tgt_size": tgt_size,
                                 "axis": axis, "mag": mag, "dice": round(dv, 4)})
                    a = agg[(axis, mag)]; a["s"] += dv; a["n"] += 1
            seen += 1
            pbar.update(1)
        if seen >= n_target:
            break
    pbar.close()

    out = Path("results/3d"); out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "exp40_context_drift.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    print(f"\nDice vs perturbation magnitude ({seen} target samples, context-only warp):")
    for axis, mags in SWEEPS.items():
        cells = "  ".join(f"{m}:{agg[(axis, m)]['s']/max(agg[(axis, m)]['n'],1):.3f}" for m in mags)
        print(f"  {axis:<14} {cells}")
    print(f"\n(mag 0 / scale 1.0 == self-context ceiling sanity)\nWrote {csv_path}")


if __name__ == "__main__":
    main()
