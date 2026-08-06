"""One-off: characterize SOFT vs HARD label transfer across temperature tau.

For each val/test task, extract (target,context) features for the headline encoder tier
(img_embed@48) and the transformer trace (encoder pre-attention + each block, res=R), then
compute label-transfer Dice for hard 1-NN and softmax(cos/tau) at a tau grid — reusing ONE
cosine matmul per (task,tier). Reports, per (tier,tau): Spearman(dice, real_dice),
size-partialled Spearman (control obj_vox), and Spearman(dice, obj_vox).

Question settled: does soft transfer (attention-like aggregation) ever beat hard on
size-partialled prediction, and at what tau does it collapse into an object-size proxy?

    .venv_thor/bin/python experiments/3d/feature_sim/tau_sweep.py \
        eval.checkpoint=/nfs/.../2026-07-25_usual-puddle-174/best.pt
"""
import sys
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "3d"))

from common import DEVICE, make_eval_loader, _source_root          # noqa: E402
from data.totalseg_classes import resolve_classes                  # noqa: E402
from feature_sim.adapters import PatchSet3DEncoderAdapter          # noqa: E402
from feature_sim.labels import grid_labels                         # noqa: E402
from feature_sim.metrics import l2norm                             # noqa: E402
from run import _load_patchset                                     # noqa: E402

TAUS = [0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
N_TASKS = 120
ENC_RES = 48


@torch.no_grad()
def transfer_sweep(tf, tl, cf, cl, taus, chunk=2048, eps=1e-8):
    """label-transfer Dice for hard 1-NN + each soft tau, sharing one cosine matmul."""
    g = tl.float().clamp(0, 1)
    keys = ["hard"] + [f"tau{t}" for t in taus]
    if g.sum() <= 0:
        return {k: float("nan") for k in keys}
    tn, cn, clf = l2norm(tf), l2norm(cf), cl.float().clamp(0, 1)
    preds = {k: tf.new_zeros(tn.shape[0]) for k in keys}
    for s in range(0, tn.shape[0], chunk):
        sim = tn[s:s + chunk] @ cn.T
        preds["hard"][s:s + chunk] = clf[sim.argmax(1)]
        for t in taus:
            preds[f"tau{t}"][s:s + chunk] = torch.softmax(sim / t, dim=1) @ clf
    out = {}
    for k, p in preds.items():
        inter = (p * g).sum()
        out[k] = (2 * inter / (p.sum() + g.sum() + eps)).item()
    return out


def _spear(x, y):
    import numpy as np
    from scipy.stats import spearmanr
    m = np.isfinite(x) & np.isfinite(y)
    return spearmanr(x[m], y[m]).statistic if m.sum() > 5 else float("nan")


def _partial(dice, real, size):
    import numpy as np
    import pandas as pd
    d = pd.DataFrame({"d": dice, "r": real, "z": size}).dropna()
    if len(d) < 10:
        return float("nan")
    rz = d["z"].rank().values

    def resid(v):
        v = v.rank().values
        b = np.polyfit(rz, v, 1)
        return v - (b[0] * rz + b[1])
    return np.corrcoef(resid(d["d"]), resid(d["r"]))[0, 1]


@hydra.main(config_path="../../../configs/experiment/3d",
            config_name="feature_sim", version_base="1.3")
def main(cfg: DictConfig) -> None:
    torch.manual_seed(cfg.eval.seed)
    _, root, is_mri = _source_root(cfg)
    val_classes = resolve_classes(cfg.data.val_classes, root, is_mri=is_mri)
    loader = make_eval_loader(cfg, val_classes, split=cfg.eval.split)
    model = _load_patchset(cfg)
    adapter = PatchSet3DEncoderAdapter(model)
    R = adapter.R

    rows = []          # (tier, key, real_dice, obj_vox, dice)
    seen = 0
    for batch in loader:
        # Per-batch crop spacing for the spacing-aware encoder's RoPE (None otherwise).
        spacing = (float(batch["spacing"][0, 0])
                   if getattr(model, "spacing_aware", False) and "spacing" in batch else None)
        sp = {"spacing": spacing} if spacing is not None else {}
        for b in range(batch["image"].shape[0]):
            if seen >= N_TASKS:
                break
            image = batch["image"][b:b + 1].to(DEVICE)
            cin = batch["context_in"][b:b + 1].to(DEVICE)
            cout = batch["context_out"][b:b + 1].to(DEVICE)
            gt = batch["label"][b]
            K = cin.shape[1]
            obj_vox = int((gt > 0).sum().item())
            with torch.no_grad():
                real = model.predict(image, cin, cout, **sp)
            inter = (real[0] * (gt.to(DEVICE) > 0)).sum().item()
            den = real[0].sum().item() + (gt > 0).sum().item()
            real_dice = (2 * inter) / den if den > 0 else 0.0

            # encoder tier img_embed@ENC_RES (dense full volume)
            tf = adapter.features(image, "img_embed", ENC_RES, **sp)[0].flatten(1).transpose(0, 1)
            tl = grid_labels(gt, ENC_RES, threshold=None).flatten().to(DEVICE)
            ctx = cin[0].squeeze(1)                       # (K,D,H,W)
            cvol = adapter.features(ctx.unsqueeze(1), "img_embed", ENC_RES, **sp)
            cf = cvol.flatten(2).transpose(1, 2).reshape(-1, cvol.shape[1])
            cl = torch.stack([grid_labels(cout[0, k], ENC_RES, threshold=None).flatten()
                              for k in range(K)]).flatten().to(DEVICE)
            for key, dice in transfer_sweep(tf, tl, cf, cl, TAUS).items():
                rows.append((f"img_embed@{ENC_RES}", key, real_dice, obj_vox, dice))

            # transformer trace (transformer_input pre-attention + each block) at res=R
            tlR = grid_labels(gt, R, threshold=None).flatten().to(DEVICE)
            clR = torch.stack([grid_labels(cout[0, k], R, threshold=None).flatten()
                               for k in range(K)]).flatten().to(DEVICE)
            for name, tq, cq in adapter.transformer_trace(image, cin, cout, **sp):
                for key, dice in transfer_sweep(tq[0], tlR, cq[0], clR, TAUS).items():
                    rows.append((name, key, real_dice, obj_vox, dice))
            seen += 1
        if seen >= N_TASKS:
            break
    print(f"\n=== tau sweep on {seen} tasks | taus={TAUS} ===")

    import numpy as np
    import pandas as pd
    df = pd.DataFrame(rows, columns=["tier", "key", "real_dice", "obj_vox", "dice"])
    for tier in df.tier.unique():
        sub = df[df.tier == tier]
        print(f"\n--- {tier} ---")
        print(f"{'variant':10s} {'rho(dice,real)':>15s} {'partial|size':>13s} {'rho(dice,size)':>15s} {'dice_mean':>10s}")
        for key in ["hard"] + [f"tau{t}" for t in TAUS]:
            g = sub[sub.key == key]
            d, r, z = g.dice.values, g.real_dice.values, g.obj_vox.values.astype(float)
            print(f"{key:10s} {_spear(d, r):>15.3f} {_partial(d, r, z):>13.3f} "
                  f"{_spear(d, z):>15.3f} {np.nanmean(d):>10.3f}")


if __name__ == "__main__":
    main()
