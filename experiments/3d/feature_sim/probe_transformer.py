"""Trace how target<->context correspondence evolves through the transformer, and settle
whether the transformer improves it (the transformer_query tier compared target POST- vs
context PRE-transformer, a mismatched probe). All at the transformer's resolution R, on the
SAME tasks, correlating each metric with the model's real Dice:

  pre      : img_embed@R                     (target & context PRE-transformer) -- baseline
  block_i  : transformer_pair after block i  (target & context POST block i, same space)

If the transformer helps correspondence, later blocks should predict Dice better than `pre`.

    python experiments/3d/feature_sim/probe_transformer.py \
        experiment=22_totalseg_train_test eval.checkpoint=/.../best.pt +limit=1500
"""
import math
import sys
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "3d"))

from common import DEVICE, make_eval_loader, _source_root              # noqa: E402
from data.totalseg_classes import resolve_classes                      # noqa: E402
from feature_sim.adapters import PatchSet3DEncoderAdapter              # noqa: E402
from feature_sim.labels import grid_labels                            # noqa: E402
from feature_sim.metrics import prototype_cosine, retrieval_at1        # noqa: E402
from feature_sim.run import _load_patchset, _avg_rank, _pearson        # noqa: E402


def _ok(v):
    return v is not None and not (isinstance(v, float) and math.isnan(v))


def _spear(xs, ys):
    pairs = [(x, y) for x, y in zip(xs, ys) if _ok(x) and _ok(y)]
    if len(pairs) < 3:
        return None
    x, y = zip(*pairs)
    return _pearson(_avg_rank(list(x)), _avg_rank(list(y)))


def _nanmean(xs):
    xs = [x for x in xs if _ok(x)]
    return sum(xs) / len(xs) if xs else float("nan")


@hydra.main(config_path="../../../configs/experiment/3d",
            config_name="feature_sim", version_base="1.3")
def main(cfg: DictConfig) -> None:
    torch.manual_seed(cfg.eval.seed)
    _, root, is_mri = _source_root(cfg)
    val_classes = resolve_classes(cfg.data.val_classes, root, is_mri=is_mri)
    loader = make_eval_loader(cfg, val_classes, split=cfg.eval.split)
    model = _load_patchset(cfg)
    ad = PatchSet3DEncoderAdapter(model)
    R = ad.R
    L = len(model.transformer.blocks)
    limit = int(cfg.get("limit", 2000))
    variants = ["pre"] + [f"block_{i+1}" for i in range(L)]
    print(f"R={R}  L={L} blocks  probing up to {limit} tasks on split={cfg.eval.split}\n")

    acc = {v: {"retr": [], "auroc": []} for v in variants}
    dice = []
    seen = 0
    for batch in loader:
        B = batch["image"].shape[0]
        # Per-batch crop spacing for the spacing-aware encoder's RoPE (None otherwise).
        spacing = (float(batch["spacing"][0, 0])
                   if getattr(model, "spacing_aware", False) and "spacing" in batch else None)
        sp = {"spacing": spacing} if spacing is not None else {}
        for b in range(B):
            image = batch["image"][b:b + 1].to(DEVICE)
            cin = batch["context_in"][b:b + 1].to(DEVICE)
            cout = batch["context_out"][b:b + 1].to(DEVICE)
            gt = batch["label"][b]
            K = cin.shape[1]
            with torch.no_grad():
                real = model.predict(image, cin, cout, **sp)[0]
            inter = (real * (gt.to(DEVICE) > 0)).sum().item()
            den = real.sum().item() + (gt > 0).sum().item()
            dice.append((2 * inter / den) if den > 0 else 0.0)

            tl = grid_labels(gt, R, threshold=None).flatten().to(DEVICE)
            cl = torch.stack([grid_labels(cout[0, k], R, threshold=None).flatten()
                              for k in range(K)]).flatten().to(DEVICE)

            # pre-transformer baseline: img_embed@R on both sides
            tf_ie = ad.features(image, "img_embed", R, **sp)[0].flatten(1).transpose(0, 1)  # (N,e)
            cvol = ad.features(cin[0].squeeze(1).unsqueeze(1), "img_embed", R, **sp)        # (K,e,R,R,R)
            ctx_ie = cvol.flatten(2).transpose(1, 2).reshape(-1, cvol.shape[1])         # (K*N,e)

            # per-block post-transformer target/context (same forward, hooked)
            layers = ad.transformer_pair_per_layer(image, cin, cout, **sp)              # [(tgt,ctx)]*L

            for name, (tf, cf) in zip(variants,
                                      [(tf_ie, ctx_ie)] + [(t[0], c[0]) for t, c in layers]):
                acc[name]["retr"].append(retrieval_at1(tf, tl, cf, cl))
                acc[name]["auroc"].append(prototype_cosine(tf, tl, cf, cl, "dense")["auroc"])
            seen += 1
            if seen >= limit:
                break
        if seen >= limit:
            break

    print(f"Probed {seen} tasks.  Spearman(metric, real_dice):\n")
    print(f"{'variant':<10} {'retr_mean':>9} {'auroc_mean':>10} {'sp(retr,dice)':>14} {'sp(auroc,dice)':>15}")
    for v in variants:
        print(f"{v:<10} {_nanmean(acc[v]['retr']):>9.3f} {_nanmean(acc[v]['auroc']):>10.3f} "
              f"{_spear(acc[v]['retr'], dice):>14.3f} {_spear(acc[v]['auroc'], dice):>15.3f}")


if __name__ == "__main__":
    main()
