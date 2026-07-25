"""Driver: load a PatchSet3D checkpoint, sweep (tier x resolution) over the shared 3D
eval loader, and write a tidy per-(task,tier,res) CSV of matching metrics + real Dice.

    python experiments/3d/feature_sim/run.py eval.checkpoint=results/.../best.pt \
        eval.model=patchset3d
"""
import csv
import sys
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "3d"))      # common / eval / evaluate

from common import DEVICE, make_eval_loader, _source_root          # noqa: E402
from data.totalseg_classes import resolve_classes                  # noqa: E402
from feature_sim.adapters import PatchSet3DEncoderAdapter          # noqa: E402
from feature_sim.labels import grid_labels, sample_points          # noqa: E402
from feature_sim.metrics import (                                  # noqa: E402
    prototype_cosine, fg_match_margin, retrieval_at1)


def plan_sweep(tiers, resolutions, budget, R):
    rows, seen = [], set()
    for tier in tiers:
        if tier == "transformer_q":
            key = (tier, R, "dense")
            if key not in seen:
                seen.add(key); rows.append({"tier": tier, "res": R, "mode": "dense"})
            continue
        for res in resolutions:
            mode = "point" if res ** 3 > budget else "dense"
            key = (tier, res, mode)
            if key not in seen:
                seen.add(key); rows.append({"tier": tier, "res": res, "mode": mode})
    return rows


def _load_patchset(cfg):
    """Rebuild PatchSet3D from the checkpoint's stored arch (mirrors eval.py:55-83)."""
    from train import build_model
    ckpt = torch.load(cfg.eval.checkpoint, map_location=DEVICE, weights_only=False)
    from omegaconf import open_dict
    with open_dict(cfg):
        cfg.model = "patchset3d"
        if ckpt.get("arch") is not None:
            cfg.arch = OmegaConf.create(ckpt["arch"])
        elif "arch" not in cfg:
            raise ValueError(
                "checkpoint has no stored arch (older run); re-supply the training "
                "arch, e.g. +model=patchset3d arch.l=2")
    model, _ = build_model(cfg)
    model = model.to(DEVICE)
    sd = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()}
    model.load_state_dict(sd)
    return model.eval()


def _rows_for_task(adapter, model, item, cfg, plan, input_res, gen):
    """One task (single batch index already unbatched to B=1 tensors). Yields dict rows."""
    fs = cfg.feature_sim
    image = item["image"].to(DEVICE)              # (1,1,D,H,W)
    cin = item["context_in"].to(DEVICE)           # (1,K,1,D,H,W)
    cout = item["context_out"].to(DEVICE)         # (1,K,D,H,W)
    gt = item["label"][0]                         # (D,H,W)
    K = cin.shape[1]
    cls = item["label_names"][0]
    obj_vox = int((gt > 0).sum().item())
    with torch.no_grad():
        real = model.predict(image, cin, cout)   # cin already (1,K,1,D,H,W) -> (1,D,H,W)
    inter = (real[0] * (gt.to(DEVICE) > 0)).sum().item()
    den = real[0].sum().item() + (gt > 0).sum().item()
    real_dice = (2 * inter) / den if den > 0 else 0.0

    ctx_imgs = cin[0].squeeze(1)                  # (K,D,H,W)
    for p in plan:
        tier, res, mode = p["tier"], p["res"], p["mode"]
        if tier == "transformer_q":
            q = adapter.transformer_query(image, cin, cout)[0]          # (N,e)
            tl = grid_labels(gt, adapter.R).flatten()
            cl = torch.stack([grid_labels(cout[0, k], adapter.R).flatten()
                              for k in range(K)]).flatten()
            # context side uses img_embed tier (concat→e projection) so its channel dim
            # matches the transformer query rep e; note this is an approximate ceiling
            # reference (post-transformer target vs pre-transformer context embeddings)
            cf = adapter.features(ctx_imgs.unsqueeze(1), "img_embed", adapter.R)
            cf = cf.flatten(2).transpose(1, 2).reshape(-1, cf.shape[1])
            yield _metric_row(cls, obj_vox, real_dice, tier, res, mode,
                              adapter.native_res("concat", input_res),
                              q.cpu(), tl, cf.cpu(), cl, K)
            continue
        if mode == "dense":
            tf = adapter.features(image, tier, res)[0]                  # (C,res,res,res)
            tf = tf.flatten(1).transpose(0, 1)                         # (res^3, C)
            tl = grid_labels(gt, res).flatten()
            cvol = adapter.features(ctx_imgs.unsqueeze(1), tier, res)  # (K,C,res^3...)
            cf = cvol.flatten(2).transpose(1, 2).reshape(-1, cvol.shape[1])
            cl = torch.stack([grid_labels(cout[0, k], res).flatten()
                              for k in range(K)]).flatten()
        else:
            tcoords, tl = sample_points(gt, fs.n_fg, fs.n_bg,
                                        band=fs.get("band"), generator=gen)
            tf = adapter.sample_features(image, tier, tcoords.to(DEVICE).unsqueeze(0))[0]
            cfs, ctx_labels = [], []
            for k in range(K):
                cc, ll = sample_points(cout[0, k].cpu(), fs.n_fg, fs.n_bg,
                                       band=fs.get("band"), generator=gen)
                cfs.append(adapter.sample_features(
                    ctx_imgs[k][None, None], tier, cc.to(DEVICE).unsqueeze(0))[0])
                ctx_labels.append(ll)
            cf = torch.cat(cfs, 0); cl = torch.cat(ctx_labels, 0); tf = tf
        yield _metric_row(cls, obj_vox, real_dice, tier, res, mode,
                          adapter.native_res(tier, input_res),
                          tf.cpu(), tl, cf.cpu(), cl, K)


def _metric_row(cls, obj_vox, real_dice, tier, res, mode, tier_native,
                tf, tl, cf, cl, K):
    proto = prototype_cosine(tf, tl, cf, cl, mode=mode)
    row = {"class": cls, "obj_vox": obj_vox, "real_dice": real_dice,
           "tier": tier, "res": res, "mode": mode, "tier_native_res": tier_native,
           "K": K, "auroc": proto["auroc"],
           "soft_dice": proto.get("soft_dice", ""), "ap": proto.get("ap", ""),
           "margin": fg_match_margin(tf, tl, cf, cl),
           "retrieval_at1": retrieval_at1(tf, tl, cf, cl)}
    return row


@hydra.main(config_path="../../../configs/experiment/3d",
            config_name="feature_sim", version_base="1.3")
def main(cfg: DictConfig) -> None:
    torch.manual_seed(cfg.eval.seed)
    _, root, is_mri = _source_root(cfg)
    if not cfg.eval.get("checkpoint"):
        raise ValueError("eval.checkpoint is required (path to a trained PatchSet3D best.pt)")
    val_classes = resolve_classes(cfg.data.val_classes, root, is_mri=is_mri)
    loader = make_eval_loader(cfg, val_classes, split=cfg.eval.split)
    model = _load_patchset(cfg)
    adapter = PatchSet3DEncoderAdapter(model)
    input_res = int(cfg.data.image_size[-1])
    tiers = list(cfg.feature_sim.tiers)
    plan = plan_sweep(tiers, list(cfg.feature_sim.resolutions),
                      int(cfg.feature_sim.budget), adapter.R)
    gen = torch.Generator().manual_seed(cfg.eval.seed)

    out_dir = Path(cfg.eval.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "feature_sim.csv"
    n = 0
    with open(csv_path, "w", newline="") as fh:
        writer = None
        for batch in loader:
            B = batch["image"].shape[0]
            for b in range(B):
                item = {k: (v[b:b + 1] if torch.is_tensor(v) else [v[b]])
                        for k, v in batch.items()}
                for row in _rows_for_task(adapter, model, item, cfg, plan,
                                          input_res, gen):
                    if writer is None:
                        writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
                        writer.writeheader()
                    writer.writerow(row); n += 1
            if n and n % 200 == 0:
                print(f"  wrote {n} rows...")
    print(f"Done. {n} rows -> {csv_path}")


if __name__ == "__main__":
    main()
