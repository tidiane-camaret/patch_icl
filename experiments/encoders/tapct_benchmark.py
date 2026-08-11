"""Transfer-Dice of frozen tap-ct-b-3d features over a class set (Hydra-configurable).

Per class: encode each subject's organ-centred crop once (crop_jitter=0 -> deterministic),
then round-robin pair each subject (target) with the next K subjects (context) and score
1-NN label transfer (feature_sim.label_transfer). Reports per-class + per-category + macro.

All knobs live in configs/experiment/3d/encoders/tap_ct.yaml:
  data.*    crop geometry / class set / subject pool (crop_spacing_mm sets feature cell size)
  encoder.* precision, compile, to_lps reorientation, resize_native, pad_hu
  metric.*  hard 1-NN vs soft softmax(cos/tau) transfer
  out.*     csv path, optional wandb

  python experiments/encoders/tapct_benchmark.py
  python experiments/encoders/tapct_benchmark.py data.crop_spacing_mm=1.0 encoder.precision=fp32
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "3d"))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling tapct_* modules

from data.totalseg_classes import resolve_classes, category_map_ct  # noqa: E402
from src.totalseg_dataloader_incontext import TotalSegInContextDataset  # noqa: E402
from feature_sim.metrics import transfer_metrics  # noqa: E402
from tapct_features import load_model, make_processor, dense_features, occ_labels  # noqa: E402
from tapct_plot import transfer_pred_grid, pred_to_image, _reorient, plot_sample  # noqa: E402

METRIC_KEYS = ("soft_dice", "soft_precision", "soft_recall",
               "hard_dice", "hard_precision", "hard_recall", "retrieval_at1")


def build_dataset(cfg):
    d = cfg.data
    classes = resolve_classes(d.classes, totalseg_root=d.root)
    T = int(d.image_size)
    ds = TotalSegInContextDataset(
        root=d.root,
        classes=list(classes),
        image_size=(T, T, T),
        split=(d.split if d.split not in (None, "null") else None),
        context_size=int(d.context_size),
        max_subjects=int(d.n_subjects) if d.n_subjects is not None else None,
        use_crop=bool(d.use_crop),
        crop_spacing_mm=float(d.crop_spacing_mm),
        crop_jitter=int(d.crop_jitter),
        mask_downsample=d.mask_downsample,
        mask_occupancy_thr=float(d.mask_occupancy_thr),
        eval_seed=int(d.eval_seed),
    )
    return ds, list(classes), T


@hydra.main(config_path="../../configs/experiment/3d/encoders",
            config_name="tap_ct", version_base="1.3")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    e, m = cfg.encoder, cfg.metric
    device = torch.device("cuda")

    ds, classes, T = build_dataset(cfg)
    model = load_model(device, use_sdpa=True)
    if e.compile:
        model = torch.compile(model, mode=e.compile_mode)
    proc = make_processor(T)                 # resize_dims=(T,T) native
    if not e.resize_native:
        proc.resize_dims = (224, 224)        # stock in-plane upsample to 224^2

    enc_kw = dict(to_lps=bool(e.to_lps), precision=e.precision)
    n_plot = int(cfg.plot.n_per_task)
    plot_dir = Path(cfg.plot.dir)
    if n_plot > 0:
        plot_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for ci, cls in enumerate(classes):
        subs = ds.label_to_subjects.get(cls, [])
        if len(subs) < 2:
            print(f"[{ci+1}/{len(classes)}] {cls}: {len(subs)} subject(s) — skipped", flush=True)
            continue

        # Encode each subject-class crop once (features on GPU fp16 to bound memory).
        feats, labels, gdims = {}, {}, None
        for s in subs:
            try:
                img, msk = ds._load_crop(s, cls)
                tf, gdims = dense_features(model, proc, img, device, **enc_kw)
                feats[s] = tf.half().to(device)
                labels[s] = occ_labels(msk, gdims, to_lps=bool(e.to_lps)).to(device)
            except Exception as ex:
                print(f"    encode fail {s}/{cls}: {ex}", flush=True)
        subs = [s for s in subs if s in feats]
        if len(subs) < 2:
            continue

        K = min(int(cfg.data.context_size), len(subs) - 1)
        acc = defaultdict(list)
        n_plotted = 0
        for i, s in enumerate(subs):
            tf, tl = feats[s].float(), labels[s]
            if tl.sum() <= 0:
                continue
            ctx_s = [subs[(i + 1 + j) % len(subs)] for j in range(K)]  # next K subjects
            cf = torch.cat([feats[c].float() for c in ctx_s], dim=0)
            cl = torch.cat([labels[c] for c in ctx_s], dim=0)
            mt = transfer_metrics(tf, tl, cf, cl, thr=float(m.thr))
            if mt["soft_dice"] != mt["soft_dice"]:  # nan -> target had no FG
                continue
            for k in METRIC_KEYS:
                acc[k].append(mt[k])

            if n_plot > 0 and n_plotted < n_plot:
                try:
                    # Draw in the same frame the features live in: reoriented iff to_lps.
                    reo = _reorient if e.to_lps else (lambda v: v.squeeze().cpu().numpy())
                    ti, tm = ds._load_crop(s, cls)
                    ck = ctx_s[0]
                    ci_, cm = ds._load_crop(ck, cls)
                    pred = pred_to_image(transfer_pred_grid(tf, cf, cl, gdims), (T, T, T))
                    plot_sample(
                        plot_dir / f"{cls}__tgt_{s}__ctx_{ck}.png", cls, s, ck,
                        float(cfg.data.crop_spacing_mm), mt,
                        reo(ti), reo(tm), reo(ci_), reo(cm),
                        pred, thr=float(cfg.plot.thr))
                    n_plotted += 1
                except Exception as ex:
                    print(f"    plot fail {s}/{cls}: {ex}", flush=True)

        del feats, labels
        torch.cuda.empty_cache()
        if not acc["soft_dice"]:
            continue
        # nanmean: hard_* is nan for pairs whose coarse cells never reach thr (thin classes).
        row = {"class": cls, "category": category_map_ct.get(cls, "?"),
               "n_pairs": len(acc["soft_dice"])}
        for k in METRIC_KEYS:
            row[k] = float(np.nanmean(acc[k])) if len(acc[k]) else float("nan")
        row["hard_frac"] = float(np.mean([v == v for v in acc["hard_dice"]]))  # non-nan share
        rows.append(row)
        print(f"[{ci+1}/{len(classes)}] {cls:34} n={row['n_pairs']:2}  "
              f"soft_d={row['soft_dice']:.3f}  hard_d={row['hard_dice']:.3f}  "
              f"r@1={row['retrieval_at1']:.3f}", flush=True)

    if not rows:
        print("no classes scored (pool too small?)"); return

    out_csv = Path(cfg.out.csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    def _mac(rs, k):
        vals = [r[k] for r in rs if r[k] == r[k]]
        return float(np.mean(vals)) if vals else float("nan")

    print("\n===== per-category (macro over classes) =====")
    by_cat = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r)
    for cat in sorted(by_cat):
        rs = by_cat[cat]
        print(f"  {cat:32} classes={len(rs):2}  "
              f"soft_d={_mac(rs,'soft_dice'):.3f}  hard_d={_mac(rs,'hard_dice'):.3f}  "
              f"r@1={_mac(rs,'retrieval_at1'):.3f}")

    print(f"\n===== MACRO over {len(rows)} classes =====")
    print(f"  soft_dice={_mac(rows,'soft_dice'):.3f}  hard_dice={_mac(rows,'hard_dice'):.3f}  "
          f"retrieval_at1={_mac(rows,'retrieval_at1'):.3f}")
    print(f"  (hard_dice averaged over classes with any non-nan pair; see hard_frac column)")
    print(f"  CSV: {out_csv}")

    if cfg.out.wandb_project:
        import wandb
        wandb.init(project=cfg.out.wandb_project, name=cfg.out.wandb_name,
                   config=OmegaConf.to_container(cfg, resolve=True))
        tbl = wandb.Table(columns=list(rows[0].keys()), data=[list(r.values()) for r in rows])
        wandb.log({"per_class": tbl,
                   "macro/soft_dice": _mac(rows, "soft_dice"),
                   "macro/hard_dice": _mac(rows, "hard_dice"),
                   "macro/retrieval_at1": _mac(rows, "retrieval_at1")})
        wandb.finish()


if __name__ == "__main__":
    main()
