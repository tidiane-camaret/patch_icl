"""Diagnose scatter-refinement sampling on a trained PatchSetCNN's own coarse prediction.

Unlike plot_sampling.py (which samples on MedSegBench GT / a frozen stage-1 ImagePFN),
this loads a `refine_mode="scatter"` PatchSetCNN checkpoint, runs its COARSE pass on the
dataset it was trained on (e.g. omnisynth_medseg), and studies how `sample_patches` params
shape the query sampling over that real coarse map.

Two outputs:
  1. A quantitative sweep (printed): for each param config, averaged over the val pool —
       core@uncertainty : of core cells, % with coarse prob in [0.3,0.7]  (uncertainty focus)
       GT-boundary recall: of true fractional-GT cells (0<gt<1), % sampled
       cluster           : mean fraction of a sampled cell's 4-neighbors also sampled (adjacency)
       coverage          : fraction of the Rf grid sampled (exploration)
  2. A figure: diverse samples (one per source) × key configs, cells tier-colored
     (boundary-core red, fg-core orange, neighbor cyan) over the image with GT (lime) and
     the coarse 0.5 contour (magenta dashed) — so you see clustering vs exploring directly.

Usage:
    python experiments/2d/multilevel/plot_scatter_sampling.py \
        --checkpoint /path/to/scatter/best.pt --n_total 400 --out results/2d/scatter_sampling.png
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, open_dict

_ROOT = str(Path(__file__).resolve().parents[3])
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "experiments", "2d"))
from common import DEVICE, build_loader                       # noqa: E402
from eval_incontext import _load_model                        # noqa: E402
from src.models.scatter_sampling import idx_to_ij, sample_patches   # noqa: E402


@torch.no_grad()
def coarse_prob(model, batch):
    """PatchSetCNN coarse pass -> (B,1,T,T) sigmoid prob (the refine sampling map source)."""
    with torch.autocast(device_type=DEVICE.type, dtype=torch.bfloat16, enabled=DEVICE.type == "cuda"):
        c = model._segment(batch["image"].to(DEVICE), batch["context_in"].to(DEVICE),
                           batch["context_out"].to(DEVICE)).float()
    return torch.sigmoid(c)


def _sample(qmap, cp, Rf):
    return sample_patches(qmap, cp["n_total"], cp["tau"], cp["blur_sigma"], cp["floor"], Rf,
                          temperature=cp["temperature"], stochastic=True,
                          n_fg_core=cp["n_fg_core"], n_boundary_core=cp["n_boundary_core"])


def behavior_metrics(qmap, gt, cp, Rf):
    """qmap,gt (B,N). Returns {core_unc, bnd_rec, clus, cover} averaged over the batch."""
    idx, is_core, _ = _sample(qmap, cp, Rf)
    B, N = qmap.shape
    sel = torch.zeros(B, N, dtype=torch.bool).scatter_(1, idx, True)
    core = torch.zeros(B, N, dtype=torch.bool).scatter_(1, idx, is_core)
    unc = (qmap > 0.3) & (qmap < 0.7)
    bnd = (gt > 0) & (gt < 1)
    s2 = sel.reshape(B, 1, Rf, Rf).float()
    adj = torch.tensor([[[[0., 1., 0.], [1., 0., 1.], [0., 1., 0.]]]])
    nbrsum = F.conv2d(s2, adj, padding=1).reshape(B, N)
    return dict(
        core_unc=float((core & unc).sum()) / max(int(core.sum()), 1),
        bnd_rec=float((sel & bnd).sum()) / max(int(bnd.sum()), 1),
        clus=float((nbrsum * sel.float()).sum() / (4 * max(int(sel.sum()), 1))),
        cover=float(sel.float().mean()))


def build_configs(base):
    """Baseline + one-knob variants for the sweep."""
    return [("baseline", base),
            ("blur_sigma=1", {**base, "blur_sigma": 1.0}),
            ("blur_sigma=6", {**base, "blur_sigma": 6.0}),
            ("floor=.001", {**base, "floor": 0.001}),
            ("floor=.05", {**base, "floor": 0.05}),
            ("temp=.3", {**base, "temperature": 0.3}),
            ("temp=2", {**base, "temperature": 2.0}),
            ("n_boundary_core=48", {**base, "n_boundary_core": 48}),
            ("n_fg_core=128", {**base, "n_fg_core": 128}),
            ("n_total=1024", {**base, "n_total": 1024})]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="scatter PatchSetCNN best.pt")
    ap.add_argument("--split", default="val")
    ap.add_argument("--n_total", type=int, default=400, help="budget M for the sweep baseline + figure")
    ap.add_argument("--tau", type=float, default=0.30)
    ap.add_argument("--blur_sigma", type=float, default=2.0)
    ap.add_argument("--floor", type=float, default=0.005)
    ap.add_argument("--n_fg_core", type=int, default=48)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--n_boundary_core", type=int, default=0)
    ap.add_argument("--sweep_pool", type=int, default=600, help="# val samples pooled for the sweep table")
    ap.add_argument("--sources", nargs="*",
                    default=["drive", "chasedb1", "nuclei", "m2caiseg", "abdomenus",
                             "promise12", "mosmedplus", "dynamicnuclear"],
                    help="omnisynth sources to show as figure rows (first 6 found)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/2d/multilevel/scatter_sampling.png")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    assert ck.get("arch", {}).get("refine_mode") == "scatter", \
        "checkpoint is not a scatter model (arch.refine_mode != 'scatter')"
    model, _ = _load_model(ck)
    model = model.to(DEVICE).eval()
    Rf = int(ck["arch"]["resolutions"][-1])
    T = int(ck["arch"]["resolution"])
    print(f"scatter PatchSetCNN: T={T} Rf={Rf} device={DEVICE} (epoch {ck.get('epoch')}, "
          f"val {ck.get('best_val_dice', float('nan')):.4f})")

    # Compose the eval config for the checkpoint's training source (mirror eval_incontext.py:
    # reconstruct cfg.synth from the stored block so the omnisynth generator reproduces).
    with initialize_config_dir(config_dir=os.path.join(_ROOT, "configs", "experiment", "2d"),
                               version_base=None):
        cfg = compose(config_name="eval_incontext",
                      overrides=[f"eval.checkpoint={args.checkpoint}",
                                 f"data.source={(ck.get('data') or {}).get('source', 'omnisynth')}",
                                 f"data.split={args.split}", "eval.batch_size=16", "eval.workers=6"])
    with open_dict(cfg):
        cfg.data.image_size = ck["image_size"]
        cfg.data.context_size = ck.get("context_size", 1)
        if ck.get("synth") is not None:
            cfg.synth = OmegaConf.create(ck["synth"])
    loader = build_loader(cfg)

    # Collect a diverse sample per source + a pool of coarse maps for the sweep.
    picked, Qs, Gs, seen = {}, [], [], 0
    for batch in loader:
        cp = coarse_prob(model, batch)
        qm = F.interpolate(cp, size=(Rf, Rf), mode="bilinear", align_corners=False)\
            .reshape(cp.shape[0], Rf * Rf).cpu()
        g64 = F.adaptive_avg_pool2d(batch["label"].float(), (Rf, Rf)).reshape(cp.shape[0], Rf * Rf)
        Qs.append(qm); Gs.append(g64)
        for b in range(cp.shape[0]):
            src = batch["dataset"][b].split("/")[-2] if "/" in batch["dataset"][b] else batch["dataset"][b]
            if src in args.sources and src not in picked:
                picked[src] = dict(img=batch["image"][b, 0].numpy(), lbl=batch["label"][b, 0].numpy(),
                                   q=qm[b].reshape(Rf, Rf).numpy(), cprob=cp[b:b + 1].cpu())
        seen += cp.shape[0]
        if len(picked) >= 6 and seen >= args.sweep_pool:
            break
    Q = torch.cat(Qs)[:args.sweep_pool]
    G = torch.cat(Gs)[:args.sweep_pool]
    print(f"pooled {Q.shape[0]} samples; sources shown: {list(picked)}")

    # ── quantitative sweep ──
    base = dict(n_total=args.n_total, tau=args.tau, blur_sigma=args.blur_sigma, floor=args.floor,
                n_fg_core=args.n_fg_core, temperature=args.temperature, n_boundary_core=args.n_boundary_core)
    print(f"\n{'config':>18} {'core@unc%':>9} {'GTbnd_rec%':>10} {'cluster':>8} {'coverage%':>9}")
    print("-" * 60)
    for cname, cp in build_configs(base):
        m = behavior_metrics(Q, G, cp, Rf)
        print(f"{cname:>18} {m['core_unc'] * 100:>8.0f}% {m['bnd_rec'] * 100:>9.0f}% "
              f"{m['clus']:>8.2f} {m['cover'] * 100:>8.0f}%")

    # ── qualitative figure: diverse samples × key configs ──
    show = [("baseline", base), ("blur_sigma=1", {**base, "blur_sigma": 1.0}),
            ("blur_sigma=6", {**base, "blur_sigma": 6.0}), ("floor=.05", {**base, "floor": 0.05}),
            ("temp=.3", {**base, "temperature": 0.3})]
    srcs = [s for s in args.sources if s in picked][:6]
    if not srcs:
        print("no requested sources present in this split; skipping figure.")
        return
    Himg = picked[srcs[0]]["img"].shape[0]
    fig, axes = plt.subplots(len(srcs), len(show), figsize=(2.8 * len(show), 2.8 * len(srcs)),
                             squeeze=False)
    for ci, (cname, cp) in enumerate(show):
        for r, src in enumerate(srcs):
            P = picked[src]; ax = axes[r][ci]
            idx, is_core, is_fg = _sample(torch.from_numpy(P["q"]).reshape(1, Rf * Rf), cp, Rf)
            ij = idx_to_ij(idx, Rf)[0]
            ax.imshow(P["img"], cmap="gray", vmin=0, vmax=1)
            if P["lbl"].max() > 0:
                ax.contour(P["lbl"], levels=[0.5], colors="lime", linewidths=1.0)
            cu = np.asarray(F.interpolate(P["cprob"], size=(Himg, Himg), mode="bilinear",
                                          align_corners=False)[0, 0])
            if cu.max() > 0.5 and cu.min() < 0.5:
                ax.contour(cu, levels=[0.5], colors="magenta", linewidths=0.9, linestyles="--")
            yy = (ij[:, 0].numpy() + 0.5) * (Himg / Rf)
            xx = (ij[:, 1].numpy() + 0.5) * (Himg / Rf)
            ic = is_core[0].numpy(); ifg = is_fg[0].numpy()
            ax.scatter(xx[~ic], yy[~ic], s=5, c="cyan", marker="s", edgecolors="none")
            ax.scatter(xx[ic & ~ifg], yy[ic & ~ifg], s=5, c="red", marker="s", edgecolors="none")
            ax.scatter(xx[ifg], yy[ifg], s=5, c="orange", marker="s", edgecolors="none")
            if ci == 0:
                ax.set_ylabel(src, fontsize=8)
            if r == 0:
                ax.set_title(cname, fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"scatter sampling on coarse pred | lime=GT  magenta=coarse.5  "
                 f"red=bnd-core orange=fg-core cyan=neigh  (n_total={args.n_total})", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=125, bbox_inches="tight")
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
