"""
Visualize / measure the proposed multilevel patch-sampling procedure (threshold
boundary core + fg-core quota + blurred-field neighbor fill) on real MedSegBench.

Sampling-map source (--source):
  ds_gt     — GT pooled directly to grid_res (res-32): genuine target-res detail.
  prev_pred — real frozen stage-1 prediction at its native res-16, upsampled to
              grid_res: carries only res-16 information (the realistic, deployable
              regime). fg/bg classification in --stats always uses the TRUE GT.

For each sample, plots two panels:
  left  — native image with GT overlay
  right — res-32 grid: boundary core (red), fg core (orange), neighbor fill (cyan)
          over the TRUE gt32 (green 0.5 contour); for prev_pred the sampling map's
          0.5 contour is drawn dashed magenta to expose the res-16 vs res-32 mismatch.

The sampler here mirrors docs/superpowers/specs/2026-06-16-multilevel-patch-sampling-design.md.

Usage:
    python experiments/2d/multilevel/plot_sampling.py --dataset busi --n_images 6
    python experiments/2d/multilevel/plot_sampling.py --source prev_pred --n_fg_core 48
    python experiments/2d/multilevel/plot_sampling.py --stats --source prev_pred
"""

import argparse
import glob
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

_ROOT = str(Path(__file__).resolve().parents[3])
sys.path.insert(0, _ROOT)
from src.datasets.medsegbench import DATA_ROOT, MedSegBenchDataset


# ---------------------------------------------------------------------------
# Proposed sampler (mirrors the design spec — inlined for the diagnostic)
# ---------------------------------------------------------------------------

def gaussian_blur(x_flat: torch.Tensor, grid_res: int, sigma: float) -> torch.Tensor:
    """(B, N) → (B, N) separable Gaussian blur on the grid_res×grid_res grid."""
    B, N = x_flat.shape
    x = x_flat.reshape(B, 1, grid_res, grid_res)
    k = int(2 * np.ceil(2 * sigma) + 1)
    coords = torch.arange(k, dtype=torch.float32, device=x.device) - (k - 1) / 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = (g / g.sum()).to(x.dtype)
    pad = k // 2
    x = F.conv2d(F.pad(x, (pad, pad, 0, 0), mode="reflect"), g.view(1, 1, 1, k))
    x = F.conv2d(F.pad(x, (0, 0, pad, pad), mode="reflect"), g.view(1, 1, k, 1))
    return x.reshape(B, N)


def sample_patches(values, n_total, tau, blur_sigma, floor, grid_res,
                   temperature=1.0, stochastic=True, n_fg_core=0, boundary_tier=True,
                   n_boundary_core=0):
    """values: (B, N) in [0,1]. Returns (idx, is_core, is_fg_core), each (B, n_total).

    Core priority tiers (all above the neighbor tier, so a single top-k selects them):
      1. boundary core : cells with |value-0.5| < tau (ranked by closeness to 0.5).
                         Disabled when boundary_tier=False (tau→0, no boundary core).
                         When n_boundary_core>0, the tau band is CAPPED to the
                         n_boundary_core cells closest to 0.5 — a distribution-invariant
                         quota so the boundary budget no longer balloons on thin-structure
                         datasets (tau still gates out pure 0/1 cells).
      2. fg core       : a fixed quota of n_fg_core foreground cells (value>=0.5) chosen
                         uniformly at random, to cover object interiors the boundary
                         misses.
    The remaining budget is filled by cells sampled near BOTH core tiers via a blurred
    proximity field + Gumbel-top-k (uniform `floor` keeps far cells in play): the field
    diffuses from boundary core ∪ fg core, so foreground neighbors also fill the budget.
    """
    d = (values - 0.5).abs()
    core_b = (d < tau) if boundary_tier else torch.zeros_like(values, dtype=torch.bool)
    if boundary_tier and n_boundary_core > 0:
        # Cap the tau band at the n_boundary_core cells closest to 0.5 (per row).
        masked_d = torch.where(core_b, d, torch.full_like(d, 2.0))   # non-core → large
        keep = masked_d.topk(min(n_boundary_core, d.shape[1]), dim=1, largest=False).indices
        core_b = torch.zeros_like(core_b).scatter_(1, keep, True) & core_b   # guard: real core only

    # ── Tier 2: forced foreground quota (random fg, excluding boundary core) ──
    fg_core = torch.zeros_like(core_b)
    if n_fg_core > 0:
        fg_pool = (values >= 0.5) & ~core_b
        key = torch.where(fg_pool, torch.rand_like(values), values.new_full((), -1.0))
        take = key.topk(n_fg_core, dim=1).indices
        fg_core = torch.zeros_like(core_b).scatter_(1, take, True) & fg_pool  # guard: <n_fg_core fg

    # ── Neighbor proximity field (around boundary core ∪ fg core) ──
    g = gaussian_blur((core_b | fg_core).float(), grid_res, blur_sigma)
    w = g + floor
    if stochastic:
        u = torch.rand_like(w).clamp(1e-6, 1 - 1e-6)
        gumbel = -torch.log(-torch.log(u))
        neigh_score = (w + 1e-12).log() + temperature * gumbel
    else:
        neigh_score = (w + 1e-12).log()

    BIG_B, BIG_F = 2e4, 1e4                             # boundary > fg core > neighbors
    score = torch.where(core_b, BIG_B - d,
            torch.where(fg_core, BIG_F, neigh_score))
    idx = score.topk(n_total, dim=1).indices
    is_fg_core = fg_core.gather(1, idx)
    is_core    = (core_b | fg_core).gather(1, idx)
    return idx, is_core, is_fg_core


# ---------------------------------------------------------------------------
# Sampling-map sources:  ds_gt (GT pooled to grid_res) vs prev_pred (real
# stage-1 prediction at its native res, upsampled to grid_res).
# ---------------------------------------------------------------------------

def load_stage1(checkpoint, device):
    """Load the frozen stage-1 ImagePFN (arch read from the .pt)."""
    from src.models.pfn_seg_2d import ImagePFN
    from src.models.pretrained_encoders import UniverSegFeatureEncoder
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    arch, img_size = ckpt["arch"], ckpt["image_size"]
    resolution = arch["resolution"]
    input_patch_size = arch.get("input_patch_size", img_size // resolution)
    image_encoder, feature_dim = None, None
    if arch.get("image_encoder", "patch") == "universeg":
        image_encoder = UniverSegFeatureEncoder(
            level=arch.get("feature_level", "all"), input_size=128,
            resize_to_input=arch.get("encoder_resize_to_input", False)).to(device)
        feature_dim = image_encoder.feature_dim
    model = ImagePFN(resolution=resolution, image_size=img_size,
                     input_patch_size=input_patch_size,
                     image_encoder=image_encoder, feature_dim=feature_dim,
                     e=arch["e"], h=arch["h"], l=arch["l"], a=arch["a"],
                     thinking_rows=arch["thinking_rows"],
                     residual_decay=arch["residual_decay"]).to(device)
    model.load_state_dict({k.removeprefix("_orig_mod."): v for k, v in ckpt["model"].items()})
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    print(f"Stage-1 loaded: resolution={resolution}, encoder={arch.get('image_encoder')}  "
          f"(prev_pred sampled at res-{resolution} → upsampled to grid)")
    return model


@torch.no_grad()
def stage1_coarse(stage1, image, ctx_in, ctx_out, grid_res, device):
    """image (n,1,H,W), ctx_* (n,K,1,H,W) → coarse pred upsampled to (n, grid_res, grid_res).

    Mirrors pipeline.coarse_predict: stage-1 logits live at its native resolution
    (res-16 here); bilinear-upsampled to grid_res so the sampling map carries only
    res-16 information — the key asymmetry vs the res-32 GT map.
    """
    all_images = torch.cat([ctx_in, image.unsqueeze(1)], dim=1).to(device)        # (n,T,1,H,W)
    all_masks  = torch.cat([ctx_out, torch.zeros_like(image.unsqueeze(1))], dim=1).to(device)
    K = ctx_in.shape[1]
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        logits = stage1(all_images, all_masks, sep=K)                              # (n, R1, R1)
    p = torch.sigmoid(logits.float())
    p = F.interpolate(p.unsqueeze(1), size=(grid_res, grid_res),
                      mode="bilinear", align_corners=False).squeeze(1)
    return p


def build_maps(ds, indices, args, stage1, device):
    """Returns (gt32, values), each (n, grid_res²).

    gt32   — GT pooled to grid_res; the truth, always used for fg/bg classification.
    values — the sampling map fed to sample_patches: gt32 for ds_gt, else the stage-1
             prediction (res-16 upsampled to grid_res) for prev_pred.
    """
    R = args.grid_res
    if args.source == "ds_gt":
        gts = []
        for i in indices:
            dsn, sidx, lv = ds.samples[i]
            mask = torch.from_numpy((ds.labels[dsn][sidx] == lv).astype("float32"))
            gts.append(F.adaptive_avg_pool2d(mask[None, None], (R, R))[0, 0].reshape(-1))
        gt32 = torch.stack(gts)
        return gt32, gt32

    # prev_pred: gather image + context, run the real stage-1.
    imgs, cin, cout, labs = [], [], [], []
    for i in indices:
        item = ds[i]
        imgs.append(item["image"]); cin.append(item["context_in"])
        cout.append(item["context_out"]); labs.append(item["label"][0])
    image, ctx_in, ctx_out = torch.stack(imgs), torch.stack(cin), torch.stack(cout)
    gt32 = torch.stack([F.adaptive_avg_pool2d(l[None, None].float(), (R, R))[0, 0].reshape(-1)
                        for l in labs])
    vals = []
    for s in range(0, image.shape[0], args.s1_chunk):
        sl = slice(s, s + args.s1_chunk)
        p = stage1_coarse(stage1, image[sl], ctx_in[sl], ctx_out[sl], R, device)
        vals.append(p.reshape(p.shape[0], -1).cpu())
    return gt32, torch.cat(vals)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_sample(ax_img, ax_grid, image, label, gt32, idx, is_core, is_fg_core,
                grid_res, title, samp_map=None):
    """image/label: (H,W) np; gt32/samp_map: (R,R) np; idx/is_core/is_fg_core: (M,) np.

    Background is always the TRUE gt32 (green 0.5 contour). When samp_map differs
    (prev_pred), its 0.5 contour is drawn in magenta to expose the res-16 vs res-32
    boundary mismatch that drives the sampling.
    """
    # ---- left: image + GT overlay ----
    ax_img.imshow(image, cmap="gray")
    ax_img.contour(label, levels=[0.5], colors="lime", linewidths=1.2)
    ax_img.imshow(np.ma.masked_where(label < 0.5, label), cmap="autumn", alpha=0.25)
    ax_img.set_title(title, fontsize=9)
    ax_img.axis("off")

    # ---- right: sampled patches over the TRUE gt32 ----
    rows = idx // grid_res
    cols = idx % grid_res
    bcore = is_core & ~is_fg_core                       # boundary core
    nb    = ~is_core                                    # neighbor fill

    ax_grid.imshow(gt32, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax_grid.contour(gt32, levels=[0.5], colors="lime", linewidths=1.0)
    if samp_map is not None:
        ax_grid.contour(samp_map, levels=[0.5], colors="magenta", linewidths=1.0, linestyles="--")
    ax_grid.scatter(cols[nb], rows[nb], s=22, c="cyan", marker="s",
                    label=f"neighbor ({int(nb.sum())})", edgecolors="none")
    ax_grid.scatter(cols[bcore], rows[bcore], s=22, c="red", marker="s",
                    label=f"boundary core ({int(bcore.sum())})", edgecolors="none")
    ax_grid.scatter(cols[is_fg_core], rows[is_fg_core], s=22, c="orange", marker="s",
                    label=f"fg core ({int(is_fg_core.sum())})", edgecolors="none")
    ax_grid.set_xlim(-0.5, grid_res - 0.5)
    ax_grid.set_ylim(grid_res - 0.5, -0.5)
    ax_grid.set_title(f"res-{grid_res} sampling", fontsize=9)
    ax_grid.legend(loc="upper right", fontsize=6, framealpha=0.6)
    ax_grid.set_xticks([]); ax_grid.set_yticks([])


def compute_stats(ds, args, stage1, device):
    """Per-dataset sampling stats over the val set, pooled across images.

    A patch is foreground if its avg-pooled GT fraction >= 0.5 (majority vote) — always
    from the TRUE GT, regardless of the sampling-map source. Two views:
      (A) selection composition — of the selected core / neighbor patches, what % are fg.
      (B) GT coverage           — of all GT fg (and bg) patches, what % are selected as
                                  core / neighbor / unsampled.
    """
    R = args.grid_res
    by_ds: dict[str, list[int]] = defaultdict(list)
    for i, (dsn, _, _) in enumerate(ds.samples):
        by_ds[dsn].append(i)

    # Per-dataset accumulators.
    A: dict[str, dict] = {}   # composition
    Bc: dict[str, dict] = {}  # coverage
    per_img = {}
    for dsn in sorted(by_ds):
        gt32, values = build_maps(ds, by_ds[dsn], args, stage1, device)   # both (n, N)
        n, N = values.shape
        idx, is_core, _ = sample_patches(values, args.n_total, args.tau, args.sigma,
                                         args.floor, R, args.temperature, stochastic=True,
                                         n_fg_core=args.n_fg_core,
                                         boundary_tier=not args.no_boundary,
                                         n_boundary_core=args.n_boundary_core)

        # Full-grid selection masks (n, N); fg/bg/boundary from the TRUE gt32.
        sel  = torch.zeros(n, N, dtype=torch.bool).scatter_(1, idx, True)
        core = torch.zeros(n, N, dtype=torch.bool).scatter_(1, idx, is_core)
        nb   = sel & ~core
        fg   = gt32 >= 0.5
        bg   = ~fg
        bnd  = (gt32 > 0) & (gt32 < 1)                  # true boundary: fractional occupancy

        A[dsn] = {
            "core_fg": int((fg & core).sum()), "core": int(core.sum()),
            "nb_fg":   int((fg & nb).sum()),   "nb":   int(nb.sum()),
        }
        Bc[dsn] = {
            "fg": int(fg.sum()), "fg_core": int((fg & core).sum()), "fg_nb": int((fg & nb).sum()),
            "bg": int(bg.sum()), "bg_core": int((bg & core).sum()), "bg_nb": int((bg & nb).sum()),
            "bnd": int(bnd.sum()), "bnd_core": int((bnd & core).sum()), "bnd_nb": int((bnd & nb).sum()),
        }
        per_img[dsn] = (n, int(core.sum()) / n)

    def _agg(d, keys):
        t = {k: 0 for k in keys}
        for v in d.values():
            for k in keys:
                t[k] += v[k]
        return t

    pct = lambda a, b: 100 * a / max(b, 1)

    # ── Table A: selection composition ─────────────────────────────────────
    print(f"\nsource={args.source}  |  fg = GT fraction >= 0.5  |  stochastic neighbor fill (seed={args.seed})")
    hdr = (f"{'dataset':>16}  {'N_img':>5}  {'core/img':>8}  "
           f"{'core%fg':>8}  {'neigh%fg':>9}  {'all%fg':>7}")
    print("\n[A] selection composition: of selected patches, what % are foreground")
    print(hdr); print("-" * len(hdr))
    for dsn in sorted(by_ds):
        a = A[dsn]; n_img, cpi = per_img[dsn]
        print(f"{dsn:>16}  {n_img:>5}  {cpi:>8.1f}  "
              f"{pct(a['core_fg'], a['core']):>7.1f}%  {pct(a['nb_fg'], a['nb']):>8.1f}%  "
              f"{pct(a['core_fg'] + a['nb_fg'], a['core'] + a['nb']):>6.1f}%")
    t = _agg(A, ["core_fg", "core", "nb_fg", "nb"])
    print("-" * len(hdr))
    print(f"{'TOTAL':>16}  {'':>5}  {'':>8}  "
          f"{pct(t['core_fg'], t['core']):>7.1f}%  {pct(t['nb_fg'], t['nb']):>8.1f}%  "
          f"{pct(t['core_fg'] + t['nb_fg'], t['core'] + t['nb']):>6.1f}%")

    # ── Table B: GT coverage ────────────────────────────────────────────────
    hdr2 = (f"{'dataset':>16}  {'fg→core':>8}  {'fg→neigh':>9}  {'fg→miss':>8}  "
            f"{'bg→core':>8}  {'bg→neigh':>9}  {'bg→miss':>8}")
    print("\n[B] GT coverage: of all GT fg (and bg) patches, where do they go")
    print(hdr2); print("-" * len(hdr2))
    for dsn in sorted(by_ds):
        b = Bc[dsn]
        fg_miss = b["fg"] - b["fg_core"] - b["fg_nb"]
        bg_miss = b["bg"] - b["bg_core"] - b["bg_nb"]
        print(f"{dsn:>16}  {pct(b['fg_core'], b['fg']):>7.1f}%  {pct(b['fg_nb'], b['fg']):>8.1f}%  "
              f"{pct(fg_miss, b['fg']):>7.1f}%  "
              f"{pct(b['bg_core'], b['bg']):>7.1f}%  {pct(b['bg_nb'], b['bg']):>8.1f}%  "
              f"{pct(bg_miss, b['bg']):>7.1f}%")
    t = _agg(Bc, ["fg", "fg_core", "fg_nb", "bg", "bg_core", "bg_nb"])
    fg_miss = t["fg"] - t["fg_core"] - t["fg_nb"]
    bg_miss = t["bg"] - t["bg_core"] - t["bg_nb"]
    print("-" * len(hdr2))
    print(f"{'TOTAL':>16}  {pct(t['fg_core'], t['fg']):>7.1f}%  {pct(t['fg_nb'], t['fg']):>8.1f}%  "
          f"{pct(fg_miss, t['fg']):>7.1f}%  "
          f"{pct(t['bg_core'], t['bg']):>7.1f}%  {pct(t['bg_nb'], t['bg']):>8.1f}%  "
          f"{pct(bg_miss, t['bg']):>7.1f}%")

    # ── Table C: true-boundary coverage ─────────────────────────────────────
    hdr3 = (f"{'dataset':>16}  {'bnd→core':>8}  {'bnd→neigh':>9}  {'bnd→miss':>8}")
    print("\n[C] boundary coverage: of true-boundary cells (0<gt<1), where do they go")
    print(hdr3); print("-" * len(hdr3))
    for dsn in sorted(by_ds):
        b = Bc[dsn]
        bnd_miss = b["bnd"] - b["bnd_core"] - b["bnd_nb"]
        print(f"{dsn:>16}  {pct(b['bnd_core'], b['bnd']):>7.1f}%  {pct(b['bnd_nb'], b['bnd']):>8.1f}%  "
              f"{pct(bnd_miss, b['bnd']):>7.1f}%")
    t = _agg(Bc, ["bnd", "bnd_core", "bnd_nb"])
    bnd_miss = t["bnd"] - t["bnd_core"] - t["bnd_nb"]
    print("-" * len(hdr3))
    print(f"{'TOTAL':>16}  {pct(t['bnd_core'], t['bnd']):>7.1f}%  {pct(t['bnd_nb'], t['bnd']):>8.1f}%  "
          f"{pct(bnd_miss, t['bnd']):>7.1f}%")


def compute_sweep(ds, args, stage1, device):
    """Sweep tau × sigma × floor over the full val set, pooled across all images.

    The expensive sampling-map computation (stage-1 for prev_pred) is done ONCE; each
    grid combo only re-runs sample_patches + mask accounting. fg/bg from the TRUE GT.
    """
    R = args.grid_res
    by_ds: dict[str, list[int]] = defaultdict(list)
    for i, (dsn, _, _) in enumerate(ds.samples):
        by_ds[dsn].append(i)

    gt_list, val_list = [], []
    for dsn in sorted(by_ds):
        gt32, values = build_maps(ds, by_ds[dsn], args, stage1, device)
        gt_list.append(gt32); val_list.append(values)
    gt32, values = torch.cat(gt_list), torch.cat(val_list)     # (Ntot, N)
    fg = gt32 >= 0.5
    n, N = values.shape
    print(f"\nsweep  source={args.source}  images={n}  M={args.n_total}  "
          f"n_fg_core={args.n_fg_core}  grid={R}  (fg = true GT >= 0.5)")

    taus   = [0.15, 0.30, 0.45]
    sigmas = [0.5, 1.0, 2.0]
    floors = [0.005, 0.02, 0.10]
    hdr = (f"{'tau':>5} {'sigma':>6} {'floor':>6}   {'core%fg':>8} {'all%fg':>7}   "
           f"{'fg→core':>8} {'fg→neigh':>9} {'fg→miss':>8}")
    print(hdr); print("-" * len(hdr))
    pct = lambda a, b: 100 * a / max(int(b), 1)
    for tau in taus:
        for sigma in sigmas:
            for floor in floors:
                idx, is_core, _ = sample_patches(values, args.n_total, tau, sigma, floor,
                                                 R, args.temperature, stochastic=True,
                                                 n_fg_core=args.n_fg_core,
                                                 boundary_tier=not args.no_boundary,
                                                 n_boundary_core=args.n_boundary_core)
                sel  = torch.zeros(n, N, dtype=torch.bool).scatter_(1, idx, True)
                core = torch.zeros(n, N, dtype=torch.bool).scatter_(1, idx, is_core)
                nb   = sel & ~core
                core_fg = (fg & core).sum(); nb_fg = (fg & nb).sum()
                core_t, nb_t, fg_t = core.sum(), nb.sum(), fg.sum()
                fg_miss = fg_t - core_fg - nb_fg
                print(f"{tau:>5.2f} {sigma:>6.1f} {floor:>6.3f}   "
                      f"{pct(core_fg, core_t):>7.1f}% {pct(core_fg + nb_fg, core_t + nb_t):>6.1f}%   "
                      f"{pct(core_fg, fg_t):>7.1f}% {pct(nb_fg, fg_t):>8.1f}% {pct(fg_miss, fg_t):>7.1f}%")


def compute_hist(ds, args, stage1, device):
    """Per-cell value distribution over the val set, for GT and the source map.

    The boundary core selects cells by |value-0.5| < tau, which is only meaningful
    relative to how values are actually distributed. Grid-cell values are strongly
    bimodal — mass piled at 0 (pure bg) and 1 (fg interior) — so a fixed tau-band
    captures a small, non-uniform slice. This reports that density explicitly.
    With --source prev_pred, build_maps returns the TRUE GT and the stage-1 prediction
    in one pass, so both distributions are reported together.
    """
    R = args.grid_res
    by_ds: dict[str, list[int]] = defaultdict(list)
    for i, (dsn, _, _) in enumerate(ds.samples):
        by_ds[dsn].append(i)

    bins = 10
    taus = [0.15, 0.30, 0.45]
    EPS  = 0.02

    def new_acc():
        return {"n": 0, "at0": 0, "at1": 0, "sum": 0.0,
                "hist": torch.zeros(bins), "band": {t: 0 for t in taus}}

    def accumulate(acc, x):
        flat = x.reshape(-1).float()
        acc["n"]   += flat.numel()
        acc["at0"] += int((flat < EPS).sum())
        acc["at1"] += int((flat > 1 - EPS).sum())
        acc["sum"] += float(flat.sum())
        acc["hist"] += torch.histc(flat, bins=bins, min=0.0, max=1.0)
        for t in taus:
            acc["band"][t] += int(((flat - 0.5).abs() < t).sum())

    has_pred = args.source == "prev_pred"
    G = {dsn: new_acc() for dsn in by_ds}
    P = {dsn: new_acc() for dsn in by_ds} if has_pred else None
    for dsn in sorted(by_ds):
        gt32, values = build_maps(ds, by_ds[dsn], args, stage1, device)
        accumulate(G[dsn], gt32)
        if has_pred:
            accumulate(P[dsn], values)

    pct = lambda a, b: 100 * a / max(b, 1)

    def print_table(label, acc_by_ds):
        print(f"\n[{label}] per-cell value distribution  "
              f"(%@0 = v<{EPS}, %@1 = v>{1 - EPS}, %mid = the rest)")
        hdr = (f"{'dataset':>16}  {'%@0':>6}  {'%@1':>6}  {'%mid':>6}  {'mean':>5}   "
               + "  ".join(f"|.5|<{t:.2f}" for t in taus))
        print(hdr); print("-" * len(hdr))
        tot = new_acc()
        for dsn in sorted(acc_by_ds):
            a = acc_by_ds[dsn]
            mid = a["n"] - a["at0"] - a["at1"]
            print(f"{dsn:>16}  {pct(a['at0'], a['n']):>5.1f}%  {pct(a['at1'], a['n']):>5.1f}%  "
                  f"{pct(mid, a['n']):>5.1f}%  {a['sum'] / max(a['n'], 1):>5.2f}   "
                  + "  ".join(f"{pct(a['band'][t], a['n']):>7.1f}%" for t in taus))
            for k in ("n", "at0", "at1", "sum"):
                tot[k] += a[k]
            tot["hist"] += a["hist"]
            for t in taus:
                tot["band"][t] += a["band"][t]
        mid = tot["n"] - tot["at0"] - tot["at1"]
        print("-" * len(hdr))
        print(f"{'TOTAL':>16}  {pct(tot['at0'], tot['n']):>5.1f}%  {pct(tot['at1'], tot['n']):>5.1f}%  "
              f"{pct(mid, tot['n']):>5.1f}%  {tot['sum'] / max(tot['n'], 1):>5.2f}   "
              + "  ".join(f"{pct(tot['band'][t], tot['n']):>7.1f}%" for t in taus))
        edges = [f"{i / bins:.1f}-{(i + 1) / bins:.1f}" for i in range(bins)]
        print("  TOTAL histogram (% of cells per 0.1-wide bin):")
        print("   " + "".join(f"{e:>10}" for e in edges))
        print("   " + "".join(f"{pct(int(tot['hist'][i]), tot['n']):>9.1f}%" for i in range(bins)))

    print(f"\nsource={args.source}  grid={R}  (GT = avg-pooled true mask; "
          f"pred = stage-1 prev_pred upsampled to grid)")
    print_table("GT", G)
    if has_pred:
        print_table("PRED (prev_pred)", P)
    else:
        print("\n(--source ds_gt: the sampling map IS the GT — "
              "run --source prev_pred for the prediction distribution.)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=None,
                    help="single MedSegBench name; if omitted, spreads across n_images datasets")
    ap.add_argument("--split", default="val")
    ap.add_argument("--image_size", type=int, default=128)
    ap.add_argument("--grid_res", type=int, default=32)
    ap.add_argument("--source", choices=["ds_gt", "prev_pred"], default="ds_gt",
                    help="sampling map: downsampled GT (res-32), or real stage-1 pred (res-16→32)")
    ap.add_argument("--stage1_checkpoint",
                    default="results/2d/pfn_seg_universeg/pfn_seg_USegall_R16q8_e256_l6_k3_think8/best.pt")
    ap.add_argument("--context_size", type=int, default=3, help="K context pairs for stage-1")
    ap.add_argument("--s1_chunk", type=int, default=64, help="stage-1 forward batch size")
    ap.add_argument("--n_images", type=int, default=10)
    ap.add_argument("--n_total", type=int, default=256)
    ap.add_argument("--tau", type=float, default=0.30)
    ap.add_argument("--sigma", type=float, default=1.0)
    ap.add_argument("--floor", type=float, default=0.005)
    ap.add_argument("--n_fg_core", type=int, default=64,
                    help="fixed quota of random foreground cells forced into the core")
    ap.add_argument("--no_boundary", action="store_true",
                    help="disable the boundary core tier (tau→0); neighbor field then "
                         "diffuses from fg_core only — the fg-sourced-neighbors test")
    ap.add_argument("--n_boundary_core", type=int, default=0,
                    help="cap the tau boundary band at the N cells closest to 0.5 (0=uncapped, "
                         "current behaviour); a distribution-invariant boundary-core quota")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stats", action="store_true",
                    help="compute per-dataset fg/bg core/neighbor stats over the val set (no plot)")
    ap.add_argument("--sweep", action="store_true",
                    help="sweep tau×sigma×floor over the val set (maps computed once); no plot")
    ap.add_argument("--hist", action="store_true",
                    help="report the per-cell value distribution (GT and prev_pred) over the "
                         "val set — exposes how non-uniform the map is around 0.5; no plot")
    ap.add_argument("--out", default="results/2d/multilevel/sampling_viz.png")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stage1 = load_stage1(args.stage1_checkpoint, device) if args.source == "prev_pred" else None

    # ---- Stats / sweep / hist modes: full val set, no plotting ----
    if args.stats or args.sweep or args.hist:
        names = [args.dataset] if args.dataset else None     # None → all datasets
        ds = MedSegBenchDataset(split=args.split, context_size=args.context_size,
                                image_size=args.image_size, datasets=names)
        fn = compute_hist if args.hist else (compute_sweep if args.sweep else compute_stats)
        fn(ds, args, stage1, device)
        return

    # Choose which datasets to load: a single one, or n_images distinct ones.
    if args.dataset:
        names = [args.dataset]
    else:
        found = sorted(glob.glob(os.path.join(DATA_ROOT, f"*_{args.image_size}.npz")))
        if not found:
            raise SystemExit(f"No *_{args.image_size}.npz files in {DATA_ROOT}")
        all_names = [os.path.basename(p).replace(f"_{args.image_size}.npz", "") for p in found]
        rng.shuffle(all_names)
        names = all_names[:args.n_images]
        print(f"--dataset not given; spreading across: {names}")

    ds = MedSegBenchDataset(split=args.split, context_size=args.context_size,
                            image_size=args.image_size, datasets=names)

    # Group sample indices by dataset (some requested names may have no samples
    # for this split and are simply absent).
    by_ds: dict[str, list[int]] = defaultdict(list)
    for i, (dsn, _, _) in enumerate(ds.samples):
        by_ds[dsn].append(i)
    for lst in by_ds.values():
        rng.shuffle(lst)

    if args.dataset:                       # single dataset → n_images samples from it
        pick = by_ds[args.dataset][:args.n_images]
    else:                                  # multiple → round-robin one per dataset
        present = list(by_ds)
        pick, k = [], 0
        while len(pick) < args.n_images and any(by_ds[p] for p in present):
            p = present[k % len(present)]
            if by_ds[p]:
                pick.append(by_ds[p].pop())
            k += 1
    n = len(pick)

    # Display image + label (left panel).
    images, labels = [], []
    for i in pick:
        item = ds[i]
        images.append(item["image"][0].numpy()); labels.append(item["label"][0].numpy())

    # Truth (gt32) + sampling map (values, = gt32 or stage-1 pred).
    R = args.grid_res
    gt32, values = build_maps(ds, pick, args, stage1, device)
    idx, is_core, is_fg_core = sample_patches(values, args.n_total, args.tau, args.sigma,
                                              args.floor, R, args.temperature,
                                              stochastic=True, n_fg_core=args.n_fg_core,
                                              boundary_tier=not args.no_boundary,
                                              n_boundary_core=args.n_boundary_core)

    fig, axes = plt.subplots(n, 2, figsize=(7, 3.2 * n), squeeze=False)
    for r in range(n):
        ds_name, sidx, lv = ds.samples[pick[r]]
        title = f"{ds_name} #{sidx} L{lv}  core={int(is_core[r].sum())}"
        samp = values[r].reshape(R, R).numpy() if args.source == "prev_pred" else None
        plot_sample(axes[r, 0], axes[r, 1], images[r], labels[r],
                    gt32[r].reshape(R, R).numpy(), idx[r].numpy(), is_core[r].numpy(),
                    is_fg_core[r].numpy(), R, title, samp_map=samp)

    fig.suptitle(f"{args.source} sampling  |  tau={args.tau} sigma={args.sigma} "
                 f"floor={args.floor} fg_core={args.n_fg_core} M={args.n_total}", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"Saved {out}  ({n} samples from {sorted(by_ds)})")


if __name__ == "__main__":
    main()
