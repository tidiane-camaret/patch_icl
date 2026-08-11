"""Driver: load a PatchSet3D checkpoint, sweep (tier x resolution) over the shared 3D
eval loader, and write a tidy per-(task,tier,res) CSV of matching metrics + real Dice.

    python experiments/3d/feature_sim/run.py eval.checkpoint=results/.../best.pt \
        eval.model=patchset3d
"""
import collections
import contextlib
import csv
import math
import sys
from pathlib import Path

import hydra
import torch
import wandb
from omegaconf import DictConfig, OmegaConf

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "3d"))      # common / eval / evaluate

from common import DEVICE, make_eval_loader, _source_root          # noqa: E402
from data.totalseg_classes import resolve_classes                  # noqa: E402
from feature_sim.adapters import (                                 # noqa: E402
    PatchSet3DEncoderAdapter, PrimusEncoderAdapter, TapCTEncoderAdapter)
from feature_sim.cost import measure_encode_cost                   # noqa: E402
from feature_sim.labels import grid_labels, sample_points          # noqa: E402
from feature_sim.metrics import (                                  # noqa: E402
    prototype_cosine, fg_match_margin, retrieval_at1, label_transfer)


def _fwd_ctx(cfg):
    """Autocast context for the model/encoder forwards. Default on (bf16, matching the
    train/eval regime) — feature_sim historically ran fp32, leaving ~2.8x on the table on
    the frozen ViT. Metrics stay fp32 (see _metric_row, which disables autocast). CPU/off
    -> nullcontext (no-op)."""
    if cfg.eval.get("autocast", True) and DEVICE.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def _maybe_compile(adapter, model, cfg):
    """Opt-in torch.compile of the heavy forward modules — the read-out transformer and the
    frozen Primus eva stack — mirroring experiments/3d/train.py (~2.2x forward on exp 36).
    Called AFTER measure_encode_cost so fvcore doesn't try to trace a compiled graph.
    dynamic=True: target vs context batch sizes differ, so avoid recompiling per shape."""
    if not cfg.eval.get("compile", False):
        return
    done = []
    if model is not None and hasattr(model, "transformer"):
        model.transformer = torch.compile(model.transformer, dynamic=True)
        done.append("transformer")
    # frozen Primus eva: on the patchset3d model's encoder, or a generic PrimusEncoderAdapter.
    enc = getattr(model, "encoder", None) if model is not None else None
    prim = getattr(enc, "primus", None) or getattr(adapter, "primus", None)
    if prim is not None and hasattr(prim, "eva"):
        prim.eva = torch.compile(prim.eva, dynamic=True)
        done.append("frozen eva")
    # frozen tap-ct: compile the whole HF ViT. The adapter always encodes one volume at a
    # time (pixel_values (1,1,T,T,T), fixed T), so shapes are static -> dynamic=False. The
    # SDPA-patched attention compiles fine (F.scaled_dot_product_attention). See tap_ct_compile.py.
    if cfg.eval.model == "tap_ct" and getattr(adapter, "model", None) is not None:
        mode = cfg.eval.get("tapct", {}).get("compile_mode", "default")
        adapter.model = torch.compile(adapter.model, mode=mode, dynamic=False)
        done.append(f"tap_ct ViT ({mode})")
    print(f"  torch.compile on: {', '.join(done) or '(nothing)'} — first batch slow")


def plan_sweep(tiers, resolutions, budget, R):
    rows, seen = [], set()
    for tier in tiers:
        if tier in ("transformer_q", "transformer_layers"):
            # Transformer probes live only at the token grid res=R (dense); transformer_layers
            # fans out to one row per block inside _rows_for_task, so it stays a single plan entry.
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


def _primus_spec(cfg):
    """Resolve (weights, primus_kwargs, preproc) for a Primus adapter from either an
    extraction sidecar (eval.primus_sidecar) or inline config (eval.primus_kwargs)."""
    import json
    sidecar = cfg.eval.get("primus_sidecar")
    if sidecar:
        meta = json.load(open(sidecar))
        pk, preproc = meta["primus_kwargs"], meta.get("preproc")
        weights = cfg.eval.get("weights") or meta.get("weights")
    else:
        if not cfg.eval.get("primus_kwargs"):
            raise ValueError("model=primus needs eval.primus_sidecar or eval.primus_kwargs")
        pk = OmegaConf.to_container(cfg.eval.primus_kwargs, resolve=True)
        preproc = (OmegaConf.to_container(cfg.eval.preproc, resolve=True)
                   if cfg.eval.get("preproc") else None)
        weights = cfg.eval.get("weights")
    return weights, pk, preproc


def _tapct_spec(cfg):
    """Resolve TapCTEncoderAdapter kwargs from eval.tapct (all optional) + data.image_size."""
    t = cfg.eval.get("tapct") or {}
    return dict(precision=t.get("precision", "bf16"),
                to_lps=bool(t.get("to_lps", True)),
                resize_native=bool(t.get("resize_native", True)),
                pad_hu=t.get("pad_hu"),
                max_layers=t.get("max_layers"),
                image_size=int(cfg.data.image_size[-1]))


def build_adapter(cfg):
    """Dispatch on cfg.eval.model -> (adapter, model_or_none).

    model_or_none is the PatchSet3D used for real_dice (a segmenter); None for a generic
    frozen encoder (Primus/CoLiPri/tap_ct), whose study reports intrinsic metrics only."""
    which = cfg.eval.model
    if which == "patchset3d":
        m = _load_patchset(cfg)
        return PatchSet3DEncoderAdapter(m), m
    if which == "primus":
        weights, pk, preproc = _primus_spec(cfg)
        # The generic adapter self-autocasts internally (its encode isn't wrapped by
        # _fwd_ctx's cost probe / standalone use), so it honours eval.autocast directly.
        return PrimusEncoderAdapter(weights_path=weights, primus_kwargs=pk,
                                    preproc=preproc, device=DEVICE.type,
                                    autocast=bool(cfg.eval.get("autocast", True))), None
    if which == "tap_ct":
        # Frozen tap-ct-b-3d: self-preprocesses (de-norm/reorient/TAP processor) and
        # self-autocasts via precision. tiers=[backbone] only. See TapCTEncoderAdapter.
        return TapCTEncoderAdapter(device=DEVICE.type, **_tapct_spec(cfg)), None
    raise ValueError(f"unknown eval.model {which!r} (expected patchset3d | primus | tap_ct)")


def _rows_for_task(adapter, model, item, cfg, plan, input_res, gen):
    """One task (single batch index already unbatched to B=1 tensors). Yields dict rows."""
    fs = cfg.feature_sim
    # Frozen-encoder adapters may cache native encodes across resolutions within a task;
    # reset per task since new items reuse tensor storage (cache is keyed by storage ptr).
    if hasattr(adapter, "reset_cache"):
        adapter.reset_cache()
    image = item["image"].to(DEVICE)              # (1,1,D,H,W)
    cin = item["context_in"].to(DEVICE)           # (1,K,1,D,H,W)
    cout = item["context_out"].to(DEVICE)         # (1,K,D,H,W)
    gt = item["label"][0]                         # (D,H,W)
    K = cin.shape[1]
    cls = item["label_names"][0]
    obj_vox = int((gt > 0).sum().item())
    # Per-task physical spacing for a spacing-aware model: thread the crop's mm/voxel into
    # both real_dice and every feature encode so the frozen ViT's RoPE matches the crop
    # (mirrors evaluate.py). Empty for non-spacing-aware models / generic encoders, so those
    # adapter/predict signatures never receive the kwarg (unchanged). See adapters.features.
    sp = ({"spacing": float(item["spacing"][0, 0])}
          if model is not None and getattr(model, "spacing_aware", False) and "spacing" in item
          else {})
    # real_dice needs a trained segmenter; a generic frozen encoder (model is None)
    # reports intrinsic metrics only -> real_dice stays None (coupling analyses drop it).
    if model is not None:
        with torch.no_grad():
            real = model.predict(image, cin, cout, **sp)   # cin (1,K,1,D,H,W) -> (1,D,H,W)
        inter = (real[0] * (gt.to(DEVICE) > 0)).sum().item()
        den = real[0].sum().item() + (gt > 0).sum().item()
        real_dice = (2 * inter) / den if den > 0 else 0.0
    else:
        real_dice = None

    ctx_imgs = cin[0].squeeze(1)                  # (K,D,H,W)
    for p in plan:
        tier, res, mode = p["tier"], p["res"], p["mode"]
        if tier == "transformer_q":
            q = adapter.transformer_query(image, cin, cout, **sp)[0]    # (N,e)
            tl = grid_labels(gt, adapter.R, threshold=None).flatten()   # soft occupancy
            cl = torch.stack([grid_labels(cout[0, k], adapter.R, threshold=None).flatten()
                              for k in range(K)]).flatten()
            # context side uses img_embed tier (concat→e projection) so its channel dim
            # matches the transformer query rep e; note this is an approximate ceiling
            # reference (post-transformer target vs pre-transformer context embeddings)
            cf = adapter.features(ctx_imgs.unsqueeze(1), "img_embed", adapter.R, **sp)
            cf = cf.flatten(2).transpose(1, 2).reshape(-1, cf.shape[1])
            yield _metric_row(cls, obj_vox, real_dice, tier, res, mode,
                              adapter.native_res("concat", input_res),
                              q, tl, cf, cl, K)
            continue
        if tier == "transformer_layers":
            # Per-block target<->context correspondence: ONE forward hooks every block and
            # returns the (target, context) img-token pair after each, both POST-transformer
            # (clean matched probe, unlike transformer_q). Emit a row per layer, tier "tf:L{i}".
            tl = grid_labels(gt, adapter.R, threshold=None).flatten()          # soft occupancy
            cl = torch.stack([grid_labels(cout[0, k], adapter.R, threshold=None).flatten()
                              for k in range(K)]).flatten()
            for li, (tq, cq) in enumerate(
                    adapter.transformer_pair_per_layer(image, cin, cout, **sp)):
                yield _metric_row(cls, obj_vox, real_dice, f"tf:L{li}", res, mode,
                                  adapter.R, tq[0], tl, cq[0], cl, K)
            continue
        if tier == "backbone_layers":
            # Frozen-encoder depth sweep: one forward captures every eva block's grid; emit a
            # row per block (tier "bb:L{i}"), the ViT analogue of the conv stage:* sweep. Same
            # prototype/retrieval metrics as the dense/point paths below, just per layer.
            if mode == "dense":
                tgrids = adapter.features_per_layer(image, res)
                cgrids = adapter.features_per_layer(ctx_imgs.unsqueeze(1), res)
                tl = grid_labels(gt, res, threshold=None).flatten()
                cl = torch.stack([grid_labels(cout[0, k], res, threshold=None).flatten()
                                  for k in range(K)]).flatten()
                for li, (tg, cvol) in enumerate(zip(tgrids, cgrids)):
                    tf = tg[0].flatten(1).transpose(0, 1)              # (res^3, C)
                    cf = cvol.flatten(2).transpose(1, 2).reshape(-1, cvol.shape[1])
                    yield _metric_row(cls, obj_vox, real_dice, f"bb:L{li}", res, mode,
                                      adapter.native_res("backbone", input_res),
                                      tf, tl, cf, cl, K)
            else:
                tcoords, tl = sample_points(gt, fs.n_fg, fs.n_bg,
                                            band=fs.get("band"), generator=gen)
                tfs = adapter.sample_features_per_layer(image, tcoords.to(DEVICE).unsqueeze(0))
                cfs = [[] for _ in tfs]
                ctx_labels = []
                for k in range(K):
                    cc, ll = sample_points(cout[0, k].cpu(), fs.n_fg, fs.n_bg,
                                           band=fs.get("band"), generator=gen)
                    per = adapter.sample_features_per_layer(
                        ctx_imgs[k][None, None], cc.to(DEVICE).unsqueeze(0))
                    for li, s in enumerate(per):
                        cfs[li].append(s[0])
                    ctx_labels.append(ll)
                cl = torch.cat(ctx_labels, 0)
                for li, tg in enumerate(tfs):
                    yield _metric_row(cls, obj_vox, real_dice, f"bb:L{li}", res, mode,
                                      adapter.native_res("backbone", input_res),
                                      tg[0], tl, torch.cat(cfs[li], 0), cl, K)
            continue
        if mode == "dense":
            tf = adapter.features(image, tier, res, **sp)[0]           # (C,res,res,res)
            tf = tf.flatten(1).transpose(0, 1)                         # (res^3, C)
            tl = grid_labels(gt, res, threshold=None).flatten()        # soft occupancy fraction
            cvol = adapter.features(ctx_imgs.unsqueeze(1), tier, res, **sp)  # (K,C,res^3...)
            cf = cvol.flatten(2).transpose(1, 2).reshape(-1, cvol.shape[1])
            cl = torch.stack([grid_labels(cout[0, k], res, threshold=None).flatten()
                              for k in range(K)]).flatten()
        else:
            tcoords, tl = sample_points(gt, fs.n_fg, fs.n_bg,
                                        band=fs.get("band"), generator=gen)
            tf = adapter.sample_features(image, tier, tcoords.to(DEVICE).unsqueeze(0), **sp)[0]
            cfs, ctx_labels = [], []
            for k in range(K):
                cc, ll = sample_points(cout[0, k].cpu(), fs.n_fg, fs.n_bg,
                                       band=fs.get("band"), generator=gen)
                cfs.append(adapter.sample_features(
                    ctx_imgs[k][None, None], tier, cc.to(DEVICE).unsqueeze(0), **sp)[0])
                ctx_labels.append(ll)
            cf = torch.cat(cfs, 0); cl = torch.cat(ctx_labels, 0); tf = tf
        yield _metric_row(cls, obj_vox, real_dice, tier, res, mode,
                          adapter.native_res(tier, input_res),
                          tf, tl, cf, cl, K)


def _metric_row(cls, obj_vox, real_dice, tier, res, mode, tier_native,
                tf, tl, cf, cl, K):
    # Run metrics on the features' device (GPU) — retrieval/margin do large (n_fg x M)
    # matmuls that are far faster there. Labels may come from CPU (target GT) or GPU
    # (context_out), so align them to the feature device rather than assuming a common one.
    dev = tf.device
    # Metrics run fp32 regardless of the forward regime: cast the (possibly bf16) features up
    # and disable autocast so the cosine matmuls / argmax ranking aren't computed in bf16
    # (the encode is autocast for speed; the similarity ranking must stay precise).
    tf, cf = tf.float(), cf.float()
    tl, cf, cl = tl.to(dev), cf.to(dev), cl.to(dev)
    with torch.autocast(DEVICE.type, enabled=False):
        proto = prototype_cosine(tf, tl, cf, cl, mode=mode)
        # Full-volume label-transfer overlap is the segmentation-quality proxy (replaces the
        # old size-collinear min-max soft_dice); dense only — point pools are 50/50 sampled so
        # precision would be distorted. Yields transfer_dice/precision/recall from one NN pass.
        lt = label_transfer(tf, tl, cf, cl) if mode == "dense" else {}
        margin = fg_match_margin(tf, tl, cf, cl)
        retr = retrieval_at1(tf, tl, cf, cl)
    row = {"class": cls, "obj_vox": obj_vox, "real_dice": real_dice,
           "tier": tier, "res": res, "mode": mode, "tier_native_res": tier_native,
           "K": K, "auroc": proto["auroc"],
           # None (not "") for absent metrics: dense rows have transfer_*, point rows have
           # ap. wandb.Table infers a column type from the first row and rejects a later ""
           # String in a Number column, so use None (an allowed optional) instead.
           "ap": proto.get("ap"),
           "transfer_dice": lt.get("transfer_dice"),
           "transfer_precision": lt.get("transfer_precision"),
           "transfer_recall": lt.get("transfer_recall"),
           "margin": margin,
           "retrieval_at1": retr}
    return row


_AGG_METRICS = ("auroc", "ap", "transfer_dice", "transfer_precision",
                "transfer_recall", "margin", "retrieval_at1")


def _num(x):
    """Coerce a cell to float, mapping "", None and nan to None (dropped from means)."""
    if x is None or x == "":
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) else v


def _mean(vals):
    vals = [v for v in (_num(x) for x in vals) if v is not None]
    return sum(vals) / len(vals) if vals else None


def aggregate_by_class(rows):
    """Collapse the per-task table to per-(tier,res,mode,class) means for wandb.

    Each config sees ~one row per subject for a class; this averages those tasks (dropping
    None/nan) into a single row, cutting the 130k full table to ~n_config x n_class rows.
    Returns (fields, agg_rows) sorted by (tier, res, class)."""
    groups = collections.defaultdict(list)
    for r in rows:
        groups[(r["tier"], r["res"], r["mode"], r["tier_native_res"], r["class"])].append(r)
    out = []
    for (tier, res, mode, native, cls) in sorted(groups, key=lambda k: (k[0], int(k[1]), k[4])):
        g = groups[(tier, res, mode, native, cls)]
        row = {"tier": tier, "res": res, "mode": mode, "tier_native_res": native,
               "class": cls, "n": len(g),
               "obj_vox_mean": _mean([r["obj_vox"] for r in g]),
               "real_dice_mean": _mean([r["real_dice"] for r in g])}
        for m in _AGG_METRICS:
            row[m + "_mean"] = _mean([r[m] for r in g])
        out.append(row)
    return list(out[0].keys()), out


def _avg_rank(a):
    """Average (tie-corrected) ranks of a 1-D list — the basis for Spearman."""
    idx = sorted(range(len(a)), key=lambda i: a[i])
    ranks = [0.0] * len(a)
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and a[idx[j + 1]] == a[idx[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[idx[k]] = (i + j) / 2.0
        i = j + 1
    return ranks


def _pearson(x, y):
    """Pearson r on paired lists (already NaN-free). None if degenerate."""
    n = len(x)
    if n < 3:
        return None
    mx, my = sum(x) / n, sum(y) / n
    sxy = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    sxx = sum((xi - mx) ** 2 for xi in x)
    syy = sum((yi - my) ** 2 for yi in y)
    d = (sxx * syy) ** 0.5
    return sxy / d if d > 0 else None


def _paired(g, col, ref="real_dice", extra=None):
    """Rows of g where col, ref (and each `extra` col) are all numeric; returns column lists."""
    cols = [col, ref] + list(extra or [])
    keep = [r for r in g if all(_num(r[c]) is not None for c in cols)]
    return {c: [_num(r[c]) for r in keep] for c in cols}, len(keep)


def _spearman(g, col, ref="real_dice"):
    p, n = _paired(g, col, ref)
    if n < 3:
        return None
    return _pearson(_avg_rank(p[col]), _avg_rank(p[ref]))


def _partial_spearman(g, col, ctrl="obj_vox", ref="real_dice"):
    """Spearman(col, ref | ctrl): Pearson of rank-residuals after regressing out rank(ctrl).
    Confirms the probe predicts Dice beyond the object-size confound."""
    p, n = _paired(g, col, ref, extra=[ctrl])
    if n < 4:
        return None
    rc, rr, rz = _avg_rank(p[col]), _avg_rank(p[ref]), _avg_rank(p[ctrl])
    mz = sum(rz) / n
    szz = sum((z - mz) ** 2 for z in rz)
    if szz == 0:
        return None

    def resid(v):
        mv = sum(v) / n
        b = sum((vi - mv) * (zi - mz) for vi, zi in zip(v, rz)) / szz
        a = mv - b * mz
        return [vi - (a + b * zi) for vi, zi in zip(v, rz)]

    return _pearson(resid(rc), resid(rr))


def summarize_by_config(rows):
    """Collapse to one row per (tier,res,mode): metric means + the metric<->Dice couplings
    the study rests on (Spearman with real_dice, plus size-partialled Spearman for
    retrieval_at1). ~21 rows — the headline table for choosing a stage/resolution."""
    groups = collections.defaultdict(list)
    for r in rows:
        groups[(r["tier"], r["res"], r["mode"], r["tier_native_res"])].append(r)
    out = []
    for (tier, res, mode, native) in sorted(groups, key=lambda k: (k[0], int(k[1]))):
        g = groups[(tier, res, mode, native)]
        row = {"tier": tier, "res": res, "mode": mode, "tier_native_res": native,
               "n": len(g), "obj_vox_mean": _mean([r["obj_vox"] for r in g]),
               "real_dice_mean": _mean([r["real_dice"] for r in g])}
        for m in _AGG_METRICS:
            row[m + "_mean"] = _mean([r[m] for r in g])
        # metric<->Dice coupling (the actual research signal — means alone hide it)
        for m in ("auroc", "margin", "retrieval_at1", "transfer_dice"):
            row["spearman_" + m] = _spearman(g, m)
        row["pearson_auroc"] = _pearson(*(lambda p: (p["auroc"], p["real_dice"]))(
            _paired(g, "auroc")[0]))
        # size-partialled Spearman: does the probe predict Dice beyond object size?
        row["partial_spearman_retr"] = _partial_spearman(g, "retrieval_at1")
        row["partial_spearman_transfer"] = _partial_spearman(g, "transfer_dice")
        out.append(row)
    return list(out[0].keys()), out


@hydra.main(config_path="../../../configs/experiment/3d",
            config_name="feature_sim", version_base="1.3")
def main(cfg: DictConfig) -> None:
    torch.manual_seed(cfg.eval.seed)
    _, root, is_mri = _source_root(cfg)
    if cfg.eval.model == "patchset3d" and not cfg.eval.get("checkpoint"):
        raise ValueError("eval.checkpoint is required (path to a trained PatchSet3D best.pt)")
    val_classes = resolve_classes(cfg.data.val_classes, root, is_mri=is_mri)
    loader = make_eval_loader(cfg, val_classes, split=cfg.eval.split)
    adapter, model = build_adapter(cfg)
    input_res = int(cfg.data.image_size[-1])
    tiers = list(cfg.feature_sim.tiers)
    # Transformer probes are PatchSet3D-only; drop them for a generic encoder.
    tf_tiers = {"transformer_q", "transformer_layers"}
    if not hasattr(adapter, "transformer_pair_per_layer"):
        dropped = [t for t in tiers if t in tf_tiers]
        if dropped:
            print(f"  adapter has no transformer probes; dropping tiers {dropped}")
        tiers = [t for t in tiers if t not in tf_tiers]
    # Fail early + actionably on encoder-tiers the adapter can't serve (the config default
    # is conv-oriented [stage:*, concat, ...]; a generic Primus/CoLiPri encoder only exposes
    # `backbone`). Without this the run dies mid-loop with a cryptic assert deep in features().
    supported = set(adapter.tiers()) | tf_tiers
    bad = [t for t in tiers if t not in supported]
    if bad:
        raise ValueError(
            f"eval.model={cfg.eval.model} encoder does not support tiers {bad}; "
            f"supported: {sorted(adapter.tiers())}. Set e.g. "
            f"'feature_sim.tiers=[{','.join(adapter.tiers())}]'.")
    plan = plan_sweep(tiers, list(cfg.feature_sim.resolutions),
                      int(cfg.feature_sim.budget), adapter.R)
    gen = torch.Generator().manual_seed(cfg.eval.seed)

    # Image-encoding cost (frozen forward): FLOPs / peak VRAM / it-s, once per run.
    cost = measure_encode_cost(adapter, input_res, DEVICE)
    print(f"  encode cost @ {input_res}^3: {cost}")

    # Opt-in torch.compile of the heavy modules (after the FLOP probe above; see _maybe_compile).
    _maybe_compile(adapter, model, cfg)

    # W&B: online when a project is configured, disabled otherwise (CSV still written).
    wb_on = bool(cfg.wandb.get("project"))
    wandb.init(project=cfg.wandb.get("project"), name=cfg.wandb.get("name"),
               mode="online" if wb_on else "disabled",
               config=OmegaConf.to_container(cfg, resolve=True))

    out_dir = Path(cfg.eval.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    cost_row = {"encoder": cfg.eval.model, "input_res": input_res, **cost}
    with open(out_dir / "encode_cost.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(cost_row)); w.writeheader(); w.writerow(cost_row)
    csv_path = out_dir / "feature_sim.csv"
    all_rows, fields, n = [], None, 0
    from tqdm import tqdm
    pbar = tqdm(loader, desc=f"feature_sim {cfg.eval.model}", unit="batch")
    with open(csv_path, "w", newline="") as fh:
        writer = None
        for batch in pbar:
            B = batch["image"].shape[0]
            for b in range(B):
                item = {k: (v[b:b + 1] if torch.is_tensor(v) else [v[b]])
                        for k, v in batch.items()}
                # Autocast wraps only the forwards inside _rows_for_task; metrics re-disable it.
                with _fwd_ctx(cfg):
                    rows = list(_rows_for_task(adapter, model, item, cfg, plan,
                                               input_res, gen))
                for row in rows:
                    if writer is None:
                        fields = list(row.keys())
                        writer = csv.DictWriter(fh, fieldnames=fields)
                        writer.writeheader()
                    writer.writerow(row); all_rows.append(row); n += 1
            pbar.set_postfix(rows=n)
    print(f"Done. {n} rows -> {csv_path}")

    if wb_on:
        wandb.log({f"encode_cost/{k}": v for k, v in cost.items() if v is not None})
    if wb_on and all_rows:
        # Per-(tier,res,mode,class) aggregated table (the full 130k per-task table is far
        # too large for wandb; the raw CSV above is the source of truth). Plus a scalar
        # mean per (tier,res) for the run-overview panels.
        afields, arows = aggregate_by_class(all_rows)
        wandb.log({"feature_sim/by_class":
                   wandb.Table(columns=afields, data=[[r[c] for c in afields] for r in arows])})
        # Config-summary (~21 rows): metric means + metric<->Dice couplings (the headline).
        sfields, srows = summarize_by_config(all_rows)
        wandb.log({"feature_sim/by_config":
                   wandb.Table(columns=sfields, data=[[r[c] for c in sfields] for r in srows])})
        agg = collections.defaultdict(lambda: collections.defaultdict(list))
        for r in all_rows:
            key = f"{r['tier']}@{r['res']}"
            for m in ("auroc", "margin", "retrieval_at1"):
                v = _num(r[m])
                if v is not None:
                    agg[key][m].append(v)
        summary = {f"feature_sim/{m}/{key}": sum(v) / len(v)
                   for key, ms in agg.items() for m, v in ms.items()}
        wandb.log(summary)
    wandb.finish()


if __name__ == "__main__":
    main()
