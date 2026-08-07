"""
Shared 3D in-context eval loop — the single source of truth used by
experiments/3d/eval.py (now) and experiments/3d/train.py's val step (later).

`evaluate_classes(model, cfg, classes)` runs ONE multi-class loader through
`model.predict()` and groups results back per class, returning per-class summary
rows + per-case records (plus optional qualitative figures). `validate(model,
loader, cls)` remains for a single-class loader. Mirrors experiments/2d/evaluate.py.

Ported from scripts/eval.py so the 3D experiments harness is self-contained;
scripts/eval.py stays as the legacy CLI benchmark.
"""

import contextlib
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.flop_counter import FlopCounterMode
from tqdm import tqdm

from grid_metrics import hard_sum, soft_sum, cos_sum


# ---------------------------------------------------------------------------
# Metrics + figures
# ---------------------------------------------------------------------------

def dice_binary(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Smooth Dice between two binary tensors of any shape."""
    pred, target = pred.bool(), target.bool()
    inter = (pred & target).sum().item()
    union = pred.sum().item() + target.sum().item()
    return (2 * inter + 1) / (union + 1)


def soft_dice_binary(prob: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> float:
    """Threshold-free Dice between a soft probability map and a binary target (any shape).

    Matches the training soft-Dice term (train.py:_soft_dice, eps=1e-6): the shape/overlap
    signal before the 0.5 threshold used by dice_binary."""
    p = prob.flatten().float()
    g = (target.flatten() > 0).float()
    inter = (p * g).sum().item()
    den = p.sum().item() + g.sum().item()
    return (2 * inter + eps) / (den + eps)


def _best_slice(mask: np.ndarray) -> int:
    counts = mask.sum(axis=(1, 2))
    return int(counts.argmax()) if counts.max() > 0 else mask.shape[0] // 2


def save_eval_figure(target_img, gt, pred, ctx_img, ctx_gt, out_path: Path, title: str = "") -> None:
    """Save a 4-panel figure: context | target | GT overlay | pred overlay."""
    def _norm(sl):
        mn, mx = sl.min(), sl.max()
        return (sl - mn) / (mx - mn + 1e-6)

    z, z_ctx = _best_slice(gt), _best_slice(ctx_gt)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), gridspec_kw={"wspace": 0.04})
    axes[0].imshow(_norm(ctx_img[z_ctx]), cmap="gray")
    axes[0].imshow(ctx_gt[z_ctx].astype(float), cmap="Reds", alpha=0.45, vmin=0, vmax=1)
    axes[0].set_title("context", fontsize=8)
    axes[1].imshow(_norm(target_img[z]), cmap="gray")
    axes[1].set_title("target", fontsize=8)
    axes[2].imshow(_norm(target_img[z]), cmap="gray")
    axes[2].imshow(gt[z].astype(float), cmap="Reds", alpha=0.45, vmin=0, vmax=1)
    axes[2].set_title("GT", fontsize=8)
    axes[3].imshow(_norm(target_img[z]), cmap="gray")
    axes[3].imshow(pred[z].astype(float), cmap="Blues", alpha=0.45, vmin=0, vmax=1)
    axes[3].set_title("pred", fontsize=8)
    for ax in axes:
        ax.axis("off")
    fig.suptitle(title, fontsize=9)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def _sample_detail(meta: dict | None) -> str:
    """One compact per-sample string for the sample table's `detail` column, adapting
    to the data source (mirrors experiments/2d/evaluate.py:_sample_detail). omniSynth3D
    meta -> "mode=<m> class=<id> sub=<i>"; anything else / missing -> "". Keeps the
    table's columns fixed across sources (totalseg items carry no meta)."""
    if not meta:
        return ""
    if "class_id" in meta:  # omniSynth3D
        return (f"mode={meta.get('target_mode', '')} "
                f"class={meta.get('class_id', '')} "
                f"sub={meta.get('sample_index', -1)}")
    return ""


# The sample-table columns are fixed (medverse is a native-resolution model, so there is
# no coarse-grid / refine family like 2D's patchset_cnn). One row per eval case, carrying
# the per-case Dice + GT/context occupancy stats + the source-adaptive `detail` string.
_SAMPLE_TABLE_COLS = ["epoch", "class", "in_train", "subject", "dice", "soft_dice", "loss",
                      "time_ms", "tgt_size", "tgt_occ", "ctx_size", "ctx_occ", "spacing",
                      "detail"]


def build_sample_table(cases: list[dict], epoch: int | None = None, train_classes=None):
    """Build a wandb.Table of per-case detail from `evaluate_classes` records.

    Shared by experiments/3d/eval.py (benchmark) and train.py's val step so both log the
    same schema. `epoch` tags the training epoch (-1 for standalone eval). Cases must be
    the enriched dicts emitted by evaluate_classes (with tgt_size/ctx_occ/detail/... keys).
    `train_classes` (set of class names seen in training) fills the `in_train` column.
    """
    import wandb
    ep = -1 if epoch is None else int(epoch)
    train_set = set(train_classes) if train_classes is not None else None
    # Optional per-layer feature-sim columns (train.py attaches fs_<rep>_<dice|retr> onto a
    # subsample of cases); absent for standalone eval, so the schema stays backward-compatible.
    fs_cols = sorted({k for c in cases for k in c if k.startswith("fs_")})
    table = wandb.Table(columns=_SAMPLE_TABLE_COLS + fs_cols)
    for c in cases:
        in_train = c["class"] in train_set if train_set is not None else None
        table.add_data(ep, c["class"], in_train, c["subject"], c["dice"],
                       c.get("soft_dice", float("nan")), c.get("loss", float("nan")),
                       c.get("time_ms", float("nan")),
                       c.get("tgt_size", float("nan")), c.get("tgt_occ", float("nan")),
                       c.get("ctx_size", float("nan")), c.get("ctx_occ", float("nan")),
                       c.get("spacing", float("nan")), c.get("detail", ""),
                       *[c.get(k, float("nan")) for k in fs_cols])
    return table


def _occupancy_stats(label_i: torch.Tensor, ctx_masks_i: torch.Tensor) -> dict:
    """GT + context foreground stats for one sample (model-independent).

    label_i: (D,H,W) target GT. ctx_masks_i: (K,D,H,W) context masks. size = foreground
    voxels; occ = foreground fraction. Context stats are averaged over the K contexts.
    """
    tgt_fg = label_i > 0
    ctx_fg = ctx_masks_i > 0
    K = max(int(ctx_masks_i.shape[0]), 1)
    return {
        "tgt_size": float(tgt_fg.sum()),
        "tgt_occ":  round(float(tgt_fg.float().mean()), 6),
        "ctx_size": float(ctx_fg.float().sum()) / K,   # mean fg voxels per context
        "ctx_occ":  round(float(ctx_fg.float().mean()), 6),
    }


def measure_flops(model, image_size: tuple, K: int, device: torch.device) -> dict:
    """GFLOPs for one predict() call with a single-sample dummy input.

    Returns {"total", "encoder", "transformer"} in GFLOPs. FlopCounterMode keys its
    per-module breakdown by class name (each top-level key aggregates its subtree), so
    the encoder / transformer shares come from the submodule class names; the small
    img/mask embeds + decoder fall outside both. encoder/transformer are None for models
    lacking those submodules (e.g. medverse). All-zero total on failure.
    """
    D, H, W = image_size
    dummy_target  = torch.zeros(1, 1, D, H, W, device=device)
    dummy_ctx_img = torch.zeros(1, K, 1, D, H, W, device=device)
    dummy_ctx_msk = torch.zeros(1, K, D, H, W, dtype=torch.long, device=device)
    try:
        with FlopCounterMode(display=False) as fc:
            model.predict(dummy_target, dummy_ctx_img, dummy_ctx_msk)
        counts = fc.get_flop_counts()

        def _share(attr):
            sub = getattr(model, attr, None)
            if sub is None:
                return None
            c = counts.get(type(sub).__name__)
            return sum(c.values()) / 1e9 if c else None

        return {"total": fc.get_total_flops() / 1e9,
                "encoder": _share("encoder"), "transformer": _share("transformer")}
    except Exception as exc:  # noqa: BLE001
        print(f"    [FLOPs] Could not count: {exc}")
        return {"total": 0.0, "encoder": None, "transformer": None}


# ---------------------------------------------------------------------------
# Per-class eval loop
# ---------------------------------------------------------------------------

def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _eval_autocast(enabled: bool):
    """bf16 CUDA autocast when enabled (else a no-op). Matches training's autocast dtype so
    a compiled encoder/transformer isn't recompiled between the train (bf16) and eval paths."""
    if enabled and torch.cuda.is_available():
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def validate(model, loader, cls: str, *, fig_dir: Path | None = None) -> tuple[dict, list[dict]]:
    """Run inference over one single-class loader.

    Returns (summary_row, cases): summary_row aggregates mean/std Dice + mean time;
    cases is a list of {class, subject, dice, time_ms}. Saves one figure per class
    to fig_dir (first batch) when provided. Uses model.predict().
    """
    cases: list[dict] = []
    fig_saved = False

    for batch in tqdm(loader, desc="eval", leave=False):
        target_img    = batch["image"]
        context_imgs  = batch["context_in"]
        context_masks = batch["context_out"]
        label         = batch["label"]
        subjects      = batch.get("subjects", [None] * target_img.shape[0])

        _sync()
        t0 = time.perf_counter()
        pred = model.predict(target_img, context_imgs, context_masks)
        _sync()
        elapsed_ms = (time.perf_counter() - t0) * 1000

        pred, label = pred.cpu(), label.cpu()

        if fig_dir is not None and not fig_saved:
            subj = subjects[0] or "s0"
            save_eval_figure(
                target_img=target_img[0].squeeze(0).numpy(),
                gt=label[0].numpy(),
                pred=pred[0].numpy(),
                ctx_img=context_imgs[0, 0].squeeze(0).numpy(),
                ctx_gt=context_masks[0, 0].numpy(),
                out_path=fig_dir / f"{cls}_{subj}.png",
                title=f"{cls}  {subj}  dice={dice_binary(pred[0], label[0]):.3f}",
            )
            fig_saved = True

        for i in range(pred.shape[0]):
            cases.append({
                "class":   cls,
                "subject": subjects[i],
                "dice":    round(dice_binary(pred[i], label[i]), 4),
                "time_ms": round(elapsed_ms / pred.shape[0], 1),
            })

    return _summarize(cls, cases), cases


def _summarize(cls: str, cases: list[dict]) -> dict:
    """Aggregate per-case records into a summary row (mean/std Dice + mean time)."""
    dice_scores = [c["dice"] for c in cases]
    times = [c["time_ms"] for c in cases]
    n = len(dice_scores)
    mean_dice = sum(dice_scores) / n if n else 0.0
    std_dice  = (sum((d - mean_dice) ** 2 for d in dice_scores) / n) ** 0.5 if n > 1 else 0.0
    mean_ms   = sum(times) / len(times) if times else 0.0
    row = {
        "class":        cls,
        "n_samples":    n,
        "mean_dice":    round(mean_dice, 4),
        "std_dice":     round(std_dice, 4),
        "mean_time_ms": round(mean_ms, 1),
    }
    # Soft-Dice / loss are only present when evaluate_classes gets a logits_fn/loss_fn
    # (train.py's val step); absent for the eval.py benchmark path.
    soft = [c["soft_dice"] for c in cases if "soft_dice" in c]
    if soft:
        row["mean_soft_dice"] = round(sum(soft) / len(soft), 4)
    losses = [c["loss"] for c in cases if "loss" in c]
    if losses:
        row["mean_loss"] = round(sum(losses) / len(losses), 4)
    for key in ("dice_ds", "dice_ds_soft", "cossim"):
        vals = [c[key] for c in cases if key in c]
        if vals:
            row[f"mean_{key}"] = round(sum(vals) / len(vals), 4)
    return row


def _locator_containment(prob, label, ratio):
    """Coarse->fine locator containment for one sample (pure geometry).

    prob : (D,H,W) tensor — soft probability or 0/1 hard mask (locator weights).
    label: (D,H,W) tensor — GT; foreground = label > 0.
    ratio: s_fine / s_coarse in (0,1). Box side per axis = max(1, round(T_a*ratio)).

    Locator center = prob-weighted centroid over ALL voxels; sum(prob) < 1e-6 -> crop
    center + locator_empty=True. The fine box (that side, clamped inside the volume) is
    placed at the locator center; the oracle box is placed at the GT-foreground centroid.
    Returns (containment, containment_oracle, locator_empty, loc_err_vox):
      containment        = |GT_fg ∩ box|        / |GT_fg|   (NaN if no GT foreground)
      containment_oracle = |GT_fg ∩ box_oracle| / |GT_fg|   (NaN if no GT foreground)
      locator_empty      = bool
      loc_err_vox        = ||center - gt_centroid|| in voxels (NaN if no GT foreground).
                           The caller scales by the coarse spacing to get loc_err_mm.
    """
    p = prob.detach().float().cpu().numpy()
    gt = (label.detach().cpu().numpy() > 0)
    T = p.shape                                    # (D, H, W)
    box = [max(1, int(round(t * ratio))) for t in T]
    idx = np.indices(T, dtype=float)               # (3, D, H, W)

    def _frac_in_box(center):
        total = float(gt.sum())
        lo = []
        for a in range(3):
            l = int(round(center[a] - box[a] / 2))
            l = max(0, min(l, T[a] - box[a]))       # clamp so the box fits in [0, T_a]
            lo.append(l)
        sub = gt[lo[0]:lo[0] + box[0], lo[1]:lo[1] + box[1], lo[2]:lo[2] + box[2]]
        return float(sub.sum()) / total

    # Locator center: prob-weighted centroid over all voxels; empty -> crop center.
    s = float(p.sum())
    if s < 1e-6:
        center = np.array([t / 2.0 for t in T])
        locator_empty = True
    else:
        center = np.array([(idx[a] * p).sum() / s for a in range(3)])
        locator_empty = False

    gt_n = float(gt.sum())
    if gt_n == 0.0:
        return float("nan"), float("nan"), locator_empty, float("nan")

    gt_centroid = np.array([(idx[a] * gt).sum() / gt_n for a in range(3)])
    containment = _frac_in_box(center)
    containment_oracle = _frac_in_box(gt_centroid)
    loc_err_vox = float(np.linalg.norm(center - gt_centroid))
    return containment, containment_oracle, locator_empty, loc_err_vox


def evaluate_classes(model, cfg, classes, *, split=None, fig_dir: Path | None = None,
                     loader=None, logits_fn=None, loss_fn=None, grid_res=None,
                     output_is_prob=False, autocast=False, reuse_logits=False):
    """Eval all `classes` through ONE multi-class loader; return (rows, cases).

    Builds a single dataset over every class (via common.make_eval_loader), so the
    scan/bbox caches load once rather than once per class. Each sample carries its
    own `label_name`, so results are grouped back per class after inference —
    yielding the same (rows, cases) shape as the old per-class loop. Classes with
    no samples get an "error" row. split defaults to cfg.eval.split.

    Pass a prebuilt `loader` (from common.make_eval_loader) to reuse one dataset
    across repeated calls — train.py's val step does this so the dataset isn't
    rebuilt (and caches reloaded) every eval epoch.

    `logits_fn(target, ctx_in, ctx_out) -> (B,1,D,H,W) raw logits` enables the soft
    monitoring metrics: when given, each case also gets `soft_dice` (threshold-free
    overlap of σ(logits) vs GT) and, if `loss_fn(logits, target)` is also given, a
    per-sample `loss`. By default the hard `dice` comes from model.predict (the benchmark
    inference). eval.py passes none of these + leaves autocast/reuse_logits off, so its path
    is byte-identical.

    `autocast=True` runs the eval forward(s) under bf16 (matches training; ~4x faster cold
    encode and no compile recompile between train/eval dtypes). `reuse_logits=True` (requires
    logits_fn) derives the hard prediction from the SAME native logits used for the soft
    metrics — one forward instead of predict + a second logits_fn pass. Both are opt-in and
    used only by train.py's val step for patchset3d, where predict == threshold(train_forward).

    Shared by experiments/3d/eval.py (benchmark) and train.py's val step.
    """
    from collections import defaultdict

    if loader is None:
        from common import make_eval_loader  # local import: common/evaluate are siblings
        split = split or cfg.eval.split
        loader = make_eval_loader(cfg, classes, split=split)

    # Each case dict carries the columns for build_sample_table: class, subject, dice,
    # time_ms, detail (source-adaptive), + tgt_size/tgt_occ/ctx_size/ctx_occ occupancy stats.
    cases_by_class: dict[str, list[dict]] = defaultdict(list)
    figs_saved: set[str] = set()

    for batch in tqdm(loader, desc="eval", leave=False):
        target_img    = batch["image"]
        context_imgs  = batch["context_in"]
        context_masks = batch["context_out"]
        label         = batch["label"]
        subjects      = batch.get("subjects", [None] * target_img.shape[0])
        label_names   = batch["label_names"]
        metas         = batch.get("meta")

        # Hard prediction. With reuse_logits (+ logits_fn), derive it from the SAME native
        # logits the soft metrics use — one forward — instead of a separate model.predict pass
        # (predict == threshold(train_forward) for patchset3d / single-ROI medverse). Default
        # path (reuse_logits=False, e.g. eval.py) is unchanged: predict is the timed inference.
        # Per-batch physical spacing for the spacing-aware frozen encoder (eval spacing is
        # fixed, so batch[0] represents the whole batch). Only forwarded to models that opt
        # in (PatchSet3D.spacing_aware) — medverse's predict/logits_fn take no spacing.
        sp_kw = ({"spacing": float(batch["spacing"][0, 0])}
                 if getattr(model, "spacing_aware", False) and "spacing" in batch else {})
        prob = None
        logits = None
        _sync()
        t0 = time.perf_counter()
        if reuse_logits and logits_fn is not None:
            with torch.no_grad(), _eval_autocast(autocast):
                logits = logits_fn(target_img, context_imgs, context_masks, **sp_kw).float()  # (B,1,D,H,W)
            pred = ((logits.clamp(0, 1) if output_is_prob else torch.sigmoid(logits)) >= 0.5
                    ).float().squeeze(1)                                              # (B,D,H,W)
        else:
            with _eval_autocast(autocast):
                pred = model.predict(target_img, context_imgs, context_masks, **sp_kw)
        _sync()
        per_sample_ms = (time.perf_counter() - t0) * 1000 / pred.shape[0]

        # Soft monitoring pass (train.py val step only): raw logits -> σ for soft Dice + the
        # training loss. Reuse the logits computed above when available, else a single-ROI
        # forward; untimed (timing stays on the hard-prediction pass above).
        if logits_fn is not None:
            if logits is None:
                with torch.no_grad(), _eval_autocast(autocast):
                    logits = logits_fn(target_img, context_imgs, context_masks, **sp_kw).float()  # (B,1,D,H,W)
            tgt = label.to(logits.device).float().unsqueeze(1)                        # (B,1,D,H,W)
            # output_is_prob (medverse): logits_fn already returns a [0,1] probability, so do
            # NOT sigmoid it again (that pins every voxel to foreground). See train.py's
            # model_output_is_prob. Clamp to [0,1] — the plain-conv head can dip slightly out of
            # range, which else drives the soft-Dice denom negative. Default False keeps eval.py's
            # logit path byte-identical.
            prob = (logits.clamp(0, 1) if output_is_prob else torch.sigmoid(logits)).cpu()
            grid_pr = grid_gt = None
            if grid_res is not None:
                grid_pr = F.adaptive_avg_pool3d(prob, (grid_res,) * 3)                 # (B,1,g,g,g)
                grid_gt = F.adaptive_avg_pool3d(label.float().unsqueeze(1).cpu(), (grid_res,) * 3)
            sample_loss = ([float(loss_fn(logits[i:i + 1], tgt[i:i + 1]).item())
                            for i in range(logits.shape[0])] if loss_fn is not None else None)

        pred, label = pred.cpu(), label.cpu()
        context_masks = context_masks.cpu()

        for i in range(pred.shape[0]):
            cls = label_names[i]
            if fig_dir is not None and cls not in figs_saved:
                subj = subjects[i] or "s0"
                save_eval_figure(
                    target_img=target_img[i].squeeze(0).cpu().numpy(),
                    gt=label[i].numpy(),
                    pred=pred[i].numpy(),
                    ctx_img=context_imgs[i, 0].squeeze(0).cpu().numpy(),
                    ctx_gt=context_masks[i, 0].cpu().numpy(),
                    out_path=fig_dir / f"{cls}_{subj}.png",
                    title=f"{cls}  {subj}  dice={dice_binary(pred[i], label[i]):.3f}",
                )
                figs_saved.add(cls)
            case = {
                "class":   cls,
                "subject": subjects[i],
                "dice":    round(dice_binary(pred[i], label[i]), 4),
                "time_ms": round(per_sample_ms, 1),
                "detail":  _sample_detail(metas[i] if metas is not None else None),
            }
            case.update(_occupancy_stats(label[i], context_masks[i]))
            # Per-sample effective spacing (mm/voxel) when the dataset reports it. Spacing is a
            # (3,) tensor; the crop path is isotropic and the spacing-aware model consumes the
            # first axis as its scalar, so log that same scalar. Absent for datasets that emit no
            # spacing (spacing key missing) -> the column stays NaN.
            if "spacing" in batch:
                case["spacing"] = round(float(batch["spacing"][i, 0]), 4)
            if prob is not None:
                case["soft_dice"] = round(soft_dice_binary(prob[i, 0], label[i]), 4)
                if sample_loss is not None:
                    case["loss"] = sample_loss[i]
                if grid_pr is not None:
                    pr, gt = grid_pr[i:i + 1], grid_gt[i:i + 1]
                    case["dice_ds"] = round(float(hard_sum(pr, gt)[0]), 4)
                    case["dice_ds_soft"] = round(float(soft_sum(pr, gt)[0]), 4)
                    case["cossim"] = round(float(cos_sum(pr, gt)[0]), 4)
            cases_by_class[cls].append(case)

    rows, all_cases = [], []
    for cls in classes:
        cases = cases_by_class.get(cls, [])
        all_cases.extend(cases)
        rows.append(_summarize(cls, cases) if cases
                    else {"class": cls, "error": "no samples"})
    return rows, all_cases


def evaluate_spacing_sweep(model, cfg, classes, spacings, *, split=None, fig_dir=None):
    """Run evaluate_classes once per physical crop spacing; tag rows with their spacing.

    Builds a constant-spacing eval loader per `s` (make_eval_loader(..., spacing=s)) and
    calls the unmodified evaluate_classes with that prebuilt loader. `idx` is stable across
    passes, so each spacing sees the same task + context subjects — only the crop spacing
    changes. Figures are saved on the first spacing only (later passes reuse the filenames).
    Returns (rows, cases): rows are per-(class, spacing); cases are all passes concatenated
    (each case already carries case["spacing"]).
    """
    from common import make_eval_loader  # local import: common/evaluate are siblings

    rows, cases = [], []
    for i, s in enumerate(spacings):
        loader = make_eval_loader(cfg, classes, split=split or cfg.eval.split, spacing=s)
        rows_s, cases_s = evaluate_classes(
            model, cfg, classes, loader=loader,
            fig_dir=fig_dir if i == 0 else None)
        for r in rows_s:
            r["spacing"] = s
        rows.extend(rows_s)
        cases.extend(cases_s)
    return rows, cases
