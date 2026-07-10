"""Unified AdamW trainer for 2D in-context segmentation models.

Fuses the former universeg_train.py with the PatchSetCNN trainer: one loop trains
either model, selected by `cfg.model`:

  - universeg     : UniverSegBaseline, native-resolution (H×W) logit.
  - patchset_cnn  : PatchSetCNN, low-resolution (R×R) logit (no decoder/upsampling).

The loop is model-agnostic. Both models are called as
    model(img, context_in=cin, context_out=cout, mode=...)  -> {"final_logit": ...}
and the GT mask is avg-pooled to the logit's spatial size — a no-op for UniverSeg
(H×W) and an R×R downsample for PatchSetCNN. So the BCE + soft-Dice loss, train-Dice
monitoring, per-epoch val Dice, best-checkpoint saving and wandb logging are shared.

The saved best.pt records model_name + arch metadata so eval.py can reload it.

Usage:
    # UniverSeg on omniSynth (default config)
    python experiments/2d/train.py synth=omniglot

    # PatchSetCNN on omniSynth, low-res 16×16 prediction
    python experiments/2d/train.py --config-name patchset_cnn_train synth=omniglot
"""

import collections
import datetime
import math
import os
import random
import socket
import sys
import tempfile
import time
from pathlib import Path

# Node-local compile caches: ~/.triton and ~/.cache live on shared NFS, so a
# cuda_utils.so compiled on a node with a newer GLIBC poisons the cache for
# nodes with an older GLIBC ("GLIBC_2.34 not found"). Key the cache by hostname
# on local /tmp so each node compiles its own artifacts. Must be set before torch.
_cache_root = os.path.join(tempfile.gettempdir(), f"{os.environ.get('USER', 'user')}_compile_{socket.gethostname()}")
os.environ.setdefault("TRITON_CACHE_DIR", os.path.join(_cache_root, "triton"))
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", os.path.join(_cache_root, "inductor"))

import hydra
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DEVICE, build_dataset, make_loader
from evaluate import validate, _target_like, _upsample_to, _as_res_list, refine_geometry
from src.models.bbox_refine import crop_resize
from pfn_train import Muon, lawa_average, soft_dice_loss


def _autocast():
    return torch.autocast(device_type=DEVICE.type, dtype=torch.bfloat16,
                          enabled=DEVICE.type == "cuda")


def _soft_sum(prob: torch.Tensor, target: torch.Tensor, eps: float = 1e-6):
    """Threshold-free Dice SUM + valid-row count (prob & target at the same res)."""
    p = prob.detach().flatten(1).float()
    g = target.detach().flatten(1).float()
    den = p.sum(1) + g.sum(1)
    ok = den > eps
    s = torch.where(ok, 2 * (p * g).sum(1) / den.clamp_min(eps), torch.zeros_like(den))
    return s.sum(), ok.sum()


def _cos_sum(prob: torch.Tensor, target: torch.Tensor, eps: float = 1e-6):
    """Scale-invariant cosine similarity SUM + valid-row count (prob & target same res).

    Σ(p·g)/(‖p‖·‖g‖) per row: magnitude cancels, so it stays a real 0→1 signal at low
    resolution where soft Dice collapses toward the mean occupancy. Rows with an empty
    GT map are skipped (matches _soft_sum's valid-row convention)."""
    p = prob.detach().flatten(1).float()
    g = target.detach().flatten(1).float()
    den = p.norm(dim=1) * g.norm(dim=1)
    ok = den > eps
    c = torch.where(ok, (p * g).sum(1) / den.clamp_min(eps), torch.zeros_like(den))
    return c.sum(), ok.sum()


def _topk_sum(prob: torch.Tensor, target: torch.Tensor, k: int, eps: float = 1e-6):
    """Top-k patch-overlap SUM + valid-row count (prob & target at the same res).

    Batched form of common.topk_overlap: per row, recall of the GT-positive patches
    within the pred top-k — |gt_pos ∩ topk(pred)| / |gt_pos| (|gt_pos| capped to k). k is
    clamped to the patch count; rows with an empty GT map are skipped."""
    p = prob.detach().flatten(1).float()
    g = target.detach().flatten(1).float()
    k = min(k, p.shape[1])
    n_pos = (g > eps).sum(1)                             # (B,) GT-positive patches per row
    ok = n_pos > 0
    m = n_pos.clamp(max=k)                               # (B,) GT set size, capped to k
    pi = p.topk(k, dim=1).indices                        # (B,k) pred top-k indices
    hit_pred = torch.zeros_like(g, dtype=torch.bool).scatter_(1, pi, True)
    gv, gi = g.topk(k, dim=1)                            # (B,k) highest GT values + indices
    inter = (hit_pred.gather(1, gi) & (gv > eps)).sum(1).float()   # positives recalled in pred top-k
    o = torch.where(ok, inter / m.clamp(min=1).float(), torch.zeros_like(inter))
    return o.sum(), ok.sum()


def _hard_sum(prob: torch.Tensor, gt: torch.Tensor, eps: float = 1e-6):
    """Hard Dice SUM + valid-row count: pred≥0.5 vs GT>0 (prob & gt at the same res)."""
    p = (prob.detach().flatten(1).float() >= 0.5).float()
    g = (gt.detach().flatten(1).float() > 0).float()
    den = p.sum(1) + g.sum(1)
    ok = den > eps
    h = torch.where(ok, 2 * (p * g).sum(1) / den.clamp_min(eps), torch.zeros_like(den))
    return h.sum(), ok.sum()


def build_model(cfg) -> tuple[torch.nn.Module, str, dict]:
    """Construct the model selected by cfg.model; return (model, name, ckpt_meta)."""
    name = cfg.get("model", "universeg")
    if name == "universeg":
        from src.models.universeg_baseline import UniverSegBaseline
        pretrained = bool(cfg.train.get("pretrained", True))
        print(f"Building UniverSeg (pretrained={pretrained}, size={cfg.data.image_size})...")
        model = UniverSegBaseline(pretrained=pretrained, input_size=cfg.data.image_size)
        return model, name, {"pretrained": pretrained}
    if name == "patchset_cnn":
        from src.models.patchset_cnn import PatchSetCNN
        a = cfg.arch
        print(f"Building PatchSetCNN (size={cfg.data.image_size}, resolution={a.resolution})...")
        # Full constructor kwargs (minus image_size, stored at ckpt top level). Built once
        # and reused as the checkpoint's `arch` so eval can rebuild via PatchSetCNN(
        # image_size=ckpt["image_size"], **ckpt["arch"]) with zero drift.
        arch = {
            "resolution": a.resolution, "enc_dims": list(a.enc_dims),
            "e": a.e, "h": a.h, "l": a.l, "a": a.a,
            "thinking_rows": a.thinking_rows, "residual_decay": a.residual_decay,
            "fourier_bands": a.get("fourier_bands", 8),
            "query_self_attn": a.get("query_self_attn", False),
            "context_id_embed": a.get("context_id_embed", False),
            "max_context": a.get("max_context", 16),
            "resolutions": list(a.resolutions) if a.get("resolutions", None) is not None else None,
            "refine_mode": a.get("refine_mode", "reencode"),
            "refine_memory": a.get("refine_memory", False),
        }
        model = PatchSetCNN(image_size=cfg.data.image_size, **arch)
        return model, name, {"arch": arch}
    raise ValueError(f"unknown model {name!r} (universeg | patchset_cnn)")


def train_epoch(model, loader, optimizers, scheduler, cfg, epoch) -> tuple[float, float, float, float, float, dict, int | None]:
    model.train()
    total, n = 0.0, 0
    soft_sum = soft_cnt = hard_sum = hard_cnt = 0.0   # on-device running sums (synced once)
    cos_sum = cos_cnt = topk_sum = topk_cnt = 0.0
    topk_k = int(cfg.train.get("topk_k", 16))
    res_list = _as_res_list(cfg.eval.get("ds_metric_res", None))   # fixed-res pooled Dice
    dsr_sums = {R: {"ss": 0.0, "sc": 0.0, "hs": 0.0, "hc": 0.0} for R in res_list}
    low_res = None                              # non-native model's coarse logit side length
    lr_hard_sum = lr_hard_cnt = 0.0            # hard Dice at that native low res (patchset_cnn)
    refine_hard_sum = refine_hard_cnt = 0.0   # refine hard Dice at Rf (on the crop)
    fused_hard_sum = fused_hard_cnt = 0.0      # fused hard Dice at Rf (stitched full image)
    fused_res = None
    pbar = tqdm(loader, desc=f"train e{epoch}")
    for batch in pbar:
        if batch is None:
            continue
        img = batch["image"].to(DEVICE, non_blocking=True)        # (B,1,H,W)
        lbl = batch["label"].to(DEVICE, non_blocking=True).float()
        cin = batch["context_in"].to(DEVICE, non_blocking=True)   # (B,K,1,H,W)
        cout = batch["context_out"].to(DEVICE, non_blocking=True)

        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
        with _autocast():
            out = model(img, context_in=cin, context_out=cout, mode="train")
            logit = out["final_logit"].float()                    # (B,1,h,w)
        target = _target_like(lbl, logit)                         # pooled to logit res
        bce = F.binary_cross_entropy_with_logits(logit, target)
        dice = soft_dice_loss(torch.sigmoid(logit), target)
        loss = bce + cfg.train.dice_weight * dice

        if out.get("refine_logit") is not None:            # multi-level: add the refine loss
            rlogit = out["refine_logit"].float()
            rtarget = crop_resize(lbl, out["refine_origin"], int(out["refine_crop"]),
                                  rlogit.shape[-1], mode="bilinear")   # soft cropped GT at T
            rbce = F.binary_cross_entropy_with_logits(rlogit, rtarget)
            rdice = soft_dice_loss(torch.sigmoid(rlogit), rtarget)
            loss = loss + float(cfg.train.get("refine_loss_weight", 1.0)) * (
                rbce + cfg.train.dice_weight * rdice)

        loss.backward()
        if cfg.train.get("grad_clip", None):
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
        for opt in optimizers:
            opt.step()
        scheduler.step()

        # Monitoring (not the loss): downsampled soft Dice (prob vs occupancy at logit
        # res) + native-resolution hard Dice (preds upscaled to original res vs GT).
        # cossim only adds signal below native res; at native res it is redundant with Dice.
        with torch.no_grad():
            prob = torch.sigmoid(logit)
            prob_nat = _upsample_to(prob, lbl.shape[-2:])
            rg = refine_geometry(out, lbl)
            # Refine models: `dice` (native hard) is scored on the fused native stitch (last
            # level), not the coarse pred. Other monitors stay on the coarse pass for now.
            pred_nat = rg["fused"] if rg is not None else prob_nat
            ss, sc = _soft_sum(prob, target)
            hs, hc = _hard_sum(pred_nat, lbl)
            soft_sum += ss; soft_cnt += sc; hard_sum += hs; hard_cnt += hc
            if logit.shape[-2:] != lbl.shape[-2:]:   # low res: cossim + top-k + hard@nativeres
                low_res = logit.shape[-1]
                cs, cc = _cos_sum(prob, target)
                ts, tc = _topk_sum(prob, target, topk_k)
                cos_sum += cs; cos_cnt += cc; topk_sum += ts; topk_cnt += tc
                hlr, clr = _hard_sum(prob, (target >= 0.5).float())   # hard Dice at the coarse grid
                lr_hard_sum += hlr; lr_hard_cnt += clr
            # fixed-res pooled Dice (UniverSeg only — see main(): ds_metric_res is ignored for
            # non-native patchset_cnn/refine, which already report their own coarse grid).
            for R in (res_list if logit.shape[-2:] == lbl.shape[-2:] else []):
                p_r = F.adaptive_avg_pool2d(prob_nat, (R, R))
                g_r = F.adaptive_avg_pool2d(lbl, (R, R))
                ss_r, sc_r = _soft_sum(p_r, g_r)
                hs_r, hc_r = _hard_sum(p_r, (g_r >= 0.5).float())
                d = dsr_sums[R]
                d["ss"] += ss_r; d["sc"] += sc_r; d["hs"] += hs_r; d["hc"] += hc_r
            if rg is not None:
                rh, rhc = _hard_sum(rg["refine_prob"], (rg["refine_target"] >= 0.5).float())
                fh, fhc = _hard_sum(rg["fused_R"], (rg["gt_R"] >= 0.5).float())
                refine_hard_sum += rh; refine_hard_cnt += rhc
                fused_hard_sum += fh; fused_hard_cnt += fhc
                fused_res = rg["Rf"]

        total += loss.item()
        n += 1
        post = {"loss": f"{total / n:.4f}",
                # native hard Dice on the fused-upsampled pred (fused for refine) — matches train/dice
                "dice": f"{float(hard_sum) / max(float(hard_cnt), 1):.4f}",
                "lr": f"{scheduler.get_last_lr()[0]:.1e}"}
        if cos_cnt:
            post["cos"] = f"{float(cos_sum) / float(cos_cnt):.4f}"
            post[f"top{topk_k}"] = f"{float(topk_sum) / float(topk_cnt):.4f}"
        pbar.set_postfix(**post)
    dsr_out = {}
    for R in (res_list if low_res is None else []):   # universeg only (native): fixed res @R
        d = dsr_sums[R]
        dsr_out[f"dice_ds@{R}"] = float(d["hs"]) / max(float(d["hc"]), 1)
        dsr_out[f"dice_ds_soft@{R}"] = float(d["ss"]) / max(float(d["sc"]), 1)
    if low_res is not None:                     # patchset_cnn: tag its native coarse grid too
        dsr_out[f"dice_ds@{low_res}"] = float(lr_hard_sum) / max(float(lr_hard_cnt), 1)
        dsr_out[f"dice_ds_soft@{low_res}"] = float(soft_sum) / max(float(soft_cnt), 1)
    if fused_res is not None:                   # refine model: per-level + fused train Dice
        dsr_out[f"dice@{fused_res}"] = float(refine_hard_sum) / max(float(refine_hard_cnt), 1)
        dsr_out[f"dice_fused@{fused_res}"] = float(fused_hard_sum) / max(float(fused_hard_cnt), 1)
    return (total / max(n, 1),
            float(soft_sum) / max(float(soft_cnt), 1),
            float(hard_sum) / max(float(hard_cnt), 1),
            float(cos_sum) / float(cos_cnt) if cos_cnt else float("nan"),
            float(topk_sum) / float(topk_cnt) if topk_cnt else float("nan"),
            dsr_out, low_res)


def _select_metric(summary: dict) -> tuple[str, float]:
    """Checkpoint-selection metric from a val summary. A refine model (identified by a
    dice_fused@R key) selects on native `dice` — which for refine is the fused prediction
    scored at full resolution. Otherwise: cossim if present, else dice. Returns
    (metric_key_without_'/mean', mean_value)."""
    is_refine = any(k.startswith("dice_fused@") and k.endswith("/mean") for k in summary)
    if is_refine:
        return "dice", summary.get("dice/mean", float("nan"))
    cossim = next((k for k in summary
                   if k.startswith("cossim@") and k.endswith("/mean")), None)
    if cossim is not None:
        return cossim[: -len("/mean")], summary[cossim]
    return "dice", summary.get("dice/mean", float("nan"))


@hydra.main(config_path="../../configs/experiment/2d", config_name="universeg_train",
            version_base=None)
def main(cfg: DictConfig):
    random.seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)
    torch.manual_seed(cfg.train.seed)
    if DEVICE.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    train_loader = make_loader(build_dataset(cfg, "train"), cfg, "train", shuffle=True)
    val_loader = make_loader(build_dataset(cfg, "val"), cfg, "val", shuffle=False)

    model, model_name, ckpt_meta = build_model(cfg)

    # Optional warm-start for retraining. Accepts a bare state_dict (old format) or the
    # new {"model": ...} checkpoint; strips the _orig_mod. prefix left by torch.compile.
    # Tolerant load: keep only tensors whose name AND shape match the current model, so a
    # checkpoint from a slightly different config still warm-starts (cf. pfn_seg.py).
    if cfg.train.get("checkpoint", None):
        raw = torch.load(cfg.train.checkpoint, map_location="cpu", weights_only=False)
        sd = raw["model"] if isinstance(raw, dict) and "model" in raw else raw
        # `replace` (not `removeprefix`): compiling a submodule (model.transformer) leaves the
        # `_orig_mod.` prefix MID-key (transformer._orig_mod.…), not just leading.
        sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
        model_sd = model.state_dict()
        compatible = {k: v for k, v in sd.items()
                      if k in model_sd and v.shape == model_sd[k].shape}
        skipped = [k for k in sd if k not in compatible]
        fresh = [k for k in model_sd if k not in compatible]
        model.load_state_dict(compatible, strict=False)
        print(f"Warm-start from {cfg.train.checkpoint}: "
              f"loaded {len(compatible)}/{len(model_sd)} tensors")
        if skipped:
            print(f"  skipped (shape mismatch / not in model): {skipped}")
        if fresh:
            print(f"  kept freshly initialized (not in checkpoint): {fresh}")

    model = model.to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {n_params/1e6:.2f}M")

    # Compile only the transformer submodule (like multilevel/train.py compiles its trainable
    # PatchSetPFNs): it is pure tensor ops so it graph-compiles cleanly, whereas the encoder's
    # crop/pool/grid_sample (adaptive_avg_pool2d with data-dependent windows) would graph-break.
    # The encoder still runs eager. UniverSeg has no `.transformer`, so it is left untouched.
    if cfg.arch.get("compile", False) and hasattr(model, "transformer"):
        model.transformer = torch.compile(model.transformer, dynamic=True)
        # Muon's Newton–Schulz orthogonalization is pure tensor ops → compile it too
        # (cf. pfn_seg.py); the encoder's crop/pool/grid_sample stays eager.
        import pfn_train
        pfn_train._newtonschulz5_batched = torch.compile(pfn_train._newtonschulz5_batched)
        print("Compiled model.transformer + Newton–Schulz (dynamic=True); "
              "encoder + bbox crop/pool run eager")

    # ── Optimizers ────────────────────────────────────────────────────────────
    # patchset_cnn trains with Muon on its transformer 2D weight matrices (Newton-
    # Schulz orthogonalized grads) + AdamW on everything else + LAWA checkpoint
    # averaging (cf. pfn_seg.py). universeg has no `transformer` submodule, so its
    # Muon group is empty and it falls back to plain AdamW with no LAWA — its path
    # is unchanged. The cosine+warmup scheduler drives AdamW only; Muon is unscheduled.
    is_patchset = (model_name == "patchset_cnn")
    muon_params = [p for n, p in model.named_parameters()
                   if p.requires_grad and p.ndim == 2 and "transformer" in n]
    adam_params = [p for n, p in model.named_parameters()
                   if p.requires_grad and not (p.ndim == 2 and "transformer" in n)]
    optimizer = torch.optim.AdamW(adam_params, lr=cfg.train.lr,
                                  weight_decay=cfg.train.get("adam_wd", 0.01))
    optimizers = [optimizer]
    if is_patchset and muon_params:
        optimizers.append(Muon(
            muon_params,
            lr=cfg.train.get("muon_lr_scale", 0.1) * cfg.train.lr,
            momentum=cfg.train.get("muon_momentum", 0.96),
            weight_decay=cfg.train.get("muon_wd", 0.1),
        ))
        print(f"Muon on {len(muon_params)} transformer matrices, "
              f"AdamW on {len(adam_params)} other tensors, LAWA k={cfg.train.get('lawa_k', 10)}")
    steps_per_epoch = max(1, len(train_loader))
    total_steps = cfg.train.epochs * steps_per_epoch
    warmup_steps = int(cfg.train.get("warmup_epochs", 1) * steps_per_epoch)

    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * prog))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # LAWA checkpoint-averaging buffer (patchset_cnn only): a CPU state_dict is pushed
    # each epoch; at eval the queue is averaged into the model, evaluated, then the raw
    # training weights are restored so optimization continues from them.
    lawa_queue = collections.deque(maxlen=cfg.train.get("lawa_k", 10)) if is_patchset else None

    # ── wandb / output dir ──────────────────────────────────────────────────────
    wandb_enabled = bool(cfg.wandb.get("enabled", True))
    # name=None → wandb auto-generates the run name; the checkpoint dir uses it (cf. pfn_seg.py).
    run_name = cfg.wandb.name
    # Log the whole resolved Hydra config (nested: train.*, data.*, synth.*, arch.*, ...)
    # as the single source of truth — no hand-maintained subset to drift out of sync.
    run = wandb.init(
        project=cfg.wandb.project, name=run_name,
        mode="online" if wandb_enabled else "disabled",
        config=OmegaConf.to_container(cfg, resolve=True),
    )
    run_name = (wandb.run.name if wandb.run is not None else None) or cfg.wandb.name or model_name
    date_str = datetime.date.today().strftime("%Y-%m-%d")
    out_dir = Path(cfg.eval.out_dir) / f"{date_str}_{run_name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "best.pt"
    print(f"Checkpoints -> {ckpt_path}")

    # ── train loop ──────────────────────────────────────────────────────────────
    best_dice = -1.0
    metric = "cossim"  # checkpoint-selection metric (→ "dice" when preds are at native res)
    topk_k = int(cfg.train.get("topk_k", 16))
    for epoch in range(cfg.train.epochs):
        t0 = time.perf_counter()
        loss, train_soft, train_hard, train_cos, train_topk, train_dsr, train_lowres = train_epoch(
            model, train_loader, optimizers, scheduler, cfg, epoch)
        log = {"epoch": epoch, "train/dice": train_hard, "train/loss": loss,
               "train/lr": scheduler.get_last_lr()[0],
               "time/epoch_s": time.perf_counter() - t0}
        if train_lowres is None:   # native model (universeg): full-res soft Dice. Named
            # dice_soft (no "ds" — not downsampled), parallel to train/dice (full-res hard)
            # and matching pfn_seg.py's train/dice_soft = soft Dice at native pred resolution.
            log["train/dice_soft"] = train_soft
        if not math.isnan(train_cos):   # coarse-grid metrics: tag with the token grid T=low_res
            log[f"train/cossim@{train_lowres}"] = train_cos
        if not math.isnan(train_topk):
            log[f"train/top{topk_k}@{train_lowres}"] = train_topk
        log.update({f"train/{k}": v for k, v in train_dsr.items()})   # dice_ds@R / dice_ds_soft@R

        if lawa_queue is not None:   # push this epoch's raw weights to the LAWA buffer
            lawa_queue.append({k: v.cpu().clone() for k, v in model.state_dict().items()})

        if epoch % cfg.train.get("eval_every", 1) == 0 or epoch == cfg.train.epochs - 1:
            # Eval (and any checkpoint saved below) uses LAWA-averaged weights; raw
            # training weights are restored after so optimization continues from them.
            saved = lawa_average(lawa_queue, model, DEVICE) if lawa_queue is not None else None
            summary, sample_table, _ = validate(
                model, val_loader, topk_k=topk_k, epoch=epoch,
                compute_flops=(epoch == 0),
                ds_metric_res=cfg.eval.get("ds_metric_res", None),
                per_group=bool(cfg.eval.get("log_per_class", True)))
            metric, mean_dice = _select_metric(summary)
            log.update(summary)
            log["val/samples"] = sample_table
            ds_soft_key = next((k for k in summary
                                if k.startswith("dice_ds_soft") and k.endswith("/mean")), None)
            ds_soft = summary.get(ds_soft_key, float("nan")) if ds_soft_key else float("nan")
            topk_key = next((k for k in summary
                             if k.startswith(f"top{topk_k}@") and k.endswith("/mean")), None)
            topk_val = summary.get(topk_key, float("nan")) if topk_key else float("nan")
            tqdm.write(f"  [e{epoch}] train loss={loss:.4f}"
                       f"  val {metric}={mean_dice:.4f} top{topk_k}={topk_val:.4f}"
                       f" ds_soft={ds_soft:.4f}"
                       f" dice={summary.get('dice/mean', float('nan')):.4f}"
                       f"  (best {metric}={max(best_dice, mean_dice):.4f})")
            if mean_dice > best_dice:
                best_dice = mean_dice
                torch.save({
                    "model": model.state_dict(),
                    "model_name": model_name,
                    "image_size": cfg.data.image_size,
                    "context_size": cfg.data.context_size,
                    "best_val_dice": best_dice, "epoch": epoch,
                    "data": OmegaConf.to_container(cfg.data, resolve=True),
                    "synth": OmegaConf.to_container(cfg.synth, resolve=True) if cfg.get("synth", None) else None,
                    **ckpt_meta,
                }, ckpt_path)
                log[f"val/best_{metric}"] = best_dice
            if saved is not None:   # restore raw training weights after LAWA-averaged eval
                model.load_state_dict(saved)
        wandb.log(log)

    print(f"Done. Best val {metric}={best_dice:.4f}. Checkpoint: {ckpt_path}")
    run.finish()


if __name__ == "__main__":
    main()
