"""
ImagePFN training script: in-context 2D segmentation via dual-axis transformer.

Trains on MedSegBench train split; evaluates periodically on val split using LAWA
checkpoint averaging.  Implements all nanoTabPFN techniques adapted to images:
  - Muon optimizer for transformer 2D weight matrices
  - AdamW + cosine LR for everything else
  - Residual decay, thinking rows (in the model)
  - LAWA: average last K checkpoints at eval time only

Usage:
    python experiments/2d/pfn_seg.py
    python experiments/2d/pfn_seg.py train.lr=5e-4 arch.l=4
    python experiments/2d/pfn_seg.py data.context_size=3 arch.thinking_rows=16
    python experiments/2d/pfn_seg.py data.dataset=abdomenus
"""

import collections
import math
import os
import random
import socket
import sys
import tempfile
from collections import defaultdict
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
import torch.nn as nn
import torch.nn.functional as F
import wandb
from omegaconf import DictConfig
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm

_ROOT = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, _ROOT)
# Import patch_icl modules BEFORE common.py inserts ic_segmentation into sys.path.
# ic_segmentation has its own src/__init__.py which would shadow our src/ if
# imported first.  By importing here we cache the correct modules in sys.modules;
# common.py's own "from src.datasets..." then finds the cached version and succeeds.
from src.datasets.medsegbench import MedSegBenchDataset
from src.models.pfn_seg_2d import ImagePFN

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DEVICE, TaggedDataset, collate, downsample_mask, hard_dice, soft_dice, log_summary


# ── Muon optimizer ────────────────────────────────────────────────────────────

def _newtonschulz5_batched(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Batched approximate matrix orthogonalization via Newton-Schulz iteration."""
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16()
    X = X / (X.norm(dim=(1, 2), keepdim=True) + eps)
    if X.size(1) > X.size(2):
        X = X.transpose(1, 2)
    for _ in range(steps):
        A = X @ X.transpose(1, 2)
        B = A @ X
        X = a * X + b * B + c * A @ B
    if G.size(1) > G.size(2):
        X = X.transpose(1, 2)
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    """
    Muon optimizer for hidden-layer 2D weight matrices.
    Applies Newton-Schulz orthogonalization to gradients before the update.
    QKV matrices of shape (3m, m) are processed as a batch of 3.
    """
    def __init__(self, params, lr: float = 3e-4, momentum: float = 0.95,
                 weight_decay: float = 0.0, steps: int = 5):
        super().__init__(params, dict(lr=lr, momentum=momentum,
                                      weight_decay=weight_decay, steps=steps))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr, mu, wd, ns = group['lr'], group['momentum'], group['weight_decay'], group['steps']
            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if 'buf' not in state:
                    state['buf'] = torch.zeros_like(g)
                buf = state['buf']
                buf.mul_(mu).add_(g)
                g = g.add(buf, alpha=mu)  # Nesterov

                # Fused QKV: shape (3m, m) → batch of 3 matrices
                if g.ndim == 2 and g.size(0) == 3 * g.size(1):
                    g_batch = g.view(3, g.size(1), g.size(1))
                    g_orth  = _newtonschulz5_batched(g_batch, steps=ns)
                    g_orth  = g_orth.view_as(g)
                    scale   = g.size(1) ** 0.5
                else:
                    g_orth = _newtonschulz5_batched(g.unsqueeze(0), steps=ns).squeeze(0)
                    scale  = max(g.size(0), g.size(1)) ** 0.5

                p.data.add_(g_orth, alpha=-lr * scale)
                if wd > 0:
                    p.data.mul_(1 - lr * wd)


# ── Augmentation ─────────────────────────────────────────────────────────────

def augment(
    images: torch.Tensor,  # (B, T, 1, H, W) float32 on device
    masks:  torch.Tensor,  # (B, T, 1, H, W) float32 on device
    K:      int,           # context count; query is at index K
    cfg,                   # cfg.aug
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched GPU augmentation. Geometric on context pairs; intensity on all images."""
    B, T, _, H, W = images.shape
    dev = images.device
    BK  = B * K

    # ── Geometric (context pairs only, joint image+mask) ───────────────────────
    c_imgs = images[:, :K].reshape(BK, 1, H, W)
    c_msks = masks[:, :K].reshape(BK, 1, H, W)

    g = cfg.geometric
    if g.hflip_p > 0:
        m = torch.rand(BK, 1, 1, 1, device=dev) < g.hflip_p
        c_imgs = torch.where(m, c_imgs.flip(-1), c_imgs)
        c_msks = torch.where(m, c_msks.flip(-1), c_msks)

    if g.vflip_p > 0:
        m = torch.rand(BK, 1, 1, 1, device=dev) < g.vflip_p
        c_imgs = torch.where(m, c_imgs.flip(-2), c_imgs)
        c_msks = torch.where(m, c_msks.flip(-2), c_msks)

    if g.rotate.p > 0:
        active = torch.rand(BK, device=dev) < g.rotate.p
        angles = (torch.rand(BK, device=dev) * 2 - 1) * g.rotate.max_angle_deg * active.float()
        rad    = torch.deg2rad(angles)
        cos_t, sin_t = torch.cos(rad), torch.sin(rad)
        z     = torch.zeros_like(cos_t)
        theta = torch.stack([cos_t, -sin_t, z, sin_t, cos_t, z], dim=1).reshape(BK, 2, 3)
        grid  = F.affine_grid(theta, (BK, 1, H, W), align_corners=False)
        c_imgs = F.grid_sample(c_imgs, grid, mode="bilinear", align_corners=False, padding_mode="zeros")
        c_msks = F.grid_sample(c_msks, grid, mode="nearest",  align_corners=False, padding_mode="zeros")

    images = torch.cat([c_imgs.reshape(B, K, 1, H, W), images[:, K:]], dim=1)
    masks  = torch.cat([c_msks.reshape(B, K, 1, H, W), masks[:, K:]],  dim=1)

    # ── Intensity (all images independently, masks unchanged) ──────────────────
    BT   = B * T
    imgs = images.reshape(BT, 1, H, W)
    ic   = cfg.intensity

    if ic.brightness.p > 0:
        m = torch.rand(BT, 1, 1, 1, device=dev) < ic.brightness.p
        d = (torch.rand(BT, 1, 1, 1, device=dev) * 2 - 1) * ic.brightness.max_delta
        imgs = torch.where(m, (imgs + d).clamp(0, 1), imgs)

    if ic.contrast.p > 0:
        lo, hi = ic.contrast.range
        m  = torch.rand(BT, 1, 1, 1, device=dev) < ic.contrast.p
        s  = torch.rand(BT, 1, 1, 1, device=dev) * (hi - lo) + lo
        mu = imgs.mean(dim=(-2, -1), keepdim=True)
        imgs = torch.where(m, ((imgs - mu) * s + mu).clamp(0, 1), imgs)

    if ic.gamma.p > 0:
        lo, hi = ic.gamma.range
        m = torch.rand(BT, 1, 1, 1, device=dev) < ic.gamma.p
        g = torch.rand(BT, 1, 1, 1, device=dev) * (hi - lo) + lo
        imgs = torch.where(m, imgs.clamp(1e-6).pow(g), imgs)

    if ic.noise.p > 0:
        m = torch.rand(BT, 1, 1, 1, device=dev) < ic.noise.p
        n = torch.randn_like(imgs) * ic.noise.std
        imgs = torch.where(m, (imgs + n).clamp(0, 1), imgs)

    return imgs.reshape(B, T, 1, H, W), masks


# ── Data helpers ──────────────────────────────────────────────────────────────

def build_split_loader(cfg, split: str, shuffle: bool) -> DataLoader:
    datasets = [cfg.data.dataset] if cfg.data.dataset else None
    ds = MedSegBenchDataset(
        split=split,
        context_size=cfg.data.context_size,
        image_size=cfg.data.image_size,
        datasets=datasets,
    )
    if split == "val" and cfg.eval.max_per_label:
        groups: dict = {}
        for i, (name, _, lv) in enumerate(ds.samples):
            groups.setdefault((name, lv), []).append(i)
        keep = []
        for indices in groups.values():
            keep.extend(random.sample(indices, min(cfg.eval.max_per_label, len(indices))))
        ds.samples = [ds.samples[i] for i in sorted(keep)]
        print(f"Val: subsampled to {len(ds.samples)} samples")
    bs = cfg.train.batch_size if split == "train" else cfg.eval.batch_size
    nw = cfg.train.workers   if split == "train" else cfg.eval.workers
    max_train = cfg.data.get("max_train_samples", None)
    sampler = (
        RandomSampler(ds, replacement=False, num_samples=max_train)
        if split == "train" and max_train is not None
        else None
    )
    return DataLoader(
        TaggedDataset(ds),
        batch_size=bs,
        shuffle=(shuffle and sampler is None),
        sampler=sampler,
        num_workers=nw,
        collate_fn=collate,
        pin_memory=DEVICE.type == "cuda",
        persistent_workers=nw > 0,
        prefetch_factor=4 if nw > 0 else None,
    )


# ── Batch construction ────────────────────────────────────────────────────────

def make_model_inputs(batch: dict, device: torch.device):
    """Stack context + query into (B, K+1, 1, H, W) tensors."""
    ctx_in  = batch["context_in"]   # (B, K, 1, H, W)
    ctx_out = batch["context_out"]  # (B, K, 1, H, W)
    img     = batch["image"]        # (B, 1, H, W)
    K = ctx_in.shape[1]
    all_images = torch.cat([ctx_in, img.unsqueeze(1)], dim=1).to(device, non_blocking=True)
    all_masks  = torch.cat([ctx_out, torch.zeros_like(img.unsqueeze(1))], dim=1).to(device, non_blocking=True)
    return all_images, all_masks, K


# ── LAWA ─────────────────────────────────────────────────────────────────────

def lawa_average(queue: collections.deque, model: nn.Module, device: torch.device):
    """Average checkpoint queue into model weights; return original state for restore."""
    if len(queue) <= 1:
        return None
    avg = {k: sum(s[k].float() for s in queue) / len(queue) for k in queue[0]}
    avg = {k: v.to(dtype=queue[0][k].dtype, device=device) for k, v in avg.items()}
    saved = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(avg)
    return saved


# ── Loss ──────────────────────────────────────────────────────────────────────

def soft_dice_loss(p: torch.Tensor, t: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    """Per-sample soft Dice loss between probability map p and soft target t.

    Both (B, Hp, Hp) in [0, 1]; smoothing eps stabilises (near-)empty samples.
    Returns 1 − Dice averaged over the batch.
    """
    p = p.flatten(1).float()
    t = t.flatten(1).float()
    num = 2 * (p * t).sum(1) + eps
    den = p.sum(1) + t.sum(1) + eps
    return (1 - num / den).mean()


# ── Training epoch ────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizers, cfg, epoch: int) -> float:
    model.train()
    total_loss, n = 0.0, 0
    pbar = tqdm(loader, desc=f"train e{epoch}", leave=False, dynamic_ncols=True)
    for batch in pbar:
        if batch is None:
            continue
        all_images, all_masks, K = make_model_inputs(batch, DEVICE)
        if cfg.aug.enabled:
            all_images, all_masks = augment(all_images, all_masks, K, cfg.aug)
        gt = batch["label"].squeeze(1).float().to(DEVICE, non_blocking=True)

        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

        with torch.autocast(device_type=DEVICE.type, dtype=torch.bfloat16,
                            enabled=DEVICE.type == "cuda"):
            logits = model(all_images, all_masks, sep=K)          # (B, Hp, Hp)
            Hp = logits.shape[-1]
            # Soft patch-level target: avg-pool the native binary mask to the patch
            # grid → per-patch foreground fraction in [0, 1]. Supervise directly at
            # Hp (no upsample) so the objective matches the head's resolution.
            target = F.adaptive_avg_pool2d(gt.unsqueeze(1), (Hp, Hp)).squeeze(1)
            bce  = F.binary_cross_entropy_with_logits(logits, target)
            dice = soft_dice_loss(torch.sigmoid(logits.float()), target)
            loss = bce + cfg.train.dice_weight * dice

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
        for opt in optimizers:
            opt.step()

        total_loss += loss.item()
        n += 1
        pbar.set_postfix(loss=f"{total_loss / n:.4f}")

    return total_loss / max(n, 1)


# ── Evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_eval(model, loader, lawa_queue, cfg, epoch: int) -> float:
    """LAWA-averaged eval on val split. Returns mean Dice."""
    saved = lawa_average(lawa_queue, model, DEVICE)
    model.eval()

    per_ds:       dict[str, list[float]] = defaultdict(list)
    per_label:    dict[str, list[float]] = defaultdict(list)
    per_ds_ds:      dict[str, list[float]] = defaultdict(list)  # low-res hard dice
    per_ds_ds_soft: dict[str, list[float]] = defaultdict(list)  # low-res soft (shape) dice
    sample_rows = []
    H = cfg.data.image_size
    running_dice, nd = 0.0, 0

    pbar = tqdm(loader, desc=f"eval  e{epoch}", leave=False, dynamic_ncols=True)
    for batch in pbar:
        if batch is None:
            continue
        all_images, all_masks, K = make_model_inputs(batch, DEVICE)

        with torch.autocast(device_type=DEVICE.type, dtype=torch.bfloat16,
                            enabled=DEVICE.type == "cuda"):
            logits = model(all_images, all_masks, sep=K)          # (B, Hp, Hp)

        Hp = logits.shape[-1]
        preds_lowres = torch.sigmoid(logits.float()).cpu()        # (B, Hp, Hp)
        if Hp != H:
            preds = F.interpolate(
                preds_lowres.unsqueeze(1), size=(H, H),
                mode="bilinear", align_corners=False,
            ).squeeze(1)
        else:
            preds = preds_lowres

        for b in range(len(batch["dataset"])):
            gt       = batch["label"][b, 0]
            ds_name  = batch["dataset"][b]
            lv       = int(batch["label_value"][b])
            d        = hard_dice(preds[b], gt)
            per_ds[ds_name].append(d)
            per_label[f"{ds_name}/label_{lv}"].append(d)
            sample_rows.append((ds_name, int(batch["sample_idx"][b]), lv, d))
            if not np.isnan(d):
                running_dice += d
                nd += 1
            # Binarize the avg-pooled GT at >= 0.5 (majority vote) so the low-res
            # target isn't OR-dilated by partially-covered boundary cells — makes
            # this comparable to the native-res metric.
            gt_lowres = (downsample_mask(gt, Hp) >= 0.5).float()
            per_ds_ds[ds_name].append(hard_dice(preds_lowres[b], gt_lowres))
            # soft (shape) dice: continuous low-res pred vs soft (un-binarized) GT
            per_ds_ds_soft[ds_name].append(soft_dice(preds_lowres[b], downsample_mask(gt, Hp)))
        pbar.set_postfix(dice=f"{running_dice / max(nd, 1):.4f}")

    if saved is not None:
        model.load_state_dict(saved)

    # Aggregate
    valid = [s for scores in per_ds.values() for s in scores if not np.isnan(s)]
    mean_dice = float(np.mean(valid)) if valid else float("nan")

    tqdm.write(f"\n  [e{epoch}] mean Dice (val): {mean_dice:.4f}")
    for name in sorted(per_ds):
        sc = [s for s in per_ds[name] if not np.isnan(s)]
        tqdm.write(f"    {name:>25}  {float(np.mean(sc)) if sc else float('nan'):.4f}")

    valid_ds = [s for scores in per_ds_ds.values() for s in scores if not np.isnan(s)]
    mean_dice_ds = float(np.mean(valid_ds)) if valid_ds else float("nan")

    valid_ds_soft = [s for scores in per_ds_ds_soft.values() for s in scores if not np.isnan(s)]
    mean_dice_ds_soft = float(np.mean(valid_ds_soft)) if valid_ds_soft else float("nan")

    # Metric naming mirrors experiments/2d/eval.py: <prefix>/mean and
    # <prefix>/dataset/<name> for dice (native), dice_ds (low-res hard),
    # dice_ds_soft (low-res soft/shape).
    _dsmean = lambda v: float(np.mean([s for s in v if not np.isnan(s)]))
    wandb.log({"epoch": epoch,
               "dice/mean": mean_dice,
               **{f"dice/dataset/{k}": _dsmean(v) for k, v in per_ds.items() if v},
               "dice_ds/mean": mean_dice_ds,
               **{f"dice_ds/dataset/{k}": _dsmean(v) for k, v in per_ds_ds.items() if v},
               "dice_ds_soft/mean": mean_dice_ds_soft,
               **{f"dice_ds_soft/dataset/{k}": _dsmean(v) for k, v in per_ds_ds_soft.items() if v}})
    return mean_dice


# ── Main ──────────────────────────────────────────────────────────────────────

@hydra.main(config_path="../../configs/experiment/2d", config_name="pfn_seg", version_base=None)
def main(cfg: DictConfig):
    random.seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)
    torch.manual_seed(cfg.train.seed)
    if DEVICE.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True

    # ── Data ──────────────────────────────────────────────────────────────────
    print("Building data loaders...")
    train_loader = build_split_loader(cfg, "train", shuffle=True)
    val_loader   = build_split_loader(cfg, "val",   shuffle=False)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = ImagePFN(
        patch_size    = cfg.arch.patch_size,
        image_size    = cfg.data.image_size,
        e             = cfg.arch.e,
        h             = cfg.arch.h,
        l             = cfg.arch.l,
        a             = cfg.arch.a,
        thinking_rows = cfg.arch.thinking_rows,
        residual_decay= cfg.arch.residual_decay,
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"ImagePFN: {total_params:,} parameters")

    # Optional warm-start. Accepts a bare state_dict (old format) or the new
    # {"model": ...} dict; strips the _orig_mod. prefix left by torch.compile.
    # Load before compile so keys match the raw module.
    if cfg.train.get("checkpoint", None):
        raw = torch.load(cfg.train.checkpoint, map_location="cpu", weights_only=False)
        sd  = raw["model"] if isinstance(raw, dict) and "model" in raw else raw
        sd  = {k.removeprefix("_orig_mod."): v for k, v in sd.items()}
        model.load_state_dict(sd)
        print(f"Loaded weights from {cfg.train.checkpoint}")

    if cfg.arch.compile:
        model = torch.compile(model, dynamic=True)
        global _newtonschulz5_batched
        _newtonschulz5_batched = torch.compile(_newtonschulz5_batched)

    # ── Optimizers (Muon for transformer 2D weights, AdamW for rest) ──────────
    muon_params = [p for n, p in model.named_parameters()
                   if p.ndim == 2 and "transformer" in n]
    adam_params = [p for n, p in model.named_parameters()
                   if not (p.ndim == 2 and "transformer" in n)]

    optimizer_muon = Muon(
        muon_params,
        lr           = cfg.train.muon_lr_scale * cfg.train.lr,
        momentum     = cfg.train.muon_momentum,
        weight_decay = cfg.train.muon_wd,
    )
    optimizer_adam = torch.optim.AdamW(
        adam_params,
        lr           = cfg.train.lr,
        weight_decay = cfg.train.adam_wd,
    )
    # Cosine LR with linear warmup for AdamW
    def lr_lambda(epoch):
        if epoch < cfg.train.warmup_epochs:
            return (epoch + 1) / cfg.train.warmup_epochs
        t = (epoch - cfg.train.warmup_epochs) / max(cfg.train.epochs - cfg.train.warmup_epochs, 1)
        return 0.5 * (1 + math.cos(math.pi * t))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer_adam, lr_lambda)

    optimizers = [optimizer_muon, optimizer_adam]

    # ── LAWA ──────────────────────────────────────────────────────────────────
    lawa_queue: collections.deque = collections.deque(maxlen=cfg.train.lawa_k)

    # ── W&B ───────────────────────────────────────────────────────────────────
    run_name = cfg.wandb.name or (
        f"pfn_seg_P{cfg.arch.patch_size}_e{cfg.arch.e}_l{cfg.arch.l}"
        f"_k{cfg.data.context_size}_think{cfg.arch.thinking_rows}"
    )
    if cfg.wandb.enabled:
        wandb.init(
            project=cfg.wandb.project,
            name=run_name,
            config={
                "arch": dict(cfg.arch),
                "train": dict(cfg.train),
                "data": dict(cfg.data),
                "params": total_params,
            },
        )
    else:
        wandb.init(mode="disabled")

    # ── Checkpoint dir ────────────────────────────────────────────────────────
    ckpt_dir = Path(cfg.eval.out_dir) / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── Training loop ─────────────────────────────────────────────────────────
    best_dice = -1.0
    epoch_pbar = tqdm(range(1, cfg.train.epochs + 1), desc="epochs", dynamic_ncols=True)
    for epoch in epoch_pbar:
        loss = train_epoch(model, train_loader, optimizers, cfg, epoch)
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        wandb.log({"epoch": epoch, "train/loss": loss, "train/lr": current_lr})
        epoch_pbar.set_postfix(loss=f"{loss:.4f}", lr=f"{current_lr:.1e}", best=f"{best_dice:.4f}")

        # Push checkpoint to LAWA buffer every epoch
        lawa_queue.append({k: v.cpu().clone() for k, v in model.state_dict().items()})

        if epoch % cfg.train.eval_every == 0 or epoch == cfg.train.epochs:
            dice = run_eval(model, val_loader, lawa_queue, cfg, epoch)
            if dice > best_dice:
                best_dice = dice
                # Save LAWA-averaged weights as best checkpoint.
                # Embed arch + image_size so eval can rebuild the model from the
                # checkpoint alone (state_dict keys may carry a _orig_mod. prefix
                # when compiled — eval strips it on load).
                saved = lawa_average(lawa_queue, model, DEVICE)
                torch.save({
                    "model":        model.state_dict(),
                    "arch":         dict(cfg.arch),
                    "image_size":   cfg.data.image_size,
                    "context_size": cfg.data.context_size,
                }, ckpt_dir / "best.pt")
                if saved:
                    model.load_state_dict(saved)
                tqdm.write(f"  [best] dice={best_dice:.4f} → {ckpt_dir}/best.pt")
            epoch_pbar.set_postfix(loss=f"{loss:.4f}", lr=f"{current_lr:.1e}", best=f"{best_dice:.4f}")

    wandb.log({"best_dice": best_dice})
    wandb.finish()
    print(f"\nTraining complete. Best val Dice: {best_dice:.4f}")


if __name__ == "__main__":
    main()
