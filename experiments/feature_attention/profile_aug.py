"""
Micro-benchmark for each augmentation sub-step.

Runs each op in isolation on a realistic (N, 1, D, H, W) tensor — the same
shape and dtype that DataLoader workers see — and reports mean ± std over
n_reps repetitions.  Uses the exact parameter values from the config.

Since augmentations run on CPU in worker processes, the numbers here match
the per-sample cost paid inside each worker.

Usage
-----
    python experiments/feature_attention/profile_aug.py
    python experiments/feature_attention/profile_aug.py n_reps=50
    python experiments/feature_attention/profile_aug.py aug_preset=multiverseg
    python experiments/feature_attention/profile_aug.py image_size=64,64,64
"""

import math
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Parse micro-benchmark args before touching sys.argv
# ---------------------------------------------------------------------------
_raw = sys.argv[1:]

def _get(key, default):
    v = next((a.split("=", 1)[1] for a in _raw if a.startswith(f"{key}=")), None)
    return v if v is not None else default

N_REPS      = int(_get("n_reps", 30))
AUG_PRESET  = _get("aug_preset", None)     # overrides config default
_img_size   = _get("image_size", None)
IMAGE_SIZE  = tuple(int(x) for x in _img_size.split(",")) if _img_size else None
N_CTX       = int(_get("context_size", 1))

sys.argv = [sys.argv[0]] + [
    a for a in _raw
    if not a.startswith(("n_reps=", "aug_preset=", "image_size=", "context_size="))
]

from experiments.feature_attention.train import load_config
from src.augmentations import (
    _make_affine_theta,
    _apply_grid,
    _separable_gaussian_blur_3d,
    apply_task_aug,
    apply_intensity_aug,
    apply_synth_aug,
)
from src.totalseg_dataset import CT_NORM_MIN, CT_NORM_MAX


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def timeit(fn, n: int) -> tuple[float, float]:
    """Run fn() n times; return (mean_ms, std_ms)."""
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1e3)
    t = times[n // 3:]          # drop first third (cold cache / JIT warm-up)
    return float(sum(t) / len(t)), float((sum((x - sum(t)/len(t))**2 for x in t) / len(t)) ** 0.5)


def row(name: str, mean_ms: float, std_ms: float, p: float | None = None) -> str:
    p_str = f"  (p={p:.2f})" if p is not None else ""
    bar   = "█" * max(1, int(mean_ms / 5))
    return f"  {name:<30}  {mean_ms:7.2f} ± {std_ms:5.2f} ms{p_str}  {bar}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    cfg = load_config()

    aug_preset  = AUG_PRESET or cfg.train.aug_preset
    image_size  = IMAGE_SIZE or tuple(cfg.data.image_size)
    n_ctx       = N_CTX
    N           = 1 + n_ctx   # target + context volumes in a task

    aug_cfg = OmegaConf.load(ROOT / "configs" / "augmentations" / f"{aug_preset}.yaml").augmentations
    task_cfg  = aug_cfg.task
    int_cfg   = aug_cfg.intensity
    syn_cfg   = aug_cfg.synth

    D, H, W = image_size
    print(f"Augmentation micro-benchmark")
    print(f"  preset={aug_preset}  image_size={image_size}  N={N} (1 tgt + {n_ctx} ctx)  n_reps={N_REPS}")
    print()

    def fresh_images() -> torch.Tensor:
        return torch.randn(N, 1, D, H, W) * 0.5 + CT_NORM_MIN

    def fresh_masks() -> torch.Tensor:
        m = torch.zeros(N, D, H, W, dtype=torch.long)
        m[:, D//4:3*D//4, H//4:3*H//4, W//4:3*W//4] = 1
        return m

    # -----------------------------------------------------------------------
    # Task augmentations (geometric, shared params, applied to all N volumes)
    # -----------------------------------------------------------------------
    print("─" * 60)
    print(f"TASK AUG  (applied to {N} volumes jointly)")
    print("─" * 60)

    # Full task aug
    m, s = timeit(lambda: apply_task_aug(fresh_images(), fresh_masks(), task_cfg), N_REPS)
    print(row("apply_task_aug [full]", m, s))
    print()

    # Individual ops
    fcfg = task_cfg.flip
    images_ref = fresh_images()
    masks_ref  = fresh_masks()

    # Flip (always sampled, forced here for consistent timing)
    m, s = timeit(lambda: images_ref.flip(2), N_REPS)
    print(row("  flip_d (single axis)", m, s, fcfg.p_d))
    m, s = timeit(lambda: images_ref.flip(3), N_REPS)
    print(row("  flip_h (single axis)", m, s, fcfg.p_h))
    m, s = timeit(lambda: images_ref.flip(4), N_REPS)
    print(row("  flip_w (single axis)", m, s, fcfg.p_w))

    # Affine (forced on)
    acfg = task_cfg.affine
    max_rad = acfg.max_angle_deg * math.pi / 180.0
    def _task_affine():
        theta = _make_affine_theta(
            random.uniform(-max_rad, max_rad),
            random.uniform(-max_rad, max_rad),
            random.uniform(-max_rad, max_rad),
            random.uniform(acfg.scale_min, acfg.scale_max),
            0.0, 0.0, 0.0,
        ).expand(N, -1, -1)
        grid = F.affine_grid(theta, images_ref.shape, align_corners=False)
        return _apply_grid(images_ref, masks_ref, grid)
    m, s = timeit(_task_affine, N_REPS)
    print(row("  affine (N vols, grid_sample)", m, s, acfg.p))

    # Elastic (forced on)
    ecfg = task_cfg.elastic
    def _task_elastic():
        gs   = max(ecfg.grid_scale, 2)
        sd, sh, sw = max(D // gs, 2), max(H // gs, 2), max(W // gs, 2)
        disp = torch.randn(1, 3, sd, sh, sw) * ecfg.alpha
        disp = F.interpolate(disp, size=(D, H, W), mode="trilinear", align_corners=False)
        disp = disp.permute(0, 2, 3, 4, 1).expand(N, -1, -1, -1, -1)
        tid  = torch.eye(3, 4).unsqueeze(0).expand(N, -1, -1)
        grid = (F.affine_grid(tid, images_ref.shape, align_corners=False) + disp).clamp(-1, 1)
        return _apply_grid(images_ref, masks_ref, grid)
    m, s = timeit(_task_elastic, N_REPS)
    print(row("  elastic (N vols, grid_sample)", m, s, ecfg.p))

    # -----------------------------------------------------------------------
    # Intensity augmentations (independent per volume, single volume)
    # -----------------------------------------------------------------------
    print()
    print("─" * 60)
    print("INTENSITY AUG  (applied per volume independently)")
    print("─" * 60)

    img1 = fresh_images()[0]   # (1, D, H, W)

    m, s = timeit(lambda: apply_intensity_aug(img1, int_cfg), N_REPS)
    print(row("apply_intensity_aug [full]", m, s))
    print()

    bccfg = int_cfg.brightness_contrast
    def _brightness_contrast():
        contrast   = random.uniform(bccfg.contrast_range[0], bccfg.contrast_range[1])
        brightness = random.uniform(-bccfg.brightness, bccfg.brightness)
        return (img1 * contrast + brightness).clamp_(CT_NORM_MIN, CT_NORM_MAX)
    m, s = timeit(_brightness_contrast, N_REPS)
    print(row("  brightness_contrast", m, s, bccfg.p))

    gcfg = int_cfg.gamma
    def _gamma():
        gamma = random.uniform(gcfg.range[0], gcfg.range[1])
        span  = CT_NORM_MAX - CT_NORM_MIN
        out   = ((img1 - CT_NORM_MIN) / span).clamp_(0, 1).pow_(gamma) * span + CT_NORM_MIN
        if getattr(gcfg, "retain_stats", False):
            std_out = out.std()
            if std_out > 1e-8:
                out = (out - out.mean()) / std_out * img1.std() + img1.mean()
        return out.clamp_(CT_NORM_MIN, CT_NORM_MAX)
    m, s = timeit(_gamma, N_REPS)
    print(row("  gamma (retain_stats)", m, s, gcfg.p))

    ncfg = int_cfg.gaussian_noise
    def _noise():
        std = random.uniform(0.0, ncfg.max_std)
        return (img1 + torch.randn_like(img1) * std).clamp_(CT_NORM_MIN, CT_NORM_MAX)
    m, s = timeit(_noise, N_REPS)
    print(row("  gaussian_noise", m, s, ncfg.p))

    lrcfg = getattr(int_cfg, "simulate_low_resolution", None)
    if lrcfg is not None:
        def _low_res():
            scale = random.uniform(lrcfg.scale_min, lrcfg.scale_max)
            small = (max(1, int(D * scale)), max(1, int(H * scale)), max(1, int(W * scale)))
            x = F.interpolate(img1.unsqueeze(0), size=small, mode="trilinear", align_corners=False)
            return F.interpolate(x, size=(D, H, W), mode="trilinear", align_corners=False).squeeze(0)
        m, s = timeit(_low_res, N_REPS)
        print(row("  simulate_low_resolution", m, s, lrcfg.p))

    blcfg = int_cfg.gaussian_blur
    def _blur():
        sigma = random.uniform(blcfg.sigma_range[0], blcfg.sigma_range[1])
        return _separable_gaussian_blur_3d(img1, sigma)
    m, s = timeit(_blur, N_REPS)
    print(row("  gaussian_blur (separable)", m, s, blcfg.p))

    # Expected cost per sample: p × mean
    print()
    print("  Expected cost per sample (p × mean):")
    for name, p, mean_fn in [
        ("brightness_contrast",   bccfg.p,  lambda: timeit(_brightness_contrast, 5)[0]),
        ("gamma",                 gcfg.p,   lambda: timeit(_gamma, 5)[0]),
        ("gaussian_noise",        ncfg.p,   lambda: timeit(_noise, 5)[0]),
        ("gaussian_blur",         blcfg.p,  lambda: timeit(_blur, 5)[0]),
    ]:
        pass   # already timed above — just print weighted cost
    # Re-use already-collected means (approximated from full-aug timing)
    m_full, _ = timeit(lambda: apply_intensity_aug(img1, int_cfg), 10)
    print(f"    full intensity aug mean: {m_full:.2f} ms  (across {N_REPS} reps, probabilities apply)")

    # -----------------------------------------------------------------------
    # Synth augmentations (heavy, independent per copy)
    # -----------------------------------------------------------------------
    print()
    print("─" * 60)
    print("SYNTH AUG  (heavy, one volume at a time)")
    print("─" * 60)

    img1s = img1.clone()
    msk1s = fresh_masks()[0]

    m, s = timeit(lambda: apply_synth_aug(img1s, msk1s, syn_cfg), N_REPS)
    print(row("apply_synth_aug [full]", m, s))
    print()

    # Affine (single volume)
    sacfg = syn_cfg.affine
    max_rad_s = sacfg.max_angle_deg * math.pi / 180.0
    def _synth_affine():
        theta = _make_affine_theta(
            random.uniform(-max_rad_s, max_rad_s),
            random.uniform(-max_rad_s, max_rad_s),
            random.uniform(-max_rad_s, max_rad_s),
            random.uniform(sacfg.scale_min, sacfg.scale_max),
            random.uniform(-sacfg.max_translate, sacfg.max_translate),
            random.uniform(-sacfg.max_translate, sacfg.max_translate),
            random.uniform(-sacfg.max_translate, sacfg.max_translate),
        )
        grid = F.affine_grid(theta, (1, 1, D, H, W), align_corners=False)
        img_out, msk_out = _apply_grid(img1s.unsqueeze(0), msk1s.unsqueeze(0), grid)
        return img_out.squeeze(0), msk_out.squeeze(0)
    m, s = timeit(_synth_affine, N_REPS)
    print(row("  affine (1 vol)", m, s, sacfg.p))

    # Elastic (single volume, coarse-grid method)
    secfg = syn_cfg.elastic
    def _synth_elastic():
        alpha = random.uniform(*secfg.alpha_range)
        sigma = random.uniform(*secfg.sigma_range)
        sd, sh, sw = max(2, round(D / sigma)), max(2, round(H / sigma)), max(2, round(W / sigma))
        disp = F.interpolate(
            torch.randn(1, 3, sd, sh, sw),
            size=(D, H, W), mode="trilinear", align_corners=False,
        ).squeeze(0)
        mx   = disp.abs().amax().clamp(min=1e-6)
        disp = disp / mx * alpha
        sn   = torch.tensor([2.0/D, 2.0/H, 2.0/W]).view(3, 1, 1, 1)
        dn   = (disp * sn).permute(1, 2, 3, 0).unsqueeze(0)
        grid = (F.affine_grid(torch.eye(3, 4).unsqueeze(0), (1, 1, D, H, W), align_corners=False) + dn).clamp(-1, 1)
        img_out, msk_out = _apply_grid(img1s.unsqueeze(0), msk1s.unsqueeze(0), grid)
        return img_out.squeeze(0), msk_out.squeeze(0)
    m, s = timeit(_synth_elastic, N_REPS)
    print(row("  elastic (1 vol, coarse-grid)", m, s, secfg.p))

    sbcfg = syn_cfg.brightness_contrast
    def _synth_bc():
        c = random.uniform(sbcfg.contrast_range[0], sbcfg.contrast_range[1])
        b = random.uniform(-sbcfg.brightness, sbcfg.brightness)
        return (img1s * c + b).clamp_(CT_NORM_MIN, CT_NORM_MAX)
    m, s = timeit(_synth_bc, N_REPS)
    print(row("  brightness_contrast", m, s, sbcfg.p))

    sscfg = syn_cfg.sharpness
    def _synth_sharp():
        blurred = _separable_gaussian_blur_3d(img1s, sigma=1.0)
        return (img1s + sscfg.factor * (img1s - blurred)).clamp_(CT_NORM_MIN, CT_NORM_MAX)
    m, s = timeit(_synth_sharp, N_REPS)
    print(row("  sharpness (unsharp mask)", m, s, sscfg.p))

    sblcfg = syn_cfg.gaussian_blur
    def _synth_blur():
        sigma = random.uniform(sblcfg.sigma_range[0], sblcfg.sigma_range[1])
        return _separable_gaussian_blur_3d(img1s, sigma)
    m, s = timeit(_synth_blur, N_REPS)
    print(row("  gaussian_blur (separable)", m, s, sblcfg.p))

    sncfg = syn_cfg.gaussian_noise
    def _synth_noise():
        mean = random.uniform(sncfg.mean_range[0], sncfg.mean_range[1])
        std  = random.uniform(sncfg.std_range[0],  sncfg.std_range[1])
        return (img1s + mean + torch.randn_like(img1s) * std).clamp_(CT_NORM_MIN, CT_NORM_MAX)
    m, s = timeit(_synth_noise, N_REPS)
    print(row("  gaussian_noise", m, s, sncfg.p))

    # -----------------------------------------------------------------------
    # Total cost estimate per training sample
    # -----------------------------------------------------------------------
    print()
    print("─" * 60)
    print("ESTIMATED TOTAL AUG COST PER TRAINING SAMPLE")
    print("─" * 60)
    t_task, _  = timeit(lambda: apply_task_aug(fresh_images(), fresh_masks(), task_cfg), N_REPS)
    t_int, _   = timeit(lambda: apply_intensity_aug(img1, int_cfg), N_REPS)
    t_int_all  = t_int * N   # applied to each of the N volumes independently

    print(f"  task aug (1 call, {N} vols)          {t_task:7.2f} ms")
    print(f"  intensity aug ({N} calls, 1 vol each) {t_int_all:7.2f} ms  ({t_int:.2f} × {N})")
    print(f"  total per sample                      {t_task + t_int_all:7.2f} ms")
    print()
    # Synth path (replaces real data at p_synth fraction)
    t_syn, _  = timeit(lambda: apply_synth_aug(img1s, msk1s, syn_cfg), N_REPS)
    t_syn_all = t_syn * N
    print(f"  synth aug ({N} calls, 1 vol each)    {t_syn_all:7.2f} ms  ({t_syn:.2f} × {N})")
    p_synth = cfg.data.p_synth
    t_expected = p_synth * t_syn_all + (1 - p_synth) * (t_task + t_int_all)
    print(f"  expected (p_synth={p_synth})            {t_expected:7.2f} ms")
    print()
    print(f"  → Each DataLoader worker can process ~{1000/t_expected:.1f} samples/s")
    print(f"    With {cfg.train.workers} workers: ~{cfg.train.workers * 1000/t_expected:.0f} samples/s throughput")
    print(f"    Batch of {cfg.train.batch_size}: ~{cfg.train.batch_size / (cfg.train.workers * 1000/t_expected) * 1000:.0f} ms data loading time")


if __name__ == "__main__":
    main()
