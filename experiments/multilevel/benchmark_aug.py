"""
Runtime benchmark for 3D augmentation pipelines.

Compares:
  - Custom PyTorch (current codebase): batched task aug + per-vol intensity aug
  - MONAI (>=1.5): equivalent transforms applied per-volume
  - batchgenerators (via nnunetv2): batch-level spatial transforms

Key finding to measure: the custom pipeline batches all K+1 volumes into a single
grid_sample call for task aug, while MONAI/BG process volumes sequentially — this
architectural difference dominates cost at large N or large image sizes.

Usage
-----
    python experiments/multilevel/benchmark_aug.py
    python experiments/multilevel/benchmark_aug.py image_size=64,64,64 n_reps=5
    python experiments/multilevel/benchmark_aug.py presets=nnunet,synth_equiv K=1,3
    python experiments/multilevel/benchmark_aug.py image_size=128,128,128 K=1,3 n_reps=50
"""

import json
import math
import random
import resource
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# CLI arg parsing (same pattern as profile_aug.py)
# ---------------------------------------------------------------------------
_raw = sys.argv[1:]

def _get(key, default):
    v = next((a.split("=", 1)[1] for a in _raw if a.startswith(f"{key}=")), None)
    return v if v is not None else default

N_REPS     = int(_get("n_reps", 30))
_sz        = _get("image_size", "128,128,128")
IMAGE_SIZES = [tuple(int(x) for x in _get("image_size", "128,128,128").split(","))]
K_VALUES   = [int(k) for k in _get("K", "1,3").split(",")]
PRESETS    = [p.strip() for p in _get("presets", "nnunet,multiverseg,synth_equiv").split(",")]
N_WORKERS  = int(_get("n_workers", 20))   # for throughput estimate only

sys.argv = [sys.argv[0]]

from src.augmentations import (
    apply_task_aug,
    apply_intensity_aug,
    apply_synth_aug,
    _make_affine_theta,
    _apply_grid,
    _separable_gaussian_blur_3d,
)
from src.totalseg_dataset import CT_NORM_MIN, CT_NORM_MAX


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def timeit(fn, n: int) -> tuple[float, float]:
    """Mean ± std wall-clock ms over n reps; discards first third as warmup."""
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1e3)
    warm = times[n // 3:]
    mean = sum(warm) / len(warm)
    std  = (sum((x - mean) ** 2 for x in warm) / len(warm)) ** 0.5
    return float(mean), float(std)


def fresh_images(N, D, H, W) -> torch.Tensor:
    return torch.randn(N, 1, D, H, W) * 0.5


def fresh_masks(N, D, H, W) -> torch.Tensor:
    m = torch.zeros(N, D, H, W, dtype=torch.long)
    m[:, D // 4:3 * D // 4, H // 4:3 * H // 4, W // 4:3 * W // 4] = 1
    return m


def detect_libraries() -> dict[str, bool]:
    libs: dict[str, bool] = {"custom": True, "monai": False, "batchgenerators": False}
    try:
        import monai  # noqa: F401
        libs["monai"] = True
    except ImportError:
        pass
    try:
        from batchgenerators.augmentations.spatial_transformations import augment_spatial  # noqa: F401
        libs["batchgenerators"] = True
    except ImportError:
        pass
    return libs


def hr(char="─", width=70) -> str:
    return char * width


# ---------------------------------------------------------------------------
# Custom PyTorch — full pipeline + per-transform breakdown
# ---------------------------------------------------------------------------

def bench_custom(aug_cfg, image_size: tuple, K: int, n_reps: int) -> dict:
    D, H, W = image_size
    N = K + 1
    results: dict[str, tuple[float, float]] = {}

    # ---- Task aug (full, N vols jointly) -----------------------------------
    def _run_task():
        apply_task_aug(fresh_images(N, D, H, W), fresh_masks(N, D, H, W), aug_cfg.task)

    m, s = timeit(_run_task, n_reps)
    results["task_full"] = (m, s)

    # ---- Intensity aug (full, N vols independently) -----------------------
    def _run_intensity():
        imgs = fresh_images(N, D, H, W)
        for i in range(N):
            apply_intensity_aug(imgs[i], aug_cfg.intensity)

    m, s = timeit(_run_intensity, n_reps)
    results["intensity_full"] = (m, s)

    # ---- Combined real-data pipeline (task + intensity) -------------------
    def _run_combined():
        imgs = fresh_images(N, D, H, W)
        msks = fresh_masks(N, D, H, W)
        imgs, msks = apply_task_aug(imgs, msks, aug_cfg.task)
        for i in range(N):
            imgs[i] = apply_intensity_aug(imgs[i], aug_cfg.intensity)

    m, s = timeit(_run_combined, n_reps)
    results["combined_full"] = (m, s)
    results["combined_per_vol"] = (m / N, s / N)

    # ---- Synth aug (N independent calls) ----------------------------------
    def _run_synth():
        imgs = fresh_images(N, D, H, W)
        msks = fresh_masks(N, D, H, W)
        for i in range(N):
            apply_synth_aug(imgs[i].clone(), msks[i].clone(), aug_cfg.synth)

    m, s = timeit(_run_synth, n_reps)
    results["synth_full"] = (m, s)
    results["synth_per_vol"] = (m / N, s / N)

    # ---- Per-transform breakdown (task aug, forced on) --------------------
    imgs_ref = fresh_images(N, D, H, W)
    msks_ref = fresh_masks(N, D, H, W)

    # Flip
    m, s = timeit(lambda: (imgs_ref.flip(2), msks_ref.flip(1)), n_reps)
    results["flip_d"] = (m, s)
    m, s = timeit(lambda: (imgs_ref.flip(3), msks_ref.flip(2)), n_reps)
    results["flip_h"] = (m, s)
    m, s = timeit(lambda: (imgs_ref.flip(4), msks_ref.flip(3)), n_reps)
    results["flip_w"] = (m, s)

    # Affine (batched over N)
    acfg     = aug_cfg.task.affine
    max_rad  = acfg.max_angle_deg * math.pi / 180.0
    def _affine():
        theta = _make_affine_theta(
            random.uniform(-max_rad, max_rad),
            random.uniform(-max_rad, max_rad),
            random.uniform(-max_rad, max_rad),
            random.uniform(acfg.scale_min, acfg.scale_max),
            random.uniform(-acfg.max_translate, acfg.max_translate),
            random.uniform(-acfg.max_translate, acfg.max_translate),
            random.uniform(-acfg.max_translate, acfg.max_translate),
        ).expand(N, -1, -1)
        grid = F.affine_grid(theta, (N, 1, D, H, W), align_corners=False)
        return _apply_grid(fresh_images(N, D, H, W), fresh_masks(N, D, H, W), grid)

    m, s = timeit(_affine, n_reps)
    results["affine_batched_N"] = (m, s)
    results["affine_per_vol"] = (m / N, s / N)

    # Elastic (batched over N)
    ecfg = aug_cfg.task.elastic
    gs   = max(getattr(ecfg, "grid_scale", 8), 2)
    sd, sh, sw = max(D // gs, 2), max(H // gs, 2), max(W // gs, 2)
    def _elastic():
        disp = torch.randn(1, 3, sd, sh, sw) * ecfg.alpha
        disp = F.interpolate(disp, size=(D, H, W), mode="trilinear", align_corners=False)
        disp = disp.permute(0, 2, 3, 4, 1).expand(N, -1, -1, -1, -1)
        tid  = torch.eye(3, 4, dtype=torch.float32).unsqueeze(0).expand(N, -1, -1)
        grid = (F.affine_grid(tid, (N, 1, D, H, W), align_corners=False) + disp).clamp(-1, 1)
        return _apply_grid(fresh_images(N, D, H, W), fresh_masks(N, D, H, W), grid)

    m, s = timeit(_elastic, n_reps)
    results["elastic_batched_N"] = (m, s)
    results["elastic_per_vol"] = (m / N, s / N)

    # Gaussian blur (intensity, single vol)
    blcfg = aug_cfg.intensity.gaussian_blur
    img1  = fresh_images(1, D, H, W)[0]
    sigma_mid = (blcfg.sigma_range[0] + blcfg.sigma_range[1]) / 2.0
    m, s = timeit(lambda: _separable_gaussian_blur_3d(img1, sigma_mid), n_reps)
    results["blur_per_vol"] = (m, s)

    return results


# ---------------------------------------------------------------------------
# MONAI equivalents  (per-volume, no batching across N)
# ---------------------------------------------------------------------------

def bench_monai(aug_cfg, image_size: tuple, K: int, n_reps: int) -> dict | None:
    try:
        from monai.transforms import (
            RandAffine,
            Rand3DElastic,
            RandGaussianNoise,
            RandGaussianSmooth,
        )
    except ImportError:
        return None

    D, H, W = image_size
    N = K + 1
    results: dict[str, tuple[float, float]] = {}

    acfg     = aug_cfg.task.affine
    ecfg     = aug_cfg.task.elastic
    ncfg     = aug_cfg.intensity.gaussian_noise
    blcfg    = aug_cfg.intensity.gaussian_blur
    max_rad  = acfg.max_angle_deg * math.pi / 180.0
    scale_rng = (acfg.scale_max - acfg.scale_min) / 2.0

    # MONAI RandAffine: (C, D, H, W) tensor, affine grid computed per call
    affine_tfm = RandAffine(
        prob=1.0,
        rotate_range=(max_rad, max_rad, max_rad),
        scale_range=(scale_rng, scale_rng, scale_rng),
        translate_range=(
            int(acfg.max_translate * D),
            int(acfg.max_translate * H),
            int(acfg.max_translate * W),
        ),
        mode="bilinear",
        padding_mode="border",
    )

    def _monai_affine_N():
        img = fresh_images(1, D, H, W)[0]   # (1, D, H, W)
        for _ in range(N):
            affine_tfm(img.clone())

    m, s = timeit(_monai_affine_N, n_reps)
    results["monai_affine_N_calls"] = (m, s)
    results["monai_affine_per_vol"] = (m / N, s / N)

    # MONAI Rand3DElastic
    gs   = max(getattr(ecfg, "grid_scale", 8), 2)
    # sigma_range: physical smoothing (voxels); magnitude_range: peak displacement (voxels)
    elastic_tfm = Rand3DElastic(
        prob=1.0,
        sigma_range=(gs * 0.8, gs * 1.6),
        magnitude_range=(ecfg.alpha * 20, ecfg.alpha * 80),   # alpha in norm coords → voxels
        mode="bilinear",
        padding_mode="border",
    )

    def _monai_elastic_N():
        img = fresh_images(1, D, H, W)[0]
        for _ in range(N):
            elastic_tfm(img.clone())

    m, s = timeit(_monai_elastic_N, n_reps)
    results["monai_elastic_N_calls"] = (m, s)
    results["monai_elastic_per_vol"] = (m / N, s / N)

    # MONAI Gaussian noise
    noise_tfm = RandGaussianNoise(prob=1.0, mean=0.0, std=ncfg.max_std / 2)

    def _monai_noise_N():
        img = fresh_images(1, D, H, W)[0]
        for _ in range(N):
            noise_tfm(img.clone())

    m, s = timeit(_monai_noise_N, n_reps)
    results["monai_noise_per_vol"] = (m / N, s / N)

    # MONAI Gaussian smooth
    sigma_mid  = (blcfg.sigma_range[0] + blcfg.sigma_range[1]) / 2.0
    smooth_tfm = RandGaussianSmooth(
        sigma_x=(sigma_mid, sigma_mid),
        sigma_y=(sigma_mid, sigma_mid),
        sigma_z=(sigma_mid, sigma_mid),
        prob=1.0,
    )

    def _monai_smooth_N():
        img = fresh_images(1, D, H, W)[0]
        for _ in range(N):
            smooth_tfm(img.clone())

    m, s = timeit(_monai_smooth_N, n_reps)
    results["monai_smooth_per_vol"] = (m / N, s / N)

    return results


# ---------------------------------------------------------------------------
# batchgenerators equivalents  (batch-level, N volumes at once)
# ---------------------------------------------------------------------------

def bench_batchgenerators(aug_cfg, image_size: tuple, K: int, n_reps: int) -> dict | None:
    try:
        from batchgenerators.augmentations.spatial_transformations import augment_spatial
        from batchgenerators.augmentations.noise_augmentations import augment_gaussian_noise
    except ImportError:
        return None

    import numpy as np

    D, H, W = image_size
    N = K + 1
    results: dict[str, tuple[float, float]] = {}

    acfg    = aug_cfg.task.affine
    ncfg    = aug_cfg.intensity.gaussian_noise
    max_rad = acfg.max_angle_deg * math.pi / 180.0

    # batchgenerators spatial aug: processes (N, C, D, H, W) batch in one call
    def _bg_spatial():
        imgs = fresh_images(N, D, H, W).numpy()          # (N, 1, D, H, W)
        msks = fresh_masks(N, D, H, W).unsqueeze(1).float().numpy()
        augment_spatial(
            imgs, msks,
            patch_size=(D, H, W),
            do_elastic_deform=True,
            alpha=(0, 900),
            sigma=(9, 13),
            do_rotation=True,
            angle_x=(-max_rad, max_rad),
            angle_y=(-max_rad, max_rad),
            angle_z=(-max_rad, max_rad),
            do_scale=True,
            scale=(acfg.scale_min, acfg.scale_max),
            border_mode_data="nearest",
            border_mode_seg="constant",
            random_crop=False,
        )

    m, s = timeit(_bg_spatial, n_reps)
    results["bg_spatial_total"] = (m, s)
    results["bg_spatial_per_vol"] = (m / N, s / N)

    # batchgenerators noise (batch)
    def _bg_noise():
        imgs = fresh_images(N, D, H, W).numpy()
        augment_gaussian_noise(imgs, noise_variance=(0, ncfg.max_std ** 2))

    m, s = timeit(_bg_noise, n_reps)
    results["bg_noise_total"] = (m, s)
    results["bg_noise_per_vol"] = (m / N, s / N)

    return results


# ---------------------------------------------------------------------------
# Scaling sweep: vary K (fixed preset + image_size)
# ---------------------------------------------------------------------------

def bench_k_scaling(aug_cfg, image_size: tuple, k_values: list, n_reps: int) -> list[dict]:
    rows = []
    for K in k_values:
        N = K + 1
        D, H, W = image_size

        def _combined():
            imgs = fresh_images(N, D, H, W)
            msks = fresh_masks(N, D, H, W)
            imgs, msks = apply_task_aug(imgs, msks, aug_cfg.task)
            for i in range(N):
                imgs[i] = apply_intensity_aug(imgs[i], aug_cfg.intensity)

        def _synth():
            imgs = fresh_images(N, D, H, W)
            msks = fresh_masks(N, D, H, W)
            for i in range(N):
                apply_synth_aug(imgs[i].clone(), msks[i].clone(), aug_cfg.synth)

        m_c, s_c = timeit(_combined, n_reps)
        m_s, s_s = timeit(_synth, n_reps)
        rows.append({
            "K": K, "N": N,
            "combined_ms": round(m_c, 2), "combined_per_vol_ms": round(m_c / N, 2),
            "synth_ms": round(m_s, 2), "synth_per_vol_ms": round(m_s / N, 2),
            "combined_sps": round(1000.0 / (m_c / N), 1),
            "synth_sps": round(1000.0 / (m_s / N), 1),
        })
    return rows


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def _row(name: str, mean_ms: float, std_ms: float,
         baseline_ms: float | None = None, label_width: int = 42) -> str:
    speedup = ""
    if baseline_ms is not None and baseline_ms > 1e-3:
        speedup = f"  {mean_ms / baseline_ms:.2f}×"
    bar = "▪" * max(1, min(40, int(mean_ms / 5)))
    return f"  {name:<{label_width}}  {mean_ms:8.2f} ± {std_ms:5.2f} ms{speedup}  {bar}"


def print_section(title: str, results: dict, N: int,
                  monai_res: dict | None, bg_res: dict | None) -> None:
    print(f"\n{hr('─')}")
    print(f"  {title}")
    print(hr("─"))

    # Combined pipeline (main metric)
    if "combined_full" in results:
        m, s = results["combined_full"]
        print(_row("combined pipeline (task+intensity)", m, s))
        mp, sp = results["combined_per_vol"]
        sps = 1000.0 / mp if mp > 0 else 0.0
        print(f"    per volume:  {mp:.2f} ± {sp:.2f} ms  →  {sps:.0f} sps (single-core)")

    # Synth pipeline
    if "synth_full" in results:
        m, s = results["synth_full"]
        baseline = results.get("combined_full", (None,))[0]
        print(_row("synth pipeline  (K+1 independent)", m, s, baseline))
        mp, sp = results["synth_per_vol"]
        sps = 1000.0 / mp if mp > 0 else 0.0
        print(f"    per volume:  {mp:.2f} ± {sp:.2f} ms  →  {sps:.0f} sps (single-core)")

    # Per-transform breakdown
    print()
    print("  Task aug breakdown (forced-on, N vols jointly):")
    baseline_affine = None
    for key in ("flip_d", "flip_h", "flip_w"):
        if key in results:
            m, s = results[key]
            print(_row(f"    {key}", m, s))
    if "affine_batched_N" in results:
        m, s = results["affine_batched_N"]
        mp, sp = results["affine_per_vol"]
        baseline_affine = mp
        print(_row(f"    affine  (batched N={N})", m, s))
        print(f"      per vol:  {mp:.2f} ± {sp:.2f} ms", end="")
        if monai_res and "monai_affine_per_vol" in monai_res:
            mm, ms = monai_res["monai_affine_per_vol"]
            print(f"   MONAI: {mm:.2f} ± {ms:.2f} ms  ({mm/mp:.2f}× vs custom)", end="")
        if bg_res and "bg_spatial_per_vol" in bg_res:
            bm, bs = bg_res["bg_spatial_per_vol"]
            print(f"   BG: {bm:.2f} ± {bs:.2f} ms  ({bm/mp:.2f}× vs custom)", end="")
        print()
    if "elastic_batched_N" in results:
        m, s = results["elastic_batched_N"]
        mp, sp = results["elastic_per_vol"]
        print(_row(f"    elastic (batched N={N})", m, s))
        print(f"      per vol:  {mp:.2f} ± {sp:.2f} ms", end="")
        if monai_res and "monai_elastic_per_vol" in monai_res:
            mm, ms_ = monai_res["monai_elastic_per_vol"]
            print(f"   MONAI: {mm:.2f} ± {ms_:.2f} ms  ({mm/mp:.2f}× vs custom)", end="")
        print()

    print()
    print("  Intensity aug breakdown (forced-on, per vol):")
    if "blur_per_vol" in results:
        m, s = results["blur_per_vol"]
        print(_row("    gaussian_blur (custom separable)", m, s))
        if monai_res and "monai_smooth_per_vol" in monai_res:
            mm, ms_ = monai_res["monai_smooth_per_vol"]
            print(f"    gaussian_blur (MONAI GaussianSmooth):  {mm:.2f} ± {ms_:.2f} ms  ({mm/m:.2f}× vs custom)")


def print_k_scaling(rows: list[dict]) -> None:
    print(f"\n{hr('─')}")
    print("  K scaling (combined real pipeline vs synth pipeline):")
    print(f"  {'K':>3}  {'N':>3}  {'combined':>12}  {'per-vol':>9}  {'sps':>6}  {'synth':>12}  {'per-vol':>9}  {'sps':>6}")
    print(f"  {'-'*3}  {'-'*3}  {'-'*12}  {'-'*9}  {'-'*6}  {'-'*12}  {'-'*9}  {'-'*6}")
    for r in rows:
        print(
            f"  {r['K']:>3}  {r['N']:>3}  {r['combined_ms']:>10.1f}ms"
            f"  {r['combined_per_vol_ms']:>7.1f}ms  {r['combined_sps']:>6.0f}"
            f"  {r['synth_ms']:>10.1f}ms  {r['synth_per_vol_ms']:>7.1f}ms  {r['synth_sps']:>6.0f}"
        )


def print_worker_estimate(rows: list[dict], n_workers: int, batch_size: int = 8) -> None:
    print(f"\n  Worker throughput estimate  (n_workers={n_workers}, batch_size={batch_size}):")
    for r in rows:
        K = r["K"]
        ms_per_vol_combined = r["combined_per_vol_ms"]
        ms_per_vol_synth    = r["synth_per_vol_ms"]
        sps_comb = 1000.0 / ms_per_vol_combined * n_workers
        sps_synt = 1000.0 / ms_per_vol_synth    * n_workers
        lat_comb = batch_size / sps_comb * 1000
        lat_synt = batch_size / sps_synt * 1000
        print(f"    K={K}  combined: {sps_comb:6.0f} sps → {lat_comb:.1f} ms/batch  |"
              f"  synth: {sps_synt:6.0f} sps → {lat_synt:.1f} ms/batch")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    libs = detect_libraries()
    print(hr("═"))
    print("  Augmentation Runtime Benchmark")
    print(hr("═"))
    print(f"  Presets     : {PRESETS}")
    print(f"  Image sizes : {IMAGE_SIZES}")
    print(f"  K values    : {K_VALUES}")
    print(f"  n_reps      : {N_REPS}")
    print(f"  Libraries   : {', '.join(k for k, v in libs.items() if v)}")

    all_results = {}

    for preset in PRESETS:
        cfg_path = ROOT / "configs" / "augmentations" / f"{preset}.yaml"
        if not cfg_path.exists():
            print(f"\n[SKIP] Config not found: {cfg_path}")
            continue

        aug_cfg = OmegaConf.load(cfg_path).augmentations

        for image_size in IMAGE_SIZES:
            D, H, W = image_size
            print(f"\n{hr('═')}")
            print(f"  preset={preset}   image_size={image_size}")
            print(hr("═"))

            # --- per-K breakdown -------------------------------------------
            for K in K_VALUES:
                N = K + 1
                title = f"K={K}  N={N}  (1 target + {K} context)"

                custom_res = bench_custom(aug_cfg, image_size, K, N_REPS)
                monai_res  = bench_monai(aug_cfg, image_size, K, N_REPS) if libs["monai"] else None
                bg_res     = bench_batchgenerators(aug_cfg, image_size, K, N_REPS) if libs["batchgenerators"] else None

                print_section(title, custom_res, N, monai_res, bg_res)

                all_results[(preset, str(image_size), K, "custom")] = {
                    k: v for k, v in custom_res.items()
                }
                if monai_res:
                    all_results[(preset, str(image_size), K, "monai")] = monai_res
                if bg_res:
                    all_results[(preset, str(image_size), K, "batchgenerators")] = bg_res

            # --- K scaling summary -----------------------------------------
            k_rows = bench_k_scaling(aug_cfg, image_size, K_VALUES, max(N_REPS // 2, 5))
            print_k_scaling(k_rows)
            print_worker_estimate(k_rows, n_workers=N_WORKERS, batch_size=8)
            all_results[(preset, str(image_size), "k_scaling")] = k_rows

    # --- Save JSON ----------------------------------------------------------
    out_dir = ROOT / "results" / "aug_benchmark"
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = out_dir / f"aug_benchmark_{timestamp}.json"

    def _serialise(v):
        if isinstance(v, tuple):
            return list(v)
        return v

    serializable = {}
    for k, v in all_results.items():
        str_key = str(k)
        if isinstance(v, dict):
            serializable[str_key] = {kk: _serialise(vv) for kk, vv in v.items()}
        elif isinstance(v, list):
            serializable[str_key] = v

    json_path.write_text(json.dumps(serializable, indent=2))
    print(f"\n{hr()}")
    print(f"  Saved → {json_path}")
    print(hr())


if __name__ == "__main__":
    main()
