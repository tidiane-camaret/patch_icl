"""Plot the SAME real in-context items augmented on CPU vs GPU, with timing.

Builds real items through common.build_dataset with `augmentations.gpu=true` so the
dataset emits RAW volumes (no worker-side aug). The same raw batch is then augmented
two ways and shown side by side:

  CPU: src.augmentations.apply_task_aug (shared geometric per task) + apply_intensity_aug
       (per volume) — exactly what the DataLoader workers run today.
  GPU: src.gpu_augment.GpuAugmentor — the batched on-device engine.

CPU and GPU draw independent randomness (exact reproduction is a non-goal), so the two
rows show representative — not identical — augmentations of the same underlying scan.
All items are forced to real mode (aug_mode=0) for a clean geometric+intensity compare.

`sharpness` is dropped when present-but-unrunnable: the CPU intensity path has no
sharpness op, and exp-42's sharpness block lacks the `factor` the GPU op needs.

Usage:
    .venv_thor/bin/python experiments/3d/plot_aug_cpu_vs_gpu.py \
        experiment=42_reg_to_all \
        augmentations.intensity.gin.p=0.5 \
        augmentations.intensity.gaussian_blur.p=0.1 \
        augmentations.intensity.gaussian_noise.p=0.1 \
        augmentations.task.affine.scale_min=1.1 augmentations.task.affine.scale_max=1.1
"""
import argparse
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from hydra import compose, initialize_config_dir
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling `common` / `plot_dataset_items`

from src.totalseg_dataloader_incontext import incontext_collate_fn   # noqa: E402
from src.augmentations import (apply_task_aug, apply_intensity_aug,   # noqa: E402
                               apply_per_image_aug, apply_synth_aug)
from src.gpu_augment import GpuAugmentor, REAL, SYNTH, SELF_CONTEXT   # noqa: E402
from common import build_dataset, _self_context                      # noqa: E402
from plot_dataset_items import _best_slice, _overlay, _label_colours  # noqa: E402

_MODE_NAME = {REAL: "real", SYNTH: "synth", SELF_CONTEXT: "self-ctx"}


def sanitize(intensity):
    """Zero any op that cannot run on BOTH paths. Returns list of dropped op names."""
    dropped = []
    sh = getattr(intensity, "sharpness", None)
    if sh is not None and getattr(sh, "p", 0) and "factor" not in sh:
        sh.p = 0.0
        dropped.append("sharpness (CPU has no sharpness op; exp-42 block lacks 'factor')")
    return dropped


def cpu_augment(raw, aug, sc_per_image, sc_intensity):
    """Per-item CPU aug, mode-aware — mirrors the DataLoader worker paths:
    real -> shared task geo + per-vol intensity; synth -> apply_synth_aug per vol
    (independent heavy); self-ctx -> shared task, then per-image jitter + intensity
    on the K context clones (the pose divergence that makes contexts differ)."""
    B = raw["image"].shape[0]
    modes = raw["aug_mode"].tolist()
    img, lbl = raw["image"].clone(), raw["label"].clone()
    cin, cout = raw["context_in"].clone(), raw["context_out"].clone()
    for b in range(B):
        all_img = torch.cat([img[b:b + 1], cin[b]], 0)          # (K+1,1,D,H,W)
        all_msk = torch.cat([lbl[b:b + 1], cout[b]], 0)         # (K+1,D,H,W)
        if modes[b] == SYNTH:
            for i in range(all_img.shape[0]):
                all_img[i], all_msk[i] = apply_synth_aug(all_img[i], all_msk[i], aug.synth)
        else:
            all_img, all_msk = apply_task_aug(all_img, all_msk, aug.task)   # shared geo
            all_img[0] = apply_intensity_aug(all_img[0], aug.intensity)     # target intensity
            for i in range(1, all_img.shape[0]):                            # contexts
                if modes[b] == SELF_CONTEXT and sc_per_image:
                    all_img[i], all_msk[i] = apply_per_image_aug(all_img[i], all_msk[i], aug.per_image)
                if modes[b] != SELF_CONTEXT or sc_intensity:
                    all_img[i] = apply_intensity_aug(all_img[i], aug.intensity)
        img[b], cin[b] = all_img[0], all_img[1:]
        lbl[b], cout[b] = all_msk[0], all_msk[1:]
    return {"image": img, "label": lbl, "context_in": cin, "context_out": cout}


def gpu_augment(raw, aug_cfg, device, sc_per_image, sc_intensity, seed=0):
    """Batched GPU aug through the production GpuAugmentor (mode dispatch on aug_mode)."""
    batch = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in raw.items()}
    for k in ("image", "label", "context_in", "context_out", "aug_mode", "spacing"):
        if k in batch and torch.is_tensor(batch[k]):
            batch[k] = batch[k].to(device)
    aug = GpuAugmentor(aug_cfg, self_context_per_image=sc_per_image,
                       self_context_intensity=sc_intensity, seed=seed)
    out = aug(batch, training=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--split", default="train")
    ap.add_argument("--n_samples", type=int, default=8)
    ap.add_argument("--out", default="results/3d/aug_cpu_vs_gpu.png")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("-h", "--help", action="store_true")
    args, overrides = ap.parse_known_args()
    if args.help:
        ap.print_help(); return

    # Force raw emission so both paths augment the same underlying items.
    overrides = list(overrides) + ["augmentations.gpu=true"]
    with initialize_config_dir(config_dir=str(ROOT / "configs" / "experiment" / "3d"),
                               version_base="1.3"):
        cfg = compose(config_name="train", overrides=overrides)

    dropped = sanitize(cfg.augmentations.intensity)
    if dropped:
        print("WARNING dropped un-runnable ops:", "; ".join(dropped))

    ds = build_dataset(cfg, args.split)
    K = cfg.data.context_size
    N = min(args.n_samples, len(ds))
    num_labels = cfg.data.get("num_labels_per_sample", 1)

    loader = DataLoader(ds, batch_size=N, shuffle=True, num_workers=0,
                        collate_fn=incontext_collate_fn)
    raw = next(iter(loader))                                    # keep the dataset's real aug_mode

    # self-context toggles (per_image jitter / intensity) forwarded to both paths
    _, sc_int, sc_pi, _ = _self_context(cfg.data, "train")
    sc_pi, sc_int = bool(sc_pi), bool(sc_int)
    modes = raw["aug_mode"].tolist()
    print("aug_mode per item:", [_MODE_NAME[m] for m in modes],
          f"| self_context_per_image={sc_pi} intensity={sc_int}")

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    # time CPU (whole batch, single thread = per-worker slice) and GPU (batched, synced)
    torch.set_num_threads(1)
    _ = gpu_augment(raw, cfg.augmentations, device, sc_pi, sc_int)     # warm up cuda/cudnn
    t0 = time.perf_counter(); cpu = cpu_augment(raw, cfg.augmentations, sc_pi, sc_int)
    cpu_ms = (time.perf_counter() - t0) * 1e3
    t0 = time.perf_counter(); gpu = gpu_augment(raw, cfg.augmentations, device, sc_pi, sc_int)
    gpu_ms = (time.perf_counter() - t0) * 1e3

    # ---- plot: per item two rows (CPU / GPU), columns = target + K contexts ----
    vol_names = ["target"] + [f"ctx {k + 1}" for k in range(K)]
    n_cols = 1 + K
    fig, axes = plt.subplots(2 * N, n_cols, figsize=(2.4 * n_cols, 2.4 * 2 * N),
                             squeeze=False,
                             gridspec_kw={"hspace": 0.03, "wspace": 0.02})
    for v, name in enumerate(vol_names):
        axes[0, v].set_title(name, fontsize=9, pad=4)

    def draw(res, r, row_idx):
        colours = _label_colours(num_labels, None)
        vols = [(res["image"][row_idx], res["label"][row_idx])]
        vols += [(res["context_in"][row_idx, k], res["context_out"][row_idx, k]) for k in range(K)]
        for v, (im, mk) in enumerate(vols):
            img_sl, mask_sl = _best_slice(im.float(), mk)
            axes[r, v].imshow(_overlay(img_sl, mask_sl, colours))

    radii = raw.get("synth_radii_mm")   # present (finite) only for ellipse-synth items
    for row_idx in range(N):
        draw(cpu, 2 * row_idx, row_idx)
        draw(gpu, 2 * row_idx + 1, row_idx)
        subj = raw["subjects"][row_idx] if "subjects" in raw else f"item {row_idx}"
        name = raw["label_names"][row_idx] if "label_names" in raw else ""
        mode = _MODE_NAME[modes[row_idx]]
        is_ellipse = (radii is not None and torch.is_tensor(radii)
                      and bool(torch.isfinite(radii[row_idx]).all()))
        tag = f"[{mode}{' · ellipse' if is_ellipse else ''}]"
        axes[2 * row_idx, 0].set_ylabel(f"CPU\n{subj}\n{name}\n{tag}", fontsize=7,
                                        rotation=0, labelpad=48, va="center")
        axes[2 * row_idx + 1, 0].set_ylabel("GPU", fontsize=8, color="tab:blue",
                                            rotation=0, labelpad=48, va="center")

    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([])

    sp = cpu_ms / gpu_ms if gpu_ms > 0 else float("nan")
    dev_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
    fig.suptitle(
        f"aug CPU vs GPU  |  {N} items x {1 + K} vols of "
        f"{tuple(cfg.data.image_size)}  |  {dev_name}\n"
        f"CPU {cpu_ms:.0f} ms (1 thread, whole batch)   "
        f"GPU {gpu_ms:.1f} ms (batched)   speedup {sp:.0f}x"
        + ("   [sharpness dropped]" if dropped else ""),
        fontsize=11, y=1.005,
    )
    fig.tight_layout(h_pad=0.2, w_pad=0.2)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"CPU {cpu_ms:.0f} ms  |  GPU {gpu_ms:.1f} ms  |  speedup {sp:.0f}x")
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
