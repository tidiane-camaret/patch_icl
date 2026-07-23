"""
Config-driven 3D in-context eval — the harness twin of experiments/3d/train.py,
mirroring experiments/2d/eval_incontext.py. Evaluates one model (default medverse)
on the TotalSegmentator test split over a class list, reporting per-class Dice,
mean inference time, and GFLOPs. Shares the loader (common.make_eval_loader) and
eval loop (evaluate.evaluate_classes) with the rest of the 3D harness.

    python experiments/3d/eval.py
    python experiments/3d/eval.py eval.model=native_resenc \
        eval.checkpoint=results/checkpoints/resenc_in_context_best.pt
    python experiments/3d/eval.py dataset=omnisynth3d eval.split=val   # eval on omniSynth-3D
"""

import datetime
import json
import random
import sys
from pathlib import Path

import hydra
import torch
import wandb
from omegaconf import DictConfig, OmegaConf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling common/evaluate (dir '3d')

from data.totalseg_classes import resolve_classes
from src.benchmark_models import load_model
from common import DEVICE, _source_root
from evaluate import measure_flops, evaluate_classes, build_sample_table


def _build_model(cfg: DictConfig):
    """Instantiate the eval model. Medverse is inference-only (no checkpoint needed);
    other models need eval.checkpoint. Handles medverse ckpt/sw_roi that load_model
    cannot forward through its `ckpt_path` param."""
    name = cfg.eval.model
    image_size = tuple(cfg.data.image_size)
    if name == "medverse":
        from src.benchmark_models.medverse import MedverseModel
        mk = {}
        if cfg.eval.get("sw_roi_size"):
            mk["sw_roi_size"] = tuple(cfg.eval.sw_roi_size)
        if cfg.eval.get("medverse_ckpt"):
            mk["ckpt_path"] = cfg.eval.medverse_ckpt
        model = MedverseModel(device=DEVICE, **mk)
        # Fine-tuned checkpoint from experiments/3d/train.py (state under "model").
        if cfg.eval.get("checkpoint"):
            ckpt = torch.load(cfg.eval.checkpoint, map_location=DEVICE, weights_only=False)
            model.load_finetuned(ckpt["model"] if "model" in ckpt else ckpt)
            print(f"  Loaded fine-tuned medverse weights from {cfg.eval.checkpoint}")
        return model
    if name == "patchset3d":
        # PatchSet3D is used directly as the eval model (it provides .predict, the only
        # method the shared eval loop needs). The architecture is rebuilt from the
        # checkpoint's stored `arch` when present (new checkpoints), else from cfg.arch
        # (re-supply the same model=patchset3d arch.* overrides used at training).
        from train import build_model
        if not cfg.eval.get("checkpoint"):
            raise ValueError("eval.checkpoint is required for patchset3d")
        ckpt = torch.load(cfg.eval.checkpoint, map_location=DEVICE, weights_only=False)
        arch = ckpt.get("arch")
        from omegaconf import open_dict
        with open_dict(cfg):
            # build_model dispatches on top-level cfg.model, which is unset unless the
            # user passed +model=patchset3d — pin it so the arch-from-checkpoint path
            # (eval.model=patchset3d only) still builds a PatchSet3D, not medverse.
            cfg.model = "patchset3d"
            if arch is not None:
                cfg.arch = OmegaConf.create(arch)   # rebuild from the stored arch
            elif "arch" not in cfg:
                raise ValueError(
                    "checkpoint has no stored arch (older run); re-supply the training "
                    "arch, e.g. +model=patchset3d arch.l=2")
        model, _ = build_model(cfg)
        model = model.to(DEVICE)
        sd = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()}
        model.load_state_dict(sd)
        model.eval()
        print(f"  Loaded PatchSet3D from {cfg.eval.checkpoint} (arch l={cfg.arch.l})")
        return model
    return load_model(name, ckpt_path=cfg.eval.get("checkpoint"),
                      image_size=image_size, device=DEVICE)


@hydra.main(config_path="../../configs/experiment/3d", config_name="eval", version_base="1.3")
def main(cfg: DictConfig) -> None:
    random.seed(cfg.eval.seed)
    torch.manual_seed(cfg.eval.seed)
    if DEVICE.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")  # TF32 tensor cores for fp32 matmuls

    source = cfg.data.get("source", "totalseg")
    if source == "anchor_synth3d":
        from common import resolve_anchor_classes
        root = cfg.paths.get("totalseg")
        classes = resolve_anchor_classes(cfg.anchor_synth, root)
    elif source == "omnisynth3d":
        # Classes come from the omniSynth3D tile-cache pool (the label_names the
        # dataset emits), not label_stats.csv. Build the bank once to list them.
        from src.datasets.omniSynth.bank_totalseg import get_or_build_totalseg_bank
        s3 = cfg.synth3d
        root = s3.get("tiles_root") or cfg.paths.totalseg
        bank = get_or_build_totalseg_bank(
            root, tuple(s3.get("size", cfg.data.image_size)),
            cfg.eval.split, tuple(resolve_classes(s3.get("classes") or (),
                                                  totalseg_root=cfg.paths.get("totalseg"))))
        classes = [bank.alphabet(c) for c in bank.task_ids()]
    else:
        _, root, is_mri = _source_root(cfg)
        classes = resolve_classes(cfg.data.val_classes, root, is_mri=is_mri)
    image_size = tuple(cfg.data.image_size)
    K = cfg.data.context_size
    model_name = cfg.eval.model

    print(f"Device      : {DEVICE}")
    print(f"Model       : {model_name}")
    print(f"Data root   : {root}  (source={cfg.data.get('source')}, split={cfg.eval.split})")
    print(f"Classes ({len(classes)}): {', '.join(classes)}")
    print(f"K={K}  image_size={image_size}  n_subjects<={cfg.eval.n_subjects}\n")

    model = _build_model(cfg)
    print(f"  Measuring FLOPs (K={K}, size={image_size})...")
    gflops = measure_flops(model, image_size, K, DEVICE)
    print(f"  GFLOPs: {gflops:.2f}\n")

    # ── wandb / output dir ───────────────────────────────────────────────────
    wb_on = bool(cfg.wandb.get("project"))
    run = wandb.init(
        project=cfg.wandb.project, name=cfg.wandb.name,
        mode="online" if wb_on else "disabled",
        config={"model": model_name, "source": cfg.data.get("source"),
                "split": cfg.eval.split, "K": K, "image_size": list(image_size),
                "n_subjects": cfg.eval.n_subjects, "classes": list(classes),
                "gflops": round(gflops, 2)},
    )
    run_name = (wandb.run.name if wandb.run is not None else None) or cfg.wandb.name or model_name
    out_dir = Path(cfg.eval.out_dir) / f"{datetime.date.today():%Y-%m-%d}_{run_name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = (out_dir / "figures") if cfg.eval.get("save_figures", True) else None

    # ── per-class eval (shared loop, also used by train.py's val step) ────────
    rows, all_cases = evaluate_classes(model, cfg, classes, fig_dir=fig_dir)
    # Full per-sample detail table (mirrors experiments/2d eval.py's sample table): one row
    # per case with Dice, timing, GT/context occupancy stats + source-adaptive `detail`.
    case_table = build_sample_table(all_cases) if wb_on else None
    for row in rows:
        cls = row["class"]
        if "error" in row:
            print(f"  {cls:<35s}  ERROR: {row['error']}")
            continue
        row["gflops"] = round(gflops, 2)
        print(f"  {cls:<35s}  dice={row['mean_dice']:.3f} ± {row['std_dice']:.3f}"
              f"  {row['mean_time_ms']:.0f}ms/sample  n={row['n_samples']}")
        if wb_on:
            wandb.log({f"class/{cls}/mean_dice": row["mean_dice"],
                       f"class/{cls}/std_dice": row["std_dice"],
                       f"class/{cls}/mean_time_ms": row["mean_time_ms"]})

    valid = [r for r in rows if "mean_dice" in r]
    if valid:
        mean_dice = sum(r["mean_dice"] for r in valid) / len(valid)
        mean_ms   = sum(r["mean_time_ms"] for r in valid) / len(valid)
        print(f"\n  Mean Dice: {mean_dice:.4f}  |  Mean time: {mean_ms:.1f} ms/sample  "
              f"|  GFLOPs: {gflops:.2f}")
        if wb_on:
            wandb.log({"mean_dice": round(mean_dice, 4), "mean_time_ms": round(mean_ms, 1),
                       "gflops": round(gflops, 2), "cases": case_table})

    # ── save outputs ─────────────────────────────────────────────────────────
    (out_dir / "eval.json").write_text(json.dumps(
        {"model": model_name, "config": OmegaConf.to_container(cfg.eval, resolve=True),
         "rows": rows}, indent=2))
    csv = ["model,class,mean_dice,std_dice,mean_time_ms,gflops,n_samples"]
    csv += [f"{model_name},{r['class']},{r['mean_dice']},{r['std_dice']},"
            f"{r.get('mean_time_ms','')},{r.get('gflops','')},{r['n_samples']}"
            for r in rows if "mean_dice" in r]
    (out_dir / "eval.csv").write_text("\n".join(csv) + "\n")
    print(f"  Saved -> {out_dir}")
    run.finish()


if __name__ == "__main__":
    main()
