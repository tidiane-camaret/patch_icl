"""Focused 2D eval for the two in-context models saved by train.py:
`universeg` (native H×W logit) and `patchset_cnn` (low-res R×R logit). Dispatches
on the checkpoint's `model_name`, rebuilds the one model, and runs the SHARED
validate() (evaluate.py) — the same loop/metrics used during training — with
figures + CSVs + FLOPs enabled.

    python experiments/2d/eval_incontext.py eval.checkpoint=results/2d/.../best.pt
    python experiments/2d/eval_incontext.py eval.checkpoint=<p> data.source=omnisynth data.split=test
"""
import datetime
import random
import sys
from pathlib import Path

import hydra
import torch
import wandb
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf, open_dict

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DEVICE, build_loader
from evaluate import validate


def _load_model(ckpt: dict):
    """Rebuild the trained model from a train.py checkpoint (dispatch on model_name)."""
    name = ckpt.get("model_name")
    img = ckpt["image_size"]
    state = {k.removeprefix("_orig_mod."): v for k, v in ckpt["model"].items()}
    if name == "universeg":
        from src.models.universeg_baseline import UniverSegBaseline
        model = UniverSegBaseline(pretrained=True, input_size=img).to(DEVICE)
    elif name == "patchset_cnn":
        from src.models.patchset_cnn import PatchSetCNN
        arch = ckpt.get("arch")
        if not arch:
            raise ValueError(
                "patchset_cnn checkpoint has no 'arch' block — it predates full-arch "
                "storage. Retrain (or re-save) so the checkpoint is self-contained.")
        model = PatchSetCNN(image_size=img, **arch).to(DEVICE)
    else:
        raise ValueError(f"unknown model_name {name!r} (universeg | patchset_cnn)")
    model.load_state_dict(state)
    return model.eval(), name


@hydra.main(config_path="../../configs/experiment/2d", config_name="eval_incontext",
            version_base=None)
def main(cfg: DictConfig):
    random.seed(cfg.eval.seed); torch.manual_seed(cfg.eval.seed)
    if DEVICE.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    if not cfg.eval.get("checkpoint"):
        raise ValueError("set eval.checkpoint=<path/to/best.pt>")
    ckpt = torch.load(cfg.eval.checkpoint, map_location="cpu", weights_only=False)
    model, model_name = _load_model(ckpt)
    # Serve images at the size the checkpoint was trained on, and reproduce the exact
    # synth config the model trained on. eval_base.yaml defaults `synth: default`
    # (controlSynth schema); an omniSynth-trained model needs the omniglot schema, so we
    # replace cfg.synth with the block train.py stored in the checkpoint. Absent for
    # older checkpoints -> leave cfg.synth as-is.
    with open_dict(cfg):
        cfg.data.image_size = ckpt["image_size"]
        cfg.data.context_size = ckpt.get("context_size", cfg.data.context_size)
        if ckpt.get("synth") is not None:
            base = OmegaConf.create(ckpt["synth"])
            # CLI `synth.*` overrides win over the checkpoint's training values, so an
            # OOD eval (e.g. synth.scene.grid=2, synth.scene.target_mode=class) is possible
            # while everything unspecified still reproduces the training distribution.
            cli = [o for o in HydraConfig.get().overrides.task
                   if o.lstrip("+~").split("=", 1)[0].startswith("synth.")]
            if cli:
                base = OmegaConf.merge(
                    base, OmegaConf.from_dotlist([o.lstrip("+~") for o in cli]).synth)
            cfg.synth = base
    print(f"Loaded {model_name} (size={ckpt['image_size']}, ctx={cfg.data.context_size}) "
          f"from {cfg.eval.checkpoint}")

    loader = build_loader(cfg)
    # wandb.project=null (or wandb.enabled=false) disables logging, per repo convention.
    wb_on = bool(cfg.wandb.get("project")) and cfg.wandb.get("enabled", True)
    run = wandb.init(project=cfg.wandb.project, name=cfg.wandb.name,
                     mode="online" if wb_on else "disabled",
                     config={"model": model_name, "checkpoint": str(cfg.eval.checkpoint),
                             "source": cfg.data.get("source"), "split": cfg.data.split,
                             "image_size": ckpt["image_size"],
                             "context_size": cfg.data.context_size})
    run_name = (wandb.run.name if wandb.run is not None else None) or cfg.wandb.name or model_name
    out_dir = Path(cfg.eval.out_dir) / f"{datetime.date.today():%Y-%m-%d}_{run_name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    figures = ({"out_dir": out_dir, "max_figures": int(cfg.eval.get("max_figures", 200)),
                "to_wandb": bool(cfg.eval.get("figures_to_wandb", False))}
               if cfg.eval.get("save_figures", False) else None)
    summary, table, _ = validate(
        model, loader, topk_k=int(cfg.eval.get("topk_k", 16)), epoch=0,
        figures=figures, patch_csv=cfg.eval.get("patch_csv"),
        synth_csv=(cfg.eval.get("synth_csv") if cfg.data.get("source") == "synthetic" else None),
        compute_flops=True)
    summary["samples"] = table
    wandb.log(summary)
    print(f"dice/mean={summary.get('dice/mean'):.4f}  "
          f"dice_ds/mean={summary.get('dice_ds/mean', float('nan')):.4f}  "
          f"flops={summary.get('flops_giga', float('nan')):.2f}G  out={out_dir}")
    run.finish()


if __name__ == "__main__":
    main()
