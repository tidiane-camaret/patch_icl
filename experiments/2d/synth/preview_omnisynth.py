"""Render a few omniSynth items (query image+mask + contexts) to a PNG for eyeballing.

Hydra-driven, exactly like experiments/2d/train.py: pass an experiment `--config-name`
and the preview builds the *same* dataset training sees (via common.build_dataset), so
the generator params, object source, backgrounds and instCopy all match. Copies are
train-only, so the preview defaults to the train split to make p_copy/n_copy visible;
each row is annotated with is_copy / copy_slot.

Any generator param is a native Hydra override on cfg.synth / cfg.data, e.g.
`synth.scene.target_mode=class`, `synth.scene.p_copy=0`, `data.context_size=1`,
`data.image_size=64`. The preview-only knobs live under a `preview` block and are appended
on the CLI with `+preview.<k>=<v>`: split, n, out, and `augment` (run items through the
experiment's aug_preset, matching what the model sees when the config sets `augment: true`).

Run: python experiments/2d/synth/preview_omnisynth.py
     python experiments/2d/synth/preview_omnisynth.py --config-name 2_omnisynth_medseg_refine
     python experiments/2d/synth/preview_omnisynth.py --config-name 2_omnisynth_medseg_refine \\
         +preview.augment=true +preview.split=val +preview.n=6 synth.scene.target_mode=class
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import hydra
import torch
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))    # experiments/2d -> `common`
_ROOT = Path(__file__).resolve().parents[3]                     # repo root (resolve out paths)
from common import build_dataset
from pfn_train import augment                                   # same batch augmenter as train.py


def _augment_item(item, K, aug_cfg):
    """Run one dataset item through the batch augmenter (B=1), mirroring
    train.py._augment_batch: contexts get geometric+intensity, the query image gets at
    most task-intensity ops, and the query mask (item['label']) is left untouched — so
    what the preview draws is exactly what the model sees when `augment: true`."""
    img = item["image"].float().unsqueeze(0)            # (1,1,H,W)
    cin = item["context_in"].float().unsqueeze(0)       # (1,K,1,H,W)
    cout = item["context_out"].float().unsqueeze(0)
    imgs = torch.cat([cin, img.unsqueeze(1)], dim=1)    # (1,T,1,H,W), query at index K
    msks = torch.cat([cout, torch.zeros_like(img.unsqueeze(1))], dim=1)
    imgs, msks = augment(imgs, msks, K, aug_cfg)
    item = dict(item)
    item["image"], item["context_in"], item["context_out"] = imgs[0, K], imgs[0, :K], msks[0, :K]
    return item


@hydra.main(config_path="../../../configs/experiment/2d", config_name="1_omnisynth_medseg",
            version_base=None)
def main(cfg: DictConfig):
    pv = cfg.get("preview", {})           # preview-only knobs: `+preview.<k>=<v>`
    split = pv.get("split", "train")
    n = int(pv.get("n", 10))
    out = Path(pv.get("out", "results/omnisynth_preview.png"))
    if not out.is_absolute():
        out = _ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)

    K = cfg.data.context_size
    image_size = cfg.data.image_size
    ds = build_dataset(cfg, split)        # same OmniSynthICLDataset the trainer uses

    # Opt-in augmentation: `+preview.augment=true` loads the same aug_preset the trainer
    # uses (cfg.aug_preset, merged with any cfg.aug) and runs each item through it, so the
    # preview matches training when the experiment sets `augment: true`.
    do_aug = bool(pv.get("augment", False))
    aug_cfg = None
    if do_aug:
        preset = cfg.get("aug_preset", "2d")
        aug_cfg = OmegaConf.load(_ROOT / "configs" / "augmentations" / f"{preset}.yaml")
        if cfg.get("aug", None):
            aug_cfg = OmegaConf.merge(aug_cfg, cfg.aug)
        do_aug = bool(aug_cfg.get("enabled", True))
        print(f"augment {'ON' if do_aug else 'OFF (enabled=false)'} (preset={preset})")

    s = cfg.synth
    scene = s.scene
    obj_source = s.get("source", "omniglot")
    if obj_source in ("medseg", "biomedparse"):
        med = s.get("medseg", {})
        ds_list = med.get("train_datasets", []) if split == "train" else med.get("val_datasets", [])
        print(f"object source: {obj_source}  {split}_datasets={list(ds_list) or 'all'}")

    print(f"omniSynth preview: split={split} target_mode={scene.target_mode} "
          f"placement={scene.placement} grid={scene.grid} max_obj={scene.max_nb_objects} "
          f"k=[{scene.k_min},{scene.k_max}] "
          f"p_copy={scene.p_copy} n_copy={scene.n_copy} K={K} image_size={image_size}")

    cols = 2 + K * 2     # query img, query mask, then K contexts (img+mask)
    fig, axes = plt.subplots(n, cols, figsize=(cols * 1.4, n * 1.4))
    axes = axes.reshape(n, cols)
    for i in range(n):
        item = ds[i]
        m = item["meta"]
        if do_aug:
            item = _augment_item(item, K, aug_cfg)
        print(f"  sample {i}: class={m['class_id']} alphabet={m['alphabet']} "
              f"target_mode={m['target_mode']} k_target={m['k_target']} "
              f"is_copy={m['is_copy']} copy_slot={m['copy_slot']}")
        panels = [("q-img", item["image"][0]), ("q-mask", item["label"][0])]
        for c in range(K):
            panels.append((f"c{c}-img", item["context_in"][c, 0]))
            panels.append((f"c{c}-msk", item["context_out"][c, 0]))
        for j, (title, im) in enumerate(panels):
            ax = axes[i, j]
            ax.imshow(im, cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(title, fontsize=7)
        axes[i, 0].set_ylabel(f"copy={m['is_copy']}\nslot={m['copy_slot']}", fontsize=6)
    fig.suptitle(f"omniSynth {split} / target_mode={scene.target_mode} "
                 f"/ p_copy={scene.p_copy} n_copy={scene.n_copy}"
                 f"{' / augmented' if do_aug else ''}")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
