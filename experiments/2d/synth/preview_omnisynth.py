"""Render a few omniSynth items (query image+mask + contexts) to a PNG for eyeballing.

Pulls every generator param from the omniglot config (default
configs/experiment/2d/synth/omniglot.yaml) so the preview matches what training
sees. Copies are train-only, so --split defaults to train to make p_copy/n_copy
visible; each row is annotated with is_copy / copy_slot.

Any trailing `key=value` args are OmegaConf dotlist overrides on the loaded config
(scene/diversity/sampling blocks), applied after --config is loaded.

Run: .venv311/bin/python experiments/2d/synth/preview_omnisynth.py
     .venv311/bin/python experiments/2d/synth/preview_omnisynth.py --split val --mode class
     .venv311/bin/python experiments/2d/synth/preview_omnisynth.py --split val scene.grid=2 scene.k_max=3
"""
import argparse
import sys; sys.path.insert(0, ".")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

from src.datasets.omniSynth import (
    OmniDiversityConfig, OmniSamplingConfig, OmniSceneConfig, OmniSynthICLDataset,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment/2d/synth/omniglot.yaml",
                    help="omniSynth generator config (diversity/scene/sampling blocks)")
    ap.add_argument("--mode", default=None, choices=["identical", "aug", "class"],
                    help="override scene.target_mode (default: use the config's value)")
    ap.add_argument("--split", default="train")
    ap.add_argument("--context_size", type=int, default=3)
    ap.add_argument("--image_size", type=int, default=128)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--out", default="results/omnisynth_preview.png")
    args, overrides = ap.parse_known_args()

    cfg = OmegaConf.load(args.config)
    # Any leftover `key=value` args are OmegaConf dotlist overrides on the loaded
    # config, e.g. `scene.grid=2 scene.k_max=3 sampling.epoch_length=10`.
    if overrides:
        bad = [o for o in overrides if "=" not in o]
        if bad:
            ap.error(f"unrecognized arguments: {' '.join(bad)} "
                     "(config overrides must be dotlist key=value, e.g. scene.grid=2)")
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    scene_kw = dict(cfg.scene)
    if args.mode is not None:
        scene_kw["target_mode"] = args.mode
    scene = OmniSceneConfig(**scene_kw)
    diversity = OmniDiversityConfig(**dict(cfg.diversity))
    sampling = OmniSamplingConfig(**dict(cfg.sampling))

    K = args.context_size
    ds = OmniSynthICLDataset(split=args.split, context_size=K, image_size=args.image_size,
                             diversity=diversity, scene=scene, sampling=sampling)

    print(f"omniSynth preview: split={args.split} target_mode={scene.target_mode} "
          f"placement={scene.placement} grid={scene.grid} k=[{scene.k_min},{scene.k_max}] "
          f"p_copy={scene.p_copy} n_copy={scene.n_copy} K={K} image_size={args.image_size}")

    cols = 2 + K * 2     # query img, query mask, then K contexts (img+mask)
    fig, axes = plt.subplots(args.n, cols, figsize=(cols * 1.4, args.n * 1.4))
    axes = axes.reshape(args.n, cols)
    for i in range(args.n):
        item = ds[i]
        m = item["meta"]
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
    fig.suptitle(f"omniSynth {args.split} / target_mode={scene.target_mode} "
                 f"/ p_copy={scene.p_copy} n_copy={scene.n_copy}")
    fig.tight_layout()
    fig.savefig(args.out, dpi=120)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
