"""Render a few omniSynth items (query image+mask + contexts) to a PNG for eyeballing.

Run: .venv311/bin/python experiments/2d/synth/preview_omnisynth.py --mode class --n 4
"""
import argparse
import sys; sys.path.insert(0, ".")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.datasets.omniSynth import OmniSceneConfig, OmniSynthICLDataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="class", choices=["identical", "aug", "class"])
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--out", default="results/omnisynth_preview.png")
    args = ap.parse_args()

    ds = OmniSynthICLDataset(split=args.split, context_size=3, image_size=64,
                             scene=OmniSceneConfig(target_mode=args.mode))
    cols = 2 + 3 * 2     # query img, query mask, then 3 contexts (img+mask)
    fig, axes = plt.subplots(args.n, cols, figsize=(cols * 1.4, args.n * 1.4))
    axes = axes.reshape(args.n, cols)
    for i in range(args.n):
        item = ds[i]
        panels = [("q-img", item["image"][0]), ("q-mask", item["label"][0])]
        for c in range(3):
            panels.append((f"c{c}-img", item["context_in"][c, 0]))
            panels.append((f"c{c}-msk", item["context_out"][c, 0]))
        for j, (title, im) in enumerate(panels):
            ax = axes[i, j]
            ax.imshow(im, cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(title, fontsize=7)
    fig.suptitle(f"omniSynth {args.split} / target_mode={args.mode}")
    fig.tight_layout()
    fig.savefig(args.out, dpi=120)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
