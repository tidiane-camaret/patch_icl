"""Config dataclasses for omniSynth (split into diversity / scene / sampling,
mirroring controlSynth's separation of concerns)."""

from dataclasses import dataclass


@dataclass
class OmniDiversityConfig:
    master_seed: int = 42
    omniglot_root: str = "/home/dpxuser/repos/omniglot/python"  # dir holding the zips
    train_zip: str = "images_background.zip"
    eval_zip: str = "images_evaluation.zip"
    val_test_split: float = 0.5   # fraction of eval-alphabet classes -> val (rest -> test)


@dataclass
class OmniSceneConfig:
    grid: int = 2                 # grid x grid cells fill the canvas
    placement: str = "grid"       # grid: glyphs centred on cell centres.
                                  # random: glyphs at uniform-random canvas positions (may
                                  # overlap). Glyph count stays grid*grid either way.
    k_min: int = 1                # target cells ~ U[k_min, k_max] (clamped to [1, grid*grid])
    k_max: int = 2
    cell_margin: float = 0.1      # glyph size = (1 - 2*margin)*cell. >0: padding inside the
                                  # cell; 0: fills the cell; <0: glyph exceeds the cell and
                                  # overflows into neighbours (rendered with union blending).
    target_mode: str = "class"    # identical | aug | class
    aug_rotate: float = 15.0      # deg; aug-mode per-placement jitter
    aug_scale: float = 0.1        # +/- log2 scale
    aug_translate: float = 0.1    # fraction of cell
    p_copy: float = 0.9           # train-only per-item prob of injecting copy slot(s)
    n_copy: int = 1               # number of context slots to copy when an item is a
                                  # copy-task (clamped to context_size); each is an exact copy of the query scene


@dataclass
class OmniSamplingConfig:
    epoch_length: int = 10000
    eval_subjects_per_task: int = 4
    eval_seed_namespace: int = 0
