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
class OmniMedSegConfig:
    """Selects a real-image object source instead of Omniglot glyphs. A "class" is a
    (dataset, label) pair (the analog of Omniglot's alphabet/character); a rendition is
    one image's whole binary mask for that label, cropped to its bbox, plus the intensity
    patch under it. Each split reads objects from that source split's own images
    (train->train, val->val/test); there is no separate test set (val doubles as test).
    train_datasets / val_datasets choose which datasets feed each split ([] = all).

    The active source is selected by the top-level `synth.source` field (omniglot |
    medseg | biomedparse), not here; this block only holds the real-object params.
    medseg     -> MedSegBench {name}_{size}.npz (label = a value in the label map).
    biomedparse-> BiomedParse pre-resized store (label = target from mask filename)."""
    data_root: str = ("/nfs/data/nii/data1/Analysis/camaret___in_context_segmentation/"
                       "ANALYSIS_20251122/data/medsegbench")
    biomedparse_root: str = ("/nfs/data/nii/data1/Analysis/camaret___in_context_segmentation/"
                             "ANALYSIS_20251122/data/biomedparse/_npy")   # source=biomedparse
    source_size: int = 128        # medseg: {name}_{size}.npz; biomedparse: {..}_{size}.npy store
    train_datasets: tuple = ()    # [] = all datasets; else subset feeding the train pool
    val_datasets: tuple = ()      # [] = all; else subset feeding val (also used as test)
    max_renditions_per_class: int = 200   # cap objects kept per class (memory)
    min_mask_px: int = 4          # drop masks smaller than this (near-empty labels)
    size_mode: str = "canvas"     # canvas: keep the object's size relative to the canvas
                                  #   (bbox_frac_of_source * canvas), aspect preserved;
                                  # cell: resize every object to the cell (uniform, glyph-like)
    size_scale: float = 1.0       # canvas mode: multiply the preserved object size


@dataclass
class OmniSceneConfig:
    grid: int = 2                 # grid x grid cells fill the canvas
    placement: str = "grid"       # grid: glyphs centred on cell centres.
                                  # random: glyphs at uniform-random canvas positions (may
                                  # overlap). Glyph count stays grid*grid either way.
    max_nb_objects: int = 0       # cap on total glyphs placed (targets + distractors);
                                  # 0 = no cap (fill all grid*grid). The filled cells are a
                                  # random subset; applies to both placement modes.
    placement_tries: int = 1      # random placement: candidate positions tried per object,
                                  # keeping the least-overlapping one (cheap anti-overlap).
                                  # 1 = fully random (no rejection); higher = less overlap.
    placement_max_overlap: float = 0.25  # accept a candidate early once its object-mask
                                  # overlap with already-placed objects is <= this fraction
    k_min: int = 1                # target cells ~ U[k_min, k_max] (clamped to [1, n_obj])
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
    background: str = "black"     # black: zero canvas (glyph default). random: a smooth
                                  # random grey field + noise. image: a random real full
                                  # image from medseg/biomedparse (see bg_source below).
                                  # Objects are painted over the background (not maxed), so
                                  # dark textures stay visible whatever the background is.
    bg_intensity: tuple = (0.2, 0.6)   # random-bg base grey level ~ U[lo, hi]
    bg_structure: float = 0.15    # amplitude of smooth low-frequency background variation
    bg_noise: float = 0.03        # gaussian noise std added to the background
    bg_source: str = "medseg"     # background=image: medseg | biomedparse (roots/size come
                                  # from the medseg block); independent of the object source.
    bg_datasets: tuple = ()       # background=image: [] = all datasets; else a subset
    bg_max_images: int = 2000     # background=image: cap on pooled background images (memory)


@dataclass
class OmniSamplingConfig:
    epoch_length: int = 10000
    eval_subjects_per_task: int = 4
    eval_seed_namespace: int = 0
