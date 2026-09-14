# Training curves — source runs

wandb project: `tidiane-camaret-ndir-universit-tsklinikum-freiburg/patchset_train`.
Both charts made with `plot_training_curves.py` (edit `RUNS`/`OUT_PATH` at the top and
rerun — it currently holds the cascade config below, not the single-level one).

## `patchset_vs_medverse_training_curves.png` — single-level

| model | runs | notes |
|---|---|---|
| patchset (ours) | `j69wl7zb` + `lldthd1z` | `80_varspacing_hard_tgt_prior`, seamless continuation (epoch counter runs straight 0→199, same recipe both runs) |
| medverse | `1avlrpb2` + `3vnqc5lm` | `69_medverse_varspacing_6_1_5` → `85_medverse_varspacing_hard_tgt_prior`. **Not a clean continuation**: `3vnqc5lm` is a weights-only warm start (own epoch counter reset to 0) that also *changed the recipe* — added the hard-target-prior aug patchset already had, plus `ram_cache`/`goal_mask`/`encoder_input_norm` changes. Spliced onto the x-axis at epoch 50 (last original val point) for display only. |

Chart cuts both curves at epoch 140 (medverse's max, the shorter of the two).

## `patchset_vs_medverse_cascade_training_curves.png` — cascade regime

| model | run | notes |
|---|---|---|
| patchset (ours) | `v48ucp2c` | `84_cascade_varspacing_GCP_h100`, from scratch (`train.checkpoint=null`), `cascade_spacings=[6,3,1.5]` |
| medverse | `dm8f4jor` | `91_medverse_multisource_intensity_augs`, from `orig_weights` (medverse's from-scratch equivalent), same `cascade_spacings`/`cascade_query_prior` scheme |

Single runs each, no continuation exists for either (checked — nothing else points at their checkpoints).
Cascade mechanics are matched exactly (spacings, query-prior p/modes/eval_mode, `cascade_train.levels`,
`goal_mask` aug, `gpu_realize_crop`, recrop workers). Remaining confounds, unavoidable — no closer-matched
medverse cascade run exists: medverse trains on `multisource` (totalseg CT + totalsegmri MRI) vs patchset's
totalseg-only, medverse carries a heavier intensity-aug ablation on top, batch_size 2 vs 4, and
`encoder_lr_scale` 0.3 (medverse, gently fine-tuning its pretrained encoder) vs 1 (patchset, full LR from
random init). Chart cuts both curves at epoch 40 (patchset's max, the shorter of the two).
