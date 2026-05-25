# patch_icl — Results Summary

*Last updated: 2026-05-25*

---

## VRAM & throughput sweep

Full report: `results/benchmarks/vram_sweep/report.md`

Reference config (current training): 128³, [8³→16³→32³], NP=512, K=1, dim=256, L=8 → **1.99 GB · 289 ms/step** at batch_size=1.

| Parameter | Range tested | VRAM range | Time range |
|-----------|-------------|------------|------------|
| `image_size` | 64³–512³ | 0.25–6.28 GB (512³ OOM) | 41–251 ms |
| `num_levels` | 1–6 | 0.92–5.71 GB (5+ OOM) | 50–139 ms |
| `n_patches_l1` | 128–8192 | 1.25–1.72 GB (8192 error) | ~96 ms flat |
| `context_size K` | 1–4 | 0.93–2.62 GB | 73–116 ms |
| `dim` | 64–512 | 0.92–1.54 GB | ~115 ms flat |
| `num_layers L` | 2–16 | 0.99–1.69 GB | 79–407 ms |

---

## Segmentation quality: MultilevelICL vs Medverse

Source: `results/benchmarks/eval/eval_20260524_*.json` — TotalSegmentator test split, K=1, 117 classes.

| Model | Mean Dice | GFLOPs | Latency |
|-------|-----------|--------|---------|
| MultilevelICL (ours) | **0.212** | 599 | 102 ms |
| Medverse | 0.105 | 2363 | 172 ms |

### Per-class Dice

| Class | MultilevelICL | Medverse |
|-------|:---:|:---:|
| liver | 0.624 | 0.747 |
| gluteus_maximus_left | 0.615 | 0.340 |
| femur_left | 0.614 | 0.465 |
| femur_right | 0.591 | 0.459 |
| gluteus_maximus_right | 0.567 | 0.344 |
| spleen | 0.540 | 0.462 |
| urinary_bladder | 0.508 | 0.337 |
| kidney_right | 0.441 | **0.771** |
| brain | 0.395 | **0.528** |
| kidney_left | 0.353 | 0.486 |
| aorta | 0.310 | 0.218 |
| pancreas | 0.224 | 0.037 |
| adrenal_gland_left | 0.009 | 0.005 |
| common_carotid_artery_left | 0.014 | 0.008 |
