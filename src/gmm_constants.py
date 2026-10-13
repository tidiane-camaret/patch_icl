"""Normalization constants for the GMM-painted synth pipeline, split out of
synth_gmm_maisi_dataset.py so consumers that only need these two numbers (e.g.
gpu_realize_crop.py, on the inference cascade path via cascade.py) don't have to import
that module's full GMM-sampling machinery (src.gpu_gmm_intensity -> scipy) just for them.
synth_gmm_maisi_dataset.py re-exports these for backward compatibility -- existing
`from src.synth_gmm_maisi_dataset import GMM_MEAN, GMM_STD` callers are unaffected.
"""
GMM_MEAN, GMM_STD = 128.0, 74.0
