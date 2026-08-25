"""GMM-synth cohort provider for the in-context dataloader v2.

Unlike TotalSegProvider (independent per-subject `load`), the synth source samples a
whole COHORT of K+1 similar masks jointly and paints them with ONE cohort-shared GMM
draw, so target and contexts cannot be loaded independently. This provider therefore
implements the engine's optional cohort hook `assemble_task` instead of `load`: it wraps
a SynthGmmMaisiDataset and drives its single cohort-sample + shared-GMM + crop/paint
implementation from the engine's per-item RNG. Emits the same item dict either way
(CPU-paint or gpu_realize native payload), so downstream collate/train.py are unchanged.
"""
import numpy as np

from data.maisi_classes import MAISI_IDX_TO_CLASS


class SynthGmmProvider:
    """Cohort-hook provider wrapping a SynthGmmMaisiDataset for InContextDataset."""

    def __init__(self, dataset):
        self.ds = dataset
        self.epoch_length = len(dataset)
        self.classes = [MAISI_IDX_TO_CLASS.get(c, str(c)) for c in dataset.cs.classes]

    def subjects_for(self, cls):
        # No independent per-subject loading on the cohort path — assemble_task owns
        # cohort selection. Present only to satisfy the VolumeProvider protocol.
        return []

    def assemble_task(self, rng, crop_spacing_mm):
        """Engine cohort hook: build one in-context item from the engine's per-item RNG.

        `nrng` (the cohort-shared GMM draw + paint noise) is derived deterministically from
        `rng`, so an eval-seeded engine RNG yields a reproducible item."""
        nrng = np.random.default_rng(rng.getrandbits(64))
        return self.ds.assemble(rng, nrng, float(crop_spacing_mm))
