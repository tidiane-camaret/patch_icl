"""TriSourceProvider — MultiSourceProvider (CT+MRI) + SynthGmmProvider as a 3rd regime.

At assemble_task time, flips a Bernoulli(p_synth) coin:
  synth: SynthGmmProvider.assemble_task → native_crop payload tagged tgt/ctx_modality='synth'
  real:  MultiSourceProvider.assemble_task → normal CT/MRI/cross native_crop payload

At cascade recrop time (load_native_crop / load), routes by modality:
  'synth': SynthGmmProvider.load_native_crop (re-crop + re-paint the same MAISI mask)
  other:   MultiSourceProvider.load_native_crop (real CT or MRI sub-provider)

Configure via data.p_synth in the experiment config. The synth sub-provider's own `cascade`
flag must match this run's resolved realize flag, same as MultiSourceProvider's
`gpu_realize_crop`: `SynthGmmProvider(dataset, cascade=_realize)` — cascade=True for a
gpu_realize_crop/cascade_spacings run (native_crop payload), cascade=False otherwise (a normal
image-bearing dict via SynthGmmMaisiDataset.assemble). Passing a hardcoded True regardless of
the run's actual mode breaks every non-cascade multisource + p_synth>0 run (see
experiments/3d/common.py's build_dataset, `source=="multisource"` branch).
"""


class TriSourceProvider:
    """MultiSourceProvider + SynthGmmProvider with a per-task synth probability."""

    def __init__(self, multi_provider, synth_provider, *, p_synth,
                 epoch_length, gpu_realize_crop=True):
        self.multi = multi_provider
        self.synth = synth_provider
        self.p_synth = float(p_synth)
        self.epoch_length = int(epoch_length)
        self.gpu_realize_crop = bool(gpu_realize_crop)
        self.classes = list({*multi_provider.classes, *synth_provider.classes})

    def subjects_for(self, cls):
        return []

    def assemble_task(self, rng, crop_spacing_mm):
        if rng.random() < self.p_synth:
            return self.synth.assemble_task(rng, crop_spacing_mm)
        return self.multi.assemble_task(rng, crop_spacing_mm)

    def load_native_crop(self, subject, cls, req, *, modality=None):
        if modality == "synth":
            return self.synth.load_native_crop(subject, cls, req)
        return self.multi.load_native_crop(subject, cls, req, modality=modality)

    def load(self, subject, cls, req, *, modality=None):
        if modality == "synth":
            return self.synth.load(subject, cls, req)
        return self.multi.load(subject, cls, req, modality=modality)
