"""_recrop_level routes level>=1 re-crop loads to the right modality sub-provider."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from cascade import _recrop_level


class _RecordingProvider:
    """Cohort-style provider: records (subject, cls, modality) per load_native_crop."""
    def __init__(self):
        self.calls = []

    def load_native_crop(self, subject, cls, req, *, modality):
        self.calls.append((subject, cls, modality))
        T = 8
        from src.providers.totalseg import NativeCrop
        from src.totalseg_dataset import resolve_ct_norm
        geom = torch.tensor([[0, 0, 0], [T, T, T], [T, T, T], [0, 0, 0]], dtype=torch.long)
        return NativeCrop(image=torch.zeros(T, T, T, dtype=torch.float16),
                          label_frac=torch.zeros(T, T, T, dtype=torch.float16),
                          class_idx=3, has_fg=False, out_sizes=[T, T, T], pad_lo=[0, 0, 0],
                          crop_geom=geom, crop_spacing_mm=req.crop_spacing_mm,
                          decim=(1, 1, 1), modality=modality, norm=resolve_ct_norm(None))


def _batch(B=2, T=8):
    return {
        "image": torch.zeros(B, 1, T, T, T),
        "subjects": ["t0", "t1"],
        "context_subjects": [["c0"], ["c1"]],
        "label_names": ["liver", "kidney"],
        "tgt_modality": ["ct", "mri"],
        "ctx_modality": ["mri", "mri"],
    }


def test_recrop_routes_by_modality():
    prov = _RecordingProvider()
    batch = _batch()
    centers = [(4, 4, 4), (4, 4, 4)]
    out = _recrop_level(prov, batch, centers, 3.0, step=0, seed=0, level=1, jitter=0,
                        realize_crop=True, mask_downsample="occupancy", occ_thr=0.1,
                        ct_spec=None, device="cpu")
    # row 0: target ct, its context mri; row 1: target mri, context mri
    assert ("t0", "liver", "ct") in prov.calls
    assert ("c0", "liver", "mri") in prov.calls
    assert ("t1", "kidney", "mri") in prov.calls
    assert ("c1", "kidney", "mri") in prov.calls
    assert out["image"].shape == (2, 1, 8, 8, 8)


def test_recrop_no_modality_keys_is_single_source_compatible():
    """Without tgt_modality/ctx_modality the provider is called with NO modality kwarg."""
    calls = []

    class _SingleSource:
        def load_native_crop(self, subject, cls, req):        # no modality kwarg
            calls.append((subject, cls))
            from src.providers.totalseg import NativeCrop
            from src.totalseg_dataset import resolve_ct_norm
            T = 8
            geom = torch.tensor([[0, 0, 0], [T, T, T], [T, T, T], [0, 0, 0]], dtype=torch.long)
            return NativeCrop(image=torch.zeros(T, T, T, dtype=torch.float16),
                              label_frac=torch.zeros(T, T, T, dtype=torch.float16),
                              class_idx=3, has_fg=False, out_sizes=[T, T, T],
                              pad_lo=[0, 0, 0], crop_geom=geom,
                              crop_spacing_mm=req.crop_spacing_mm, decim=(1, 1, 1),
                              norm=resolve_ct_norm(None))

    batch = {"image": torch.zeros(1, 1, 8, 8, 8), "subjects": ["t0"],
             "context_subjects": [["c0"]], "label_names": ["liver"]}
    _recrop_level(_SingleSource(), batch, [(4, 4, 4)], 3.0, step=0, seed=0, level=1,
                  jitter=0, realize_crop=True, mask_downsample="occupancy", occ_thr=0.1,
                  ct_spec=None, device="cpu")
    assert calls == [("t0", "liver"), ("c0", "liver")]
