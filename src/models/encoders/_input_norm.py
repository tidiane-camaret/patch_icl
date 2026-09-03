"""Shared input-normalization stem for the from-scratch / TS conv encoders.

One module, four modes. `passthrough | reframe | zscore` are the existing encoder
`_norm` behaviors, extracted verbatim so plainconv_ts / resenc_ts / nnunet_ts stop
duplicating them. `instance` is the modality-agnostic mode: per-sample z-score of the
tensor as received, with NO inversion to a HU frame — nothing modality-specific, so
it is correct regardless of which per-modality normalization the dataloader ran. See
docs/superpowers/specs/2026-09-03-modality-agnostic-normalization-design.md.
"""
import torch
import torch.nn as nn

_INPUT_NORMS = ("passthrough", "reframe", "zscore", "instance")


class InputRenorm(nn.Module):
    def __init__(self, mode, *, loader_spec=None, target_spec=None,
                 affine=False, eps=1e-8):
        super().__init__()
        if mode not in _INPUT_NORMS:
            raise ValueError(f"unknown input_norm {mode!r} ({'|'.join(_INPUT_NORMS)})")
        if mode in ("reframe", "zscore") and loader_spec is None:
            raise ValueError(f"input_norm={mode!r} needs loader_spec")
        if mode == "reframe" and target_spec is None:
            raise ValueError("input_norm='reframe' needs target_spec")
        if affine and mode != "instance":
            raise ValueError(f"affine=True is only supported for mode='instance', not {mode!r}")
        self.mode = mode
        self._loader = loader_spec
        self._target = target_spec
        self.eps = float(eps)
        self.affine = bool(affine)
        if self.affine:
            self.gamma = nn.Parameter(torch.ones(1, 1, 1, 1, 1))
            self.beta = nn.Parameter(torch.zeros(1, 1, 1, 1, 1))

    def _maybe_grow_affine(self, c, device, dtype):
        if not self.affine or self.gamma.shape[1] == c:
            return
        self.gamma = nn.Parameter(torch.ones(1, c, 1, 1, 1, device=device, dtype=dtype))
        self.beta = nn.Parameter(torch.zeros(1, c, 1, 1, 1, device=device, dtype=dtype))

    def _per_sample_zscore(self, v):
        flat = v.reshape(v.shape[0], -1)
        mu = flat.mean(dim=1).reshape(-1, 1, 1, 1, 1)
        sig = flat.std(dim=1).reshape(-1, 1, 1, 1, 1)
        return (v - mu) / (sig + self.eps)

    def forward(self, x):
        x = x.float()
        if self.mode == "passthrough":
            return x
        if self.mode == "instance":
            out = self._per_sample_zscore(x)
            if self.affine:
                self._maybe_grow_affine(out.shape[1], out.device, out.dtype)
                out = out * self.gamma + self.beta
            return out
        hu = x * self._loader.std + self._loader.mean
        if self.mode == "reframe":
            t = self._target
            return (hu.clamp(t.clip_lo, t.clip_hi) - t.mean) / t.std
        return self._per_sample_zscore(hu)   # zscore
