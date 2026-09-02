"""Query mask-token prior for the NON-cascade PatchSet3D training path.

The cascade feeds level i-1's prediction into level i's query mask token
(``cascade.py::_build_query_prior``). This module is the single-forward analogue:
at a configurable fraction of training steps it seeds the query mask token with
the target's own mask instead of the support-mean occupancy prior, so the model
learns to consume a spatial prior channel it can be handed at inference (an
interactive scribble, an atlas, a bbox, a cheap coarse pass).

Eval always runs with no prior (mode ``none``): ``val/dice`` stays comparable to a
prior-free run and there is no GT to leak at inference time. A ``val/dice`` that
collapses across epochs under ``query_prior=gt`` is the expected "model learned to
copy the prior" signal.

``data.query_prior``::

    none                                 # default — support-mean prior (unchanged)
    gt                                   # every step: target GT as the query prior
    {modes: [none, gt], p: [0.5, 0.5]}   # per-step seeded categorical draw

Step 2 adds perturbed-GT modes (dilate / erode / coarsen / component-drop /
affine-jitter) — they hook into ``build_query_prior`` and extend ``_QP_MODES``.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

import torch

_QP_MODES = ("none", "gt")


@dataclass(frozen=True)
class QueryPriorSpec:
    """Resolved ``data.query_prior``. A scalar mode -> single-mode spec; a mapping
    ``{modes, p}`` -> a categorical drawn once per training step."""

    modes: tuple
    weights: tuple

    @property
    def active(self) -> bool:
        """True when at least one mode injects a prior (i.e. not the pure-`none` default)."""
        return any(m != "none" for m in self.modes)


def resolve_query_prior_spec(query_prior) -> QueryPriorSpec:
    """``data.query_prior`` (None | bool | str | mapping) -> QueryPriorSpec."""
    if query_prior is None or query_prior is False:
        return QueryPriorSpec(("none",), (1.0,))
    if query_prior is True:
        return QueryPriorSpec(("gt",), (1.0,))
    if isinstance(query_prior, str):
        if query_prior not in _QP_MODES:
            raise ValueError(f"data.query_prior={query_prior!r} not in {_QP_MODES}")
        return QueryPriorSpec((query_prior,), (1.0,))
    # mapping (plain dict or OmegaConf DictConfig) — duck-typed to skip an omegaconf import.
    if not hasattr(query_prior, "get") or "modes" not in query_prior:
        raise ValueError(
            f"data.query_prior={query_prior!r}: expected a mode string "
            f"({'|'.join(_QP_MODES)}), a bool, or a mapping with a 'modes' list.")
    modes = tuple(str(x) for x in query_prior["modes"])
    if not modes:
        raise ValueError("data.query_prior.modes is empty")
    for m in modes:
        if m not in _QP_MODES:
            raise ValueError(f"data.query_prior mode {m!r} not in {_QP_MODES}")
    p = query_prior.get("p", None)
    if p is None:
        weights = tuple(1.0 for _ in modes)
    else:
        weights = tuple(float(x) for x in p)
        if len(weights) != len(modes):
            raise ValueError(f"data.query_prior.p (len {len(weights)}) must match "
                             f".modes (len {len(modes)})")
    if any(w < 0 for w in weights) or sum(weights) <= 0:
        raise ValueError(f"data.query_prior.p={list(weights)} must be non-negative with "
                         f"a positive sum")
    return QueryPriorSpec(modes, weights)


def draw_query_prior_mode(spec: QueryPriorSpec, key: str) -> str:
    """One mode per step: the single mode for a fixed spec, else a seeded categorical draw
    (keyed on ``f"{seed}_{epoch}_{step}"`` so it is reproducible and RNG-independent)."""
    if len(spec.modes) == 1:
        return spec.modes[0]
    return random.Random(key).choices(spec.modes, weights=spec.weights, k=1)[0]


@torch.no_grad()
def build_query_prior(mode: str, label: torch.Tensor, *, perturb_cfg=None,
                      spacing_mm=None, gen: torch.Generator | None = None):
    """``(B,D,H,W)`` augmented target GT -> ``(B,1,D,H,W)`` soft prior in [0,1], or None.

    ``label`` is already on the query's T^3 crop grid (the same tensor the loss target is
    pooled from), so no geometric warp is needed — the model's ``_prior_occupancy``
    downsamples it to the R^3 mask-token lattice.

    ``perturb_cfg`` (data.prior_perturb): when set with ``spacing_mm`` + ``gen``, degrade the
    prior via ``src.mask_transforms.perturb_prior_mask`` (dilate/erode/shift/noise, radii in
    mm) so the model does not just learn to copy a clean GT.
    """
    if mode == "none":
        return None
    if mode == "gt":
        prior = label.detach().unsqueeze(1).float().clamp_(0.0, 1.0)
    else:
        raise ValueError(f"unknown query_prior mode {mode!r}")
    if perturb_cfg is not None and spacing_mm is not None and gen is not None:
        from src.mask_transforms import perturb_prior_mask
        prior = perturb_prior_mask(prior, perturb_cfg, float(spacing_mm), gen)
    return prior
