# Modality-Agnostic Normalization — Prep Seams Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land three no-op seams so a later CT+MRI joint-training change is config-only: a shared model input-normalization stem, a `modality` field on the data payload, and a de-pinnable clamp frame in the GPU augmentor.

**Architecture:** Loose dataloader contract (per-modality clip → z-score), strict model stem (sole canonical frame, no descriptor passed). This PR extracts the duplicated encoder `_norm` into one `InputRenorm` module and adds an `instance` mode (per-volume renorm, no HU inversion) that is unreachable until selected; adds a `modality: str` field that rides the payload unused; parametrizes the augmentor's hardcoded CT clamp bounds behind a default-off constructor arg. Every current config stays byte-identical.

**Tech Stack:** Python 3.12, PyTorch, Hydra/OmegaConf, pytest ≥ 9.1.1, `dynamic_network_architectures` (nnU-Net).

**Spec:** `docs/superpowers/specs/2026-09-03-modality-agnostic-normalization-design.md`

## Global Constraints

- **No behavior change to any current config.** `experiments/3d/experiment/70_patchset_varspacing_6_1_5.yaml` and every other experiment must produce identical results. Per-encoder `input_norm` defaults unchanged: `plainconv_ts` = `zscore`, `resenc_ts` = `passthrough`, `nnunet_ts` = `reframe`.
- **Existing tests must keep passing unchanged**, including `tests/test_plainconv_ts_encoder.py` and `tests/test_resenc_ts_encoder.py`, which call `enc._norm(x)` directly and pass `loader_ct_norm=` / `target_ct_norm=` as encoder constructor kwargs — those signatures and the `enc._norm` method must survive.
- **`CtNormSpec`** (`src/totalseg_dataset.py:36`) is a frozen dataclass `CtNormSpec(clip_lo, clip_hi, mean, std)` with `.norm_min` / `.norm_max` properties. `resolve_ct_norm(spec)` maps `None`→`fingerprint_1228`, a str→preset, a `CtNormSpec`→itself, a mapping→`CtNormSpec`.
- **The z-score math to preserve bit-exactly** (current `_norm`, non-passthrough): `hu = x.float() * loader_spec.std + loader_spec.mean`; then `reframe` → `(hu.clamp(target.clip_lo, target.clip_hi) - target.mean) / target.std`; `zscore` → `flat = hu.reshape(hu.shape[0], -1); mu = flat.mean(dim=1).reshape(-1,1,1,1,1); sig = flat.std(dim=1).reshape(-1,1,1,1,1); (hu - mu) / (sig + 1e-8)`.
- **Commit after every task.** This repo's `configs/` tree and NFS mount have a known `git commit` hang; if `git commit` hangs > 20s, stop and report — do not retry. Run `git` via `PATH="/software/anaconda3/envs/git/bin:$PATH"` (git is not on the default PATH).
- Tests run with `python -m pytest <path> -v` from the repo root (`/home/dpxuser/dev/patch_icl`). No `[tool.pytest.ini_options]` — pass explicit paths.

---

## File Structure

**Create:**
- `src/models/encoders/_input_norm.py` — `InputRenorm(nn.Module)`, the single shared input-normalization stem + the `_INPUT_NORMS` enum. One responsibility: map a raw input tensor to the encoder's canonical frame.
- `tests/test_input_norm.py` — unit tests for `InputRenorm` (all four modes, affine flag, bit-parity vs. the inline math).
- `tests/test_modality_seam.py` — the `modality` field flows `LoadResult` → item dict → `batch["modality"]`.

**Modify:**
- `src/models/encoders/plainconv_ts.py` — hold an `InputRenorm`; `_norm` delegates; accept `instance` in `input_norm`.
- `src/models/encoders/resenc_ts.py` — same.
- `src/models/encoders/nnunet_ts.py` — same; build `target_spec` from plans `CTNormalization`.
- `src/incontext_dataset_v2.py` — `LoadResult.modality: str = "ct"`; item dict carries `"modality"`.
- `src/providers/totalseg.py` — `NativeCrop.modality: str = "ct"`; `build_native_crop(..., modality="ct")`; `load()` / `load_native_crop()` pass `self.modality`; `LoadResult(..., modality=self.modality)`.
- `src/providers/native_grid.py` — `LoadResult(..., modality=self.modality)` (or `"ct"` if the provider has no `modality` attr).
- `src/totalseg_dataloader_incontext.py` — `incontext_collate_fn` emits `out["modality"]`.
- `src/gpu_augment.py` — `clamp` param on `_batched_gin_ipa` / `_batched_bias_field` / `_batched_intensity`; `GpuAugmentor(clamp_frame=None)`; relaxed guard.
- `tests/test_gpu_augment.py` — add a `clamp_frame` test; assert default path unchanged.

---

## Task 1: `InputRenorm` shared stem module

**Files:**
- Create: `src/models/encoders/_input_norm.py`
- Test: `tests/test_input_norm.py`

**Interfaces:**
- Consumes: `src.totalseg_dataset.CtNormSpec`, `resolve_ct_norm`.
- Produces:
  - `_INPUT_NORMS: tuple[str, ...] = ("passthrough", "reframe", "zscore", "instance")`
  - `class InputRenorm(nn.Module)`:
    - `__init__(self, mode: str, *, loader_spec: CtNormSpec | None = None, target_spec: CtNormSpec | None = None, affine: bool = False, eps: float = 1e-8)`
    - `forward(self, x: torch.Tensor) -> torch.Tensor` — `x` is `(B, 1, D, H, W)` or `(B, C, D, H, W)`; returns same shape, `float32`.
    - modes: `passthrough` → `x.float()`; `reframe` → invert loader frame to HU then clamp+z-score into `target_spec`; `zscore` → invert to HU then per-sample z-score (no clip); `instance` → per-sample z-score of `x.float()` directly (NO HU inversion), then optional learned per-channel affine.
    - `affine=True` registers `self.gamma` / `self.beta` as `nn.Parameter` of shape `(1, C, 1, 1, 1)` with `C=1` default (lazily sized on first `forward` if `C != 1`). `affine=False` registers no params/buffers.
    - raises `ValueError` for an unknown `mode`, and for `reframe`/`zscore` when `loader_spec is None`, and for `reframe` when `target_spec is None`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_input_norm.py`:

```python
"""Unit tests for the shared InputRenorm stem (src/models/encoders/_input_norm.py)."""
import pytest
import torch

from src.models.encoders._input_norm import InputRenorm, _INPUT_NORMS
from src.totalseg_dataset import CT_NORM_PRESETS


LD = CT_NORM_PRESETS["fingerprint_1228"]
TG = CT_NORM_PRESETS["d297"]


def test_enum_has_instance():
    assert _INPUT_NORMS == ("passthrough", "reframe", "zscore", "instance")


def test_passthrough_is_identity_float():
    m = InputRenorm("passthrough")
    x = torch.randn(2, 1, 8, 8, 8, dtype=torch.float64)
    out = m(x)
    assert out.dtype == torch.float32
    assert torch.equal(out, x.float())


def test_reframe_matches_inline_math():
    m = InputRenorm("reframe", loader_spec=LD, target_spec=TG)
    x = torch.randn(3, 1, 8, 8, 8)
    hu = x.float() * LD.std + LD.mean
    want = (hu.clamp(TG.clip_lo, TG.clip_hi) - TG.mean) / TG.std
    assert torch.allclose(m(x), want, atol=1e-6)


def test_zscore_matches_inline_math():
    m = InputRenorm("zscore", loader_spec=LD)
    x = torch.randn(3, 1, 8, 8, 8)
    hu = x.float() * LD.std + LD.mean
    flat = hu.reshape(hu.shape[0], -1)
    mu = flat.mean(dim=1).reshape(-1, 1, 1, 1, 1)
    sig = flat.std(dim=1).reshape(-1, 1, 1, 1, 1)
    want = (hu - mu) / (sig + 1e-8)
    assert torch.allclose(m(x), want, atol=1e-6)


def test_instance_is_zscore_without_hu_inversion():
    m = InputRenorm("instance")
    x = torch.randn(3, 1, 8, 8, 8) * 5.0 + 2.0
    flat = x.float().reshape(x.shape[0], -1)
    mu = flat.mean(dim=1).reshape(-1, 1, 1, 1, 1)
    sig = flat.std(dim=1).reshape(-1, 1, 1, 1, 1)
    want = (x.float() - mu) / (sig + 1e-8)
    assert torch.allclose(m(x), want, atol=1e-6)
    # per-sample standardized: mean ~0, std ~1
    per = m(x).reshape(3, -1)
    assert torch.allclose(per.mean(dim=1), torch.zeros(3), atol=1e-5)
    assert torch.allclose(per.std(dim=1), torch.ones(3), atol=1e-3)


def test_instance_affine_has_params_and_defaults_identity():
    m = InputRenorm("instance", affine=True)
    names = [n for n, _ in m.named_parameters()]
    assert names == ["gamma", "beta"]
    x = torch.randn(2, 1, 8, 8, 8)
    base = InputRenorm("instance")
    assert torch.allclose(m(x), base(x), atol=1e-6)   # gamma=1, beta=0 at init


def test_affine_false_registers_no_state():
    m = InputRenorm("instance", affine=False)
    assert list(m.parameters()) == []
    assert list(m.buffers()) == []
    assert m.state_dict() == {}


def test_unknown_mode_raises():
    with pytest.raises(ValueError):
        InputRenorm("bogus")


def test_zscore_requires_loader_spec():
    with pytest.raises(ValueError):
        InputRenorm("zscore")


def test_reframe_requires_both_specs():
    with pytest.raises(ValueError):
        InputRenorm("reframe", loader_spec=LD)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_input_norm.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.models.encoders._input_norm'`

- [ ] **Step 3: Write minimal implementation**

Create `src/models/encoders/_input_norm.py`:

```python
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

from src.totalseg_dataset import CtNormSpec  # noqa: F401  (type reference for callers)

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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_input_norm.py -v`
Expected: PASS (10 tests)

- [ ] **Step 5: Commit**

```bash
PATH="/software/anaconda3/envs/git/bin:$PATH" git add src/models/encoders/_input_norm.py tests/test_input_norm.py
PATH="/software/anaconda3/envs/git/bin:$PATH" git commit -m "feat: shared InputRenorm stem with instance mode (no-op seam)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017LE3uxB7pNGutyT2deEGhp"
```

---

## Task 2: Refactor the three conv encoders onto `InputRenorm`

**Files:**
- Modify: `src/models/encoders/plainconv_ts.py` (`_INPUT_NORMS` at module scope; `__init__` ~lines 70-95; `_norm` ~lines 122-134)
- Modify: `src/models/encoders/resenc_ts.py` (`_INPUT_NORMS` ~line 43; `__init__` ~lines 79-98; `_norm` ~lines 126-138)
- Modify: `src/models/encoders/nnunet_ts.py` (`_INPUT_NORMS` ~line 40; `__init__` ~lines 104-126; `_norm` ~lines 185-197)
- Test: `tests/test_plainconv_ts_encoder.py`, `tests/test_resenc_ts_encoder.py` (existing — must still pass; add `instance` cases)

**Interfaces:**
- Consumes: `src.models.encoders._input_norm.InputRenorm`, `_INPUT_NORMS` (Task 1).
- Produces: no signature change. Each encoder keeps `input_norm=` (now also accepts `"instance"`), `loader_ct_norm=`, and (plainconv/resenc only) `target_ct_norm=`. Each gains `self.input_renorm: InputRenorm` and keeps a thin `_norm(self, x)` that returns `self.input_renorm(x)`. `self.input_norm` (the string) stays for the existing assertions.

- [ ] **Step 1: Add `instance` cases to the existing encoder tests (failing)**

Append to `tests/test_plainconv_ts_encoder.py`:

```python
def test_instance_norm_no_hu_inversion():
    from src.models.encoders.plainconv_ts import PlainConvTSEncoder
    enc = PlainConvTSEncoder(resolution=8, n_stages=4, stages=(1, 2, 3),
                             input_norm="instance", frozen=False,
                             device="cpu", precision="fp32")
    x = torch.randn(2, 1, 16, 16, 16) * 3.0 + 1.0
    flat = x.float().reshape(2, -1)
    mu = flat.mean(dim=1).reshape(-1, 1, 1, 1, 1)
    sig = flat.std(dim=1).reshape(-1, 1, 1, 1, 1)
    assert torch.allclose(enc._norm(x), (x.float() - mu) / (sig + 1e-8), atol=1e-6)
```

Append to `tests/test_resenc_ts_encoder.py`:

```python
def test_instance_norm_no_hu_inversion():
    from src.models.encoders.resenc_ts import ResEncTSEncoder
    enc = ResEncTSEncoder(resolution=8, n_stages=4, stages=(1, 2, 3),
                          input_norm="instance", frozen=False,
                          device="cpu", precision="fp32")
    x = torch.randn(2, 1, 16, 16, 16) * 3.0 + 1.0
    flat = x.float().reshape(2, -1)
    mu = flat.mean(dim=1).reshape(-1, 1, 1, 1, 1)
    sig = flat.std(dim=1).reshape(-1, 1, 1, 1, 1)
    assert torch.allclose(enc._norm(x), (x.float() - mu) / (sig + 1e-8), atol=1e-6)
```

- [ ] **Step 2: Run tests to verify the new cases fail**

Run: `python -m pytest tests/test_plainconv_ts_encoder.py::test_instance_norm_no_hu_inversion tests/test_resenc_ts_encoder.py::test_instance_norm_no_hu_inversion -v`
Expected: FAIL — `ValueError: unknown input_norm 'instance'` (from the encoder's `_INPUT_NORMS` check).

- [ ] **Step 3: Refactor `plainconv_ts.py`**

At module scope, replace `_INPUT_NORMS = ("passthrough", "reframe", "zscore")` with:

```python
from src.models.encoders._input_norm import InputRenorm, _INPUT_NORMS
```

(Delete the local `_INPUT_NORMS` tuple. Keep the existing `from src.totalseg_dataset import resolve_ct_norm` import.)

In `__init__`, replace the block:

```python
        self.input_norm = str(input_norm)
        if self.input_norm not in _INPUT_NORMS:
            raise ValueError(f"unknown input_norm {input_norm!r} ({'|'.join(_INPUT_NORMS)})")
```

...and the later:

```python
        # Frames for the reframe/zscore paths (unused under passthrough).
        self._loader_spec = resolve_ct_norm(loader_ct_norm)
        self._target_spec = resolve_ct_norm(target_ct_norm)
```

with:

```python
        self.input_norm = str(input_norm)
        if self.input_norm not in _INPUT_NORMS:
            raise ValueError(f"unknown input_norm {input_norm!r} ({'|'.join(_INPUT_NORMS)})")
        # Shared stem: passthrough/reframe/zscore = the previous inline _norm; instance =
        # modality-agnostic per-sample renorm (no HU inversion). Frames unused for
        # passthrough/instance but cheap to resolve.
        self.input_renorm = InputRenorm(
            self.input_norm,
            loader_spec=resolve_ct_norm(loader_ct_norm),
            target_spec=resolve_ct_norm(target_ct_norm))
```

Replace the whole `_norm` method body with:

```python
    def _norm(self, x):
        """Delegates to the shared InputRenorm stem (see _input_norm.py)."""
        return self.input_renorm(x)
```

- [ ] **Step 4: Refactor `resenc_ts.py`** — identical edits

Same three edits as Step 3, in `resenc_ts.py`:
- Replace the local `_INPUT_NORMS` tuple (~line 43) with `from src.models.encoders._input_norm import InputRenorm, _INPUT_NORMS`.
- In `__init__`, replace the `self.input_norm` validation + the `self._loader_spec` / `self._target_spec` lines with the `InputRenorm(...)` construction block from Step 3 (verbatim — `resenc_ts` also has `loader_ct_norm` and `target_ct_norm` kwargs).
- Replace `_norm` body with the one-line delegation.

- [ ] **Step 5: Refactor `nnunet_ts.py`** — target_spec from plans

`nnunet_ts` has `loader_ct_norm` but **no** `target_ct_norm` kwarg; its reframe target is the plans `CTNormalization` (`self.ct_clip`, `self.ct_mean`, `self.ct_std`, set from `fip` at ~line 124-126). The `InputRenorm` must be built **after** those are set.

- Replace the local `_INPUT_NORMS` tuple (~line 40) with `from src.models.encoders._input_norm import InputRenorm, _INPUT_NORMS`. Add `from src.totalseg_dataset import CtNormSpec` to the imports (keep the existing `resolve_ct_norm` import).
- In `__init__`, keep the early `self.input_norm` validation as-is (it runs before `fip` is read — leave it). Keep `self._loader_spec = resolve_ct_norm(loader_ct_norm)` where it is.
- Immediately **after** the three lines that set `self.ct_clip` / `self.ct_mean` / `self.ct_std`, add:

```python
        # Shared stem. reframe target = this checkpoint's plans CTNormalization.
        self.input_renorm = InputRenorm(
            self.input_norm,
            loader_spec=self._loader_spec,
            target_spec=CtNormSpec(clip_lo=self.ct_clip[0], clip_hi=self.ct_clip[1],
                                   mean=self.ct_mean, std=self.ct_std))
```

- Replace the whole `_norm` method body (~lines 185-197) with:

```python
    def _norm(self, x):
        """Delegates to the shared InputRenorm stem (see _input_norm.py)."""
        return self.input_renorm(x)
```

- [ ] **Step 6: Run the full encoder test suites**

Run: `python -m pytest tests/test_plainconv_ts_encoder.py tests/test_resenc_ts_encoder.py -v`
Expected: PASS — all pre-existing tests (including `test_passthrough_norm_is_identity`, `test_reframe_norm_matches_manual_roundtrip`, `test_zscore_norm_path_runs`) plus the two new `test_instance_norm_no_hu_inversion`.

- [ ] **Step 7: Smoke-check nnunet_ts import path**

Run: `python -c "import ast,sys; ast.parse(open('src/models/encoders/nnunet_ts.py').read()); print('nnunet_ts.py parses')"`
Expected: prints `nnunet_ts.py parses`. (A live `NnUNetTSEncoder` build needs a weights dir; the encoder test suites above already exercise the shared `InputRenorm` through plainconv/resenc. The `reframe` math is identical and covered by `tests/test_input_norm.py::test_reframe_matches_inline_math`.)

- [ ] **Step 8: Verify exp 70 config still resolves**

Run: `python experiments/3d/train.py experiment=70_patchset_varspacing_6_1_5 --cfg job 2>&1 | grep -E "encoder_input_norm|encoder: plainconv_ts"`
Expected: `encoder: plainconv_ts` and `encoder_input_norm: zscore` present, no traceback.

- [ ] **Step 9: Commit**

```bash
PATH="/software/anaconda3/envs/git/bin:$PATH" git add src/models/encoders/plainconv_ts.py src/models/encoders/resenc_ts.py src/models/encoders/nnunet_ts.py tests/test_plainconv_ts_encoder.py tests/test_resenc_ts_encoder.py
PATH="/software/anaconda3/envs/git/bin:$PATH" git commit -m "refactor: conv encoders share InputRenorm stem; add instance mode

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017LE3uxB7pNGutyT2deEGhp"
```

---

## Task 3: `modality` field seam

**Files:**
- Modify: `src/incontext_dataset_v2.py` (`LoadResult` dataclass ~lines 30-35; item dict ~lines 188-199)
- Modify: `src/providers/totalseg.py` (`NativeCrop` ~lines 30-49; `build_native_crop` ~lines 59-88; `load` return ~line 258; `load_native_crop` return ~lines 287-290)
- Modify: `src/providers/native_grid.py` (`LoadResult` return ~line 133)
- Modify: `src/totalseg_dataloader_incontext.py` (`incontext_collate_fn` ~lines 1551-1582)
- Test: `tests/test_modality_seam.py`

**Interfaces:**
- Consumes: nothing new.
- Produces:
  - `LoadResult.modality: str` (dataclass field, default `"ct"`).
  - `NativeCrop.modality: str` (dataclass field, default `"ct"`).
  - `build_native_crop(..., modality: str = "ct")` keyword-only arg (added to the existing `*,` block).
  - Item dict from `InContextDataset.__getitem__` (non-realize branch) carries `"modality": str`.
  - `batch["modality"]: list[str]` of length B from `incontext_collate_fn` when items carry it.

- [ ] **Step 1: Write the failing test**

Create `tests/test_modality_seam.py`:

```python
"""The `modality` field rides LoadResult -> item dict -> batch, unused downstream."""
import random

import torch

from src.incontext_dataset_v2 import LoadResult
from src.totalseg_dataloader_incontext import incontext_collate_fn


def test_loadresult_modality_defaults_ct():
    r = LoadResult(image=torch.zeros(1, 4, 4, 4), label=torch.zeros(4, 4, 4),
                   spacing=torch.ones(3), crop_geom=torch.zeros(4, 3, dtype=torch.long))
    assert r.modality == "ct"


def test_loadresult_modality_settable():
    r = LoadResult(image=torch.zeros(1, 4, 4, 4), label=torch.zeros(4, 4, 4),
                   spacing=torch.ones(3), crop_geom=torch.zeros(4, 3, dtype=torch.long),
                   modality="mri")
    assert r.modality == "mri"


def _item(modality):
    return {
        "image": torch.zeros(1, 4, 4, 4),
        "label": torch.zeros(4, 4, 4, dtype=torch.long),
        "context_in": torch.zeros(1, 1, 4, 4, 4),
        "context_out": torch.zeros(1, 4, 4, 4, dtype=torch.long),
        "subject": "s0", "context_subjects": ["s1"], "label_name": "liver",
        "spacing": torch.ones(3), "aug_mode": torch.tensor(0, dtype=torch.long),
        "crop_geom": torch.zeros(4, 3, dtype=torch.long),
        "modality": modality,
    }


def test_collate_emits_modality_list():
    batch = incontext_collate_fn([_item("ct"), _item("mri")])
    assert batch["modality"] == ["ct", "mri"]


def test_collate_omits_modality_when_absent():
    it = _item("ct")
    del it["modality"]
    batch = incontext_collate_fn([it, it])
    assert "modality" not in batch
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_modality_seam.py -v`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'modality'` on `test_loadresult_modality_settable`, and `KeyError`/assert failure on the collate tests.

- [ ] **Step 3: Add the `modality` field to `LoadResult` and `NativeCrop`**

In `src/incontext_dataset_v2.py`, the `LoadResult` dataclass — add the field last (after `crop_geom`):

```python
@dataclass
class LoadResult:
    image: torch.Tensor                # (1, T, T, T) f32, normalized
    label: torch.Tensor               # (T, T, T) i64, binary {0,1}
    spacing: torch.Tensor              # (3,) mm/voxel of the output
    crop_geom: torch.Tensor            # (4, 3) i64: starts, crop_sizes, out_sizes, pad_lo
    modality: str = "ct"              # "ct" | "mri" — rides for aug/analysis; encoder path ignores it
```

In `src/providers/totalseg.py`, the `NativeCrop` dataclass — add the field last (after `decim`):

```python
    decim: tuple                  # per-axis integer decimation factor (>=1)
    modality: str = "ct"         # "ct" | "mri" — carried for the GPU realize/aug frame
```

- [ ] **Step 4: Populate `modality` at the provider return sites**

In `src/providers/totalseg.py`:
- `load()` return (~line 258):
  ```python
  return LoadResult(image=image_t, label=label_t, spacing=spacing, crop_geom=geom,
                    modality=self.modality)
  ```
- `build_native_crop` signature (~line 59) — add to the keyword-only block:
  ```python
  def build_native_crop(crop_ct, crop_lbl, class_idx, out_sizes, pad_lo, geom, *,
                        crop_spacing_mm, ct_spec=None, modality="ct"):
  ```
  and its `NativeCrop(...)` return (~line 85) — add `modality=modality` last.
- `load_native_crop()` return (~line 287) — pass it through:
  ```python
  return build_native_crop(
      crop_ct, crop_lbl, _ALL_CLASSES_IDX.get(cls, -1), out_sizes, pad_lo, geom,
      crop_spacing_mm=float(req.crop_spacing_mm),
      ct_spec=(self.ct_spec if self.modality == "ct" else None),
      modality=self.modality)
  ```

In `src/providers/native_grid.py`, the `LoadResult` return (~line 133):
```python
        return LoadResult(image=image_t, label=label_t, spacing=spacing, crop_geom=geom,
                          modality=getattr(self, "modality", "ct"))
```

- [ ] **Step 5: Carry `modality` into the item dict**

In `src/incontext_dataset_v2.py`, `__getitem__`, the non-realize return dict (~lines 188-199) — add one key:

```python
        return {
            "image": image_t,
            "label": label_t,
            "context_in": ctx_in,
            "context_out": ctx_out,
            "subject": subj,
            "context_subjects": ctx_subjects,
            "label_name": cls,
            "spacing": tgt.spacing,
            "aug_mode": torch.tensor(0, dtype=torch.long),
            "crop_geom": tgt.crop_geom,
            "modality": tgt.modality,
        }
```

(The `gpu_realize_crop` branch return at ~line 153 and the `cohort_mode` branch are left unchanged — cascade/realize-path modality is a later follow-up per the spec's out-of-scope; `NativeCrop.modality` is populated but not yet collated.)

- [ ] **Step 6: Emit `batch["modality"]` from the collate**

In `src/totalseg_dataloader_incontext.py`, `incontext_collate_fn`, after the `if "aug_mode" in batch[0]:` block (~line 1571):

```python
    if "modality" in batch[0]:
        out["modality"] = [b["modality"] for b in batch]  # (B,) list[str], unused downstream
```

- [ ] **Step 7: Run the test to verify it passes**

Run: `python -m pytest tests/test_modality_seam.py -v`
Expected: PASS (5 tests)

- [ ] **Step 8: Regression — v2 dataloader + collate tests**

Run: `python -m pytest tests/ experiments/3d/tests/ -k "dataloader or collate or dataset_v2 or incontext or provider or cascade_provider" -v`
Expected: PASS (no test references a fixed key set that a new `modality` key breaks).

- [ ] **Step 9: Commit**

```bash
PATH="/software/anaconda3/envs/git/bin:$PATH" git add src/incontext_dataset_v2.py src/providers/totalseg.py src/providers/native_grid.py src/totalseg_dataloader_incontext.py tests/test_modality_seam.py
PATH="/software/anaconda3/envs/git/bin:$PATH" git commit -m "feat: modality field on LoadResult/NativeCrop/batch (no-op seam)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017LE3uxB7pNGutyT2deEGhp"
```

---

## Task 4: `gpu_augment.py` de-pinnable clamp frame

**Files:**
- Modify: `src/gpu_augment.py` (`_batched_gin_ipa` ~line 115; `_batched_bias_field` ~line 134; `_batched_intensity` ~lines 143-254; `GpuAugmentor.__init__` ~lines 407-420; the 3 `_batched_intensity` call sites at ~446, ~479, ~487)
- Test: `tests/test_gpu_augment.py`

**Interfaces:**
- Consumes: `CT_NORM_MIN`, `CT_NORM_MAX` (existing module imports) as the default.
- Produces:
  - `_batched_gin_ipa(vols, cfg, gen, clamp=None)` — `clamp` is `None` (→ `(CT_NORM_MIN, CT_NORM_MAX)`) or `(lo, hi)`.
  - `_batched_bias_field(vols, magnitude, coarse, gen, clamp=None)` — same.
  - `_batched_intensity(vols, cfg, gen, clamp=None)` — same; passes `clamp` down to the two helpers it calls.
  - `GpuAugmentor.__init__(..., clamp_frame=None)` — `None` keeps the CT-frame guard; a `(lo, hi)` tuple sets `self._clamp` and skips the guard. Stored as `self._clamp: tuple[float, float] | None`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_gpu_augment.py` (keep existing imports; add any missing):

```python
def test_batched_intensity_clamp_default_is_ct_frame():
    import torch
    from src.gpu_augment import _batched_intensity, CT_NORM_MIN, CT_NORM_MAX

    class _NC:  # gaussian-noise-only cfg forcing a large clamp excursion
        class gaussian_noise:
            p = 1.0
            max_std = 50.0
    g = torch.Generator().manual_seed(0)
    vols = torch.zeros(2, 1, 8, 8, 8)
    out = _batched_intensity(vols, _NC, g)
    assert out.max() <= CT_NORM_MAX + 1e-4
    assert out.min() >= CT_NORM_MIN - 1e-4


def test_batched_intensity_clamp_override():
    import torch
    from src.gpu_augment import _batched_intensity

    class _NC:
        class gaussian_noise:
            p = 1.0
            max_std = 50.0
    g = torch.Generator().manual_seed(0)
    vols = torch.zeros(2, 1, 8, 8, 8)
    out = _batched_intensity(vols, _NC, g, clamp=(-4.0, 4.0))
    assert out.max() <= 4.0 + 1e-4
    assert out.min() >= -4.0 - 1e-4


def test_gpu_augmentor_clamp_frame_skips_ct_guard():
    from src.gpu_augment import GpuAugmentor
    # A non-default ct_norm normally raises; clamp_frame set -> allowed.
    aug = GpuAugmentor(aug_cfg=None,
                       ct_norm={"clip_lo": -500.0, "clip_hi": 500.0, "mean": 0.0, "std": 100.0},
                       clamp_frame=(-3.0, 3.0))
    assert aug._clamp == (-3.0, 3.0)


def test_gpu_augmentor_default_still_guards():
    import pytest
    from src.gpu_augment import GpuAugmentor
    with pytest.raises(NotImplementedError):
        GpuAugmentor(aug_cfg=None,
                     ct_norm={"clip_lo": -500.0, "clip_hi": 500.0, "mean": 0.0, "std": 100.0})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_gpu_augment.py -k "clamp or clamp_frame or ct_guard" -v`
Expected: FAIL — `_batched_intensity() got an unexpected keyword argument 'clamp'`; `GpuAugmentor.__init__() got an unexpected keyword argument 'clamp_frame'`.

- [ ] **Step 3: Thread `clamp` through the three functions**

`_batched_gin_ipa` (~line 115) — add the param and resolve at the return:

```python
def _batched_gin_ipa(vols, cfg, gen, clamp=None):
    ...
    lo, hi = clamp if clamp is not None else (CT_NORM_MIN, CT_NORM_MAX)
    return out.clamp(lo, hi)
```

`_batched_bias_field` (~line 134):

```python
def _batched_bias_field(vols, magnitude, coarse, gen, clamp=None):
    ...
    lo, hi = clamp if clamp is not None else (CT_NORM_MIN, CT_NORM_MAX)
    return (vols * field.exp()).clamp(lo, hi)
```

`_batched_intensity` (~line 143) — add the param, resolve `lo`/`hi`/`span` once at the top, replace every `CT_NORM_MIN` → `lo`, `CT_NORM_MAX` → `hi` inside the function body (the clamp calls at ~180, ~195, ~211, ~227 and the gamma remaps at ~189, ~203 which currently use `CT_NORM_MIN` + `span`), and pass `clamp=clamp` to the two helper calls (~156, ~163):

```python
def _batched_intensity(vols, cfg, gen, clamp=None):
    ...
    N = vols.shape[0]
    device = vols.device
    lo, hi = clamp if clamp is not None else (CT_NORM_MIN, CT_NORM_MAX)
    span = hi - lo
    ...
        aug = _batched_gin_ipa(vols, gin, gen, clamp=clamp)
    ...
        aug = _batched_bias_field(vols, bf.magnitude, int(getattr(bf, "coarse", 4)), gen, clamp=clamp)
    ...
    # every remaining `CT_NORM_MIN` -> `lo`, `CT_NORM_MAX` -> `hi` in this function
```

Concrete replacements inside `_batched_intensity` (do not touch occurrences elsewhere in the module):
- `aug = aug.clamp(CT_NORM_MIN, CT_NORM_MAX)` → `aug = aug.clamp(lo, hi)`
- `((vols - CT_NORM_MIN) / span).clamp(0, 1).pow(gamma) * span + CT_NORM_MIN` → `((vols - lo) / span).clamp(0, 1).pow(gamma) * span + lo` (both the gamma and inverted-gamma lines)
- `((aug - m_out) / (s_out + 1e-8) * s_in + m_in).clamp(CT_NORM_MIN, CT_NORM_MAX)` → `... .clamp(lo, hi)`
- `aug = (vols + sc.factor * (vols - blur)).clamp(CT_NORM_MIN, CT_NORM_MAX)` → `.clamp(lo, hi)`
- `aug = (vols + noise).clamp(CT_NORM_MIN, CT_NORM_MAX)` → `.clamp(lo, hi)`

- [ ] **Step 4: Add `clamp_frame` to `GpuAugmentor` and pass it at the call sites**

`__init__` (~line 407) — new param, relaxed guard, stored tuple:

```python
    def __init__(self, aug_cfg, self_context_per_image: bool = False,
                 self_context_intensity: bool = False, seed: int = 0, ct_norm=None,
                 clamp_frame=None):
        # Intensity ops clamp to CT_NORM_MIN/MAX (the default CT frame) unless an explicit
        # clamp_frame (lo, hi) is given — the seam for a non-CT / multi-modality frame.
        self._clamp = None if clamp_frame is None else (float(clamp_frame[0]), float(clamp_frame[1]))
        if self._clamp is None and resolve_ct_norm(ct_norm) != DEFAULT_CT_NORM:
            raise NotImplementedError(
                "GpuAugmentor is pinned to the default CT frame (fingerprint_1228); "
                f"data.ct_norm={ct_norm!r} needs an explicit clamp_frame=(lo, hi).")
        self.cfg = aug_cfg
        self.self_context_per_image = bool(self_context_per_image)
        self.self_context_intensity = bool(self_context_intensity)
        self._seed = seed
        self._step = 0
```

The three `_batched_intensity(...)` call sites — add `clamp=self._clamp`:
- ~line 446 (`apply`): `vols = _batched_intensity(vols, cfg.intensity, int_gen, clamp=self._clamp)`
- ~line 479 (`__call__`, synth): `v = _batched_intensity(v, cfg.synth, gen, clamp=self._clamp)`
- ~line 487 (`__call__`, intensity): `v = _batched_intensity(v, cfg.intensity, gen, clamp=self._clamp)`

- [ ] **Step 5: Run the new tests + the full gpu_augment suites**

Run: `python -m pytest tests/test_gpu_augment.py experiments/3d/tests/test_gpu_augment_capture.py -v`
Expected: PASS — all pre-existing tests (default path untouched) plus the 4 new ones.

- [ ] **Step 6: Commit**

```bash
PATH="/software/anaconda3/envs/git/bin:$PATH" git add src/gpu_augment.py tests/test_gpu_augment.py
PATH="/software/anaconda3/envs/git/bin:$PATH" git commit -m "feat: GpuAugmentor clamp_frame seam (default = CT frame, unchanged)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017LE3uxB7pNGutyT2deEGhp"
```

---

## Task 5: Full-suite regression + parity verification + logs

**Files:**
- Modify: `docs/logs.md` (append an entry)

**Interfaces:**
- Consumes: everything from Tasks 1-4.
- Produces: a `docs/logs.md` entry; no code.

- [ ] **Step 1: Run the whole test suite**

Run: `python -m pytest tests/ experiments/3d/tests/ -q`
Expected: PASS (same pass/skip count as before the branch, plus the ~19 new tests from Tasks 1/3/4 and 2 from Task 2). Record the exact `N passed / M skipped` line.

- [ ] **Step 2: exp 70 config-resolution parity**

Run: `python experiments/3d/train.py experiment=70_patchset_varspacing_6_1_5 --cfg job --resolve > /tmp/exp70_after.yaml 2>&1; git stash; python experiments/3d/train.py experiment=70_patchset_varspacing_6_1_5 --cfg job --resolve > /tmp/exp70_before.yaml 2>&1; git stash pop; diff /tmp/exp70_before.yaml /tmp/exp70_after.yaml && echo "PARITY OK"`
Expected: `PARITY OK` (empty diff). (Use `PATH="/software/anaconda3/envs/git/bin:$PATH"` for the `git stash` calls.)

- [ ] **Step 3: `_norm` numeric parity for the three encoder defaults**

Run:

```bash
python -c "
import torch
from src.models.encoders.plainconv_ts import PlainConvTSEncoder
from src.models.encoders.resenc_ts import ResEncTSEncoder
torch.manual_seed(0)
x = torch.randn(3, 1, 16, 16, 16)
pc = PlainConvTSEncoder(resolution=8, n_stages=4, stages=(1,2,3), input_norm='zscore', device='cpu', precision='fp32')
re = ResEncTSEncoder(resolution=8, n_stages=4, stages=(1,2,3), input_norm='passthrough', device='cpu', precision='fp32')
ld = pc.input_renorm._loader
hu = x.float()*ld.std + ld.mean
flat = hu.reshape(3,-1); mu = flat.mean(1).reshape(-1,1,1,1,1); sig = flat.std(1).reshape(-1,1,1,1,1)
assert torch.allclose(pc._norm(x), (hu-mu)/(sig+1e-8), atol=1e-6), 'plainconv zscore drift'
assert torch.equal(re._norm(x), x.float()), 'resenc passthrough drift'
print('ENCODER _norm PARITY OK')
"
```

Expected: `ENCODER _norm PARITY OK`

- [ ] **Step 4: Append the logs entry**

Add to `docs/logs.md`:

```markdown
## 2026-09-03 — modality-agnostic normalization: prep seams (no-op)

Landed three seams from
docs/superpowers/specs/2026-09-03-modality-agnostic-normalization-design.md so a later
CT+MRI joint run is config-only. Zero behavior change — exp 70 `--cfg job --resolve` diff
empty; encoder `_norm` numerically identical for every current default.

- `src/models/encoders/_input_norm.py` (new): `InputRenorm` stem, modes
  `passthrough | reframe | zscore | instance`. The first three are the previous inline
  `_norm` extracted verbatim; `instance` = per-sample z-score of the tensor as received,
  NO HU inversion (modality-agnostic), optional default-off learned affine.
- `plainconv_ts` / `resenc_ts` / `nnunet_ts`: `_norm` now delegates to a shared
  `InputRenorm` instance; `_INPUT_NORMS` moved to `_input_norm.py`; `nnunet_ts` passes its
  plans `CTNormalization` as `target_spec`. Per-encoder defaults unchanged
  (zscore / passthrough / reframe). `input_norm='instance'` now accepted (unreachable
  until a config selects it).
- `modality: str = "ct"` on `LoadResult` + `NativeCrop`; `build_native_crop(modality=)`;
  provider returns populate it; v2 item dict carries `"modality"`; `incontext_collate_fn`
  emits `batch["modality"]` (list[str], nothing reads it). Cascade/realize collate not
  threaded — later follow-up.
- `src/gpu_augment.py`: `_batched_intensity` / `_batched_gin_ipa` / `_batched_bias_field`
  take `clamp=(lo,hi)` (None → the CT-frame constants); `GpuAugmentor(clamp_frame=...)`
  stores it and, when set, skips the `ct_norm != DEFAULT` guard. Default path byte-identical.

Tests: `tests/test_input_norm.py` (new), `tests/test_modality_seam.py` (new),
`instance` cases in `tests/test_{plainconv,resenc}_ts_encoder.py`, `clamp_frame` cases in
`tests/test_gpu_augment.py`. Full suite: <N> passed / <M> skipped.
```

Fill `<N>` / `<M>` from Step 1.

- [ ] **Step 5: Commit**

```bash
PATH="/software/anaconda3/envs/git/bin:$PATH" git add docs/logs.md
PATH="/software/anaconda3/envs/git/bin:$PATH" git commit -m "docs: log modality-agnostic normalization prep seams

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_017LE3uxB7pNGutyT2deEGhp"
```

---

## Self-Review

**1. Spec coverage:**
- Spec §2 (invariant stem, `instance` mode, affine sub-flag, shared module de-dups 3 encoders, `nnunet_ts` target via `target_spec`) → Tasks 1 + 2. ✓
- Spec §5.1 (`_input_norm.py`, refactor, parity test) → Task 1 + Task 2 Steps 6/7 + Task 5 Step 3. ✓
- Spec §5.2 (`modality` on `LoadResult`/`NativeCrop`/`batch["modality"]`, unused) → Task 3. ✓
- Spec §5.3 (`gpu_augment.py` `clamp_frame` arg, instance-level bounds, relaxed guard) → Task 4. ✓
- Spec §6 (every current config identical; `--cfg job` diff empty; forward-parity) → Task 2 Step 8 + Task 5 Steps 2/3. ✓
- Spec "Out of scope" (`MultiModalProvider`, `[0,1]` augmentor rewrite, MRI+gpu_realize, `feat_norm` tuning, `data.ct_norm` semantics) → not in any task. ✓ (`robust` mode deliberately deferred per spec §5.1 "not on the critical path" — the enum ships without it; a later PR adds the mean/std→median/IQR swap.)

**2. Placeholder scan:** No "TBD"/"add error handling"/"similar to Task N". Every code step has full code. Line numbers are approximate ("~line N") because they will drift as edits land — each is paired with an anchoring code snippet or symbol name. `<N>`/`<M>` in Task 5 Step 4 are explicitly filled from Step 1 output.

**3. Type consistency:** `InputRenorm(mode, *, loader_spec, target_spec, affine, eps)` — same signature in Task 1 def, Task 2 call sites (plainconv/resenc pass `loader_spec=`/`target_spec=`; nnunet passes `loader_spec=`/`target_spec=`). `_INPUT_NORMS` is a 4-tuple everywhere. `modality: str = "ct"` — same field name and default on `LoadResult`, `NativeCrop`, `build_native_crop`, item dict key, `batch["modality"]`. `clamp` param name consistent across the 3 augmentor functions; `clamp_frame` (constructor) vs `self._clamp` (stored tuple) vs `clamp=` (function kwarg) — distinct by design, used consistently.
