# Multi-source (CT + MRI) cascade support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let `data.source=multisource` (the joint CT/MRI cohort provider) run through the N-level PatchSet3D cascade with `gpu_realize_crop=true` + `ram_cache=true`.

**Architecture:** `NativeCrop` gains a per-crop `CtNormSpec` so CT and MRI crops each normalize correctly (this alone unblocks single-source MRI GPU-realize). `MultiSourceProvider` gains modality-dispatching `load` / `load_native_crop` and a native-payload branch in `assemble_task`. The target and context modality (scalars per task row) are threaded through both collate functions and `realize_cascade_level0` into `cascade._recrop_level`, which routes each re-crop load to the right sub-provider. Single-source paths stay byte-identical — the `modality=` kwarg is only passed when the batch carries the keys.

**Tech Stack:** Python, PyTorch, Hydra/OmegaConf, pytest. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-09-07-multisource-cascade-design.md`

## Global Constraints

- Repo: `patch_icl`. Run everything from the repo root `/home/dpxuser/dev/patch_icl`.
- Python env is node-specific (see memory `feedback_python_env`). For unit tests, `python -m pytest` under whatever env is active is fine — the unit tests here are pure-CPU (torch tensors only, no CUDA, no model). If imports fail, prefer `.venv_nero` (Ampere) or the env `which python` resolves to.
- Log every change to `docs/logs.md` (project rule, `CLAUDE.md`).
- Write tests only where they carry their weight (project rule). Each task below specifies its tests.
- Commit frequently — one commit per task minimum, at the TDD boundaries shown.
- Do NOT reformat or refactor code outside the lines each task names.
- `CtNormSpec` is `src.totalseg_dataset.CtNormSpec` — a frozen-ish dataclass with fields `clip_lo, clip_hi, mean, std` (all float). `resolve_ct_norm(x)` maps `None`→default CT spec, a preset name→spec, a `CtNormSpec`→itself, a mapping `{clip_lo,clip_hi,mean,std}`→spec.
- MRI per-subject stats live in `TotalSegProvider._ct_stats` (a `dict[str, dict]`), populated in `__init__` only when `modality == "mri"`. Each entry has exactly the keys `{clip_lo, clip_hi, mean, std}` (`src.totalseg_dataset.mri_stats`), so `resolve_ct_norm(self._ct_stats[subject])` yields a `CtNormSpec`.
- `normalize_ct` and `normalize_mri` are both pointwise `clip(lo,hi) → (x-mean)/std`; only the constants differ (global vs per-subject). So one `CtNormSpec` per crop reproduces either.

---

## File Structure

| File | Responsibility | Change |
|------|----------------|--------|
| `src/providers/totalseg.py` | `NativeCrop` dataclass, `build_native_crop`, `TotalSegProvider.load_native_crop` | Add `NativeCrop.norm`; `build_native_crop` takes a resolved `norm`; `load_native_crop` passes CT global spec / per-subject MRI spec |
| `src/gpu_realize_crop.py` | GPU realization + native-crop collate | `_realize_member` uses `nc.norm`; `native_crop_collate_fn` passes `tgt_modality`/`ctx_modality` through |
| `src/providers/multisource.py` | Cohort CT/MRI task assembler | `gpu_realize_crop` ctor flag; native branch in `assemble_task` + `*_modality` keys; `load`/`load_native_crop` modality dispatch |
| `src/totalseg_dataloader_incontext.py` | `incontext_collate_fn` | Pass `tgt_modality`/`ctx_modality` through |
| `experiments/3d/cascade.py` | N-level cascade loop | `realize_cascade_level0` re-attaches `*_modality`; `_recrop_level` per-task modality routing |
| `experiments/3d/common.py` | dataset/loader builders + cascade guard | multisource branch wires `gpu_realize_crop` + `ram_cache`; `_assert_cascade_supported` allows `multisource`, drops the MRI+realize block |
| `experiments/3d/tests/test_gpu_realize_crop.py` | realize parity tests | Add MRI-spec + mixed-modality member cases |
| `src/providers/test_multisource.py` | provider logic tests | Add native-emission + dispatch cases |
| `experiments/3d/tests/test_cascade_guard.py` | guard tests | Update MRI case; add multisource-allowed case |
| `experiments/3d/tests/test_recrop_modality.py` | NEW — `_recrop_level` routing test | Create |
| `experiments/3d/_check_multisource.py` | NFS integration check | Add cascade-override build assertion |
| `docs/logs.md` | change log | Append entry + runnable command |

---

## Task 1: `NativeCrop` carries its normalization spec

**Files:**
- Modify: `src/providers/totalseg.py` — `NativeCrop` (dataclass, ~line 30-50), `build_native_crop` (~line 60-90), `TotalSegProvider.load_native_crop` (~line 262-293)
- Modify: `src/gpu_realize_crop.py` — `_realize_member` (~line 34-75)
- Test: `experiments/3d/tests/test_gpu_realize_crop.py`

**Interfaces:**
- Consumes: nothing new.
- Produces:
  - `NativeCrop.norm: Optional[CtNormSpec] = None` — new dataclass field (last, with default, so existing positional constructions in tests still work).
  - `build_native_crop(crop_ct, crop_lbl, class_idx, out_sizes, pad_lo, geom, *, crop_spacing_mm, norm=None, modality="ct")` — the `ct_spec` keyword is **renamed to `norm`**. It is used for BOTH the pre-decimation HU clip (as `ct_spec` was) AND stored on the payload as `NativeCrop.norm`.
  - `_realize_member` normalizes with `nc.norm if nc.norm is not None else ct_spec`.

- [ ] **Step 1: Write the failing test**

Add to `experiments/3d/tests/test_gpu_realize_crop.py`:

```python
from src.totalseg_dataset import CtNormSpec, normalize_mri


def test_realize_member_uses_per_crop_norm_spec_mri():
    """A NativeCrop carrying an MRI-style per-subject spec is normalized with THAT
    spec, not the CT fingerprint passed to realize_native_crops."""
    D = 24
    # MRI-ish intensities: strictly positive, ~[0, 900]
    img = (np.linspace(10, 890, D, dtype=np.float32)[:, None, None]
           + np.linspace(0, 200, D, dtype=np.float32)[None, :, None]).astype(np.float16)
    img = np.broadcast_to(img, (D, D, D)).copy()
    lbl = np.zeros((D, D, D), np.uint8); lbl[8:16, 8:16, 8:16] = 3

    mri_stats = {"clip_lo": 5.0, "clip_hi": 950.0, "mean": 300.0, "std": 120.0}
    mri_spec = resolve_ct_norm(mri_stats)
    ct_spec = resolve_ct_norm(None)                    # deliberately WRONG for this crop

    crop_ct, crop_lbl, out_sizes, pad_lo, geom = organ_crop_arrays(
        img, lbl, (12, 12, 12), [1.5, 1.5, 1.5], image_size=(8, 8, 8),
        crop_mm=1.5, jitter=0, rng=random.Random(0))
    nc = build_native_crop(crop_ct, crop_lbl, 3, out_sizes, pad_lo, geom,
                           crop_spacing_mm=1.5, norm=mri_spec, modality="mri")
    assert nc.norm == mri_spec

    out = realize_native_crops([[nc]], T=8, mask_downsample="soft", occ_thr=0.5,
                               ct_spec=ct_spec, device="cpu")
    # Reference: normalize_mri applied BEFORE the same crop+resample (decim=1 here).
    ref_i, _, _ = crop_and_place(
        img, lbl, 3, (12, 12, 12), 8, crop_spacing_mm=1.5,
        native_spacing=(1.5, 1.5, 1.5), jitter=0, rng=random.Random(0),
        mask_downsample="soft", occ_thr=0.5,
        normalize_fn=lambda a: normalize_mri(a, mri_stats))
    assert (out["image"][0] - ref_i).abs().max() < 2e-2


def test_realize_batch_mixes_ct_and_mri_members():
    """One batch, target CT (global spec) + context MRI (per-subject spec)."""
    D = 24
    img_ct = _smooth_vol(D)
    img_mri = np.broadcast_to(
        np.linspace(10, 890, D, dtype=np.float32)[:, None, None], (D, D, D)).astype(np.float16).copy()
    lbl = np.zeros((D, D, D), np.uint8); lbl[10:14, 10:14, 10:14] = 3

    ct_spec = resolve_ct_norm(None)
    mri_spec = resolve_ct_norm({"clip_lo": 5.0, "clip_hi": 950.0, "mean": 300.0, "std": 120.0})

    def _mk(image_np, spec, modality):
        cc, cl, os_, pl, g = organ_crop_arrays(
            image_np, lbl, (12, 12, 12), [1.5, 1.5, 1.5], image_size=(8, 8, 8),
            crop_mm=1.5, jitter=0, rng=random.Random(0))
        return build_native_crop(cc, cl, 3, os_, pl, g, crop_spacing_mm=1.5,
                                 norm=spec, modality=modality)

    members = [[_mk(img_ct, ct_spec, "ct"), _mk(img_mri, mri_spec, "mri")]]
    out = realize_native_crops(members, T=8, mask_downsample="occupancy", occ_thr=0.1,
                               ct_spec=ct_spec, device="cpu")
    assert out["image"].shape == (1, 1, 8, 8, 8)
    assert out["context_in"].shape == (1, 1, 1, 8, 8, 8)
    # CT air ≈ ct_spec.norm_min (≈ -1.66); MRI context min ≈ (clip_lo-mean)/std of its spec.
    assert abs(float(out["image"][0].min()) - ct_spec.norm_min) < 0.2
    assert abs(float(out["context_in"][0, 0].min()) - mri_spec.norm_min) < 0.2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest experiments/3d/tests/test_gpu_realize_crop.py::test_realize_member_uses_per_crop_norm_spec_mri experiments/3d/tests/test_gpu_realize_crop.py::test_realize_batch_mixes_ct_and_mri_members -v`
Expected: FAIL — `build_native_crop() got an unexpected keyword argument 'norm'` (and `NativeCrop` has no `norm`).

- [ ] **Step 3: Add `NativeCrop.norm`**

In `src/providers/totalseg.py`, add to the `NativeCrop` dataclass as the LAST field (after `modality`):

```python
    modality: str = "ct"         # "ct" | "mri" — carried for the GPU realize/aug frame
    norm: "CtNormSpec | None" = None  # per-crop normalization spec (global CT / per-subject MRI)
```

`CtNormSpec` is already importable — extend the existing import at the top of the file:

```python
from src.totalseg_dataset import (_ALL_CLASSES_IDX, normalize_ct, normalize_mri,
                                  resolve_ct_norm, CtNormSpec)
```

- [ ] **Step 4: Rename `build_native_crop`'s `ct_spec` param to `norm` and store it**

In `src/providers/totalseg.py`, `build_native_crop`:

```python
def build_native_crop(crop_ct, crop_lbl, class_idx, out_sizes, pad_lo, geom, *,
                      crop_spacing_mm, norm=None, modality="ct"):
```

Update the docstring line `Image: HU-clipped to \`ct_spec\` FIRST ...` → `\`norm\``. In the body, change the clip:

```python
    img_t = torch.from_numpy(np.array(crop_ct)).float()
    if norm is not None:
        img_t = img_t.clamp(norm.clip_lo, norm.clip_hi)
```

and the return:

```python
    return NativeCrop(image=img_t.half(), label_frac=frac_t.half(),
                      class_idx=int(class_idx), has_fg=has_fg,
                      out_sizes=list(out_sizes), pad_lo=list(pad_lo),
                      crop_geom=geom, crop_spacing_mm=float(crop_spacing_mm),
                      decim=decim, modality=modality, norm=norm)
```

- [ ] **Step 5: Pass the right spec from `load_native_crop`**

In `src/providers/totalseg.py`, `TotalSegProvider.load_native_crop`, replace the final `return build_native_crop(...)` block:

```python
        # Per-crop normalization spec: global CT fingerprint for CT, per-subject
        # foreground stats for MRI (normalize_mri is the same pointwise clip+z-score
        # form, so one CtNormSpec reproduces it). The GPU realize step reads nc.norm.
        norm = (self.ct_spec if self.modality == "ct"
                else resolve_ct_norm(self._ct_stats[subject]))
        return build_native_crop(
            crop_ct, crop_lbl, _ALL_CLASSES_IDX.get(cls, -1), out_sizes, pad_lo, geom,
            crop_spacing_mm=float(req.crop_spacing_mm),
            norm=norm, modality=self.modality)
```

- [ ] **Step 6: `_realize_member` honours `nc.norm`**

In `src/gpu_realize_crop.py`, `_realize_member`, change the normalize line:

```python
    spec = nc.norm if getattr(nc, "norm", None) is not None else ct_spec
    src = normalize_ct_gpu(nc.image.to(device), spec)[None, None]
```

Update the docstring: `CT-normalize FIRST` → `normalize FIRST (per-crop nc.norm — CT fingerprint or per-subject MRI stats)`.

- [ ] **Step 7: Fix the existing `_native_crop_from` test helper**

In `experiments/3d/tests/test_gpu_realize_crop.py`, the helper `_native_crop_from` (~line 26) still calls `build_native_crop(..., ct_spec=...)`. Rename that kwarg:

```python
    return build_native_crop(crop_ct, crop_lbl, cls_idx, out_sizes, pad_lo, geom,
                             crop_spacing_mm=spacing,
                             norm=(spec if spec is not None else resolve_ct_norm(None)))
```

- [ ] **Step 8: Run the full realize test file**

Run: `python -m pytest experiments/3d/tests/test_gpu_realize_crop.py -v`
Expected: PASS — the two new tests plus all pre-existing parity tests (they exercised `ct_spec=`; now `norm=`).

- [ ] **Step 9: Grep for other `build_native_crop` callers**

Run: `grep -rn "build_native_crop" src/ experiments/ --include=*.py`
Expected: only `src/providers/totalseg.py` (def + the one call) and `experiments/3d/tests/test_gpu_realize_crop.py`. If any other caller passes `ct_spec=`, rename it to `norm=` there too.

- [ ] **Step 10: Commit**

```bash
git add src/providers/totalseg.py src/gpu_realize_crop.py experiments/3d/tests/test_gpu_realize_crop.py
git commit -m "feat: per-crop CtNormSpec on NativeCrop (unblocks MRI GPU-realize)

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_019ZbXsSuph6AU8dMhWhycxf"
```

---

## Task 2: `MultiSourceProvider` — modality dispatch + native emission

**Files:**
- Modify: `src/providers/multisource.py` — `__init__` (~line 21-32), `load` (~line 55-56), add `load_native_crop`, `assemble_task` (~line 74-126)
- Test: `src/providers/test_multisource.py`

**Interfaces:**
- Consumes: `NativeCrop` (Task 1) via `self.subs[mod].load_native_crop(...)`.
- Produces:
  - `MultiSourceProvider(sub_providers, *, context_size, regime_p=..., epoch_length=1000, gpu_realize_crop=False)`.
  - `provider.load(subject, cls, req, *, modality)` — dispatches to `self.subs[modality].load`; `modality=None` (or omitted) raises `RuntimeError`.
  - `provider.load_native_crop(subject, cls, req, *, modality)` — dispatches to `self.subs[modality].load_native_crop`.
  - `assemble_task` return dict, `gpu_realize_crop=False` (painted): existing keys **plus** `"tgt_modality": str`, `"ctx_modality": str`.
  - `assemble_task` return dict, `gpu_realize_crop=True` (native): `{"native_crop": [tgt_nc, *ctx_ncs], "subject": str, "context_subjects": list[str], "label_name": str, "tgt_modality": str, "ctx_modality": str, "aug_mode": tensor(0), "meta": {...}}` — **no** `"image"` key.

- [ ] **Step 1: Write the failing tests**

Add to `src/providers/test_multisource.py`. Extend `_FakeSub` with a `load_native_crop` and a call log:

```python
class _FakeSubNC(_FakeSub):
    def __init__(self, modality, class_to_subjects):
        super().__init__(modality, class_to_subjects)
        self.nc_loaded = []

    def load_native_crop(self, subject, cls, req):
        self.nc_loaded.append((subject, cls))
        # a stand-in payload; the provider only forwards it
        return {"_fake_nc": True, "modality": self.modality,
                "subject": subject, "spacing": req.crop_spacing_mm}


def _mk_nc(regime_p=(1 / 3, 1 / 3, 1 / 3), context_size=1, gpu_realize_crop=False):
    ct = _FakeSubNC("ct", {"a": ["ca0", "ca1", "ca2"],
                           "b": ["cb0", "cb1", "cb2", "cb3"],
                           "c": ["cc0", "cc1", "cc2"]})
    mri = _FakeSubNC("mri", {"b": ["mb0", "mb1", "mb2"],
                             "c": ["mc0", "mc1", "mc2"],
                             "d": ["md0", "md1", "md2"]})
    prov = MultiSourceProvider({"ct": ct, "mri": mri}, context_size=context_size,
                               regime_p=regime_p, epoch_length=99,
                               gpu_realize_crop=gpu_realize_crop)
    return prov, ct, mri


def test_painted_item_has_explicit_modality_keys():
    prov, _, _ = _mk_nc()
    it = prov.assemble_task(random.Random(0), 3.0)
    assert it["tgt_modality"] == it["meta"]["tgt_mod"]
    assert it["ctx_modality"] == it["meta"]["ctx_mod"]
    assert "image" in it                       # painted path unchanged


def test_native_emission_shape_and_no_image():
    prov, ct, mri = _mk_nc(context_size=2, gpu_realize_crop=True)
    it = prov.assemble_task(random.Random(0), 4.0)
    assert "image" not in it and "native_crop" in it
    assert len(it["native_crop"]) == 3         # target + 2 contexts
    assert set(it) >= {"native_crop", "subject", "context_subjects", "label_name",
                       "tgt_modality", "ctx_modality", "aug_mode", "meta"}
    assert it["tgt_modality"] == it["meta"]["tgt_mod"]
    assert it["ctx_modality"] == it["meta"]["ctx_mod"]
    assert int(it["aug_mode"]) == 0
    # every payload came from the sub-provider matching its slot's modality
    assert it["native_crop"][0]["modality"] == it["tgt_modality"]
    for m in it["native_crop"][1:]:
        assert m["modality"] == it["ctx_modality"]


def test_native_emission_cross_regime_routes_both_subproviders():
    prov, ct, mri = _mk_nc(regime_p=(0.0, 0.0, 1.0), gpu_realize_crop=True)
    for _ in range(50):
        it = prov.assemble_task(random.Random(_), 3.0)
        if it["tgt_modality"] != it["ctx_modality"]:
            break
    else:
        raise AssertionError("no cross-modality task in 50 draws")
    assert ct.nc_loaded and mri.nc_loaded       # both sub-providers were hit


def test_load_dispatch_by_modality():
    prov, ct, mri = _mk_nc()
    req = LoadRequest(rng=random.Random(0), crop_spacing_mm=3.0)
    prov.load("mb0", "b", req, modality="mri")
    assert mri.loaded[-1] == ("mb0", "b") and not ct.loaded
    prov.load_native_crop("cb0", "b", req, modality="ct")
    assert ct.nc_loaded[-1] == ("cb0", "b")


def test_load_without_modality_raises():
    prov, _, _ = _mk_nc()
    req = LoadRequest(rng=random.Random(0), crop_spacing_mm=3.0)
    with pytest.raises(RuntimeError, match="modality"):
        prov.load("mb0", "b", req)
```

`LoadRequest` import: add `from src.incontext_dataset_v2 import LoadRequest` at the top of the test file if not already present (it imports `LoadResult` today — extend that line).

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest src/providers/test_multisource.py -v -k "native_emission or dispatch or modality_keys or without_modality"`
Expected: FAIL — `MultiSourceProvider.__init__() got an unexpected keyword argument 'gpu_realize_crop'`.

- [ ] **Step 3: Add the `gpu_realize_crop` ctor flag**

In `src/providers/multisource.py`, `__init__` signature and body:

```python
    def __init__(self, sub_providers, *, context_size, regime_p=(1 / 3, 1 / 3, 1 / 3),
                 epoch_length=1000, gpu_realize_crop=False):
```

After `self.epoch_length = int(epoch_length)`:

```python
        self.gpu_realize_crop = bool(gpu_realize_crop)
```

- [ ] **Step 4: Add modality-dispatching load methods**

In `src/providers/multisource.py`, replace the `load` stub (the `def load(self, *a, **k): raise RuntimeError(...)` block) with:

```python
    def load(self, subject, cls, req, *, modality=None):
        """Dispatch a single-case load to the modality-locked sub-provider.
        Level-0 uses assemble_task; this is the cascade re-crop (level>=1) entry."""
        if modality is None:
            raise RuntimeError("MultiSourceProvider.load needs modality= "
                               "(cohort provider; level-0 goes through assemble_task)")
        return self.subs[modality].load(subject, cls, req)

    def load_native_crop(self, subject, cls, req, *, modality):
        """Native-crop re-crop for the GPU-realize cascade path (level>=1)."""
        return self.subs[modality].load_native_crop(subject, cls, req)
```

- [ ] **Step 5: Emit modality keys + the native branch in `assemble_task`**

In `src/providers/multisource.py`, `assemble_task`, replace everything from `def _load(mod, subj):` to the end of the method with:

```python
        meta = {"regime": regime, "tgt_mod": tgt_mod, "ctx_mod": ctx_mod,
                "fallback": bool(regime == "cross" and tgt_mod == ctx_mod)}

        if self.gpu_realize_crop:
            req = LoadRequest(rng=rng, crop_spacing_mm=float(crop_spacing_mm))
            tgt_nc = self.subs[tgt_mod].load_native_crop(tgt_subj, cls, req)
            ctx_ncs = [self.subs[ctx_mod].load_native_crop(s, cls, req) for s in ctx_subjs]
            return {
                "native_crop": [tgt_nc, *ctx_ncs],
                "subject": tgt_subj,
                "context_subjects": list(ctx_subjs),
                "label_name": cls,
                "tgt_modality": tgt_mod,
                "ctx_modality": ctx_mod,
                "aug_mode": torch.tensor(0, dtype=torch.long),
                "meta": meta,
            }

        def _load(mod, subj):
            return self.subs[mod].load(
                subj, cls, LoadRequest(rng=rng, crop_spacing_mm=float(crop_spacing_mm)))

        tgt = _load(tgt_mod, tgt_subj)
        ctx = [_load(ctx_mod, s) for s in ctx_subjs]

        return {
            "image": tgt.image,
            "label": tgt.label,
            "context_in": torch.stack([r.image for r in ctx]),
            "context_out": torch.stack([r.label for r in ctx]),
            "spacing": tgt.spacing,
            "crop_geom": tgt.crop_geom,
            "subject": tgt_subj,
            "context_subjects": list(ctx_subjs),
            "label_name": cls,
            "modality": tgt_mod,
            "tgt_modality": tgt_mod,
            "ctx_modality": ctx_mod,
            "aug_mode": torch.tensor(0, dtype=torch.long),
            "meta": meta,
        }
```

(The `LoadRequest` import at the top of `multisource.py` already exists.)

- [ ] **Step 6: Run the multisource test file**

Run: `python -m pytest src/providers/test_multisource.py -v`
Expected: PASS — new tests plus all pre-existing ones (`test_item_dict_shape_k2` etc. still hold; the painted dict is a superset now).

- [ ] **Step 7: Commit**

```bash
git add src/providers/multisource.py src/providers/test_multisource.py
git commit -m "feat: MultiSourceProvider modality dispatch + native-crop emission

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_019ZbXsSuph6AU8dMhWhycxf"
```

---

## Task 3: Thread `tgt_modality` / `ctx_modality` through the collates

**Files:**
- Modify: `src/gpu_realize_crop.py` — `native_crop_collate_fn` (~line 109-120)
- Modify: `src/totalseg_dataloader_incontext.py` — `incontext_collate_fn` (~line 1551-1584)
- Modify: `experiments/3d/cascade.py` — `realize_cascade_level0` (~line 442-457)
- Test: `experiments/3d/tests/test_gpu_realize_crop.py`

**Interfaces:**
- Consumes: item dicts from Task 2 carrying `"tgt_modality"` / `"ctx_modality"` (scalars).
- Produces: batch dicts where, **when the items carry them**, `batch["tgt_modality"]` and `batch["ctx_modality"]` are `list[str]` of length B (row-parallel to `batch["subjects"]`). Absent otherwise (single-source items).

- [ ] **Step 1: Write the failing test**

Add to `experiments/3d/tests/test_gpu_realize_crop.py`:

```python
def test_native_collate_passes_modality_when_present():
    from src.gpu_realize_crop import native_crop_collate_fn
    spec = resolve_ct_norm(None)
    img = _smooth_vol(24); lbl = np.zeros((24, 24, 24), np.uint8); lbl[10:14, 10:14, 10:14] = 3
    nc = _native_crop_from(img, lbl, 3, (12, 12, 12), 8, 1.5)
    items = [
        {"native_crop": [nc, nc], "subject": "s0", "context_subjects": ["c0"],
         "label_name": "liver", "tgt_modality": "ct", "ctx_modality": "mri",
         "aug_mode": torch.tensor(0)},
        {"native_crop": [nc, nc], "subject": "s1", "context_subjects": ["c1"],
         "label_name": "liver", "tgt_modality": "mri", "ctx_modality": "mri",
         "aug_mode": torch.tensor(0)},
    ]
    b = native_crop_collate_fn(items)
    assert b["tgt_modality"] == ["ct", "mri"]
    assert b["ctx_modality"] == ["mri", "mri"]


def test_native_collate_omits_modality_when_absent():
    from src.gpu_realize_crop import native_crop_collate_fn
    spec = resolve_ct_norm(None)
    img = _smooth_vol(24); lbl = np.zeros((24, 24, 24), np.uint8); lbl[10:14, 10:14, 10:14] = 3
    nc = _native_crop_from(img, lbl, 3, (12, 12, 12), 8, 1.5)
    items = [{"native_crop": [nc, nc], "subject": "s0", "context_subjects": ["c0"],
              "label_name": "liver", "aug_mode": torch.tensor(0)}]
    b = native_crop_collate_fn(items)
    assert "tgt_modality" not in b and "ctx_modality" not in b
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest experiments/3d/tests/test_gpu_realize_crop.py -v -k "collate_passes_modality or collate_omits_modality"`
Expected: FAIL — `KeyError: 'tgt_modality'`.

- [ ] **Step 3: `native_crop_collate_fn` passthrough**

In `src/gpu_realize_crop.py`, `native_crop_collate_fn`, before `return out`:

```python
    if "tgt_modality" in batch[0]:
        out["tgt_modality"] = [b["tgt_modality"] for b in batch]
        out["ctx_modality"] = [b["ctx_modality"] for b in batch]
    return out
```

- [ ] **Step 4: `incontext_collate_fn` passthrough**

In `src/totalseg_dataloader_incontext.py`, `incontext_collate_fn`, next to the existing `if "modality" in batch[0]:` block:

```python
    if "tgt_modality" in batch[0]:
        out["tgt_modality"] = [b["tgt_modality"] for b in batch]  # (B,) list[str], cascade re-crop routing
        out["ctx_modality"] = [b["ctx_modality"] for b in batch]
```

- [ ] **Step 5: `realize_cascade_level0` passthrough**

In `experiments/3d/cascade.py`, `realize_cascade_level0`, after the `out["label_names"] = ...` line and before `out["aug_mode"] = ...`:

```python
    if "tgt_modality" in batch:
        out["tgt_modality"] = list(batch["tgt_modality"])
        out["ctx_modality"] = list(batch["ctx_modality"])
```

- [ ] **Step 6: Run the realize test file**

Run: `python -m pytest experiments/3d/tests/test_gpu_realize_crop.py -v`
Expected: PASS (new + existing).

- [ ] **Step 7: Commit**

```bash
git add src/gpu_realize_crop.py src/totalseg_dataloader_incontext.py experiments/3d/cascade.py experiments/3d/tests/test_gpu_realize_crop.py
git commit -m "feat: thread tgt/ctx modality through the cascade collates

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_019ZbXsSuph6AU8dMhWhycxf"
```

---

## Task 4: `_recrop_level` routes each re-crop load by modality

**Files:**
- Modify: `experiments/3d/cascade.py` — `_recrop_level` (~line 158-229)
- Test: `experiments/3d/tests/test_recrop_modality.py` (new)

**Interfaces:**
- Consumes: `batch["tgt_modality"]` / `batch["ctx_modality"]` (Task 3), optional; `provider.load_native_crop(subj, cls, req, *, modality)` / `provider.load(subj, cls, req, *, modality)` (Task 2).
- Produces: no signature change to `_recrop_level`. Internally, each entry in the `tasks` list is now a 6-tuple `(b, k, subject, center, rk, mod)` where `mod` is the row's target/context modality string or `None`. `modality=` is forwarded to the provider **only when `mod is not None`**.

- [ ] **Step 1: Write the failing test**

Create `experiments/3d/tests/test_recrop_modality.py`:

```python
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
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest experiments/3d/tests/test_recrop_modality.py -v`
Expected: FAIL — `test_recrop_routes_by_modality` raises `TypeError` (`load_native_crop() missing 1 required keyword-only argument: 'modality'`) because `_recrop_level` does not pass it yet.

- [ ] **Step 3: Tag tasks with modality and route the loads**

In `experiments/3d/cascade.py`, `_recrop_level`. Replace the flat-load-list construction:

```python
    subs, ctxs, clss = batch["subjects"], batch["context_subjects"], batch["label_names"]
    sp = float(spacing)
    tmods = batch.get("tgt_modality")
    cmods = batch.get("ctx_modality")

    # Flat load list: per b, the target (k == -1) then its K contexts, in order.
    # 6th field = the slot's modality ("ct"/"mri") or None for single-source providers.
    tasks = []
    for b in range(len(subs)):
        tmod = tmods[b] if tmods is not None else None
        cmod = cmods[b] if cmods is not None else None
        tasks.append((b, -1, subs[b], centers[b], f"{seed}_{step}_{level}_{b}", tmod))
        for k, cs in enumerate(ctxs[b]):
            tasks.append((b, k, cs, None, f"{seed}_{step}_{level}_{b}_{k}", cmod))
```

Update the `realize_crop` branch's `_load_nc`:

```python
    if realize_crop:
        def _load_nc(t):
            b, _k, subj, center, rk, mod = t
            req = LoadRequest(rng=random.Random(rk), crop_spacing_mm=sp,
                              center=center, jitter=jitter)
            if mod is not None:
                return provider.load_native_crop(subj, clss[b], req, modality=mod)
            return provider.load_native_crop(subj, clss[b], req)
```

Update the CPU branch's `_load`:

```python
    def _load(t):
        b, _k, subj, center, rk, mod = t
        req = LoadRequest(rng=random.Random(rk), crop_spacing_mm=sp,
                          center=center, jitter=jitter)
        if mod is not None:
            return provider.load(subj, clss[b], req, modality=mod)
        return provider.load(subj, clss[b], req)
```

The two downstream unpacks are unaffected: `_regroup(flat, [t[0] for t in tasks], len(subs))` reads `t[0]`, and `for (b, k, *_), r in zip(tasks, results)` star-absorbs the extra field.

- [ ] **Step 4: Run the routing test file**

Run: `python -m pytest experiments/3d/tests/test_recrop_modality.py -v`
Expected: PASS.

- [ ] **Step 5: Run the cascade unit tests that touch `_recrop_level`**

Run: `python -m pytest experiments/3d/tests/ -v -k "recrop or cascade"`
Expected: PASS — no regression in existing cascade tests.

- [ ] **Step 6: Commit**

```bash
git add experiments/3d/cascade.py experiments/3d/tests/test_recrop_modality.py
git commit -m "feat: modality-routed re-crop loads in cascade._recrop_level

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_019ZbXsSuph6AU8dMhWhycxf"
```

---

## Task 5: Wire `common.py` — build path + relax the guard

**Files:**
- Modify: `experiments/3d/common.py` — multisource branch of `build_dataset` (~line 336-380), `_assert_cascade_supported` (~line 233-294)
- Test: `experiments/3d/tests/test_cascade_guard.py`

**Interfaces:**
- Consumes: `MultiSourceProvider(..., gpu_realize_crop=...)` (Task 2); `TotalSegProvider(..., ram_cache=...)` (existing kwarg).
- Produces: a train-split multisource dataset whose provider has `gpu_realize_crop=True` (and both sub-providers `ram_cache=True`) when `data.cascade_spacings` is set and `data.gpu_realize_crop` is not explicitly false; eval-split provider stays painted. `_assert_cascade_supported` no longer raises for `source='multisource'`, and no longer raises for `source='totalsegmri'` + `gpu_realize_crop=True`.

- [ ] **Step 1: Write the failing guard tests**

In `experiments/3d/tests/test_cascade_guard.py`:

Replace `test_rejects_mri_source_with_gpu_realize` with:

```python
def test_allows_mri_source_with_gpu_realize():
    # NativeCrop now carries a per-subject CtNormSpec, so MRI GPU-realize normalizes
    # correctly (2026-09-07 multisource-cascade spec).
    _assert_cascade_supported(_cfg(data={"source": "totalsegmri"}))
```

Add:

```python
def test_allows_multisource_source():
    _assert_cascade_supported(_cfg(data={"source": "multisource"}))


def test_allows_multisource_with_gpu_realize_and_ram_cache():
    _assert_cascade_supported(_cfg(data={"source": "multisource",
                                         "gpu_realize_crop": True, "ram_cache": True}))
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest experiments/3d/tests/test_cascade_guard.py -v -k "multisource or mri_source"`
Expected: FAIL — `test_allows_multisource_source` raises `ValueError: ... not a v2 TotalSeg source`; `test_allows_mri_source_with_gpu_realize` raises `ValueError: ... MRI ...`.

- [ ] **Step 3: Relax `_assert_cascade_supported`**

In `experiments/3d/common.py`, `_assert_cascade_supported`:

Change the source check (currently `if d.get("source", "totalseg") not in _TOTALSEG_SOURCES:`):

```python
    _cascade_sources = _TOTALSEG_SOURCES | {"multisource"}
    if d.get("source", "totalseg") not in _cascade_sources:
        raise ValueError(f"data.cascade_spacings: source {d.get('source')!r} is not a "
                         f"cascade-capable source ({sorted(_cascade_sources)}).")
```

Delete the MRI block entirely:

```python
    if _gr and d.get("source", "totalseg") == "totalsegmri":
        raise ValueError(
            "GPU realize does not yet support per-subject MRI normalization (the NativeCrop "
            "payload carries no modality and realize_native_crops applies the CT fingerprint "
            "unconditionally); set data.gpu_realize_crop=false for MRI cascade runs.")
```

(Leave the `_gr and ram_cache is False` check immediately above it in place.)

- [ ] **Step 4: Run the guard test file**

Run: `python -m pytest experiments/3d/tests/test_cascade_guard.py -v`
Expected: PASS — all, including the two new + the flipped MRI case. `test_allows_mri_source_without_gpu_realize` still passes.

- [ ] **Step 5: Wire the multisource build branch**

In `experiments/3d/common.py`, the `if d.get("source") == "multisource":` branch of `build_dataset`.

Just before the `for src, mod, spec, root in _multisource_specs(cfg, which):` loop, compute the realize/ram-cache flags (mirror the `loader_v2` branch's formula):

```python
        _casc = bool(d.get("cascade_spacings"))
        _realize = is_train and bool(d.get("gpu_realize_crop", _casc))
        _ram = _realize and bool(d.get("ram_cache", False))
```

In the `TotalSegProvider(...)` construction inside the loop, change the last kwarg `ram_cache=False` to:

```python
                modality=mod, ct_norm=d.get("ct_norm"), ram_cache=_ram)
```

In the `MultiSourceProvider(...)` construction, add the flag:

```python
        provider = MultiSourceProvider(
            subs, context_size=d.context_size,
            regime_p=tuple(sm.get("regime_p", (1 / 3, 1 / 3, 1 / 3))),
            epoch_length=((d.get("max_ds_len_train") or 1000) if is_train
                          else int(sm.eval_epoch_length)),
            gpu_realize_crop=_realize)
```

- [ ] **Step 6: Config-resolution smoke (no data)**

Run:

```bash
python -c "
import sys; sys.path.insert(0, 'experiments/3d')
from hydra import initialize, compose
with initialize(version_base=None, config_path='../../configs'):
    cfg = compose(config_name='experiment/3d/config', overrides=[
        'experiment=81_multisource_ct_mri', 'cluster=nfs',
        'data.train_spacing_range=null', 'data.crop_spacing_mm=6',
        '+data.cascade_spacings=[6,3]', '+data.gpu_realize_crop=true',
        '+data.ram_cache=true', '+train.cascade_loss_weights=[1,1]',
        '+data.cascade_query_prior.modes=[pred,none,gt]',
        '+data.cascade_query_prior.p=[0.6,0.3,0.1]',
        '+data.cascade_query_prior.eval_mode=pred'])
from common import _assert_cascade_supported
_assert_cascade_supported(cfg)
print('guard OK; source=', cfg.data.source, 'cascade_spacings=', list(cfg.data.cascade_spacings))
"
```

Expected: prints `guard OK; source= multisource cascade_spacings= [6, 3]`. If the config path / name differs, adjust — the point is `_assert_cascade_supported(cfg)` returns without raising.

- [ ] **Step 7: Commit**

```bash
git add experiments/3d/common.py experiments/3d/tests/test_cascade_guard.py
git commit -m "feat: allow source=multisource in cascade; wire gpu_realize_crop + ram_cache

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_019ZbXsSuph6AU8dMhWhycxf"
```

---

## Task 6: Integration check, smoke run, docs

**Files:**
- Modify: `experiments/3d/_check_multisource.py`
- Modify: `docs/logs.md`
- Test: the extended `_check_multisource.py` + (conditionally) a real training step.

**Interfaces:**
- Consumes: everything from Tasks 1-5.
- Produces: nothing importable — this task validates end to end and documents the command.

- [ ] **Step 1: Extend `_check_multisource.py`**

Read the file first (`experiments/3d/_check_multisource.py`) to match its style (it builds a cfg from `["experiment=81_multisource_ct_mri", "cluster=nfs"]` and asserts `ds.cohort_mode`). Add a function that builds the dataset with the cascade overrides and checks the native path:

```python
def check_cascade_build():
    """Train-split multisource under the cascade overrides: provider emits native crops."""
    cfg = _compose([
        "experiment=81_multisource_ct_mri", "cluster=nfs",
        "data.train_spacing_range=null", "data.crop_spacing_mm=6",
        "+data.cascade_spacings=[6,3]", "+data.gpu_realize_crop=true",
        "+data.ram_cache=true", "+train.cascade_loss_weights=[1,1]",
        "+data.cascade_query_prior.modes=[pred,none,gt]",
        "+data.cascade_query_prior.p=[0.6,0.3,0.1]",
        "+data.cascade_query_prior.eval_mode=pred",
    ])
    from common import _assert_cascade_supported, build_dataset
    _assert_cascade_supported(cfg)
    ds = build_dataset(cfg, "train")
    from src.providers.multisource import MultiSourceProvider
    assert isinstance(ds.provider, MultiSourceProvider)
    assert ds.provider.gpu_realize_crop is True
    item = ds[0]
    assert "image" not in item and "native_crop" in item
    assert item["tgt_modality"] in ("ct", "mri")
    assert item["ctx_modality"] in ("ct", "mri")
    from src.providers.totalseg import NativeCrop
    assert all(isinstance(x, NativeCrop) for x in item["native_crop"])
    assert all(x.norm is not None for x in item["native_crop"])
    print("check_cascade_build OK:", item["tgt_modality"], "->", item["ctx_modality"])
```

Reuse the module's existing compose helper if it has one (it imports `build_dataset, resolve_multisource_classes` from `common`); name the call to match. Wire `check_cascade_build()` into the script's `__main__` alongside the existing checks.

- [ ] **Step 2: Run the integration check (needs NFS data)**

Run: `python experiments/3d/_check_multisource.py`
Expected: the existing checks pass AND `check_cascade_build OK: <mod> -> <mod>`.
If the NFS mount / data is unavailable in this environment, note that in `docs/logs.md` and rely on Tasks 1-5's unit tests + Step 4 below.

- [ ] **Step 3: Check for a usable GPU on this node**

Run: `nvidia-smi --query-gpu=memory.free,memory.total --format=csv,noheader`
- If it errors or shows no free memory → skip Step 4, go to Step 5.
- If a GPU has substantial free memory → do Step 4.

- [ ] **Step 4: Real cascade smoke run (only if Step 3 found a free GPU)**

Run (small caps; adjust `train.checkpoint` path or drop it if the file is not reachable):

```bash
python experiments/3d/train.py experiment=81_multisource_ct_mri cluster=nfs \
  +data.ram_cache=true \
  augmentations.goal_mask.p=0.4 \
  augmentations.goal_mask.ops=[dilate,erode,boundary,sobel] \
  data.source_mix.per_source_train_classes=[all,all] \
  train.batch_size=2 train.lr=1.0e-5 train.warmup_epochs=1 \
  data.train_spacing_range=null \
  data.crop_spacing_mm=6 +data.cascade_spacings=[6,3] \
  +data.cascade_crop_jitter=null +data.gpu_realize_crop=true \
  +data.cascade_query_prior.modes=[pred,none,gt] \
  +data.cascade_query_prior.p=[0.6,0.3,0.1] \
  +data.cascade_query_prior.eval_mode=pred \
  +data.cascade_query_prior_hard=false \
  +train.cascade_loss_weights=[1,1] \
  train.epochs=1 data.max_train_subjects=8 data.max_val_subjects=8 \
  data.max_ds_len_train=16 data.source_mix.eval_epoch_length=16 \
  wandb.name=null wandb.project=null
```

Expected: one epoch completes — a `run_cascade` step (both levels), a `evaluate_cascade` val pass, no exception. Watch for: `RuntimeError` about `modality`, shape mismatch in `realize_native_crops`, or NaN cascade-forward guards.

- [ ] **Step 5: Full unit-test sweep**

Run:

```bash
python -m pytest src/providers/test_multisource.py \
  experiments/3d/tests/test_gpu_realize_crop.py \
  experiments/3d/tests/test_recrop_modality.py \
  experiments/3d/tests/test_cascade_guard.py -v
```

Expected: all PASS.

- [ ] **Step 6: Log to `docs/logs.md`**

Append a dated entry summarizing: `NativeCrop.norm` (per-crop CtNormSpec, unblocks MRI GPU-realize); `MultiSourceProvider` modality dispatch + native emission; `tgt_modality`/`ctx_modality` threaded through collates → `_recrop_level`; `common.py` guard relaxed for `multisource` + MRI. Include the exact runnable command:

```
python experiments/3d/train.py experiment=81_multisource_ct_mri \
  +data.ram_cache=true \
  augmentations.goal_mask.p=0.4 augmentations.goal_mask.ops=[dilate,erode,boundary,sobel] \
  data.source_mix.per_source_train_classes=[all,all] \
  train.checkpoint=/nfs/data/nii/data1/Analysis/camaret___in_context_segmentation/ANALYSIS_20251122/results/patch_icl/3d_train/2026-09-06_82_multisource_ct_mri_train_all_classes/best.pt \
  train.batch_size=2 train.lr=1.0e-5 train.warmup_epochs=1 \
  data.train_spacing_range=null \
  data.crop_spacing_mm=6 +data.cascade_spacings=[6,3] \
  +data.cascade_crop_jitter=null +data.gpu_realize_crop=true \
  +data.cascade_query_prior.modes=[pred,none,gt] +data.cascade_query_prior.p=[0.6,0.3,0.1] \
  +data.cascade_query_prior.eval_mode=pred +data.cascade_query_prior_hard=false \
  +train.cascade_loss_weights=[1,1] wandb.name=83_multisource_cascade
```

Note the `data.train_spacing_range=null` addition (the `multisource_ct_mri.yaml` default conflicts with `cascade_spacings`) and that `ram_cache=true` now allocates **two** RAM caches (CT + MRI sub-providers).

- [ ] **Step 7: Commit**

```bash
git add experiments/3d/_check_multisource.py docs/logs.md
git commit -m "test+docs: multisource cascade integration check + logs

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_019ZbXsSuph6AU8dMhWhycxf"
```

---

## Self-Review

**Spec coverage:**

| Spec section | Task |
|---|---|
| Component 1 — `NativeCrop` self-describes normalization | Task 1 |
| Component 2 — `MultiSourceProvider` load methods + native emission | Task 2 |
| Component 3 — modality threading to the re-crop loop | Task 3 (collates + `realize_cascade_level0`) + Task 4 (`_recrop_level`) |
| Component 4 — `build_dataset`, `_assert_cascade_supported`, RAM cache | Task 5 |
| Component 5 — eval | Task 4's routing + Task 6 Step 4 smoke (eval pass); no code change, per spec |
| Non-goal: cascade sample-table `detail` column | not implemented (explicit non-goal) |
| Testing section | Tasks 1-6 tests; GPU smoke = Task 6 Step 3-4 |
| `docs/logs.md` + runnable command | Task 6 Step 6 |

**Placeholder scan:** No TBD/TODO. Every code step has literal code. `_check_multisource.py` edits (Task 6 Step 1) say "match its style / reuse its compose helper" rather than a verbatim diff because the file's helper names must be read first — the assertion body is given in full.

**Type consistency:**
- `NativeCrop.norm` — added in Task 1, read in Task 1 (`_realize_member`), Task 4 (test fixtures), Task 6 (`x.norm is not None`). Consistent.
- `build_native_crop(..., norm=..., modality=...)` — defined Task 1, called Task 1 (`load_native_crop`, test helper), Task 4 (fixtures). No remaining `ct_spec=` caller after Task 1 Step 9.
- `MultiSourceProvider(..., gpu_realize_crop=bool)` — defined Task 2, used Task 5 build branch, asserted Task 6.
- `provider.load(subj, cls, req, *, modality=None)` / `load_native_crop(subj, cls, req, *, modality)` — defined Task 2, called Task 4 `_recrop_level` (kwarg only when `mod is not None`).
- `batch["tgt_modality"]` / `batch["ctx_modality"]` — `list[str]` length B, produced Task 3 (both collates + `realize_cascade_level0`), consumed Task 4 (`_recrop_level` reads `batch.get(...)`).
- `assemble_task` native dict keys — defined Task 2, consumed by `native_crop_collate_fn` (Task 3) and asserted in Task 6.
