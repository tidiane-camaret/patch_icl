# PatchSet3D Iris Decoder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `arch.decoder="iris"` to `PatchSet3D` — a literal reproduction of Iris's
contextual-stream task encoding (§4.2, Eq. 3–4) and mask-decoding module (§5, Eq. 5–6), as a
third decoder option alongside today's `fine_filter`/`conv`.

**Architecture:** Two new self-contained modules that do not reuse `_build_conv_decoder`'s
FiLM/z-score/token-residual fusion tricks: `_iris_task_encode` (support-only, PixelShuffle-fused
mask + support features → `m` learned tokens `T_c`, independent of the query and of the main
transformer) and `_decode_iris` (bidirectional cross-attention between `T_c` and the query's
pre-transformer image embedding, then a plain skip-connected conv up-path, read out by one
global per-volume filter dot product). The existing main cross-context transformer (`_attn`)
runs completely unchanged and unmodified — this decoder simply doesn't use its output.

**Tech Stack:** PyTorch (`nn.MultiheadAttention`, `nn.Conv3d`, `nn.LayerNorm`), pytest.

**Spec:** `docs/superpowers/specs/2026-09-17-patchset3d-iris-decoder-design.md`

## Global Constraints

- No FiLM, no per-skip `InstanceNorm3d` z-score, no static head-conv, no token-only residual in
  the new decoder path — those are `_build_conv_decoder`'s own adaptations, not what Iris does.
- `_attn` gets **zero code changes** — the iris path reads `sup_feat`/`qry_feat` directly in
  `forward()`, before/parallel to the (unchanged) call to `_attn`.
- `e % iris_pixelshuffle_r**3 == 0` must be asserted at construction, not discovered at first
  forward.
- Every new `arch.*` constructor kwarg must be threaded through
  `experiments/3d/train.py::build_model`'s `arch` dict via `a.get(key, default)` — a kwarg that
  exists only in `PatchSet3D.__init__` is unreachable from any Hydra config (this bit the
  `pool_token` feature's own design doc; don't repeat it).
- Existing experiment numbers 93 and 94 are already taken
  (`93_multisource_synth_texture.yaml`, `94_multisource_synth_heterogeneity.yaml`) — the new
  experiment config is `95_iris_decoder.yaml`.
- Tiny-model tests in `tests/test_patchset3d.py` use `e=32` by convention
  (`resolution=4, enc_dims=(8,8,8), e=32, h=64, l=2, a=2, thinking_rows=2`) — `iris_pixelshuffle_r`
  must be overridden to `2` in tests (`32 % 4**3 != 0`, but `32 % 2**3 == 0`).

---

### Task 1: 3D PixelShuffle / PixelUnshuffle helpers

**Files:**
- Modify: `src/models/patchset3d.py` — insert after `_mask_tiles_3d` (ends line 54), before
  `class MaskConvEmbed` (line 57)
- Test: `tests/test_patchset3d.py` — insert after `test_mask_tiles_3d_shape_and_occupancy`
  (ends line 18), before the mid-file `from src.models.patchset3d import PatchSet3D` (line 19)

**Interfaces:**
- Produces: `_pixel_shuffle_3d(x: torch.Tensor, r: int) -> torch.Tensor` — `(N,C,D,H,W) ->
  (N,C/r^3,D*r,H*r,W*r)`. `_pixel_unshuffle_3d(x: torch.Tensor, r: int) -> torch.Tensor` — exact
  inverse. Both module-level functions in `src/models/patchset3d.py`, used by Task 3's
  `_iris_task_encode`.

- [ ] **Step 1: Write the failing test**

Insert into `tests/test_patchset3d.py` right after line 18 (`assert torch.allclose(tiles[0, 0],
torch.ones(8))`), before line 19's import:

```python
from src.models.patchset3d import _pixel_shuffle_3d, _pixel_unshuffle_3d


def test_pixel_shuffle_3d_roundtrip():
    for C, D, r in [(64, 4, 2), (24, 8, 2), (128, 2, 4)]:
        x = torch.randn(2, C, D, D, D)
        shuffled = _pixel_shuffle_3d(x, r)
        assert shuffled.shape == (2, C // r ** 3, D * r, D * r, D * r)
        back = _pixel_unshuffle_3d(shuffled, r)
        assert torch.allclose(back, x)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/dpxuser/dev/patch_icl && python -m pytest tests/test_patchset3d.py::test_pixel_shuffle_3d_roundtrip -v`
Expected: FAIL with `ImportError: cannot import name '_pixel_shuffle_3d'`

- [ ] **Step 3: Write minimal implementation**

Insert into `src/models/patchset3d.py` right after line 54 (end of `_mask_tiles_3d`), before
line 57 (`class MaskConvEmbed`):

```python
def _pixel_shuffle_3d(x: torch.Tensor, r: int) -> torch.Tensor:
    """(N,C,D,H,W) -> (N,C/r^3,D*r,H*r,W*r), C % r^3 == 0. 3D analog of nn.PixelShuffle
    (channel-to-space rearrangement, no interpolation/conv) -- Iris Eq 3."""
    N, C, D, H, W = x.shape
    Co = C // (r ** 3)
    x = x.reshape(N, Co, r, r, r, D, H, W)
    x = x.permute(0, 1, 5, 2, 6, 3, 7, 4)
    return x.reshape(N, Co, D * r, H * r, W * r).contiguous()


def _pixel_unshuffle_3d(x: torch.Tensor, r: int) -> torch.Tensor:
    """(N,C,D,H,W) -> (N,C*r^3,D/r,H/r,W/r) -- exact inverse of _pixel_shuffle_3d -- Iris Eq 4."""
    N, C, D, H, W = x.shape
    Dd, Hh, Ww = D // r, H // r, W // r
    x = x.reshape(N, C, Dd, r, Hh, r, Ww, r)
    x = x.permute(0, 1, 3, 5, 7, 2, 4, 6)
    return x.reshape(N, C * (r ** 3), Dd, Hh, Ww).contiguous()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/dpxuser/dev/patch_icl && python -m pytest tests/test_patchset3d.py::test_pixel_shuffle_3d_roundtrip -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
export PATH="/software/anaconda3/envs/git/bin:$PATH"    # git is not on the default PATH on this node
cd /home/dpxuser/dev/patch_icl
git add src/models/patchset3d.py tests/test_patchset3d.py
git commit -m "feat(patchset3d): add 3D PixelShuffle/PixelUnshuffle helpers (Iris Eq 3-4)"
```

---

### Task 2: `_build_iris_decoder` constructor wiring

**Files:**
- Modify: `src/models/patchset3d.py:229` (constructor signature), `src/models/patchset3d.py:519-522`
  (decoder dispatch), insert new method after line 580 (after `_build_conv_decoder`, before line
  582's `@property def grid_size`)
- Test: `tests/test_patchset3d.py`

**Interfaces:**
- Consumes: `_ConvNormAct` (already defined, line 155) for the skip-fusion conv blocks;
  `self.encoder.fine_stage_size`/`fine_stage_channels` (already exist on every fine-capable
  encoder).
- Produces: constructor kwargs `iris_pixelshuffle_r: int = 4, iris_m: int = 10,
  iris_ctx_layers: int = 2`; instance attrs `self.iris_r`, `self.iris_ctx_conv`,
  `self.iris_ctx_query` (`(m,e)` param), `self.iris_ctx_cross`/`self.iris_ctx_self` (each an
  `nn.ModuleList` of `nn.MultiheadAttention`), `self.iris_ctx_mlp` (`nn.ModuleList` of MLPs),
  `self.iris_ctx_norms` (`nn.ModuleList` of 3-`nn.LayerNorm` lists), `self.iris_t2f`/
  `self.iris_f2t` (`nn.MultiheadAttention`), `self._iris_stage_order`/`self._iris_sides` (lists),
  `self.iris_token_proj` (`nn.Linear`), `self.iris_blocks` (`nn.ModuleList` of
  `nn.Sequential(_ConvNormAct, _ConvNormAct)`), `self.iris_class_embed` (`nn.Linear`). Used by
  Task 3 (`_iris_task_encode`) and Task 4 (`_decode_iris`).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_patchset3d.py` (end of file, after the existing `pool_token` test block):

```python
# --- arch.decoder="iris": literal Iris §4.2+§5 task-encoding/decoding modules ---

_IRIS_KW = dict(image_size=(16, 16, 16), fine_decode=True, fine_stage=1, decoder="iris",
               iris_pixelshuffle_r=2, iris_m=4, iris_ctx_layers=1)


def test_iris_decoder_builds_expected_modules():
    m = PatchSet3D(resolution=4, enc_dims=(8, 8, 8), e=32, h=64, l=2, a=2, thinking_rows=2,
                   **_IRIS_KW)
    assert m.decoder_kind == "iris"
    assert m.iris_ctx_query.shape == (4, 32)
    assert m.iris_class_embed.out_features == m.iris_blocks[-1][-1].conv.out_channels


def test_iris_decoder_rejects_bad_pixelshuffle_r():
    try:
        PatchSet3D(resolution=4, enc_dims=(8, 8, 8), e=32, h=64, l=2, a=2, thinking_rows=2,
                  image_size=(16, 16, 16), fine_decode=True, fine_stage=1, decoder="iris",
                  iris_pixelshuffle_r=3)          # 32 % 27 != 0
        assert False, "should have raised"
    except AssertionError as exc:
        assert "iris_pixelshuffle_r" in str(exc)


def test_decoder_invalid_raises_mentions_iris():
    try:
        PatchSet3D(resolution=4, enc_dims=(8, 8, 8), e=32, h=64, l=2, a=2, thinking_rows=2,
                  image_size=(16, 16, 16), fine_decode=True, fine_stage=1, decoder="bogus")
        assert False, "should have raised"
    except ValueError as exc:
        assert "iris" in str(exc)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/dpxuser/dev/patch_icl && python -m pytest tests/test_patchset3d.py -k iris_decoder_builds -v`
Expected: FAIL with `TypeError: __init__() got an unexpected keyword argument 'decoder'` is
already accepted today, but `'iris_pixelshuffle_r'`/`'iris_m'`/`'iris_ctx_layers'` are not —
expect `TypeError: __init__() got an unexpected keyword argument 'iris_pixelshuffle_r'`.

- [ ] **Step 3: Write minimal implementation**

3a. In `src/models/patchset3d.py`, at line 229 (`decoder_dim: int = 64,`), insert three new
constructor kwargs immediately after it (before `pool_token: bool = False,` at line 230):

```python
        decoder_dim: int = 64,
        iris_pixelshuffle_r: int = 4,
        iris_m: int = 10,
        iris_ctx_layers: int = 2,
        pool_token: bool = False,
```

3b. Replace lines 519-522 (the `elif self.decoder_kind == "conv": ... else: raise ValueError(...)`
block):

```python
            elif self.decoder_kind == "conv":
                self._build_conv_decoder(e, int(image_size[0]), resolution, int(decoder_dim))
            else:
                raise ValueError(f"arch.decoder {self.decoder_kind!r} (fine_filter | conv)")
```

with:

```python
            elif self.decoder_kind == "conv":
                self._build_conv_decoder(e, int(image_size[0]), resolution, int(decoder_dim))
            elif self.decoder_kind == "iris":
                self._build_iris_decoder(e, int(image_size[0]), resolution, int(decoder_dim), a,
                                         int(iris_pixelshuffle_r), int(iris_m),
                                         int(iris_ctx_layers))
            else:
                raise ValueError(f"arch.decoder {self.decoder_kind!r} (fine_filter | conv | iris)")
```

3c. Insert this new method after line 580 (end of `_build_conv_decoder`), before line 582
(`@property\n    def grid_size`):

```python
    def _build_iris_decoder(self, e: int, in_size: int, resolution: int, c_d: int, a: int,
                            r: int, m: int, ctx_layers: int):
        """arch.decoder=iris: literal Iris task-encoding (Eq 3-4, §4.2) + mask-decoding (Eq 5-6,
        §5) modules -- see docs/superpowers/specs/2026-09-17-patchset3d-iris-decoder-design.md.
        No FiLM/z-score/token-residual fusion tricks (those are _build_conv_decoder's own
        adaptations, not what Iris does)."""
        assert e % (r ** 3) == 0, (
            f"arch.iris_pixelshuffle_r={r} requires e % r^3 == 0 (e={e}, r^3={r ** 3})")
        self.iris_r = r
        c_shuf = e // (r ** 3)
        self.iris_ctx_conv = nn.Conv3d(c_shuf + 1, c_shuf, 1)          # +1 = concatenated mask
        self.iris_ctx_query = nn.Parameter(torch.empty(m, e))
        nn.init.normal_(self.iris_ctx_query, std=0.02)
        self.iris_ctx_cross = nn.ModuleList(
            [nn.MultiheadAttention(e, a, batch_first=True) for _ in range(ctx_layers)])
        self.iris_ctx_self = nn.ModuleList(
            [nn.MultiheadAttention(e, a, batch_first=True) for _ in range(ctx_layers)])
        self.iris_ctx_mlp = nn.ModuleList(
            [nn.Sequential(nn.Linear(e, 4 * e), nn.GELU(), nn.Linear(4 * e, e))
             for _ in range(ctx_layers)])
        self.iris_ctx_norms = nn.ModuleList(
            [nn.ModuleList([nn.LayerNorm(e) for _ in range(3)]) for _ in range(ctx_layers)])

        self.iris_t2f = nn.MultiheadAttention(e, a, batch_first=True)
        self.iris_f2t = nn.MultiheadAttention(e, a, batch_first=True)
        stages = sorted(self.fine_stage,
                        key=lambda st: self.encoder.fine_stage_size(in_size, st))
        self._iris_stage_order = [self.fine_stage.index(st) for st in stages]
        self._iris_sides = [self.encoder.fine_stage_size(in_size, st) for st in stages]
        chans = [self.encoder.fine_stage_channels(st) for st in stages]
        dims = [max(c_d // (2 ** i), 8) for i in range(len(stages))]
        self.iris_token_proj = nn.Linear(e, c_d)
        self.iris_blocks = nn.ModuleList()
        prev = c_d
        for i in range(len(stages)):
            self.iris_blocks.append(nn.Sequential(_ConvNormAct(prev + chans[i], dims[i]),
                                                  _ConvNormAct(dims[i], dims[i])))
            prev = dims[i]
        self.iris_class_embed = nn.Linear(e, prev)      # C_m = final taper width
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/dpxuser/dev/patch_icl && python -m pytest tests/test_patchset3d.py -k "iris_decoder_builds or iris_decoder_rejects_bad or decoder_invalid_raises_mentions_iris" -v`
Expected: 3 PASS

- [ ] **Step 5: Commit**

```bash
export PATH="/software/anaconda3/envs/git/bin:$PATH"
cd /home/dpxuser/dev/patch_icl
git add src/models/patchset3d.py tests/test_patchset3d.py
git commit -m "feat(patchset3d): construct arch.decoder=iris's task-encoding/decoding modules"
```

---

### Task 3: `_iris_task_encode` (Eq. 3–4, support-only)

**Files:**
- Modify: `src/models/patchset3d.py` — insert new method after `_decode_conv` (ends line 956),
  before `def forward` (line 958)
- Test: `tests/test_patchset3d.py`

**Interfaces:**
- Consumes: `self.img_embed` (existing `nn.Linear`/MLP, line 372), `self.iris_r`,
  `self.iris_ctx_conv`, `self.iris_ctx_query`, `self.iris_ctx_cross`/`_self`/`_mlp`/`_norms`
  (Task 2), `_pixel_shuffle_3d`/`_pixel_unshuffle_3d` (Task 1).
- Produces: `_iris_task_encode(self, sup_feat, context_out, B, K) -> torch.Tensor` — `sup_feat:
  (B,K*N,Cf)` (raw, pre-`img_embed` encoder grid tokens, `self._grid_tokens`'s support output),
  `context_out: (B,K,D,H,W)` (support GT masks) → returns `T_c: (B, iris_m, e)`. Consumed by
  Task 5's `forward()`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_patchset3d.py`:

```python
def test_iris_task_encode_shape_and_backward():
    m = PatchSet3D(resolution=4, enc_dims=(8, 8, 8), e=32, h=64, l=2, a=2, thinking_rows=2,
                   **_IRIS_KW)
    B, K, Cf = 2, 3, m.encoder.out_ch
    sup_feat = torch.randn(B, K * m.N, Cf, requires_grad=True)
    context_out = (torch.rand(B, K, 16, 16, 16) > 0.5).float()
    T_c = m._iris_task_encode(sup_feat, context_out, B, K)
    assert T_c.shape == (B, 4, 32)          # (B, iris_m, e)
    T_c.mean().backward()
    assert sup_feat.grad is not None
    missing = [n for n, p in m.named_parameters()
              if p.requires_grad and p.grad is None and n.startswith("iris_ctx")]
    assert not missing, f"no grad reached: {missing}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/dpxuser/dev/patch_icl && python -m pytest tests/test_patchset3d.py::test_iris_task_encode_shape_and_backward -v`
Expected: FAIL with `AttributeError: 'PatchSet3D' object has no attribute '_iris_task_encode'`

- [ ] **Step 3: Write minimal implementation**

Insert into `src/models/patchset3d.py` after line 956 (end of `_decode_conv`), before line 958
(`def forward`):

```python
    def _iris_task_encode(self, sup_feat, context_out, B, K):
        """Iris §4.2 contextual stream (Eq 3-4), support-only, independent of the query and the
        main transformer. sup_feat: (B,K*N,Cf) raw (pre-img_embed) encoder grid tokens.
        context_out: (B,K,D,H,W) support GT masks. Returns T_c: (B, iris_m, e)."""
        R = self.resolution
        F_s = self.img_embed(sup_feat).reshape(B * K, R, R, R, -1).permute(0, 4, 1, 2, 3)
        r, side = self.iris_r, R * self.iris_r
        y_s = F.interpolate(context_out.reshape(B * K, 1, *context_out.shape[-3:]).float(),
                            size=(side, side, side), mode="trilinear", align_corners=False)
        shuffled = _pixel_shuffle_3d(F_s.contiguous(), r)
        fused = self.iris_ctx_conv(torch.cat([shuffled, y_s.to(shuffled.dtype)], dim=1))
        Fhat_s = _pixel_unshuffle_3d(fused, r)
        kv = Fhat_s.flatten(2).transpose(1, 2).reshape(B, K * R ** 3, -1)
        q = self.iris_ctx_query.unsqueeze(0).expand(B, -1, -1)
        for cross, selfattn, mlp, norms in zip(self.iris_ctx_cross, self.iris_ctx_self,
                                               self.iris_ctx_mlp, self.iris_ctx_norms):
            n1, n2, n3 = norms
            q = q + cross(n1(q), kv, kv)[0]
            q = q + selfattn(n2(q), n2(q), n2(q))[0]
            q = q + mlp(n3(q))
        return q
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/dpxuser/dev/patch_icl && python -m pytest tests/test_patchset3d.py::test_iris_task_encode_shape_and_backward -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
export PATH="/software/anaconda3/envs/git/bin:$PATH"
cd /home/dpxuser/dev/patch_icl
git add src/models/patchset3d.py tests/test_patchset3d.py
git commit -m "feat(patchset3d): add _iris_task_encode (Iris Eq 3-4 contextual stream)"
```

---

### Task 4: `_decode_iris` (Eq. 5–6)

**Files:**
- Modify: `src/models/patchset3d.py` — insert new method immediately after `_iris_task_encode`
  (Task 3), before `def forward`
- Test: `tests/test_patchset3d.py`

**Interfaces:**
- Consumes: `self.iris_t2f`/`self.iris_f2t`/`self.iris_token_proj`/`self.iris_blocks`/
  `self.iris_class_embed`/`self._iris_sides`/`self._iris_stage_order` (Task 2), `self.grid_size`
  (existing property, line 583).
- Produces: `_decode_iris(self, F_q, T, fine) -> torch.Tensor` — `F_q: (B,N,e)` (query's
  pre-transformer image embedding), `T: (B,iris_m,e)` (Task 3's `T_c`), `fine`: tuple of
  query-only unpooled encoder stage maps in `self.fine_stage` order → returns `(B,1,G,G,G)`
  logits, `G = self.grid_size`. Consumed by Task 5's `forward()`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_patchset3d.py`:

```python
def test_decode_iris_shape_and_backward():
    m = PatchSet3D(resolution=4, enc_dims=(8, 8, 8), e=32, h=64, l=2, a=2, thinking_rows=2,
                   **_IRIS_KW)
    B, N, e = 2, m.N, 32
    F_q = torch.randn(B, N, e, requires_grad=True)
    T = torch.randn(B, 4, e, requires_grad=True)
    fine = (torch.randn(B, 8, 8, 8, 8),)      # (B,Cf=8,S=8,...) matches fine_stage=1, enc_dims=(8,8,8)
    logit = m._decode_iris(F_q, T, fine)
    assert logit.shape == (B, 1, m.grid_size, m.grid_size, m.grid_size)
    logit.mean().backward()
    assert F_q.grad is not None and T.grad is not None
    missing = [n for n, p in m.named_parameters()
              if p.requires_grad and p.grad is None
              and n.startswith(("iris_t2f", "iris_f2t", "iris_token_proj", "iris_blocks",
                                "iris_class_embed"))]
    assert not missing, f"no grad reached: {missing}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/dpxuser/dev/patch_icl && python -m pytest tests/test_patchset3d.py::test_decode_iris_shape_and_backward -v`
Expected: FAIL with `AttributeError: 'PatchSet3D' object has no attribute '_decode_iris'`

- [ ] **Step 3: Write minimal implementation**

Insert into `src/models/patchset3d.py` immediately after Task 3's `_iris_task_encode` method:

```python
    def _decode_iris(self, F_q, T, fine):
        """Iris §5 mask decoding module (Eq 5-6), literal reproduction -- see
        docs/superpowers/specs/2026-09-17-patchset3d-iris-decoder-design.md. F_q: (B,N,e)
        query's PRE-transformer image embedding. T: (B,iris_m,e) = _iris_task_encode's T_c.
        fine: query-only unpooled encoder stage maps, self.fine_stage order."""
        t2, _ = self.iris_t2f(T, F_q, F_q)          # tokens attend image
        T2 = T + t2
        f2, _ = self.iris_f2t(F_q, T2, T2)          # image attends updated tokens
        Fq2 = F_q + f2

        B = Fq2.shape[0]
        R = self.resolution
        x = self.iris_token_proj(Fq2).transpose(1, 2).reshape(B, -1, R, R, R)
        for i, block in enumerate(self.iris_blocks):
            s = self._iris_sides[i]
            x = F.interpolate(x, size=(s, s, s), mode="trilinear", align_corners=False)
            skip = fine[self._iris_stage_order[i]]
            x = block(torch.cat([x, skip], dim=1))
        mask_features = x

        class_embed = self.iris_class_embed(T2.mean(dim=1))
        logit = torch.einsum('bc,bcdhw->bdhw', class_embed, mask_features).unsqueeze(1)
        g = self.grid_size
        if logit.shape[-1] != g:
            logit = F.interpolate(logit, size=(g, g, g), mode="trilinear", align_corners=False)
        return logit
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/dpxuser/dev/patch_icl && python -m pytest tests/test_patchset3d.py::test_decode_iris_shape_and_backward -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
export PATH="/software/anaconda3/envs/git/bin:$PATH"
cd /home/dpxuser/dev/patch_icl
git add src/models/patchset3d.py tests/test_patchset3d.py
git commit -m "feat(patchset3d): add _decode_iris (Iris Eq 5-6 mask decoding module)"
```

---

### Task 5: Wire `forward()` + end-to-end tests

**Files:**
- Modify: `src/models/patchset3d.py:997-1004` (end of `forward`)
- Test: `tests/test_patchset3d.py`

**Interfaces:**
- Consumes: `self._iris_task_encode` (Task 3), `self._decode_iris` (Task 4), `self.img_embed`,
  existing `forward()` locals `sup_feat`/`qry_feat`/`context_out`/`B`/`K`/`fine`.
- Produces: `forward()`'s `final_logit` now correctly routes through the iris modules when
  `self.decoder_kind == "iris"`; no change to the returned dict's keys/shapes contract.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_patchset3d.py`:

```python
def test_iris_decoder_end_to_end_shape_and_backward():
    m = PatchSet3D(resolution=4, enc_dims=(8, 8, 8), e=32, h=64, l=2, a=2, thinking_rows=2,
                   **_IRIS_KW)
    img, cin, cout = _dummy_batch(B=2, K=3, S=16)
    out = m(img, context_in=cin, context_out=cout, mode="train")
    assert out["final_logit"].shape == (2, 1, 4, 4, 4)
    out["final_logit"].mean().backward()
    missing = [n for n, p in m.named_parameters() if p.requires_grad and p.grad is None]
    assert not missing, f"no grad reached: {missing}"


def test_iris_decoder_noop_when_not_selected():
    """decoder != 'iris' (existing configs): forward() takes the original branch, no iris_*
    attributes exist, output shape/behavior unchanged."""
    m = PatchSet3D(resolution=4, enc_dims=(8, 8, 8), e=32, h=64, l=2, a=2, thinking_rows=2,
                   image_size=(16, 16, 16), fine_decode=True, fine_stage=1, decoder="conv")
    assert not hasattr(m, "iris_t2f") and not hasattr(m, "iris_ctx_query")
    img, cin, cout = _dummy_batch(B=2, K=2, S=16)
    out = m(img, context_in=cin, context_out=cout, mode="train")
    assert out["final_logit"].shape == (2, 1, 4, 4, 4)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/dpxuser/dev/patch_icl && python -m pytest tests/test_patchset3d.py::test_iris_decoder_end_to_end_shape_and_backward -v`
Expected: FAIL with `AttributeError: 'PatchSet3D' object has no attribute 'fine_proj'` (forward
currently falls through to `_decode`'s `fine_filter` body for any unrecognized `decoder_kind`,
which was never built for `decoder="iris"`)

- [ ] **Step 3: Write minimal implementation**

In `src/models/patchset3d.py`, replace lines 997-1004 (the tail of `forward`, from
`sup_feat, qry_feat = self._grid_tokens(...)` through the final `return`):

```python
        sup_feat, qry_feat = self._grid_tokens(feat_map, B, T, K)
        q, mask_support, mask_query, regs = self._attn(
            sup_feat, qry_feat, self._occupancy(context_out), K, spacing=spacing,
            query_prior=query_prior, cascade_regs=cascade_regs, pool_feat=pool_feat)
        logit = self._decode(q, fine)
        return {"final_logit": logit, "mask_support": mask_support, "mask_query": mask_query,
               "registers": regs}
```

with:

```python
        sup_feat, qry_feat = self._grid_tokens(feat_map, B, T, K)
        q, mask_support, mask_query, regs = self._attn(
            sup_feat, qry_feat, self._occupancy(context_out), K, spacing=spacing,
            query_prior=query_prior, cascade_regs=cascade_regs, pool_feat=pool_feat)
        if self.decoder_kind == "iris":
            # Eq 3-4 task encoding (support-only) + Eq 5-6 decoding: independent of `q`/the main
            # transformer above, which still runs unchanged for parity (see design spec).
            T_c = self._iris_task_encode(sup_feat, context_out, B, K)
            qry_img_pre = self.img_embed(qry_feat)
            logit = self._decode_iris(qry_img_pre, T_c, fine)
        else:
            logit = self._decode(q, fine)
        return {"final_logit": logit, "mask_support": mask_support, "mask_query": mask_query,
               "registers": regs}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/dpxuser/dev/patch_icl && python -m pytest tests/test_patchset3d.py -v`
Expected: **ALL** tests in the file PASS (this also verifies the no-op guarantee for every
existing `decoder`/`fine_decode`/`pool_token`/`seq_compress`/`cascade_registers` combination
already covered by the file's existing test suite).

- [ ] **Step 5: Commit**

```bash
export PATH="/software/anaconda3/envs/git/bin:$PATH"
cd /home/dpxuser/dev/patch_icl
git add src/models/patchset3d.py tests/test_patchset3d.py
git commit -m "feat(patchset3d): wire arch.decoder=iris into forward()"
```

---

### Task 6: Config wiring + experiment file + docs

**Files:**
- Modify: `experiments/3d/train.py:437` (`build_model`'s `arch` dict)
- Modify: `configs/experiment/3d/model/m2_patchset_decoder.yaml:47-48` (decoder comment + new
  knobs)
- Create: `configs/experiment/3d/experiment/95_iris_decoder.yaml`
- Modify: `docs/logs.md` (append entry)

**Interfaces:**
- Consumes: Task 2's constructor kwargs `iris_pixelshuffle_r`/`iris_m`/`iris_ctx_layers`.
- Produces: `arch.decoder=iris` becomes reachable from a Hydra config/CLI override; a runnable
  experiment file for a direct A/B against `92_multisource_synth`'s `conv` decoder.

- [ ] **Step 1: Write the failing test**

This task is config/plumbing, not new model code — verified by a Hydra compose check rather than
a pytest unit test (no existing test file covers `build_model`/Hydra compose; introducing one
would be disproportionate to this task per the repo's "tests only when necessary" guideline).
Confirm the current failure state first:

`experiments/3d`'s directory name isn't a valid Python package segment (leading digit), so this
codebase's own convention (see `tests/experiments_common_shim.py`,
`tests/test_grid_metrics_3d.py`) is `sys.path.insert(0, "experiments/3d")` then a plain `from
train import build_model` — follow that here too, not a dotted `experiments.3d.train` import
(which would raise `SyntaxError`):

Run: `cd /home/dpxuser/dev/patch_icl && python -c "
import sys; sys.path.insert(0, 'experiments/3d')
from omegaconf import OmegaConf
from train import build_model
cfg = OmegaConf.create({'model': 'patchset3d', 'data': {'image_size': [16,16,16]},
                        'arch': {'resolution': 4, 'enc_dims': [8,8,8], 'e': 32, 'h': 64,
                                 'l': 2, 'a': 2, 'thinking_rows': 2, 'residual_decay': 0.95,
                                 'fine_decode': True, 'fine_stage': 1, 'decoder': 'iris',
                                 'iris_pixelshuffle_r': 2}})
build_model(cfg)
"`
Expected: FAIL with `TypeError: __init__() got an unexpected keyword argument` is NOT what
happens — `build_model` silently drops `iris_pixelshuffle_r` (not in its hard-coded key list) and
constructs with the Python default `iris_pixelshuffle_r=4`, which then fails inside
`PatchSet3D.__init__`'s assert: `AssertionError: arch.iris_pixelshuffle_r=4 requires e % r^3 ==
0 (e=32, r^3=64)` — i.e. the override from the config is silently ignored. This is the bug this
task fixes.

- [ ] **Step 2: (already run above)**

- [ ] **Step 3: Write minimal implementation**

3a. In `experiments/3d/train.py`, at line 437 (`"decoder_dim": a.get("decoder_dim", 64),`),
insert three new lines immediately after it, before line 438's closing `}`:

```python
            "decoder_dim": a.get("decoder_dim", 64),
            "iris_pixelshuffle_r": a.get("iris_pixelshuffle_r", 4),
            "iris_m": a.get("iris_m", 10),
            "iris_ctx_layers": a.get("iris_ctx_layers", 2),
        }
```

3b. In `configs/experiment/3d/model/m2_patchset_decoder.yaml`, after line 48
(`decoder_dim: 64               # base width of the conv decoder (halves per level, >=8)`),
insert:

```yaml
  # iris = literal Iris (docs/methods/iris.md §4.2+§5) task-encoding + decoding: PixelShuffle
  # -fused support-only contextual-stream task tokens (Eq 3-4) cross-attend the query's
  # pre-transformer image embedding (Eq 5), then a plain skip-connected conv up-path + one
  # global per-volume filter dot product (Eq 6) -- no FiLM/z-score/token-residual tricks.
  # Bypasses `q` (the main transformer's query-row output) entirely for this decoder's logits.
  # See docs/superpowers/specs/2026-09-17-patchset3d-iris-decoder-design.md.
  iris_pixelshuffle_r: 4        # PixelShuffle factor for Eq 3-4's mask fusion; e % r^3 must == 0
  iris_m: 10                    # learned contextual-stream query tokens (Iris's m, paper default)
  iris_ctx_layers: 2            # {cross-attn,self-attn,MLP} blocks building T_c (paper's default)
```

3c. Create `configs/experiment/3d/experiment/95_iris_decoder.yaml`:

```yaml
# @package _global_
# 95_iris_decoder — 92_multisource_synth with the literal Iris task-encoding + decoding modules
# (arch.decoder=iris, docs/methods/iris.md §4.2 + §5) swapped in for the progressive conv decoder.
#
#   python experiments/3d/train.py experiment=95_iris_decoder \
#     train.checkpoint=<a 92-line best.pt> train.resume_weights_only=true

defaults:
  - 92_multisource_synth
  - _self_

arch:
  decoder: iris

wandb:
  name: 95_iris_decoder
```

3d. Append to `docs/logs.md`:

```markdown
**arch.decoder=iris** (`src/models/patchset3d.py`): literal reproduction of Iris's contextual
-stream task encoding (§4.2, Eq 3-4 — PixelShuffle-fused support features+mask -> `m=10` learned
tokens `T_c`, support-only, independent of the query and of the main transformer) and mask
-decoding module (§5, Eq 5-6 — bidirectional cross-attention between `T_c` and the query's
pre-transformer image embedding, then a plain skip-connected conv up-path read out by one global
per-volume filter dot product). Deliberately does not reuse `_build_conv_decoder`'s
FiLM/z-score/token-residual fusion tricks. New constructor kwargs `iris_pixelshuffle_r`
(default 4, needs `e % r^3 == 0`), `iris_m` (default 10), `iris_ctx_layers` (default 2), threaded
through `train.py::build_model`. New experiment `95_iris_decoder.yaml` (92 + `arch.decoder:
iris`) for a direct A/B. Design: `docs/superpowers/specs/2026-09-17-patchset3d-iris-decoder
-design.md`. TDD throughout (PixelShuffle round-trip, construction/shape/gradient tests per new
method, end-to-end forward/backward, no-op guarantee for every other `decoder` value) — **not
yet validated**, no eval run against 95 has happened.
```

- [ ] **Step 4: Run test to verify it passes**

Re-run the Step 1 command with `'e': 32` unchanged but add `'iris_pixelshuffle_r': 2` already
present in the override (it is, above) — this time it must succeed:

Run: `cd /home/dpxuser/dev/patch_icl && python -c "
import sys; sys.path.insert(0, 'experiments/3d')
from omegaconf import OmegaConf
from train import build_model
cfg = OmegaConf.create({'model': 'patchset3d', 'data': {'image_size': [16,16,16]},
                        'arch': {'resolution': 4, 'enc_dims': [8,8,8], 'e': 32, 'h': 64,
                                 'l': 2, 'a': 2, 'thinking_rows': 2, 'residual_decay': 0.95,
                                 'fine_decode': True, 'fine_stage': 1, 'decoder': 'iris',
                                 'iris_pixelshuffle_r': 2}})
model, name = build_model(cfg)
assert model.decoder_kind == 'iris' and model.iris_r == 2
print('OK')
"`
Expected: prints `OK`

Also verify the new experiment file composes cleanly under Hydra. `train.py`'s own
`@hydra.main` decorator (line 1008) uses `config_path="../../configs/experiment/3d"` (relative
to `experiments/3d/train.py`'s own directory) and `config_name="train"`, `version_base="1.3"` —
use the absolute form of that same directory with `initialize_config_dir` so it resolves
identically regardless of the shell's cwd:

Run: `cd /home/dpxuser/dev/patch_icl && python -c "
from hydra import compose, initialize_config_dir
with initialize_config_dir(
        config_dir='/home/dpxuser/dev/patch_icl/configs/experiment/3d', version_base='1.3'):
    cfg = compose(config_name='train', overrides=['experiment=95_iris_decoder'])
    assert cfg.arch.decoder == 'iris'
    print('OK')
"`
Expected: prints `OK`

- [ ] **Step 5: Commit**

```bash
export PATH="/software/anaconda3/envs/git/bin:$PATH"
cd /home/dpxuser/dev/patch_icl
git add experiments/3d/train.py configs/experiment/3d/model/m2_patchset_decoder.yaml \
       configs/experiment/3d/experiment/95_iris_decoder.yaml docs/logs.md
git commit -m "feat(patchset3d): thread arch.decoder=iris through Hydra config + train.py"
```
