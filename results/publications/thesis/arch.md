# PatchSet3D — Architecture Notation Reference (thesis copy)

Forked from `results/publications/article/arch.md` on 2026-09-23 so the
thesis can diverge independently (more room for extra ablations/analysis
than the CVPR paper). Re-sync manually if the shared architecture changes
in ways that matter to both documents.

Working reference for keeping notation consistent across
`ics_3d_medical/sections/methodology_*.tex`, the TikZ figures
(`ics_3d_medical/imgs/method/arch_patchset.tex`), and the actual
implementation (`src/models/patchset3d.py`, `src/models/pfn_seg_2d.py`,
`experiments/3d/cascade.py`). Values tagged **[ref-config]** are the
concrete numbers of the checkpoint currently used for every eval-table row
(`experiment=92_multisource_synth`, model `patchset3d`, see
`docs/datasets/eval_expansion_status.md`) — they instantiate the notation,
they are not part of the notation itself.

Where the implementation has simplified or diverged from what the
methodology chapter currently states, this doc says so explicitly (flagged
**[divergence]**) rather than silently picking one — those are exactly the
spots that need a decision before the thesis text is finalized.

---

## 1. Notation

| Symbol | Meaning | [ref-config] value |
|---|---|---|
| $I^t$ | target volume | — |
| $\{(I^c_k, L^c_k)\}_{k=1}^K$ | context (image, mask) pairs | $K=1$ |
| $K$ | context-set size | 1 |
| $T$ | volumes per task, $T=K+1$ | 2 |
| $D,H,W$ | crop grid side (isotropic here: $D=H=W$) | 128 |
| $\mathrm{Enc}_\theta$ | shared image encoder (weights shared across all $T$ volumes) | from-scratch PlainConvUNet |
| $R$ | token-grid resolution per axis | 16 |
| $N$ | cells per volume, $N=R^3$ | 4096 |
| $C_f$ | encoder channel width at the tapped coarse stage(s) | 768 |
| $e$ | transformer token width | 768 |
| $a$ | attention heads | 12 |
| $d=e/a$ | head dim | 64 |
| $h$ | MLP hidden width | 3072 |
| $l$ | transformer layers | 4 |
| $c$ | slot-axis width: image slot + label slot | 2 |
| $p$ | mask occupancy tile side (voxels/cell edge) | 8 |
| $n_t$ | thinking/register rows | 8 |
| $\mathbf z^{\text{img}}, \mathbf z^{\text{lbl}}$ | image-slot / label-slot token streams | — |
| $r$ | row index into the assembled sequence ($n_t$ registers + $K{\cdot}N$ support cells + $N$ query cells) | $8+4096+4096=8200$ |
| $M$ | number of cascade levels | 3 |
| $r_1 < \dots < r_M$ | cascade spacings (mm/voxel at each level) | $6, 3, 1.5$ |
| $\hat L^t_\ell$ | predicted target mask at level $\ell$ | $(B,1,128,128,128)$ per level |
| $\mathbf r_\ell$ | carried register state entering level $\ell$ (Eq. register-carry) | unused (`cascade_registers=false`) |

Note the deliberate symbol reuse already present in `3_method.tex`: non-bold
$r_1,\dots,r_M$ are cascade **spacings**, bold $\mathbf r_\ell$ is the **register**
state (Eq. register-carry). Keep the bolding — it's the only thing distinguishing
them and both names are already load-bearing in the draft.

---

## 2. Main pipeline: Encode → Attend → Decode

```
 I^t, {I^c_k}, {L^c_k}
        │
        ▼ stack T = K+1 volumes
 ┌───────────────────────────┐
 │   Enc_θ  (shared, §2.1)   │   one encoder pass over all T volumes
 └───────────────────────────┘
        │  coarse: (T·N, C_f)         │  fine: unpooled per-stage maps (query only)
        ▼                              │  (feeds Decode's skip path, §4)
 ┌───────────────────────────┐         │
 │   Tokenize (img + lbl)    │         │
 └───────────────────────────┘         │
        │  z^img, z^lbl : (T·N, e) each│
        ▼                              │
 ┌───────────────────────────┐         │
 │  l × bi-axial attention   │  (§3 / "sample-row attention")
 │   - slot axis  (img↔lbl)  │
 │   - row axis   (cell↔cell,│
 │     cross-volume, RoPE)   │
 └───────────────────────────┘         │
        │  q : (N, e)  query image slot│
        ▼                              ▼
 ┌───────────────────────────────────────┐
 │             Decode (§4)                │
 └───────────────────────────────────────┘
        │
        ▼
   L̂^t  :  (D,H,W) logit volume
```

### 2.1 Encode

A single shared encoder $\mathrm{Enc}_\theta$ (from-scratch nnU-Net-style plain-conv
stack, no ImageNet/plans.json pretraining) is applied once to all $T$ stacked
volumes (the $K$ context images and the target). It produces a feature pyramid;
two disjoint subsets of stages are read out of it:

- **coarse stages**, resampled to a common $R^3$ grid and concatenated on
  channels → one feature vector $f_i \in \mathbb R^{C_f}$ per cell $i$ of every
  volume. This is what feeds the transformer (§2.2/§3).
- **fine stages**, kept at their *native*, unpooled resolution (query volume
  only, unless a decoder head needs support rows too — see §5) → the skip
  pyramid the decoder fuses back in (§4). This is what lets the model emit a
  full-$D^3$ mask from an $R^3$ ($R \ll D$) token grid without ever running the
  heavy transformer at full resolution.

**[ref-config]** encoder stages `[32,64,256,512]` channels at strides
`[1,2,2,2]` → native sizes `[128,64,32,16]`. Coarse taps = stages `{2,3}`
(`256+512=768` ch, resampled to $R=16$). Fine taps = stages `{0,1}`
(`32` ch @ `128³`, `64` ch @ `64³`).

The target volume's image content is encoded exactly like a context image — no
architectural asymmetry here. The asymmetry between target and context is
introduced entirely at the label side, in tokenization (§2.2).

### 2.2 Tokenize

Every cell $i$ of every volume becomes **two** tokens sharing one row but two
"slots" $c \in \{\text{img}, \text{lbl}\}$:

$$z^{\text{img}}_i = \mathrm{ImgEmbed}(f_i), \qquad z^{\text{lbl}}_i = \mathrm{LblEmbed}(\text{occ}_i)$$

$\text{occ}_i \in \{0,1\}^{p^3}$ (or a soft probability in $[0,1]^{p^3}$) is the
$p\times p\times p$ occupancy tile of the mask at cell $i$, downsampled from the
native mask by the same $R^3$ cell grid. $\mathrm{LblEmbed}$ is one shared module
for every mask-bearing row — context rows and the query row go through the
*same* weights, never a per-role copy.

- **Context rows** always carry the real ground-truth tile: $\text{occ}$ from
  $L^c_k$.
- **The query row never carries ground truth.** Its label slot is either (a)
  the support-mean occupancy (no prior — the single-level default), or (b) the
  query prior threaded in from the previous cascade level (§6).

A shared additive Fourier position (or 3D-axial RoPE, see §3) marks each cell's
$(i,j,k)$ grid location; $n_t$ learned **thinking rows** are prepended to the
whole set, shared across every volume, giving the attention stack a
fixed-size read/write scratch space untied to any single cell.

### 2.3 Attend

$l$ stacked layers, each alternating two attention axes over the row/slot
token grid — this *is* the paper's bi-axial design (§sec:biaxial), detailed in
§3 below because it's the piece most worth pinning down precisely for the
method section.

### 2.4 Decode

The query's post-attention **image**-slot tokens $q \in \mathbb R^{N \times e}$
(the label slot is discarded at readout — see §3's "decode source" note) are
projected back to a spatial $R^3$ field and progressively fused with the fine
encoder skip maps (§2.1) via upsample → concat/FiLM → conv, one step per fine
stage, coarsest first. A final $1{\times}1{\times}1$ head plus a token-only
residual path produce the full-resolution logit $\hat L^t \in \mathbb
R^{D\times H\times W}$. (Two alternative heads exist and share the same
tokens-in/logits-out contract — a per-cell dynamic-filter head and a literal
Iris-style task-token decoder — but the deployed checkpoint uses the
progressive conv decoder above; not detailed further here since it isn't the
publication's design.)

**[ref-config]** fine stages `{1,0}` (coarse→fine order), taper widths
`[64,32]`, output grid `16·8=128` = input resolution exactly (a full-resolution
prediction, no final resize).

---

## 3. Sample/row attention (the bi-axial mechanism)

At any layer, the live token tensor is $X \in \mathbb R^{r \times c \times e}$:
$r$ rows = ($n_t$ registers) $\cup$ ($K{\cdot}N$ support cells) $\cup$ ($N$
query cells); $c=2$ columns = {img, lbl} slots. Each layer applies two
attentions in sequence:

**Slot axis** (fixed row, attend over $c$): the image and label tokens *at the
same cell* attend to each other. This is a direct, literal implementation of
`3_method.tex`'s "Image/label axis" — label evidence sharpens the image
representation and vice versa, at every cell independently.

**Row axis** (fixed slot, attend over $r$): every cell of every volume — support,
query, and the registers — attends over the *entire* row sequence at once,
using 3D-axial RoPE so relative attention is a function of physical
$(i,j,k)$ distance (positions are scaled by `spacing / rope_train_mm`, so a
6mm-pitch coarse level and a 1.5mm-pitch fine level place cells at the correct
*physical* distance from each other despite sharing the same $R^3$ token
count). Masking variants (mutually exclusive):

| mode | connectivity |
|---|---|
| `full_attn` (**[ref-config]**, dense) | every row reads every row, unmasked |
| context-read-only (`full_attn=False`, `query_self_attn=True`) | query rows read {registers, support, other query rows}; support rows read only {registers, support} (never see the query) |
| `register_routed` | each volume's $N$ cells attend **only within their own block**; the $n_t$ registers are the *sole* cross-volume path (all-to-all with every row) — blocks the direct cross-context token-matching shortcut entirely |

**[divergence]** `3_method.tex` §sec:backbone describes spatial self-attention
("Patch axis") and cross-context attention as two separate stages inherited
from the ResEnc bottleneck design (within-volume self-attention, *then*
read-only cross-attention into context). The row axis above does **not**
separate these: one dense (or masked) attention jointly does spatial
self-attention *and* cross-volume matching, because cells from every volume
and every spatial position already live in the same flat $r$-length sequence —
there is no separate "volume" tensor axis to stage a second attention over.
`3_method.tex`'s "Patch axis" bullet should either be dropped (row axis already
covers it) or rewritten to describe this one joint mechanism; picking one is a
prerequisite for closing the `\TODO` on Eq. factorization in §sec:biaxial.

**Decode source.** Only the query row's **image** slot is read out for
decoding (§2.4) — not the label slot — even though the label slot's own
cross-context attention performs a genuine content-based retrieval against the
real support masks (support populates it with ground truth; nothing else
does). The image slot is simply the more direct channel: it started from real
encoder features rather than a synthetic prior, for every row, always.

---

## 4. Pool token

An optional extra summary row per volume ($K{+}1$ rows total: one per support,
one for the query), inserted as a **prefix**, ahead of the registers — the same
insertion point cascade-register memory uses (§6), so it never disturbs the
per-cell $N$-token invariant the row-axis RoPE/masking logic depends on.

$$\text{pool}_v = \mathrm{PoolProj}\Big(\underbrace{\textstyle\sum_{\text{voxel} \in \text{fg}} \hat f_v(\text{voxel})}_{\text{foreground-masked average, finest fine stage}}\Big) + \text{pool\_type}$$

computed from the **finest** requested fine-decode stage, masked by that
volume's own foreground (real ground truth for support; the query prior — or
support-mean fallback — for the query, matching the label-slot rule in §2.2),
z-scored per-volume before pooling. This is the same idea as Iris's
foreground-stream token $T_f$ (§4.1 of `docs/methods/iris.md`), reintroduced
here as an *additional* row rather than a replacement for the per-cell tokens —
a cheap "what does this structure look like overall" signal that the dense
per-cell attention doesn't have to reconstruct implicitly from 4096 rows.

Requires `fine_decode=True` (needs the unpooled stage maps); incompatible with
`register_routed` (extra prefix rows break that mode's per-volume block
partitioning, same reason as §6). **[ref-config]: off** in the checkpoint used
for every current eval-table row — implemented but not the production path.

---

## 5. Query prior

The mechanism that actually connects consecutive cascade levels (§6 of
`3_method.tex`, Eq. query-prior). At level $\ell{>}1$, instead of the
support-mean fallback (§2.2), the target's label slot is seeded from level
$\ell{-}1$'s own prediction:

$$\mathbf z^{\text{lbl}}_{\ell,0} = \mathrm{LblEmbed}\big(\mathrm{occ}(\,\mathrm{Warp}_{\ell-1\to\ell}(\hat L^t_{\ell-1})\,)\big)$$

$\mathrm{Warp}_{\ell-1\to\ell}$ is **not** a plain upsample: level $\ell{-}1$'s
logit lives on level $\ell{-}1$'s own *augmented crop grid* (different center,
different physical FOV, possibly a different flip/affine draw under training
augmentation). The warp closed-form-composes the two levels' augmentation
grids and crop geometries and resamples with one `grid_sample` call
(`experiments/3d/cascade.py::_warp_prior_m2` / `_warp_prior_cropgeom`, exact
for affine-only augmentation) — this is what makes "the previous level's
belief" land on the correct voxels of the new, more-zoomed-in crop, not just a
naive resize of the same window.

Modes (`data.cascade_query_prior`, drawn once per level per step at train
time, deterministic at eval): `none` (support-mean, as if $\ell{=}1$), `pred`
(the real upstream prediction, warped as above), `gt`/`gt_coarse`/`gt_fine`
(oracle — training-only, teaches the network to *use* a prior at all before it
has to trust its own noisy one). **[ref-config]** train mix
`{pred: .8, none: .1, gt: .1}`; eval always deterministic `pred` — the honest
end-to-end cascade, no oracle leakage into any reported number.

**Prior perturbation** (train-only): the built prior is deliberately degraded
(random dilate/erode/shift/additive noise, magnitude drawn per step) before
it's tokenized, so a `gt` draw doesn't teach the network to trust a
suspiciously perfect prior it will never see at eval — it has to learn to
correct a plausibly-wrong one, matching what a real `pred` draw looks like.

**[divergence] none currently** — this section matches `3_method.tex` Eq.
query-prior closely; the one gap is that the paper doesn't yet mention the
geometry-warp step (`Warp`) at all, presenting the operation as a plain
"Upsample". Worth an explicit sentence once §sec:cascade's `\TODO`s are
addressed, since "upsample" undersells what actually has to happen for the
prior to land correctly on a re-centered, re-scaled crop.

---

## 6. Cascade registers

The *other* candidate cross-level channel (`3_method.tex` §sec:registers, Eq.
register-carry) — carries level $\ell$'s own post-attention **thinking-row**
state forward as extra input rows at level $\ell{+}1$, in addition to (not
instead of) the query-prior mask channel above:

$$\mathbf r_{\ell+1,0} = W_r\,\mathbf r_{\ell,\text{final}} + \mathbf e_{\ell+1}$$

Implementation: level $\ell$'s thinking rows, mean-pooled over the slot axis
$c$, are projected ($W_r$ = `cascade_proj`, a plain `Linear(e,e)`) and tagged
with a learned, level-independent type vector (`cascade_type`) that marks them
as *carried memory*, distinct from level $\ell{+}1$'s own **fresh** thinking
rows (which are always still present too — this mechanism adds rows, it never
replaces the thinking rows). Inserted as a prefix, same slot as pool tokens
(§4) — ahead of the registers, before support/query.

Unlike the query-prior channel (§5), which is explicitly detached between
levels ("each level keeps its own loss" — no gradient crosses the boundary),
cascade-register gradient is **not** detached: level $\ell{+}1$'s loss can
backprop through `cascade_proj`/`cascade_type` and into level $\ell$'s own
attention, giving level $\ell$ a direct training signal to extract whatever
helps the *next* level, not just whatever its own loss rewards. This is the
one place in the cascade where consecutive levels are trained with a shared
gradient path rather than being chained only through the (detached) rendered
prediction.

Mutually exclusive with `register_routed` (§3) — the block-mask partitioning
that mode relies on assumes no prefix rows besides the model's own thinking
rows; extra carried-memory rows are unaccounted for in that mask and would be
silently mis-routed.

**[ref-config]: off** in the production checkpoint. A dedicated ablation
checkpoint (`arch.cascade_registers=true`, otherwise identical recipe,
`docs/datasets/eval_expansion_status.md` §"patchset3d `cascade_registers`
checkpoint vs exp92_orig") gave a **mixed** result — some single-level source
gains, but the cascade path itself regressed on the one directly-comparable
source (hu_lwk1: 0.1287 → 0.0935) — so this mechanism is documented here as
implemented and evaluated, not as the paper's adopted design. State that
explicitly if §sec:registers is kept in the submission rather than quietly
presenting it as settled.
