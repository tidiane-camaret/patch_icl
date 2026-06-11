# modded-nanoTabPFN: Paper and Repo Summary

Source paper: *Speedrunning Tabular Foundation Model Pretraining* (Öztürk, Pfefferle, Hutter — arXiv 2606.03681, ICML 2026 Workshop)
Repo: `github.com/borawhocodess/modded-nanotabpfn` (local: `/home/dpxuser/repos/modded-nanotabpfn/train_nano.py`)

---

## What is TabPFN / Prior-Fitted Networks?

TabPFN trains a transformer entirely on **synthetic datasets** sampled from a prior distribution (TabICL prior: random classification tasks up to 1,000 rows, 20 features, 8 classes). At inference time, the real training set is passed as context tokens and the model makes predictions with **no gradient updates** — pure in-context learning. The prior generates unlimited synthetic tasks cheaply, so pretraining is on data quantity, not real data diversity.

---

## The Speedrun Format

Inspired by modded-nanoGPT, the competition fixes a downstream accuracy target (ROC AUC matching a Random Forest baseline = **0.8068** on 38 subsampled TabArena classification tasks) and asks contributors to reach it in minimum wallclock time on one NVIDIA L40S GPU, modifying a single self-contained training script. The leaderboard has 9 records so far, compressing 74.32 minutes to **0.92 minutes** (81× speedup, 22× fewer synthetic datasets).

---

## Architecture

### Tensor layout

All tensors use a 4D convention: `(batch, rows, cols, embedding)`.
Rows = datapoints, cols = features + target column. This allows two separate attention axes with no special masking constructs.

### Full forward pass

```
x (features)  → FeatureEncoder    → (b, rows, cols, e)  ─┐
y (labels)    → TargetEncoder     → (b, rows, 1,    e)   ├─ cat on col dim
                                                           ↓
                              (b, rows, cols+1, e)
                                                           ↓
                         ThinkingRows  →  prepend n_think rows
                              (b, rows+n_think, cols+1, e)
                                                           ↓
                         TransformerEncoderStack  (L blocks, residual_decay^i)
                                                           ↓
                         output[:, sep:, :-1, :].mean(dim=2)
                              (b, test_rows, e)
                                                           ↓
                         Decoder  (MLP)  →  class logits
```

`sep` = number of train rows (after thinking rows are prepended, sep is incremented by `n_think` so thinking rows are treated as train rows in attention).

### FeatureEncoder — repeated feature grouping

Each column `j` is encoded jointly with shifted neighbors using shifts `(0, 1, 3, 7, 15, ...)` = `2^i - 1`:

```python
x = torch.stack([x[:, :, (idxs + (2**i - 1)) % n_cols] for i in range(group_size)], dim=-1)
# → (b, rows, cols, group_size)
self.linear_layer(x)  # → (b, rows, cols, e)
```

`group_size=5` means shifts `(0, 1, 3, 7, 15)`. All pairwise differences are distinct, so no pair of columns co-occurs in more than one group (holds for any table with ≥ `max_shift + 1` columns). Per-column normalization is applied using train-row statistics (`mean`, `std` over `[:, :sep]`).

### TargetEncoder

Labels `y_train` are padded with the train mean to fill test rows, then projected with a linear layer. This avoids exposing test labels while still providing a valid-shaped tensor.

### ThinkingRows

```python
# n_think learnable embeddings, broadcast across all columns
thinking = self.row_tokens.unsqueeze(0).unsqueeze(2).expand(b, -1, c, -1)
x = torch.cat([thinking, x], dim=1)
sep += self.num_thinking_rows
```

Thinking rows act as a persistent latent memory: they appear before train rows, participate in row-axis attention as keys/values (all rows attend to them), and are not output rows. Attention maps show data rows concentrate attention on thinking rows, particularly in early layers. `n_think=24` in the final record.

### TransformerEncoderLayer (compiled)

Three sub-operations per block, all pre-norm with `LowerPrecisionRMSNorm`:

**1. Feature-axis attention** (across columns per row):
```python
x = src.reshape(b*rows, cols, e)
# standard full SDPA over cols
```
Each row independently attends across all its feature positions.

**2. Sample-axis attention** (across rows per column, asymmetric):
```python
x = src.transpose(1,2).reshape(b*cols, rows, e)
q_left, q_right = q.split([sep, rows-sep], dim=2)   # train / test queries
k_train = k[:, :, :sep, :]   # only train keys
v_train = v[:, :, :sep, :]   # only train values

x_left  = SDPA(q_left,  k_train, v_train)   # train rows attend to train rows
x_right = SDPA(q_right, k_train, v_train)   # test  rows attend to train rows only
```
This enforces the in-context learning constraint: test rows can read from train rows but not from each other.

**3. MLP** with GELU activation.

The `@torch.compile(dynamic=True)` decorator covers the full block forward pass.

### Residual decay

```python
for i, block in enumerate(self.transformer_blocks):
    x = x * (self.residual_decay ** i)  # 0.95^i before block i
    x = block(x, sep=sep)
```

Exponentially down-weights earlier-layer contributions to the final residual stream.

### Decoder (mean-pool)

```python
output = output[:, sep:, :-1, :].mean(dim=2)
```

- `sep:` — select test rows only
- `:-1` — drop the target column position
- `.mean(dim=2)` — mean-pool over all feature positions → one vector per test row

This "mean-pool decoder" replaced slicing the target token in Record 9 and was discovered by LLM-driven autoresearch.

---

## Training Setup

### Prior

A static HDF5 dump of 256,000 synthetic TabICL datasets. Each epoch iterates through `steps=32` mini-batches of `batch_size=2`, advancing a pointer through the dump (wraps around if exhausted). Evaluation budget is separate from wallclock: the training loop stops as soon as the downstream target is hit.

### Optimizer split

2D weight matrices **inside `transformer_encoder`** → **Muon**.
Everything else (1D params, encoders, decoder) → **schedule-free AdamW**.

```python
optimizer_muon = Muon(muon_params, lr=0.1*c.lr, momentum=0.96, weight_decay=0.1)
optimizer_adam = schedulefree.AdamWScheduleFree(adam_params, lr=c.lr, weight_decay=0.01, warmup_steps=1000)
```

Muon applies batched Newton-Schulz (5 steps) to compute the steepest-descent direction on the orthogonal manifold. For QKV matrices of shape `3e × e`, it detects the `size(0) == 3*size(1)` case and processes all three as a batch.

### LAWA (Latest Weight Averaging)

```python
lawa_queue.append({k: v.cpu().clone() for k, v in model.state_dict().items()})
# at eval: average queue → temp model → evaluate → restore training weights
```

Keeps a FIFO buffer of the last `K=10` checkpoints. At eval time, weights are averaged into a temporary model used only for scoring. Training continues from the unaveraged weights. Free accuracy improvement with no change to the training trajectory.

### Training precision

`torch.autocast(device_type="cuda", dtype=torch.bfloat16)` for forward + loss. `LowerPrecisionRMSNorm` disables autocast internally (applies RMSNorm in fp32 when input is bf16/fp16) to prevent norm instability, matching TabPFN v2.6 behavior.

---

## Speedrun Records Summary

| # | Time (min) | Datasets | Key change | Main driver |
|---|-----------|----------|-----------|-------------|
| 1 | 74.32 | 80,576 | Baseline (nanoTabPFN) | — |
| 2 | 54.41 | 45,824 | Muon optimizer | Sample efficiency |
| 3 | 10.10 | 13,184 | SDPA + bf16 + LR 10⁻⁴→10⁻³ + wider (192→256) | SDPA and LR |
| 4 | 9.26 | 13,184 | Batched Muon + `torch.compile` | Throughput |
| 5 | 7.57 | 11,200 | Residual decay 0.95^i | Sample efficiency |
| 6 | 3.88 | 9,664 | RMSNorm + 16 thinking rows | Thinking rows dominant |
| 7 | 3.48 | 8,768 | LAWA (K=10) + AdamW weight decay 0.01 | Eval quality |
| 8 | 2.15 | 4,992 | Repeated feature grouping (group size 3→5) | Sample efficiency |
| 9 | 0.92 | 3,648 | HPO via autoresearch + Muon WD + mean-pool decoder | Stacked tuning |

---

## Final Configuration (Record 9 vs Baseline)

| Param | Baseline | Record 9 |
|-------|----------|----------|
| Layers | 6 | 5 |
| Heads | 6 | 4 |
| Embedding | 192 | 256 |
| MLP hidden | 768 | 768 |
| Norm | post-LayerNorm | pre-LowerPrecisionRMSNorm |
| Feature group size | 1 | 5 |
| Thinking rows | 0 | 24 |
| Residual decay | 1.0 | 0.95 |
| Decoder input | target token | mean of feature tokens |
| Optimizer | schedule-free AdamW | AdamW + Muon |
| LR | 10⁻⁴ | 10⁻³ |
| AdamW weight decay | 0 | 0.01 |
| Muon LR | — | 0.1 × AdamW LR |
| Muon momentum | — | 0.96 |
| Muon weight decay | — | 0.1 |
| Gradient clip | 1.0 | 2.0 |
| LAWA | — | K=10 |
| Batch size | 1 | 2 |
| Steps per epoch | 64 | 32 |
| Precision | fp32 | bf16 autocast |

---

## Applicability to patch_icl

patch_icl shares the same core structure as TabPFN: a transformer trained to do in-context prediction given K context (input, label) pairs, predicting on a query input without gradient updates. Several techniques transfer directly:

| Technique | patch_icl analog |
|-----------|-----------------|
| **Thinking rows/tokens** | Prepend learnable tokens to the K context pairs in the bottleneck attention (both within-volume self-attn and cross-context cross-attn stages). The biggest single gain in the speedrun. |
| **Residual decay** (`0.95^i`) | Apply to ResEncInContext3D or ViTInContext3D encoder blocks. Zero extra parameters. |
| **LAWA** | Average last K model checkpoints at eval time; restore training weights after. No training changes. |
| **Mean-pool output** | Instead of decoding from a single target bottleneck token, mean-pool over all target spatial positions before the decoder. |
| **Muon optimizer** | Apply to 2D weight matrices (attention projections, MLP weights) in the encoder; AdamW for everything else (norms, biases, encoders). |
| **Repeated feature grouping** | Less direct — could inform how patch embeddings are grouped or how channel info is encoded, but the 1D column structure doesn't map cleanly to 3D patches. |
