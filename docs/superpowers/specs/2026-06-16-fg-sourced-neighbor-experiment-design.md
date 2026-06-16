# fg-sourced neighbor fill — diagnostic experiment

Date: 2026-06-16
Scope: `experiments/2d/multilevel/plot_sampling.py` (diagnostic only; no pipeline change)

## Question

The multilevel sampler (`sample_patches`) fills its patch budget from three tiers:
boundary core (`|value-0.5| < tau`), a fixed foreground quota (`n_fg_core` random
fg cells), and a neighbor field. The neighbor field today is `blur(boundary_core)` —
it diffuses outward from the boundary cells only; `fg_core` contributes nothing to it.

Hypothesis to test: if we **heavily sample `fg_core`** and source the neighbor field
from the foreground instead, does the true object boundary still get covered "for
free" by those fg neighbors — enough to drop the explicit boundary tier?

Known catch this experiment must quantify: `blur(fg_core)` peaks in the foreground
*interior* (cells surrounded by fg ≈ 1) and is only ~0.5 at the fg/bg edge, so the
field does not obviously peak on the boundary. Whether the boundary is nonetheless
adequately covered is the empirical question. A negative result (boundary gets
missed) is itself the finding — it would justify keeping the boundary tier.

## Changes

### 1. `sample_patches` — union neighbor field + one new arg

The neighbor proximity field is sourced from **both** core tiers:

```python
g = gaussian_blur((core_b | fg_core).float(), grid_res, blur_sigma)   # was: core_b only
```

so neighbors diffuse from the boundary core and the sampled foreground together. No
`neighbor_source` flag — the union is unconditional.

One new arg, `boundary_tier: bool` (default `True`): when `False`, the boundary core is
disabled (`core_b` becomes all-False, i.e. `tau→0`), leaving only the `n_fg_core`
foreground cells as core, so the field reduces to `blur(fg_core)`.

Floor, Gumbel noise, score stacking, and the final top-k are unchanged.

The three experimental regimes are reached from existing knobs + the union:

| run | `core_b` | `fg_core` | neighbor field |
|-----|----------|-----------|----------------|
| `--n_fg_core 0` | on | empty | `blur(core_b)` — today's baseline |
| `--n_fg_core 160` | on | 160 | `blur(core_b ∪ fg_core)` — union |
| `--n_fg_core 160 --no_boundary` | empty | 160 | `blur(fg_core)` — pure fg hypothesis |

Consequence: with `n_fg_core>0` the default neighbor field now includes the foreground,
so the boundary-only baseline is reached via `--n_fg_core 0` rather than being the
default. This `sample_patches` is the inlined diagnostic copy (docstring line 47), so
the real pipeline is unaffected.

### 2. CLI flag

- `--no_boundary` (action flag → `boundary_tier=False`)

Threaded into every `sample_patches` call site: `--stats`, `--sweep`, and the plot path.

### 3. Extend `compute_stats` with a boundary-coverage block

Boundary cells are defined on the **true GT**, consistent with the existing fg/bg
classification: a cell is a boundary cell iff `0 < gt32 < 1` (fractional occupancy at
`grid_res`). Add a third printed block:

```
[C] boundary coverage: of true-boundary cells (0<gt<1), where do they go
          dataset   bnd→core  bnd→neigh  bnd→miss
             ...        ...       ...        ...
            TOTAL       ...       ...        ...
```

`bnd→miss` (fraction of true boundary cells receiving no patch) is the headline metric.
This reuses the per-image selection masks already built in `compute_stats`; no new mode.

## Out of scope (YAGNI)

- No `--compare` mode — the head-to-head is run by invoking the script twice.
- No plotting changes.
- No edge-detector / gradient neighbor field.
- No `n_fg_core` sweep axis.

## How the experiment is run

```bash
# baseline: boundary-only neighbors (no fg quota)
python experiments/2d/multilevel/plot_sampling.py --stats --source prev_pred --n_fg_core 0

# hypothesis: fg-sourced neighbors, boundary tier dropped
python experiments/2d/multilevel/plot_sampling.py --stats --source prev_pred --n_fg_core 160 \
    --no_boundary

# control: union (boundary + fg), tier kept
python experiments/2d/multilevel/plot_sampling.py --stats --source prev_pred --n_fg_core 160
```

Compare the `[C]` blocks. If `bnd→miss` under the hypothesis run stays close to
baseline, fg neighbors cover the boundary and the tier is droppable; if it rises
sharply, the boundary tier is doing real work.
