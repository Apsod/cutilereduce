# CutileReduce

CutileReduce is an experimental CUDA Tile reduction generator for kernels that
can be expressed as monoidal folds. The project is based on the theory in
[`theory.pdf`](theory.pdf): many memory-efficient deep learning kernels can be
seen as tiled folds over associative operations, and useful backward passes can
often be derived from local gradient rules for those monoids.

The current implementation focuses on a practical subset of that model:
forward and backward kernels for fold-style reductions. A user supplies:

- a `FoldSpec` describing the logical axes, input, semantic, and carrier buffers,
  fold dimension, algebra, and work models;
- a forward pass specification:
  - a tile-local `map_fold` function that maps input tiles to one or more
    monoidal summary tiles;
  - a `combine` function that merges two summaries;
- a backward pass specification:
  - an `embed` function that turns semantic outputs and output gradients into
    a compact gradient representation;
  - a tile-local `map_finalize` function that maps input tiles, input-gradient
    accumulators, and the compact gradient representation into input tile
    gradients. For general/non-commutative folds, `map_finalize` also threads the
    ordered fold state.

Concrete problem sizes and tiling choices are supplied when constructing or
tuning a `FoldPlan`. CutileReduce then builds CUDA Tile kernels that reduce
over the fold dimension inside each program and write the final summary.
Optional `to_semantic` and `to_output` hooks can convert the execution
representation back into semantic outputs.

## Current Scope

This repository is still a prototype. The current landed version supports:

- commutative fold-style reductions with forward and backward execution;
- general/non-commutative fold plans for ordered reductions that need prefix
  state;
- basic spec construction, cost estimation, and sweep utilities;
- working example-style experiments in [`examples/xentropy.py`](examples/xentropy.py),
  [`examples/attention.py`](examples/attention.py), and
  [`examples/affine_attention.py`](examples/affine_attention.py).

Not yet landed:

- decomposition along non-batch, non-fold dimensions: these inner dimensions
  must currently fit in a single program tile and cannot themselves be split
  across programs or selected as the loop axis;
- multiple fold/loop axes, such as joint or factorized cross-entropy kernels;
- first-class scan and scan-fold APIs beyond the internal scan used to combine
  ordered partial folds;
- broader API polish and packaging around the prototype examples.
- support for multi-device distributed folds.

## Function Interface

Fold callbacks are passed with `fold_functions(...)`. In the type sketches
below, `TID` is the CUDA Tile loop index info, `Input` is the tuple of input
tiles, `State` is the tuple of execution/carrier tiles, `Semantic` is the tuple of
semantic tiles, `OutputGrad` is the tuple of semantic-gradient tiles,
`Embed` is the tuple returned by `embed`, and `InputGrad` is the tuple of
input-gradient tiles.

Forward callbacks:

```python
map_fold: (TID, *Input) -> State
combine: (*State, *State) -> State
map_fold_combine: (TID, *Input, State) -> State  # optional fused form
to_semantic: (*State) -> Semantic                    # optional
to_output: (*Semantic) -> Tensor | tuple[Tensor, ...] # optional host wrapper
```

When `map_fold_combine` is omitted, a map-fold stage computes
`combine(*acc, *map_fold(tid, *inputs))`. Supplying `map_fold_combine`
lets an example fuse tile-local mapping with accumulator update, as in
[`examples/attention.py`](examples/attention.py).

Backward callbacks:

```python
embed: (*Semantic, *OutputGrad) -> Embed

# Commutative folds:
map_finalize: (TID, *Input, *InputGrad, *Embed) -> InputGrad

# General/non-commutative folds:
map_finalize: (TID, *Input, *InputGrad, *Embed, *State) -> tuple[InputGrad, State]
```

For general folds, `State` in `map_finalize` is the exclusive prefix state before
the current tile, and the returned `State` is the inclusive state after that
tile.

Backward plans always contain two stages: `embed`, then `map_finalize` with
recomputation. Declare the `embed` results in `make_fold_spec(embedded={...})`,
in callback return order, using their logical axes and storage dtypes. The
`map_finalize` callback receives tiles with those logical shapes; any flattening
for matrix multiplication belongs inside that callback. Embedded outputs must
include every batch axis used by the embed stage.

Embed is tiled independently over its batch axes, with inner reduction axes
resident. Its analytical sweep ranks traffic and residency, followed by empirical
tuning. The gradient stage loads this shared representation instead of repeating
embedding in each program. Saved plans use format version 3; older plans need
retuning.

On CTK 13.4 or later, `build_fold(...,
programmatic_dependent_launch=True)` uses the embed stage as a programmatic
dependent-launch producer. Each embed block signals its dependent
map-finalize block after its output is ready; the latter waits before reading
embedded storage. The executor prepares all intermediate and gradient buffers
before launching embed, leaving the two launches adjacent on the stream.

## Early Benchmarks

### Scaling results

The following scaling measurements were collected on an RTX 5080 with
`--torch-compile --programmatic-dependent-launch --benchmark-memory
--accuracy-matrix`. Each row retunes the workload at the stated square problem
size: $B=V=N$ for xentropy and $L=R=N$ for both attention examples. Timings are
steady-state milliseconds.

#### Cross entropy

| N | CuTile forward | PyTorch forward | CuTile forward + backward | PyTorch forward + backward |
| ---: | ---: | ---: | ---: | ---: |
| 1,024 | 0.060 | 0.056 | 0.386 | 0.372 |
| 2,048 | 0.077 | 0.086 | 0.389 | 0.389 |
| 4,096 | 0.132 | 0.244 | 0.589 | 0.461 |
| 8,192 | 0.464 | 0.498 | 1.4 | 1.2 |
| 16,384 | 1.3 | 1.6 | 4.3 | 4.5 |

#### Attention

| N | CuTile forward | PyTorch SDPA forward | CuTile forward + backward | PyTorch SDPA forward + backward |
| ---: | ---: | ---: | ---: | ---: |
| 1,024 | 0.330 | 0.338 | 0.994 | 0.953 |
| 2,048 | 0.954 | 0.908 | 3.3 | 2.9 |
| 4,096 | 3.6 | 3.1 | 11.4 | 10.4 |
| 8,192 | 13.7 | 11.5 | 42.9 | 39.3 |
| 16,384 | 53.6 | 44.7 | 171.6 | 152.9 |

#### Affine LWS attention

| N | CuTile forward | PyTorch BF16 forward | CuTile forward + backward | PyTorch BF16 forward + backward |
| ---: | ---: | ---: | ---: | ---: |
| 1,024 | 0.227 | 0.230 | 0.758 | 0.584 |
| 2,048 | 0.586 | 0.823 | 2.2 | 1.9 |
| 4,096 | 1.9 | 3.4 | 7.6 | 7.5 |
| 8,192 | 8.7 | 13.0 | 36.8 | 30.5 |
| 16,384 | 32.8 | 54.1 | 143.7 | 288.6 |

#### Accuracy against PyTorch FP64

Each `worst` entry is the maximum MAE over all input gradients
for that workload and size.

| Function | N | CuTile output | CuTile worst | PyTorch BF16 output | PyTorch BF16 worst |
| --- | ---: | ---: | ---: | ---: | ---: |
| xentropy | 1,024 | $2.8\mathrm{e}{-7}$ | $8.8\mathrm{e}{-5}$ | $7.9\mathrm{e}{-3}$ | $9.0\mathrm{e}{-5}$ |
| xentropy | 2,048 | $2.7\mathrm{e}{-7}$ | $8.2\mathrm{e}{-5}$ | $7.7\mathrm{e}{-3}$ | $8.3\mathrm{e}{-5}$ |
| xentropy | 4,096 | $3.9\mathrm{e}{-7}$ | $8.0\mathrm{e}{-5}$ | $1.6\mathrm{e}{-2}$ | $1.1\mathrm{e}{-4}$ |
| attention | 1,024 | $6.0\mathrm{e}{-5}$ | $8.8\mathrm{e}{-4}$ | $8.8\mathrm{e}{-5}$ | $1.1\mathrm{e}{-3}$ |
| attention | 2,048 | $4.4\mathrm{e}{-5}$ | $6.3\mathrm{e}{-4}$ | $6.3\mathrm{e}{-5}$ | $7.8\mathrm{e}{-4}$ |
| attention | 4,096 | $3.2\mathrm{e}{-5}$ | $4.4\mathrm{e}{-4}$ | $4.5\mathrm{e}{-5}$ | $5.5\mathrm{e}{-4}$ |
| affine attention | 1,024 | $1.9\mathrm{e}{-4}$ | $3.1\mathrm{e}{-2}$ | $1.7\mathrm{e}{-2}$ | $7.5\mathrm{e}{-1}$ |
| affine attention | 2,048 | $1.9\mathrm{e}{-4}$ | $4.3\mathrm{e}{-2}$ | $2.4\mathrm{e}{-2}$ | $1.6$ |
| affine attention | 4,096 | $1.9\mathrm{e}{-4}$ | $6.1\mathrm{e}{-2}$ | $3.8\mathrm{e}{-2}$ | $3.0$ |

## Theory

The paper develops the Local Gradient Property for monoidal folds and scans.
For commutative folds, the key simplification is that the gradient of a global
product with respect to a local factor can be computed from the global product
and that local factor, without carrying a prefix. This is the theoretical case
targeted by the commutative examples in the current implementation.

## Constructing inputs and running examples

### Inspecting plans and analytical costs

The plan topology is independent of tuning. Starting from a `FoldSpec`, enumerate
symbolic forward, backward, or compatible joint plans:

```python
from cutilereduce.fold import forward_plans, backward_plans, joint_plan_templates
from cutilereduce.util.spec import SPECMAP
from examples.xentropy import xentropy_spec

spec = xentropy_spec()
print([path.name for path in forward_plans(spec)])   # full, partial
print([path.name for path in backward_plans(spec)])  # commutative
templates = joint_plan_templates(spec)
for template in templates:
    print(template.name, [name for name, _ in template.stage_costs])
    print(template.estimated_time)  # Symbolic sum of stage times, in seconds.

template = next(t for t in templates if t.name == "full+commutative")
values = {
    "b": 32, "v": 128, "d": 64,
    "full_b_tile": 16, "full_v_tile": 32,
    "backward_b_tile": 16, "backward_v_tile": 32,
    "backward_group_count": 1, "embed_b_tile": 16,
}
print(template.parameters)  # Names of sizes and tiling parameters.
print(template.evaluate({"v": 128}))  # Remaining parameters stay symbolic.
print(float(template.evaluate(values, hardware=SPECMAP["rtx5080"])) * 1e3)
plan = template.bind(values)  # Concrete FoldPlan, ready to build or save.
```

Each path object owns its stage sequence and lowering. A commutative backward
path can select another eligible loop axis with `path.with_loop("b")`.
`compose_plans(forward, backward)` validates dependencies; general checkpointed
backward requires a partial scan forward plan and shares its partition count
and fold tile. `plan_fold(spec, sizes, tiles=...)` is a convenience for manual
selection, while `tune_fold_plan(spec, functions, sizes,
hardware=...)` searches concrete choices. `build_fold(plan, functions)` creates
the executable callable. Saved version 3 plans remain loadable with
`load_fold_plan(spec, path, sizes)`.

`FoldSpec.mk_inputs` takes all logical axis sizes and returns tensors in input
spec order. Shapes, dtypes, and `requires_grad` come from the spec. Floating
inputs use normal random values by default. Named callbacks receive an allocated
tensor and a read-only mapping of all sizes; they may initialize in place or
return a tensor with the same shape, dtype, and device. Returned inputs are
leaves, so scaling during initialization does not create an autograd history.
Integer inputs require an explicit initializer to define their valid range.
Buffer `default` values remain kernel padding values.

```python
from examples.xentropy import xentropy_spec

ctx, trg, targets = xentropy_spec().mk_inputs(
    {"b": 32, "v": 128, "d": 64},
    device="cuda",  # use "cpu" for host-side experiments
    ctx=lambda tensor, sizes: tensor.normal_().mul_(sizes["d"] ** -0.5),
    trg=lambda tensor, sizes: tensor.normal_().mul_(sizes["d"] ** -0.5),
    targets=lambda tensor, sizes: tensor.random_(sizes["v"]),
)
```

Run examples from the repository root:

```console
uv run -m examples.attention --heads 4 --l 4096 --r 4096 --validate
uv run -m examples.xentropy --batch 1024 --v 4096 --d 128
uv run -m examples.affine_attention --help
```

The shared `ExampleParser` generates dimension arguments from `spec.axes`.
Examples supply size defaults and optional descriptive aliases (`--heads` for
`--h`, for example). Tuning, seed, compilation, benchmark, and plan persistence
options are shared; algorithm-specific options stay with their example.

All three kernel examples call `examples.cli.main` with their spec, fold
functions, size defaults, aliases, initializers, and references. The shared CLI
handles parsing, seeding, tuning or loading a plan, saving, building, validation,
and benchmarking. It builds one callable and passes it to the shared execution
helpers. `--torch-compile` selects the compiled callable for both validation and
benchmarking.
`--validate` or `--accuracy-matrix` enables validation in all examples.
All examples support `--forward-only` for tuning and running only the forward
pass. It currently cannot be combined with `--torch-compile`.
