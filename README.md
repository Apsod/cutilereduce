# CutileReduce

CutileReduce is an experimental CUDA Tile reduction generator for kernels that
can be expressed as monoidal folds. The project is based on the theory in
[`theory.pdf`](theory.pdf): many memory-efficient deep learning kernels can be
seen as tiled folds over associative operations, and useful backward passes can
often be derived from local gradient rules for those monoids.

The current implementation focuses on a practical subset of that model:
forward and backward kernels for fold-style reductions. A user supplies:

- a `Spec` describing the logical grid, input/output buffers, fold dimension,
  work model, and concrete tiling;
- a tile-local `map_reduce` function that maps input tiles to one or more
  monoidal summary tiles;
- a `combine` function that merges two summaries.

CutileReduce then builds CUDA Tile kernels that reduce over the fold dimension
inside each program and write the final summary. Optional `to_semantic` and
`to_output` hooks can convert the execution representation back into semantic
outputs.

## Current Scope

This repository is still a prototype. The current landed version supports:

- commutative fold-style reductions with forward and backward execution;
- general/non-commutative fold plans for ordered reductions that need prefix
  state;
- basic spec construction, cost estimation, and sweep utilities;
- working example-style experiments in [`xentropy.py`](xentropy.py),
  [`attention.py`](attention.py), and
  [`affine_attention.py`](affine_attention.py).

Not yet landed:

- decomposition along non-batch, non-fold dimensions: these inner dimensions
  must currently fit in a single program tile and cannot themselves be split
  across programs or selected as the loop axis;
- multiple fold/loop axes, such as joint or factorized cross-entropy kernels;
- first-class scan and scan-fold APIs beyond the internal scan used to combine
  ordered partial folds;
- broader API polish and packaging around the prototype examples.

## Function Interface

Fold callbacks are passed with `fold_functions(...)`. In the type sketches
below, `TID` is the CUDA Tile loop index info, `Input` is the tuple of input
tiles, `State` is the tuple of execution/carrier tiles, `Output` is the tuple of
semantic output tiles, `OutputGrad` is the tuple of output-gradient tiles,
`Embed` is the tuple returned by `embed`, and `InputGrad` is the tuple of
input-gradient tiles.

Forward callbacks:

```python
map_reduce: (TID, *Input) -> State
combine: (*State, *State) -> State
map_reduce_combine: (TID, *Input, State) -> State  # optional fused form
to_semantic: (*State) -> Output                    # optional
to_output: (*Output) -> Tensor | tuple[Tensor, ...] # optional host wrapper
```

When `map_reduce_combine` is omitted, a map-fold stage computes
`combine(*acc, *map_reduce(tid, *inputs))`. Supplying `map_reduce_combine`
lets an example fuse tile-local mapping with accumulator update, as in
[`attention.py`](attention.py).

Backward callbacks:

```python
embed: (*Output, *OutputGrad) -> Embed

# Commutative folds:
finalize: (TID, *Input, *InputGrad, *Embed) -> InputGrad

# General/non-commutative folds:
finalize: (TID, *Input, *InputGrad, *Embed, *State) -> tuple[InputGrad, State]
```

For general folds, `State` in `finalize` is the exclusive prefix state before
the current tile, and the returned `State` is the inclusive state after that
tile. `map_reduce_backward` is still accepted as a compatibility alias for this
stateful general-fold `finalize` callback.

## Early Benchmarks

The following measurements were collected on an NVIDIA GeForce RTX 5080. Each
script used its built-in analytical sweep followed by empirical tuning over 20
forward candidates. Backward tuning considered 40 xentropy candidates, 60
attention candidates, and 20 affine-attention candidates. The tables report
steady-state wall-clock CUDA timings from `torch.utils.benchmark` with
`num_threads=1`; compilation and tuning time are excluded.

These are local prototype measurements rather than portable performance
claims. In particular, the PyTorch baselines differ by example: standard
attention uses PyTorch SDPA, while affine LWS attention uses the explicit
PyTorch formulation in `affine_attention.py`.

Each example also accepts `--benchmark-memory`. This runs a separate pass over
the already selected and warmed plan and reports incremental peak
PyTorch-managed CUDA allocation for forward and forward plus backward. It does
not instrument or alter the timing samples. Use
`--benchmark-seconds 0 --benchmark-memory` for a memory-only run. Driver,
compiler, CUDA-context, and
other allocations made outside PyTorch's caching allocator are not included.

Tuned plans can be stored as versioned JSON and reused without repeating the
analytical or empirical sweep:

```console
uv run attention.py --save-plan plans/attention-rtx5080.json
uv run attention.py --load-plan plans/attention-rtx5080.json \
  --benchmark-seconds 0 --benchmark-memory
```

The file records each forward and backward stage kind, dimension extents and
tiles, program counts, and loop axis, along with the algebra, CuTile version,
and optional tuning-target metadata. Loading validates the format, algebra,
axes, and requested problem sizes before reconstructing the kernels. It stores
planning decisions rather than Python functions or machine code: CuTile 1.5
already maintains its own cubin cache keyed by compiler version, GPU
architecture, compiler options, and generated bytecode. Consequently, a loaded
plan skips tuning and recompiles only when CuTile cannot reuse a compatible
cached kernel.

### Cross Entropy

The xentropy problem has batch size $B=16{,}384$, vocabulary size
$V=16{,}384$, and embedding dimension $D=128$. Its two BF16 inputs have shapes
`ctx: [B, D]` and `trg: [V, D]`; the logical `[B, V]` logits matrix is never
materialized by CutileReduce.

The selected forward is `map_fold_partial -> fold`, with a
`[b=128, v=32, d=128]` map tile. The selected backward uses a
`[b=128, v=32, d=128]` tile, loops over `v`, and uses four contention groups.

| Mode | CutileReduce | PyTorch |
| --- | ---: | ---: |
| Forward | 1.1 ms | 2.1 ms |
| Forward + backward | 4.6 ms | 6.3 ms |

Incremental peak PyTorch-managed CUDA allocation for the selected plan:

| Mode | CutileReduce | PyTorch |
| --- | ---: | ---: |
| Forward | 0.500 MiB | 1,024.157 MiB |
| Forward + backward | 20.250 MiB | 1,536.062 MiB |

### Attention

The grouped-query attention problem has $H=4$ key/value heads, length
$L=4{,}096$, right/key length $R=4{,}096$, $G=8$ query groups per key/value
head, and $D_{QK}=D_V=128$. The BF16 inputs are:

- `query: [H, L, G, DQK]` = `[4, 4096, 8, 128]`;
- `key: [H, R, DQK]` = `[4, 4096, 128]`;
- `value: [H, R, DV]` = `[4, 4096, 128]`.

This corresponds to 32 query heads and four key/value heads. The selected
forward is a single `map_fold` using a
`[h=1, l=8, g=8, r=64, dqk=128, dv=128]` tile. The selected backward uses
`[h=1, l=4, g=8, r=32, dqk=128, dv=128]`, loops over `r`, and uses four
contention groups.

| Mode | CutileReduce | PyTorch SDPA |
| --- | ---: | ---: |
| Forward | 2.9 ms | 2.9 ms |
| Forward + backward | 14.8 ms | 10.1 ms |

Incremental peak PyTorch-managed CUDA allocation for the selected plan:

| Mode | CutileReduce | PyTorch SDPA |
| --- | ---: | ---: |
| Forward | 64.500 MiB | 64.501 MiB |
| Forward + backward | 176.500 MiB | 361.001 MiB |

The generated forward matches the highly specialized PyTorch SDPA baseline at
this shape. Its generated backward is currently about 1.47x slower.

### Affine LWS Attention

Affine LWS attention is the general, non-commutative fold example. It has
$H=2$, $L=2{,}048$, $R=2{,}048$, $G=2$, $D_{QK}=D_V=64$, and bias dimension
$D_B=32$. Its BF16 inputs are:

- `query: [H, L, G, DQK]` = `[2, 2048, 2, 64]`;
- `key: [H, R, DQK]` = `[2, 2048, 64]`;
- `value: [H, R, DV]` = `[2, 2048, 64]`;
- `bias_query: [H, L, G, DB]` = `[2, 2048, 2, 32]`;
- `bias_key: [H, R, DB]` = `[2, 2048, 32]`.

The selected forward is a single `map_fold` with a
`[h=1, l=64, g=1, r=16, dqk=64, dv=64, db=32]` tile. The selected full
recomputation backward uses
`[h=2, l=16, g=2, r=16, dqk=64, dv=64, db=32]`.

| Mode | CutileReduce | PyTorch FP32 | PyTorch BF16 |
| --- | ---: | ---: | ---: |
| Forward | 185.0 us | 1,497.5 us | 610.3 us |
| Forward + backward | 1.4 ms | 3.9 ms | 1.6 ms |

Incremental peak PyTorch-managed CUDA allocation for the selected plan:

| Mode | CutileReduce | PyTorch FP32 | PyTorch BF16 |
| --- | ---: | ---: | ---: |
| Forward | 2.062 MiB | 325.500 MiB | 160.000 MiB |
| Forward + backward | 10.562 MiB | 393.500 MiB | 193.500 MiB |

At this shape, CutileReduce is about 8.1x faster than the explicit PyTorch FP32
forward and 2.8x faster for forward plus backward. It is also about 3.3x faster
than the explicit BF16 forward and 1.1x faster for forward plus backward.

Against the FP32 reference, forward mean absolute error is `2.12e-4`. Gradient
MAE is `3.49e-3` for query, `5.57e-4` for key, `7.12e-4` for value, `1.35e-2`
for bias query, and `5.01e-3` for bias key. The implementation uses BF16
tensor-core operands with FP32 execution carriers and FP32 gradient
accumulation; gradients returned for the BF16 example inputs are BF16.

## Theory

The paper develops the Local Gradient Property for monoidal folds and scans.
For commutative folds, the key simplification is that the gradient of a global
product with respect to a local factor can be computed from the global product
and that local factor, without carrying a prefix. This is the theoretical case
targeted by the commutative examples in the current implementation.
