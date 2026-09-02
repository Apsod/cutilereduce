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

- grouped reductions across multiple programs;
- scan and scan-fold execution;
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

(EARLIER VERSION: PLACEHOLDER)

The current benchmark paths use the built-in sweep to generate and evaluate
candidate configurations. These numbers should still be read as early local
measurements rather than portable performance claims.

All timings below are wall-clock CUDA timings reported by
`torch.utils.benchmark`, in milliseconds, with `num_threads=1`.

### Cross Entropy

| Mode | CutileReduce | PyTorch |
| --- | ---: | ---: |
| Forward | 2.0 ms | 6.8 ms |
| Forward + backward | 18.6 ms | 20.6 ms |

### Attention

| Mode | CutileReduce | PyTorch naive | PyTorch SDPA |
| --- | ---: | ---: | ---: |
| Forward | 7.7 ms | 56.7 ms | 5.0 ms |
| Forward + backward | 75.2 ms | 179.9 ms | 22.7 ms |

The attention result is mostly a sanity check against an already excellent
hand-tuned kernel. PyTorch SDPA/FlashAttention is a very strong target for
standard attention, and CutileReduce should not be expected to match it broadly
without much more attention-specific tuning. The more relevant target is custom
commutative folds where PyTorch does not already have a specialized fused
kernel.

## Theory

The paper develops the Local Gradient Property for monoidal folds and scans.
For commutative folds, the key simplification is that the gradient of a global
product with respect to a local factor can be computed from the global product
and that local factor, without carrying a prefix. This is the theoretical case
targeted by the commutative examples in the current implementation.
