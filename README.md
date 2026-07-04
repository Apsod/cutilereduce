# CutileReduce

CutileReduce is an experimental CUDA Tile reduction generator for kernels that
can be expressed as monoidal folds. The project is based on the theory in
[`theory.pdf`](theory.pdf): many memory-efficient deep learning kernels can be
seen as tiled folds over associative operations, and useful backward passes can
often be derived from local gradient rules for those monoids.

The current implementation focuses on the simplest practical subset of that
model: forward kernels for commutative folds. A user supplies:

- a `Spec` describing the logical grid, input/output buffers, fold dimension,
  work model, and concrete tiling;
- a tile-local `map_reduce` function that maps input tiles to one or more
  monoidal summary tiles;
- a `combine` function that merges two summaries.

`mk_fwd_no_group` then builds a CUDA Tile forward kernel that reduces over the
fold dimension inside each program and writes the final summary. Optional
`to_semantic` and `to_output` hooks can convert the execution representation
back into semantic outputs.

## Current Scope

This repository is still a prototype. The current landed version supports:

- commutative fold-style reductions;
- ungrouped execution only (`groups == 1`, via `mk_fwd_no_group`);
- basic spec construction, cost estimation, and sweep utilities;
- example-style experiments for attention-like and cross-entropy-like folds.

Not yet landed:

- grouped reductions across multiple programs;
- non-commutative/general monoidal folds that require prefix state;
- scan and scan-fold execution;
- backward kernels. The backward path is actively in progress, but it is not
  part of the usable API yet.

## Early Forward Benchmarks

The current forward path has only been tried with one-shot configurations chosen
from the sweep estimator. There is no autotuning yet, and these numbers should
be read as early local measurements rather than portable performance claims.

All timings below are wall-clock CUDA timings reported by
`torch.utils.benchmark`, in milliseconds, with `num_threads=1`.

| Example | Shape | CutileReduce | PyTorch baseline |
| --- | --- | ---: | ---: |
| Cross entropy | `b=32768`, `v=32768`, `d=128`, bf16 inputs | 10.0 ms | 27.4 ms |
| Attention | `l=4096`, `r=4096`, `h=4`, `g=8`, `dqk=128`, `dv=128`, bf16 inputs | 17.7 ms | 65.2 ms naive, 4.8 ms SDPA |

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
targeted by the current forward-only implementation.
