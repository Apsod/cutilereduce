# `cutilereduce.experimental`

You describe **what** a fold computes — its inputs, a map, and an algebra — and
the compiler decides **how**: backward, tiling, loops, kernel boundaries,
storage, and write strategy. Correctness comes from the algebra's laws;
performance from the planner and backend.

This package is the intended **replacement** for `cutilereduce.fold`. The
callback-based path remains as a correctness and performance reference, not
as a supported alternative interface. Design rationale and longer worked
examples are in [DESIGN.md](DESIGN.md).

**Scope:** folds, commutative and non-commutative, with forward and backward,
written in the named-axis DSL. User-facing scans, multi-device folds, and
order-sensitive numerics are out of scope for now.

## What you write

| Piece | Form | Meaning |
|---|---|---|
| `FoldSpec` | `inputs={name: TensorSpec(axes, dtype)}`, `fold_axes`, `batch_axes` | Axes express *dependence*, not layout. `fold_axes` are eliminated by the algebra; all other axes survive unless the map contracts or reduces them. |
| `to_factor` | `(ctx, inputs) -> S` or `FoldFactor(value, where=mask)` | One logical interaction, in the tensor DSL. A masked factor *is* the semantic identity (zero cotangent). |
| `to_output` | `(ctx, S) -> O` (default identity) | What the caller observes. Unobserved fields are pruned from backward. |
| algebra | built-in (`Add`, `LogSumExp`, `LogWeightedSum`, `Affine`, `AffineLWS`, `Product(...)`) or a custom `FoldAlgebra` | How factors combine. |

```python
op = FoldOperator(spec, algebra, FoldMaps(to_factor=..., to_output=...))
from cutilereduce.experimental import CpuBackend, CpuTarget, compile, analyze, tune
hardware = CpuTarget.detect()  # explicit discovery at the application boundary
backend = CpuBackend()
f = compile(op, sizes, backend=backend, hardware=hardware, backward=True)
# Optional: analytical pruning/top-k, correctness validation, empirical selection.
result = tune(op, sizes, inputs, backend=backend, hardware=hardware,
              reference=reference, backward=True, candidate_count=20)
```

**You never write:** backward code, tile sizes, loops, kernels, layouts, or
store/atomic decisions. Schedule choices may be *overridden*, never required.

## Defining an algebra (optional)

```text
Monoid[T]           structure, identity, combine, commutative
                    lower_fold / lower_scan     optional; must equal repeated combine
FoldAlgebra[S,E,A]  semantic: Monoid S        what the map produces
                    execution: Monoid E       numerically useful carrier
                    accumulator: semigroup A  combines embedded cotangents
                    Phi: S→E   Psi: E→S
                    embed:    E × S* → A
                    finalize: [E_prefix ×] S × A → S*
                    finalize_context: NONE | PREFIX
```

A `TensorField` declares the *inner* axes that belong to one algebra element
(e.g. `LogWeightedSum(("dv",))`: one weighting shared by a `dv`-vector
payload). Values may carry further axes; algebra operations are pointwise over
every axis they do not eliminate and never see batch axes, tiles, or the map.

## Your obligations

1. `combine` is associative and `identity` exact; a fully masked fold returns
   the identity without NaNs.
2. An optimized fold equals `fold_E ∘ Phi` applied factorwise — **not**
   `Phi ∘ fold_S` — and is valid over *any* sub-region, not only the full fold.
3. `embed`/`finalize` give the true gradient of `Psi ∘ fold`. `S.commutative`
   requires `finalize_context=NONE`.
4. `E.commutative` is declared honestly; the planner reorders partial carriers
   only when it is true.
5. Invalid mathematical domains are yours to avoid.
6. Floating-point expressions may be reassociated or contracted.

## Pipeline and ownership

| Stage | Package | In → Out | Owns | Never |
|---|---|---|---|---|
| Trace | `compiler` | `FoldOperator` → `FoldProgram` | Symbolic regions (`map_fold`, `project_output`, `embed_output_vjp`, `map_finalize_map_vjp`), map VJPs, commutativity facts | Sizes, target, storage |
| Optimize | `ir` | `TensorProgram` → `TensorProgram` | Simplification, DCE, contraction discovery/classification, structural zeros | Materialization beyond declared outputs |
| Plan | `planning` | `FoldProgram` → `ScheduleSpace` | Hardware-independent templates (forward: online / tree / partial / checkpoint-scan; backward: embed → finalize), owner/loop partitions, resident vs streamed inputs, store vs atomic commits | Blocks, warps, threads, SIMD |
| Specialize | `backends/<t>` | template + sizes + target → `SpecializedSchedule` | Concrete tiles, program counts, buffers, selection | Semantics or algebra |
| Emit / run | `backends/<t>` | `SpecializedSchedule` → callable | Codegen, compilation, launch, autograd binding | Silent fallbacks |

## Invariants

- **Dependencies point one way:** backends → planning → compiler → algebra → IR.
- **No hidden materialization.** Only inputs, declared outputs, saved `E`, and
  planner-chosen buffers become memory; broadcasts stay lazy.
- **A region is not a kernel.** Region boundaries are logical; kernel and stage
  boundaries come only from the schedule.
- **Masking ≠ padding.** User masks map to the *semantic* identity; tiling
  out-of-bounds lanes map to the *execution* identity and belong to the planner.
- **Forward and backward are planned separately.** Prefix-free backward
  composes with any forward plan; prefix-dependent backward requires a forward
  plan that persists compatible checkpoints.
- **Unsupported is an explicit error** — no placeholder or numerically
  unstable substitutions (e.g. `log1p`/`expm1` on cuTile).

## Public surface

The top-level module exports operator/algebra authoring, core tensor DSL types,
`CpuBackend` / `CpuTarget`, `CuTileBackend` / `CuTileTarget`, and
`analyze` / `compile` / `tune`. Compilation selects analytically; tuning is explicit.
Both return autograd-compatible execution when backward is requested and supported.

Inspection and overrides live in their owning packages: `planning.analysis` for
candidates/reports, `planning.driver` for report selection/tuning,
`planning.ssa` and `planning.bound` for scheduled IR, and each backend's emission
modules for generated source. These APIs remain experimental. Former top-level
reexports and competing target/CPU selection helpers have been removed.

## Analytical planning and tuning

`PlanningProblem` derives symbolic region costs from the IR and composes them
with parameterized schedules. It supports CPU-only analytical sweeps, resource
pruning, empirical CPU/cuTile tuning, and saved concrete decisions. See
[PLANNING.md](PLANNING.md) for formulas, assumptions, examples, and backend limits.

## Status

| | Reference | CPU | cuTile |
|---|---|---|---|
| Forward: online | ✓ | ✓ | ✓ |
| Forward: tree / partial / checkpoint | ✓ | ✓ | ✗ |
| Backward: prefix-free | ✓ | ✓ | ✓ |
| Backward: prefix-dependent | ✓ | ✓ | ✗ |
| Launch + autograd runtime | — | ✓ | ✓ (online, prefix-free) |

Not yet supported: VJPs outside the current DSL subset; automatic derivation
of fused fold-accumulate updates; numerical rewrite policies (strict order,
compensated sums); a single target-generic compile entry point.

### Extension contracts

Algebra authors supply carriers, identity/combine, execution conversions and
embed/finalize using the tensor DSL. Backend authors supply specialization,
legal schedule domains/constraints, realization into shared scheduled IR,
physical region refinements, compilation and an execution/timing adapter.

The shared planner enumerates choices. Shared cost visitors analyze bound loops
and tiled SSA, combining region work/dependencies with repetition, storage and
commits. Backends specialize physical cost rules without replacing the entire
costing system. Analytical selection retains the implementations that empirical
tuning later compiles; emission does not independently choose a new schedule.

Hardware descriptions are separate from model policy. `CpuTarget` describes CPU
resources; `CuTileTarget` describes a single GPU and optional throughput data.
`BackendModel` in `planning.analysis` holds estimator assumptions and scheduling
capabilities. `ExecutionTarget` in `planning.target` is a low-level tile-binding
capability record, with no cost hooks or selection methods.

Distributed execution and user-facing scans remain outside this milestone.
Scope names and dependency/ownership accounting can support hierarchical
placement later; they do not imply an implemented cluster/node/device scheduler.
GPU runtime/source support is implemented, but this refactor's GPU execution
remains unverified on device hardware.
