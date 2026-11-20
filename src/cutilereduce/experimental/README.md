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
f = compile_cpu(op, sizes, backward=True)   # torch.autograd-compatible callable
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

The intended public surface is the table in *What you write* plus one compile
entry point per target. Everything else exported today is an **inspection and
override tier**, kept for convenience and without stability guarantees:

| Purpose | Entry points |
|---|---|
| Inspect lowered IR and contraction fusion | `operator.lower()`, `compile_fold(...).format()`, `TensorProgram.contractions` / `.reductions` / `.materializations`, `compile_*_region` |
| Inspect or override schedules | `build_schedule_space`, `format_plan`, `schedule_backward_program`, `schedule_cpu`, `CpuScheduleConfig` |
| Inspect generated code | `emit_cpp_source`, `specialize_cutile_schedule`, `emit_cutile_source`, `emit_cutile_backward_source` |
| Reference execution | `execute_tiled_fold_cpu`, `execute_tiled_backward_cpu` |

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

### Planned changes

- **`batch_axes` become inferred.** Tracing already determines each carrier
  field's actual dependencies; outer axes are those surviving into `E` minus
  the algebra's declared inner axes. Explicit declaration is inherited from
  the callback path, which cannot see through user code. The inner/outer
  choice is a modeling decision expressed by the algebra (one `LWS(("dv",))`
  vs `dv` independent `LWS(())` instances sharing a weighting), not by the
  spec. Whether to split an inner axis across programs (recomputing the shared
  fields) is then a planner cost decision rather than a hard restriction.
- **The cost model moves toward the IR.** Work (contraction FLOPs from the
  named-axis partition, reductions, transcendentals) is schedule-independent
  and computed from IR regions; traffic and recomputation follow from the plan
  (residency, partitions, recompute in backward); rates and weights come from
  the target.
