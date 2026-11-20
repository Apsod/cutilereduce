# Analytical planning and tuning

`PlanningProblem` combines a lowered `FoldProgram`, an explicit
`BackendModel`, and optional logical-size bindings. Construction, region
analysis, schedule costing, and candidate sweeps require no device discovery,
compilation, or kernel execution. The legacy callback path remains unchanged.

```python
from cutilereduce.experimental.planning.analysis import PlanningProblem
from cutilereduce.experimental.backends.cutile.analysis import cutile_analytical_target
from examples.experimental_attention import operator

program = operator().lower()
target = cutile_analytical_target(
    workers=16,
    contraction_flops_per_second=1e12,
    scalar_ops_per_second=1e11,
    transcendental_ops_per_second=1e10,
    bytes_per_second=1e11,
    launch_seconds=5e-6,
)
symbolic = PlanningProblem(program, target)
print(symbolic.regions["map_fold"].work.contraction_flops)
# 2*h*l*g*r*dqk + 2*h*l*g*r*dv

cost = symbolic.estimate(parameters={
    "tile.h": 1, "tile.l": 32, "tile.g": 1, "tile.r": 32,
}, backward=True)
print(cost.work)
print(cost.subs({"h": 4, "l": 1024, "g": 8, "r": 1024,
                 "dqk": 128, "dv": 128}).estimated_seconds)
```

These example rates are illustrative. Unspecified rates remain symbolic.
Logical extents, schedule parameters, and rates can all be partially bound.
`RegionCost.work` separates contraction FLOPs, arithmetic, comparisons,
transcendentals, conversions, reductions, and scans. SSA producers are counted
once, including shared producers across product components. Generic structured
fold/scan combine regions are analyzed recursively and counted once per
invocation rather than once per result field. Unknown operations raise errors.

Region argument/result byte counts describe shapes, not loads or stores.
Only schedule costing assigns traffic. Work expressions are polynomial when
possible; reductions use saturating subtraction, and tile iteration counts
retain ceilings, minima and maxima. Counts use FMA=2 and linear-work reduction
and scan conventions. Tree span, backend instruction count, and numerical
rounding errors are not inferred from those conventions.

## Schedule analysis

```python
sizes = {"h": 4, "l": 1024, "g": 8, "r": 1024, "dqk": 128, "dv": 128}
problem = PlanningProblem(program, target, sizes)
sweep = problem.sweep(backward=True, max_candidates=256)
for entry in sweep.shortlist(10, per_family=1):
    print(entry.candidate, entry.cost.estimated_seconds)
sweep.save("/tmp/attention-sweep.json")
```

`ScheduleCost.stages` exposes forward/backward work, reads, direct writes,
atomic traffic, parallel tasks, serial steps, resident-byte bounds and estimated
time. Aggregates include separate forward/backward time, saved state, and peak
auxiliary storage. Peak auxiliary storage excludes user inputs, visible outputs
and input-gradient destinations; stage allocations expose destination sizes.
Online cuTile projection is fused with its fold: no global carrier is stored
unless backward needs it. Forward partial buffers overlap final carriers/checkpoints; backward saved state
and accumulators overlap. This is a conservative lifetime model, not buffer reuse.

Counts use full padded tiles, including tails and idle partitions, and are upper
bounds for dense execution. Non-fold inputs are modeled resident across a fold
loop. Backward destinations are persistent or streamed according to loop-axis
dependence; absent owner axes imply atomic commits. Symbolically unresolved
ownership conservatively uses atomics. Binding before constructing an estimate
can consequently produce tighter traffic estimates than binding afterward.

Resident storage is a conservative dense-SSA liveness estimate plus input and
carrier/gradient tiles. It is not physical register or shared-memory allocation;
`max_resident_bytes` is an opt-in capacity proxy. `max_auxiliary_bytes` limits the
explicit auxiliary-storage estimate. `tile_domain_bounds()` derives necessary
polynomial input-residency bounds (other tile dimensions set to one) to narrow
initial domains before enumerating full SSA/carry capacity checks. Feasibility rejection records remain in
the sweep. Time uses a heuristic roofline with worker saturation and launch
latency; CPU defaults are uncalibrated vector-width proxies. No cache, spilling,
MMA instruction selection, or empirical occupancy model is currently inferred.

The default prefix-free training search independently enumerates forward and
backward tile domains, topology alternatives, partition counts, backward loop
axes and loop-program counts. The tile grids form a Cartesian product, visited
in cyclic diagonals so a budget does not exhaust backward variants of only the
first forward shape. Equal stage tiles are visited first; subsequent diagonals
decouple them. Ordered prefix/checkpoint schedules retain shared chunk layouts;
independent checkpoint reconstruction is not implemented. CPU supports
online/tree/partial/checkpoint families. cuTile supports online forward and
prefix-free backward. Ordered backward requires compatible checkpoint forward.
Unsupported combinations are explicit rejections. `sweep.truncated` reports a
candidate budget cutoff; it does not mean the entire domain was explored.
The default cap is 1,000,000 candidates in the planner and CPU search examples;
smaller budgets can be supplied explicitly.

Supply `parameter_domains` to override forward/backward tile domains, explore
embed tiles or partition choices, or pass concrete `ScheduleCandidate` objects
to `sweep`:

```python
sweep = problem.sweep(backward=True, max_candidates=256, parameter_domains={
    "backward_tile.l": (16, 32),
    "embed_tile.l": (32, 64),
    "loop_programs": (1, 2),
})
```

Allowed parameters are `tile.<axis>`, `backward_tile.<axis>`,
`embed_tile.<batch-axis>`, `partition_count`, and `loop_programs`. Inner axes
remain resident in the present backends. Floating-point contraction precision
is declared on the operator; changing it requires lowering another program.
The current search does not invent precision permissions or carrier layouts.

## Empirical tuning and reuse

`tune_cpu` and `tune_cutile` consume the same analytical candidates. Both require
inputs and a reference callable, validate outputs and (for training) input VJPs,
and benchmark complete compatible executions. Validation uses all-ones output
cotangents. Set tolerances appropriate for the declared precision. cuTile timing
uses synchronized CUDA events. Training calls use fresh autograd executions and
fresh gradient accumulation buffers rather than accumulating `.grad` across runs.

```python
from cutilereduce.experimental.backends.cpu.tuning import tune_cpu, compile_cpu_candidate
from cutilereduce.experimental.planning.tuning import load_candidate

# op is a FoldOperator and inputs follow its declared input order.
result = tune_cpu(op, sizes, inputs, reference=reference, backward=True,
                  candidate_count=5)
result.save("/tmp/selected-plan.json")
f = result.executable
candidate = load_candidate(result.problem, "/tmp/selected-plan.json")
f_again = compile_cpu_candidate(op, result.problem, candidate)
```

For cuTile, use `tune_cutile(..., analytical_target=target)` and
`compile_cutile_candidate` on a CUDA-capable host. The runtime emits saved
execution-carrier stores, separate embed/finalize launches, and an autograd
binding. Ordered cuTile backward and partial/tree cuTile forward still require
new backend implementations and are rejected. GPU compilation is lazy at first
launch, before steady-state timing.

`tune_candidates` is the common compile/validate/measure driver for custom
adapters. It deduplicates analytical candidates, retains a configurable minimum
per family, records failures, and saves the best valid measured executable.
Selected-plan files check the program fingerprint, shape and backend on reload;
recompilation uses the current backend. Hardware rates/limits should be supplied
again when reusing a decision on a different machine. Sweep JSON is an inspection
report; symbolic expression strings in it are not executable serialization.

Run the CPU-only demonstration with:

```console
uv run -m examples.experimental_planning --backend cutile --length 1024 --backward --json /tmp/sweep.json
```

## CPU realization and buffer reuse

The CPU region emitter now groups compatible FP32/FP64 expressions into native
SIMD loops, including named-axis broadcasting, comparisons/masks, and common
transcendentals. Trailing sum/max reductions can consume expressions directly;
a pointwise result and its trailing row sum can be produced in one pass. Batched
dot contractions use fused multiply/reduce loops; matrix contractions continue
to use ATen GEMMs. Actual tensor strides select SIMD contiguous/broadcast loads
or a scalar strided loop, with scalar tails. Cheap shared arithmetic can be
recomputed inside its consumers to avoid a whole-tensor temporary.

Last-use analysis runs on the emitted dependencies. A compatible contiguous
internal temporary can become the output of a pointwise loop in place. Formal
arguments, yielded values, and potentially aliased fallback values are protected;
the emitter does not mutate caller inputs or saved backward state. Unique-owner
gradient destinations use uninitialized allocation and are fully overwritten;
atomic destinations remain zero initialized. Persistent gradient accumulators
copy their first contribution before later in-place adds, except when one loop
iteration makes borrowing the contribution through its commit safe.

Mixed input dtypes, unsupported elementary operations, non-trailing reductions,
and scans retain the general ATen implementation. Fusion does not introduce
precision narrowing or fast-math compiler flags. The local extension uses
`-march=native`; AVX2-capable builds use ATen's vectorized math implementation.
This is local compilation, not a portable binary distribution.

`examples.experimental_cpu_single_tile` is the reproducible attention comparison
against eager PyTorch. It validates outputs and random-cotangent input gradients,
checks one tile and direct gradient stores, and alternates benchmark order.
Compilation is excluded from its steady-state timings.

Single-owner forward schedules return their local output/carrier tensors directly;
a single full backward tile similarly returns its local gradients, bypassing
scatter buffers and copies. These are ownership optimizations, not changes to
saved-state or gradient semantics. Analytical operation counts still describe
logical IR work: native fusion removes intermediate allocations and may duplicate
cheap arithmetic. The dense resident-byte model remains a conservative proxy,
and timing rates have not been calibrated to the new native kernels.

The CPU tile experiment and raw local measurements are in
[benchmarks/experimental_cpu_attention](../../../benchmarks/experimental_cpu_attention/README.md).
`examples.experimental_cpu_tiles` compares query/fold tiles and independent
backward layouts, validating random-cotangent gradients at each thread count.

Linux builds enable OpenMP compile/link flags when PyTorch provides OpenMP, and
build flags participate in the extension cache key. Singleton task schedules
avoid installing an outer parallel guard so their GEMMs can use intra-op workers.
Generated regions and tensor-operation worker closures use `NoGradGuard`:
OpenMP workers do not inherit the caller's grad-mode TLS, and the compiler's
explicit backward owns differentiation.

## Shared structural accounting

[ACCOUNTING.md](ACCOUNTING.md) describes the DSL-derived `RegionFacts` and the
backend-independent `ScheduleDomain` / `BufferAccounting` layer. Analytical
stages expose their accounting domains, and scheduled gradient ownership uses
the same absent-axis contribution rules. These quantities are separate from
CPU cache assumptions and GPU occupancy/communication implementations.

The benchmark can select independent training schedules directly from the
analytical search before compiling or measuring them:

```bash
uv run -m examples.experimental_cpu_tiles --length 512 --threads 1 4 \
  --only full query_128 full_forward_backward_query_128 \
  --analytical-shortlist 1 --json /tmp/cpu-independent.json
```

The report records each thread count's analytical candidate count, truncation,
model assumptions and selections alongside correctness checks and measurements.
With singleton head/group axes, five query and five fold choices per stage,
one-thread training has 7,500 candidates (the former coupled grid had 300).
This is the full coarse grid, not every legal integer tile or schedule.

## Backend and sweep contracts

The user-facing pipeline is:

```python
from cutilereduce.experimental import analyze, compile, tune, CpuBackend, CpuTarget
hardware = CpuTarget.detect()
backend = CpuBackend()
report = analyze(op, sizes, backend=backend, hardware=hardware, backward=True)
f = compile(op, sizes, backend=backend, hardware=hardware, backward=True)
result = tune(op, sizes, inputs, backend=backend, hardware=hardware,
              reference=reference, backward=True, candidate_count=20)
```

`analyze` performs no discovery, compilation or execution. `compile` selects the
analytical winner, or accepts an explicit candidate. `tune` analytically prunes,
retains the top-k (20 by default, including one per forward family), validates
outputs/random-cotangent gradients against the reference, measures complete
workloads and selects the fastest valid candidate. Failed candidates are recorded.
For custom shortlists, `report.shortlist(...)`, `planning.driver.compile_report`
and `tune_report` operate on the same retained implementations. Timing does not
implicitly calibrate the analytical model.

Backends implement `specialize`, `realize`, `compile` and `execution_adapter`:

- Specialization returns a `BackendProgram`: lowered program, hardware, shape,
  schedule space and `BackendModel` (rates, capabilities and physical rules).
- The space supplies templates; the model supplies domains and legality flags.
  Backend-specific integer domains use `extra_parameter_domains`. Shared search
  contains no CPU/cuTile-name dispatch.
- Realization fixes a candidate into an `Implementation` with `BoundExecution`
  (recursive bound loops) or `TiledExecution` (tiled SSA). More execution IR kinds
  require shared visitors; missing rules are explicit errors.
- Shared analysis derives logical work, padded repetition, traffic, storage and
  commit costs from that IR, applying specialized physical region/stage rules.
- Compilation consumes the retained realization. The execution adapter owns
  synchronization and timing, not scheduling or winner selection.

CPU emission and costing consume the same bound forward/backward choices and
region fusion/reuse registry. cuTile costs its fused online stage, saved execution
buffers, embed stage and finalize SSA without requiring a GPU. Both concrete
sweeps use this path, including `PlanningProblem.sweep` when supplied a model from
`cpu_analytical_target` or `cutile_analytical_target`.

`PlanningProblem.estimate` also supports partially bound, template-level symbolic
estimates. Fully bound backend estimates and sweeps realize concrete IR. Generic
models without a realizer provide coarse template estimates; they do not claim
physical implementation fidelity. Frozen fixtures verify numerical parity for
ragged Add, partial folds, asymmetric attention and ordered checkpoints.

Hardware, binding and modeling have distinct roles: CPU/GPU target records
provide resources; low-level `ExecutionTarget` binds legal tiles; `BackendModel`
supplies cost assumptions and scheduling capabilities. GPU peaks may remain
symbolic; ranking requires bound rates. Multi-GPU targets explicitly reject
executable distributed specialization in this milestone. Scope labels remain
available for future hierarchy without promising placement control not offered
by ATen or a GPU runtime.

Saved tuning plans use format version 2 and include program, sizes, backend,
hardware and cost-model identity. Older plans and changed hardware/models must be
selected/tuned again. They contain concrete decisions rather than pickled code.
The removed CPU selector and `ScheduleSpace` cost-hook selectors are not parallel
fallback paths.

The low-level CPU compiler preserves an explicitly supplied backward program
when it selects forward decisions. This is an inspection/debugging override,
not a search conditioned on that backward implementation. Optimizing the
remaining schedule choices around a fixed stage is deferred.
