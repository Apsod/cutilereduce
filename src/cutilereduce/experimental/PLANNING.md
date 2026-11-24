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

`api.tune`, `tune_cpu` and `tune_cutile` use stage-wise search by default.
Training enumerates a union of forward, finalize and (where separately tiled)
embed choices, rather than the Cartesian product of their grids. Analytical
estimates still come from the batched scheduled-IR cost model. Explicit `analyze`
retains whole-plan enumeration; `analyze(..., stagewise=True)` requests the union.
An explicit candidate list or sweep is honored and its executable stages deduplicated.
Optional `StageSearchBackend.stage_candidates` enables factorized analytical
enumeration. Other staged backends retain full analytical enumeration until they
provide a generator covering their own executable units and coupled choices.

Each unique executable stage is compiled, validated against reference artifacts,
and timed once. `candidate_count` limits the shortlist **per stage and interface
class**, with forward-family diversity, rather than limiting whole training
plans. The planner finds the compatible composition with minimum summed measured
stage latency, validates its complete forward/VJP, and measures its complete
execution. A failed composition is recorded and the next compatible path tried;
individually valid stages remain reusable. `stage_measurements`, `selected_stages`,
`mode` and `tuning_metadata` expose what was actually searched and measured.
Stage tests use random cotangents and check upstream artifacts for mutation;
complete-execution validation also checks output/VJP correctness. Set tolerances
appropriate for declared precision. CPU timing uses wall time; cuTile uses
synchronized CUDA events. Gradient accumulation buffers are freshly allocated
and initialized for each call, and `.grad` is not accumulated across iterations.

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

`tune_report` orchestrates staged tuning for a `StagedBackend` and a timing adapter
with `measure_stage`. `tune_report(..., stagewise=False)` explicitly requests
whole-plan tuning. Backends/adapters without staging support retain whole-plan
`tune_candidates` behavior and report `mode="whole_plan"`.
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

`make bench_cpu` tunes forward and finalize independently at each thread count,
validates/times the single CPU embed implementation, and benchmarks the compatible
composition against eager PyTorch and SDPA. The per-stage shortlist is controlled
by `BENCH_CPU_SHORTLIST` (default 3); tuning repetitions use
`BENCH_CPU_TUNE_ITERATIONS` (default 10). The report's `tuning_searches` contains
analytical search metadata, stage measurements/failures and the selected stages.
The direct CLI equivalent is:

```bash
uv run -m examples.experimental_cpu_tiles --length 512 --threads 1 4 \
  --only full --tune-shortlist 3 --tune-iterations 10 \
  --json /tmp/cpu-independent.json
```

The optional `--analytical-shortlist` retains whole-plan analytical finalist
comparisons without empirical stage selection. For that older inspection path:

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

## Batched analytical evaluation

CPU and cuTile models classify candidates by topology, backward loop axis and
resolved commit pattern. One representative per class is lifted to parameterized
shared execution IR. The same cost visitors build its symbolic metric DAG, then
multi-output CSE and NumPy evaluate candidate columns in bounded batches (4096
rows by default). Numeric rows retain cheap lazy cost views; inspecting detailed
stages materializes their scalar description on demand. Regions and physical
stage estimates are cached, as are prepared class evaluators on a problem.

Concrete implementations are materialized when selected for compilation/tuning,
not for every analytical row. A finalist is retained after realization. The
`batched=False` sweep option preserves scalar evaluation for parity checks.
`evaluation_metadata` reports mode, structural groups and scalar fallback count.
Custom backends opt in through `BackendModel.batch_key`: its classes must preserve
operation structure and physical decisions when geometry is parameterized.
Models without this hook, and models whose hardware rates remain symbolic, keep
the scalar path rather than substituting an approximate cost model.

For the 2048-token CPU attention grid, measured searches after this change took
approximately 2.8 s cold / 0.7 s warm for 7,500 candidates and 4.4 s cold / 1.6 s
warm for 15,000 candidates on the existing i7-1165G7 benchmark host. The classes
numbered 36 / 48 and selected the same independent forward/backward tiles. These
timings include candidate enumeration and symbolic preparation, not compilation.
The benchmark now prints and records analytical search elapsed time separately.


## Executable stage contracts and independence

The backend returns a `StagedPlan` for a concrete schedule, through the optional
`StagedBackend` protocol in `planning/backend.py`. This describes executable
boundaries, not one stage per semantic tensor region. CPU currently exposes its
fused forward driver, embed, and finalize; cuTile exposes the corresponding three
kernel launches. Map/fold/scan/projection inside a fused driver cannot be timed or
selected independently through this interface. A backend exposing separate
callable units can return finer stages without changing shared selection logic.

A stage declares immutable `requires` artifacts, fresh `produces` artifacts,
implementation identity, its own schedule parameters, analytical cost selectors,
and any shared-resource `couplings`. Artifact representations include shape,
dtype, layout and schedule-dependent saved-state geometry. Data dependence does
not imply tuning dependence: embed can be tuned using a reference carrier if the
carrier interface is fixed. Ordered finalize requires matching forward checkpoint
partitioning, so winners with different checkpoint contracts cannot compose.
Embed consumes only the final carrier and therefore does not multiply with the
number of checkpoint layouts.

Responsibility is split explicitly:

- The backend declares fusion, callable boundaries, layouts/placement, resource
  coupling and stage-specific parameter influence; compiles stage runners;
  supplies reference fixtures; and assembles the selected runners.
- Shared planning checks artifact availability, compatible representations,
  unique output versions, execution order and coupling bindings. It deduplicates
  stage identities, ranks/prunes stage choices and composes measured winners with
  dynamic programming over interface states. It validates complete optima rather
  than every Cartesian combination.
- The execution adapter times isolated stages and complete executions, including
  target-specific synchronization. The backend runner owns fresh/reset scratch
  and accumulation state.

The contract is a backend assertion checked structurally and by reference tests,
not a general proof that arbitrary native code is separable. Undeclared mutation,
aliased outputs and overlapping resource usage invalidate independent timing;
these units must be fused or jointly tuned. Future cluster/node/device/CTA-level
backends can encode placement, communication buffers and collective topology in
artifact representations and couplings. Concurrent stages need a joint timing
unit; current composition deliberately supports sequential execution only.

Saved version-2 plans additionally record selected stage keys, interfaces,
parameters, individual measurements and tuning mode. The combined concrete
candidate remains available for existing reload/compilation APIs.
