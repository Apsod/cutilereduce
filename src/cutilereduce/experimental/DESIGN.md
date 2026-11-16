# Experimental fold compiler: design notes

Long-form design notes and rationale. The user-facing contract, scope, and
status live in [README.md](README.md); where the two disagree, the README is
authoritative.

This package explores a user interface in which a fold is described by its
logical tensor interactions and its algebra, rather than by handwritten fused
tile callbacks. Its output is a target-independent tensor program: it makes
contractions, reductions, surviving axes, dtypes, and materialization
boundaries explicit before target-specific CPU or cuTile emission.

## Package organization and dependencies

The implementation is organized by responsibility:

```text
experimental/
  ir/                 Tensor SSA, operation contracts, map VJPs, rewrite passes
  algebra/            Monoid/FoldAlgebra definitions and fold/scan fallbacks
  compiler/           FoldSpec, maps, operator regions, algebra component graphs
  planning/           Schedule templates, write ownership, scheduled SSA, targets
  backends/
    reference.py      Deterministic tensor-region and tiled-schedule interpreter
    cpu/
      schedule.py     CPU schedule specialization and storage decisions
      regions.py      Tensor-region → ATen C++ emission
      codegen.py      Scheduled stages → C++ loops, buffers, and writes
      runtime.py      Extension compilation, argument binding, autograd execution
      cost.py         Analytical schedule costs and selection
      benchmark.py    Timing and memory measurements
    cutile/
      schedule.py     CuTile schedule specialization
      regions.py      Tensor-region → CuTile function emission
      codegen.py      Scheduled stages → CuTile kernels
```

Dependencies point from backends to planning/compiler/algebra/IR, never from
those layers to a backend. The tensor IR is independent of the fold algebra;
algebras depend on the IR, compiler regions depend on both, and planning
consumes the resulting operator program. Monoid fold/scan fallbacks belong to
`algebra.lowering`, not to a target backend. The reference interpreter consumes
scheduled SSA without owning CuTile specialization.

Backend region emitters interpret tensor operations; scheduled emitters own
loops, loads, stores, and gradient commits. CPU runtime execution is separate
from source generation. CuTile currently provides specialization and source
generation, not an end-to-end launch runtime.

The existing public imports from `cutilereduce.experimental` remain available.
Backend entry points are also available from `experimental.backends.cpu` and
`experimental.backends.cutile`. Internal imports should use the canonical
subpackages above; the former flat implementation module paths are removed.

The experiment separates three concerns:

- `FoldSpec` describes input tensors, identifies the axes that the algebra must
  fold, and records batch axes as user intent for validation and later planning.
- `Monoid` owns one structured value and its identity and combine operations.
  It may also provide optimized axis-polymorphic fold or scan lowerings.
  `FoldAlgebra` relates semantic
  and execution monoids `S` and `E` to a gradient-accumulator partial
  semigroup `A`.
- `TensorBuilder` records the user's map and the algebra lowering as an
  axis-aware tensor IR.

## Elementary tensor expressions and gradient cuts

Maps, output maps, and monoid implementations share the same symbolic tensor
DSL. In addition to arithmetic, contractions, reductions, scans, and explicit
comparisons, `TensorBuilder` provides:

- `powi(x, n)` (also `x ** n`) for a static Python integer exponent;
  `sqrt`, `rsqrt`, `abs`, `sin`, `cos`, `tanh`, and `sigmoid`;
- `minimum`, `maximum`, and `clamp(x, min=..., max=...)`, with constant bounds;
- `log1p`, `expm1`, and derived stable `softplus` and `logaddexp` expressions;
- boolean `logical_and`, `logical_or`, `logical_not` (`&`, `|`, `~`), and integer
  `floor_divide` and `remainder` (`//`, `%`), with Python sign conventions;
- `stop_gradient(x)`, a forward identity that cuts only that use's VJP path.

Operands broadcast by named axes and must have matching tensor dtypes; use
explicit casts when necessary. Symbolic tensors cannot control Python `if`
statements: use `ctx.compare(...)` and `ctx.where(...)`. Integer/boolean
operations and casts across discrete dtypes do not propagate gradients.
The VJP of `abs` at zero is zero; binary minimum/maximum divide tie gradients
equally; clamp passes the input gradient at both boundaries. Constant bounds
are not differentiable. Invalid mathematical domains, including integer
division by zero, remain the caller's responsibility.

Integer powers lower to multiplication chains, including reciprocal chains
for negative exponents. CPU interpretation and native CPU emission support
the entire operation set. CuTile supports the set except `log1p` and `expm1`:
emission raises an explicit error rather than replacing them with unstable
naive formulas. Recipes containing those operations have the same limitation.

`examples/experimental_teacher_tower.py` constructs a three-predictor
cross-entropy tower using ordinary `Product` algebras. Each next predictor's
logits are detached only in their teacher-weight role; their own student loss
still differentiates them. The real teacher uses compact integer class indices
and `axis_index`, not a materialized one-hot tensor. Individual losses and their
sum are exposed by the output map.

## Describing a fold

A `TensorSpec` gives each input its logical axes and dtype. Axis names express
dependence, not a physical tile layout:

```python
from cutilereduce.experimental import FoldSpec, TensorSpec

spec = FoldSpec(
    inputs={
        "query": TensorSpec(("h", "l", "g", "dqk"), "bfloat16"),
        "key": TensorSpec(("h", "r", "dqk"), "bfloat16"),
        "value": TensorSpec(("h", "r", "dv"), "bfloat16"),
    },
    batch_axes=("h", "l", "g"),
    fold_axes=("r",),
)
```

The algebra acts only on `fold_axes`. Every other dependency survives unless
the map itself contracts or reduces it; elementwise operations, contractions,
and reductions therefore batch over untouched axes automatically. Declared
`batch_axes` record which surviving axes belong to the outer fold problem.
Other axes are inner axes used by input interactions or fields of the algebra.
The distinction is useful for validation and later scheduling, but batch axes
are not passed to a monoid fold lowering. The current IR preserves input axis order but
deliberately makes no promises about physical storage, tiling, or CUDA layouts.

## Writing the map

The map describes one logical interaction with the fold algebra. It receives a
`TensorBuilder` and symbolic input values and returns the semantic structure
expected by the algebra:

```python
from cutilereduce.experimental import StructuredValue

def attention_map(ctx, inputs):
    scores = ctx.contract(
        inputs["query"],
        inputs["key"],
        over=("dqk",),
        dtype="float32",
    )
    return StructuredValue(
        {
            "log_weight": scores,
            "value": ctx.cast(inputs["value"], "float32"),
        }
    )
```

The map builds expressions; it does not allocate tensors. In particular,
`value` still depends only on `(h, r, dv)`. Returning it as part of a factor
lifted over `(h, l, g, r)` does not broadcast or materialize an
`(h, l, g, r, dv)` tensor.

`contract(lhs, rhs, over=...)` is the semantic contraction primitive. Dot
products, matrix-vector products, matrix products, and their batched variants
are instances of the same operation and are classified from their named axes.
Elementwise operations use lazy named-axis broadcasting.

A contraction may additionally carry a `ContractionPrecision` contract.  This
separates the logical result dtype from the operand representation and
accumulator dtype used by a realization.  For example, an attention numerator
can explicitly permit BF16 operands with an FP32 accumulator:

```python
from cutilereduce.experimental import ContractionPrecision

tensor_core_bf16 = ContractionPrecision(
    operand_dtype="bfloat16",
    accumulator_dtype="float32",
    allow_operand_rounding=True,
)
algebra = LogWeightedSum(("dv",), contraction_precision=tensor_core_bf16)
```

`allow_operand_rounding` is deliberately opt-in.  It permits the execution
carrier's FP32 softmax weights to be rounded for a BF16 MMA; without it, the
IR requires either matching operand dtypes or an explicit semantic cast.  The
contract belongs to the operation/algebra, not to a CuTile-specific user map.

Logical coordinates are available without exposing physical tile IDs:

```python
is_target = ctx.compare("equal", ctx.axis_index("v"), inputs["target"])
```

This permits compact targets, causal masks, and relative-position expressions.
The backend lowers an axis index to the current tile origin plus a local range.

When a predicate removes a complete interaction from the fold, return a
`FoldFactor`:

```python
return FoldFactor(
    value=StructuredValue({"log_weight": scores, "value": inputs["value"]}),
    where=ctx.compare(
        "less_equal", ctx.axis_index("r"), ctx.axis_index("l")
    ),
)
```

The false branch is the semantic monoid identity and therefore has zero input
cotangent. This is distinct from an unmasked value that merely happens to equal
the identity. Physical padding is separate: the schedule supplies an execution
domain to the fold region, and invalid lanes contribute the execution-carrier
identity after `to_execution`. This avoids assuming that `Phi` maps the
semantic identity to the execution identity. Backward regions use the same
domain to zero local semantic cotangents before the map VJP reduces axes shared
by multiple factors. Masking only the reduced input gradient would incorrectly
retain contributions from padded lanes. A backend may skip tiles proven
empty from simple axis-index comparisons. Optimized monoid folds must return
their exact identity, without NaNs, when every factor is masked.

## Constructing an operator

`FoldOperator` is the complete unscheduled, observable fold definition. It
bundles the logical tensor specification, fold algebra, factor map, and output
map:

```python
from cutilereduce.experimental import FoldMaps, FoldOperator

operator = FoldOperator(
    spec=spec,
    algebra=LogWeightedSum(("dv",)),
    maps=FoldMaps(
        to_factor=attention_map,
        to_output=lambda ctx, semantic: StructuredValue(
            {"value": semantic["value"]}
        ),
    ),
)
```

Only `to_factor: inputs -> S` and `to_output: S -> O` are user-authored.
Their VJPs are generated from tensor IR. Calling `operator.lower()` performs
purely symbolic lowering and produces a schedule-independent
`FoldProgram` containing:

```text
map_fold                         map -> Phi -> local E-fold
project_output                   Psi -> to_output
embed_output_vjp                 to_output* -> embed
map_finalize_map_vjp             finalize -> map*                 # commutative
map_finalize_map_vjp_accumulate  threaded E prefix, finalize, map* # general
execution_identity              scalar E identity
execution_combine               pointwise E combine
semantic_commutative            guarantees prefix-free finalization
execution_commutative           whether partial E values may be reordered
```

Exactly one map-finalize region is present. Commutative folds use the stateless
form. General folds use the accumulating form, which consumes an exclusive
partition-boundary execution checkpoint, threads it through its local fold
range, and returns the updated carrier. A partial forward plan persists local
`map_fold` carriers and exclusive-scans them with `E.combine` to produce those
checkpoints and the global execution result; `project_output` observes only
the latter. These `TensorRegion`s have formal symbolic arguments and SSA
results; their boundaries do not imply loads, stores, or materialization.
Together with the fold specification, execution structure, and commutativity
law, these regions define the logical contract. `build_schedule_space(program)`
constructs hardware-independent forward and backward schedule templates.
Templates are alternatives, not executable programs; their symbolic iteration
parameters have shape-bounded domains. `ScheduleSpace.analyze_shape` exposes
those domains without choosing a target, while `select_forward` and
`select_backward` may use target support and an optional target cost hook.

Semantic and execution commutativity are recorded separately. Commutativity
of `S` guarantees that backward finalization is prefix-free; the operator
rejects `S.commutative` together with `FinalizeContext.PREFIX`.
Commutativity of `E` independently determines whether a planner may reorder
partial execution carriers. The converse implication is not required: a
noncommutative semantic monoid may still admit prefix-free finalization.

The standalone `compile_fold_*` functions remain lower-level compatibility
entry points for lowering standalone programs.

## Schedule selection and target specialization

The `FoldProgram` and its schedule space are independent of problem extents and
target. A target may analyze shape-bounded choices, reject unsupported
templates, or provide a cost hook. The initial implementation uses
deterministic policies and explicit overrides; it does not run a general
constraint solver.

CuTile currently implements the online forward template. Specialization binds
shape and tile overrides, creates a self-contained `SpecializedSchedule`, and
emission consumes that object rather than separately supplied program, stage,
and sizes:

```python
from cutilereduce.experimental import (
    TargetModel,
    build_schedule_space,
    emit_cutile_source,
    specialize_cutile_schedule,
)

program = operator.lower()
space = build_schedule_space(program)
cutile = TargetModel(
    "cutile",
    frozenset({"tensor_regions", "tiled_loops", "multi_stage"}),
    supported_schedules=frozenset({"online", "independent", "prefix"}),
)
forward = space.select_forward(sizes, cutile)
specialized = specialize_cutile_schedule(
    program, forward, sizes, tiles={"l": 32, "r": 64}, target=cutile
)
source = emit_cutile_source(specialized)
```

The specialization owns its source `FoldProgram`, selected schedule template,
shape, bound parameters, target description, and concrete stage DAG. The same
wrapper is used for the CuTile backward emitter. CPU compilation likewise
retains a `SpecializedSchedule`; its target-specific implementation is the
CPU-scheduled plan. `execute_tiled_fold_cpu` remains a correctness interpreter
for tile coverage, remainder handling, carrier threading, and region
composition, not the optimized CPU backend.

```text
parallel output owners:
    state_E = execution_identity()
    for fold_tile in serial fold tiles:
        state_E = map_fold_combine(input_tile, state_E)
    store project_output(state_E)
```

`map_fold_combine` is a single symbolic region for updating a running carrier
from one input tile. Its generic lowering is `combine(state_E,
fold_E(Phi(map(input_tile))))`; a monoid may provide a fused accumulate
lowering when it can avoid materializing that tile's partial carrier. For
example, the LWS execution carrier lowers its numerator contraction directly
into the rescaled running numerator. `map_fold` remains available to schedules
that need independently materialized partial carriers, followed by a later
tree or staged combine.

`lower_backward_stage` lowers the existing scheduled-backward description to
the same SSA stage form: persistent gradients become loop carries, streamed
gradients commit inside the loop, and persistent gradients commit after it.

An `AxisPartition(axis, extent, tile, programs)` describes both levels of
ownership for one logical axis. Programs own balanced, disjoint spans of tile
indices; the stage-local serial loop visits the tiles in the current program's
span. Thus an axis may be parallel between programs and serial within each
program without being represented as two different logical axes. Setting
`programs=1` is the ordinary single-program serial fold. The same structure is
used by the CPU correctness interpreters and cuTile source emission.

Each loop stage classifies input bindings by dependence on its loop axis.
Bindings without that axis are resident: the backend loads their tile once
before the local loop. Bindings carrying the loop axis are streamed and loaded
for the current tile inside the loop. For example, attention over `r` keeps
`query` resident while streaming `key` and `value`; cross-entropy over `v`
keeps `context` and the compact target index resident while streaming class
embeddings. Backward applies the same rule to the embedded accumulator.

Backward is initially a two-stage `TiledPlan`. An `embed_output_vjp` map stage
materializes one accumulator tile per output tile. A `finalize_map_vjp` stage
then reads that accumulator while traversing its owned factor tiles. Gradients
independent of the local loop are loop-carried and committed at program end;
factor-local gradients can commit each iteration. Shared destinations use
explicit relaxed atomic adds, while destinations with unique ownership use
stores. The dependency and its accumulator buffer are explicit plan objects.

Nothing in this topology denotes blocks, warps, cores, or SIMD lanes. A CPU
backend may map owners to worker tasks, while a cuTile backend may map them to
grid blocks. Forward partial-carrier materialization is a separate schedule
topology rather than implicit behavior of this one. Backward may already split
its loop axis between programs; the resulting shared gradient commits remain
explicit.

`build_schedule_space(program)` constructs hardware-independent recursive
schedule templates. The initial forward candidates are an online carrier fold, a fused tree, a
materialized partial fold, and—when prefix-dependent backward requires
it—a partial fold followed by an exclusive checkpoint scan. Partitions may
contain their own online fold over smaller chunks; this nesting is part of the
plan rather than a CPU or GPU concept.

Forward and backward schedule templates are separate. Prefix-free backward composes with
any forward plan that supplies the global execution carrier. Prefix-dependent
backward additionally declares its boundary-prefix requirement, and therefore
only composes with a forward persistence plan that supplies compatible
checkpoints.

`schedule_cpu` binds a plan to concrete extents, chunk sizes, partition counts,
parallel/serial loops, and private or materialized storage. These CPU choices
do not appear in `FoldPlan`. Nested thread-pool parallelism is suppressed by
the initial scheduler.

The first executable path is:

```python
compiled = compile_cpu(
    operator,
    sizes={"h": 2, "l": 128, "g": 1, "r": 128, "dqk": 64, "dv": 64},
    backward=True,
    # Optional and independent of the forward chunk schedule:
    backward_schedule=CpuScheduleConfig(chunk_sizes=(("r", 32),)),
)
output = compiled(query, key, value)
```

An explicit scheduled backward can be supplied while schedule selection is
still experimental:

```python
problem = operator.lower()
backward_program = schedule_backward_program(
    problem,
    sizes,
    loop_axis="l",
    tiles={"l": 32, "r": 64},
)
compiled = compile_cpu(
    operator,
    sizes,
    backward=True,
    backward_program=backward_program,
)
```

It generates and caches a size-specialized PyTorch C++ extension for
contiguous CPU tensors. Prefix-free algebraic backward is exposed through a
custom `torch.autograd.Function`. Its partitioning and chunking are scheduled
independently of forward. Prefix-dependent backward shares only the persisted
forward partition boundaries while retaining an independent local chunk size.
The generic region emitter covers the current tensor IR. Contractions are
canonicalized from named axes into `B/M/N/K` and lowered uniformly through
ATen `mm` or `bmm`, and accumulated contractions use `addmm` or `baddbmm`;
there are no algebra- or workload-specific CPU emitters.
Intermediate tensors therefore follow the selected region and tile boundaries.

The executable backend accepts online, fused-tree, and materialized-partial
forward plans. Chunk and partition extents compile to cached specialized
`map_fold` and `map_fold_combine` functions. A generated `scheduled_forward`
C++ driver maps the Cartesian tile grid of all batch axes through
`at::parallel_for`; inputs that
carry an owner axis are sliced while shared inputs remain shared. Each owner
program traverses fold-axis tiles. Serial online folds retain one execution
state and invoke the shared `map_fold_combine` region for map-fold leaves;
nested online folds combine each completed child carrier immediately. Tree
reductions retain their partial carriers, and checkpoint plans preserve their
partition-boundary state. Each program projects the result and returns saved
execution state in one extension call. Parallel CPU
loops lower to `at::parallel_for`; no Python code participates between forward
tiles. Online carrier storage is independent of loop length, but ATen operations
can still allocate temporary tensors; configured scratch budgets are not yet
enforced. Prefix plans persist exclusive partition-boundary carriers; generated accumulating backward
loads each boundary, performs the local structured execution scan, threads the
updated prefix through local chunks, and merges the resulting input gradients.
For a supplied prefix-free `ScheduledBackwardProgram`, the extension instead
generates one scheduled backward entry point: CPU programs run through
`at::parallel_for`, persistent gradient tiles remain local across the serial
tile loop, and completed tiles use direct stores or C++20 relaxed atomic
additions. Prefix-free CPU compilation constructs this fused scheduled program
automatically when one is not supplied: batch axes form the program grid, the
fold axis is traversed locally, uniquely owned gradients use direct stores,
and multiply-owned gradients commit completed local tiles with relaxed atomic
adds. Prefix-dependent schedules continue to use the existing executor.

`estimate_cpu_plan` counts region operations, materialized carrier/output
bytes, exposed parallel tasks, and serial steps. `select_cpu_plan` evaluates a
small deterministic set of chunk and partition configurations against the
detected worker/vector width and supplies the default plan used by
`compile_cpu`. This is an analytical first model; benchmark calibration and
cache-aware traffic tiers remain future refinements.

### Scheduled programs and shared gradient writes

Tensor IR specifies contractions, reductions, and value dependencies, but it
does not prescribe concurrent writers or memory scope. After an abstract plan
has been tiled, the scheduled-program IR represents those physical decisions
explicitly:

- `Parallel` partitions logical axes into independently executing programs;
- `SerialLoop` is a program-local tiled loop;
- `LocalZero` and `Accumulate` describe loop-carried resident tiles;
- `Store` commits a uniquely owned output tile;
- `AtomicAdd` commits a contended tile with relaxed ordering.

Backward output lifetime and write contention are separate properties. A
gradient whose axes contain the local loop axis is streamed and committed at
each loop iteration. A gradient without the loop axis is accumulated locally
and committed once at program end. Independently, any nontrivial parallel
program axis absent from the gradient axes creates multiple writers and turns
the commit into a relaxed atomic add.

For attention with programs partitioned over `r` and a local loop over `l`,
this gives the characteristic schedule:

```text
parallel r-program:
    local dK, dV = zero
    loop l-tile:
        dQ-tile, dK-delta, dV-delta = map_finalize_map_vjp(...)
        accumulate dK, dV
        atomic_add relaxed dQ
    store dK, dV
```

Only tile-sized gradient state is private. The logical per-program `dQ`
contributions are never materialized as a program-by-output tensor. A target
backend maps the same scheduled operations to CPU threads and atomics, GPU
programs and device atomics, or another explicit execution hierarchy.

## Specifying a fold algebra

A fold algebra contains two monoids and a gradient-accumulator semigroup:

- `semantic` (`S`): factors produced by the map and semantic fold results;
- `execution` (`E`): the numerically or operationally useful reduction
  carrier;
- `accumulator` (`A`): the partial semigroup used to combine embedded prefix
  cotangents in scan backward.

Each monoid owns its structure, identity, and combine. `A`
requires only an associative combine on its valid domain; an identity is not
part of the `FoldAlgebra` contract. Each
`TensorField` declares only its inner axes. Values retain all of their actual
named-axis dependencies, while `fold` is told only which axes it must
eliminate:

```python
from cutilereduce.experimental import (
    FoldAlgebra,
    Monoid,
    StructuredValue,
    TensorField,
    TensorStructure,
)

class ExampleExecutionMonoid(Monoid):
    def __init__(self):
        vector = TensorField(("d",), "float32")
        self.structure = TensorStructure({"total": vector})

    def identity(self, ctx):
        ...

    def combine(self, ctx, left, right):
        ...

    def lower_fold(self, ctx, values, *, fold_axes):
        # May use reductions, contractions, scans, and elementwise operations.
        ...
```

`lower_fold` and `lower_scan` are optional optimizations. Compiler-owned
`lower_fold(monoid, ...)` and `lower_scan(monoid, ...)` emit structured
associative IR operations using `identity` and `combine` when no specialization
is provided. A scan returns both exclusive per-element prefixes and its final
carrier, without prescribing whether either is materialized by a backend.
The built-in Add monoid expands its scan to `scan_sum`, and products derive
component scans. Coupled monoids such as LWS and affine-LWS currently retain
the generic associative scan with its explicit combine region. They should
only override `lower_scan` when they can emit an actual equivalent tensor
program; naming an opaque monoid-specific operation is not considered a
lowering.

`FoldAlgebra` then owns only `Phi: S -> E`, `Psi: E -> S`,
`embed: E x S* -> A`, and `finalize`. Its `semantic` and `execution`
attributes are `Monoid`s, while `accumulator` is a `PartialSemigroup`. For an
ordinary fold, `embed` is called once and `A.combine` is unnecessary. For a
scan, embedded prefix cotangents are suffix-scanned with `A.combine` before
each call to `finalize`:

```text
A_i = combine_A(j > i, embed(prefix_j, grad_prefix_j))
grad_factor_i = finalize(prefix_i, factor_i, A_i)
```

`StructuredValue` remains a flat record of tensor fields belonging to one
algebra element. Recursive algebra constructions use the separate,
semantically neutral `CompositeValue`, whose named children are themselves
structured or composite algebra values. `CompositeStructure` is its schema.
The wrapper says only how values are nested; the owning algebra determines how
the children interact.

For example, cross entropy can use a componentwise product:

```python
from cutilereduce.experimental import Add, LogSumExp, Product

algebra = Product(
    normalizer=LogSumExp(),
    target=Add(),
)

def xentropy_map(ctx, inputs):
    logits = ctx.contract(inputs["context"], inputs["targets"], over=("d",))
    target_logits = ctx.where(
        inputs["is_target"],
        logits,
        ctx.full(logits.axes, 0.0),
    )
    return algebra.value(
        normalizer=StructuredValue({"value": logits}),
        target=StructuredValue({"value": target_logits}),
    )
```

`Product` is the concise public constructor for `ProductAlgebra`. Its semantic
and execution `ProductMonoid`s derive identities, combines, and folds
componentwise; its `ProductSemigroup` combines accumulators componentwise; while
the algebra delegates `Phi`, `Psi`, `embed`, and `finalize`. The two components still inhabit one
IR graph: the logits contraction is shared, their reductions can later be
scheduled together, and semantic leaves materialize under stable paths such as
`normalizer.value` and `target.value`.

The neutral composite representation also admits non-componentwise
constructions. `SemidirectProductMonoid` stores its acting and payload operands
in a `CompositeValue` and derives identity and an ordered coupled combine from
an action lowering. `SemidirectProductAlgebra` additionally derives
componentwise `Phi` and `Psi`. With the right-oriented convention:

```text
(a_s, a_v) o (b_s, b_v)
    = (a_s o b_s, action(b_s, a_v) o b_v)
```

The generic algebra construction cannot infer an efficient fold or a resolved accumulator
factorization from an arbitrary action alone. Concrete instances provide those
operations. Once the IR has a generic structured ordered-fold region, `Phi`,
the action-aware `E.combine`, and the identity will also be sufficient for a
correct unoptimized forward fallback.

`LogScaleAffine`, also exported as `Affine`, is the first concrete instance:

```python
from cutilereduce.experimental import Affine

algebra = Affine(("d",))
factor = algebra.value(
    acting=StructuredValue({"value": log_scale}),
    payload=StructuredValue({"value": translation}),
)
```

It represents `x -> exp(s) * x + v`, so ordered combination is:

```text
a o b = (a.s + b.s, exp(b.s) * a.v + b.v)
```

Its optimized region fold uses a reverse exclusive sum scan of log scales and
a weighted sum of payloads. This makes ordering explicit in the tensor IR
without fixing whether the region is a hardware tile or the complete fold.

`AffineLWS` is the bounded, normalized variant in which the additive payload
is itself a log-weighted sum:

```python
from cutilereduce.experimental import AffineLWS

algebra = AffineLWS(("d",))
factor = algebra.value(
    acting=StructuredValue({"value": log_scale}),
    payload=StructuredValue(
        {"log_weight": log_weight, "value": payload}
    ),
)
```

Its ordered operation is:

```text
(a.alpha, a.beta) o (b.alpha, b.beta)
    = (a.alpha + b.alpha,
       reweight(b.alpha, a.beta) LWS-combine b.beta)

reweight(s, beta).log_weight = beta.log_weight + s
reweight(s, beta).value      = beta.value
```

The execution carrier stores the payload in the stable
`{maximum, normalizer, numerator}` LWS representation. Consequently the
action changes only a log-domain field, and projecting the observed payload
produces a convex combination rather than an unbounded affine translation.
Its gradient accumulator is `Add x LWS({weight, value})`, matching the stable
carrier-resolved factorization for prefix-dependent scan backward.

The monoid structure declarations are checked during symbolic lowering. Field names,
required inner axes, and dtypes must agree with mapped, lifted, folded, and
projected values.

The interface separates monoid operations from the carrier-resolved
relationships:

```text
Monoid[T]:
    identity: 1 -> T
    combine:  T x T -> T
    lower_fold: optional optimized T[...,F] -> T[...]
    lower_scan: optional optimized (T[...,F], T[...]) -> (T[...,F], T[...])

FoldAlgebra[S,E,A]:
    to_execution: S -> E                         # Phi
    to_semantic:  E -> S                         # Psi
    embed:        E x S* -> A
    finalize:     S x A -> S*                    # commutative
    finalize:     E_prefix x S x A -> S*         # general
```

Every `Monoid` explicitly declares `commutative`. The planner uses the
execution monoid's declaration to decide whether partial carriers may be
permuted; associativity still permits order-preserving regrouping for a
non-commutative monoid. `FoldAlgebra.finalize_context` independently declares
whether `finalize` requires no ordered context or an execution prefix:

```text
FinalizeContext.NONE
FinalizeContext.PREFIX
```

The prefix requirement is not inferred from the Python method signature and
does not define monoid commutativity. Products propagate commutativity only
when every component monoid is commutative, and require prefix context when
any component algebra requires it. Semidirect-product monoids conservatively
declare themselves non-commutative.

These operations construct tensor IR and therefore operate pointwise over
every axis they do not explicitly eliminate. A specialized monoid fold must be
equivalent to repeated `combine`. Unsupported operations fail explicitly
rather than silently selecting a placeholder implementation.

Monoid identities are single algebra elements: their fields contain only the
inner axes declared by `S` or `E`. They do not receive batch or other outer axes.
`Phi`, `Psi`, combine, embed, and finalize are likewise elementwise and simply
preserve the dependencies of their operands. The fold lowering is different:
it receives `fold_axes` because its job is specifically to eliminate those
axes. Any lifting of an identity across outer axes belongs to the surrounding
tensor program, not to the algebra definition.

The current forward compiler needs `to_execution`, `E.fold`, and
`to_semantic`. LWS uses
an LWS semantic monoid, a stable execution-carrier monoid, and another LWS
monoid over `{scalar, value}` as its accumulator. Its `embed` and `finalize` realize
the stable LWS gradient factorization.

### Correctness contract for execution folding

Let `lift: S -> E` be the factorwise semantic-to-execution map (denoted
`Phi` below). An optimized execution lowering must be equivalent to lifting
every factor first and then folding in the execution monoid:

```text
E.fold(to_execution^F(S[...,F])) == fold_E_F(map(Phi, S[...,F]))
```

In mathematical shorthand:

```text
forward region = E.fold o to_execution^F
```

It is explicitly **not**:

```text
Phi o fold_S
```

The distinction is essential when the execution carrier resolves information
destroyed by the semantic operation. For tie-counting max, each local factor
`x` must first become `(maximum=x, count=1)` and those carriers must be folded
in `E`. Computing `max` in `S` first and then lifting it cannot recover the
number of tied maximizers. The optimized lowering may replace the literal
factorwise lift and combine sequence with reductions or contractions, but it
must preserve `fold_E o Phi^F` semantics.

The final semantic fold result is obtained only afterward:

```text
partial carriers = E.fold(to_execution(factor region 1)), ...
combined carrier = fold(E.combine, partial carriers)
semantic result  = Psi(combined carrier)
```

Crucially, an invocation of `E.fold` need not cover the complete logical
fold domain. It lowers an arbitrary factor region to a valid (possibly
partial) `E` value. The same lowering can therefore be used for one tile, many
tiles handled by a loop, a persistent work assignment, or the entire fold.
For non-commutative algebras, combining partial carriers must additionally
preserve logical factor order.

The algebra does not choose among those possibilities. Tiling, loops,
persistence, reduction trees, partial-state materialization, and kernel
boundaries belong to an execution planner. `E.fold` is deliberately
agnostic about all of them. Applying `Psi` before all required partial carriers
have been combined is only valid when the selected region was the complete
fold domain.

### Map/algebra composition and fusion ownership

The user-supplied map is external to the fold algebra. Consequently, fused
operations such as `map_fold_E` or `map_finalize_map*` are not methods of
`FoldAlgebra`. Their canonical forward composition is:

```text
inputs -> map -> S[...,F] -> to_execution -> E[...,F] -> E.fold -> partial E
```

and the canonical local backward composition is:

```text
inputs -> map -> S_local
prefix?, S_local, A -> finalize -> S_local*
inputs, S_local* -> map* -> input cotangents
```

The compiler may recognize either chain and replace it with a fused tensor
program, eliminating the factor or factor-cotangent intermediates. Such a
rewrite belongs to the compiler's optimization/lowering registry because it
crosses the boundary between the user map and the algebra. The algebra owns
only operations over its `S`, `E`, and `A` structures; it neither receives nor
owns a `map_region`.

Like `E.fold` itself, these fused rewrites are region-polymorphic and do
not prescribe staging. A planner may instantiate them over any chosen region
and decide whether their partial carriers or cotangents remain local, are
combined in a loop, or are persisted for later work.

The prototype can already lower the unfused local backward composition as an
independent program:

```python
from cutilereduce.experimental import compile_map_finalize_vjp_region

local_backward = compile_map_finalize_vjp_region(
    spec,
    algebra,
    xentropy_map,
)

print(local_backward.format())
```

This program has `accumulator.*` inputs and performs:

```text
region inputs -> xentropy_map -> local S
local S, accumulator A -> finalize -> local S*
local S* -> VJP(xentropy_map) -> partial input cotangents
```

If `v` denotes the fold axis, an invocation may bind it to a tile-local range.
The context cotangent is then reduced over that local `v` range and is a
partial cotangent to be accumulated with results from other ranges, while the
per-`v` target cotangents retain `v`. Nothing in the resulting IR declares a
tile or kernel boundary: "tile-local" describes one possible planner
instantiation of the region.

The default is the commutative form, with no prefix carrier. A non-commutative
algebra requests an explicit execution prefix for the same region lowering:

```python
local_backward = compile_map_finalize_vjp_region(
    spec,
    Affine(("d",)),
    affine_map,
)
```

The affine algebra's `FinalizeContext.PREFIX` declaration automatically
introduces an exclusive tile-boundary `prefix.*` input. The region scans its
mapped execution carriers to obtain factorwise prefixes for `finalize` and
returns `updated_prefix.*`, the inclusive carrier at the outgoing tile
boundary. Its
accumulator stores the sum of scale cotangents and represents prefix-scale
weighted payload cotangents with an LWS payload. Thus its combine never first
forms `exp(prefix_scale) * payload_cotangent`. `finalize` uses the LWS
log-weight relative to the local prefix, then transports the normalized
payload through the local factor. Prefix-dependent finalize is needed for
this non-commutative case, not for commutative folds.

Both forms expose the composition needed for a future fusion rewrite, but the
rewrite itself remains a compiler optimization rather than a `FoldAlgebra`
method.

## Fold-axis-polymorphic efficient lowering

An algebra lowering inspects symbolic values and is given the fold axes, but it
does not need a batch-axis parameter. Operations preserve every axis they do
not explicitly reduce or contract, so the computation works pointwise over all
remaining dependencies. A lowering should not depend on application-specific
axis names or ranks. The built-in `LogWeightedSum(("dv",))` implements:

```text
maximum[...]     = max_F(log_weight[...,F])
weights[...,F]   = exp(log_weight[...,F] - maximum[...])
normalizer[...]  = sum_F(weights[...,F])
numerator[...,I] = contract_F(weights[...,F], value[...,F,I])
```

Here `...` denotes whatever non-fold dependencies the operands actually have,
not a batch shape supplied to the algebra.

For the attention specification above, the final contraction sees:

```text
weights axes: (h, l, g, r)
value axes:   (h, r, dv)
reduce axes:  (r)
```

It therefore records the following contraction-local partition:

```text
batch:    (h)
lhs free: (l, g)
rhs free: (dv)
reduce:   (r)
hint:     batched_gemm
```

The contraction's `batch` category means axes shared by these two operands and
is distinct from `FoldSpec.batch_axes`. Thus `h` is a GEMM batch axis, while
`l` and `g` are free matrix axes even though all three are outer batch axes of
the fold problem.

The algebra asks for a contraction over the fold role; it does not mention
attention axes or request a matrix multiplication explicitly. The tensor IR
derives the GEMM-shaped operation from axis relationships. Because broadcasting
remains implicit, no value-vector grid over `(l, g, r, dv)` is introduced.

An efficient lowering is expected when:

- the map retains useful contractions rather than expanding them into scalar
  multiply-and-sum expressions;
- algebra lowerings express weighted reductions as contractions;
- values retain their true axis dependencies;
- dtypes for accumulation are selected explicitly; and
- intermediates remain symbolic until an explicit materialization boundary.

The IR records an implementation hint; it does not yet prove that a backend can
realize that hint efficiently. Backend layout selection, tiling, fusion, and
hardware scheduling remain separate later lowering stages.

## Tensor-IR optimization pipeline

The map, monoid operations, algebra relationships, and generated `map*` all
inhabit one tensor graph. Optimizations can therefore recognize patterns that
cross any of those construction boundaries without making the pattern an
operation of `Monoid` or `FoldAlgebra`.

Every compiled program currently runs these passes in order:

1. Constant and identity simplification removes operations such as `x * 1`,
   `x + 0`, `x / 1`, and redundant casts.
2. Dead-code elimination retains only nodes reachable from declared outputs.
3. Contraction discovery rewrites a named-axis sum of products into an
   explicit contraction when every reduced axis occurs in both operands. It
   also recognizes a zero-masked product when the mask can be applied to one
   operand without expanding that operand's axes:

   ```text
   reduce_sum(multiply(lhs, rhs), over=R)
       -> contract(lhs, rhs, over=R)

   reduce_sum(where(mask, multiply(lhs, rhs), 0), over=R)
       -> contract(where(mask, lhs, 0), rhs, over=R)
   ```

   Masks spanning axes exclusive to both operands are left as generic tensor
   expressions, since applying such a mask to either operand would introduce
   an unnecessary expanded intermediate.

4. Contraction canonicalization recomputes the named-axis partition and its
   dot/GEMV/GEMM/batched implementation classification.
5. Materialization planning currently enforces the conservative prototype
   policy that only declared outputs are stored. Choosing internal staging
   boundaries remains the responsibility of a later execution planner.

For LWS backward, contraction discovery turns the logical expression

```text
reduce_sum(
    probability[h,l,g,r] * output_cotangent[h,l,g,dv],
    over=(l,g),
)
```

directly into a batched GEMM-shaped contraction producing `[h,r,dv]`. The
logical `[h,l,g,r,dv]` product consequently disappears from the optimized IR.

### Future: accumulating folds and numerical semantics

An execution planner may repeatedly update an existing carrier with arbitrary
factor regions. Its canonical meaning is already expressible without another
fundamental algebra operation:

```text
updated_E = E.combine(
    state_E,
    E.fold(to_execution(map(region))),
)
```

LWS supplies an optimized fold-accumulate lowering for this composition. It
chooses the combined maximum before forming weights, accumulates the
weighted-value contraction directly into the state numerator, and avoids
constructing a distinct partial carrier. CPU serial online folds and CuTile
loops invoke this shared lowering through `map_fold_combine`. Without a
specialization, its generic expansion remains `combine(state, fold(values))`.
Deriving this optimized update automatically from ordinary `combine` and
`fold` definitions remains future compiler work.

Floating-point tensor expressions require an explicit rewrite policy so that
user maps and algebra lowerings can distinguish expressions that permit
real-algebra transformations from expressions whose evaluation structure must
be preserved. The tensor IR should attach capabilities such as the following
to operations or expression regions:

```text
strict       preserve operation order and grouping
contract     permit recognized multiply/reduce contraction or FMA formation
reassociate  permit real-algebra reassociation and distribution
```

Optimization passes must require the relevant capability before applying
identities such as `x * 1 -> x`, sum-of-products contraction, distribution
through reductions, or reassociation of exponential scales.

For example, a compensated or Kahan-like execution carrier can build its
update from strict tensor operations so that generic simplification and
contraction passes cannot rewrite away the compensation scheme. A regular sum
lowering can explicitly permit contraction or reassociation. This remains
necessary even when both are implementations of the same mathematical
additive monoid.

The current IR does not yet carry these policies; its optimization passes
assume permissive algebraic manipulation. User-defined numerical schemes that
depend on strict evaluation order are therefore outside the current supported
scope.

The current optimizer's rewrites are local. Serial online carrier residency
is explicit in the CPU driver, but general temporary-buffer reuse, in-place
accumulation, inference of optimized online updates, and numerical-mode
selection remain future compiler/backend work.

## Compiling and inspecting a program

```python
from cutilereduce.experimental import LogWeightedSum, compile_fold

program = compile_fold(
    spec,
    LogWeightedSum(("dv",)),
    attention_map,
)

print(program.format())
```

`compile_fold` performs the following steps:

1. Create symbolic input nodes with their actual axis dependencies.
2. Trace `map` to obtain semantic factors retaining their actual dependencies.
3. Apply `to_execution` pointwise to obtain carriers retaining the fold axes.
4. Invoke `execution.fold` with only the fold axes to obtain a partial
   execution carrier in which those axes have been eliminated.
5. Invoke `to_semantic` to recover semantic outputs and materialize their fields.

The current prototype instantiates this symbolic region over a range spanning
the complete declared fold axes. That is a limitation of the current compiler,
not part of the `execution.fold` contract. A subsequent planner-oriented
representation should be able to instantiate the same lowering over arbitrary
regions, combine their partial `E` values, and apply `Psi` only to the combined
carrier.

Inputs and final outputs are currently the only materialization boundaries.
All other nodes describe expressions available for subsequent fusion and
scheduling. `TensorProgram.contractions`, `.reductions`, and
`.materializations` provide structured inspection in addition to `format()`.

## Visible output mapping

The semantic monoid result need not be the public result of a fold. Compilation
accepts an optional `output_function: S -> O`, with identity as the default:

```python
def attention_output(ctx, semantic):
    del ctx
    return StructuredValue({"output": semantic["value"]})

program = compile_fold(
    spec,
    LogWeightedSum(("dv",)),
    attention_map,
    output_function=attention_output,
)
```

The complete forward composition is:

```text
inputs -> map -> S^F -> Phi^F -> E.fold -> E -> Psi -> S
                                                           \
                                                            output_function -> O
```

Backward differentiates the output function with respect to the semantic
result before embedding the output cotangent:

```text
saved E -> Psi -> S
S, O* -> output_function* -> S*
saved E, S* -> embed -> A
```

For attention, only the normalized LWS payload is visible. Consequently,
`output_function*` produces a structural zero for the log-weight cotangent and
the supplied output cotangent for the payload. Structural zeros are explicit
IR values rather than allocated zero tensors, allowing algebra lowering,
simplification, and dead-code elimination to remove unused gradient paths.

The current compiler performs this pruning within each compiled program.
Pruning saved `E` or intermediate `A` fields across separately planned regions
requires inter-region liveness analysis and belongs to the execution planner.

## Commutative fold backward and `map*`

`compile_fold_forward` and `compile_fold_backward` produce separate programs.
The forward program exposes `saved_execution.*` materializations needed by the
backward program. The backward program accepts those carriers, semantic output
cotangents, and the original inputs; it does not contain the forward fold.

The current backward lowering does not differentiate the optimized execution
fold. Instead it composes the two intended differentiation layers:

```text
E_global, S*_global -> algebra.embed -> A
recomputed map inputs -> S_local
S_local, A -> algebra.finalize -> S*_local
VJP(map, S*_local) -> input cotangents
```

The tensor-IR VJP currently handles elementwise arithmetic, `exp`, `log`,
casts, lazy broadcasting, `where`, sum reductions, shared-expression
cotangent accumulation, and contractions. Contraction VJPs are emitted as new
contractions, so the cross-entropy map produces GEMM-shaped gradients for both
input matrices.

`Add` and `LogSumExp` implement the accumulator operations needed by the
cross-entropy product. The shared logits expression receives and adds the
cotangents returned by both component finalizers:

```text
g_logits = exp(logits - logsumexp) * g_normalizer
         + is_target * g_target
```

This local cotangent remains symbolic and feeds the two contraction VJPs; it is
not an output materialization. LWS uses the same construction with an LWS
accumulator monoid. Prefix-dependent folds persist partition-boundary execution
carriers and generated backward threads them through independently chunked
local regions.

## Current scope

Implemented:

- named-axis input and expression types;
- lazy elementwise broadcasting;
- casts, reductions, and binary contractions;
- contraction-axis partitioning and implementation hints;
- flat structured values for individual algebras and recursive composite
  values for algebra constructions;
- independently structured semantic and execution monoids and an accumulator
  partial semigroup;
- monoid-owned identity and combine, optional optimized fold lowerings, and
  generic structured associative fold/scan fallbacks;
- componentwise `ProductAlgebra`, plus `Add` and `LogSumExp` instances;
- a right-oriented `SemidirectProductMonoid` construction and concrete
  prefix-differentiable `LogScaleAffine` instance;
- forward fold lowering and explicit final materialization;
- separate commutative forward and backward compilers with an explicit saved
  execution-state contract;
- algebraic `embed` and `finalize` for `Add`, `LogSumExp`, LWS, and affine;
- reverse-mode logical-map differentiation, including contraction VJPs;
- constant simplification, dead-code elimination, contraction discovery and
  canonicalization, and conservative output materialization passes;
- a stable `LogWeightedSum` lowering polymorphic over fold-axis names, ranks,
  and all untouched dependencies;
- recursive hardware-independent online/tree/partial/scan fold plans;
- concrete CPU scheduling and analytical plan selection;
- generated size-specialized PyTorch C++ extensions for forward and
  prefix-free backward;
- an explicit scheduled-backward SSA form with loop-carried local gradients,
  streamed commits, relaxed atomic adds, and liveness intervals;
- analytical scheduled-memory accounting and separate forward/backward/full
  CPU timing helpers;
- an algebra-component graph which retains SSA producer sharing, Product
  independence, and semidirect actor-to-payload dependencies; splitting a
  shared producer requires an explicit recompute-or-materialize decision;
- uniform named-axis CPU contraction lowering through ATen `mm` and `bmm`,
  with `addmm` and `baddbmm` for accumulated contractions.

Not implemented yet:

- VJP rules beyond the current bounded tensor-IR subset;
- accumulator suffix scans for differentiable scans;
- additional global/algebra-aware optimization passes;
- automatic inference of optimized fold-accumulate lowerings beyond supplied
  monoid specializations;
- explicit carrier ordering/merge capabilities, including compensated sums;
- explicit SIMD lowering beyond compiler auto-vectorization.

The package is intentionally independent of the existing executable fold
implementation while these interfaces are still being tested.
