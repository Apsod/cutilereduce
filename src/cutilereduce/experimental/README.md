# Experimental algebra-driven fold IR

This package explores a user interface in which a fold is described by its
logical tensor interactions and its algebra, rather than by handwritten fused
tile callbacks. Its output is a target-independent tensor program: it makes
contractions, reductions, surviving axes, dtypes, and materialization
boundaries explicit, but does not emit CUDA Tile or PyTorch code.

The experiment separates three concerns:

- `FoldSpec` describes input tensors, identifies the axes that the algebra must
  fold, and records batch axes as user intent for validation and later planning.
- `Monoid` owns one structured value, its identity and combine operations, and
  an optional axis-polymorphic fold lowering. `FoldAlgebra` relates three such
  monoids: semantic `S`, execution `E`, and adjoint `A`.
- `TensorBuilder` records the user's map and the algebra lowering as an
  axis-aware tensor IR.

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
are not passed to `Monoid.lower_fold`. The current IR preserves input axis order but
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

## Specifying a fold algebra

A fold algebra contains three monoids:

- `semantic` (`S`): factors produced by the map and semantic fold results;
- `execution` (`E`): the numerically or operationally useful reduction
  carrier;
- `adjoint` (`A`): the accumulator used by the eventual backward lowering.

Each monoid owns its structure, identity, combine, and fold lowering. Each
`TensorField` declares only its inner axes. Values retain all of their actual
named-axis dependencies, while `lower_fold` is told only which axes it must
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

    def lower_identity(self, ctx):
        ...

    def lower_combine(self, ctx, left, right):
        ...

    def lower_fold(self, ctx, values, *, fold_axes):
        # May use reductions, contractions, scans, and elementwise operations.
        ...
```

`FoldAlgebra` then owns only `Phi: S -> E`, `Psi: E -> S`,
`embed: E x S* -> A`, and `finalize`, while its `semantic`, `execution`, and
`adjoint` attributes are independently typed `Monoid` instances.

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

`Product` is the concise public constructor for `ProductAlgebra`. Its three
`ProductMonoid`s derive identities, combines, and folds componentwise, while
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

The generic algebra construction cannot infer an efficient fold or a resolved adjoint
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

The monoid structure declarations are checked at tracing time. Field names,
required inner axes, and dtypes must agree with mapped, lifted, folded, and
projected values.

The interface separates monoid operations from the carrier-resolved
relationships:

```text
Monoid[T]:
    lower_identity: 1 -> T
    lower_combine:  T x T -> T
    lower_fold:     T[...,F] -> T[...]

FoldAlgebra[S,E,A]:
    lower_phi:      S -> E
    lower_psi:      E -> S
    lower_embed:    E x S* -> A
    lower_finalize: S x A -> S*                 # commutative
    lower_finalize: E_prefix x S x A -> S*      # general
```

These operations construct tensor IR and therefore operate pointwise over
every axis they do not explicitly eliminate. A specialized monoid fold must be
equivalent to repeated `lower_combine`. Unsupported operations fail explicitly
rather than silently selecting a placeholder implementation.

Identities are single algebra elements: their fields contain only the inner
axes declared by `E` or `A`. They do not receive batch or other outer axes.
`Phi`, `Psi`, combine, embed, and finalize are likewise elementwise and simply
preserve the dependencies of their operands. The fold lowering is different:
it receives `fold_axes` because its job is specifically to eliminate those
axes. Any lifting of an identity across outer axes belongs to the surrounding
tensor program, not to the algebra definition.

The current forward compiler needs `Phi`, `E.lower_fold`, and `Psi`. LWS uses
an LWS semantic monoid, a stable execution-carrier monoid, and another LWS
monoid over `{scalar, value}` as its adjoint. Its `embed` and `finalize` realize
the stable LWS gradient factorization.

### Correctness contract for execution folding

Let `lift: S -> E` be the factorwise semantic-to-execution map (denoted
`Phi` below). An optimized execution lowering must be equivalent to lifting
every factor first and then folding in the execution monoid:

```text
E.lower_fold(Phi^F(S[...,F])) == fold_E_F(map(Phi, S[...,F]))
```

In mathematical shorthand:

```text
forward region = E.lower_fold o Phi^F
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
partial carriers = E.lower_fold(Phi(factor region 1)), ...
combined carrier = fold(E.combine, partial carriers)
semantic result  = Psi(combined carrier)
```

Crucially, an invocation of `E.lower_fold` need not cover the complete logical
fold domain. It lowers an arbitrary factor region to a valid (possibly
partial) `E` value. The same lowering can therefore be used for one tile, many
tiles handled by a loop, a persistent work assignment, or the entire fold.
For non-commutative algebras, combining partial carriers must additionally
preserve logical factor order.

The algebra does not choose among those possibilities. Tiling, loops,
persistence, reduction trees, partial-state materialization, and kernel
boundaries belong to an execution planner. `E.lower_fold` is deliberately
agnostic about all of them. Applying `Psi` before all required partial carriers
have been combined is only valid when the selected region was the complete
fold domain.

### Map/algebra composition and fusion ownership

The user-supplied map is external to the fold algebra. Consequently, fused
operations such as `map_fold_E` or `map_finalize_map*` are not methods of
`FoldAlgebra`. Their canonical forward composition is:

```text
inputs -> map -> S[...,F] -> Phi -> E[...,F] -> E.lower_fold -> partial E
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

Like `E.lower_fold` itself, these fused rewrites are region-polymorphic and do
not prescribe staging. A planner may instantiate them over any chosen region
and decide whether their partial carriers or cotangents remain local, are
combined in a loop, or are persisted for later work.

The prototype can already trace the unfused local backward composition as an
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

This program has `adjoint.*` inputs and performs:

```text
region inputs -> xentropy_map -> local S
local S, adjoint A -> finalize -> local S*
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
    with_prefix=True,
)
```

For affine composition this introduces `prefix.acting.value` and
`prefix.payload.value` inputs. Given global adjoint state containing the total
scale and output cotangents, `finalize` reconstructs the suffix scale from the
total, prefix, and local scales and produces the local scale and payload
cotangents. Prefix-dependent finalize is needed for this general
non-commutative case, not for commutative folds.

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
   explicit contraction when every reduced axis occurs in both operands:

   ```text
   reduce_sum(multiply(lhs, rhs), over=R)
       -> contract(lhs, rhs, over=R)
   ```

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
    E.lower_fold(Phi(map(region))),
)
```

For LWS, a future compiler rewrite can recognize this complete composition and
lower it as an online-softmax update. In particular, it can choose the combined
maximum before forming weights, accumulate the weighted-value contraction
directly into the state numerator, and avoid constructing a distinct partial
carrier. This is a derived compiler lowering target with an exact expansion to
`combine` and `fold`, not a required fourth `Monoid` operation.

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

The current optimizer deliberately stops short of these transformations. Its
rewrites are local, and internal state residency, in-place accumulation,
online-fold fusion, and numerical-mode selection remain future planner/backend
work.

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
3. Apply `Phi` pointwise to obtain execution carriers retaining the fold axes.
4. Invoke `execution.lower_fold` with only the fold axes to obtain a partial
   execution carrier in which those axes have been eliminated.
5. Invoke `Psi` to recover semantic outputs and materialize their fields.

The current prototype instantiates this trace over a single region spanning
the complete declared fold axes. That is a limitation of the current compiler,
not part of the `execution.lower_fold` contract. A subsequent planner-oriented
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

`Add` and `LogSumExp` implement the adjoint operations needed by the
cross-entropy product. The shared logits expression receives and adds the
cotangents returned by both component finalizers:

```text
g_logits = exp(logits - logsumexp) * g_normalizer
         + is_target * g_target
```

This local cotangent remains symbolic and feeds the two contraction VJPs; it is
not an output materialization. LWS uses the same construction with an LWS
adjoint monoid. General planner construction of prefix context and concrete
backward scheduling remain future work.

## Current scope

Implemented:

- named-axis input and expression types;
- lazy elementwise broadcasting;
- casts, reductions, and binary contractions;
- contraction-axis partitioning and implementation hints;
- flat structured values for individual algebras and recursive composite
  values for algebra constructions;
- independently structured semantic, execution, and adjoint monoids;
- monoid-owned identity, combine, and optimized region-fold lowering;
- componentwise `ProductAlgebra`, plus `Add` and `LogSumExp` instances;
- a right-oriented `SemidirectProductMonoid` construction and concrete
  prefix-differentiable `LogScaleAffine` instance;
- forward fold tracing and explicit final materialization;
- separate commutative forward and backward compilers with an explicit saved
  execution-state contract;
- algebraic `embed` and `finalize` for `Add`, `LogSumExp`, LWS, and affine;
- reverse-mode logical-map differentiation, including contraction VJPs;
- constant simplification, dead-code elimination, contraction discovery and
  canonicalization, and conservative output materialization passes;
- a stable `LogWeightedSum` lowering polymorphic over fold-axis names, ranks,
  and all untouched dependencies.

Not implemented yet:

- scalar algebra definitions and a generic combine-based fallback;
- generic structured ordered-fold regions and a combine-derived fallback;
- execution-carrier combination across partitions;
- VJP rules beyond the current bounded tensor-IR subset;
- planner construction of prefix context for arbitrary non-commutative folds;
- adjoint suffix scans for differentiable scans;
- additional global/algebra-aware optimization passes;
- accumulating-fold recognition and online carrier-update lowering;
- explicit carrier ordering/merge capabilities, including compensated sums;
- concrete layouts, tile sizes, code generation, or execution.

The package is intentionally independent of the existing executable fold
implementation while these interfaces are still being tested.
