# Region facts and general schedule accounting

This layer provides structural facts and logical counts before applying CPU/GPU
performance models. It lifts the legacy domain and buffer accounting without
requiring user-supplied matrix-work descriptions or intermediate-buffer specs.

## Facts derived from the DSL

`FoldProgram.region_facts` is a cached, read-only mapping from region name to
`RegionFacts`. It contains:

- every SSA value's named axes, dtype, symbolic size, and transitive formal-input
  sources;
- named formal arguments and results;
- each contraction's batch, left-free, right-free and reduction axis partitions,
  precision contract, symbolic B/M/N/K cardinalities and FLOPs;
- the existing logical operation-cost summary.

Axes describe the dependencies/shapes represented by optimized IR; the analysis
does not prove arbitrary expressions constant. Formal-input sources describe
primal dataflow, including through `stop_gradient`, rather than derivative paths.
Contraction cardinalities do not prescribe physical flattening, layout or MMA
instructions. In particular, `l*g` can be the logical M cardinality while `l`
and `g` remain distinct scheduling/dependency axes.

```python
program = operator.lower()
facts = program.region_facts["map_fold"]
qk = facts.contractions[0]
print(qk.axes)
print(qk.dimensions())        # symbolic B/M/N/K
print(qk.flops())
print(facts.argument("key").axes)
```

## Domain and scope quantities

`ScheduleDomain` gives each named axis an `AxisDomain(extent, tile, owners)`.
`owners` counts independent assignments along that axis at the scope being
examined. Quantities remain symbolic and support partial substitution:

```text
num_tiles  = ceil(extent / tile)
span_tiles = ceil(num_tiles / owners)
span       = span_tiles * tile
```

Spans are maximum padded per-owner spans, not exact spans for every owner.
`active_owners` accounts for idle assignments. `Measure` from `core.measure` is
reused: its legacy `programs` spelling means owners at this boundary, while
`tiles_per_program` means maximum tiles per owner. No GPU program semantics are
required by these formulas.

`iteration_axes` declares which axes can be partitioned among owners;
`loop_axes` identifies serial traversal within each owner. Every serially
traversed axis must be declared for the padded traffic formula to describe a
complete execution. Inner dimensions can be included in iteration_axes by a
future planner; the accounting itself does not impose legacy tile residency.

`ExecutionScope(name, parent)` supplies a boundary label and ancestry. For
example, a target may name cluster/node/device or device/CTA/warp boundaries.
Domain extents are explicit at each boundary: a child can use its parent's
maximum span as its extent when that describes the selected decomposition.
The scope object does not implement hierarchical scheduling, communication,
resource capacities, or synchronization.

## Dependencies, replicas and contribution ownership

For a `ValueFacts`, `domain.buffer(value)` exposes:

- full, tile, and maximum per-owner span bytes;
- persistent versus streamed status relative to the declared local loops;
- owner partitions addressing distinct destinations;
- owner replication from iteration axes absent from the value;
- missing local-loop axes requiring local accumulation;
- logical access volume and padded tile-loop access bounds.

With `P[a]` owners along each axis, a value carrying axes `V` has:

```text
owners_per_target = product(P[a] for partitioned a absent from V)
logical_accessed_bytes = full_value_bytes * owners_per_target
padded_accessed_bytes = tile_value_bytes * total_owners
                       * product(span_tiles[a] for local-loop a present in V)
```

Loop-invariant values are assumed retained at the specified owner boundary.
Owner counts include assigned idle owners; logical accesses are not an optimized
physical traffic minimum. `replicated_bytes` counts extra owner replicas, not
network-link transfers. Cache hits, multicast, packing, physical layout padding,
transaction sizes, and collective algorithms belong to later realization/models.

`ContributionOwnership` applies when the enclosing operation declares values to
be partial contributions (cotangents or partial monoid carriers). Replicated
primal values do not become additive merely because an axis is absent; use
`owner_replicas` for their multiplicity. The operation supplies the combination
semantics. Contribution ownership chooses neither atomics nor a reduction tree/collective. The same absent-axis calculation
applies whether the contributors are threads or nodes.

## Integration

The analytical planner now uses these domains and facts for forward/backward
input tile sizes, traffic bounds, persistent gradients, and contributor counts.
Each stage exposes `accounting_domain`, including in JSON inspection reports.
CPU realization analysis obtains contraction dimensions and value sizes from
`RegionFacts`. Backend timing models still own cache/throughput/latency priors.

`ScheduledBackwardProgram.contributions` exposes the generic requirements.
Existing CPU/cuTile executors retain their store/relaxed-atomic implementation
through `resolve_writes`. `resolve_writes(program, shared_commit=None)` leaves
shared reductions unresolved for another implementation to select. Requesting a
plain store for multiple contributors is rejected.

Tests use nontrivial `h,l,g,r`, compare every domain measure and buffer
multiplicity against the legacy accounting, and compare generic contributor
sets with scheduled write ownership. Equal tile volumes are explicitly tested:
head/query tile trades change key/value residency, and `l/g` trades can change
replication through tails even when GEMM M is identical.

Run the target-free worked example with:

```console
uv run -m examples.experimental_accounting
```

This delivers the general accounting foundation. Physical register allocation,
GPU occupancy/pipelining heuristics, hierarchical schedule generation, and
communication algorithms remain backend/planner work.
