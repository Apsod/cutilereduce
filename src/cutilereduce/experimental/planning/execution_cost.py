"""Resource accounting from concrete scheduled SSA, independent of backend names."""

from cutilereduce.experimental.ir.cost import DTYPE_BYTES, OperationCost, elements, product
from .accounting import ExecutionScope, ScheduleDomain, ceildiv
from .analysis import RegionInvocation, estimate_execution_stage


def analyze_tiled_stage(problem, stage, *, direction="backward"):
    """Analyze explicit calls, loop-carried additions and commits.

    Uses the existing dense padded upper-bound convention, including idle owners.
    This first visitor covers the scheduled gradient-stage instruction subset.
    Unknown instructions fail instead of silently receiving zero cost.
    """
    partitions = stage.effective_partitions
    sizes = dict(problem.sizes)
    tiles = {p.axis: p.tile for p in partitions}
    dimensions = sizes | tiles
    loop_axes = (stage.loop.axis.axis,) if stage.loop is not None else ()
    domain = ScheduleDomain.from_maps(sizes, tiles=tiles,
        owners={p.axis: p.programs for p in partitions},
        iteration_axes=tuple(a for a in sizes if a in tiles), loop_axes=loop_axes,
        scope=ExecutionScope("backward_owner" if stage.loop is not None else "embed_owner"))
    owners = product(p.programs for p in partitions)
    steps = (ceildiv(stage.loop.axis.count, next(p.programs for p in partitions
             if p.axis == stage.loop.axis.axis)) if stage.loop is not None else 1)
    work = OperationCost()
    reads = writes = atomics = resident = persistent = allocated = 0
    invocations = []

    def nbytes(value, extents=dimensions):
        return elements(value.signature.axes, extents) * DTYPE_BYTES[value.signature.dtype]

    def visit(operation, repetitions):
        nonlocal work, reads, writes, atomics, resident, persistent, allocated
        if operation.op == "call_region":
            name = dict(operation.attributes)["region"]
            cost = problem.bound_region(name, dimensions)
            work += cost.work.scale(repetitions)
            invocations.append(RegionInvocation(name, cost, repetitions, owners))
            reads += sum(domain.buffer(value).padded_accessed_bytes
                         for _, value in problem.facts[name].arguments)
            resident += cost.live_bytes_upper_bound + sum(v for _, v in cost.argument_bytes)
        elif operation.op == "add":
            value, = operation.outputs
            work += OperationCost(arithmetic=elements(value.signature.axes, dimensions) * repetitions)
        elif operation.op == "local_zero":
            persistent += sum(nbytes(v) for v in operation.outputs)
        elif operation.op in {"store", "atomic_add_relaxed"}:
            value, = operation.inputs
            amount = nbytes(value) * repetitions
            if operation.op == "store":
                writes += amount
            else:
                atomics += amount
            # Each destination owns one full gradient allocation, independent
            # of whether its commits occur within or after the serial loop.
            allocated += nbytes(value, sizes)
        else:
            raise NotImplementedError(f"no scheduled cost rule for {operation.op!r}")

    for operation in stage.initialization:
        visit(operation, owners)
    if stage.loop is not None:
        for operation in stage.loop.body:
            visit(operation, owners * steps)
    for operation in (*stage.epilogue, *stage.commits):
        visit(operation, owners)
    return estimate_execution_stage(problem.target, stage.name, direction, work,
        reads, writes, atomics, resident + persistent, allocated, owners, steps,
        tuple(invocations), domain)


def analyze_implementation(problem, implementation, baseline=None):
    """Derive costs from the backend's concrete shared execution-IR root."""
    from .backend import BoundExecution, TiledExecution
    execution = implementation.execution
    if isinstance(execution, BoundExecution):
        return analyze_bound_plan(problem, execution.forward, backward=execution.backward,
                                  gradient_stage=execution.gradient_stage)
    if isinstance(execution, TiledExecution):
        return analyze_online_plan(problem, execution.forward, execution.backward)
    raise NotImplementedError("backend did not provide supported execution IR")


def analyze_bound_plan(problem, forward, *, backward=None, gradient_stage=None):
    """Cost the bound recursive fold IR also consumed by native code generation.

    Keeps the established padded-domain/linear-combine convention. Topologies,
    partitioning and tile bounds are read from bound loops, never family names.
    """
    from cutilereduce.core import symbolic as sym
    from .bound import BoundLoop, BoundLeaf, BoundLoopKind
    from .templates import FoldTopology, RegionCall, MapFoldRegion, Reference
    from .analysis import ScheduleCost

    sizes = dict(forward.sizes)
    axis = problem.program.spec.fold_axes[0]
    batch = problem.program.spec.batch_axes
    first = forward.forward[0]
    root = first.bindings[0].expression
    partition = root if isinstance(root, BoundLoop) and root.operation == "map" else None
    fold = partition.body if partition is not None else root
    if not isinstance(fold, BoundLoop) or fold.operation != "carrier_fold":
        raise NotImplementedError("expected a bound carrier-fold execution")
    if not isinstance(fold.body, BoundLeaf) or not isinstance(fold.body.expression, MapFoldRegion):
        raise NotImplementedError("nested carrier-fold execution requires a cost rule")
    tiles = {a: dict(first.tile_sizes)[a] for a in (*batch, axis)}
    tiles[axis] = fold.iteration.grain
    dims = sizes | tiles
    partitions = partition.iteration.count if partition is not None else 1
    chunks = ceildiv(ceildiv(sizes[axis], partitions), fold.iteration.grain)
    owners = product(ceildiv(sizes[a], tiles[a]) for a in batch)
    online = fold.topology is FoldTopology.ONLINE and fold.iteration.loop is not BoundLoopKind.PARALLEL
    domain = ScheduleDomain.from_maps(sizes, tiles=tiles,
        owners={**{a: ceildiv(sizes[a], tiles[a]) for a in batch}, axis: partitions},
        iteration_axes=(*batch, axis), loop_axes=(axis,), scope=ExecutionScope("forward_owner"))
    stages = []

    def region(name, extents=dims):
        return problem.bound_region(name, extents)

    def outputs(name, extents=dims):
        return sum(value for _, value in region(name, extents).result_bytes)

    def emit(name, work, read, write, resident, allocated, tasks, steps=1, calls=(), accounting=None, direction="forward"):
        stages.append(estimate_execution_stage(problem.target, name, direction, work,
            read, write, 0, resident, allocated, tasks, steps, calls, accounting))

    local = region("map_fold_combine" if online else "map_fold")
    count = owners * partitions * chunks
    tasks = owners * partitions * (1 if online else chunks)
    calls = [RegionInvocation("map_fold_combine" if online else "map_fold", local, count, tasks)]
    work = local.work.scale(count)
    if not online:
        combines = owners * sym.SaturateSub(chunks, 1)
        combine = region("execution_combine")
        work += combine.work.scale(combines)
        calls.append(RegionInvocation("execution_combine", combine, combines, owners))
    inputs = [domain.buffer(value) for _, value in problem.facts["map_fold"].arguments]
    carrier = outputs("map_fold", sizes)
    local_carrier = outputs("map_fold")
    partials = carrier * partitions if partition is not None else 0
    scratch = carrier * chunks if not online else 0
    emit("map_fold", work, sum(v.padded_accessed_bytes for v in inputs), partials or carrier,
         local.live_bytes_upper_bound + sum(v.tile_bytes for v in inputs) + 2 * local_carrier,
         partials or carrier, tasks, chunks if online else 1, tuple(calls), domain)
    checkpoints = 0
    for stage in forward.forward[1:]:
        expressions = tuple(binding.expression for binding in stage.bindings)
        scans = any(isinstance(e, BoundLoop) and e.operation in {"exclusive_scan", "inclusive_scan"}
                    for e in expressions)
        projection = any(isinstance(e, BoundLeaf) and isinstance(e.expression, RegionCall)
                         and e.expression.region == "project_output" for e in expressions)
        if projection:
            # Native emission runs projection within the same output owner tile.
            local_dims = dims
            project = region("project_output", local_dims)
            project_owners = product(ceildiv(sizes[a], local_dims[a]) for a in batch)
            project_domain = ScheduleDomain.from_maps(sizes, tiles={a: local_dims[a] for a in tiles},
                owners={a: ceildiv(sizes[a], local_dims[a]) for a in batch},
                iteration_axes=batch, scope=ExecutionScope("project_owner"))
            emit("project_output", project.work.scale(project_owners), carrier,
                 outputs("project_output", sizes), project.live_bytes_upper_bound + outputs("map_fold", local_dims),
                 outputs("project_output", sizes), project_owners,
                 calls=(RegionInvocation("project_output", project, project_owners, project_owners),), accounting=project_domain)
        elif partials and (scans or any(isinstance(e, BoundLoop) and e.operation == "carrier_fold"
             and isinstance(e.body, BoundLeaf) and isinstance(e.body.expression, Reference) for e in expressions)):
            checkpoints = carrier * partitions if scans and backward is not None else 0
            combine = region("execution_combine", sizes)
            count = sym.SaturateSub(partitions, 1)
            emit("scan_partials" if checkpoints else "combine_partials", combine.work.scale(count),
                 partials, carrier + checkpoints, region("execution_combine").live_bytes_upper_bound + 2 * local_carrier,
                 carrier + checkpoints, owners, partitions,
                 (RegionInvocation("execution_combine", combine, count, owners),))
        else:
            raise NotImplementedError(f"no execution accounting rule for bound stage {stage.name!r}")
    saved = carrier + checkpoints if backward is not None else 0
    auxiliary = sym.Max(partials + carrier + checkpoints, scratch + carrier)
    if backward is not None:
        embed_stage = backward.backward[0]
        call = embed_stage.bindings[0].expression
        if not isinstance(call, BoundLeaf) or not isinstance(call.expression, RegionCall):
            raise NotImplementedError("expected a bound embed region call")
        embed = region(call.expression.region, sizes)
        accumulator = outputs(call.expression.region, sizes)
        embed_domain = ScheduleDomain.from_maps(sizes, tiles=sizes,
            owners={a: 1 for a in batch}, iteration_axes=batch, scope=ExecutionScope("embed_owner"))
        emit("embed_output_vjp", embed.work, sum(v for _, v in embed.argument_bytes), accumulator,
             embed.live_bytes_upper_bound + sum(v for _, v in embed.argument_bytes), accumulator, 1,
             calls=(RegionInvocation(call.expression.region, embed, 1, 1),), accounting=embed_domain, direction="backward")
        if gradient_stage is None:
            raise NotImplementedError("a realized gradient stage is required")
        gradient = analyze_tiled_stage(problem, gradient_stage)
        from dataclasses import replace
        stages.append(replace(gradient, name="map_finalize_map_vjp"))
        auxiliary = sym.Max(auxiliary, saved + accumulator)
    return ScheduleCost(tuple(stages), auxiliary, saved, "IR-derived bound execution; CPU physical region refinements")


def analyze_online_plan(problem, forward, backward=None):
    """Analyze a fused online fold DAG and its explicit saved-state buffers."""
    from cutilereduce.core import symbolic as sym
    from .analysis import ScheduleCost
    from .ssa import TiledPlan
    plan = forward.implementation
    if not isinstance(plan, TiledPlan) or len(plan.stages) != 1:
        raise NotImplementedError("expected a single online tiled stage")
    stage, = plan.stages
    if stage.loop is None:
        raise NotImplementedError("online stage requires a serial loop")
    sizes = dict(forward.shape)
    partitions = stage.effective_partitions
    tiles = {p.axis: p.tile for p in partitions}
    dims = sizes | tiles
    domain = ScheduleDomain.from_maps(sizes, tiles=tiles,
        owners={p.axis: p.programs for p in partitions},
        iteration_axes=(*problem.program.spec.batch_axes, *problem.program.spec.fold_axes),
        loop_axes=(stage.loop.axis.axis,), scope=ExecutionScope("forward_owner"))
    owners = product(p.programs for p in partitions)
    steps = stage.loop.axis.count
    calls = tuple(o for o in stage.loop.body if o.op == "call_region")
    if len(calls) != 1 or any(o.op != "call_region" for o in stage.loop.body):
        raise NotImplementedError("online body must contain one region call")
    local = problem.bound_region(dict(calls[0].attributes)["region"], dims)
    project_call, = stage.epilogue
    project = problem.bound_region(dict(project_call.attributes)["region"], dims)
    work = local.work.scale(owners * steps) + project.work.scale(owners)
    input_uses = [domain.buffer(v) for _, v in problem.facts["map_fold"].arguments]
    carrier = problem.bound_region("map_fold", dims)
    local_carrier = sum(v for _, v in carrier.result_bytes)
    resident = sym.Max(local.live_bytes_upper_bound + sum(v.tile_bytes for v in input_uses) + 2 * local_carrier,
                       project.live_bytes_upper_bound + local_carrier)
    writes = sum(elements(op.inputs[0].signature.axes, sizes) * DTYPE_BYTES[op.inputs[0].signature.dtype]
                 for op in stage.commits)
    saved = sum(elements(signature.axes, sizes) * DTYPE_BYTES[signature.dtype]
                for buffer in plan.buffers for _, signature in buffer.signatures)
    first = estimate_execution_stage(problem.target, "map_fold_online", "forward", work,
        sum(v.padded_accessed_bytes for v in input_uses), writes + saved,
        resident=resident, allocated=writes + saved, tasks=owners, steps=steps, domain=domain)
    stages = [first]
    auxiliary = saved
    if backward is not None:
        embed_stage, final_stage = backward.implementation.stages
        embed = analyze_tiled_stage(problem, embed_stage)
        # The embed call has no loop; retain the established live-SSA convention.
        from dataclasses import replace
        stages.extend((replace(embed, name="embed_output_vjp"),
                       replace(analyze_tiled_stage(problem, final_stage), name="map_finalize_map_vjp")))
        auxiliary = sym.Max(auxiliary, saved + embed.allocated_bytes)
    return ScheduleCost(tuple(stages), auxiliary, saved, "IR-derived tiled execution; generic physical estimates")
