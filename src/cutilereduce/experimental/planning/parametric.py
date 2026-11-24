"""Parameterize shared execution IR within a fixed structural decision class."""

from dataclasses import replace

from cutilereduce.core import symbolic as sym
from .accounting import ceildiv
from .backend import BoundExecution, TiledExecution
from .bound import BoundLoop
from .execution_cost import analyze_implementation


def schedule_class(problem, candidate):
    """Group only decisions that preserve operations and gradient commit modes.

    Numeric geometry remains symbolic. Missing result axes with multiple owners
    determine whether each result uses store or atomic commits. Backends with
    additional data-dependent algorithm choices must supply their own classifier.
    """
    p = dict(candidate.parameters)
    axes = (*problem.program.spec.batch_axes, *problem.program.spec.fold_axes)
    loop = candidate.loop_axis or problem.program.spec.fold_axes[0]
    commits = ()
    if candidate.backward:
        name = "map_finalize_map_vjp_accumulate" if problem.program.requires_prefix else "map_finalize_map_vjp"
        owners = {a: (p.get("partition_count", 1) if problem.program.requires_prefix else p.get("loop_programs", 1))
                  if a == loop else ceildiv(problem.sizes[a], min(problem.sizes[a],
                      p.get(f"backward_tile.{a}", p[f"tile.{a}"]))) for a in axes}
        commits = tuple(any(owners[a] > 1 for a in axes if a not in value.axes)
                        for name, value in problem.facts[name].results if name.startswith("partial_grad_input."))
    extras = tuple((k, p[k]) for k in problem.target.extra_parameter_domains if k in p)
    singleton = tuple((a, p.get(f"backward_tile.{a}", p[f"tile.{a}"]) == 1)
                      for a in axes if problem.sizes[a] == 1)
    return candidate.family, candidate.backward, loop, commits, extras, singleton


def parameter_values(problem, candidate):
    p = dict(candidate.parameters)
    for a in (*problem.program.spec.batch_axes, *problem.program.spec.fold_axes):
        p[f"backward_tile.{a}"] = p.get(f"backward_tile.{a}", p[f"tile.{a}"])
        p[f"tile.{a}"] = min(problem.sizes[a], p[f"tile.{a}"])
        if a in problem.program.spec.batch_axes:
            p[f"embed_tile.{a}"] = min(problem.sizes[a], p.get(f"embed_tile.{a}", p[f"tile.{a}"]))
    p.setdefault("partition_count", 1)
    p.setdefault("loop_programs", 1)
    return p


def parameterize(problem, candidate):
    """Realize once, lift geometry to symbols, then use the regular IR visitor."""
    implementation = problem.target.realization_factory(problem, candidate)
    axes = (*problem.program.spec.batch_axes, *problem.program.spec.fold_axes)
    values = parameter_values(problem, candidate)
    def tile(axis, prefix):
        if problem.sizes[axis] == 1 and values.get(f"{prefix}.{axis}", 1) == 1:
            return 1
        return sym.Symbol(f"{prefix}.{axis}") if axis in axes else problem.sizes[axis]

    def bound(plan, prefix):
        if plan is None:
            return None
        def expression(node):
            if not isinstance(node, BoundLoop):
                return node
            iteration = node.iteration
            axis = iteration.logical.axis
            if iteration.logical.kind == "partitions":
                count = sym.Symbol("partition_count")
                grain = ceildiv(problem.sizes[axis], count)
            else:
                grain = tile(axis, prefix)
                count = ceildiv(problem.sizes[axis], grain)
            return replace(node, iteration=replace(iteration, grain=grain, count=count), body=expression(node.body))
        def stage(s):
            return replace(s, tile_sizes=tuple((a, tile(a, prefix)) for a, _ in s.tile_sizes),
                           bindings=tuple(replace(b, expression=expression(b.expression)) for b in s.bindings))
        return replace(plan, forward=tuple(map(stage, plan.forward)), backward=tuple(map(stage, plan.backward)))

    def tiled(stage, prefix):
        if stage is None:
            return None
        def owners(p):
            if stage.loop is not None and p.axis == stage.loop.axis.axis:
                if prefix == "tile":
                    return 1
                return sym.Symbol("partition_count" if problem.program.requires_prefix else "loop_programs")
            return ceildiv(p.extent, tile(p.axis, prefix))
        partitions = tuple(replace(p, tile=tile(p.axis, prefix), programs=owners(p))
                           for p in stage.effective_partitions)
        return replace(stage,
            parallel_axes=tuple(replace(a, tile=tile(a.axis, prefix)) for a in stage.parallel_axes),
            partitions=partitions,
            loop=replace(stage.loop, axis=replace(stage.loop.axis, tile=tile(stage.loop.axis.axis, prefix)))
                 if stage.loop is not None else None)

    execution = implementation.execution
    if isinstance(execution, BoundExecution):
        symbolic = BoundExecution(bound(execution.forward, "tile"), bound(execution.backward, "backward_tile"),
                                  tiled(execution.gradient_stage, "backward_tile"))
    elif isinstance(execution, TiledExecution):
        f = execution.forward
        b = execution.backward
        f = replace(f, implementation=replace(f.implementation,
            stages=tuple(tiled(s, "tile") for s in f.implementation.stages)))
        if b is not None:
            b = replace(b, implementation=replace(b.implementation,
                stages=tuple(tiled(s, "embed_tile" if s.loop is None else "backward_tile")
                             for s in b.implementation.stages)))
        symbolic = TiledExecution(f, b)
    else:
        raise NotImplementedError("no parametric lowering for this execution IR")
    return analyze_implementation(problem, replace(implementation, execution=symbolic))
