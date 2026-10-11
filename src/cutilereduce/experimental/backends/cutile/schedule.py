"""Specialize abstract forward and backward schedules for the CuTile target."""

from __future__ import annotations

import math
from dataclasses import replace

from cutilereduce.experimental.ir.core import TensorSignature
from cutilereduce.experimental.compiler.operator import FoldProgram
from cutilereduce.experimental.planning.templates import ForwardScheduleTemplate
from cutilereduce.experimental.planning.target import TargetModel
from cutilereduce.experimental.planning.scheduled import (
    ScheduledAxis,
    TileBinding,
    schedule_backward_program,
)
from cutilereduce.experimental.planning.ssa import (
    AxisPartition,
    LoopCarry,
    ScheduledLoop,
    ScheduledOperation,
    ScheduledValue,
    StageDependency,
    TiledBuffer,
    TiledPlan,
    TiledStage,
    lower_backward_stage,
    SpecializedSchedule,
)


def _build_tiled_stage(
    problem: FoldProgram,
    sizes: dict[str, int],
    *,
    tiles: dict[str, int],
) -> TiledStage:
    """Build the initial single-owner, serial-fold schedule topology."""
    if set(sizes) != set(problem.spec.axes):
        raise ValueError("sizes must define every logical axis exactly once")
    owner_names = problem.spec.batch_axes
    fold_names = problem.spec.fold_axes
    if len(fold_names) != 1:
        raise NotImplementedError("the first tiled stage supports one fold axis")

    def axis(name: str) -> ScheduledAxis:
        return ScheduledAxis(
            name, min(tiles.get(name, sizes[name]), sizes[name]), sizes[name]
        )

    scheduled = (*owner_names, *fold_names)
    arguments = tuple(
        TileBinding(
            name,
            spec.axes,
            spec.dtype,
            tuple(value for value in scheduled if value in spec.axes),
        )
        for name, spec in problem.spec.inputs.items()
    )
    next_id = 0

    def value(name: str, signature: TensorSignature) -> ScheduledValue:
        nonlocal next_id
        result = ScheduledValue(next_id, name, signature)
        next_id += 1
        return result

    parameters = tuple(
        value(binding.name, TensorSignature(binding.axes, binding.dtype))
        for binding in arguments
    )
    identity_outputs = tuple(
        value(
            f"initial.{name}",
            TensorSignature(
                problem.execution_identity.nodes[node_id].axes,
                problem.execution_identity.nodes[node_id].dtype,
            ),
        )
        for name, node_id in problem.execution_identity.results
    )
    initialization = (
        ScheduledOperation(
            "call_region",
            (),
            identity_outputs,
            (("region", "execution_identity"),),
        ),
    )
    accumulator_arguments = tuple(
        (name, node_id)
        for name, node_id in problem.map_fold_combine.arguments
        if name.startswith("accumulator.")
    )
    carry_arguments = tuple(
        value(
            f"iter.{name.removeprefix('accumulator.')}",
            TensorSignature(
                problem.map_fold_combine.nodes[node_id].axes,
                problem.map_fold_combine.nodes[node_id].dtype,
            ),
        )
        for name, node_id in accumulator_arguments
    )
    next_state = tuple(
        value(
            f"next.{name}",
            TensorSignature(
                problem.map_fold_combine.nodes[node_id].axes,
                problem.map_fold_combine.nodes[node_id].dtype,
            ),
        )
        for name, node_id in problem.map_fold_combine.results
    )
    final_state = tuple(
        value(f"final.{name}", item.signature)
        for name, item in zip(
            (name for name, _ in problem.execution_identity.results),
            identity_outputs,
            strict=True,
        )
    )
    loop = ScheduledLoop(
        axis(fold_names[0]),
        tuple(
            LoopCarry(initial, argument, yielded, result)
            for initial, argument, yielded, result in zip(
                identity_outputs,
                carry_arguments,
                next_state,
                final_state,
                strict=True,
            )
        ),
        (
            ScheduledOperation(
                "call_region",
                (*parameters, *carry_arguments),
                next_state,
                (
                    ("region", "map_fold_combine"),
                    (
                        "domain_axes",
                        (*problem.spec.batch_axes, *problem.spec.fold_axes),
                    ),
                ),
            ),
        ),
    )
    visible = tuple(
        value(
            name,
            TensorSignature(
                problem.project_output.nodes[node_id].axes,
                problem.project_output.nodes[node_id].dtype,
            ),
        )
        for name, node_id in problem.project_output.results
    )
    epilogue = (
        ScheduledOperation(
            "call_region",
            final_state,
            visible,
            (("region", "project_output"),),
        ),
    )
    commits = tuple(
        ScheduledOperation("store", (item,), (), (("destination", item.name),))
        for item in visible
    )
    return TiledStage(
        "single_owner_tiled_fold",
        tuple(axis(name) for name in owner_names),
        arguments,
        parameters,
        initialization,
        loop,
        epilogue,
        commits,
        tuple(
            AxisPartition(name, sizes[name], axis(name).tile, axis(name).count)
            for name in owner_names
        )
        + (
            AxisPartition(
                fold_names[0],
                sizes[fold_names[0]],
                axis(fold_names[0]).tile,
                1,
            ),
        ),
    )


def specialize_cutile_schedule(
    program: FoldProgram,
    template: ForwardScheduleTemplate,
    sizes: dict[str, int],
    *,
    tiles: dict[str, int] | None = None,
    target: TargetModel | None = None,
) -> SpecializedSchedule:
    """Bind an online schedule to CuTile tiles and build its executable DAG."""
    from cutilereduce.experimental.planning.templates import build_schedule_space

    space = build_schedule_space(program)
    if template not in space.forward:
        raise ValueError("forward schedule must belong to the program's schedule space")
    if template.name != "online":
        raise NotImplementedError(
            "the initial CuTile target implements the online schedule only"
        )
    if set(sizes) != set(program.spec.axes):
        raise ValueError("sizes must define every logical axis exactly once")
    if any(type(value) is not int or value <= 0 for value in sizes.values()):
        raise ValueError("all logical extents must be positive integers")
    target = target or TargetModel(
        "cutile",
        frozenset(
            {"tensor_regions", "tiled_loops", "multi_stage", "relaxed_atomic_add"}
        ),
        supported_schedules=frozenset({"online", "independent", "prefix"}),
    )
    if target.name != "cutile":
        raise ValueError("specialize_cutile_schedule requires a CuTile target")
    target.require("tensor_regions", "tiled_loops")
    if not target.supports_schedule(template.name):
        raise NotImplementedError(f"CuTile target does not support {template.name!r}")
    scheduled_axes = (*program.spec.batch_axes, *program.spec.fold_axes)
    effective_tiles = target.bind_tiles(sizes, scheduled_axes, tiles)
    stage = _build_tiled_stage(program, sizes, tiles=effective_tiles)
    parameters = tuple(
        (f"tile.{axis}", effective_tiles[axis])
        for axis in (*program.spec.batch_axes, *program.spec.fold_axes)
    )
    return SpecializedSchedule(
        program=program,
        template=template,
        target=target,
        shape=tuple((axis, sizes[axis]) for axis in program.spec.axes),
        parameters=parameters,
        implementation=TiledPlan((stage,)),
    )


def _build_tiled_backward_plan(
    problem: FoldProgram,
    sizes: dict[str, int],
    *,
    loop_axis: str,
    tiles: dict[str, int],
    loop_programs: int = 1,
    program_axes: tuple[str, ...] | None = None,
) -> TiledPlan:
    """Build separate embed and finalize stages sharing hierarchical tiling."""
    region = problem.embed_output_vjp
    if region is None:
        raise ValueError("backward tiled planning requires an embed region")
    next_id = 0

    def value(name: str, node_id: int) -> ScheduledValue:
        nonlocal next_id
        node = region.nodes[node_id]
        result = ScheduledValue(next_id, name, TensorSignature(node.axes, node.dtype))
        next_id += 1
        return result

    bindings = tuple(
        TileBinding(
            name,
            region.nodes[node_id].axes,
            region.nodes[node_id].dtype,
            region.nodes[node_id].axes,
        )
        for name, node_id in region.arguments
    )
    parameters = tuple(value(name, node_id) for name, node_id in region.arguments)
    results = tuple(value(name, node_id) for name, node_id in region.results)
    embed = TiledStage(
        "embed_output_vjp",
        tuple(
            ScheduledAxis(
                axis,
                min(tiles.get(axis, sizes[axis]), sizes[axis]),
                sizes[axis],
            )
            for axis in problem.spec.batch_axes
        ),
        bindings,
        parameters,
        (),
        None,
        (
            ScheduledOperation(
                "call_region",
                parameters,
                results,
                (("region", "embed_output_vjp"),),
            ),
        ),
        tuple(
            ScheduledOperation("store", (item,), (), (("destination", item.name),))
            for item in results
        ),
        tuple(
            AxisPartition(
                axis,
                sizes[axis],
                min(tiles.get(axis, sizes[axis]), sizes[axis]),
                math.ceil(sizes[axis] / min(tiles.get(axis, sizes[axis]), sizes[axis])),
            )
            for axis in problem.spec.batch_axes
        ),
    )
    scheduled = schedule_backward_program(
        problem,
        sizes,
        loop_axis=loop_axis,
        tiles=tiles,
        program_axes=program_axes,
        loop_programs=loop_programs,
    )
    finalize = replace(lower_backward_stage(scheduled), name="finalize_map_vjp")
    buffer = TiledBuffer(
        "accumulator",
        tuple(
            (name, item.signature)
            for (name, _), item in zip(region.results, results, strict=True)
        ),
    )
    return TiledPlan(
        (embed, finalize),
        (buffer,),
        (StageDependency(embed.name, finalize.name, buffer.name),),
    )


def specialize_cutile_backward_schedule(
    program: FoldProgram,
    template,
    sizes: dict[str, int],
    *,
    loop_axis: str,
    tiles: dict[str, int] | None = None,
    loop_programs: int = 1,
    program_axes: tuple[str, ...] | None = None,
    target: TargetModel | None = None,
) -> SpecializedSchedule:
    """Bind a backward template to CuTile dimensions and a concrete stage DAG."""
    from cutilereduce.experimental.planning.templates import build_schedule_space

    space = build_schedule_space(program)
    if template not in space.backward:
        raise ValueError(
            "backward schedule must belong to the program's schedule space"
        )
    if set(sizes) != set(program.spec.axes):
        raise ValueError("sizes must define every logical axis exactly once")
    if any(type(value) is not int or value <= 0 for value in sizes.values()):
        raise ValueError("all logical extents must be positive integers")
    target = target or TargetModel(
        "cutile",
        frozenset(
            {"tensor_regions", "tiled_loops", "multi_stage", "relaxed_atomic_add"}
        ),
        supported_schedules=frozenset({"online", "independent", "prefix"}),
    )
    if target.name != "cutile":
        raise ValueError("specialize_cutile_backward_schedule requires CuTile")
    target.require("tensor_regions", "tiled_loops", "multi_stage")
    if not target.supports_schedule(template.name):
        raise NotImplementedError(f"CuTile target does not support {template.name!r}")
    scheduled_axes = (*program.spec.batch_axes, *program.spec.fold_axes)
    effective_tiles = target.bind_tiles(sizes, scheduled_axes, tiles)
    implementation = _build_tiled_backward_plan(
        program,
        sizes,
        loop_axis=loop_axis,
        tiles=effective_tiles,
        loop_programs=loop_programs,
        program_axes=program_axes,
    )
    parameters = tuple(
        (f"tile.{axis}", value) for axis, value in effective_tiles.items()
    )
    parameters += (("loop_programs", loop_programs),)
    return SpecializedSchedule(
        program,
        template,
        target,
        tuple((axis, sizes[axis]) for axis in program.spec.axes),
        parameters,
        implementation,
    )
