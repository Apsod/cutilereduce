"""Target-neutral single-owner tiled folds and a PyTorch CPU interpreter."""

from __future__ import annotations

import itertools
import math
from dataclasses import replace
from typing import Any

from .ir import TensorRegion, TensorSignature
from .operator import FoldProgram
from .planning import ForwardScheduleTemplate
from .target import TargetModel
from .scheduled import ScheduledAxis, TileBinding, schedule_backward_program
from .scheduled_ssa import (
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
                    ("domain_axes", (*problem.spec.batch_axes, *problem.spec.fold_axes)),
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
    from .planning import build_schedule_space

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
        frozenset({"tensor_regions", "tiled_loops", "multi_stage", "relaxed_atomic_add"}),
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
    from .planning import build_schedule_space

    space = build_schedule_space(program)
    if template not in space.backward:
        raise ValueError("backward schedule must belong to the program's schedule space")
    if set(sizes) != set(program.spec.axes):
        raise ValueError("sizes must define every logical axis exactly once")
    if any(type(value) is not int or value <= 0 for value in sizes.values()):
        raise ValueError("all logical extents must be positive integers")
    target = target or TargetModel(
        "cutile",
        frozenset({"tensor_regions", "tiled_loops", "multi_stage", "relaxed_atomic_add"}),
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
    parameters = tuple((f"tile.{axis}", value) for axis, value in effective_tiles.items())
    parameters += (("loop_programs", loop_programs),)
    return SpecializedSchedule(
        program,
        template,
        target,
        tuple((axis, sizes[axis]) for axis in program.spec.axes),
        parameters,
        implementation,
    )


def _align(value: Any, source: tuple[str, ...], target: tuple[str, ...]):
    present = tuple(axis for axis in target if axis in source)
    if present != source:
        value = value.permute(*(source.index(axis) for axis in present))
    for index, axis in enumerate(target):
        if axis not in source:
            value = value.unsqueeze(index)
    return value


def _execute_region(
    region: TensorRegion,
    arguments: dict[str, Any],
    *,
    axis_origins: dict[str, int] | None = None,
    axis_sizes: dict[str, int] | None = None,
    domain_valid: Any | None = None,
) -> dict[str, Any]:
    import torch

    values = {node_id: arguments[name] for name, node_id in region.arguments}
    axis_origins = axis_origins or {}
    axis_sizes = axis_sizes or {}
    dtypes = {
        "float32": torch.float32,
        "float64": torch.float64,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bool": torch.bool,
        "int32": torch.int32,
        "int64": torch.int64,
    }

    def axis_extent(axis: str) -> int:
        if axis in axis_sizes:
            return axis_sizes[axis]
        extent = next(
            (
                arguments[name].shape[region.nodes[node_id].axes.index(axis)]
                for name, node_id in region.arguments
                if axis in region.nodes[node_id].axes
            ),
            None,
        )
        if extent is not None:
            return extent
        # AD can eliminate every primal argument carrying a reduced axis.
        # Explicit validity still supplies that physical domain's extent.
        if domain_valid is not None:
            domain = next(node for node in region.nodes if node.op == "domain_valid")
            if domain_valid.ndim != len(domain.axes):
                raise ValueError("domain validity must match domain axes")
            return domain_valid.shape[domain.axes.index(axis)]
        raise ValueError(f"cannot infer the extent of axis {axis!r}")

    for node in region.nodes:
        if node.id in values:
            continue
        inputs = [values[value] for value in node.inputs]
        input_nodes = [region.nodes[value] for value in node.inputs]
        attrs = dict(node.attributes)
        if node.op == "domain_valid":
            shape = tuple(axis_extent(axis) for axis in node.axes)
            if domain_valid is None:
                value = torch.ones(shape, dtype=torch.bool)
            else:
                if domain_valid.dtype != torch.bool or tuple(domain_valid.shape) != shape:
                    raise ValueError("domain validity must be a boolean tensor over domain axes")
                value = domain_valid
        elif node.op == "axis_index":
            axis = attrs["axis"]
            extent = axis_sizes.get(axis)
            if extent is None:
                extent = axis_extent(axis)
            value = torch.arange(
                axis_origins.get(axis, 0),
                axis_origins.get(axis, 0) + extent,
                dtype=dtypes[node.dtype],
            )
        elif node.op == "constant":
            value = torch.tensor(attrs["value"], dtype=dtypes[node.dtype])
        elif node.op in {"zero", "full"}:
            shape = tuple(axis_extent(axis) for axis in node.axes)
            fill = 0 if node.op == "zero" else attrs["value"]
            value = torch.full(shape, fill, dtype=dtypes[node.dtype])
        elif node.op in {"add", "subtract", "multiply", "divide"}:
            left = _align(inputs[0], input_nodes[0].axes, node.axes)
            right = _align(inputs[1], input_nodes[1].axes, node.axes)
            value = {
                "add": torch.add,
                "subtract": torch.subtract,
                "multiply": torch.multiply,
                "divide": torch.divide,
            }[node.op](left, right)
        elif node.op in {"neg", "exp", "log", "sqrt", "rsqrt", "abs", "sin", "cos", "tanh", "sigmoid", "log1p", "expm1", "logical_not"}:
            value = getattr(torch, node.op)(_align(inputs[0], input_nodes[0].axes, node.axes))
        elif node.op in {"minimum", "maximum", "logical_and", "logical_or", "floor_divide", "remainder"}:
            value = getattr(torch, node.op)(*[_align(item, source.axes, node.axes) for item, source in zip(inputs, input_nodes, strict=True)])
        elif node.op == "stop_gradient":
            value = _align(inputs[0], input_nodes[0].axes, node.axes).detach()
        elif node.op == "clamp":
            value = torch.clamp(_align(inputs[0], input_nodes[0].axes, node.axes), min=attrs["min"], max=attrs["max"])
        elif node.op == "powi":
            from .ops import multiplication_chain
            base = _align(inputs[0], input_nodes[0].axes, node.axes)
            exponent = attrs["exponent"]
            powers = [base]
            for left, right in multiplication_chain(abs(exponent)) if exponent else ():
                powers.append(powers[left] * powers[right])
            value = powers[-1] if exponent else torch.ones_like(base)
            if exponent < 0:
                value = torch.reciprocal(value)
        elif node.op == "cast":
            value = inputs[0].to(dtypes[node.dtype])
        elif node.op in {"where", "mask_identity"}:
            aligned = [
                _align(item, source.axes, node.axes)
                for item, source in zip(inputs, input_nodes, strict=True)
            ]
            value = torch.where(*aligned)
        elif node.op.startswith("compare_"):
            target = node.axes
            left = _align(inputs[0], input_nodes[0].axes, target)
            right = _align(inputs[1], input_nodes[1].axes, target)
            value = {
                "compare_eq": torch.eq,
                "compare_equal": torch.eq,
                "compare_ne": torch.ne,
                "compare_not_equal": torch.ne,
                "compare_lt": torch.lt,
                "compare_less": torch.lt,
                "compare_le": torch.le,
                "compare_less_equal": torch.le,
                "compare_gt": torch.gt,
                "compare_greater": torch.gt,
                "compare_ge": torch.ge,
                "compare_greater_equal": torch.ge,
            }[node.op](left, right)
        elif node.op.startswith("reduce_"):
            dims = tuple(input_nodes[0].axes.index(axis) for axis in attrs["over"])
            value = (
                inputs[0].sum(dims) if node.op == "reduce_sum" else inputs[0].amax(dims)
            )
        elif node.op in {"contract", "contract_accumulate"}:
            axes = tuple(dict.fromkeys((*input_nodes[0].axes, *input_nodes[1].axes)))
            letters = dict(
                zip(
                    axes,
                    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ",
                    strict=False,
                )
            )

            def encode(xs: tuple[str, ...]) -> str:
                return "".join(letters[axis] for axis in xs)

            equation = f"{encode(input_nodes[0].axes)},{encode(input_nodes[1].axes)}->{encode(node.axes)}"
            value = torch.einsum(equation, inputs[0], inputs[1])
            if node.op == "contract_accumulate":
                value = value + inputs[2]
        else:
            raise NotImplementedError(f"CPU mock does not support {node.op!r}")
        values[node.id] = value
    return {name: values[node_id] for name, node_id in region.results}


def _ranges(axis: ScheduledAxis):
    return tuple(
        (start, min(axis.tile, axis.extent - start))
        for start in range(0, axis.extent, axis.tile)
    )


def execute_tiled_fold_cpu(
    problem: FoldProgram,
    program: TiledStage,
    inputs: dict[str, Any],
) -> dict[str, Any]:
    """Interpret a tiled schedule on CPU using PyTorch tensor primitives."""
    import torch

    regions = {
        "execution_identity": problem.execution_identity,
        "map_fold": problem.map_fold,
        "map_fold_combine": problem.map_fold_combine,
        "execution_combine": problem.execution_combine,
        "project_output": problem.project_output,
    }
    loop = program.loop
    if loop is None:
        raise ValueError("CPU fold mock requires a loop stage")
    if (
        len(program.initialization) != 1
        or len(loop.body) != 1
        or len(program.epilogue) != 1
    ):
        raise ValueError("CPU fold mock requires the single-owner fold stage shape")

    def called_region(operation: ScheduledOperation) -> str:
        if operation.op != "call_region":
            raise ValueError("CPU fold mock expected a region call")
        region = dict(operation.attributes).get("region")
        if not isinstance(region, str) or region not in regions:
            raise ValueError(f"unknown scheduled region {region!r}")
        return region

    identity_region = called_region(program.initialization[0])
    tile_region = called_region(loop.body[0])
    epilogue_region = called_region(program.epilogue[0])
    loop_partition = next(
        partition
        for partition in program.effective_partitions
        if partition.axis == loop.axis.axis
    )
    if loop_partition.programs != 1:
        raise ValueError("complete-output CPU mock requires one fold partition")
    owner_partitions = tuple(
        partition
        for partition in program.effective_partitions
        if partition.axis != loop.axis.axis
    )
    owner_programs = tuple(range(partition.programs) for partition in owner_partitions)
    fold_start, fold_end = loop_partition.tile_span(0)
    fold_ranges = tuple(
        (
            tile * loop_partition.tile,
            min(
                loop_partition.tile, loop_partition.extent - tile * loop_partition.tile
            ),
        )
        for tile in range(fold_start, fold_end)
    )
    outputs: dict[str, Any] = {}
    logical_sizes = {
        axis: inputs[name].shape[spec.axes.index(axis)]
        for name, spec in problem.spec.inputs.items()
        for axis in spec.axes
    }

    def slices(binding: TileBinding, locations: dict[str, tuple[int, int]]):
        return tuple(
            slice(*((locations[axis][0], sum(locations[axis]))))
            if axis in locations
            else slice(None)
            for axis in binding.axes
        )

    for owner in itertools.product(*owner_programs):
        owner_location = {
            partition.axis: (
                partition.element_span(coordinate)[0],
                partition.element_span(coordinate)[1]
                - partition.element_span(coordinate)[0],
            )
            for partition, coordinate in zip(owner_partitions, owner, strict=True)
        }
        state = _execute_region(regions[identity_region], {}, axis_sizes=logical_sizes)
        for folded in fold_ranges:
            locations = owner_location | {loop.axis.axis: folded}
            tile_inputs = {
                binding.name: inputs[binding.name][slices(binding, locations)]
                for binding in program.bindings
            }
            combine_inputs = dict(tile_inputs)
            combine_inputs.update(
                {
                    f"accumulator.{name.removeprefix('execution.')}": value
                    for name, value in state.items()
                }
            )
            state = _execute_region(
                regions[tile_region],
                combine_inputs,
                axis_origins={axis: start for axis, (start, _) in locations.items()},
                axis_sizes={axis: size for axis, (_, size) in locations.items()},
            )
        tile = _execute_region(regions[epilogue_region], state)
        for name, value in tile.items():
            result_node = regions[epilogue_region].nodes[
                dict(regions[epilogue_region].results)[name]
            ]
            if name not in outputs:
                shape = tuple(
                    next(
                        (
                            axis.extent
                            for axis in program.effective_partitions
                            if axis.axis == logical
                        ),
                        value.shape[result_node.axes.index(logical)],
                    )
                    for logical in result_node.axes
                )
                outputs[name] = torch.empty(shape, dtype=value.dtype)
            output_slices = tuple(
                slice(
                    *(
                        (owner_location[axis][0], sum(owner_location[axis]))
                        if axis in owner_location
                        else (None, None)
                    )
                )
                for axis in result_node.axes
            )
            outputs[name][output_slices] = value
    return outputs


def execute_tiled_backward_cpu(
    problem: FoldProgram,
    plan: TiledPlan,
    inputs: dict[str, Any],
    saved_execution: dict[str, Any],
    grad_output: dict[str, Any],
) -> dict[str, Any]:
    """Deterministically interpret a prefix-free two-stage backward plan."""
    import torch

    if problem.requires_prefix:
        raise NotImplementedError(
            "the initial tiled backward interpreter is prefix-free"
        )
    if len(plan.stages) != 2 or plan.stages[1].loop is None:
        raise ValueError("backward CPU mock expects embed and finalize stages")
    embed_region = problem.embed_output_vjp
    finalize_region = (
        problem.map_finalize_map_vjp_accumulate
        if problem.requires_prefix
        else problem.map_finalize_map_vjp
    )
    if embed_region is None or finalize_region is None:
        raise ValueError("backward CPU mock requires backward regions")

    embed_arguments = {
        name: (
            saved_execution[name.removeprefix("saved_execution.")]
            if name.startswith("saved_execution.")
            else grad_output[name.removeprefix("grad_output.")]
        )
        for name, _ in embed_region.arguments
    }
    accumulator = _execute_region(embed_region, embed_arguments)
    stage = plan.stages[1]
    loop = stage.loop
    assert loop is not None
    loop_partition = next(
        item for item in stage.effective_partitions if item.axis == loop.axis.axis
    )
    outer_partitions = tuple(
        item for item in stage.effective_partitions if item.axis != loop.axis.axis
    )
    results = dict(finalize_region.results)
    gradients: dict[str, Any] = {}
    for input_name, spec in problem.spec.inputs.items():
        result_name = f"partial_grad_input.{input_name}"
        if result_name in results:
            gradients[input_name] = torch.zeros_like(inputs[input_name])

    def tile(value: Any, axes: tuple[str, ...], locations: dict[str, tuple[int, int]]):
        index = tuple(
            slice(locations[axis][0], sum(locations[axis]))
            if axis in locations
            else slice(None)
            for axis in axes
        )
        return value[index]

    coordinates = tuple(
        itertools.product(
            *(range(partition.programs) for partition in outer_partitions),
            range(loop_partition.programs),
        )
    )
    for coordinate in coordinates:
        outer_coordinate = coordinate[:-1]
        loop_program = coordinate[-1]
        outer_locations = {
            partition.axis: (
                partition.element_span(owner)[0],
                partition.element_span(owner)[1] - partition.element_span(owner)[0],
            )
            for partition, owner in zip(outer_partitions, outer_coordinate, strict=True)
        }
        persistent: dict[str, Any] = {}
        start, end = loop_partition.tile_span(loop_program)
        for tile_index in range(start, end):
            fold_start = tile_index * loop_partition.tile
            fold_size = min(loop_partition.tile, loop_partition.extent - fold_start)
            locations = outer_locations | {loop_partition.axis: (fold_start, fold_size)}
            arguments = {}
            for name, node_id in finalize_region.arguments:
                axes = finalize_region.nodes[node_id].axes
                if name.startswith("accumulator."):
                    arguments[name] = tile(accumulator[name], axes, locations)
                else:
                    arguments[name] = tile(inputs[name], axes, locations)
            partials = _execute_region(
                finalize_region,
                arguments,
                axis_origins={axis: start for axis, (start, _) in locations.items()},
                axis_sizes={axis: size for axis, (_, size) in locations.items()},
            )
            for input_name, destination in gradients.items():
                result_name = f"partial_grad_input.{input_name}"
                if result_name not in partials:
                    continue
                axes = finalize_region.nodes[results[result_name]].axes
                contribution = partials[result_name]
                if loop_partition.axis in axes:
                    index = tuple(
                        slice(locations[axis][0], sum(locations[axis]))
                        if axis in locations
                        else slice(None)
                        for axis in problem.spec.inputs[input_name].axes
                    )
                    destination[index] += contribution
                else:
                    persistent[input_name] = (
                        persistent.get(input_name, torch.zeros_like(contribution))
                        + contribution
                    )
        for input_name, contribution in persistent.items():
            spec = problem.spec.inputs[input_name]
            index = tuple(
                slice(outer_locations[axis][0], sum(outer_locations[axis]))
                if axis in outer_locations
                else slice(None)
                for axis in spec.axes
            )
            gradients[input_name][index] += contribution
    return gradients


def format_tiled_fold(program: TiledStage) -> str:
    loop = program.loop
    if loop is None:
        raise ValueError("fold formatting requires a loop stage")
    owners = ", ".join(
        f"{axis.axis}[tile={axis.tile}, programs={axis.programs}]"
        for axis in program.effective_partitions
        if axis.axis != loop.axis.axis
    )
    fold = next(
        axis for axis in program.effective_partitions if axis.axis == loop.axis.axis
    )
    folds = f"{fold.axis}[tile={fold.tile}, programs={fold.programs}]"
    return "\n".join(
        (
            f"tiled_fold {program.name} {{",
            f"  parallel owners ({owners}) {{",
            "    %state = call @execution_identity",
            f"    serial fold ({folds}) iter_args(%state) {{",
            "      %next = call @map_fold_combine(tile_inputs, %state)",
            "      yield %next",
            "    }",
            "    %output = call @project_output(%state)",
            "    store %output",
            "  }",
            "}",
        )
    )


__all__ = [
    "execute_tiled_backward_cpu",
    "execute_tiled_fold_cpu",
    "format_tiled_fold",
    "specialize_cutile_backward_schedule",
    "specialize_cutile_schedule",
]
