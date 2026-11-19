"""cuTile Python source emission for target-neutral tiled stages."""

from __future__ import annotations

from typing import Any

from cutilereduce.experimental.ir.core import TensorRegion
from cutilereduce.experimental.compiler.operator import FoldProgram
from cutilereduce.experimental.ir.passes import (
    AxisAffineIndex,
    discover_tile_mask,
    realize_contraction_precision,
)
from cutilereduce.experimental.planning.ssa import (
    ScheduledOperation,
    SpecializedSchedule,
    TiledPlan,
    TiledStage,
)


from .regions import _CT_DTYPE, _tuple, _literal, _identifier, _shape, _emit_region


def _called_region(operation: ScheduledOperation) -> str:
    region = dict(operation.attributes).get("region")
    if operation.op != "call_region" or not isinstance(region, str):
        raise ValueError("cuTile stage lowering expected a named region call")
    return region


def _axis_index_arguments(
    region: TensorRegion,
    partitions: dict[str, Any],
    tile_coordinates: dict[str, str],
    sizes: dict[str, int],
) -> tuple[str, ...]:
    result = []
    for node in region.nodes:
        if node.op != "axis_index":
            continue
        axis = node.attribute("axis")
        partition = partitions.get(axis)
        if partition is None:
            result.append(f"ct.arange({sizes[axis]}, dtype={_CT_DTYPE[node.dtype]})")
        else:
            coordinate = tile_coordinates.get(axis, "0")
            result.append(
                f"({coordinate} * {partition.tile} + "
                f"ct.arange({partition.tile}, dtype={_CT_DTYPE[node.dtype]}))"
            )
    return tuple(result)


def _physical_valid_lines(
    axes: tuple[str, ...],
    partitions: dict[str, Any],
    tile_coordinates: dict[str, str],
) -> list[str]:
    lines = []
    terms = []
    for dimension, axis in enumerate(axes):
        partition = partitions[axis]
        coordinate = tile_coordinates.get(axis, "0")
        shape = tuple(
            partition.tile if index == dimension else 1 for index in range(len(axes))
        )
        name = f"valid_{axis}"
        lines.append(
            f"{name} = ct.reshape(({coordinate} * {partition.tile} + "
            f"ct.arange({partition.tile}, dtype=ct.int32)) < {partition.extent}, {_tuple(shape)})"
        )
        terms.append(name)
    lines.append(f"physical_valid = {' & '.join(terms)}")
    return lines


def _tile_mask_guard(
    region: TensorRegion,
    partitions: dict[str, Any],
    tile_coordinates: dict[str, str],
) -> str | None:
    """Return a scalar may-contain-valid-factors guard for simple index masks."""
    comparison = discover_tile_mask(region)
    if comparison is None:
        return None

    def interval(value: AxisAffineIndex) -> tuple[str, str] | None:
        if value.axis is None:
            literal = _literal(value.offset)
            return literal, literal
        axis = value.axis
        partition = partitions.get(axis)
        if partition is None:
            return None
        coordinate = tile_coordinates.get(axis, "0")
        lower = f"({coordinate} * {partition.tile})"
        upper = f"ct.minimum({lower} + {partition.tile - 1}, {partition.extent - 1})"
        if value.offset:
            offset = _literal(value.offset)
            lower = f"({lower} + {offset})"
            upper = f"({upper} + {offset})"
        return lower, upper

    left = interval(comparison.left)
    right = interval(comparison.right)
    if left is None or right is None:
        return None
    left_low, left_high = left
    right_low, right_high = right
    op = comparison.operation
    if op in {"less", "lt"}:
        return f"{left_low} < {right_high}"
    if op in {"less_equal", "le"}:
        return f"{left_low} <= {right_high}"
    if op in {"greater", "gt"}:
        return f"{left_high} > {right_low}"
    if op in {"greater_equal", "ge"}:
        return f"{left_high} >= {right_low}"
    if op in {"equal", "eq"}:
        return f"({left_low} <= {right_high}) & ({right_low} <= {left_high})"
    if op in {"not_equal", "ne"}:
        return f"({left_low} != {left_high}) | ({right_low} != {right_high}) | ({left_low} != {right_low})"
    return None


def _initial_state_lines(
    identity: TensorRegion,
    map_fold_combine: TensorRegion,
    sizes: dict[str, int],
    tiles: dict[str, int],
) -> list[str]:
    """Lift logical carrier identities to the physical loop-carry tile shape."""
    accumulator_nodes = tuple(
        map_fold_combine.nodes[node_id]
        for name, node_id in map_fold_combine.arguments
        if name.startswith("accumulator.")
    )
    identity_nodes = tuple(
        identity.nodes[node_id] for _, node_id in identity.results
    )
    if len(identity_nodes) != len(accumulator_nodes):
        raise ValueError("execution identity and accumulator fields disagree")
    values = []
    for index, (source, target) in enumerate(
        zip(identity_nodes, accumulator_nodes, strict=True)
    ):
        retained_axes = tuple(axis for axis in target.axes if axis in source.axes)
        if retained_axes != source.axes:
            raise ValueError("execution identity axes are incompatible with accumulator")
        source_shape = tuple(
            (tiles.get(axis, sizes[axis])) if axis in source.axes else 1
            for axis in target.axes
        )
        values.append(
            f"ct.broadcast_to(ct.reshape(state[{index}], {_tuple(source_shape)}), "
            f"{_tuple(_shape(target.axes, sizes, tiles))})"
        )
    return ["    state = execution_identity()", f"    state = {_tuple(tuple(values))}"]


def emit_cutile_source(
    specialized: SpecializedSchedule,
) -> str:
    """Emit cuTile source from a self-contained specialized schedule."""
    if getattr(specialized.target, "name", specialized.target) != "cutile":
        raise ValueError("emit_cutile_source requires a CuTile schedule")
    if not isinstance(specialized.program, FoldProgram):
        raise TypeError("specialized schedule does not contain a FoldProgram")
    if not isinstance(specialized.implementation, TiledPlan):
        raise TypeError("specialized schedule does not contain a TiledPlan")
    if len(specialized.implementation.stages) != 1:
        raise NotImplementedError("forward CuTile emission expects one stage")
    problem = specialized.program
    stage = specialized.implementation.stages[0]
    sizes = dict(specialized.shape)
    if set(sizes) != set(problem.spec.axes):
        raise ValueError("sizes must define every logical axis exactly once")
    loop = stage.loop
    if loop is None:
        raise NotImplementedError("initial cuTile emitter requires a loop stage")
    if (
        len(stage.initialization) != 1
        or len(loop.body) != 1
        or len(stage.epilogue) != 1
    ):
        raise NotImplementedError(
            "initial cuTile emitter supports the fold-stage shape"
        )
    regions = {
        "execution_identity": problem.execution_identity,
        "map_fold_combine": realize_contraction_precision(problem.map_fold_combine),
        "project_output": problem.project_output,
    }
    tiles = {partition.axis: partition.tile for partition in stage.effective_partitions}
    helpers = [
        _emit_region(name, region, sizes, tiles) for name, region in regions.items()
    ]
    buffers = tuple(binding.name for binding in stage.bindings)
    destinations = tuple(
        dict(commit.attributes)["destination"] for commit in stage.commits
    )
    identity_positions = {
        name: index for index, (name, _) in enumerate(regions["execution_identity"].results)
    }
    try:
        project_state = ", ".join(
            f"state[{identity_positions[name]}]"
            for name, _ in regions["project_output"].arguments
        )
    except KeyError as error:
        raise ValueError(
            "project output requires an execution field not produced by identity"
        ) from error
    parameters = ", ".join(
        (*buffers, *(f"out_{name.replace('.', '_')}" for name in destinations))
    )
    grid = stage.grid_partitions
    loop_partition = next(
        value for value in stage.effective_partitions if value.axis == loop.axis.axis
    )
    partition_by_axis = {value.axis: value for value in stage.effective_partitions}
    lines = ["@ct.kernel", f"def {stage.name}({parameters}):", "    pid = ct.bid(0)"]
    coordinates: dict[str, str] = {}
    for partition in grid:
        coordinate = f"program_{partition.axis}"
        lines.extend(
            (
                f"    {coordinate} = pid % {partition.programs}",
                f"    pid = pid // {partition.programs}",
            )
        )
        coordinates[partition.axis] = coordinate
    loop_coordinate = coordinates.get(loop_partition.axis, "0")
    base, remainder = divmod(loop_partition.tile_count, loop_partition.programs)
    lines.extend(
        (
            f"    loop_size = {base}",
            f"    loop_extra = {loop_coordinate} < {remainder}",
            f"    loop_offset = {base} * {loop_coordinate} + ct.minimum({loop_coordinate}, {remainder})",
            *_initial_state_lines(
                regions["execution_identity"],
                regions["map_fold_combine"],
                sizes,
                tiles,
            ),
        )
    )
    lines.extend(
        _binding_load(
            binding,
            indent="    ",
            coordinates=coordinates,
            sizes=sizes,
            tiles=tiles,
        )
        for binding in stage.resident_bindings
    )
    lines.extend(
        (
            "    for local_tile in range(loop_size):",
            "        loop_tile = loop_offset + local_tile",
        )
    )

    def load_lines(indent: str) -> list[str]:
        tile_coordinates = {**coordinates, loop_partition.axis: "loop_tile"}
        result = [
            f"{indent}{line}"
            for line in _physical_valid_lines(
                (*problem.spec.batch_axes, *problem.spec.fold_axes),
                partition_by_axis,
                tile_coordinates,
            )
        ]
        region = regions["map_fold_combine"]
        guard = _tile_mask_guard(problem.map_fold, partition_by_axis, tile_coordinates)
        body_indent = indent
        if guard is not None:
            result.append(f"{indent}if {guard}:")
            body_indent += "    "
        for binding in stage.streamed_bindings:
            result.append(
                _binding_load(
                    binding,
                    indent=body_indent,
                    coordinates=tile_coordinates,
                    sizes=sizes,
                    tiles=tiles,
                )
            )
        arguments = [
            f"tile_{binding.name.replace('.', '_')}" for binding in stage.bindings
        ]
        arguments.append("*state")
        arguments.extend(
            _axis_index_arguments(region, partition_by_axis, tile_coordinates, sizes)
        )
        arguments.append("physical_valid")
        result.append(f"{body_indent}state = map_fold_combine({', '.join(arguments)})")
        return result

    lines.extend(load_lines("        "))
    lines.extend(
        (
            "    if loop_extra:",
            f"        loop_tile = loop_offset + {base}",
        )
    )
    lines.extend(load_lines("        "))
    lines.append(f"    visible = project_output({project_state})")
    owner_indices = tuple(
        coordinates.get(axis, "0") for axis in problem.spec.batch_axes
    )
    for index, destination in enumerate(destinations):
        result = problem.project_output.nodes[
            dict(problem.project_output.results)[destination]
        ]
        shape = _shape(result.axes, sizes, tiles)
        indices = tuple(coordinates.get(axis, "0") for axis in result.axes)
        lines.append(
            f"    ct.store(out_{destination.replace('.', '_')}, "
            f"{_tuple(indices or owner_indices)}, "
            f"ct.reshape(visible[{index}], {_tuple(shape)}))"
        )
    source = "\n\n".join(
        (
            "import cuda.tile as ct",
            *helpers,
            "\n".join(lines),
        )
    )
    compile(source, "<generated-cutile>", "exec")
    return source


def _tile_indices(
    axes: tuple[str, ...],
    coordinates: dict[str, str],
    tiles: dict[str, int],
) -> tuple[str, ...]:
    del tiles
    return tuple(coordinates.get(axis, "0") for axis in axes)


def _binding_load(
    binding: Any,
    *,
    indent: str,
    coordinates: dict[str, str],
    sizes: dict[str, int],
    tiles: dict[str, int],
) -> str:
    name = f"tile_{_identifier(binding.name)}"
    return (
        f"{indent}{name} = ct.load({_identifier(binding.name)}, "
        f"{_tuple(_tile_indices(binding.axes, coordinates, tiles))}, "
        f"{_tuple(_shape(binding.axes, sizes, tiles))}, "
        "padding_mode=ct.PaddingMode.ZERO)"
    )


def _atomic_indices(
    axes: tuple[str, ...],
    coordinates: dict[str, str],
    sizes: dict[str, int],
    tiles: dict[str, int],
) -> tuple[str, ...]:
    shape = _shape(axes, sizes, tiles)
    result = []
    for dimension, (axis, extent) in enumerate(zip(axes, shape, strict=True)):
        base = f"({coordinates[axis]} * {tiles[axis]})" if axis in coordinates else "0"
        local_shape = tuple(
            extent if index == dimension else 1 for index in range(len(axes))
        )
        index = f"ct.reshape(ct.arange({extent}), {_tuple(local_shape)})"
        if base != "0":
            index = f"({base} + {index})"
        result.append(index)
    return tuple(result)


def _emit_embed_stage(
    problem: FoldProgram,
    stage: TiledStage,
    sizes: dict[str, int],
) -> str:
    if stage.loop is not None or len(stage.epilogue) != 1:
        raise NotImplementedError("embed emission expects one map-only stage")
    region = problem.embed_output_vjp
    if region is None:
        raise ValueError("backward plan requires embed_output_vjp")
    tiles = {partition.axis: partition.tile for partition in stage.effective_partitions}
    helper_name = "embed_output_vjp_region"
    helper = _emit_region(helper_name, region, sizes, tiles)
    destinations = tuple(dict(op.attributes)["destination"] for op in stage.commits)
    parameters = tuple(_identifier(binding.name) for binding in stage.bindings) + tuple(
        f"out_{name.replace('.', '_')}" for name in destinations
    )
    lines = [
        "@ct.kernel",
        f"def {stage.name}({', '.join(parameters)}):",
        "    pid = ct.bid(0)",
    ]
    coordinates: dict[str, str] = {}
    for partition in stage.grid_partitions:
        coordinate = f"program_{partition.axis}"
        lines.extend(
            (
                f"    {coordinate} = pid % {partition.programs}",
                f"    pid = pid // {partition.programs}",
            )
        )
        coordinates[partition.axis] = coordinate
    loaded = []
    for binding in stage.bindings:
        shape = _shape(binding.axes, sizes, tiles)
        indices = _tile_indices(binding.axes, coordinates, tiles)
        name = f"tile_{binding.name.replace('.', '_')}"
        lines.append(
            f"    {name} = ct.load({_identifier(binding.name)}, {_tuple(indices)}, {_tuple(shape)}, "
            "padding_mode=ct.PaddingMode.ZERO)"
        )
        loaded.append(name)
    lines.append(f"    embedded = {helper_name}({', '.join(loaded)})")
    for index, (destination, (_, node_id)) in enumerate(
        zip(destinations, region.results, strict=True)
    ):
        node = region.nodes[node_id]
        indices = _tile_indices(node.axes, coordinates, tiles)
        lines.append(
            f"    ct.store(out_{destination.replace('.', '_')}, {_tuple(indices)}, "
            f"embedded[{index}])"
        )
    return f"{helper}\n\n" + "\n".join(lines)


def _emit_finalize_stage(
    problem: FoldProgram,
    stage: TiledStage,
    sizes: dict[str, int],
) -> str:
    loop = stage.loop
    if loop is None:
        raise NotImplementedError("finalize emission expects a loop stage")
    if problem.requires_prefix:
        raise NotImplementedError(
            "cuTile backward emission does not yet thread noncommutative prefixes"
        )
    region = (
        problem.map_finalize_map_vjp_accumulate
        if problem.requires_prefix
        else problem.map_finalize_map_vjp
    )
    if region is None:
        raise ValueError("backward plan requires a finalize region")
    tiles = {partition.axis: partition.tile for partition in stage.effective_partitions}
    helper = _emit_region("map_finalize_map_vjp", region, sizes, tiles)
    destinations = tuple(
        dict(operation.attributes)["destination"]
        for operation in (*loop.body, *stage.commits)
        if operation.op in {"store", "atomic_add_relaxed"}
    )
    parameters = tuple(_identifier(binding.name) for binding in stage.bindings) + tuple(
        f"grad_{name.replace('.', '_')}" for name in dict.fromkeys(destinations)
    )
    lines = [
        "@ct.kernel",
        f"def {stage.name}({', '.join(parameters)}):",
        "    pid = ct.bid(0)",
    ]
    coordinates: dict[str, str] = {}
    for partition in stage.grid_partitions:
        coordinate = f"program_{partition.axis}"
        lines.extend(
            (
                f"    {coordinate} = pid % {partition.programs}",
                f"    pid = pid // {partition.programs}",
            )
        )
        coordinates[partition.axis] = coordinate
    partition = next(
        item for item in stage.effective_partitions if item.axis == loop.axis.axis
    )
    partition_by_axis = {item.axis: item for item in stage.effective_partitions}
    loop_coordinate = coordinates.get(partition.axis, "0")
    base, remainder = divmod(partition.tile_count, partition.programs)
    lines.extend(
        (
            f"    loop_size = {base}",
            f"    loop_extra = {loop_coordinate} < {remainder}",
            f"    loop_offset = {base} * {loop_coordinate} + ct.minimum({loop_coordinate}, {remainder})",
        )
    )
    for carry in loop.carries:
        shape = _shape(carry.argument.signature.axes, sizes, tiles)
        lines.append(
            f"    state_{carry.argument.name.replace('.', '_')} = "
            f"ct.zeros({_tuple(shape)}, {_CT_DTYPE[carry.argument.signature.dtype]})"
        )
    resident_parameters = {
        parameter.id: f"tile_{_identifier(binding.name)}"
        for binding, parameter in zip(stage.bindings, stage.parameters, strict=True)
        if binding in stage.resident_bindings
    }
    lines.extend(
        _binding_load(
            binding,
            indent="    ",
            coordinates=coordinates,
            sizes=sizes,
            tiles=tiles,
        )
        for binding in stage.resident_bindings
    )
    call = next(operation for operation in loop.body if operation.op == "call_region")

    def emit_iteration(indent: str) -> list[str]:
        tile_coordinates = {**coordinates, partition.axis: "loop_tile"}
        result = [
            f"{indent}{line}"
            for line in _physical_valid_lines(
                (*problem.spec.batch_axes, *problem.spec.fold_axes),
                partition_by_axis,
                tile_coordinates,
            )
        ]
        guard = _tile_mask_guard(problem.map_fold, partition_by_axis, tile_coordinates)
        body_indent = indent
        if guard is not None:
            result.append(f"{indent}if {guard}:")
            body_indent += "    "
        loaded: dict[int, str] = dict(resident_parameters)
        for binding, parameter in zip(stage.bindings, stage.parameters, strict=True):
            if binding in stage.resident_bindings:
                continue
            name = f"tile_{binding.name.replace('.', '_')}"
            result.append(
                _binding_load(
                    binding,
                    indent=body_indent,
                    coordinates=tile_coordinates,
                    sizes=sizes,
                    tiles=tiles,
                )
            )
            loaded[parameter.id] = name
        arguments = [loaded[value.id] for value in call.inputs]
        arguments.extend(
            _axis_index_arguments(region, partition_by_axis, tile_coordinates, sizes)
        )
        arguments.append("physical_valid")
        outputs = tuple(
            f"partial_{value.name.replace('.', '_')}" for value in call.outputs
        )
        result.append(
            f"{body_indent}{', '.join(outputs)} = map_finalize_map_vjp({', '.join(arguments)})"
        )
        names = {
            value.id: name for value, name in zip(call.outputs, outputs, strict=True)
        }
        for operation in loop.body:
            if operation.op == "add":
                carry_name = f"state_{operation.inputs[0].name.replace('.', '_')}"
                result.append(
                    f"{body_indent}{carry_name} += {names[operation.inputs[1].id]}"
                )
                names[operation.outputs[0].id] = carry_name
            elif operation.op in {"store", "atomic_add_relaxed"}:
                destination = dict(operation.attributes)["destination"]
                source = names[operation.inputs[0].id]
                signature = operation.inputs[0].signature
                iteration_coordinates = {
                    **coordinates,
                    partition.axis: "loop_tile",
                }
                if operation.op == "store":
                    indices = _tile_indices(
                        signature.axes, iteration_coordinates, tiles
                    )
                    result.append(
                        f"{body_indent}ct.store(grad_{destination.replace('.', '_')}, "
                        f"{_tuple(indices)}, {source})"
                    )
                else:
                    indices = _atomic_indices(
                        signature.axes, iteration_coordinates, sizes, tiles
                    )
                    result.append(
                        f"{body_indent}ct.atomic_add(grad_{destination.replace('.', '_')}, "
                        f"{_tuple(indices)}, {source}, memory_order=ct.MemoryOrder.RELAXED)"
                    )
        return result

    lines.extend(
        (
            "    for local_tile in range(loop_size):",
            "        loop_tile = loop_offset + local_tile",
        )
    )
    lines.extend(emit_iteration("        "))
    lines.extend(("    if loop_extra:", f"        loop_tile = loop_offset + {base}"))
    lines.extend(emit_iteration("        "))
    carry_results = {carry.result.id: carry for carry in loop.carries}
    for operation in stage.commits:
        destination = dict(operation.attributes)["destination"]
        carry = carry_results[operation.inputs[0].id]
        source = f"state_{carry.argument.name.replace('.', '_')}"
        signature = operation.inputs[0].signature
        if operation.op == "store":
            indices = _tile_indices(signature.axes, coordinates, tiles)
            lines.append(
                f"    ct.store(grad_{destination.replace('.', '_')}, "
                f"{_tuple(indices)}, {source})"
            )
        else:
            indices = _atomic_indices(signature.axes, coordinates, sizes, tiles)
            lines.append(
                f"    ct.atomic_add(grad_{destination.replace('.', '_')}, "
                f"{_tuple(indices)}, {source}, memory_order=ct.MemoryOrder.RELAXED)"
            )
    return f"{helper}\n\n" + "\n".join(lines)


def emit_cutile_backward_source(
    specialized: SpecializedSchedule,
) -> str:
    """Emit inspectable cuTile kernels from a specialized backward schedule."""
    if getattr(specialized.target, "name", specialized.target) != "cutile":
        raise ValueError("emit_cutile_backward_source requires a CuTile schedule")
    if not isinstance(specialized.program, FoldProgram):
        raise TypeError("specialized schedule does not contain a FoldProgram")
    if not isinstance(specialized.implementation, TiledPlan):
        raise TypeError("specialized schedule does not contain a TiledPlan")
    problem = specialized.program
    plan = specialized.implementation
    sizes = dict(specialized.shape)
    if len(plan.stages) != 2:
        raise NotImplementedError("backward emission expects embed and finalize stages")
    embed, finalize = plan.stages
    source = "\n\n".join(
        (
            "import cuda.tile as ct",
            _emit_embed_stage(problem, embed, sizes),
            _emit_finalize_stage(problem, finalize, sizes),
        )
    )
    compile(source, "<generated-cutile-backward>", "exec")
    return source


__all__ = ["emit_cutile_backward_source", "emit_cutile_source"]
