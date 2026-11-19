"""Deterministic PyTorch interpretation of tensor regions and tiled schedules."""

from __future__ import annotations

import itertools
from typing import Any

from cutilereduce.experimental.ir.core import TensorRegion
from cutilereduce.experimental.compiler.operator import FoldProgram
from cutilereduce.experimental.planning.scheduled import ScheduledAxis, TileBinding
from cutilereduce.experimental.planning.ssa import (
    ScheduledOperation,
    TiledPlan,
    TiledStage,
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
    tensors = [
        value
        for value in (*arguments.values(), domain_valid)
        if isinstance(value, torch.Tensor)
    ]
    devices = {value.device for value in tensors}
    if len(devices) > 1:
        raise ValueError("reference execution requires tensor inputs on one device")
    device = next(iter(devices), None)
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
                value = torch.ones(shape, dtype=torch.bool, device=device)
            else:
                if (
                    domain_valid.dtype != torch.bool
                    or tuple(domain_valid.shape) != shape
                ):
                    raise ValueError(
                        "domain validity must be a boolean tensor over domain axes"
                    )
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
                device=device,
            )
        elif node.op == "constant":
            value = torch.tensor(
                attrs["value"], dtype=dtypes[node.dtype], device=device
            )
        elif node.op in {"zero", "full"}:
            shape = tuple(axis_extent(axis) for axis in node.axes)
            fill = 0 if node.op == "zero" else attrs["value"]
            value = torch.full(shape, fill, dtype=dtypes[node.dtype], device=device)
        elif node.op in {"add", "subtract", "multiply", "divide"}:
            left = _align(inputs[0], input_nodes[0].axes, node.axes)
            right = _align(inputs[1], input_nodes[1].axes, node.axes)
            value = {
                "add": torch.add,
                "subtract": torch.subtract,
                "multiply": torch.multiply,
                "divide": torch.divide,
            }[node.op](left, right)
        elif node.op in {
            "neg",
            "exp",
            "exp2",
            "log",
            "log2",
            "sqrt",
            "rsqrt",
            "abs",
            "sin",
            "cos",
            "tanh",
            "sigmoid",
            "log1p",
            "expm1",
            "logical_not",
        }:
            value = getattr(torch, node.op)(
                _align(inputs[0], input_nodes[0].axes, node.axes)
            )
        elif node.op in {
            "minimum",
            "maximum",
            "logical_and",
            "logical_or",
            "floor_divide",
            "remainder",
        }:
            value = getattr(torch, node.op)(
                *[
                    _align(item, source.axes, node.axes)
                    for item, source in zip(inputs, input_nodes, strict=True)
                ]
            )
        elif node.op == "stop_gradient":
            value = _align(inputs[0], input_nodes[0].axes, node.axes).detach()
        elif node.op == "clamp":
            value = torch.clamp(
                _align(inputs[0], input_nodes[0].axes, node.axes),
                min=attrs["min"],
                max=attrs["max"],
            )
        elif node.op == "powi":
            from cutilereduce.experimental.ir.ops import multiplication_chain

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
            precision = attrs.get("precision")
            if precision is not None and precision.operand_dtype is not None:
                # Model a low-precision operand conversion while retaining the
                # declared accumulator dtype for the reference reduction.
                lhs = inputs[0].to(dtypes[precision.operand_dtype]).to(
                    dtypes[node.dtype]
                )
                rhs = inputs[1].to(dtypes[precision.operand_dtype]).to(
                    dtypes[node.dtype]
                )
            else:
                lhs, rhs = inputs[:2]
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
            value = torch.einsum(equation, lhs, rhs)
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
