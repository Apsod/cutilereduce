"""Generated C++ execution for size-specialized experimental tensor regions."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from .cpu import CpuExpr, CpuLeaf, CpuLoop, CpuScheduleConfig, CpuScheduledPlan
from .ir import ContractionAxes, TensorNode, TensorProgram, TensorRegion
from .operator import FoldOperator, FoldProgram
from .planning import (
    FoldTopology,
    ForwardScheduleTemplate,
    MapFoldRegion,
    RegionCall,
    build_schedule_space,
)
from .passes import program_to_region
from .scheduled import (
    CommitMode,
    LocalLifetime,
    ScheduledBackwardProgram,
    schedule_backward_program,
)
from .scheduled_ssa import SpecializedSchedule, liveness, scheduled_to_ssa
from .target import TargetModel


_CPP_DTYPE = {
    "float32": "at::kFloat",
    "float64": "at::kDouble",
    "bfloat16": "at::kBFloat16",
    "float16": "at::kHalf",
    "bool": "at::kBool",
    "int32": "at::kInt",
    "int64": "at::kLong",
}


def _attr(node: TensorNode, name: str, default: Any = None) -> Any:
    return dict(node.attributes).get(name, default)


def _number(value: int | float) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float) and math.isinf(value):
        sign = "-" if value < 0 else ""
        return f"{sign}std::numeric_limits<float>::infinity()"
    return repr(value)


def _cpp_name(name: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in name)


def _axes_literal(axes: tuple[str, ...], sizes: dict[str, int]) -> str:
    return "{" + ", ".join(str(sizes[axis]) for axis in axes) + "}"


def _dims_literal(dims: tuple[int, ...]) -> str:
    return "{" + ", ".join(str(dim) for dim in dims) + "}"


def _align(expression: str, source: tuple[str, ...], target: tuple[str, ...]) -> str:
    if not source or source == target:
        return expression
    present = tuple(axis for axis in target if axis in source)
    result = expression
    if present != source:
        permutation = tuple(source.index(axis) for axis in present)
        result = f"({result}).permute({_dims_literal(permutation)})"
    for index, axis in enumerate(target):
        if axis not in source:
            result = f"({result}).unsqueeze({index})"
    return result


def _product(axes: tuple[str, ...], sizes: dict[str, int]) -> int:
    return math.prod(sizes[axis] for axis in axes)


def _canonical_matrix_operand(
    expression: str,
    source_axes: tuple[str, ...],
    target_axes: tuple[str, ...],
    shape: tuple[int, ...],
    *,
    source_dtype: str,
    target_dtype: str,
) -> str:
    """Permute and flatten a bounded region value for an ATen GEMM call."""
    if set(source_axes) != set(target_axes) or len(source_axes) != len(target_axes):
        raise ValueError(
            "contraction operand axes do not match the contraction partition"
        )
    result = expression
    if source_dtype != target_dtype:
        result = f"({result}).to({_CPP_DTYPE[target_dtype]})"
    if source_axes != target_axes:
        permutation = tuple(source_axes.index(axis) for axis in target_axes)
        result = f"({result}).permute({_dims_literal(permutation)})"
    return f"({result}).reshape({_dims_literal(shape)})"


def _emit_contraction(
    node: TensorNode,
    input_nodes: list[TensorNode],
    inputs: list[str],
    partition: ContractionAxes,
    sizes: dict[str, int],
) -> str:
    """Lower a named-axis contraction to mm/bmm through canonical B/M/N/K."""
    batch = partition.batch
    lhs_free = partition.lhs_free
    rhs_free = partition.rhs_free
    reduce = partition.reduce
    batch_size = _product(batch, sizes)
    lhs_size = _product(lhs_free, sizes)
    rhs_size = _product(rhs_free, sizes)
    reduce_size = _product(reduce, sizes)

    lhs_axes = (*batch, *lhs_free, *reduce)
    rhs_axes = (*batch, *reduce, *rhs_free)
    if batch:
        lhs_shape = (batch_size, lhs_size, reduce_size)
        rhs_shape = (batch_size, reduce_size, rhs_size)
    else:
        lhs_shape = (lhs_size, reduce_size)
        rhs_shape = (reduce_size, rhs_size)
    lhs = _canonical_matrix_operand(
        inputs[0],
        input_nodes[0].axes,
        lhs_axes,
        lhs_shape,
        source_dtype=input_nodes[0].dtype,
        target_dtype=node.dtype,
    )
    rhs = _canonical_matrix_operand(
        inputs[1],
        input_nodes[1].axes,
        rhs_axes,
        rhs_shape,
        source_dtype=input_nodes[1].dtype,
        target_dtype=node.dtype,
    )
    product = f"at::bmm({lhs}, {rhs})" if batch else f"at::mm({lhs}, {rhs})"

    canonical_axes = (*batch, *lhs_free, *rhs_free)
    canonical_shape = tuple(sizes[axis] for axis in canonical_axes)
    if node.op == "contract_accumulate":
        accumulator = _align(inputs[2], input_nodes[2].axes, canonical_axes)
        if input_nodes[2].dtype != node.dtype:
            accumulator = f"({accumulator}).to({_CPP_DTYPE[node.dtype]})"
        matrix_shape = (
            (batch_size, lhs_size, rhs_size) if batch else (lhs_size, rhs_size)
        )
        accumulator = (
            f"({accumulator}).expand({_dims_literal(canonical_shape)})"
            f".reshape({_dims_literal(matrix_shape)})"
        )
        operation = "baddbmm" if batch else "addmm"
        product = f"at::{operation}({accumulator}, {lhs}, {rhs})"
    result = f"({product}).reshape({_dims_literal(canonical_shape)})"
    if canonical_axes != node.axes:
        if set(canonical_axes) != set(node.axes):
            raise ValueError("contraction result axes do not match its partition")
        permutation = tuple(canonical_axes.index(axis) for axis in node.axes)
        result = f"({result}).permute({_dims_literal(permutation)})"
    return result


def _emit_region(
    name: str,
    region: TensorRegion,
    sizes: dict[str, int],
    *,
    nested_names: dict[int, str] | None = None,
) -> str:
    function = _cpp_name(name)
    argument_names = {node_id: _cpp_name(arg) for arg, node_id in region.arguments}
    parameters = ", ".join(
        f"const torch::Tensor& {argument_names[node_id]}"
        for _, node_id in region.arguments
    )
    lines = [f"std::vector<torch::Tensor> {function}({parameters}) {{"]
    values: dict[int, str] = dict(argument_names)
    for node in region.nodes:
        if node.id in argument_names:
            continue
        if node.id in values:
            continue
        if node.op.startswith("associative_scan_"):
            combine = _attr(node, "combine_region")
            if not isinstance(combine, TensorProgram) or nested_names is None:
                raise TypeError("associative scan is missing its combine region")
            group = tuple(
                candidate
                for candidate in region.nodes
                if candidate.op.startswith("associative_scan_")
                and candidate.inputs == node.inputs
                and _attr(candidate, "combine_region") == combine
                and _attr(candidate, "over") == _attr(node, "over")
            )
            scan_axis = _attr(node, "over")[0]
            prefix_nodes = {
                tuple(_attr(candidate, "result_path")): candidate
                for candidate in group
                if candidate.op == "associative_scan_prefix"
            }
            final_nodes = {
                tuple(_attr(candidate, "result_path")): candidate
                for candidate in group
                if candidate.op == "associative_scan_final"
            }
            paths = tuple(prefix_nodes)
            leaf_count = len(paths)
            factor_ids = node.inputs[:leaf_count]
            initial_ids = node.inputs[leaf_count:]
            if len(initial_ids) != leaf_count:
                raise ValueError("structured scan inputs do not match results")
            scan_index = f"scan_{node.id}"
            state_names = {
                path: f"state_{node.id}_{index}" for index, path in enumerate(paths)
            }
            for path, initial_id in zip(paths, initial_ids, strict=True):
                lines.append(f"  auto {state_names[path]} = {values[initial_id]};")
                prefix_node = prefix_nodes[path]
                prefix_name = f"v{prefix_node.id}"
                lines.append(
                    f"  auto {prefix_name} = at::empty({_axes_literal(prefix_node.axes, sizes)}, at::TensorOptions().dtype({_CPP_DTYPE[prefix_node.dtype]}));"
                )
                values[prefix_node.id] = prefix_name
            lines.append(
                f"  for (int64_t {scan_index} = 0; {scan_index} < {sizes[scan_axis]}; ++{scan_index}) {{"
            )
            factor_names = {}
            for path, factor_id in zip(paths, factor_ids, strict=True):
                factor_node = region.nodes[factor_id]
                dimension = factor_node.axes.index(scan_axis)
                factor_name = f"factor_{node.id}_{len(factor_names)}"
                lines.append(
                    f"    auto {factor_name} = {values[factor_id]}.select({dimension}, {scan_index});"
                )
                factor_names[path] = factor_name
                prefix_node = prefix_nodes[path]
                prefix_dimension = prefix_node.axes.index(scan_axis)
                lines.append(
                    f"    {values[prefix_node.id]}.select({prefix_dimension}, {scan_index}).copy_({state_names[path]});"
                )
            combine_region = program_to_region(combine)
            combine_arguments = []
            for argument, _ in combine_region.arguments:
                side, *path_parts = argument.split(".")
                path = tuple(path_parts)
                combine_arguments.append(
                    state_names[path] if side == "left" else factor_names[path]
                )
            combined_name = f"combined_{node.id}"
            lines.append(
                f"    auto {combined_name} = {nested_names[id(combine)]}({', '.join(combine_arguments)});"
            )
            for index, path in enumerate(paths):
                lines.append(f"    {state_names[path]} = {combined_name}[{index}];")
            lines.append("  }")
            for path, final_node in final_nodes.items():
                values[final_node.id] = state_names[path]
            continue
        variable = f"v{node.id}"
        inputs = [values[node_id] for node_id in node.inputs]
        input_nodes = [region.nodes[node_id] for node_id in node.inputs]
        match node.op:
            case "domain_valid":
                expression = (
                    f"at::ones({_axes_literal(node.axes, sizes)}, "
                    f"at::TensorOptions().dtype(at::kBool))"
                )
            case "axis_index":
                axis = _attr(node, "axis")
                expression = (
                    f"at::arange({sizes[axis]}, "
                    f"at::TensorOptions().dtype({_CPP_DTYPE[node.dtype]}))"
                )
            case "constant":
                expression = (
                    f"at::scalar_tensor({_number(_attr(node, 'value'))}, "
                    f"at::TensorOptions().dtype({_CPP_DTYPE[node.dtype]}))"
                )
            case "zero":
                expression = (
                    f"at::zeros({_axes_literal(node.axes, sizes)}, "
                    f"at::TensorOptions().dtype({_CPP_DTYPE[node.dtype]}))"
                )
            case "full":
                expression = (
                    f"at::full({_axes_literal(node.axes, sizes)}, "
                    f"{_number(_attr(node, 'value'))}, "
                    f"at::TensorOptions().dtype({_CPP_DTYPE[node.dtype]}))"
                )
            case "add" | "subtract" | "multiply" | "divide":
                symbols = {"add": "+", "subtract": "-", "multiply": "*", "divide": "/"}
                left = _align(inputs[0], input_nodes[0].axes, node.axes)
                right = _align(inputs[1], input_nodes[1].axes, node.axes)
                expression = f"({left}) {symbols[node.op]} ({right})"
            case "neg" | "exp" | "log" | "sqrt" | "rsqrt" | "abs" | "sin" | "cos" | "tanh" | "sigmoid" | "log1p" | "expm1" | "logical_not":
                operand = _align(inputs[0], input_nodes[0].axes, node.axes)
                expression = f"at::{node.op}({operand})"
            case "minimum" | "maximum" | "logical_and" | "logical_or" | "floor_divide" | "remainder":
                operands = [_align(value, source.axes, node.axes) for value, source in zip(inputs, input_nodes, strict=True)]
                expression = f"at::{node.op}({', '.join(operands)})"
            case "stop_gradient":
                expression = f"({_align(inputs[0], input_nodes[0].axes, node.axes)}).detach()"
            case "clamp":
                expression = _align(inputs[0], input_nodes[0].axes, node.axes)
                for bound in ("min", "max"):
                    value = _attr(node, bound)
                    if value is not None:
                        expression = f"at::clamp_{bound}({expression}, {_number(value)})"
            case "powi":
                from .ops import multiplication_chain
                exponent = _attr(node, "exponent")
                powers = [_align(inputs[0], input_nodes[0].axes, node.axes)]
                for left, right in multiplication_chain(abs(exponent)) if exponent else ():
                    name = f"power_{node.id}_{len(powers)}"
                    lines.append(f"  auto {name} = ({powers[left]}) * ({powers[right]});")
                    powers.append(name)
                expression = powers[-1] if exponent else f"at::ones_like({powers[0]})"
                if exponent < 0:
                    expression = f"at::reciprocal({expression})"
            case "cast":
                expression = f"({inputs[0]}).to({_CPP_DTYPE[node.dtype]})"
            case "where" | "mask_identity":
                operands = [
                    _align(value, input_node.axes, node.axes)
                    for value, input_node in zip(inputs, input_nodes, strict=True)
                ]
                expression = f"at::where({operands[0]}, {operands[1]}, {operands[2]})"
            case op if op.startswith("compare_"):
                comparison = {
                    "compare_eq": "==",
                    "compare_ne": "!=",
                    "compare_lt": "<",
                    "compare_le": "<=",
                    "compare_gt": ">",
                    "compare_ge": ">=",
                    "compare_equal": "==",
                    "compare_not_equal": "!=",
                    "compare_less": "<",
                    "compare_less_equal": "<=",
                    "compare_greater": ">",
                    "compare_greater_equal": ">=",
                }[op]
                target = node.axes
                left = _align(inputs[0], input_nodes[0].axes, target)
                right = _align(inputs[1], input_nodes[1].axes, target)
                expression = f"({left}) {comparison} ({right})"
            case op if op.startswith("reduce_"):
                over = _attr(node, "over")
                dims = tuple(input_nodes[0].axes.index(axis) for axis in over)
                if op == "reduce_sum":
                    expression = f"at::sum({inputs[0]}, {_dims_literal(dims)})"
                else:
                    expression = f"at::amax({inputs[0]}, {_dims_literal(dims)}, false)"
            case op if op.startswith("scan_"):
                over = _attr(node, "over")
                dimension = input_nodes[0].axes.index(over)
                reverse = bool(_attr(node, "reverse", False))
                exclusive = bool(_attr(node, "exclusive", False))
                source = inputs[0]
                if reverse:
                    source = f"at::flip({source}, {{{dimension}}})"
                expression = f"at::cumsum({source}, {dimension})"
                if exclusive:
                    expression = f"({expression}) - ({source})"
                if reverse:
                    expression = f"at::flip({expression}, {{{dimension}}})"
            case "contract" | "contract_accumulate":
                partition = _attr(node, "contraction_axes")
                if not isinstance(partition, ContractionAxes):
                    raise TypeError("contract node has invalid axis metadata")
                expression = _emit_contraction(
                    node, input_nodes, inputs, partition, sizes
                )
            case _:
                raise NotImplementedError(
                    f"CPU C++ lowering does not support tensor op {node.op!r}"
                )
        lines.append(f"  auto {variable} = {expression};")
        values[node.id] = variable
    results = ", ".join(values[node_id] for _, node_id in region.results)
    lines.append(f"  return {{{results}}};")
    lines.append("}")
    return "\n".join(lines)


def _scheduled_local_sizes(
    program: ScheduledBackwardProgram,
    sizes: dict[str, int],
) -> tuple[dict[str, int], ...]:
    axes = (*program.program_axes, program.loop_axis)
    choices = []
    for axis in axes:
        remainder = axis.extent % axis.tile
        choices.append((axis.tile,) if remainder == 0 else (axis.tile, remainder))
    variants = []
    for extents in itertools.product(*choices):
        local = dict(sizes)
        for axis, extent in zip(axes, extents, strict=True):
            local[axis.axis] = extent
        if local not in variants:
            variants.append(local)
    return tuple(variants)


def _emit_scheduled_backward(
    problem: FoldProgram,
    program: ScheduledBackwardProgram,
    sizes: dict[str, int],
    *,
    nested_names: dict[int, str],
) -> tuple[list[str], str]:
    """Emit a program-local loop with tile-resident gradients and commits."""
    if not liveness(scheduled_to_ssa(program)):
        raise ValueError("scheduled backward must define SSA values")
    if problem.requires_prefix or program.threads_prefix:
        raise NotImplementedError(
            "scheduled CPU lowering initially supports prefix-free backward"
        )
    region = problem.map_finalize_map_vjp
    if region is None:
        raise ValueError("scheduled CPU backward requires a commutative region")
    if tuple(name for name, _ in region.arguments) != tuple(
        binding.name for binding in program.arguments
    ):
        raise ValueError("scheduled arguments do not match backward region")
    result_indices = {name: index for index, (name, _) in enumerate(region.results)}
    for gradient in program.gradients:
        if gradient.result_name not in result_indices:
            raise ValueError(f"missing scheduled result {gradient.result_name!r}")
        node = region.nodes[dict(region.results)[gradient.result_name]]
        if (
            gradient.commit_mode is CommitMode.ATOMIC_ADD_RELAXED
            and node.dtype != "float32"
        ):
            raise ValueError(
                "scheduled CPU atomic gradients initially require float32, got "
                f"{gradient.input_name!r}: {node.dtype}"
            )

    local_sizes = _scheduled_local_sizes(program, sizes)
    variant_names = [
        f"scheduled_backward_region_{index}" for index in range(len(local_sizes))
    ]
    variants = [
        _emit_region(
            name,
            region,
            shape,
            nested_names=nested_names,
        )
        for name, shape in zip(variant_names, local_sizes, strict=True)
    ]

    parameters = ", ".join(
        f"const torch::Tensor& {_cpp_name(name)}" for name, _ in region.arguments
    )
    lines = [f"std::vector<torch::Tensor> scheduled_backward({parameters}) {{"]
    for gradient in program.gradients:
        node = region.nodes[dict(region.results)[gradient.result_name]]
        lines.append(
            f"  auto grad_{_cpp_name(gradient.input_name)} = at::zeros("
            f"{_axes_literal(gradient.axes, sizes)}, "
            f"at::TensorOptions().dtype({_CPP_DTYPE[node.dtype]}));"
        )
    program_count = math.prod(axis.count for axis in program.program_axes)
    lines.append(
        f"  at::parallel_for(0, {program_count}, 1, [&](int64_t begin, int64_t end) {{"
    )
    lines.append(
        "    for (int64_t program_id = begin; program_id < end; ++program_id) {"
    )
    lines.append("      int64_t program_remainder = program_id;")
    for axis in reversed(program.program_axes):
        name = _cpp_name(axis.axis)
        lines.extend(
            (
                f"      const int64_t index_{name} = program_remainder % {axis.count};",
                f"      program_remainder /= {axis.count};",
                f"      const int64_t start_{name} = index_{name} * {axis.tile};",
                f"      const int64_t size_{name} = std::min<int64_t>({axis.tile}, {axis.extent} - start_{name});",
            )
        )
    for gradient in program.gradients:
        if gradient.lifetime is LocalLifetime.PERSISTENT:
            lines.append(f"      torch::Tensor local_{_cpp_name(gradient.input_name)};")
    loop = program.loop_axis
    loop_name = _cpp_name(loop.axis)
    lines.extend(
        (
            f"      for (int64_t start_{loop_name} = 0; start_{loop_name} < {loop.extent}; start_{loop_name} += {loop.tile}) {{",
            f"        const int64_t size_{loop_name} = std::min<int64_t>({loop.tile}, {loop.extent} - start_{loop_name});",
        )
    )
    scheduled_axes = {axis.axis: axis for axis in (*program.program_axes, loop)}
    for binding in program.arguments:
        variable = _cpp_name(binding.name)
        lines.append(f"        auto tile_{variable} = {variable};")
        for axis in binding.sliced_axes:
            dimension = binding.axes.index(axis)
            axis_name = _cpp_name(axis)
            lines.append(
                f"        tile_{variable} = tile_{variable}.narrow({dimension}, "
                f"start_{axis_name}, size_{axis_name});"
            )
    call_arguments = ", ".join(
        f"tile_{_cpp_name(binding.name)}" for binding in program.arguments
    )
    for index, (name, shape) in enumerate(zip(variant_names, local_sizes, strict=True)):
        conditions = (
            " && ".join(
                f"size_{_cpp_name(axis)} == {shape[axis]}" for axis in scheduled_axes
            )
            or "true"
        )
        keyword = "if" if index == 0 else "else if"
        lines.append(f"        {keyword} ({conditions}) {{")
        lines.append(f"          auto partial = {name}({call_arguments});")

        def destination_lines(gradient, source: str, indent: str) -> list[str]:
            destination = f"destination_{_cpp_name(gradient.input_name)}"
            result = [
                f"{indent}auto {destination} = grad_{_cpp_name(gradient.input_name)};"
            ]
            for axis in gradient.axes:
                if axis not in scheduled_axes:
                    continue
                dimension = gradient.axes.index(axis)
                axis_name = _cpp_name(axis)
                result.append(
                    f"{indent}{destination} = {destination}.narrow({dimension}, "
                    f"start_{axis_name}, size_{axis_name});"
                )
            if gradient.commit_mode is CommitMode.STORE:
                result.append(f"{indent}{destination}.copy_({source});")
            elif gradient.commit_mode is CommitMode.ATOMIC_ADD_RELAXED:
                result.append(f"{indent}atomic_add_relaxed({destination}, {source});")
            else:
                raise ValueError("scheduled write policy is unresolved")
            return result

        for gradient in program.gradients:
            result_index = result_indices[gradient.result_name]
            source = f"partial[{result_index}]"
            local = f"local_{_cpp_name(gradient.input_name)}"
            if gradient.lifetime is LocalLifetime.PERSISTENT:
                lines.extend(
                    (
                        f"          if (!{local}.defined()) {local} = at::zeros_like({source});",
                        f"          {local}.add_({source});",
                    )
                )
            else:
                lines.extend(destination_lines(gradient, source, "          "))
        lines.append("        }")
    lines.append(
        '        else { TORCH_CHECK(false, "missing scheduled tile variant"); }'
    )
    lines.append("      }")
    for gradient in program.gradients:
        if gradient.lifetime is LocalLifetime.PERSISTENT:
            lines.extend(
                destination_lines(
                    gradient,
                    f"local_{_cpp_name(gradient.input_name)}",
                    "      ",
                )
            )
    lines.extend(("    }", "  });"))
    results = ", ".join(
        f"grad_{_cpp_name(gradient.input_name)}" for gradient in program.gradients
    )
    lines.extend((f"  return {{{results}}};", "}"))
    return variants, "\n".join(lines)


def _emit_scheduled_forward_tile(
    problem: FoldProgram,
    scheduled: CpuScheduledPlan,
    sizes: dict[str, int],
    extents: tuple[int, ...],
    *,
    nested_names: dict[int, str],
    function_name: str,
    name_suffix: str,
    map_fold_name: str,
    combine_name: str,
    identity_name: str,
    project_name: str,
) -> tuple[list[str], str, int]:
    """Emit the scheduled fold for one independently owned output tile."""
    fold_axis = problem.spec.fold_axes[0]
    full_extent = sizes[fold_axis]
    variants = []
    variant_names = {}
    for extent in extents:
        if extent == full_extent:
            variant_names[extent] = map_fold_name
            continue
        local_sizes = dict(sizes)
        local_sizes[fold_axis] = extent
        name = f"scheduled_map_fold_{extent}_{name_suffix}"
        variant_names[extent] = name
        variants.append(
            _emit_region(
                name,
                problem.map_fold,
                local_sizes,
                nested_names=nested_names,
            )
        )

    accumulate_names = {}
    for extent in extents:
        local_sizes = dict(sizes)
        local_sizes[fold_axis] = extent
        name = f"scheduled_map_fold_combine_{extent}_{name_suffix}"
        accumulate_names[extent] = name
        variants.append(
            _emit_region(
                name, problem.map_fold_combine, local_sizes, nested_names=nested_names
            )
        )

    parameters = ", ".join(
        f"const torch::Tensor& {_cpp_name(name)}" for name in problem.spec.inputs
    )
    lines = [f"std::vector<torch::Tensor> {function_name}({parameters}) {{"]
    lines.append(
        "  auto run_map_fold = [&](int64_t start, int64_t extent) "
        "-> std::vector<torch::Tensor> {"
    )
    for index, (name, spec) in enumerate(problem.spec.inputs.items()):
        variable = _cpp_name(name)
        if fold_axis in spec.axes:
            dimension = spec.axes.index(fold_axis)
            lines.append(
                f"    auto sliced_{index} = {variable}.narrow({dimension}, start, extent).contiguous();"
            )
        else:
            lines.append(f"    const auto& sliced_{index} = {variable};")
    map_arguments = []
    input_indices = {name: index for index, name in enumerate(problem.spec.inputs)}
    for name, _ in problem.map_fold.arguments:
        if name not in input_indices:
            raise ValueError(f"map_fold argument {name!r} is not an operator input")
        map_arguments.append(f"sliced_{input_indices[name]}")
    for extent in extents:
        lines.append(
            f"    if (extent == {extent}) return {variant_names[extent]}("
            f"{', '.join(map_arguments)});"
        )
    lines.extend(
        (
            '    TORCH_CHECK(false, "missing scheduled map-fold tile variant");',
            "  };",
        )
    )

    carrier_indices = {
        name: index for index, (name, _) in enumerate(problem.map_fold.results)
    }
    carrier_nodes = {
        name: problem.map_fold.nodes[node_id] for name, node_id in problem.map_fold.results
    }
    lines.append(
        "  auto run_map_fold_combine = [&](int64_t start, int64_t extent, "
        "const std::vector<torch::Tensor>& state) -> std::vector<torch::Tensor> {"
    )
    for index, (name, spec) in enumerate(problem.spec.inputs.items()):
        variable = _cpp_name(name)
        if fold_axis in spec.axes:
            lines.append(
                f"    auto sliced_{index} = {variable}.narrow("
                f"{spec.axes.index(fold_axis)}, start, extent).contiguous();"
            )
        else:
            lines.append(f"    const auto& sliced_{index} = {variable};")
    accumulate_arguments = []
    for name, node_id in problem.map_fold_combine.arguments:
        target_axes = problem.map_fold_combine.nodes[node_id].axes
        if name in input_indices:
            accumulate_arguments.append(
                _align(
                    f"sliced_{input_indices[name]}",
                    problem.spec.inputs[name].axes,
                    target_axes,
                )
            )
        elif name.startswith("accumulator."):
            result_name = "execution." + name.removeprefix("accumulator.")
            accumulate_arguments.append(
                _align(
                    f"state[{carrier_indices[result_name]}]",
                    carrier_nodes[result_name].axes,
                    target_axes,
                )
            )
        else:
            raise ValueError(f"unrecognized map_fold_combine argument {name!r}")
    accumulate_indices = {
        name: index for index, (name, _) in enumerate(problem.map_fold_combine.results)
    }
    accumulate_nodes = dict(problem.map_fold_combine.results)
    reordered = ", ".join(
        _align(
            f"result[{accumulate_indices[name]}]",
            problem.map_fold_combine.nodes[accumulate_nodes[name]].axes,
            carrier_nodes[name].axes,
        )
        for name in carrier_indices
    )
    for extent in extents:
        lines.extend(
            (
                f"    if (extent == {extent}) {{",
                f"      auto result = {accumulate_names[extent]}({', '.join(accumulate_arguments)});",
                f"      return {{{reordered}}};",
                "    }",
            )
        )
    lines.extend(
        ('    TORCH_CHECK(false, "missing scheduled accumulate tile variant");', "  };")
    )
    identity_indices = {
        name: index for index, (name, _) in enumerate(problem.execution_identity.results)
    }
    identity_nodes = dict(problem.execution_identity.results)
    identity_parts = []
    for name, node_id in problem.map_fold.results:
        carrier = problem.map_fold.nodes[node_id]
        identity_node = problem.execution_identity.nodes[identity_nodes[name]]
        aligned = _align(
            f"identity[{identity_indices[name]}]", identity_node.axes, carrier.axes
        )
        identity_parts.append(f"({aligned}).expand({_axes_literal(carrier.axes, sizes)})")
    identity_values = ", ".join(identity_parts)
    lines.extend(
        (
            "  auto initial_state = [&]() -> std::vector<torch::Tensor> {",
            f"    auto identity = {identity_name}();",
            f"    return {{{identity_values}}};",
            "  };",
        )
    )
    combine_arguments = []
    for name, node_id in problem.execution_combine.arguments:
        side, suffix = name.split(".", 1)
        result_name = f"execution.{suffix}"
        if result_name not in carrier_indices or side not in {"left", "right"}:
            raise ValueError("execution_combine arguments do not match map_fold results")
        combine_arguments.append(
            _align(
                f"{side}[{carrier_indices[result_name]}]",
                carrier_nodes[result_name].axes,
                problem.execution_combine.nodes[node_id].axes,
            )
        )
    combine_indices = {
        name: index for index, (name, _) in enumerate(problem.execution_combine.results)
    }
    combine_nodes = dict(problem.execution_combine.results)
    combined_values = ", ".join(
        _align(
            f"result[{combine_indices[name]}]",
            problem.execution_combine.nodes[combine_nodes[name]].axes,
            carrier_nodes[name].axes,
        )
        for name in carrier_indices
    )
    lines.extend(
        (
            "  auto combine_carriers = [&](const std::vector<torch::Tensor>& left,",
            "                              const std::vector<torch::Tensor>& right) -> std::vector<torch::Tensor> {",
            f"    auto result = {combine_name}({', '.join(combine_arguments)});",
            f"    return {{{combined_values}}};",
            "  };",
            "  auto combine_many = [&](std::vector<std::vector<torch::Tensor>> values, bool tree) {",
            '    TORCH_CHECK(!values.empty(), "scheduled fold produced no carriers");',
            "    if (!tree) {",
            "      auto state = std::move(values[0]);",
            "      for (size_t i = 1; i < values.size(); ++i)",
            "        state = combine_carriers(state, values[i]);",
            "      return state;",
            "    }",
            "    while (values.size() > 1) {",
            "      std::vector<std::vector<torch::Tensor>> next;",
            "      next.reserve((values.size() + 1) / 2);",
            "      for (size_t i = 0; i < values.size(); i += 2) {",
            "        if (i + 1 == values.size()) next.push_back(std::move(values[i]));",
            "        else next.push_back(combine_carriers(values[i], values[i + 1]));",
            "      }",
            "      values = std::move(next);",
            "    }",
            "    return std::move(values[0]);",
            "  };",
        )
    )

    lambda_names: dict[int, str] = {}
    lambda_definitions: list[str] = []

    def emit_expression(expression: CpuExpr) -> str:
        key = id(expression)
        if key in lambda_names:
            return lambda_names[key]
        name = f"eval_carrier_{len(lambda_names)}"
        lambda_names[key] = name
        if isinstance(expression, CpuLeaf):
            if not isinstance(expression.expression, MapFoldRegion):
                raise TypeError("scheduled forward carrier leaf must call map_fold")
            body = ["    return run_map_fold(start, extent);"]
        elif isinstance(expression, CpuLoop) and expression.operation == "carrier_fold":
            if (
                expression.iteration.loop.value != "parallel"
                and expression.topology is FoldTopology.ONLINE
            ):
                grain = expression.iteration.grain
                if isinstance(expression.body, CpuLeaf):
                    if not isinstance(expression.body.expression, MapFoldRegion):
                        raise TypeError("online fold leaf must call map_fold")
                    update = "state = run_map_fold_combine(start + offset, local, state);"
                else:
                    child = emit_expression(expression.body)
                    update = f"state = combine_carriers(state, {child}(start + offset, local));"
                body = [
                    "    auto state = initial_state();",
                    f"    for (int64_t offset = 0; offset < extent; offset += {grain}) {{",
                    f"      const int64_t local = std::min<int64_t>({grain}, extent - offset);",
                    f"      {update}",
                    "    }",
                    "    return state;",
                ]
                definition = [
                    f"  std::function<std::vector<torch::Tensor>(int64_t, int64_t)> {name};",
                    f"  {name} = [&](int64_t start, int64_t extent) {{",
                    *body,
                    "  };",
                ]
                lambda_definitions.extend(definition)
                return name
            child = emit_expression(expression.body)
            grain = expression.iteration.grain
            tree = "true" if expression.topology is FoldTopology.TREE else "false"
            body = [
                f"    const int64_t count = (extent + {grain} - 1) / {grain};",
                "    std::vector<std::vector<torch::Tensor>> values(count);",
            ]
            loop_body = [
                f"const int64_t offset = index * {grain};",
                f"const int64_t local = std::min<int64_t>({grain}, extent - offset);",
                f"values[index] = {child}(start + offset, local);",
            ]
            if expression.iteration.loop.value == "parallel":
                body.extend(
                    [
                        "    at::parallel_for(0, count, 1, [&](int64_t begin, int64_t end) {",
                        "      for (int64_t index = begin; index < end; ++index) {",
                        *(f"        {line}" for line in loop_body),
                        "      }",
                        "    });",
                    ]
                )
            else:
                body.extend(
                    [
                        "    for (int64_t index = 0; index < count; ++index) {",
                        *(f"      {line}" for line in loop_body),
                        "    }",
                    ]
                )
            body.append(f"    return combine_many(std::move(values), {tree});")
        else:
            raise TypeError("unsupported scheduled forward carrier expression")
        definition = [
            f"  std::function<std::vector<torch::Tensor>(int64_t, int64_t)> {name};",
            f"  {name} = [&](int64_t start, int64_t extent) {{",
            *body,
            "  };",
        ]
        lambda_definitions.extend(definition)
        return name

    root = scheduled.forward[0].bindings[0].expression
    prefix_count = 0
    if isinstance(root, CpuLoop) and root.operation == "map":
        child = emit_expression(root.body)
        grain = root.iteration.grain
        lines.extend(lambda_definitions)
        lines.extend(
            (
                f"  const int64_t partition_count = ({full_extent} + {grain} - 1) / {grain};",
                "  std::vector<std::vector<torch::Tensor>> partials(partition_count);",
            )
        )
        map_body = [
            f"const int64_t start = index * {grain};",
            f"const int64_t extent = std::min<int64_t>({grain}, {full_extent} - start);",
            f"partials[index] = {child}(start, extent);",
        ]
        if root.iteration.loop.value == "parallel":
            lines.extend(
                (
                    "  at::parallel_for(0, partition_count, 1, [&](int64_t begin, int64_t end) {",
                    "    for (int64_t index = begin; index < end; ++index) {",
                    *(f"      {line}" for line in map_body),
                    "    }",
                    "  });",
                )
            )
        else:
            lines.extend(
                (
                    "  for (int64_t index = 0; index < partition_count; ++index) {",
                    *(f"    {line}" for line in map_body),
                    "  }",
                )
            )
        combine_expression = scheduled.forward[1].bindings[0].expression
        if (
            isinstance(combine_expression, CpuLoop)
            and combine_expression.operation == "exclusive_scan"
        ):
            prefix_count = root.iteration.count
            if problem.execution_identity.arguments:
                raise ValueError("execution identity must not take arguments")
            lines.extend(
                (
                    f"  auto identity = {identity_name}();",
                    "  std::vector<std::vector<torch::Tensor>> prefixes;",
                    "  prefixes.reserve(partials.size());",
                    "  std::vector<torch::Tensor> prefix_state;",
                    "  prefix_state.reserve(identity.size());",
                    "  for (size_t i = 0; i < identity.size(); ++i) {",
                    "    std::vector<int64_t> shape(partials[0][i].dim() - identity[i].dim(), 1);",
                    "    shape.insert(shape.end(), identity[i].sizes().begin(), identity[i].sizes().end());",
                    "    prefix_state.push_back(identity[i].reshape(shape).expand_as(partials[0][i]).clone());",
                    "  }",
                    "  for (const auto& partial : partials) {",
                    "    prefixes.push_back(prefix_state);",
                    "    prefix_state = combine_carriers(prefix_state, partial);",
                    "  }",
                )
            )
        topology = (
            combine_expression.topology
            if isinstance(combine_expression, CpuLoop)
            and combine_expression.topology is not None
            else FoldTopology.ONLINE
        )
        tree = "true" if topology is FoldTopology.TREE else "false"
        lines.append(f"  auto execution = combine_many(partials, {tree});")
    else:
        evaluator = emit_expression(root)
        lines.extend(lambda_definitions)
        lines.append(f"  auto execution = {evaluator}(0, {full_extent});")

    project_arguments = []
    for name, node_id in problem.project_output.arguments:
        candidate = name.replace("saved_execution.", "execution.")
        if candidate not in carrier_indices:
            raise ValueError("project_output arguments do not match the carrier")
        project_arguments.append(
            _align(
                f"execution[{carrier_indices[candidate]}]",
                carrier_nodes[candidate].axes,
                problem.project_output.nodes[node_id].axes,
            )
        )
    lines.append(f"  auto outputs = {project_name}({', '.join(project_arguments)});")
    lines.extend(
        (
            "  std::vector<torch::Tensor> result;",
            "  result.insert(result.end(), outputs.begin(), outputs.end());",
            "  result.insert(result.end(), execution.begin(), execution.end());",
        )
    )
    if prefix_count:
        lines.extend(
            (
                "  for (const auto& prefix : prefixes)",
                "    result.insert(result.end(), prefix.begin(), prefix.end());",
            )
        )
    lines.extend(("  return result;", "}"))
    return variants, "\n".join(lines), prefix_count


def _emit_scheduled_forward(
    problem: FoldProgram,
    scheduled: CpuScheduledPlan,
    sizes: dict[str, int],
    extents: tuple[int, ...],
    *,
    nested_names: dict[int, str],
) -> tuple[list[str], str]:
    """Tile all forward batch axes and emit one native owner-program grid."""
    owner_axes = problem.spec.batch_axes
    stage_tiles = dict(scheduled.forward[0].tile_sizes)
    tile_sizes = {
        axis: min(sizes[axis], stage_tiles.get(axis, sizes[axis]))
        for axis in owner_axes
    }
    extent_choices = []
    for axis in owner_axes:
        tile = tile_sizes[axis]
        remainder = sizes[axis] % tile
        extent_choices.append((tile,) if remainder == 0 else (tile, remainder))
    owner_shapes = tuple(itertools.product(*extent_choices)) if owner_axes else ((),)

    functions: list[str] = []
    tile_functions: dict[tuple[int, ...], str] = {}
    prefix_count: int | None = None
    for shape_index, shape in enumerate(owner_shapes):
        local_sizes = dict(sizes)
        local_sizes.update(zip(owner_axes, shape, strict=True))
        suffix = f"owner_{shape_index}"
        names = {
            "map": f"map_fold_{suffix}",
            "combine": f"execution_combine_{suffix}",
            "identity": f"execution_identity_{suffix}",
            "project": f"project_output_{suffix}",
            "driver": f"scheduled_forward_{suffix}",
        }
        functions.extend(
            _emit_region(
                names[key],
                region,
                local_sizes,
                nested_names=nested_names,
            )
            for key, region in (
                ("map", problem.map_fold),
                ("combine", problem.execution_combine),
                ("identity", problem.execution_identity),
                ("project", problem.project_output),
            )
        )
        variants, driver, local_prefix_count = _emit_scheduled_forward_tile(
            problem,
            scheduled,
            local_sizes,
            extents,
            nested_names=nested_names,
            function_name=names["driver"],
            name_suffix=suffix,
            map_fold_name=names["map"],
            combine_name=names["combine"],
            identity_name=names["identity"],
            project_name=names["project"],
        )
        functions.extend((*variants, driver))
        tile_functions[shape] = names["driver"]
        if prefix_count is None:
            prefix_count = local_prefix_count
        elif prefix_count != local_prefix_count:
            raise ValueError("owner-tile variants disagree on prefix count")
    prefix_count = prefix_count or 0

    parameters = ", ".join(
        f"const torch::Tensor& {_cpp_name(name)}" for name in problem.spec.inputs
    )
    lines = [f"std::vector<torch::Tensor> scheduled_forward({parameters}) {{"]
    output_nodes = [
        problem.project_output.nodes[node_id]
        for _, node_id in problem.project_output.results
    ]
    carrier_nodes = [
        problem.map_fold.nodes[node_id] for _, node_id in problem.map_fold.results
    ]
    for index, node in enumerate(output_nodes):
        lines.append(
            f"  auto output_{index} = at::empty({_axes_literal(node.axes, sizes)}, "
            f"at::TensorOptions().dtype({_CPP_DTYPE[node.dtype]}));"
        )
    for index, node in enumerate(carrier_nodes):
        lines.append(
            f"  auto carrier_{index} = at::empty({_axes_literal(node.axes, sizes)}, "
            f"at::TensorOptions().dtype({_CPP_DTYPE[node.dtype]}));"
        )
    for prefix in range(prefix_count):
        for index, node in enumerate(carrier_nodes):
            lines.append(
                f"  auto prefix_{prefix}_{index} = at::empty("
                f"{_axes_literal(node.axes, sizes)}, "
                f"at::TensorOptions().dtype({_CPP_DTYPE[node.dtype]}));"
            )
    owner_count = math.prod(
        math.ceil(sizes[axis] / tile_sizes[axis]) for axis in owner_axes
    )
    lines.extend(
        (
            f"  at::parallel_for(0, {owner_count}, 1, [&](int64_t begin, int64_t end) {{",
            "    for (int64_t owner = begin; owner < end; ++owner) {",
            "      int64_t owner_remainder = owner;",
        )
    )
    for axis in reversed(owner_axes):
        axis_name = _cpp_name(axis)
        count = math.ceil(sizes[axis] / tile_sizes[axis])
        lines.extend(
            (
                f"      const int64_t tile_index_{axis_name} = owner_remainder % {count};",
                f"      owner_remainder /= {count};",
                f"      const int64_t start_{axis_name} = tile_index_{axis_name} * {tile_sizes[axis]};",
                f"      const int64_t extent_{axis_name} = std::min<int64_t>({tile_sizes[axis]}, {sizes[axis]} - start_{axis_name});",
            )
        )
    input_arguments = []
    for input_index, (input_name, spec) in enumerate(problem.spec.inputs.items()):
        variable = _cpp_name(input_name)
        lines.append(f"      auto tile_input_{input_index} = {variable};")
        for axis in owner_axes:
            if axis not in spec.axes:
                continue
            dimension = spec.axes.index(axis)
            axis_name = _cpp_name(axis)
            lines.append(
                f"      tile_input_{input_index} = tile_input_{input_index}.narrow("
                f"{dimension}, start_{axis_name}, extent_{axis_name});"
            )
        lines.append(f"      tile_input_{input_index} = tile_input_{input_index}.contiguous();")
        input_arguments.append(f"tile_input_{input_index}")
    for shape_index, shape in enumerate(owner_shapes):
        condition = " && ".join(
            f"extent_{_cpp_name(axis)} == {extent}"
            for axis, extent in zip(owner_axes, shape, strict=True)
        ) or "true"
        keyword = "if" if shape_index == 0 else "else if"
        lines.append(f"      {keyword} ({condition}) {{")
        lines.append(
            f"        auto local = {tile_functions[shape]}({', '.join(input_arguments)});"
        )
        result_nodes = [
            *output_nodes,
            *carrier_nodes,
            *(node for _ in range(prefix_count) for node in carrier_nodes),
        ]
        result_variables = [
            *(f"output_{index}" for index in range(len(output_nodes))),
            *(f"carrier_{index}" for index in range(len(carrier_nodes))),
            *(
                f"prefix_{prefix}_{index}"
                for prefix in range(prefix_count)
                for index in range(len(carrier_nodes))
            ),
        ]
        for result_index, (node, destination) in enumerate(
            zip(result_nodes, result_variables, strict=True)
        ):
            lines.append(f"        auto destination_{result_index} = {destination};")
            for axis in owner_axes:
                if axis not in node.axes:
                    continue
                dimension = node.axes.index(axis)
                axis_name = _cpp_name(axis)
                lines.append(
                    f"        destination_{result_index} = destination_{result_index}.narrow("
                    f"{dimension}, start_{axis_name}, extent_{axis_name});"
                )
            lines.append(
                f"        destination_{result_index}.copy_(local[{result_index}]);"
            )
        lines.append("      }")
    lines.extend(
        (
            '      else { TORCH_CHECK(false, "missing scheduled owner-tile variant"); }',
            "    }",
            "  });",
            "  std::vector<torch::Tensor> result;",
        )
    )
    for index in range(len(output_nodes)):
        lines.append(f"  result.push_back(output_{index});")
    for index in range(len(carrier_nodes)):
        lines.append(f"  result.push_back(carrier_{index});")
    for prefix in range(prefix_count):
        for index in range(len(carrier_nodes)):
            lines.append(f"  result.push_back(prefix_{prefix}_{index});")
    lines.extend(("  return result;", "}"))
    return functions, "\n".join(lines)


def emit_cpp_source(
    problem: FoldProgram,
    sizes: dict[str, int],
    *,
    backward: bool,
    scheduled_backward: ScheduledBackwardProgram | None = None,
    scheduled_forward: CpuScheduledPlan | None = None,
    forward_extents: tuple[int, ...] = (),
) -> str:
    """Emit one extension containing the independently callable plan regions."""
    regions = {
        "map_fold": problem.map_fold,
        "project_output": problem.project_output,
        "execution_identity": problem.execution_identity,
        "execution_combine": problem.execution_combine,
    }
    if backward:
        if problem.embed_output_vjp is None:
            raise ValueError("planning problem has no backward regions")
        regions["embed_output_vjp"] = problem.embed_output_vjp
        backward_region = (
            problem.map_finalize_map_vjp or problem.map_finalize_map_vjp_accumulate
        )
        if backward_region is None:
            raise ValueError("planning problem has no map-finalize region")
        regions["map_finalize_map_vjp"] = backward_region
    nested_programs = []
    for region in regions.values():
        for node in region.nodes:
            combine = _attr(node, "combine_region")
            if isinstance(combine, TensorProgram) and all(
                combine != existing for existing in nested_programs
            ):
                nested_programs.append(combine)
    nested_names = {
        id(program): f"combine_region_{index}"
        for index, program in enumerate(nested_programs)
    }
    nested_functions = [
        _emit_region(
            nested_names[id(program)],
            program_to_region(program),
            sizes,
            nested_names=nested_names,
        )
        for program in nested_programs
    ]
    region_functions = [
        *(
            _emit_region(
                name,
                region,
                sizes,
                nested_names=nested_names,
            )
            for name, region in regions.items()
        )
    ]
    scheduled_functions: list[str] = []
    if scheduled_forward is not None:
        variants, kernel = _emit_scheduled_forward(
            problem,
            scheduled_forward,
            sizes,
            forward_extents,
            nested_names=nested_names,
        )
        scheduled_functions.extend((*variants, kernel))
    if scheduled_backward is not None:
        variants, kernel = _emit_scheduled_backward(
            problem,
            scheduled_backward,
            sizes,
            nested_names=nested_names,
        )
        scheduled_functions.extend((*variants, kernel))
    functions = "\n\n".join(
        [
            *nested_functions,
            *region_functions,
            *scheduled_functions,
        ]
    )
    bindings = "\n".join(
        f'  m.def("{name}", &{_cpp_name(name)}, '
        "py::call_guard<py::gil_scoped_release>());"
        for name in regions
    )
    if scheduled_backward is not None:
        bindings += (
            '\n  m.def("scheduled_backward", &scheduled_backward, '
            "py::call_guard<py::gil_scoped_release>());"
        )
    if scheduled_forward is not None:
        bindings += (
            '\n  m.def("scheduled_forward", &scheduled_forward, '
            "py::call_guard<py::gil_scoped_release>());"
        )
    atomic_helper = ""
    if scheduled_backward is not None:
        atomic_helper = """
void atomic_add_relaxed(torch::Tensor destination, torch::Tensor source) {
  TORCH_CHECK(destination.scalar_type() == at::kFloat, "atomic destination must be float32");
  TORCH_CHECK(source.scalar_type() == at::kFloat, "atomic source must be float32");
  auto contiguous = source.contiguous();
  TORCH_CHECK(destination.sizes() == contiguous.sizes(), "atomic tile shape mismatch");
  auto* destination_ptr = destination.data_ptr<float>();
  const auto* source_ptr = contiguous.data_ptr<float>();
  for (int64_t linear = 0; linear < contiguous.numel(); ++linear) {
    int64_t remainder = linear;
    int64_t offset = 0;
    for (int64_t dimension = contiguous.dim() - 1; dimension >= 0; --dimension) {
      const int64_t coordinate = remainder % contiguous.size(dimension);
      remainder /= contiguous.size(dimension);
      offset += coordinate * destination.stride(dimension);
    }
    std::atomic_ref<float>(destination_ptr[offset]).fetch_add(
        source_ptr[linear], std::memory_order_relaxed);
  }
}
"""
    return f"""#include <torch/extension.h>
#include <ATen/Parallel.h>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <functional>
#include <limits>
#include <vector>
{atomic_helper}

{functions}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {{
{bindings}
}}
"""


def _region_arguments(region: TensorRegion, values: dict[str, Any]) -> list[Any]:
    arguments = []
    for name, _ in region.arguments:
        candidates = (name, name.replace("saved_execution.", "execution."))
        value = next((values[key] for key in candidates if key in values), None)
        if value is None:
            raise ValueError(f"missing region argument {name!r}")
        arguments.append(value)
    return arguments


def _bind_results(region: TensorRegion, tensors: list[Any]) -> dict[str, Any]:
    if len(tensors) != len(region.results):
        raise ValueError("compiled region returned the wrong number of values")
    return {
        name: tensor for (name, _), tensor in zip(region.results, tensors, strict=True)
    }


@dataclass
class CompiledCpuFold:
    operator: FoldOperator
    problem: FoldProgram
    scheduled: CpuScheduledPlan
    modules: dict[int, Any]
    compile_seconds: float
    backward_program: ScheduledBackwardProgram | None = None
    specialized: SpecializedSchedule | None = None

    @property
    def supports_backward(self) -> bool:
        return self.problem.embed_output_vjp is not None

    @property
    def module(self):
        fold_axis = self.problem.spec.fold_axes[0]
        return self.modules[dict(self.scheduled.sizes)[fold_axis]]

    def _validate_inputs(self, inputs: tuple[Any, ...]) -> dict[str, Any]:
        import torch

        torch_dtypes = {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "bool": torch.bool,
            "int32": torch.int32,
            "int64": torch.int64,
        }
        specs = tuple(self.problem.spec.inputs.items())
        if len(inputs) != len(specs):
            raise ValueError(f"expected {len(specs)} inputs, got {len(inputs)}")
        sizes = dict(self.scheduled.sizes)
        values = {}
        for tensor, (name, spec) in zip(inputs, specs, strict=True):
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"input {name!r} is not a torch.Tensor")
            expected_shape = tuple(sizes[axis] for axis in spec.axes)
            if tensor.device.type != "cpu" or not tensor.is_contiguous():
                raise ValueError(f"input {name!r} must be a contiguous CPU tensor")
            if tensor.dtype != torch_dtypes[spec.dtype]:
                raise ValueError(
                    f"input {name!r} has dtype {tensor.dtype}, "
                    f"expected {torch_dtypes[spec.dtype]}"
                )
            if tuple(tensor.shape) != expected_shape:
                raise ValueError(
                    f"input {name!r} has shape {tuple(tensor.shape)}, "
                    f"expected {expected_shape}"
                )
            values[name] = tensor
        return values

    def forward_values(self, *inputs) -> tuple[tuple[Any, ...], dict[str, Any]]:
        values = self._validate_inputs(inputs)
        if hasattr(self.module, "scheduled_forward"):
            flat = list(self.module.scheduled_forward(*inputs))
            output_count = len(self.problem.project_output.results)
            carrier_count = len(self.problem.map_fold.results)
            outputs = tuple(flat[:output_count])
            carrier_values = flat[output_count : output_count + carrier_count]
            execution = _bind_results(self.problem.map_fold, carrier_values)
            saved = dict(execution)
            prefix_values = flat[output_count + carrier_count :]
            if len(prefix_values) % carrier_count:
                raise ValueError("scheduled forward returned malformed prefixes")
            for index in range(len(prefix_values) // carrier_count):
                offset = index * carrier_count
                for (name, _), value in zip(
                    self.problem.map_fold.results,
                    prefix_values[offset : offset + carrier_count],
                    strict=True,
                ):
                    suffix = name.removeprefix("execution.")
                    saved[f"prefix_execution.{index}.{suffix}"] = value
            return outputs, saved
        fold_axis = self.problem.spec.fold_axes[0]
        fold_extent = dict(self.scheduled.sizes)[fold_axis]
        forward = self.scheduled.forward
        root = forward[0].bindings[0].expression
        if isinstance(root, CpuLoop) and root.operation == "map":
            partials = self._execute_map(root, inputs, 0, fold_extent)
            if not partials:
                raise ValueError("partitioned fold produced no partial carriers")
            combine = forward[1].bindings[0].expression
            combine_expression = forward[1].bindings[0].expression
            topology = (
                combine.topology
                if isinstance(combine, CpuLoop) and combine.topology is not None
                else FoldTopology.ONLINE
            )
            execution = self._combine_many(partials, topology)
            prefixes = (
                self._exclusive_prefixes(partials)
                if isinstance(combine_expression, CpuLoop)
                and combine_expression.operation == "exclusive_scan"
                else []
            )
        else:
            execution = self._execute_carrier(root, inputs, 0, fold_extent)
            prefixes = []
        values.update(execution)
        outputs = _bind_results(
            self.problem.project_output,
            list(
                self.module.project_output(
                    *_region_arguments(self.problem.project_output, values)
                )
            ),
        )
        saved = dict(execution)
        for index, prefix in enumerate(prefixes):
            for name, value in prefix.items():
                suffix = name.removeprefix("execution.")
                saved[f"prefix_execution.{index}.{suffix}"] = value
        return tuple(
            outputs[name] for name, _ in self.problem.project_output.results
        ), saved

    def _exclusive_prefixes(
        self, partials: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        identity = _bind_results(
            self.problem.execution_identity,
            list(self.module.execution_identity()),
        )
        first = partials[0]
        state = {
            name: value.reshape(
                (1,) * (first[name].ndim - value.ndim) + tuple(value.shape)
            )
            .expand_as(first[name])
            .clone()
            for name, value in identity.items()
        }
        prefixes = []
        for partial in partials:
            prefixes.append(state)
            state = self._combine(state, partial)
        return prefixes

    def _slice_inputs(
        self, inputs: tuple[Any, ...], start: int, extent: int
    ) -> dict[str, Any]:
        fold_axis = self.problem.spec.fold_axes[0]
        values = {}
        for tensor, (name, spec) in zip(
            inputs, self.problem.spec.inputs.items(), strict=True
        ):
            if fold_axis in spec.axes:
                dimension = spec.axes.index(fold_axis)
                tensor = tensor.narrow(dimension, start, extent).contiguous()
            values[name] = tensor
        return values

    def _map_fold(
        self, inputs: tuple[Any, ...], start: int, extent: int
    ) -> dict[str, Any]:
        module = self.modules[extent]
        values = self._slice_inputs(inputs, start, extent)
        return _bind_results(
            self.problem.map_fold,
            list(module.map_fold(*_region_arguments(self.problem.map_fold, values))),
        )

    def _combine(self, left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
        values = {}
        for name, value in left.items():
            suffix = name.removeprefix("execution.")
            values[f"left.{suffix}"] = value
        for name, value in right.items():
            suffix = name.removeprefix("execution.")
            values[f"right.{suffix}"] = value
        return _bind_results(
            self.problem.execution_combine,
            list(
                self.module.execution_combine(
                    *_region_arguments(self.problem.execution_combine, values)
                )
            ),
        )

    def _combine_many(
        self, values: list[dict[str, Any]], topology: FoldTopology
    ) -> dict[str, Any]:
        if not values:
            raise ValueError("cannot combine an empty carrier collection")
        if topology is FoldTopology.ONLINE:
            state = values[0]
            for value in values[1:]:
                state = self._combine(state, value)
            return state
        level = values
        while len(level) > 1:
            next_level = []
            for index in range(0, len(level), 2):
                if index + 1 == len(level):
                    next_level.append(level[index])
                else:
                    next_level.append(self._combine(level[index], level[index + 1]))
            level = next_level
        return level[0]

    def _execute_carrier(
        self,
        expression: CpuExpr,
        inputs: tuple[Any, ...],
        start: int,
        extent: int,
    ) -> dict[str, Any]:
        if isinstance(expression, CpuLeaf):
            if not isinstance(expression.expression, MapFoldRegion):
                raise TypeError("carrier leaf must invoke map_fold")
            return self._map_fold(inputs, start, extent)
        if (
            not isinstance(expression, CpuLoop)
            or expression.operation != "carrier_fold"
        ):
            raise TypeError("unsupported executable carrier expression")
        ranges = self._ranges(start, extent, expression.iteration.grain)
        if expression.iteration.loop.value == "parallel" and len(ranges) > 1:
            with ThreadPoolExecutor(
                max_workers=min(self.scheduled.target.worker_count, len(ranges))
            ) as pool:
                values = list(
                    pool.map(
                        lambda item: self._execute_carrier(
                            expression.body, inputs, *item
                        ),
                        ranges,
                    )
                )
        else:
            values = [
                self._execute_carrier(expression.body, inputs, *item) for item in ranges
            ]
        return self._combine_many(values, expression.topology or FoldTopology.ONLINE)

    def _execute_map(
        self,
        expression: CpuLoop,
        inputs: tuple[Any, ...],
        start: int,
        extent: int,
    ) -> list[dict[str, Any]]:
        ranges = self._ranges(start, extent, expression.iteration.grain)
        if expression.iteration.loop.value != "parallel" or len(ranges) <= 1:
            return [
                self._execute_carrier(expression.body, inputs, *item) for item in ranges
            ]
        with ThreadPoolExecutor(
            max_workers=min(self.scheduled.target.worker_count, len(ranges))
        ) as pool:
            return list(
                pool.map(
                    lambda item: self._execute_carrier(expression.body, inputs, *item),
                    ranges,
                )
            )

    @staticmethod
    def _ranges(start: int, extent: int, grain: int) -> list[tuple[int, int]]:
        result = []
        offset = start
        end = start + extent
        while offset < end:
            local_extent = min(grain, end - offset)
            result.append((offset, local_extent))
            offset += local_extent
        return result

    def backward_values(
        self,
        inputs: tuple[Any, ...],
        execution: dict[str, Any],
        output_cotangents: tuple[Any, ...],
    ) -> tuple[Any | None, ...]:
        if self.problem.embed_output_vjp is None:
            raise ValueError("CPU fold was compiled without backward regions")
        backward_region = (
            self.problem.map_finalize_map_vjp
            or self.problem.map_finalize_map_vjp_accumulate
        )
        if backward_region is None:
            raise ValueError("CPU fold has no map-finalize backward region")
        values = self._validate_inputs(inputs)
        values.update(execution)
        if len(output_cotangents) != len(self.problem.project_output.results):
            raise ValueError("output cotangent count does not match outputs")
        for (name, _), cotangent in zip(
            self.problem.project_output.results,
            output_cotangents,
            strict=True,
        ):
            values[f"grad_output.{name}"] = cotangent.contiguous()
        accumulator = _bind_results(
            self.problem.embed_output_vjp,
            list(
                self.module.embed_output_vjp(
                    *_region_arguments(self.problem.embed_output_vjp, values)
                )
            ),
        )
        values.update(accumulator)
        if self.backward_program is not None:
            tensors = list(
                self.module.scheduled_backward(
                    *_region_arguments(backward_region, values)
                )
            )
            gradients = {
                gradient.result_name: tensor
                for gradient, tensor in zip(
                    self.backward_program.gradients, tensors, strict=True
                )
            }
        elif not self.problem.requires_prefix:
            gradients = self._independent_backward(inputs, values)
        else:
            gradients = self._prefix_backward(inputs, execution, values)
        result_nodes = dict(backward_region.results)
        input_gradients = []
        for name, spec in self.problem.spec.inputs.items():
            result_name = f"partial_grad_input.{name}"
            value = gradients.get(result_name) if spec.differentiable else None
            if value is not None:
                axes = backward_region.nodes[result_nodes[result_name]].axes
                if axes != spec.axes:
                    value = value.permute(tuple(axes.index(axis) for axis in spec.axes))
            input_gradients.append(value)
        return tuple(input_gradients)

    def _merge_input_gradients(self, results: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        fold_axis = self.problem.spec.fold_axes[0]
        region = self.problem.map_finalize_map_vjp
        if region is None:
            raise ValueError("independent backward region is unavailable")
        result_nodes = dict(region.results)
        gradients = {}
        for name, spec in self.problem.spec.inputs.items():
            key = f"partial_grad_input.{name}"
            parts = [result[key] for result in results if key in result]
            if not parts:
                continue
            if fold_axis in spec.axes:
                axes = region.nodes[result_nodes[key]].axes
                gradients[key] = torch.cat(parts, dim=axes.index(fold_axis))
            else:
                value = parts[0]
                for part in parts[1:]:
                    value = value + part
                gradients[key] = value
        return gradients

    def _independent_backward(
        self, inputs: tuple[Any, ...], common_values: dict[str, Any]
    ) -> dict[str, Any]:
        region = self.problem.map_finalize_map_vjp
        if region is None:
            raise ValueError("prefix-free backward region is unavailable")
        root = self.scheduled.backward[-1].bindings[0].expression
        fold_extent = dict(self.scheduled.sizes)[self.problem.spec.fold_axes[0]]
        chunks = self._backward_chunks(root, 0, fold_extent)

        def execute(item):
            start, extent = item
            local_values = dict(common_values)
            local_values.update(self._slice_inputs(inputs, start, extent))
            return _bind_results(
                region,
                list(
                    self.modules[extent].map_finalize_map_vjp(
                        *_region_arguments(region, local_values)
                    )
                ),
            )

        if len(chunks) > 1:
            with ThreadPoolExecutor(
                max_workers=min(self.scheduled.target.worker_count, len(chunks))
            ) as pool:
                results = list(pool.map(execute, chunks))
        else:
            results = [execute(chunks[0])]
        return self._merge_input_gradients(results)

    def _backward_chunks(
        self, expression: CpuExpr, start: int, extent: int
    ) -> list[tuple[int, int]]:
        if isinstance(expression, CpuLeaf):
            return [(start, extent)]
        if not isinstance(expression, CpuLoop):
            raise TypeError("unsupported backward schedule expression")
        result = []
        for local_start, local_extent in self._ranges(
            start, extent, expression.iteration.grain
        ):
            result.extend(
                self._backward_chunks(expression.body, local_start, local_extent)
            )
        return result

    def _prefix_backward(
        self,
        inputs: tuple[Any, ...],
        saved: dict[str, Any],
        common_values: dict[str, Any],
    ) -> dict[str, Any]:
        region = self.problem.map_finalize_map_vjp_accumulate
        if region is None:
            raise ValueError("prefix backward region is unavailable")
        forward_root = self.scheduled.forward[0].bindings[0].expression
        backward_root = self.scheduled.backward[-1].bindings[0].expression
        if not isinstance(forward_root, CpuLoop) or forward_root.operation != "map":
            raise ValueError("prefix backward requires partitioned forward")
        fold_axis = self.problem.spec.fold_axes[0]
        fold_extent = dict(self.scheduled.sizes)[fold_axis]
        partition_ranges = self._ranges(0, fold_extent, forward_root.iteration.grain)
        fold_gradients: dict[str, list[Any]] = {
            name: [] for name in self.problem.spec.inputs
        }
        reduced_gradients: dict[str, Any] = {}
        for partition_index, (partition_start, partition_extent) in enumerate(
            partition_ranges
        ):
            prefix = {
                name.removeprefix(f"prefix_execution.{partition_index}."): value
                for name, value in saved.items()
                if name.startswith(f"prefix_execution.{partition_index}.")
            }
            prefix = {f"prefix.{name}": value for name, value in prefix.items()}
            body = (
                backward_root.body
                if isinstance(backward_root, CpuLoop)
                else backward_root
            )
            grain = (
                body.iteration.grain if isinstance(body, CpuLoop) else partition_extent
            )
            for chunk_start, chunk_extent in self._ranges(
                partition_start, partition_extent, grain
            ):
                local_values = dict(common_values)
                local_values.update(
                    self._slice_inputs(inputs, chunk_start, chunk_extent)
                )
                local_values.update(prefix)
                result = _bind_results(
                    region,
                    list(
                        self.modules[chunk_extent].map_finalize_map_vjp(
                            *_region_arguments(region, local_values)
                        )
                    ),
                )
                prefix = {
                    name.replace("updated_prefix.", "prefix."): value
                    for name, value in result.items()
                    if name.startswith("updated_prefix.")
                }
                for name, spec in self.problem.spec.inputs.items():
                    key = f"partial_grad_input.{name}"
                    if key not in result:
                        continue
                    if fold_axis in spec.axes:
                        fold_gradients[name].append(result[key])
                    else:
                        previous = reduced_gradients.get(name)
                        reduced_gradients[name] = (
                            result[key] if previous is None else previous + result[key]
                        )
        gradients = {
            f"partial_grad_input.{name}": value
            for name, value in reduced_gradients.items()
        }
        import torch

        for name, parts in fold_gradients.items():
            if not parts:
                continue
            result_name = f"partial_grad_input.{name}"
            dimension = region.nodes[dict(region.results)[result_name]].axes.index(fold_axis)
            gradients[f"partial_grad_input.{name}"] = torch.cat(parts, dim=dimension)
        return gradients

    def __call__(self, *inputs):
        if self.supports_backward and any(
            getattr(value, "requires_grad", False) for value in inputs
        ):
            return _CompiledFoldFunction.apply(self, *inputs)
        outputs, _ = self.forward_values(*inputs)
        return outputs[0] if len(outputs) == 1 else outputs


class _CompiledFoldFunction:
    """Lazily defined torch.autograd.Function to keep torch import optional."""

    @classmethod
    def apply(cls, compiled: CompiledCpuFold, *inputs):
        import torch

        class Function(torch.autograd.Function):
            @staticmethod
            def forward(ctx, *flat_inputs):
                outputs, execution = compiled.forward_values(*flat_inputs)
                ctx.input_count = len(flat_inputs)
                ctx.execution_names = tuple(execution)
                ctx.save_for_backward(
                    *flat_inputs,
                    *(execution[name] for name in ctx.execution_names),
                )
                return outputs[0] if len(outputs) == 1 else outputs

            @staticmethod
            def backward(ctx, *output_cotangents):
                saved = ctx.saved_tensors
                flat_inputs = tuple(saved[: ctx.input_count])
                execution = dict(
                    zip(
                        ctx.execution_names,
                        saved[ctx.input_count :],
                        strict=True,
                    )
                )
                return compiled.backward_values(
                    flat_inputs, execution, output_cotangents
                )

        return Function.apply(*inputs)


def compile_cpu(
    operator: FoldOperator,
    sizes: dict[str, int],
    *,
    plan: ForwardScheduleTemplate | None = None,
    backward: bool = False,
    verbose: bool = False,
    schedule: CpuScheduleConfig | None = None,
    backward_schedule: CpuScheduleConfig | None = None,
    backward_program: ScheduledBackwardProgram | None = None,
) -> CompiledCpuFold:
    """Compile size-specialized symbolic regions into a cached C++ extension."""
    from torch.utils.cpp_extension import load_inline

    problem = operator.lower(backward=backward)
    schedule_space = build_schedule_space(problem)
    if backward_program is not None:
        if not backward:
            raise ValueError("a scheduled backward program requires backward=True")
        if problem.requires_prefix or backward_program.threads_prefix:
            raise NotImplementedError(
                "scheduled CPU backward initially supports prefix-free folds"
            )
        for axis in (*backward_program.program_axes, backward_program.loop_axis):
            if axis.axis not in sizes or sizes[axis.axis] != axis.extent:
                raise ValueError(
                    f"scheduled extent for {axis.axis!r} does not match sizes"
                )
    if plan is None and schedule is None:
        from .cpu_cost import select_cpu_plan

        candidates = schedule_space.forward
        if backward and problem.requires_prefix:
            candidates = tuple(
                candidate
                for candidate in candidates
                if candidate.name == "partial_scan_checkpoints"
            )
        choice = select_cpu_plan(problem, candidates, sizes)
        selected = choice.plan
        schedule = choice.config
    else:
        selected = plan or schedule_space.forward[0]
    if (
        backward
        and problem.requires_prefix
        and selected.name != "partial_scan_checkpoints"
    ):
        raise ValueError(
            "prefix-dependent backward requires partial_scan_checkpoints forward"
        )
    from .cpu import schedule_cpu

    forward_scheduled = schedule_cpu(
        problem,
        selected,
        sizes,
        config=schedule,
    )
    scheduled = forward_scheduled
    if backward:
        backward_plan = schedule_space.backward[0]
        training = schedule_space.compose(selected, backward_plan)
        if problem.requires_prefix:
            forward_partition_count = None
            root = forward_scheduled.forward[0].bindings[0].expression
            if isinstance(root, CpuLoop) and root.operation == "map":
                forward_partition_count = root.iteration.count
            configured_chunks = (
                backward_schedule.chunk_sizes if backward_schedule is not None else ()
            )
            configured_scratch = (
                backward_schedule.scratch_bytes_per_worker
                if backward_schedule is not None
                else None
            )
            configured_stages = (
                backward_schedule.stages if backward_schedule is not None else ()
            )
            backward_schedule = CpuScheduleConfig(
                chunk_sizes=configured_chunks,
                partition_count=forward_partition_count,
                scratch_bytes_per_worker=configured_scratch,
                stages=configured_stages,
            )
        backward_scheduled = schedule_cpu(
            problem,
            backward_plan,
            sizes,
            target=forward_scheduled.target,
            config=backward_schedule,
        )
        scheduled = CpuScheduledPlan(
            training,
            forward_scheduled.sizes,
            forward_scheduled.target,
            forward_scheduled.forward,
            backward_scheduled.backward,
        )
    if backward and backward_program is None and not problem.requires_prefix:
        if len(problem.spec.fold_axes) != 1:
            raise NotImplementedError(
                "automatic fused CPU backward currently requires one fold axis"
            )
        automatic_config = backward_schedule or CpuScheduleConfig()
        loop_axis = problem.spec.fold_axes[0]
        program_axes = problem.spec.batch_axes
        explicit_tiles = dict(automatic_config.chunk_sizes)
        partition_counts = {axis: 1 for axis in program_axes}
        while (
            math.prod(partition_counts.values()) < scheduled.target.worker_count
            and program_axes
        ):
            axis = max(
                program_axes,
                key=lambda candidate: sizes[candidate]
                / partition_counts[candidate],
            )
            if partition_counts[axis] >= sizes[axis]:
                break
            partition_counts[axis] = min(
                sizes[axis], partition_counts[axis] * 2
            )
        tiles = {
            axis: min(
                sizes[axis],
                explicit_tiles.get(
                    axis,
                    math.ceil(sizes[axis] / partition_counts[axis]),
                ),
            )
            for axis in program_axes
        }
        tiles[loop_axis] = automatic_config.chunk_size(
            loop_axis, sizes[loop_axis]
        )
        backward_program = schedule_backward_program(
            problem,
            sizes,
            loop_axis=loop_axis,
            program_axes=program_axes,
            tiles=tiles,
        )
    fold_axis = problem.spec.fold_axes[0]
    full_extent = sizes[fold_axis]

    def required_extents(
        expression: CpuExpr,
        extent: int,
    ) -> set[int]:
        if isinstance(expression, CpuLeaf):
            requires_specialized_module = isinstance(
                expression.expression, MapFoldRegion
            ) or (
                isinstance(expression.expression, RegionCall)
                and expression.expression.region.startswith("map_finalize")
            )
            return {extent} if requires_specialized_module else set()
        if not isinstance(expression, CpuLoop):
            return set()
        result: set[int] = set()
        offset = 0
        while offset < extent:
            local = min(expression.iteration.grain, extent - offset)
            result.update(required_extents(expression.body, local))
            offset += local
        return result

    extents = {full_extent}
    for stage in (*scheduled.forward, *scheduled.backward):
        for binding in stage.bindings:
            extents.update(required_extents(binding.expression, full_extent))
    modules = {}
    compile_start = time.perf_counter()
    for extent in sorted(extents):
        local_sizes = dict(sizes)
        local_sizes[fold_axis] = extent
        source = emit_cpp_source(
            problem,
            local_sizes,
            backward=backward,
            scheduled_backward=(backward_program if extent == full_extent else None),
            scheduled_forward=(scheduled if extent == full_extent else None),
            forward_extents=(tuple(sorted(extents)) if extent == full_extent else ()),
        )
        key = json.dumps(
            {
                "source": source,
                "sizes": sorted(local_sizes.items()),
                "target": scheduled.target.architecture,
            },
            sort_keys=True,
        ).encode()
        name = "cutilereduce_cpu_" + hashlib.sha256(key).hexdigest()[:16]
        extra_cflags = ["-O3", "-march=native"]
        if backward_program is not None:
            extra_cflags.append("-std=c++20")
        modules[extent] = load_inline(
            name=name,
            cpp_sources=source,
            functions=None,
            extra_cflags=extra_cflags,
            with_cuda=False,
            verbose=verbose,
        )
    specialization_template = scheduled.source
    specialized = SpecializedSchedule(
        program=problem,
        template=specialization_template,
        target=TargetModel(
            "cpu",
            frozenset({"tensor_regions", "tiled_loops", "multi_stage", "relaxed_atomic_add"}),
        ),
        shape=scheduled.sizes,
        parameters=tuple(schedule.chunk_sizes) if schedule is not None else (),
        implementation=scheduled,
    )
    return CompiledCpuFold(
        operator,
        problem,
        scheduled,
        modules,
        time.perf_counter() - compile_start,
        backward_program,
        specialized,
    )


__all__ = ["CompiledCpuFold", "compile_cpu", "emit_cpp_source"]
