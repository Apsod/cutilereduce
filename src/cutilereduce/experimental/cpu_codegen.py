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
)
from .scheduled_ssa import SpecializedSchedule, liveness, scheduled_to_ssa
from .target import TargetModel


_CPP_DTYPE = {
    "float32": "at::kFloat",
    "bfloat16": "at::kBFloat16",
    "float16": "at::kHalf",
    "bool": "at::kBool",
    "int32": "at::kInt",
    "int64": "at::kLong",
}


def _attr(node: TensorNode, name: str, default: Any = None) -> Any:
    return dict(node.attributes).get(name, default)


def _number(value: int | float) -> str:
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


def _einsum_equation(
    lhs: tuple[str, ...], rhs: tuple[str, ...], out: tuple[str, ...]
) -> str:
    axes = tuple(dict.fromkeys((*lhs, *rhs, *out)))
    alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if len(axes) > len(alphabet):
        raise ValueError("too many axes for initial C++ einsum lowering")
    letters = dict(zip(axes, alphabet[: len(axes)], strict=True))

    def encode(values):
        return "".join(letters[axis] for axis in values)

    return f"{encode(lhs)},{encode(rhs)}->{encode(out)}"


def _cpp_scalar(dtype: str) -> str:
    return {
        "float32": "float",
        "bfloat16": "at::BFloat16",
        "float16": "at::Half",
    }[dtype]


def _strides(axes: tuple[str, ...], sizes: dict[str, int]) -> dict[str, int]:
    stride = 1
    result = {}
    for axis in reversed(axes):
        result[axis] = stride
        stride *= sizes[axis]
    return result


def _offset(
    axes: tuple[str, ...], sizes: dict[str, int], coordinates: dict[str, str]
) -> str:
    strides = _strides(axes, sizes)
    terms = [
        f"({coordinates[axis]}) * {strides[axis]}"
        for axis in axes
        if strides[axis] != 0
    ]
    return " + ".join(terms) or "0"


def _argument_ancestor(region: TensorRegion, node_id: int) -> TensorNode | None:
    node = region.nodes[node_id]
    while node.op == "cast":
        node = region.nodes[node.inputs[0]]
    return node if node.op == "argument" else None


def _emit_fused_lws_map_fold(
    name: str,
    region: TensorRegion,
    sizes: dict[str, int],
    *,
    parallel: bool,
) -> str | None:
    """Recognize and emit the stable attention-shaped LWS fold directly."""
    results = dict(region.results)
    required = {"execution.maximum", "execution.normalizer", "execution.numerator"}
    if set(results) != required:
        return None
    maximum = region.nodes[results["execution.maximum"]]
    normalizer = region.nodes[results["execution.normalizer"]]
    numerator = region.nodes[results["execution.numerator"]]
    if maximum.op != "reduce_max" or normalizer.op != "reduce_sum":
        return None
    score = region.nodes[maximum.inputs[0]]
    if score.op != "contract" or numerator.op != "contract":
        return None
    score_partition = _attr(score, "contraction_axes")
    numerator_partition = _attr(numerator, "contraction_axes")
    if not isinstance(score_partition, ContractionAxes) or not isinstance(
        numerator_partition, ContractionAxes
    ):
        return None
    if len(score_partition.reduce) != 1 or len(numerator_partition.reduce) != 1:
        return None
    fold_axis = numerator_partition.reduce[0]
    reduction_axis = score_partition.reduce[0]
    if fold_axis not in score.axes:
        return None
    lhs = _argument_ancestor(region, score.inputs[0])
    rhs = _argument_ancestor(region, score.inputs[1])
    payload = _argument_ancestor(region, numerator.inputs[1])
    if lhs is None or rhs is None or payload is None:
        return None
    argument_by_id = {node_id: name for name, node_id in region.arguments}
    if any(node.id not in argument_by_id for node in (lhs, rhs, payload)):
        return None
    outer_axes = maximum.axes
    inner_axes = tuple(axis for axis in numerator.axes if axis not in outer_axes)
    outer_size = math.prod(sizes[axis] for axis in outer_axes)
    inner_size = math.prod(sizes[axis] for axis in inner_axes)
    parameter_names = {node_id: _cpp_name(arg) for arg, node_id in region.arguments}
    parameters = ", ".join(
        f"const torch::Tensor& {parameter_names[node_id]}"
        for _, node_id in region.arguments
    )
    lines = [f"std::vector<torch::Tensor> {_cpp_name(name)}({parameters}) {{"]
    for node in (lhs, rhs, payload):
        variable = parameter_names[node.id]
        lines.append(
            f"  const auto* {variable}_ptr = "
            f"{variable}.data_ptr<{_cpp_scalar(node.dtype)}>();"
        )
    output_shape = _axes_literal(outer_axes, sizes)
    numerator_shape = _axes_literal(numerator.axes, sizes)
    loop_open = (
        f"  at::parallel_for(0, {outer_size}, 1, [&](int64_t begin, int64_t end) {{"
        if parallel
        else f"  {{ int64_t begin = 0, end = {outer_size};"
    )
    loop_close = "  });" if parallel else "  }"
    lines.extend(
        [
            f"  auto maximum = at::empty({output_shape}, at::TensorOptions().dtype(at::kFloat));",
            f"  auto normalizer = at::empty({output_shape}, at::TensorOptions().dtype(at::kFloat));",
            f"  auto numerator = at::zeros({numerator_shape}, at::TensorOptions().dtype(at::kFloat));",
            "  auto* maximum_ptr = maximum.data_ptr<float>();",
            "  auto* normalizer_ptr = normalizer.data_ptr<float>();",
            "  auto* numerator_ptr = numerator.data_ptr<float>();",
            loop_open,
            "    for (int64_t outer = begin; outer < end; ++outer) {",
            "      int64_t outer_remainder = outer;",
        ]
    )
    coordinates: dict[str, str] = {}
    for axis in reversed(outer_axes):
        variable = f"i_{_cpp_name(axis)}"
        coordinates[axis] = variable
        lines.append(
            f"      const int64_t {variable} = outer_remainder % {sizes[axis]};"
        )
        lines.append(f"      outer_remainder /= {sizes[axis]};")
    score_coordinates = dict(coordinates)
    score_coordinates[fold_axis] = "fold"
    score_coordinates[reduction_axis] = "reduce"
    lhs_offset = _offset(lhs.axes, sizes, score_coordinates)
    rhs_offset = _offset(rhs.axes, sizes, score_coordinates)
    score_code = [
        "        float score = 0.0f;",
        f"        for (int64_t reduce = 0; reduce < {sizes[reduction_axis]}; ++reduce) {{",
        f"          score += static_cast<float>({parameter_names[lhs.id]}_ptr[{lhs_offset}]) * static_cast<float>({parameter_names[rhs.id]}_ptr[{rhs_offset}]);",
        "        }",
    ]
    lines.extend(
        [
            "      float local_maximum = -std::numeric_limits<float>::infinity();",
            f"      for (int64_t fold = 0; fold < {sizes[fold_axis]}; ++fold) {{",
            *score_code,
            "        local_maximum = std::max(local_maximum, score);",
            "      }",
            "      float local_normalizer = 0.0f;",
            f"      for (int64_t fold = 0; fold < {sizes[fold_axis]}; ++fold) {{",
            *score_code,
            "        const float weight = std::exp(score - local_maximum);",
            "        local_normalizer += weight;",
            f"        for (int64_t inner = 0; inner < {inner_size}; ++inner) {{",
            "          int64_t inner_remainder = inner;",
        ]
    )
    payload_coordinates = dict(coordinates)
    payload_coordinates[fold_axis] = "fold"
    for axis in reversed(inner_axes):
        variable = f"i_{_cpp_name(axis)}"
        payload_coordinates[axis] = variable
        lines.append(
            f"          const int64_t {variable} = inner_remainder % {sizes[axis]};"
        )
        lines.append(f"          inner_remainder /= {sizes[axis]};")
    payload_offset = _offset(payload.axes, sizes, payload_coordinates)
    lines.extend(
        [
            f"          numerator_ptr[outer * {inner_size} + inner] += weight * static_cast<float>({parameter_names[payload.id]}_ptr[{payload_offset}]);",
            "        }",
            "      }",
            "      maximum_ptr[outer] = local_maximum;",
            "      normalizer_ptr[outer] = local_normalizer;",
            "    }",
            loop_close,
            "  return {maximum, normalizer, numerator};",
            "}",
        ]
    )
    return "\n".join(lines)


def _emit_region(
    name: str,
    region: TensorRegion,
    sizes: dict[str, int],
    *,
    parallel_map_fold: bool,
    nested_names: dict[int, str] | None = None,
) -> str:
    if name == "map_fold":
        fused = _emit_fused_lws_map_fold(
            name, region, sizes, parallel=parallel_map_fold
        )
        if fused is not None:
            return fused
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
            case "neg":
                expression = f"-({inputs[0]})"
            case "exp" | "log":
                expression = f"at::{node.op}({inputs[0]})"
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
                equation = _einsum_equation(
                    input_nodes[0].axes, input_nodes[1].axes, node.axes
                )
                expression = f'at::einsum("{equation}", {{{inputs[0]}, {inputs[1]}}})'
                if node.op == "contract_accumulate":
                    expression = f"({expression}) + ({inputs[2]})"
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
            parallel_map_fold=False,
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


def emit_cpp_source(
    problem: FoldProgram,
    sizes: dict[str, int],
    *,
    backward: bool,
    parallel_map_fold: bool = True,
    scheduled_backward: ScheduledBackwardProgram | None = None,
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
            parallel_map_fold=False,
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
                parallel_map_fold=parallel_map_fold,
                nested_names=nested_names,
            )
            for name, region in regions.items()
        )
    ]
    scheduled_functions: list[str] = []
    if scheduled_backward is not None:
        variants, kernel = _emit_scheduled_backward(
            problem,
            scheduled_backward,
            sizes,
            nested_names=nested_names,
        )
        scheduled_functions = [*variants, kernel]
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
        return tuple(
            gradients.get(f"partial_grad_input.{name}") if spec.differentiable else None
            for name, spec in self.problem.spec.inputs.items()
        )

    def _merge_input_gradients(self, results: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        fold_axis = self.problem.spec.fold_axes[0]
        gradients = {}
        for name, spec in self.problem.spec.inputs.items():
            key = f"partial_grad_input.{name}"
            parts = [result[key] for result in results if key in result]
            if not parts:
                continue
            if fold_axis in spec.axes:
                gradients[key] = torch.cat(parts, dim=spec.axes.index(fold_axis))
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
            dimension = self.problem.spec.inputs[name].axes.index(fold_axis)
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
            backward_schedule = CpuScheduleConfig(
                chunk_sizes=configured_chunks,
                partition_count=forward_partition_count,
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
    fold_axis = problem.spec.fold_axes[0]
    full_extent = sizes[fold_axis]

    def required_extents(
        expression: CpuExpr,
        extent: int,
        *,
        parallel_active: bool = False,
    ) -> tuple[set[int], set[int]]:
        if isinstance(expression, CpuLeaf):
            requires_specialized_module = isinstance(
                expression.expression, MapFoldRegion
            ) or (
                isinstance(expression.expression, RegionCall)
                and expression.expression.region.startswith("map_finalize")
            )
            extents = {extent} if requires_specialized_module else set()
            return extents, extents if parallel_active else set()
        if not isinstance(expression, CpuLoop):
            return set(), set()
        result: set[int] = set()
        parallel_result: set[int] = set()
        nested_parallel = (
            parallel_active or expression.iteration.loop.value == "parallel"
        )
        offset = 0
        while offset < extent:
            local = min(expression.iteration.grain, extent - offset)
            child, child_parallel = required_extents(
                expression.body,
                local,
                parallel_active=nested_parallel,
            )
            result.update(child)
            parallel_result.update(child_parallel)
            offset += local
        return result, parallel_result

    extents = {full_extent}
    parallel_extents: set[int] = set()
    for stage in (*scheduled.forward, *scheduled.backward):
        for binding in stage.bindings:
            required, parallel_required = required_extents(
                binding.expression, full_extent
            )
            extents.update(required)
            parallel_extents.update(parallel_required)
    modules = {}
    compile_start = time.perf_counter()
    for extent in sorted(extents):
        local_sizes = dict(sizes)
        local_sizes[fold_axis] = extent
        source = emit_cpp_source(
            problem,
            local_sizes,
            backward=backward,
            parallel_map_fold=extent not in parallel_extents,
            scheduled_backward=(backward_program if extent == full_extent else None),
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
