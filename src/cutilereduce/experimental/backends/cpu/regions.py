"""Lower tensor regions to ATen C++ operations, independently of schedules."""

from __future__ import annotations

import math
from typing import Any

from cutilereduce.experimental.ir.core import (
    ContractionAxes,
    ContractionPrecision,
    TensorNode,
    TensorProgram,
    TensorRegion,
)
from cutilereduce.experimental.ir.passes import program_to_region


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
    precision = _attr(node, "precision")
    if precision is not None and not isinstance(precision, ContractionPrecision):
        raise TypeError("contract node has invalid precision metadata")
    if precision is not None and precision.operand_dtype is not None:
        # CPU has no equivalent tensor-core instruction. Preserve the same
        # numerical contract by quantizing operands before the FP32 GEMM.
        inputs = [
            f"(({value}).to({_CPP_DTYPE[precision.operand_dtype]}).to({_CPP_DTYPE[node.dtype]}))"
            for value in inputs[:2]
        ] + inputs[2:]
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
            case (
                "neg"
                | "exp"
                | "log"
                | "sqrt"
                | "rsqrt"
                | "abs"
                | "sin"
                | "cos"
                | "tanh"
                | "sigmoid"
                | "log1p"
                | "expm1"
                | "logical_not"
            ):
                operand = _align(inputs[0], input_nodes[0].axes, node.axes)
                expression = f"at::{node.op}({operand})"
            case (
                "minimum"
                | "maximum"
                | "logical_and"
                | "logical_or"
                | "floor_divide"
                | "remainder"
            ):
                operands = [
                    _align(value, source.axes, node.axes)
                    for value, source in zip(inputs, input_nodes, strict=True)
                ]
                expression = f"at::{node.op}({', '.join(operands)})"
            case "stop_gradient":
                expression = (
                    f"({_align(inputs[0], input_nodes[0].axes, node.axes)}).detach()"
                )
            case "clamp":
                expression = _align(inputs[0], input_nodes[0].axes, node.axes)
                for bound in ("min", "max"):
                    value = _attr(node, bound)
                    if value is not None:
                        expression = (
                            f"at::clamp_{bound}({expression}, {_number(value)})"
                        )
            case "powi":
                from cutilereduce.experimental.ir.ops import multiplication_chain

                exponent = _attr(node, "exponent")
                powers = [_align(inputs[0], input_nodes[0].axes, node.axes)]
                for left, right in (
                    multiplication_chain(abs(exponent)) if exponent else ()
                ):
                    name = f"power_{node.id}_{len(powers)}"
                    lines.append(
                        f"  auto {name} = ({powers[left]}) * ({powers[right]});"
                    )
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
