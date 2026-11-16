"""Lower tensor regions to CuTile function bodies, independently of schedules."""

from __future__ import annotations

import math
from typing import Any

from cutilereduce.experimental.ir.core import (
    ContractionAxes,
    ContractionPrecision,
    TensorNode,
    TensorRegion,
)


_CT_DTYPE = {
    "bool": "ct.bool_",
    "int32": "ct.int32",
    "int64": "ct.int64",
    "float16": "ct.float16",
    "bfloat16": "ct.bfloat16",
    "float32": "ct.float32",
    "float64": "ct.float64",
}


def _product(values: tuple[int, ...]) -> int:
    return math.prod(values) if values else 1


def _tuple(values: tuple[Any, ...]) -> str:
    body = ", ".join(map(str, values))
    if len(values) == 1:
        body += ","
    return f"({body})"


def _attribute(node: TensorNode, name: str, default: Any = None) -> Any:
    return dict(node.attributes).get(name, default)


def _literal(value: Any) -> str:
    if isinstance(value, float) and math.isinf(value):
        return "float('-inf')" if value < 0 else "float('inf')"
    return repr(value)


def _identifier(name: str) -> str:
    return name.replace(".", "_")


def _shape(
    axes: tuple[str, ...], sizes: dict[str, int], tiles: dict[str, int]
) -> tuple[int, ...]:
    return tuple(tiles.get(axis, sizes[axis]) for axis in axes)


def _aligned(
    expression: str,
    source: tuple[str, ...],
    target: tuple[str, ...],
    sizes: dict[str, int],
    tiles: dict[str, int],
) -> str:
    present = tuple(axis for axis in target if axis in source)
    result = expression
    if present != source:
        permutation = tuple(source.index(axis) for axis in present)
        result = f"ct.permute({result}, {_tuple(permutation)})"
    aligned_shape = tuple(
        tiles.get(axis, sizes[axis]) if axis in source else 1 for axis in target
    )
    if aligned_shape != _shape(source, sizes, tiles):
        result = f"ct.reshape({result}, {_tuple(aligned_shape)})"
    target_shape = _shape(target, sizes, tiles)
    if aligned_shape != target_shape:
        result = f"ct.broadcast_to({result}, {_tuple(target_shape)})"
    return result


def _contract(
    node: TensorNode,
    input_nodes: list[TensorNode],
    inputs: list[str],
    sizes: dict[str, int],
    tiles: dict[str, int],
) -> str:
    partition = _attribute(node, "contraction_axes")
    if not isinstance(partition, ContractionAxes):
        raise TypeError("contract node is missing contraction-axis metadata")
    precision = _attribute(node, "precision")
    if precision is not None and not isinstance(precision, ContractionPrecision):
        raise TypeError("contract node has invalid precision metadata")
    batch = partition.batch
    lhs_order = (*batch, *partition.lhs_free, *partition.reduce)
    rhs_order = (*batch, *partition.rhs_free, *partition.reduce)

    def reorder(value: str, source: tuple[str, ...], target: tuple[str, ...]) -> str:
        if source == target:
            return value
        permutation = tuple(source.index(axis) for axis in target)
        return f"ct.permute({value}, {_tuple(permutation)})"

    lhs = reorder(inputs[0], input_nodes[0].axes, lhs_order)
    rhs = reorder(inputs[1], input_nodes[1].axes, rhs_order)
    if precision is not None and precision.operand_dtype is not None:
        operand = _CT_DTYPE[precision.operand_dtype]
        lhs = f"({lhs}).astype({operand})"
        rhs = f"({rhs}).astype({operand})"
    batch_size = _product(_shape(batch, sizes, tiles))
    batch_shape = (batch_size,) if batch else ()
    m = _product(_shape(partition.lhs_free, sizes, tiles))
    n = _product(_shape(partition.rhs_free, sizes, tiles))
    k = _product(_shape(partition.reduce, sizes, tiles))
    lhs = f"ct.reshape({lhs}, {_tuple((*batch_shape, m, k))})"
    rhs = f"ct.reshape({rhs}, {_tuple((*batch_shape, n, k))})"
    rhs = f"ct.transpose({rhs})" if not batch else f"ct.transpose({rhs}, 1, 2)"
    accumulator = (
        f"ct.reshape({inputs[2]}, {_tuple((*batch_shape, m, n))})"
        if node.op == "contract_accumulate"
        else f"ct.zeros({_tuple((*batch_shape, m, n))}, {_CT_DTYPE[node.dtype]})"
    )
    result = f"ct.mma({lhs}, {rhs}, {accumulator})"
    result_shape = _shape(node.axes, sizes, tiles)
    if result_shape != (*batch_shape, m, n):
        result = f"ct.reshape({result}, {_tuple(result_shape)})"
    return result


def _emit_region(
    name: str,
    region: TensorRegion,
    sizes: dict[str, int],
    tiles: dict[str, int],
) -> str:
    argument_names = {
        node_id: argument.replace(".", "_") for argument, node_id in region.arguments
    }
    index_nodes = tuple(node for node in region.nodes if node.op == "axis_index")
    domain_nodes = tuple(node for node in region.nodes if node.op == "domain_valid")
    parameters = ", ".join(
        (
            *argument_names.values(),
            *(f"axis_index_{node.attribute('axis')}" for node in index_nodes),
            *(("domain_valid",) if domain_nodes else ()),
        )
    )
    lines = ["@ct.function", f"def {name}({parameters}):"]
    values: dict[int, str] = dict(argument_names)
    values.update(
        (node.id, f"axis_index_{node.attribute('axis')}") for node in index_nodes
    )
    values.update((node.id, "domain_valid") for node in domain_nodes)
    for node in region.nodes:
        if node.id in values:
            continue
        variable = f"v{node.id}"
        inputs = [values[value] for value in node.inputs]
        input_nodes = [region.nodes[value] for value in node.inputs]
        match node.op:
            case "constant":
                expression = _literal(_attribute(node, "value"))
            case "zero" | "full":
                fill = 0 if node.op == "zero" else _attribute(node, "value")
                expression = (
                    f"ct.full({_tuple(_shape(node.axes, sizes, tiles))}, "
                    f"{_literal(fill)}, {_CT_DTYPE[node.dtype]})"
                )
            case "add" | "subtract" | "multiply" | "divide":
                symbol = {
                    "add": "+",
                    "subtract": "-",
                    "multiply": "*",
                    "divide": "/",
                }[node.op]
                left = _aligned(inputs[0], input_nodes[0].axes, node.axes, sizes, tiles)
                right = _aligned(
                    inputs[1], input_nodes[1].axes, node.axes, sizes, tiles
                )
                expression = f"({left}) {symbol} ({right})"
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
                | "stop_gradient"
                | "logical_not"
                | "sigmoid"
            ):
                operand = _aligned(
                    inputs[0], input_nodes[0].axes, node.axes, sizes, tiles
                )
                if node.op == "stop_gradient":
                    expression = operand
                elif node.op == "neg":
                    expression = f"-({operand})"
                elif node.op == "logical_not":
                    expression = f"~({operand})"
                elif node.op == "sigmoid":
                    work = (
                        f"({operand}).astype(ct.float32)"
                        if node.dtype in {"float16", "bfloat16"}
                        else operand
                    )
                    name = f"sigmoid_exp_{node.id}"
                    lines.append(f"    {name} = ct.exp(-ct.abs({work}))")
                    expression = f"ct.where(({work}) >= 0, 1 / (1 + {name}), {name} / (1 + {name}))"
                    expression = f"({expression}).astype({_CT_DTYPE[node.dtype]})"
                else:
                    expression = f"ct.{node.op}({operand})"
            case "log1p" | "expm1":
                raise NotImplementedError(
                    f"CuTile has no stable lowering for {node.op}"
                )
            case (
                "minimum"
                | "maximum"
                | "logical_and"
                | "logical_or"
                | "floor_divide"
                | "remainder"
            ):
                left, right = [
                    _aligned(value, source.axes, node.axes, sizes, tiles)
                    for value, source in zip(inputs, input_nodes, strict=True)
                ]
                if node.op in {"minimum", "maximum"}:
                    expression = f"ct.{node.op}({left}, {right}, propagate_nan=True)"
                elif node.op in {"logical_and", "logical_or"}:
                    expression = (
                        f"({left}) {'&' if node.op == 'logical_and' else '|'} ({right})"
                    )
                else:
                    # CuTile integer // rounds toward negative infinity.
                    quotient = f"integer_q_{node.id}"
                    lines.append(f"    {quotient} = ({left}) // ({right})")
                    expression = (
                        quotient
                        if node.op == "floor_divide"
                        else f"(({left}) - {quotient} * ({right}))"
                    )
            case "clamp":
                expression = _aligned(
                    inputs[0], input_nodes[0].axes, node.axes, sizes, tiles
                )
                for bound, op in (("min", "maximum"), ("max", "minimum")):
                    value = _attribute(node, bound)
                    if value is not None:
                        expression = f"ct.{op}({expression}, {_literal(value)}, propagate_nan=True)"
            case "powi":
                from cutilereduce.experimental.ir.ops import multiplication_chain

                exponent = _attribute(node, "exponent")
                powers = [
                    _aligned(inputs[0], input_nodes[0].axes, node.axes, sizes, tiles)
                ]
                for left, right in (
                    multiplication_chain(abs(exponent)) if exponent else ()
                ):
                    name = f"power_{node.id}_{len(powers)}"
                    lines.append(f"    {name} = ({powers[left]}) * ({powers[right]})")
                    powers.append(name)
                expression = (
                    powers[-1]
                    if exponent
                    else f"ct.full({_tuple(_shape(node.axes, sizes, tiles))}, 1, {_CT_DTYPE[node.dtype]})"
                )
                if exponent < 0:
                    expression = f"1 / ({expression})"
            case "cast":
                expression = f"({inputs[0]}).astype({_CT_DTYPE[node.dtype]})"
            case "where" | "mask_identity":
                aligned = [
                    _aligned(value, source.axes, node.axes, sizes, tiles)
                    for value, source in zip(inputs, input_nodes, strict=True)
                ]
                expression = f"ct.where({', '.join(aligned)})"
            case op if op.startswith("compare_"):
                symbol = {
                    "compare_eq": "==",
                    "compare_equal": "==",
                    "compare_ne": "!=",
                    "compare_not_equal": "!=",
                    "compare_lt": "<",
                    "compare_less": "<",
                    "compare_le": "<=",
                    "compare_less_equal": "<=",
                    "compare_gt": ">",
                    "compare_greater": ">",
                    "compare_ge": ">=",
                    "compare_greater_equal": ">=",
                }[op]
                target = node.axes
                left = _aligned(inputs[0], input_nodes[0].axes, target, sizes, tiles)
                right = _aligned(inputs[1], input_nodes[1].axes, target, sizes, tiles)
                expression = f"({left}) {symbol} ({right})"
            case op if op.startswith("reduce_"):
                over = tuple(_attribute(node, "over"))
                axes = tuple(input_nodes[0].axes.index(axis) for axis in over)
                function = "sum" if op == "reduce_sum" else "max"
                expression = f"ct.{function}({inputs[0]}, axis={axes[0] if len(axes) == 1 else _tuple(axes)})"
            case op if op.startswith("scan_"):
                over = _attribute(node, "over")
                axis_name = over[0] if isinstance(over, tuple) else over
                axis = input_nodes[0].axes.index(axis_name)
                expression = (
                    f"ct.cumsum({inputs[0]}, axis={axis}, "
                    f"reverse={bool(_attribute(node, 'reverse', False))})"
                )
                if _attribute(node, "exclusive", False):
                    expression = f"({expression}) - ({inputs[0]})"
            case "contract" | "contract_accumulate":
                expression = _contract(node, input_nodes, inputs, sizes, tiles)
            case _:
                raise NotImplementedError(
                    f"cuTile emission does not support tensor op {node.op!r}"
                )
        lines.append(f"    {variable} = {expression}")
        values[node.id] = variable
    results = tuple(values[node_id] for _, node_id in region.results)
    lines.append(f"    return {_tuple(results)}")
    return "\n".join(lines)
