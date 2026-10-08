"""Reverse-mode differentiation for logical map expressions in the tensor IR."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from .ir import ContractionAxes, TensorBuilder, TensorValue
from .ops import ELEMENTWISE_OPS, FLOAT_DTYPES


def _sum_to(
    ctx: TensorBuilder, value: TensorValue, axes: tuple[str, ...]
) -> TensorValue:
    reduce_axes = tuple(axis for axis in value.axes if axis not in axes)
    if reduce_axes:
        value = ctx.reduce("sum", value, reduce_axes)
    missing = tuple(axis for axis in axes if axis not in value.axes)
    if missing:
        value = value * ctx.full(axes, 1.0, value.dtype)
    return value


def _accumulate(
    gradients: dict[int, TensorValue], node_id: int, value: TensorValue
) -> None:
    previous = gradients.get(node_id)
    gradients[node_id] = value if previous is None else previous + value


def _contract_vjp(
    ctx: TensorBuilder,
    cotangent: TensorValue,
    other: TensorValue,
    *,
    reduce: tuple[str, ...],
    target_axes: tuple[str, ...],
) -> TensorValue:
    if reduce:
        result = ctx.contract(cotangent, other, over=reduce)
    else:
        result = cotangent * other
    return _sum_to(ctx, result, target_axes)


def vjp(
    ctx: TensorBuilder,
    outputs: Iterable[tuple[TensorValue, TensorValue]],
    *,
    wrt: Mapping[str, TensorValue],
) -> dict[str, TensorValue]:
    """Build a VJP of the primal graph reachable from ``outputs``.

    Cotangents may be arbitrary later IR expressions. Only primal ancestors are
    traversed, so algebra finalization is composed with, rather than
    differentiated as part of, the logical map.
    """
    output_pairs = tuple(outputs)
    stop_ids = {value.id for value in wrt.values()}
    primal_ids: set[int] = set()
    pending = [primal.id for primal, _ in output_pairs]
    while pending:
        node_id = pending.pop()
        if node_id in primal_ids:
            continue
        primal_ids.add(node_id)
        if node_id in stop_ids:
            continue
        if ctx.nodes[node_id].op == "stop_gradient":
            continue
        pending.extend(ctx.nodes[node_id].inputs)

    gradients: dict[int, TensorValue] = {}
    for primal, cotangent in output_pairs:
        _accumulate(gradients, primal.id, _sum_to(ctx, cotangent, primal.axes))

    for node_id in sorted(primal_ids, reverse=True):
        cotangent = gradients.get(node_id)
        if cotangent is None:
            continue
        if node_id in stop_ids:
            continue
        node = ctx.nodes[node_id]
        if node.dtype not in FLOAT_DTYPES:
            continue
        inputs = tuple(TensorValue(ctx, value) for value in node.inputs)
        contract = ELEMENTWISE_OPS.get(node.op)
        if contract is not None and contract.differentiation == "none":
            continue

        def add_input(index: int, value: TensorValue) -> None:
            operand = inputs[index]
            if operand.dtype not in FLOAT_DTYPES:
                return
            value = ctx.cast(value, operand.dtype)
            _accumulate(gradients, operand.id, _sum_to(ctx, value, operand.axes))

        match node.op:
            case "add":
                add_input(0, cotangent)
                add_input(1, cotangent)
            case "subtract":
                add_input(0, cotangent)
                add_input(1, -cotangent)
            case "multiply":
                add_input(0, cotangent * inputs[1])
                add_input(1, cotangent * inputs[0])
            case "divide":
                add_input(0, cotangent / inputs[1])
                add_input(
                    1,
                    -(cotangent * inputs[0]) / (inputs[1] * inputs[1]),
                )
            case "neg":
                add_input(0, -cotangent)
            case "exp":
                add_input(0, cotangent * TensorValue(ctx, node_id))
            case "log":
                add_input(0, cotangent / inputs[0])
            case "powi":
                exponent = node.attribute("exponent")
                if exponent != 0:
                    add_input(0, cotangent * exponent * ctx.powi(inputs[0], exponent - 1))
            case "sqrt":
                add_input(0, cotangent / (2.0 * TensorValue(ctx, node_id)))
            case "rsqrt":
                add_input(0, -0.5 * cotangent * ctx.powi(TensorValue(ctx, node_id), 3))
            case "sin":
                add_input(0, cotangent * ctx.cos(inputs[0]))
            case "cos":
                add_input(0, -cotangent * ctx.sin(inputs[0]))
            case "tanh":
                add_input(0, cotangent * (1.0 - ctx.powi(TensorValue(ctx, node_id), 2)))
            case "sigmoid":
                output = TensorValue(ctx, node_id)
                add_input(0, cotangent * output * (1.0 - output))
            case "log1p":
                add_input(0, cotangent / (1.0 + inputs[0]))
            case "expm1":
                add_input(0, cotangent * ctx.exp(inputs[0]))
            case "abs":
                zero = ctx.full(cotangent.axes, 0.0, cotangent.dtype)
                add_input(0, ctx.where(ctx.compare("gt", inputs[0], 0), cotangent,
                    ctx.where(ctx.compare("lt", inputs[0], 0), -cotangent, zero)))
            case "minimum" | "maximum":
                zero = ctx.full(cotangent.axes, 0.0, cotangent.dtype)
                tie = ctx.compare("eq", inputs[0], inputs[1])
                for index in (0, 1):
                    wins = ctx.compare("lt" if node.op == "minimum" else "gt", inputs[index], inputs[1-index])
                    add_input(index, ctx.where(tie, cotangent * 0.5, ctx.where(wins, cotangent, zero)))
            case "clamp":
                gradient = cotangent
                zero = ctx.full(cotangent.axes, 0.0, cotangent.dtype)
                for bound, comparison in (("min", "ge"), ("max", "le")):
                    value = node.attribute(bound)
                    if value is not None:
                        gradient = ctx.where(ctx.compare(comparison, inputs[0], value), gradient, zero)
                add_input(0, gradient)
            case "cast":
                if node.dtype in FLOAT_DTYPES and inputs[0].dtype in FLOAT_DTYPES:
                    add_input(0, ctx.cast(cotangent, inputs[0].dtype))
            case "where" | "mask_identity":
                zero = ctx.full(cotangent.axes, 0.0, cotangent.dtype)
                add_input(1, ctx.where(inputs[0], cotangent, zero))
                if node.op == "where":
                    add_input(2, ctx.where(inputs[0], zero, cotangent))
            case "contract" | "contract_accumulate":
                partition = node.attribute("contraction_axes")
                if not isinstance(partition, ContractionAxes):
                    raise TypeError("contract node has invalid axis metadata")
                add_input(
                    0,
                    _contract_vjp(
                        ctx,
                        cotangent,
                        inputs[1],
                        reduce=partition.rhs_free,
                        target_axes=inputs[0].axes,
                    ),
                )
                if node.op == "contract_accumulate":
                    add_input(2, cotangent)
                add_input(
                    1,
                    _contract_vjp(
                        ctx,
                        cotangent,
                        inputs[0],
                        reduce=partition.lhs_free,
                        target_axes=inputs[1].axes,
                    ),
                )
            case "reduce_sum":
                add_input(
                    0,
                    cotangent * ctx.full(inputs[0].axes, 1.0, cotangent.dtype),
                )
            case "input" | "axis_index" | "constant" | "full" | "zero":
                pass
            case _:
                raise NotImplementedError(f"no VJP rule for tensor op {node.op!r}")

    result = {}
    for name, value in wrt.items():
        if value.dtype not in FLOAT_DTYPES:
            continue
        gradient = gradients.get(value.id)
        if gradient is not None:
            result[name] = gradient
    return result


__all__ = ["vjp"]
