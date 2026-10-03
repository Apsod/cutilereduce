"""Reverse-mode differentiation for logical map expressions in the tensor IR."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from .ir import ContractionAxes, TensorBuilder, TensorValue


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
        inputs = tuple(TensorValue(ctx, value) for value in node.inputs)

        def add_input(index: int, value: TensorValue) -> None:
            operand = inputs[index]
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
            case "cast":
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
        gradient = gradients.get(value.id)
        if gradient is not None:
            result[name] = gradient
    return result


__all__ = ["vjp"]
