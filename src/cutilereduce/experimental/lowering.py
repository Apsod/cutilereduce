"""Selection of optimized or generic structured monoid lowerings."""

from __future__ import annotations

from typing import cast

from .algebra.base import (
    AlgebraValue,
    CompositeValue,
    Monoid,
    ScanResult,
    StructuredValue,
    value_leaves,
)
from .ir import TensorBuilder, TensorProgram, TensorValue


def _replace_leaves(
    template: AlgebraValue,
    leaves: dict[tuple[str, ...], TensorValue],
    prefix: tuple[str, ...] = (),
) -> AlgebraValue:
    if isinstance(template, StructuredValue):
        return StructuredValue(
            {
                name: leaves[(*prefix, name)]
                for name in template.fields
            }
        )
    return CompositeValue(
        {
            name: _replace_leaves(component, leaves, (*prefix, name))
            for name, component in template.components.items()
        }
    )


def _combine_region[ValueT: AlgebraValue](
    monoid: Monoid[ValueT],
    template: ValueT,
    *,
    removed_axes: tuple[str, ...],
) -> TensorProgram:
    remaining_axes = tuple(
        axis
        for axis in next(iter(value_leaves(template)))[1].builder.axes
        if axis not in removed_axes
    )
    ctx = TensorBuilder(remaining_axes)
    left_leaves: dict[tuple[str, ...], TensorValue] = {}
    right_leaves: dict[tuple[str, ...], TensorValue] = {}
    for path, value in value_leaves(template):
        axes = tuple(axis for axis in value.axes if axis not in removed_axes)
        name = ".".join(path)
        left_leaves[path] = ctx.input(f"left.{name}", axes, value.dtype)
        right_leaves[path] = ctx.input(f"right.{name}", axes, value.dtype)
    left = cast(ValueT, _replace_leaves(template, left_leaves))
    right = cast(ValueT, _replace_leaves(template, right_leaves))
    combined = monoid.combine(ctx, left, right)
    monoid.validate(combined, label="generic combine result")
    for path, value in value_leaves(combined):
        ctx.materialize(".".join(path), value)
    return ctx.finish()


def _generic_inputs(
    values: AlgebraValue, initial: AlgebraValue
) -> tuple[TensorValue, ...]:
    return tuple(value for _, value in value_leaves(values)) + tuple(
        value for _, value in value_leaves(initial)
    )


def lower_fold[ValueT: AlgebraValue](
    ctx: TensorBuilder,
    monoid: Monoid[ValueT],
    values: ValueT,
    *,
    fold_axes: tuple[str, ...],
) -> ValueT:
    specialized = monoid.lower_fold(ctx, values, fold_axes=fold_axes)
    if specialized is not None:
        return specialized
    monoid.validate(values, label="generic fold values")
    initial = monoid.identity(ctx)
    region = _combine_region(monoid, values, removed_axes=fold_axes)
    inputs = _generic_inputs(values, initial)
    leaves = {
        path: ctx.associative_result(
            "associative_fold",
            inputs,
            axes=tuple(axis for axis in value.axes if axis not in fold_axes),
            dtype=value.dtype,
            region=region,
            result_path=path,
            over=fold_axes,
        )
        for path, value in value_leaves(values)
    }
    return cast(ValueT, _replace_leaves(values, leaves))


def lower_scan[ValueT: AlgebraValue](
    ctx: TensorBuilder,
    monoid: Monoid[ValueT],
    values: ValueT,
    *,
    scan_axis: str,
    initial: ValueT,
) -> ScanResult[ValueT]:
    specialized = monoid.lower_scan(
        ctx, values, scan_axis=scan_axis, initial=initial
    )
    if specialized is not None:
        return specialized
    monoid.validate(values, label="generic scan values")
    monoid.validate(initial, label="generic scan initial")
    region = _combine_region(monoid, values, removed_axes=(scan_axis,))
    inputs = _generic_inputs(values, initial)
    prefix_leaves = {}
    final_leaves = {}
    for path, value in value_leaves(values):
        final_axes = tuple(axis for axis in value.axes if axis != scan_axis)
        prefix_leaves[path] = ctx.associative_result(
            "associative_scan_prefix",
            inputs,
            axes=value.axes,
            dtype=value.dtype,
            region=region,
            result_path=path,
            over=(scan_axis,),
        )
        final_leaves[path] = ctx.associative_result(
            "associative_scan_final",
            inputs,
            axes=final_axes,
            dtype=value.dtype,
            region=region,
            result_path=path,
            over=(scan_axis,),
        )
    return ScanResult(
        prefixes=cast(ValueT, _replace_leaves(values, prefix_leaves)),
        final=cast(ValueT, _replace_leaves(values, final_leaves)),
    )


__all__ = ["lower_fold", "lower_scan"]
