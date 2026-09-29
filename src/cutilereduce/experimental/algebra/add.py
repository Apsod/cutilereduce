"""Additive monoid and its carrier-resolved fold algebra."""

from __future__ import annotations

from .base import (
    FinalizeContext,
    FoldAlgebra,
    Monoid,
    ScanResult,
    StructuredValue,
    TensorField,
    TensorStructure,
)
from ..ir import TensorBuilder


class AddMonoid(Monoid[StructuredValue]):
    commutative = True
    def __init__(self, inner_axes: tuple[str, ...] = (), dtype: str = "float32"):
        self.field = TensorField(inner_axes, dtype)
        self.structure = TensorStructure({"value": self.field})

    def identity(self, ctx: TensorBuilder) -> StructuredValue:
        return StructuredValue(
            {"value": ctx.full(self.field.inner_axes, 0.0, self.field.dtype)}
        )

    def combine(
        self,
        ctx: TensorBuilder,
        left: StructuredValue,
        right: StructuredValue,
    ) -> StructuredValue:
        del ctx
        self.validate(left, label="left additive value")
        self.validate(right, label="right additive value")
        return StructuredValue({"value": left["value"] + right["value"]})

    def lower_fold(
        self,
        ctx: TensorBuilder,
        values: StructuredValue,
        *,
        fold_axes: tuple[str, ...],
    ) -> StructuredValue:
        self.validate(values, label="additive values")
        return StructuredValue(
            {"value": ctx.reduce("sum", values["value"], fold_axes)}
        )

    def lower_scan(
        self,
        ctx: TensorBuilder,
        values: StructuredValue,
        *,
        scan_axis: str,
        initial: StructuredValue,
    ) -> ScanResult[StructuredValue]:
        self.validate(values, label="additive scan values")
        self.validate(initial, label="additive scan initial")
        prefixes = initial["value"] + ctx.scan(
            "sum", values["value"], over=scan_axis, exclusive=True
        )
        final = initial["value"] + ctx.reduce(
            "sum", values["value"], (scan_axis,)
        )
        return ScanResult(
            prefixes=StructuredValue({"value": prefixes}),
            final=StructuredValue({"value": final}),
        )


class Add(FoldAlgebra[StructuredValue, StructuredValue, StructuredValue]):
    finalize_context = FinalizeContext.NONE
    def __init__(self, inner_axes: tuple[str, ...] = (), dtype: str = "float32"):
        monoid = AddMonoid(inner_axes, dtype)
        self.semantic = monoid
        self.execution = monoid
        self.accumulator = monoid

    def to_execution(
        self, ctx: TensorBuilder, semantic: StructuredValue
    ) -> StructuredValue:
        del ctx
        self.semantic.validate(semantic, label="add semantic factor")
        return semantic

    def to_semantic(
        self, ctx: TensorBuilder, execution: StructuredValue
    ) -> StructuredValue:
        del ctx
        self.execution.validate(execution, label="add execution carrier")
        return execution

    def embed(
        self,
        ctx: TensorBuilder,
        execution: StructuredValue,
        semantic_cotangent: StructuredValue,
    ) -> StructuredValue:
        del ctx, execution
        self.semantic.validate(
            semantic_cotangent, label="add semantic cotangent"
        )
        return semantic_cotangent

    def finalize(
        self,
        ctx: TensorBuilder,
        factor: StructuredValue,
        accumulator: StructuredValue,
        *,
        prefix: StructuredValue | None = None,
    ) -> StructuredValue:
        del ctx, factor, prefix
        self.accumulator.validate(accumulator, label="add accumulator")
        return accumulator


__all__ = ["Add", "AddMonoid"]
