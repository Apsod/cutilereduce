"""Additive monoid and its carrier-resolved fold algebra."""

from __future__ import annotations

from .base import FoldAlgebra, Monoid, StructuredValue, TensorField, TensorStructure
from ..ir import TensorBuilder


class AddMonoid(Monoid[StructuredValue]):
    def __init__(self, inner_axes: tuple[str, ...] = (), dtype: str = "float32"):
        self.field = TensorField(inner_axes, dtype)
        self.structure = TensorStructure({"value": self.field})

    def lower_identity(self, ctx: TensorBuilder) -> StructuredValue:
        return StructuredValue(
            {"value": ctx.full(self.field.inner_axes, 0.0, self.field.dtype)}
        )

    def lower_combine(
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


class Add(FoldAlgebra[StructuredValue, StructuredValue, StructuredValue]):
    def __init__(self, inner_axes: tuple[str, ...] = (), dtype: str = "float32"):
        monoid = AddMonoid(inner_axes, dtype)
        self.semantic = monoid
        self.execution = monoid
        self.adjoint = monoid

    def lower_phi(
        self, ctx: TensorBuilder, semantic: StructuredValue
    ) -> StructuredValue:
        del ctx
        self.semantic.validate(semantic, label="add semantic factor")
        return semantic

    def lower_psi(
        self, ctx: TensorBuilder, execution: StructuredValue
    ) -> StructuredValue:
        del ctx
        self.execution.validate(execution, label="add execution carrier")
        return execution

    def lower_embed(
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

    def lower_finalize(
        self,
        ctx: TensorBuilder,
        factor: StructuredValue,
        adjoint: StructuredValue,
        *,
        prefix: StructuredValue | None = None,
    ) -> StructuredValue:
        del ctx, factor, prefix
        self.adjoint.validate(adjoint, label="add adjoint")
        return adjoint


__all__ = ["Add", "AddMonoid"]
