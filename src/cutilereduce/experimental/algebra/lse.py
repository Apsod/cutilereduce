"""Numerically stable log-sum-exp fold algebra."""

from __future__ import annotations

from .base import (
    FinalizeContext,
    FoldAlgebra,
    Monoid,
    StructuredValue,
    TensorField,
    TensorStructure,
)
from .lws import LogWeightedMonoid
from ..ir import TensorBuilder


class LogAddExpMonoid(Monoid[StructuredValue]):
    commutative = True
    def __init__(self, dtype: str = "float32"):
        self.dtype = dtype
        self.structure = TensorStructure({"value": TensorField(dtype=dtype)})

    def identity(self, ctx: TensorBuilder) -> StructuredValue:
        return StructuredValue(
            {"value": ctx.constant(float("-inf"), self.dtype)}
        )

    def combine(
        self,
        ctx: TensorBuilder,
        left: StructuredValue,
        right: StructuredValue,
    ) -> StructuredValue:
        self.validate(left, label="left logaddexp value")
        self.validate(right, label="right logaddexp value")
        maximum = ctx.where(
            ctx.compare("greater", left["value"], right["value"]),
            left["value"],
            right["value"],
        )
        return StructuredValue(
            {
                "value": maximum
                + ctx.log(
                    ctx.exp(left["value"] - maximum)
                    + ctx.exp(right["value"] - maximum)
                )
            }
        )

    def lower_fold(
        self,
        ctx: TensorBuilder,
        values: StructuredValue,
        *,
        fold_axes: tuple[str, ...],
    ) -> StructuredValue:
        self.validate(values, label="logaddexp values")
        maximum = ctx.reduce("max", values["value"], fold_axes)
        normalizer = ctx.reduce(
            "sum", ctx.exp(values["value"] - maximum), fold_axes
        )
        return StructuredValue({"value": maximum + ctx.log(normalizer)})

class LSEExecutionMonoid(Monoid[StructuredValue]):
    commutative = True
    def __init__(self, dtype: str = "float32"):
        self.dtype = dtype
        scalar = TensorField(dtype=dtype)
        self.structure = TensorStructure(
            {"maximum": scalar, "normalizer": scalar}
        )

    def identity(self, ctx: TensorBuilder) -> StructuredValue:
        return StructuredValue(
            {
                "maximum": ctx.constant(float("-inf"), self.dtype),
                "normalizer": ctx.constant(0.0, self.dtype),
            }
        )

    def combine(
        self,
        ctx: TensorBuilder,
        left: StructuredValue,
        right: StructuredValue,
    ) -> StructuredValue:
        self.validate(left, label="left LSE carrier")
        self.validate(right, label="right LSE carrier")
        maximum = ctx.where(
            ctx.compare("greater", left["maximum"], right["maximum"]),
            left["maximum"],
            right["maximum"],
        )
        return StructuredValue(
            {
                "maximum": maximum,
                "normalizer": ctx.exp(left["maximum"] - maximum)
                * left["normalizer"]
                + ctx.exp(right["maximum"] - maximum)
                * right["normalizer"],
            }
        )

    def lower_fold(
        self,
        ctx: TensorBuilder,
        values: StructuredValue,
        *,
        fold_axes: tuple[str, ...],
    ) -> StructuredValue:
        self.validate(values, label="LSE execution carriers")
        maximum = ctx.reduce("max", values["maximum"], fold_axes)
        normalizer = ctx.reduce(
            "sum",
            ctx.exp(values["maximum"] - maximum) * values["normalizer"],
            fold_axes,
        )
        return StructuredValue(
            {"maximum": maximum, "normalizer": normalizer}
        )

class LogSumExp(
    FoldAlgebra[StructuredValue, StructuredValue, StructuredValue]
):
    finalize_context = FinalizeContext.NONE
    def __init__(self, dtype: str = "float32"):
        scalar = TensorField(dtype=dtype)
        self.semantic = LogAddExpMonoid(dtype)
        self.execution = LSEExecutionMonoid(dtype)
        self.accumulator = LogWeightedMonoid({"value": scalar}, dtype)

    def to_execution(
        self, ctx: TensorBuilder, semantic: StructuredValue
    ) -> StructuredValue:
        self.semantic.validate(semantic, label="LSE semantic factor")
        value = semantic["value"]
        return StructuredValue(
            {
                "maximum": value,
                "normalizer": ctx.full(value.axes, 1.0, value.dtype),
            }
        )

    def to_semantic(
        self, ctx: TensorBuilder, execution: StructuredValue
    ) -> StructuredValue:
        self.execution.validate(execution, label="LSE execution carrier")
        return StructuredValue(
            {
                "value": execution["maximum"]
                + ctx.log(execution["normalizer"])
            }
        )

    def embed(
        self,
        ctx: TensorBuilder,
        execution: StructuredValue,
        semantic_cotangent: StructuredValue,
    ) -> StructuredValue:
        semantic = self.to_semantic(ctx, execution)
        self.semantic.validate(
            semantic_cotangent, label="LSE semantic cotangent"
        )
        return StructuredValue(
            {
                "log_weight": semantic["value"] * -1.0,
                "value": semantic_cotangent["value"],
            }
        )

    def finalize(
        self,
        ctx: TensorBuilder,
        factor: StructuredValue,
        accumulator: StructuredValue,
        *,
        prefix: StructuredValue | None = None,
    ) -> StructuredValue:
        del prefix
        self.semantic.validate(factor, label="LSE semantic factor")
        self.accumulator.validate(accumulator, label="LSE accumulator")
        return StructuredValue(
            {
                "value": ctx.exp(
                    factor["value"] + accumulator["log_weight"]
                )
                * accumulator["value"]
            }
        )


__all__ = ["LSEExecutionMonoid", "LogAddExpMonoid", "LogSumExp"]
