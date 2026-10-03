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
        return StructuredValue({"value": ctx.constant(float("-inf"), self.dtype)})

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
        nonempty = ctx.compare("not_equal", maximum, float("-inf"))
        safe_maximum = ctx.where(nonempty, maximum, ctx.constant(0.0, self.dtype))
        normalizer = ctx.where(
            nonempty,
            ctx.exp(left["value"] - safe_maximum)
            + ctx.exp(right["value"] - safe_maximum),
            ctx.constant(1.0, self.dtype),
        )
        return StructuredValue(
            {
                "value": ctx.where(
                    nonempty,
                    maximum + ctx.log(normalizer),
                    ctx.constant(float("-inf"), self.dtype),
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
        nonempty = ctx.compare("not_equal", maximum, float("-inf"))
        safe_maximum = ctx.where(nonempty, maximum, ctx.constant(0.0, self.dtype))
        normalizer = ctx.reduce(
            "sum",
            ctx.where(
                nonempty,
                ctx.exp(values["value"] - safe_maximum),
                ctx.full(values["value"].axes, 0.0, self.dtype),
            ),
            fold_axes,
        )
        return StructuredValue(
            {
                "value": ctx.where(
                    nonempty,
                    maximum + ctx.log(normalizer),
                    ctx.constant(float("-inf"), self.dtype),
                )
            }
        )


class LSEExecutionMonoid(Monoid[StructuredValue]):
    commutative = True

    def __init__(self, dtype: str = "float32"):
        self.dtype = dtype
        scalar = TensorField(dtype=dtype)
        self.structure = TensorStructure({"maximum": scalar, "normalizer": scalar})

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
        nonempty = ctx.compare("not_equal", maximum, float("-inf"))
        safe_maximum = ctx.where(nonempty, maximum, ctx.constant(0.0, self.dtype))
        return StructuredValue(
            {
                "maximum": maximum,
                "normalizer": ctx.where(
                    nonempty,
                    ctx.exp(left["maximum"] - safe_maximum),
                    ctx.constant(0.0, self.dtype),
                )
                * left["normalizer"]
                + ctx.where(
                    nonempty,
                    ctx.exp(right["maximum"] - safe_maximum),
                    ctx.constant(0.0, self.dtype),
                )
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
        nonempty = ctx.compare("not_equal", maximum, float("-inf"))
        safe_maximum = ctx.where(nonempty, maximum, ctx.constant(0.0, self.dtype))
        normalizer = ctx.reduce(
            "sum",
            ctx.where(
                nonempty,
                ctx.exp(values["maximum"] - safe_maximum),
                ctx.full(values["maximum"].axes, 0.0, self.dtype),
            )
            * values["normalizer"],
            fold_axes,
        )
        return StructuredValue({"maximum": maximum, "normalizer": normalizer})


class LogSumExp(FoldAlgebra[StructuredValue, StructuredValue, StructuredValue]):
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
            {"value": execution["maximum"] + ctx.log(execution["normalizer"])}
        )

    def embed(
        self,
        ctx: TensorBuilder,
        execution: StructuredValue,
        semantic_cotangent: StructuredValue,
    ) -> StructuredValue:
        semantic = self.to_semantic(ctx, execution)
        self.semantic.validate(semantic_cotangent, label="LSE semantic cotangent")
        return StructuredValue(
            {
                "log_weight": -semantic["value"],
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
                "value": ctx.exp(factor["value"] + accumulator["log_weight"])
                * accumulator["value"]
            }
        )


__all__ = ["LSEExecutionMonoid", "LogAddExpMonoid", "LogSumExp"]
