"""Log-weighted monoids and the log-weighted-sum fold algebra."""

from __future__ import annotations

from collections.abc import Mapping

from .base import (
    FinalizeContext,
    FoldAlgebra,
    Monoid,
    StructuredValue,
    TensorField,
    TensorStructure,
)
from ..ir import TensorBuilder, TensorValue


class LogWeightedMonoid(Monoid[StructuredValue]):
    """LWS over an arbitrary flat record of payload tensor fields."""

    commutative = True

    def __init__(
        self,
        payload_fields: Mapping[str, TensorField],
        dtype: str = "float32",
    ):
        if "log_weight" in payload_fields:
            raise ValueError("LWS payload cannot contain a 'log_weight' field")
        self.dtype = dtype
        self.payload_fields = dict(payload_fields)
        self.structure = TensorStructure(
            {"log_weight": TensorField(dtype=dtype), **self.payload_fields}
        )

    def identity(self, ctx: TensorBuilder) -> StructuredValue:
        return StructuredValue(
            {
                "log_weight": ctx.constant(float("-inf"), self.dtype),
                **{
                    name: ctx.full(field.inner_axes, 0.0, field.dtype)
                    for name, field in self.payload_fields.items()
                },
            }
        )

    def combine(
        self,
        ctx: TensorBuilder,
        left: StructuredValue,
        right: StructuredValue,
    ) -> StructuredValue:
        self.validate(left, label="left LWS value")
        self.validate(right, label="right LWS value")
        left_is_high = ctx.compare(
            "greater", left["log_weight"], right["log_weight"]
        )
        maximum = ctx.where(
            left_is_high, left["log_weight"], right["log_weight"]
        )
        left_weight = ctx.exp(left["log_weight"] - maximum)
        right_weight = ctx.exp(right["log_weight"] - maximum)
        normalizer = left_weight + right_weight
        log_weight = maximum + ctx.log(normalizer)
        fields: dict[str, TensorValue] = {"log_weight": log_weight}
        for name in self.payload_fields:
            fields[name] = (
                left_weight * left[name] + right_weight * right[name]
            ) / normalizer
        return StructuredValue(fields)

    def lower_fold(
        self,
        ctx: TensorBuilder,
        values: StructuredValue,
        *,
        fold_axes: tuple[str, ...],
    ) -> StructuredValue:
        self.validate(values, label="LWS values")
        maximum = ctx.reduce("max", values["log_weight"], fold_axes)
        weights = ctx.exp(values["log_weight"] - maximum)
        normalizer = ctx.reduce("sum", weights, fold_axes)
        fields: dict[str, TensorValue] = {
            "log_weight": maximum + ctx.log(normalizer)
        }
        for name in self.payload_fields:
            fields[name] = (
                ctx.contract(weights, values[name], over=fold_axes)
                / normalizer
            )
        return StructuredValue(fields)

class LWSExecutionMonoid(Monoid[StructuredValue]):
    """Stable execution carrier for an LWS with one tensor payload."""

    commutative = True

    def __init__(self, value_axes: tuple[str, ...], dtype: str = "float32"):
        scalar = TensorField(dtype=dtype)
        self.value_field = TensorField(value_axes, dtype)
        self.structure = TensorStructure(
            {
                "maximum": scalar,
                "normalizer": scalar,
                "numerator": self.value_field,
            }
        )

    def identity(self, ctx: TensorBuilder) -> StructuredValue:
        return StructuredValue(
            {
                "maximum": ctx.constant(float("-inf")),
                "normalizer": ctx.constant(0.0),
                "numerator": ctx.full(
                    self.value_field.inner_axes, 0.0, self.value_field.dtype
                ),
            }
        )

    def combine(
        self,
        ctx: TensorBuilder,
        left: StructuredValue,
        right: StructuredValue,
    ) -> StructuredValue:
        self.validate(left, label="left LWS execution carrier")
        self.validate(right, label="right LWS execution carrier")
        left_is_high = ctx.compare("greater", left["maximum"], right["maximum"])
        maximum = ctx.where(left_is_high, left["maximum"], right["maximum"])
        left_scale = ctx.exp(left["maximum"] - maximum)
        right_scale = ctx.exp(right["maximum"] - maximum)
        return StructuredValue(
            {
                "maximum": maximum,
                "normalizer": left_scale * left["normalizer"]
                + right_scale * right["normalizer"],
                "numerator": left_scale * left["numerator"]
                + right_scale * right["numerator"],
            }
        )

    def lower_fold(
        self,
        ctx: TensorBuilder,
        values: StructuredValue,
        *,
        fold_axes: tuple[str, ...],
    ) -> StructuredValue:
        self.validate(values, label="LWS execution carriers")
        maximum = ctx.reduce("max", values["maximum"], fold_axes)
        scales = ctx.exp(values["maximum"] - maximum)
        normalizer = ctx.reduce(
            "sum", scales * values["normalizer"], fold_axes
        )
        numerator = ctx.contract(
            scales, values["numerator"], over=fold_axes
        )
        return StructuredValue(
            {
                "maximum": maximum,
                "normalizer": normalizer,
                "numerator": numerator,
            }
        )

class LogWeightedSum(
    FoldAlgebra[StructuredValue, StructuredValue, StructuredValue]
):
    """Stable log-weighted summation over one tensor payload."""

    finalize_context = FinalizeContext.NONE

    def __init__(self, value_axes: tuple[str, ...], dtype: str = "float32"):
        scalar = TensorField(dtype=dtype)
        value = TensorField(value_axes, dtype)
        self.semantic = LogWeightedMonoid({"value": value}, dtype)
        self.execution = LWSExecutionMonoid(value_axes, dtype)
        self.accumulator = LogWeightedMonoid(
            {"scalar": scalar, "value": value}, dtype
        )
        self.value_axes = value_axes

    def to_execution(
        self, ctx: TensorBuilder, semantic: StructuredValue
    ) -> StructuredValue:
        self.semantic.validate(semantic, label="LWS semantic factor")
        log_weight = semantic["log_weight"]
        return StructuredValue(
            {
                "maximum": log_weight,
                "normalizer": ctx.full(log_weight.axes, 1.0, log_weight.dtype),
                "numerator": semantic["value"],
            }
        )

    def to_semantic(
        self, ctx: TensorBuilder, execution: StructuredValue
    ) -> StructuredValue:
        self.execution.validate(execution, label="LWS execution carrier")
        return StructuredValue(
            {
                "log_weight": execution["maximum"]
                + ctx.log(execution["normalizer"]),
                "value": execution["numerator"] / execution["normalizer"],
            }
        )

    def embed(
        self,
        ctx: TensorBuilder,
        execution: StructuredValue,
        semantic_cotangent: StructuredValue,
    ) -> StructuredValue:
        result = self.to_semantic(ctx, execution)
        self.semantic.validate(
            semantic_cotangent, label="LWS semantic cotangent"
        )
        return StructuredValue(
            {
                "log_weight": result["log_weight"] * -1.0,
                "scalar": semantic_cotangent["log_weight"]
                - ctx.contract(
                    semantic_cotangent["value"],
                    result["value"],
                    over=self.value_axes,
                ),
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
        self.semantic.validate(factor, label="LWS local factor")
        self.accumulator.validate(accumulator, label="LWS accumulator")
        scale = ctx.exp(factor["log_weight"] + accumulator["log_weight"])
        return StructuredValue(
            {
                "log_weight": scale
                * (
                    accumulator["scalar"]
                    + ctx.contract(
                        accumulator["value"],
                        factor["value"],
                        over=self.value_axes,
                    )
                ),
                "value": scale * accumulator["value"],
            }
        )


__all__ = [
    "LWSExecutionMonoid",
    "LogWeightedMonoid",
    "LogWeightedSum",
]
