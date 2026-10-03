"""Log-scale affine fold algebra, a concrete semidirect-product instance."""

from __future__ import annotations

from typing import cast

from .add import Add, AddMonoid
from .base import (
    AlgebraValue,
    CompositeValue,
    StructuredValue,
    TensorField,
)
from .lws import LogWeightedMonoid
from .product import ProductSemigroup
from .semidirect import SemidirectProductAlgebra, SemidirectProductMonoid
from ..ir import TensorBuilder


def _scale_payload(
    ctx: TensorBuilder,
    acting: AlgebraValue,
    value: AlgebraValue,
) -> AlgebraValue:
    acting_value = cast(StructuredValue, acting)["value"]
    payload_value = cast(StructuredValue, value)["value"]
    return StructuredValue({"value": ctx.exp(acting_value) * payload_value})


class AffineMonoid(SemidirectProductMonoid):
    def __init__(
        self, payload_axes: tuple[str, ...] = (), dtype: str = "float32"
    ):
        super().__init__(
            AddMonoid(dtype=dtype),
            AddMonoid(payload_axes, dtype),
            action=_scale_payload,
        )

    def lower_fold(
        self,
        ctx: TensorBuilder,
        values: CompositeValue,
        *,
        fold_axes: tuple[str, ...],
    ) -> CompositeValue:
        self.validate(values, label="affine values")
        if len(fold_axes) != 1:
            raise ValueError("log-scale affine fold requires one ordered axis")
        fold_axis = fold_axes[0]
        scale = cast(StructuredValue, values["acting"])["value"]
        payload = cast(StructuredValue, values["payload"])["value"]
        suffix_scale = ctx.scan(
            "sum", scale, over=fold_axis, exclusive=True, reverse=True
        )
        return CompositeValue(
            {
                "acting": StructuredValue(
                    {"value": ctx.reduce("sum", scale, fold_axes)}
                ),
                "payload": StructuredValue(
                    {
                        "value": ctx.reduce(
                            "sum", ctx.exp(suffix_scale) * payload, fold_axes
                        )
                    }
                ),
            }
        )

class LogScaleAffine(SemidirectProductAlgebra):
    """Affine maps ``x -> exp(scale) * x + payload`` under composition."""

    def __init__(
        self, payload_axes: tuple[str, ...] = (), dtype: str = "float32"
    ):
        scale = Add(dtype=dtype)
        payload = Add(payload_axes, dtype=dtype)
        super().__init__(scale, payload, semantic_action=_scale_payload)
        affine = AffineMonoid(payload_axes, dtype)
        self.semantic = affine
        self.execution = affine
        self.accumulator = ProductSemigroup(
            {
                "acting": AddMonoid(dtype=dtype),
                "payload": LogWeightedMonoid(
                    {"value": TensorField(payload_axes, dtype)}, dtype
                ),
            }
        )
        self.payload_axes = payload_axes

    def embed(
        self,
        ctx: TensorBuilder,
        execution: CompositeValue,
        semantic_cotangent: CompositeValue,
    ) -> CompositeValue:
        self.execution.validate(execution, label="affine execution carrier")
        self.semantic.validate(
            semantic_cotangent, label="affine semantic cotangent"
        )
        execution_scale = cast(StructuredValue, execution["acting"])
        scale_cotangent = cast(StructuredValue, semantic_cotangent["acting"])
        payload_cotangent = cast(StructuredValue, semantic_cotangent["payload"])
        return CompositeValue(
            {
                "acting": StructuredValue(
                    {"value": scale_cotangent["value"]}
                ),
                "payload": StructuredValue(
                    {
                        "log_weight": execution_scale["value"],
                        "value": payload_cotangent["value"],
                    }
                ),
            }
        )

    def finalize(
        self,
        ctx: TensorBuilder,
        factor: CompositeValue,
        accumulator: CompositeValue,
        *,
        prefix: CompositeValue | None = None,
    ) -> CompositeValue:
        if prefix is None:
            raise ValueError("log-scale affine finalize requires an E prefix")
        self.semantic.validate(factor, label="affine local factor")
        self.accumulator.validate(accumulator, label="affine accumulator")
        self.execution.validate(prefix, label="affine execution prefix")
        local_scale = cast(StructuredValue, factor["acting"])["value"]
        prefix_scale = cast(StructuredValue, prefix["acting"])["value"]
        prefix_payload = cast(StructuredValue, prefix["payload"])["value"]
        accumulated_scale = cast(StructuredValue, accumulator["acting"])[
            "value"
        ]
        accumulated_payload = cast(
            StructuredValue, accumulator["payload"]
        )
        relative_payload = (
            ctx.exp(accumulated_payload["log_weight"] - prefix_scale)
            * accumulated_payload["value"]
        )
        local_payload_cotangent = ctx.exp(-local_scale) * relative_payload
        scale_from_payload = prefix_payload * relative_payload
        if self.payload_axes:
            scale_from_payload = ctx.reduce(
                "sum", scale_from_payload, self.payload_axes
            )
        return CompositeValue(
            {
                "acting": StructuredValue(
                    {"value": accumulated_scale + scale_from_payload}
                ),
                "payload": StructuredValue(
                    {"value": local_payload_cotangent}
                ),
            }
        )


Affine = LogScaleAffine


__all__ = ["Affine", "AffineMonoid", "LogScaleAffine"]
