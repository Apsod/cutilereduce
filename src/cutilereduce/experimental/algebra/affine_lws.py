"""Log-scale affine action on a log-weighted-sum payload."""

from __future__ import annotations

from typing import cast

from .add import Add, AddMonoid
from .base import (
    AlgebraValue,
    CompositeValue,
    StructuredValue,
    TensorField,
)
from .lws import LWSExecutionMonoid, LogWeightedMonoid, LogWeightedSum
from .product import ProductSemigroup
from .semidirect import SemidirectProductAlgebra, SemidirectProductMonoid
from ..ir import TensorBuilder


def _reweight_semantic(
    ctx: TensorBuilder,
    acting: AlgebraValue,
    payload: AlgebraValue,
) -> AlgebraValue:
    del ctx
    scale = cast(StructuredValue, acting)["value"]
    value = cast(StructuredValue, payload)
    return StructuredValue(
        {
            "log_weight": value["log_weight"] + scale,
            "value": value["value"],
        }
    )


def _reweight_execution(
    ctx: TensorBuilder,
    acting: AlgebraValue,
    payload: AlgebraValue,
) -> AlgebraValue:
    del ctx
    scale = cast(StructuredValue, acting)["value"]
    value = cast(StructuredValue, payload)
    return StructuredValue(
        {
            "maximum": value["maximum"] + scale,
            "normalizer": value["normalizer"],
            "numerator": value["numerator"],
        }
    )


class AffineLWSMonoid(SemidirectProductMonoid):
    """``Add`` acting by log-weight shifts on an LWS payload."""

    def __init__(
        self, value_axes: tuple[str, ...] = (), dtype: str = "float32"
    ):
        super().__init__(
            AddMonoid(dtype=dtype),
            LogWeightedMonoid(
                {"value": TensorField(value_axes, dtype)}, dtype
            ),
            action=_reweight_semantic,
        )

    def lower_fold(
        self,
        ctx: TensorBuilder,
        values: CompositeValue,
        *,
        fold_axes: tuple[str, ...],
    ) -> CompositeValue:
        from .lowering import lower_fold

        self.validate(values, label="affine-LWS values")
        if len(fold_axes) != 1:
            raise ValueError("affine-LWS fold requires one ordered axis")
        fold_axis = fold_axes[0]
        scale = cast(StructuredValue, values["acting"])["value"]
        payload = cast(StructuredValue, values["payload"])
        suffix_scale = ctx.scan(
            "sum", scale, over=fold_axis, exclusive=True, reverse=True
        )
        reweighted = StructuredValue(
            {
                "log_weight": payload["log_weight"] + suffix_scale,
                "value": payload["value"],
            }
        )
        return CompositeValue(
            {
                "acting": StructuredValue(
                    {"value": ctx.reduce("sum", scale, fold_axes)}
                ),
                "payload": lower_fold(
                    ctx, self.payload, reweighted, fold_axes=fold_axes
                ),
            }
        )

class AffineLWSExecutionMonoid(SemidirectProductMonoid):
    def __init__(
        self, value_axes: tuple[str, ...] = (), dtype: str = "float32"
    ):
        super().__init__(
            AddMonoid(dtype=dtype),
            LWSExecutionMonoid(value_axes, dtype),
            action=_reweight_execution,
        )

    def lower_fold(
        self,
        ctx: TensorBuilder,
        values: CompositeValue,
        *,
        fold_axes: tuple[str, ...],
    ) -> CompositeValue:
        from .lowering import lower_fold

        self.validate(values, label="affine-LWS execution values")
        if len(fold_axes) != 1:
            raise ValueError("affine-LWS fold requires one ordered axis")
        fold_axis = fold_axes[0]
        scale = cast(StructuredValue, values["acting"])["value"]
        payload = cast(StructuredValue, values["payload"])
        suffix_scale = ctx.scan(
            "sum", scale, over=fold_axis, exclusive=True, reverse=True
        )
        reweighted = StructuredValue(
            {
                "maximum": payload["maximum"] + suffix_scale,
                "normalizer": payload["normalizer"],
                "numerator": payload["numerator"],
            }
        )
        return CompositeValue(
            {
                "acting": StructuredValue(
                    {"value": ctx.reduce("sum", scale, fold_axes)}
                ),
                "payload": lower_fold(
                    ctx, self.payload, reweighted, fold_axes=fold_axes
                ),
            }
        )

class AffineLWS(SemidirectProductAlgebra):
    """Log-scale affine action with LWS-style payload addition."""

    def __init__(
        self, value_axes: tuple[str, ...] = (), dtype: str = "float32"
    ):
        scale = Add(dtype=dtype)
        payload = LogWeightedSum(value_axes, dtype)
        super().__init__(
            scale,
            payload,
            semantic_action=_reweight_semantic,
            execution_action=_reweight_execution,
        )
        self.semantic = AffineLWSMonoid(value_axes, dtype)
        self.execution = AffineLWSExecutionMonoid(value_axes, dtype)
        scalar = TensorField(dtype=dtype)
        vector = TensorField(value_axes, dtype)
        self.accumulator = ProductSemigroup(
            {
                "acting": AddMonoid(dtype=dtype),
                "payload": LogWeightedMonoid(
                    {"weight": scalar, "value": vector}, dtype
                ),
            }
        )
        self.value_axes = value_axes

    def embed(
        self,
        ctx: TensorBuilder,
        execution: CompositeValue,
        semantic_cotangent: CompositeValue,
    ) -> CompositeValue:
        semantic = self.to_semantic(ctx, execution)
        self.semantic.validate(
            semantic_cotangent, label="affine-LWS semantic cotangent"
        )
        scale = cast(StructuredValue, semantic["acting"])["value"]
        payload = cast(StructuredValue, semantic["payload"])
        scale_cotangent = cast(
            StructuredValue, semantic_cotangent["acting"]
        )["value"]
        payload_cotangent = cast(
            StructuredValue, semantic_cotangent["payload"]
        )
        return CompositeValue(
            {
                "acting": StructuredValue({"value": scale_cotangent}),
                "payload": StructuredValue(
                    {
                        "log_weight": scale - payload["log_weight"],
                        "weight": payload_cotangent["log_weight"]
                        - ctx.contract(
                            payload_cotangent["value"],
                            payload["value"],
                            over=self.value_axes,
                        ),
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
            raise ValueError("affine-LWS finalize requires an E prefix")
        self.semantic.validate(factor, label="affine-LWS local factor")
        self.accumulator.validate(accumulator, label="affine-LWS accumulator")
        prefix_scale = cast(StructuredValue, prefix["acting"])["value"]
        prefix_execution_payload = cast(StructuredValue, prefix["payload"])
        prefix_normalizer = prefix_execution_payload["normalizer"]
        has_prefix_payload = ctx.compare(
            "greater", prefix_normalizer, 0.0
        )
        safe_prefix_normalizer = ctx.where(
            has_prefix_payload,
            prefix_normalizer,
            ctx.full(prefix_normalizer.axes, 1.0, prefix_normalizer.dtype),
        )
        prefix_payload = StructuredValue(
            {
                "log_weight": prefix_execution_payload["maximum"]
                + ctx.log(safe_prefix_normalizer),
                "value": prefix_execution_payload["numerator"]
                / safe_prefix_normalizer,
            }
        )
        local_scale = cast(StructuredValue, factor["acting"])["value"]
        local_payload = cast(StructuredValue, factor["payload"])
        accumulated_scale = cast(StructuredValue, accumulator["acting"])[
            "value"
        ]
        accumulated_payload = cast(
            StructuredValue, accumulator["payload"]
        )

        prefix_inner = accumulated_payload["weight"] + ctx.contract(
            accumulated_payload["value"],
            prefix_payload["value"],
            over=self.value_axes,
        )
        local_inner = accumulated_payload["weight"] + ctx.contract(
            accumulated_payload["value"],
            local_payload["value"],
            over=self.value_axes,
        )
        prefix_scale_factor = ctx.exp(
            accumulated_payload["log_weight"]
            - prefix_scale
            + prefix_payload["log_weight"]
        )
        local_scale_factor = ctx.exp(
            accumulated_payload["log_weight"]
            - prefix_scale
            - local_scale
            + local_payload["log_weight"]
        )
        return CompositeValue(
            {
                "acting": StructuredValue(
                    {
                        "value": accumulated_scale
                        + prefix_scale_factor * prefix_inner
                    }
                ),
                "payload": StructuredValue(
                    {
                        "log_weight": local_scale_factor * local_inner,
                        "value": local_scale_factor
                        * accumulated_payload["value"],
                    }
                ),
            }
        )


__all__ = [
    "AffineLWS",
    "AffineLWSExecutionMonoid",
    "AffineLWSMonoid",
]
