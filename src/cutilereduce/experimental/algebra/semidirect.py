"""Semidirect-product monoids and a log-scale affine fold algebra."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

from .add import Add, AddMonoid
from .base import (
    AlgebraValue,
    CompositeStructure,
    CompositeValue,
    FoldAlgebra,
    Monoid,
    StructuredValue,
    TensorField,
    TensorStructure,
)
from .product import ProductMonoid
from ..ir import TensorBuilder


type ActionLowering = Callable[
    [TensorBuilder, AlgebraValue, AlgebraValue], AlgebraValue
]


class SemidirectProductMonoid(Monoid[CompositeValue]):
    """A right-oriented semidirect product derived from an action."""

    def __init__(
        self,
        acting: Monoid[Any],
        payload: Monoid[Any],
        *,
        action: ActionLowering,
    ):
        self.acting = acting
        self.payload = payload
        self.action = action
        self.structure = CompositeStructure(
            {"acting": acting.structure, "payload": payload.structure}
        )

    def lower_identity(self, ctx: TensorBuilder) -> CompositeValue:
        return CompositeValue(
            {
                "acting": self.acting.lower_identity(ctx),
                "payload": self.payload.lower_identity(ctx),
            }
        )

    def lower_combine(
        self, ctx: TensorBuilder, left: CompositeValue, right: CompositeValue
    ) -> CompositeValue:
        self.validate(left, label="left semidirect value")
        self.validate(right, label="right semidirect value")
        acted = self.action(ctx, right["acting"], left["payload"])
        return CompositeValue(
            {
                "acting": self.acting.lower_combine(
                    ctx, left["acting"], right["acting"]
                ),
                "payload": self.payload.lower_combine(
                    ctx, acted, right["payload"]
                ),
            }
        )


class SemidirectProductAlgebra(
    FoldAlgebra[CompositeValue, CompositeValue, CompositeValue]
):
    """Partially derived fold algebra for a semidirect construction.

    Its component relationships derive Phi and Psi. A concrete resolved
    algebra still supplies its adjoint monoid, embed, and finalize.
    """

    def __init__(
        self,
        acting: FoldAlgebra[Any, Any, Any],
        payload: FoldAlgebra[Any, Any, Any],
        *,
        semantic_action: ActionLowering,
        execution_action: ActionLowering | None = None,
    ):
        self.acting = acting
        self.payload = payload
        self.semantic = SemidirectProductMonoid(
            acting.semantic, payload.semantic, action=semantic_action
        )
        self.execution = SemidirectProductMonoid(
            acting.execution,
            payload.execution,
            action=execution_action or semantic_action,
        )
        self.adjoint = ProductMonoid(
            {"acting": acting.adjoint, "payload": payload.adjoint}
        )

    def value(
        self, *, acting: AlgebraValue, payload: AlgebraValue
    ) -> CompositeValue:
        result = CompositeValue({"acting": acting, "payload": payload})
        self.semantic.validate(result, label="semantic semidirect product")
        return result

    def lower_phi(
        self, ctx: TensorBuilder, semantic: CompositeValue
    ) -> CompositeValue:
        self.semantic.validate(semantic, label="semantic semidirect product")
        return CompositeValue(
            {
                "acting": self.acting.lower_phi(ctx, semantic["acting"]),
                "payload": self.payload.lower_phi(ctx, semantic["payload"]),
            }
        )

    def lower_psi(
        self, ctx: TensorBuilder, execution: CompositeValue
    ) -> CompositeValue:
        self.execution.validate(execution, label="semidirect carrier")
        return CompositeValue(
            {
                "acting": self.acting.lower_psi(ctx, execution["acting"]),
                "payload": self.payload.lower_psi(ctx, execution["payload"]),
            }
        )


class AffineMonoid(SemidirectProductMonoid):
    def __init__(
        self, payload_axes: tuple[str, ...] = (), dtype: str = "float32"
    ):
        def scale_payload(
            ctx: TensorBuilder,
            acting: AlgebraValue,
            value: AlgebraValue,
        ) -> AlgebraValue:
            acting_value = cast(StructuredValue, acting)["value"]
            payload_value = cast(StructuredValue, value)["value"]
            return StructuredValue(
                {"value": ctx.exp(acting_value) * payload_value}
            )

        super().__init__(
            AddMonoid(dtype=dtype),
            AddMonoid(payload_axes, dtype),
            action=scale_payload,
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
            raise ValueError("log-scale affine fold currently requires one ordered axis")
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


class AffineAdjointMonoid(Monoid[CompositeValue]):
    def __init__(
        self, payload_axes: tuple[str, ...] = (), dtype: str = "float32"
    ):
        scalar = TensorField(dtype=dtype)
        count = TensorField(dtype="int32")
        payload = TensorField(payload_axes, dtype)
        self.structure = CompositeStructure(
            {
                "acting": TensorStructure(
                    {"total": scalar, "count": count, "cotangent": scalar}
                ),
                "payload": TensorStructure({"cotangent": payload}),
            }
        )
        self.payload_axes = payload_axes
        self.dtype = dtype

    def lower_identity(self, ctx: TensorBuilder) -> CompositeValue:
        return CompositeValue(
            {
                "acting": StructuredValue(
                    {
                        "total": ctx.constant(0.0, self.dtype),
                        "count": ctx.constant(0, "int32"),
                        "cotangent": ctx.constant(0.0, self.dtype),
                    }
                ),
                "payload": StructuredValue(
                    {
                        "cotangent": ctx.full(
                            self.payload_axes, 0.0, self.dtype
                        )
                    }
                ),
            }
        )

    def lower_combine(
        self, ctx: TensorBuilder, left: CompositeValue, right: CompositeValue
    ) -> CompositeValue:
        self.validate(left, label="left affine adjoint")
        self.validate(right, label="right affine adjoint")
        left_scale = cast(StructuredValue, left["acting"])
        right_scale = cast(StructuredValue, right["acting"])
        left_payload = cast(StructuredValue, left["payload"])
        right_payload = cast(StructuredValue, right["payload"])
        left_present = ctx.compare("greater", left_scale["count"], 0)
        return CompositeValue(
            {
                "acting": StructuredValue(
                    {
                        "total": ctx.where(
                            left_present,
                            left_scale["total"],
                            right_scale["total"],
                        ),
                        "count": left_scale["count"] + right_scale["count"],
                        "cotangent": left_scale["cotangent"]
                        + right_scale["cotangent"],
                    }
                ),
                "payload": StructuredValue(
                    {
                        "cotangent": left_payload["cotangent"]
                        + right_payload["cotangent"]
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

        def scale_payload(
            ctx: TensorBuilder,
            acting: AlgebraValue,
            value: AlgebraValue,
        ) -> AlgebraValue:
            acting_value = cast(StructuredValue, acting)["value"]
            payload_value = cast(StructuredValue, value)["value"]
            return StructuredValue(
                {"value": ctx.exp(acting_value) * payload_value}
            )

        super().__init__(
            scale, payload, semantic_action=scale_payload
        )
        affine = AffineMonoid(payload_axes, dtype)
        self.semantic = affine
        self.execution = affine
        self.adjoint = AffineAdjointMonoid(payload_axes, dtype)
        self.payload_axes = payload_axes

    def lower_embed(
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
                    {
                        "total": execution_scale["value"],
                        "count": ctx.full(
                            execution_scale["value"].axes, 1, "int32"
                        ),
                        "cotangent": scale_cotangent["value"],
                    }
                ),
                "payload": StructuredValue(
                    {"cotangent": payload_cotangent["value"]}
                ),
            }
        )

    def lower_finalize(
        self,
        ctx: TensorBuilder,
        factor: CompositeValue,
        adjoint: CompositeValue,
        *,
        prefix: CompositeValue | None = None,
    ) -> CompositeValue:
        if prefix is None:
            raise ValueError("log-scale affine finalize requires an E prefix")
        self.semantic.validate(factor, label="affine local factor")
        self.adjoint.validate(adjoint, label="affine adjoint")
        self.execution.validate(prefix, label="affine execution prefix")
        local_scale = cast(StructuredValue, factor["acting"])["value"]
        prefix_scale = cast(StructuredValue, prefix["acting"])["value"]
        prefix_payload = cast(StructuredValue, prefix["payload"])["value"]
        adjoint_scale = cast(StructuredValue, adjoint["acting"])
        payload_cotangent = cast(StructuredValue, adjoint["payload"])[
            "cotangent"
        ]
        suffix_scale = adjoint_scale["total"] - prefix_scale - local_scale
        local_payload_cotangent = ctx.exp(suffix_scale) * payload_cotangent
        scale_from_payload = (
            ctx.exp(suffix_scale + local_scale)
            * prefix_payload
            * payload_cotangent
        )
        if self.payload_axes:
            scale_from_payload = ctx.reduce(
                "sum", scale_from_payload, self.payload_axes
            )
        return CompositeValue(
            {
                "acting": StructuredValue(
                    {
                        "value": adjoint_scale["cotangent"]
                        + scale_from_payload
                    }
                ),
                "payload": StructuredValue(
                    {"value": local_payload_cotangent}
                ),
            }
        )


Affine = LogScaleAffine
SemidirectProduct = SemidirectProductAlgebra


__all__ = [
    "Affine",
    "AffineAdjointMonoid",
    "AffineMonoid",
    "LogScaleAffine",
    "SemidirectProduct",
    "SemidirectProductAlgebra",
    "SemidirectProductMonoid",
]
