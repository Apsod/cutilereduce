"""Componentwise products of monoids and fold algebras."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .base import CompositeStructure, CompositeValue, FoldAlgebra, Monoid
from ..ir import TensorBuilder


class ProductMonoid(Monoid[CompositeValue]):
    def __init__(self, components: Mapping[str, Monoid[Any]]):
        if not components:
            raise ValueError("a monoid product requires at least one component")
        self.components = dict(components)
        self.structure = CompositeStructure(
            {name: monoid.structure for name, monoid in self.components.items()}
        )

    def lower_identity(self, ctx: TensorBuilder) -> CompositeValue:
        return CompositeValue(
            {
                name: monoid.lower_identity(ctx)
                for name, monoid in self.components.items()
            }
        )

    def lower_combine(
        self, ctx: TensorBuilder, left: CompositeValue, right: CompositeValue
    ) -> CompositeValue:
        self.validate(left, label="left product value")
        self.validate(right, label="right product value")
        return CompositeValue(
            {
                name: monoid.lower_combine(ctx, left[name], right[name])
                for name, monoid in self.components.items()
            }
        )

    def lower_fold(
        self,
        ctx: TensorBuilder,
        values: CompositeValue,
        *,
        fold_axes: tuple[str, ...],
    ) -> CompositeValue:
        self.validate(values, label="product values")
        return CompositeValue(
            {
                name: monoid.lower_fold(
                    ctx, values[name], fold_axes=fold_axes
                )
                for name, monoid in self.components.items()
            }
        )


class ProductAlgebra(
    FoldAlgebra[CompositeValue, CompositeValue, CompositeValue]
):
    """A named product whose relationships delegate componentwise."""

    def __init__(self, components: Mapping[str, FoldAlgebra[Any, Any, Any]]):
        if not components:
            raise ValueError("an algebra product requires at least one component")
        self.components = dict(components)
        self.semantic = ProductMonoid(
            {name: algebra.semantic for name, algebra in self.components.items()}
        )
        self.execution = ProductMonoid(
            {name: algebra.execution for name, algebra in self.components.items()}
        )
        self.adjoint = ProductMonoid(
            {name: algebra.adjoint for name, algebra in self.components.items()}
        )

    def value(self, **components) -> CompositeValue:
        value = CompositeValue(components)
        self.semantic.validate(value, label="semantic product")
        return value

    def lower_phi(
        self, ctx: TensorBuilder, semantic: CompositeValue
    ) -> CompositeValue:
        self.semantic.validate(semantic, label="semantic product")
        return CompositeValue(
            {
                name: algebra.lower_phi(ctx, semantic[name])
                for name, algebra in self.components.items()
            }
        )

    def lower_psi(
        self, ctx: TensorBuilder, execution: CompositeValue
    ) -> CompositeValue:
        self.execution.validate(execution, label="execution product")
        return CompositeValue(
            {
                name: algebra.lower_psi(ctx, execution[name])
                for name, algebra in self.components.items()
            }
        )

    def lower_embed(
        self,
        ctx: TensorBuilder,
        execution: CompositeValue,
        semantic_cotangent: CompositeValue,
    ) -> CompositeValue:
        return CompositeValue(
            {
                name: algebra.lower_embed(
                    ctx, execution[name], semantic_cotangent[name]
                )
                for name, algebra in self.components.items()
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
        return CompositeValue(
            {
                name: algebra.lower_finalize(
                    ctx,
                    factor[name],
                    adjoint[name],
                    prefix=None if prefix is None else prefix[name],
                )
                for name, algebra in self.components.items()
            }
        )


def Product(**components: FoldAlgebra[Any, Any, Any]) -> ProductAlgebra:
    """Construct a named n-ary product fold algebra."""
    return ProductAlgebra(components)


__all__ = ["Product", "ProductAlgebra", "ProductMonoid"]
