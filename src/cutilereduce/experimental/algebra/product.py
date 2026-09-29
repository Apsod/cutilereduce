"""Componentwise products of monoids and fold algebras."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .base import (
    CompositeStructure,
    CompositeValue,
    FinalizeContext,
    FoldAlgebra,
    Monoid,
    PartialSemigroup,
    ScanResult,
)
from ..ir import TensorBuilder


class ProductSemigroup(PartialSemigroup[CompositeValue]):
    def __init__(self, components: Mapping[str, PartialSemigroup[Any]]):
        if not components:
            raise ValueError("a semigroup product requires at least one component")
        self.components = dict(components)
        self.commutative = all(
            monoid.commutative for monoid in self.components.values()
        )
        self.structure = CompositeStructure(
            {name: monoid.structure for name, monoid in self.components.items()}
        )

    def combine(
        self, ctx: TensorBuilder, left: CompositeValue, right: CompositeValue
    ) -> CompositeValue:
        self.validate(left, label="left product value")
        self.validate(right, label="right product value")
        return CompositeValue(
            {
                name: monoid.combine(ctx, left[name], right[name])
                for name, monoid in self.components.items()
            }
        )


class ProductMonoid(ProductSemigroup, Monoid[CompositeValue]):
    def __init__(self, components: Mapping[str, Monoid[Any]]):
        self.monoids = dict(components)
        super().__init__(components)

    def identity(self, ctx: TensorBuilder) -> CompositeValue:
        return CompositeValue(
            {
                name: monoid.identity(ctx)
                for name, monoid in self.monoids.items()
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
        from ..lowering import lower_fold

        return CompositeValue(
            {
                name: lower_fold(
                    ctx, monoid, values[name], fold_axes=fold_axes
                )
                for name, monoid in self.monoids.items()
            }
        )

    def lower_scan(
        self,
        ctx: TensorBuilder,
        values: CompositeValue,
        *,
        scan_axis: str,
        initial: CompositeValue,
    ) -> ScanResult[CompositeValue]:
        from ..lowering import lower_scan

        scans = {
            name: lower_scan(
                ctx,
                monoid,
                values[name],
                scan_axis=scan_axis,
                initial=initial[name],
            )
            for name, monoid in self.monoids.items()
        }
        return ScanResult(
            prefixes=CompositeValue(
                {name: scan.prefixes for name, scan in scans.items()}
            ),
            final=CompositeValue(
                {name: scan.final for name, scan in scans.items()}
            ),
        )


class ProductAlgebra(
    FoldAlgebra[CompositeValue, CompositeValue, CompositeValue]
):
    """A named product whose relationships delegate componentwise."""

    def __init__(self, components: Mapping[str, FoldAlgebra[Any, Any, Any]]):
        if not components:
            raise ValueError("an algebra product requires at least one component")
        self.components = dict(components)
        self.finalize_context = (
            FinalizeContext.PREFIX
            if any(
                algebra.finalize_context is FinalizeContext.PREFIX
                for algebra in self.components.values()
            )
            else FinalizeContext.NONE
        )
        self.semantic = ProductMonoid(
            {name: algebra.semantic for name, algebra in self.components.items()}
        )
        self.execution = ProductMonoid(
            {name: algebra.execution for name, algebra in self.components.items()}
        )
        self.accumulator = ProductSemigroup(
            {
                name: algebra.accumulator
                for name, algebra in self.components.items()
            }
        )

    def value(self, **components) -> CompositeValue:
        value = CompositeValue(components)
        self.semantic.validate(value, label="semantic product")
        return value

    def to_execution(
        self, ctx: TensorBuilder, semantic: CompositeValue
    ) -> CompositeValue:
        self.semantic.validate(semantic, label="semantic product")
        return CompositeValue(
            {
                name: algebra.to_execution(ctx, semantic[name])
                for name, algebra in self.components.items()
            }
        )

    def to_semantic(
        self, ctx: TensorBuilder, execution: CompositeValue
    ) -> CompositeValue:
        self.execution.validate(execution, label="execution product")
        return CompositeValue(
            {
                name: algebra.to_semantic(ctx, execution[name])
                for name, algebra in self.components.items()
            }
        )

    def embed(
        self,
        ctx: TensorBuilder,
        execution: CompositeValue,
        semantic_cotangent: CompositeValue,
    ) -> CompositeValue:
        return CompositeValue(
            {
                name: algebra.embed(
                    ctx, execution[name], semantic_cotangent[name]
                )
                for name, algebra in self.components.items()
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
        return CompositeValue(
            {
                name: algebra.finalize(
                    ctx,
                    factor[name],
                    accumulator[name],
                    prefix=None if prefix is None else prefix[name],
                )
                for name, algebra in self.components.items()
            }
        )


def Product(**components: FoldAlgebra[Any, Any, Any]) -> ProductAlgebra:
    """Construct a named n-ary product fold algebra."""
    return ProductAlgebra(components)


__all__ = [
    "Product",
    "ProductAlgebra",
    "ProductMonoid",
    "ProductSemigroup",
]
