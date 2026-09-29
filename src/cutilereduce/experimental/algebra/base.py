"""General contracts for carrier-resolved fold algebras."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Never

from ..ir import TensorBuilder, TensorValue


@dataclass(frozen=True)
class TensorField:
    """One field of an algebraic structure, excluding lifted outer axes."""

    inner_axes: tuple[str, ...] = ()
    dtype: str = "float32"


@dataclass(frozen=True)
class TensorStructure:
    fields: Mapping[str, TensorField]

    def validate(self, value: AlgebraValue, *, label: str) -> None:
        if not isinstance(value, StructuredValue):
            raise ValueError(f"{label} must be a structured value")
        expected = set(self.fields)
        actual = set(value.fields)
        if actual != expected:
            raise ValueError(
                f"{label} fields must be {sorted(expected)}, got {sorted(actual)}"
            )
        for name, field in self.fields.items():
            missing = set(field.inner_axes) - set(value[name].axes)
            if missing:
                raise ValueError(
                    f"{label}.{name} is missing inner axes {sorted(missing)}"
                )
            if value[name].dtype != field.dtype:
                raise ValueError(
                    f"{label}.{name} must have dtype {field.dtype}, "
                    f"got {value[name].dtype}"
                )


@dataclass(frozen=True)
class StructuredValue:
    """The flat tensor-field record belonging to one algebra element."""

    fields: Mapping[str, TensorValue]

    def __getitem__(self, name: str) -> TensorValue:
        return self.fields[name]


@dataclass(frozen=True)
class CompositeValue:
    """A named recursive composition of independently structured values."""

    components: Mapping[str, AlgebraValue]

    def __getitem__(self, name: str) -> AlgebraValue:
        return self.components[name]


type AlgebraValue = StructuredValue | CompositeValue


@dataclass(frozen=True)
class CompositeStructure:
    """Schemas for a named recursive algebra composition."""

    components: Mapping[str, AlgebraStructure]

    def validate(self, value: AlgebraValue, *, label: str) -> None:
        if not isinstance(value, CompositeValue):
            raise ValueError(f"{label} must be a composite value")
        expected = set(self.components)
        actual = set(value.components)
        if actual != expected:
            raise ValueError(
                f"{label} components must be {sorted(expected)}, "
                f"got {sorted(actual)}"
            )
        for name, structure in self.components.items():
            structure.validate(value[name], label=f"{label}.{name}")


type AlgebraStructure = TensorStructure | CompositeStructure


class FinalizeContext(Enum):
    NONE = "none"
    PREFIX = "prefix"


def value_leaves(
    value: AlgebraValue, prefix: tuple[str, ...] = ()
) -> Iterator[tuple[tuple[str, ...], TensorValue]]:
    """Yield stable field paths and tensor leaves from an algebra value."""
    if isinstance(value, StructuredValue):
        for name, tensor in value.fields.items():
            yield (*prefix, name), tensor
        return
    for name, component in value.components.items():
        yield from value_leaves(component, (*prefix, name))


class PartialSemigroup[ValueT: AlgebraValue]:
    """A possibly partial associative combine operation."""

    structure: AlgebraStructure
    commutative: bool

    def _unsupported(self, operation: str) -> Never:
        raise NotImplementedError(
            f"{type(self).__name__} does not implement {operation} lowering"
        )

    def validate(self, value: AlgebraValue, *, label: str) -> None:
        self.structure.validate(value, label=label)

    def combine(
        self,
        ctx: TensorBuilder,
        left: ValueT,
        right: ValueT,
    ) -> ValueT:
        """Combine two values pointwise over untouched axes."""
        del ctx, left, right
        self._unsupported("combine")


class Monoid[ValueT: AlgebraValue](PartialSemigroup[ValueT]):
    """A partial semigroup with an identity and optional optimized lowerings."""

    def identity(self, ctx: TensorBuilder) -> ValueT:
        """Construct one identity, carrying only the monoid's inner axes."""
        del ctx
        self._unsupported("identity")

    def lower_fold(
        self,
        ctx: TensorBuilder,
        values: ValueT,
        *,
        fold_axes: tuple[str, ...],
    ) -> ValueT | None:
        """Optionally emit an optimized fold; ``None`` selects the fallback."""
        del ctx, values, fold_axes
        return None

    def lower_scan(
        self,
        ctx: TensorBuilder,
        values: ValueT,
        *,
        scan_axis: str,
        initial: ValueT,
    ) -> ScanResult[ValueT] | None:
        """Optionally emit an optimized exclusive scan and final carrier."""
        del ctx, values, scan_axis, initial
        return None


@dataclass(frozen=True)
class ScanResult[ValueT: AlgebraValue]:
    prefixes: ValueT
    final: ValueT


class FoldAlgebra[
    SemanticT: AlgebraValue,
    ExecutionT: AlgebraValue,
    AccumulatorT: AlgebraValue,
]:
    """Semantic/execution monoids and their gradient accumulator relation."""

    semantic: Monoid[SemanticT]
    execution: Monoid[ExecutionT]
    accumulator: PartialSemigroup[AccumulatorT]
    finalize_context: FinalizeContext

    def _unsupported(self, operation: str) -> Never:
        raise NotImplementedError(
            f"{type(self).__name__} does not implement {operation} lowering"
        )

    def to_execution(
        self, ctx: TensorBuilder, semantic: SemanticT
    ) -> ExecutionT:
        """Lift semantic factors into execution carriers, Phi: S -> E."""
        del ctx, semantic
        self._unsupported("Phi")

    def to_semantic(
        self, ctx: TensorBuilder, execution: ExecutionT
    ) -> SemanticT:
        """Project an execution carrier to its semantic value, Psi: E -> S."""
        del ctx, execution
        self._unsupported("Psi")

    def embed(
        self,
        ctx: TensorBuilder,
        execution: ExecutionT,
        semantic_cotangent: SemanticT,
    ) -> AccumulatorT:
        """Embed a global execution state and semantic cotangent into A."""
        del ctx, execution, semantic_cotangent
        self._unsupported("embed")

    def finalize(
        self,
        ctx: TensorBuilder,
        factor: SemanticT,
        accumulator: AccumulatorT,
        *,
        prefix: ExecutionT | None = None,
    ) -> SemanticT:
        """Produce a local semantic cotangent, optionally using an E prefix."""
        del ctx, factor, accumulator, prefix
        self._unsupported("finalize")


__all__ = [
    "AlgebraStructure",
    "AlgebraValue",
    "FoldAlgebra",
    "FinalizeContext",
    "Monoid",
    "PartialSemigroup",
    "ScanResult",
    "CompositeStructure",
    "CompositeValue",
    "StructuredValue",
    "TensorField",
    "TensorStructure",
    "value_leaves",
]
