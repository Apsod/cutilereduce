"""General contracts for carrier-resolved fold algebras."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
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


class Monoid[ValueT: AlgebraValue]:
    """One structured monoid and its tensor-IR lowering operations."""

    structure: AlgebraStructure

    def _unsupported(self, operation: str) -> Never:
        raise NotImplementedError(
            f"{type(self).__name__} does not implement {operation} lowering"
        )

    def validate(self, value: AlgebraValue, *, label: str) -> None:
        self.structure.validate(value, label=label)

    def lower_identity(self, ctx: TensorBuilder) -> ValueT:
        """Construct one identity, carrying only the monoid's inner axes."""
        del ctx
        self._unsupported("identity")

    def lower_combine(
        self,
        ctx: TensorBuilder,
        left: ValueT,
        right: ValueT,
    ) -> ValueT:
        """Combine two values pointwise over untouched axes."""
        del ctx, left, right
        self._unsupported("combine")

    def lower_fold(
        self,
        ctx: TensorBuilder,
        values: ValueT,
        *,
        fold_axes: tuple[str, ...],
    ) -> ValueT:
        """Fold an arbitrary region without prescribing its staging."""
        del ctx, values, fold_axes
        self._unsupported("fold")


class FoldAlgebra[
    SemanticT: AlgebraValue,
    ExecutionT: AlgebraValue,
    AdjointT: AlgebraValue,
]:
    """Relationships among semantic, execution, and adjoint monoids."""

    semantic: Monoid[SemanticT]
    execution: Monoid[ExecutionT]
    adjoint: Monoid[AdjointT]

    def _unsupported(self, operation: str) -> Never:
        raise NotImplementedError(
            f"{type(self).__name__} does not implement {operation} lowering"
        )

    def lower_phi(
        self, ctx: TensorBuilder, semantic: SemanticT
    ) -> ExecutionT:
        """Lift semantic factors into execution carriers, Phi: S -> E."""
        del ctx, semantic
        self._unsupported("Phi")

    def lower_psi(
        self, ctx: TensorBuilder, execution: ExecutionT
    ) -> SemanticT:
        """Project an execution carrier to its semantic value, Psi: E -> S."""
        del ctx, execution
        self._unsupported("Psi")

    def lower_embed(
        self,
        ctx: TensorBuilder,
        execution: ExecutionT,
        semantic_cotangent: SemanticT,
    ) -> AdjointT:
        """Embed a global execution state and semantic cotangent into A."""
        del ctx, execution, semantic_cotangent
        self._unsupported("embed")

    def lower_finalize(
        self,
        ctx: TensorBuilder,
        factor: SemanticT,
        adjoint: AdjointT,
        *,
        prefix: ExecutionT | None = None,
    ) -> SemanticT:
        """Produce a local semantic cotangent, optionally using an E prefix."""
        del ctx, factor, adjoint, prefix
        self._unsupported("finalize")


__all__ = [
    "AlgebraStructure",
    "AlgebraValue",
    "FoldAlgebra",
    "Monoid",
    "CompositeStructure",
    "CompositeValue",
    "StructuredValue",
    "TensorField",
    "TensorStructure",
    "value_leaves",
]
