"""Generic semidirect-product monoids and fold-algebra construction."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .base import (
    AlgebraValue,
    CompositeStructure,
    CompositeValue,
    FinalizeContext,
    FoldAlgebra,
    Monoid,
)
from ..ir import TensorBuilder


type ActionLowering = Callable[
    [TensorBuilder, AlgebraValue, AlgebraValue], AlgebraValue
]


class SemidirectProductMonoid(Monoid[CompositeValue]):
    """A right-oriented semidirect product derived from an action."""

    commutative = False

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

    def identity(self, ctx: TensorBuilder) -> CompositeValue:
        return CompositeValue(
            {
                "acting": self.acting.identity(ctx),
                "payload": self.payload.identity(ctx),
            }
        )

    def combine(
        self, ctx: TensorBuilder, left: CompositeValue, right: CompositeValue
    ) -> CompositeValue:
        self.validate(left, label="left semidirect value")
        self.validate(right, label="right semidirect value")
        acted = self.action(ctx, right["acting"], left["payload"])
        return CompositeValue(
            {
                "acting": self.acting.combine(
                    ctx, left["acting"], right["acting"]
                ),
                "payload": self.payload.combine(
                    ctx, acted, right["payload"]
                ),
            }
        )


class SemidirectProductAlgebra(
    FoldAlgebra[CompositeValue, CompositeValue, CompositeValue]
):
    """Partially derived fold algebra for a semidirect construction.

    Its component relationships derive Phi and Psi. A concrete resolved
    algebra still supplies its accumulator semigroup, embed, and finalize.
    """

    finalize_context = FinalizeContext.PREFIX

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

    def value(
        self, *, acting: AlgebraValue, payload: AlgebraValue
    ) -> CompositeValue:
        result = CompositeValue({"acting": acting, "payload": payload})
        self.semantic.validate(result, label="semantic semidirect product")
        return result

    def to_execution(
        self, ctx: TensorBuilder, semantic: CompositeValue
    ) -> CompositeValue:
        self.semantic.validate(semantic, label="semantic semidirect product")
        return CompositeValue(
            {
                "acting": self.acting.to_execution(ctx, semantic["acting"]),
                "payload": self.payload.to_execution(ctx, semantic["payload"]),
            }
        )

    def to_semantic(
        self, ctx: TensorBuilder, execution: CompositeValue
    ) -> CompositeValue:
        self.execution.validate(execution, label="semidirect carrier")
        return CompositeValue(
            {
                "acting": self.acting.to_semantic(ctx, execution["acting"]),
                "payload": self.payload.to_semantic(ctx, execution["payload"]),
            }
        )


SemidirectProduct = SemidirectProductAlgebra


__all__ = [
    "SemidirectProduct",
    "SemidirectProductAlgebra",
    "SemidirectProductMonoid",
]
