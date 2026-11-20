"""Backend-independent facts derived from named-axis tensor regions."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
from typing import Any

from cutilereduce.core import symbolic as sym
from .core import ContractionAxes, ContractionPrecision, TensorRegion
from .cost import DTYPE_BYTES, RegionCost, analyze_region, elements


def axis_extents(axes, values=None):
    return {axis: sym.Symbol(axis) for axis in axes} | dict(values or {})


@dataclass(frozen=True)
class ValueFacts:
    node_id: int
    axes: tuple[str, ...]
    dtype: str
    sources: tuple[str, ...] = ()

    def numel(self, extents: Mapping[str, Any] | None = None):
        return elements(self.axes, axis_extents(self.axes, extents))

    def nbytes(self, extents: Mapping[str, Any] | None = None):
        return DTYPE_BYTES[self.dtype] * self.numel(extents)

    def depends_on(self, axes):
        """Named-axis dependence, not a claim about numeric variation."""
        return bool(set(self.axes) & set(axes))


@dataclass(frozen=True)
class ContractionFacts:
    node_id: int
    operands: tuple[int, int]
    axes: ContractionAxes
    dtype: str
    precision: ContractionPrecision | None = None
    accumulated: bool = False

    def dimensions(self, extents=None):
        """Logical B/M/N/K cardinalities; no physical flattening is prescribed."""
        all_axes = (*self.axes.batch, *self.axes.lhs_free, *self.axes.rhs_free, *self.axes.reduce)
        sizes = axis_extents(all_axes, extents)
        return dict(zip(("B", "M", "N", "K"),
                        (elements(axes, sizes) for axes in
                         (self.axes.batch, self.axes.lhs_free, self.axes.rhs_free, self.axes.reduce)), strict=True))

    def flops(self, extents=None):
        dimensions = self.dimensions(extents)
        return 2 * dimensions["B"] * dimensions["M"] * dimensions["N"] * dimensions["K"]


@dataclass(frozen=True)
class RegionFacts:
    axes: tuple[str, ...]
    values: tuple[ValueFacts, ...]
    arguments: tuple[tuple[str, ValueFacts], ...]
    results: tuple[tuple[str, ValueFacts], ...]
    contractions: tuple[ContractionFacts, ...]
    cost: RegionCost

    def argument(self, name):
        return dict(self.arguments)[name]

    def result(self, name):
        return dict(self.results)[name]


def analyze_region_facts(region: TensorRegion) -> RegionFacts:
    """Infer dependencies and contraction roles once, before target selection.

    Sources are transitive formal-argument dependencies. Axes retain the IR's
    declared shape/dependence, including logical indices and lazy broadcasting.
    This neither materializes values nor chooses layouts or execution scopes.
    """
    argument_names = {node_id: name for name, node_id in region.arguments}
    source_sets: dict[int, set[str]] = {}
    values = []
    contractions = []
    for node in region.nodes:
        sources = ({argument_names[node.id]} if node.id in argument_names else
                   set().union(*(source_sets[i] for i in node.inputs)))
        source_sets[node.id] = sources
        values.append(ValueFacts(node.id, node.axes, node.dtype,
                                 tuple(name for name, _ in region.arguments if name in sources)))
        if node.op in {"contract", "contract_accumulate"}:
            partition = node.attribute("contraction_axes")
            if not isinstance(partition, ContractionAxes):
                raise TypeError("contraction has invalid named-axis metadata")
            contractions.append(ContractionFacts(
                node.id, (node.inputs[0], node.inputs[1]), partition, node.dtype,
                dict(node.attributes).get("precision"), node.op == "contract_accumulate",
            ))
    return RegionFacts(region.axes, tuple(values),
                       tuple((name, values[i]) for name, i in region.arguments),
                       tuple((name, values[i]) for name, i in region.results),
                       tuple(contractions), analyze_region(region))
