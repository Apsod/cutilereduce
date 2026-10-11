"""SSA dataflow annotated with independently schedulable algebra components."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Any

from cutilereduce.experimental.algebra import (
    FoldAlgebra,
    ProductAlgebra,
    SemidirectProductAlgebra,
    value_leaves,
)
from cutilereduce.experimental.ir.core import (
    TensorRegion,
    TensorRegionBuilder,
    TensorSignature,
)
from cutilereduce.experimental.compiler.fold import FoldFactor
from cutilereduce.experimental.compiler.operator import FoldOperator


type ComponentPath = tuple[str, ...]


class ComponentRelation(Enum):
    PRIMITIVE = "primitive"
    PRODUCT = "product"
    SEMIDIRECT_ACTOR = "semidirect_actor"
    SEMIDIRECT_PAYLOAD = "semidirect_payload"


class SharedProducerPolicy(Enum):
    RECOMPUTE = "recompute"
    MATERIALIZE = "materialize"


@dataclass(frozen=True)
class AlgebraComponent:
    path: ComponentPath
    relation: ComponentRelation
    dependencies: tuple[ComponentPath, ...] = ()


@dataclass(frozen=True)
class SharedProducer:
    node_id: int
    consumers: tuple[ComponentPath, ...]


@dataclass(frozen=True)
class FusionGroup:
    name: str
    components: tuple[ComponentPath, ...]


@dataclass(frozen=True)
class AlgebraicComponentGraph:
    factor_map: TensorRegion
    backward: TensorRegion | None
    components: tuple[AlgebraComponent, ...]
    shared_producers: tuple[SharedProducer, ...]
    fusion_groups: tuple[FusionGroup, ...]
    shared_producer_policy: SharedProducerPolicy | None = None

    @property
    def component_paths(self) -> tuple[ComponentPath, ...]:
        return tuple(component.path for component in self.components)


def _factor_map_region(operator: FoldOperator) -> TensorRegion:
    ctx = TensorRegionBuilder(operator.spec.axes)
    inputs = {
        name: ctx.expect(name, TensorSignature(spec.axes, spec.dtype))
        for name, spec in operator.spec.inputs.items()
    }
    mapped = operator.maps.to_factor(ctx, inputs)
    factor = mapped.value if isinstance(mapped, FoldFactor) else mapped
    operator.algebra.semantic.validate(factor, label="map result")
    for path, value in value_leaves(factor):
        ctx.yield_value(f"factor.{'.'.join(path)}", value)
    return ctx.finish_region()


def _components(
    algebra: FoldAlgebra[Any, Any, Any],
    path: ComponentPath = (),
    dependencies: tuple[ComponentPath, ...] = (),
) -> tuple[AlgebraComponent, ...]:
    if isinstance(algebra, ProductAlgebra):
        result = []
        for name, component in algebra.components.items():
            child = (*path, name)
            nested = _components(component, child, dependencies)
            result.extend(
                replace(value, relation=ComponentRelation.PRODUCT)
                if value.relation is ComponentRelation.PRIMITIVE
                else value
                for value in nested
            )
        return tuple(result)
    if isinstance(algebra, SemidirectProductAlgebra):
        acting_path = (*path, "acting")
        acting = _components(algebra.acting, acting_path, dependencies)
        acting_paths = tuple(component.path for component in acting)
        acting = tuple(
            replace(component, relation=ComponentRelation.SEMIDIRECT_ACTOR)
            if component.relation is ComponentRelation.PRIMITIVE
            else component
            for component in acting
        )
        payload = _components(
            algebra.payload,
            (*path, "payload"),
            (*dependencies, *acting_paths),
        )
        payload = tuple(
            replace(component, relation=ComponentRelation.SEMIDIRECT_PAYLOAD)
            if component.relation is ComponentRelation.PRIMITIVE
            else component
            for component in payload
        )
        return (*acting, *payload)
    return (AlgebraComponent(path, ComponentRelation.PRIMITIVE, dependencies),)


def _ancestors(region: TensorRegion, roots: tuple[int, ...]) -> set[int]:
    result: set[int] = set()
    pending = list(roots)
    while pending:
        node_id = pending.pop()
        if node_id in result:
            continue
        result.add(node_id)
        pending.extend(region.nodes[node_id].inputs)
    return result


def _shared_producers(
    region: TensorRegion, components: tuple[AlgebraComponent, ...]
) -> tuple[SharedProducer, ...]:
    consumers: dict[int, set[ComponentPath]] = {}
    for component in components:
        prefix = "factor"
        if component.path:
            prefix += "." + ".".join(component.path)
        prefix += "."
        roots = tuple(
            node_id for name, node_id in region.results if name.startswith(prefix)
        )
        for node_id in _ancestors(region, roots):
            node = region.nodes[node_id]
            if node.op in {"argument", "constant", "full", "zero"}:
                continue
            consumers.setdefault(node_id, set()).add(component.path)
    return tuple(
        SharedProducer(
            node_id,
            tuple(
                path
                for path in (component.path for component in components)
                if path in paths
            ),
        )
        for node_id, paths in sorted(consumers.items())
        if len(paths) > 1
    )


def build_algebraic_graph(operator: FoldOperator) -> AlgebraicComponentGraph:
    """Build maximal fusion while retaining component and SSA dependencies."""
    factor_map = _factor_map_region(operator)
    components = _components(operator.algebra)
    problem = operator.lower(backward=True)
    backward = (
        problem.map_finalize_map_vjp_accumulate
        if problem.requires_prefix
        else problem.map_finalize_map_vjp
    )
    return AlgebraicComponentGraph(
        factor_map,
        backward,
        components,
        _shared_producers(factor_map, components),
        (FusionGroup("fused", tuple(value.path for value in components)),),
    )


def split_algebraic_graph(
    graph: AlgebraicComponentGraph,
    groups: tuple[FusionGroup, ...],
    *,
    shared_producer_policy: SharedProducerPolicy | None = None,
) -> AlgebraicComponentGraph:
    """Split components, requiring an explicit policy for cross-group sharing."""
    flattened = tuple(path for group in groups for path in group.components)
    if len(set(flattened)) != len(flattened) or set(flattened) != set(
        graph.component_paths
    ):
        raise ValueError("fusion groups must partition every component exactly once")
    owner = {
        component: group.name for group in groups for component in group.components
    }
    crosses_groups = any(
        len({owner[path] for path in producer.consumers}) > 1
        for producer in graph.shared_producers
    )
    if crosses_groups and shared_producer_policy is None:
        raise ValueError(
            "splitting a shared producer requires recompute or materialize"
        )
    return replace(
        graph,
        fusion_groups=groups,
        shared_producer_policy=shared_producer_policy,
    )


def format_algebraic_graph(graph: AlgebraicComponentGraph) -> str:
    lines = ["algebraic_graph {"]
    for component in graph.components:
        path = ".".join(component.path) or "<root>"
        dependencies = ", ".join(".".join(value) for value in component.dependencies)
        suffix = f" depends=[{dependencies}]" if dependencies else ""
        lines.append(f"  component {path} [{component.relation.value}]{suffix}")
    for producer in graph.shared_producers:
        consumers = ", ".join(".".join(path) for path in producer.consumers)
        lines.append(f"  shared %{producer.node_id} -> [{consumers}]")
    for group in graph.fusion_groups:
        values = ", ".join(".".join(path) for path in group.components)
        lines.append(f"  fusion {group.name} [{values}]")
    if graph.shared_producer_policy is not None:
        lines.append(f"  shared_policy {graph.shared_producer_policy.value}")
    lines.append("}")
    return "\n".join(lines)


__all__ = [
    "AlgebraComponent",
    "AlgebraicComponentGraph",
    "ComponentPath",
    "ComponentRelation",
    "FusionGroup",
    "SharedProducer",
    "SharedProducerPolicy",
    "build_algebraic_graph",
    "format_algebraic_graph",
    "split_algebraic_graph",
]
