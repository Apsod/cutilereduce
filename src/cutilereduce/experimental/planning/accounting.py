"""Symbolic domain, storage and ownership accounting independent of a target.

These are logical counts or padded upper bounds, not cache-miss, transaction,
register-allocation or communication-algorithm predictions. A scope names the
boundary being accounted for; it does not select a hardware mechanism.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Number, Real
import math
from typing import Any

from cutilereduce.core import symbolic as sym
from cutilereduce.core.measure import Measure
from cutilereduce.experimental.ir.cost import bind, product
from cutilereduce.experimental.ir.facts import ValueFacts


def ceildiv(extent, divisor):
    if isinstance(extent, int) and isinstance(divisor, int):
        return (extent + divisor - 1) // divisor
    return sym.ceiling(sym.sympify(extent) / divisor)


def _positive(value, name):
    if isinstance(value, Number):
        if (not isinstance(value, Real) or isinstance(value, bool)
                or (isinstance(value, float) and not math.isfinite(value))
                or value <= 0 or int(value) != value):
            raise ValueError(f"{name} must be a positive integer or symbolic expression")
    elif not isinstance(value, sym.Expr):
        raise TypeError(f"{name} must be a positive integer or symbolic expression")


@dataclass(frozen=True)
class ExecutionScope:
    """A target-supplied boundary label, optionally nested in another scope."""

    name: str
    parent: ExecutionScope | None = None

    def __post_init__(self):
        if not self.name:
            raise ValueError("execution scope needs a name")
        if self.parent is not None and self.name in self.parent.path:
            raise ValueError("scope names must be unique along an ancestry path")

    @property
    def path(self):
        return (*(self.parent.path if self.parent else ()), self.name)


@dataclass(frozen=True)
class AxisDomain:
    axis: str
    extent: Any
    tile: Any
    owners: Any = 1

    def __post_init__(self):
        if not self.axis:
            raise ValueError("axis needs a name")
        for name in ("extent", "tile", "owners"):
            _positive(getattr(self, name), name)

    @property
    def num_tiles(self):
        return ceildiv(self.extent, self.tile)

    @property
    def span_tiles(self):
        return ceildiv(self.num_tiles, self.owners)

    @property
    def span(self):
        return self.span_tiles * self.tile

    @property
    def active_owners(self):
        return sym.Min(self.owners, self.num_tiles)

    def measure(self, measure: Measure):
        return {
            Measure.extent: self.extent, Measure.tile: self.tile,
            Measure.programs: self.owners, Measure.num_tiles: self.num_tiles,
            Measure.tiles_per_program: self.span_tiles, Measure.span_tiles: self.span_tiles,
            Measure.span: self.span,
        }[measure]

    def subs(self, values):
        return AxisDomain(self.axis, bind(self.extent, values), bind(self.tile, values), bind(self.owners, values))


@dataclass(frozen=True)
class ContributionOwnership:
    """Required accumulation, deliberately independent of stores/atomics/trees."""

    scope: ExecutionScope
    owner_axes: tuple[str, ...]
    local_axes: tuple[str, ...]
    owners_per_target: Any

    @property
    def requires_reduction(self):
        return self.owners_per_target > 1


def contribution_ownership(value_axes, owner_counts: Mapping[str, Any], *, loop_axes=(), scope=None):
    """Account for values declared to be partial contributions, e.g. cotangents.

    Axis absence alone does not make replicated primal values additive. Their
    ownership/combination semantics must be supplied by the enclosing operation.
    """
    for count in owner_counts.values():
        _positive(count, "owner count")
    missing = tuple(axis for axis, count in owner_counts.items()
                    if axis not in value_axes and not (isinstance(count, Number) and count == 1))
    return ContributionOwnership(
        scope or ExecutionScope("owner"), missing,
        tuple(axis for axis in loop_axes if axis not in value_axes),
        product(owner_counts[axis] for axis in missing),
    )


@dataclass(frozen=True)
class ScheduleDomain:
    scope: ExecutionScope
    axes: tuple[AxisDomain, ...]
    iteration_axes: tuple[str, ...]
    loop_axes: tuple[str, ...] = ()

    def __post_init__(self):
        names = tuple(a.axis for a in self.axes)
        if len(set(names)) != len(names):
            raise ValueError("domain axes must be unique")
        if len(set(self.iteration_axes)) != len(self.iteration_axes) or len(set(self.loop_axes)) != len(self.loop_axes):
            raise ValueError("iteration and loop axes must be unique")
        if set(self.iteration_axes) - set(names) or set(self.loop_axes) - set(self.iteration_axes):
            raise ValueError("loops must be iteration axes and iteration axes must be in the domain")
        if any(a.axis not in self.iteration_axes and not (isinstance(a.owners, Number) and a.owners == 1) for a in self.axes):
            raise ValueError("only declared iteration axes may have multiple owners")

    @classmethod
    def from_maps(cls, extents, *, tiles=None, owners=None, iteration_axes=None, loop_axes=(), scope=None):
        tiles, owners = dict(tiles or {}), dict(owners or {})
        if (set(tiles) | set(owners)) - set(extents):
            raise ValueError("tile and owner maps name an unknown domain axis")
        return cls(scope or ExecutionScope("owner"),
                   tuple(AxisDomain(a, n, tiles.get(a, n), owners.get(a, 1)) for a, n in extents.items()),
                   tuple(extents) if iteration_axes is None else tuple(iteration_axes), tuple(loop_axes))

    def axis(self, name):
        for axis in self.axes:
            if axis.axis == name:
                return axis
        raise KeyError(name)

    def measures(self, measure: Measure):
        return {a.axis: a.measure(measure) for a in self.axes}

    @property
    def owner_count(self):
        return product(self.axis(a).owners for a in self.iteration_axes)

    @property
    def active_owners(self):
        return product(self.axis(a).active_owners for a in self.iteration_axes)

    def subs(self, values):
        return ScheduleDomain(self.scope, tuple(a.subs(values) for a in self.axes), self.iteration_axes, self.loop_axes)

    def to_dict(self):
        def scalar(value):
            if isinstance(value, sym.Expr):
                return str(value)
            return int(value) if int(value) == value else float(value)
        return {"scope": list(self.scope.path), "iteration_axes": list(self.iteration_axes),
                "loop_axes": list(self.loop_axes),
                "axes": {a.axis: {"extent": scalar(a.extent), "tile": scalar(a.tile),
                                   "owners": scalar(a.owners), "num_tiles": scalar(a.num_tiles),
                                   "span_tiles": scalar(a.span_tiles), "span": scalar(a.span)}
                         for a in self.axes}}

    def buffer(self, value: ValueFacts):
        if set(value.axes) - {a.axis for a in self.axes}:
            raise ValueError("value names axes absent from the domain")
        return BufferAccounting(value, self)


@dataclass(frozen=True)
class BufferAccounting:
    value: ValueFacts
    domain: ScheduleDomain

    def nbytes(self, measure: Measure):
        return self.value.nbytes(self.domain.measures(measure))

    @property
    def tile_bytes(self):
        return self.nbytes(Measure.tile)

    @property
    def span_bytes(self):
        """Maximum padded value span owned by one owner."""
        return self.nbytes(Measure.span)

    @property
    def total_bytes(self):
        return self.nbytes(Measure.extent)

    @property
    def streamed(self):
        return self.value.depends_on(self.domain.loop_axes)

    @property
    def persistent(self):
        return not self.streamed

    @property
    def replication_axes(self):
        return tuple(a for a in self.domain.iteration_axes if a not in self.value.axes)

    @property
    def owner_replicas(self):
        return product(self.domain.axis(a).owners for a in self.replication_axes)

    @property
    def ownership(self):
        """Contribution ownership when this value denotes partial results.

        For input/replicated primal values, use owner_replicas: duplication does
        not by itself prescribe a reduction or its algebra.
        """
        return contribution_ownership(self.value.axes,
                                      {a: self.domain.axis(a).owners for a in self.domain.iteration_axes},
                                      loop_axes=self.domain.loop_axes, scope=self.domain.scope)

    @property
    def target_partitions(self):
        return product(self.domain.axis(a).owners for a in self.domain.iteration_axes if a in self.value.axes)

    @property
    def replicated_bytes(self):
        """Additional logical owner replicas, not physical link traffic."""
        return self.total_bytes * sym.SaturateSub(self.owner_replicas, 1)

    @property
    def logical_accessed_bytes(self):
        """Exact full-domain bytes times replication across independent owners."""
        return self.total_bytes * self.owner_replicas

    @property
    def padded_accessed_bytes(self):
        """Tile-loop access bound; loop-invariant values are retained per owner.

        The schedule must account for every serially traversed axis in loop_axes.
        This counts logical accesses, not physical memory/cache/network transfers.
        """
        repeats = product(self.domain.axis(a).span_tiles for a in self.domain.loop_axes if a in self.value.axes)
        return self.tile_bytes * self.domain.owner_count * repeats
