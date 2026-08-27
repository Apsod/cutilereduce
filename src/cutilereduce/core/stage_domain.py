from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
from enum import Enum, Flag, auto

import sympy

from .axis import Axis, AxisId, Axes, axis_id
from .buffer import BufferRole
from .utilities import ceil_div, forward, prod, TupleSet

class AxisRole(Enum):
    Batch = "batch"
    Fold = "fold"
    Scan = "scan"
    Inner = "inner"

    @property
    def is_outer(self) -> bool:
        return self != AxisRole.Inner
    
    @property
    def is_inner(self) -> bool:
        return self == AxisRole.Inner
    
    @property
    def is_fold(self) -> bool:
        return self == AxisRole.Fold

    @property
    def is_scan(self) -> bool:
        return self == AxisRole.Scan

@dataclass(frozen=True)
class StageAxis:
    axis: Axis
    role: AxisRole
    extent: int | sympy.Expr
    tile: int | sympy.Expr
    programs: int | sympy.Expr = 1

    @property
    def id(self) -> AxisId:
        return self.axis.id

    @property
    def name(self) -> str:
        return self.axis.name

    @property
    def resolved(self):
        return all(type(x) is int for x in [self.extent, self.tile, self.programs])

    @property
    def num_tiles(self):
        return ceil_div(self.extent, self.tile)

    @property
    def max_span_tiles(self):
        return ceil_div(self.num_tiles, self.programs)

    @property
    def max_span(self):
        return self.max_span_tiles * self.tile

    @property
    def relaxed_span(self):
        return sympy.sympify(self.extent) / sympy.sympify(self.programs)

    @property
    def in_loop(self):
        return self.max_span_tiles > 1

    @property
    def is_partitioned(self):
        return self.programs > 1

@dataclass(frozen=True)
class StageAxes(TupleSet[StageAxis]):
    @staticmethod
    def key(x):
        return axis_id(x)

    @property
    def axes(self) -> Axes:
        return Axes(values=tuple(a.axis for a in self))

    @property
    def ids(self) -> tuple[AxisId, ...]:
        return self.keys

    @property
    def outer(self):
        return self.subset(lambda x: x.role.is_outer)

    @property
    def inner(self):
        return self.subset(lambda x: x.role.is_inner)

    @property
    def fold(self):
        return self.subset(lambda x: x.role.is_fold)

    @property
    def scan(self):
        return self.subset(lambda x: x.role.is_scan)

    @property
    def loop(self):
        return self.subset(lambda x: x.in_loop)

    @property
    def resolved(self):
        return all(x.resolved for x in self)

@dataclass(frozen=True)
class StageDomain:
    name: str
    axes: StageAxes

    def get(self, axis: Axis | AxisId) -> StageAxis:
        return self.axes.get(axis)

    def index(self, axis: Axis | AxisId) -> int:
        return self.axes.index(axis)
    
    resolved = forward('axes', 'resolved')
    outer_axes = forward('axes', 'outer')
    inner_axes = forward('axes', 'inner')
    fold_axes = forward('axes', 'fold')
    scan_axes = forward('axes', 'scan')
    loop_axes = forward('axes', 'loop')

    @property
    def fold_axis(self):
        axis, = self.fold_axes
        return axis

    @property
    def scan_axis(self):
        axis, = self.scan_axes
        return axis

    @property
    def loop_axis(self):
        match tuple(self.loop_axes):
            case (axis,):
                return axis
            case ():
                return None
            case _:
                raise ValueError(f'multiple loop axes: {self.loop_axes}')

    @property
    def task_grid(self):
        return tuple(a.programs for a in self.outer_axes)

    @property
    def tasks(self):
        return prod(self.task_grid)
