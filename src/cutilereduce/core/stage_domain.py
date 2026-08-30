from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum

import sympy

from .axis import Axis, AxisId, Axes, axis_id
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
class ComputeAxis:
    axis: Axis
    role: AxisRole
    extent: int | sympy.Expr
    tile: int | sympy.Expr

    @property
    def id(self) -> AxisId:
        return self.axis.id

    @property
    def name(self) -> str:
        return self.axis.name

    @property
    def resolved(self) -> bool:
        return all(type(x) is int for x in (self.extent, self.tile))

    @property
    def num_tiles(self):
        return ceil_div(self.extent, self.tile)


@dataclass(frozen=True)
class ProgramAxis:
    axis: Axis
    source: AxisId
    programs: int | sympy.Expr

    @property
    def id(self) -> AxisId:
        return self.axis.id

    @property
    def name(self) -> str:
        return self.axis.name

    @property
    def extent(self):
        return self.programs

    @property
    def resolved(self) -> bool:
        return type(self.programs) is int


@dataclass(frozen=True)
class ComputeAxes(TupleSet[ComputeAxis]):
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
    def resolved(self):
        return all(x.resolved for x in self)


@dataclass(frozen=True)
class ProgramAxes(TupleSet[ProgramAxis]):
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
    def resolved(self):
        return all(x.resolved for x in self)

    def for_source(self, axis: Axis | AxisId) -> ProgramAxis | None:
        source = axis_id(axis)
        matches = tuple(a for a in self if a.source == source)
        match matches:
            case (axis,):
                return axis
            case ():
                return None
            case _:
                raise ValueError(f"multiple program axes for source {source}: {matches}")


StageAxis = ComputeAxis
StageAxes = ComputeAxes


@dataclass(frozen=True)
class StageDomain:
    name: str
    compute_axes: ComputeAxes
    program_axes: ProgramAxes
    loop: AxisId | None = None

    @classmethod
    def make(cls, name: str, axes: ComputeAxes) -> StageDomain:
        program_axes = ProgramAxes(values=tuple(
            ProgramAxis(axis=a.axis, source=a.id, programs=a.num_tiles)
            for a in axes.outer
        ))
        return cls(name=name, compute_axes=axes, program_axes=program_axes)

    @property
    def axes(self) -> ComputeAxes:
        return self.compute_axes

    def get(self, axis: Axis | AxisId) -> ComputeAxis:
        return self.compute_axes.get(axis)

    def get_storage(self, axis: Axis | AxisId) -> ComputeAxis | ProgramAxis:
        id = axis_id(axis)
        if id in self.compute_axes:
            return self.compute_axes.get(id)
        return self.program_axes.get(id)

    def index(self, axis: Axis | AxisId) -> int:
        return self.compute_axes.index(axis)

    def storage_index(self, axis: Axis | AxisId) -> int:
        return self.storage_axes.index(axis)

    def resolve(self, axes: Iterable[Axis | AxisId]) -> ComputeAxes:
        return ComputeAxes(values=tuple(self.get(axis) for axis in axes))

    def resolve_storage(self, axes: Iterable[Axis | AxisId]) -> tuple[ComputeAxis | ProgramAxis, ...]:
        return tuple(self.get_storage(axis) for axis in axes)

    @property
    def resolved(self):
        return self.compute_axes.resolved and self.program_axes.resolved

    outer_axes = forward("compute_axes", "outer")
    inner_axes = forward("compute_axes", "inner")
    fold_axes = forward("compute_axes", "fold")
    scan_axes = forward("compute_axes", "scan")

    @property
    def storage_axes(self) -> tuple[ComputeAxis | ProgramAxis, ...]:
        return (*self.compute_axes, *self.program_axes)

    @property
    def loop_axes(self) -> ComputeAxes:
        if self.loop is not None:
            return self.resolve((self.loop,))
        return self.compute_axes.subset(lambda a: self.max_span_tiles(a) > 1)

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
                raise ValueError(f"multiple loop axes: {self.loop_axes}")

    def program_axis_for(self, axis: Axis | AxisId) -> ProgramAxis | None:
        return self.program_axes.for_source(axis)

    def programs_for(self, axis: Axis | AxisId):
        program_axis = self.program_axis_for(axis)
        return 1 if program_axis is None else program_axis.programs

    def max_span_tiles(self, axis: Axis | AxisId):
        compute_axis = self.get(axis)
        return ceil_div(compute_axis.num_tiles, self.programs_for(compute_axis))

    def max_span(self, axis: Axis | AxisId):
        compute_axis = self.get(axis)
        return self.max_span_tiles(compute_axis) * compute_axis.tile

    @property
    def task_grid(self):
        return tuple(a.programs for a in self.program_axes)

    @property
    def tasks(self):
        return prod(self.task_grid)
