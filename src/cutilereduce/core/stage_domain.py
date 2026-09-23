from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum

from . import symbolic as expr

from .axis import Axis, AxisId, Axes, axis_id
from .utilities import ceil_div, forward, prod, TupleSet
from .measure import Measure, MeasureRef


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
    extent: int | expr.Expr
    tile: int | expr.Expr

    @classmethod
    def make(
        cls,
        axis: Axis,
        role: AxisRole,
        *,
        extent: int | expr.Expr | None = None,
        tile: int | expr.Expr | None = None,
        namespace: str | None = None,
    ) -> ComputeAxis:
        extent = extent if extent is not None else expr.Symbol(axis.name)
        if tile is None:
            tile = (
                extent
                if role == AxisRole.Inner
                else expr.Symbol(f"{namespace or 'stage'}_{axis.name}_tile")
            )
        return cls(axis=axis, role=role, extent=expr.sympify(extent), tile=expr.sympify(tile))

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
        """Exact compatibility helper for concrete lowering."""
        if not self.resolved:
            raise ValueError("num_tiles requires concrete extent and tile")
        return ceil_div(self.extent, self.tile)



@dataclass(frozen=True)
class ProgramAxis:
    axis: Axis
    source: AxisId
    programs: int | expr.Expr

    @classmethod
    def make(
        cls,
        axis: Axis,
        source: AxisId,
        programs: int | expr.Expr,
    ) -> ProgramAxis:
        return cls(axis=axis, source=source, programs=expr.sympify(programs))

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
                raise ValueError(
                    f"multiple program axes for source {source}: {matches}"
                )


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
        program_axes = ProgramAxes(
            values=tuple(
                ProgramAxis(
                    axis=a.axis,
                    source=a.id,
                    programs=cls.measure_symbol(name, a, Measure.num_tiles),
                )
                for a in axes.outer
            )
        )
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

    def resolve_storage(
        self, axes: Iterable[Axis | AxisId]
    ) -> tuple[ComputeAxis | ProgramAxis, ...]:
        return tuple(self.get_storage(axis) for axis in axes)

    @property
    def resolved(self):
        return self.compute_axes.resolved and self.program_axes.resolved

    @property
    def parameters(self):
        """Symbolic extent, tile, and program parameters owned by this domain."""
        values = {}
        for axis in self.compute_axes:
            for value in (axis.extent, axis.tile):
                for symbol in getattr(value, "free_symbols", ()):
                    values[str(symbol)] = symbol
        for axis in self.program_axes:
            for symbol in getattr(axis.programs, "free_symbols", ()):
                values[str(symbol)] = symbol
        return values

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
        return self.compute_axes.subset(
            lambda a: (
                type(self.measure(a, Measure.span_tiles)) is int
                and self.measure(a, Measure.span_tiles) > 1
            )
        )

    @property
    def fold_axis(self):
        (axis,) = self.fold_axes
        return axis

    @property
    def scan_axis(self):
        (axis,) = self.scan_axes
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
        return self.measure(axis, Measure.span_tiles)

    def max_span(self, axis: Axis | AxisId):
        return self.measure(axis, Measure.span)

    def measure_ref(self, axis: Axis | AxisId, measure: Measure) -> MeasureRef:
        """Return a typed reference without exposing expression naming rules."""
        return MeasureRef(axis_id(axis), measure)

    @staticmethod
    def measure_symbol(
        stage_name: str,
        axis: Axis | AxisId,
        measure: Measure,
        *,
        extent=None,
        tile=None,
    ):
        if measure is Measure.num_tiles and type(extent) is int and type(tile) is int:
            return ceil_div(extent, tile)
        return expr.Symbol(f"{stage_name}_{axis_id(axis).name}_{measure.value}")

    def _concrete(self, value, substitutions):
        if isinstance(value, expr.Expr):
            value = value.subs(substitutions)
        return value if type(value) is int else None

    def measure(self, axis: Axis | AxisId, measure: Measure):
        """Return the symbolic or concrete expression for a stage measure."""
        if measure is Measure.extent:
            return self.get(axis).extent
        if measure is Measure.tile:
            return self.get(axis).tile
        if measure is Measure.programs:
            return self.programs_for(axis)
        compute_axis = self.get(axis)
        if measure is Measure.num_tiles:
            extent = self._concrete(compute_axis.extent, {})
            tile = self._concrete(compute_axis.tile, {})
            if extent is not None and tile is not None:
                return ceil_div(extent, tile)
            return self.measure_symbol(self.name, compute_axis, measure)
        if measure is Measure.tiles_per_program:
            num_tiles = self.measure(compute_axis, Measure.num_tiles)
            programs = self.programs_for(compute_axis.id)
            if type(num_tiles) is int and type(programs) is int:
                return ceil_div(num_tiles, programs)
            return self.measure_symbol(self.name, compute_axis, measure)
        if measure is Measure.span_tiles:
            return self.measure_symbol(self.name, compute_axis, measure) \
                if not isinstance(self.measure(axis, Measure.tiles_per_program), int) \
                else self.measure(axis, Measure.tiles_per_program)
        if measure is Measure.span:
            span_tiles = self.measure(axis, Measure.span_tiles)
            tile = compute_axis.tile
            if type(span_tiles) is int and type(tile) is int:
                return span_tiles * tile
            return self.measure_symbol(self.name, compute_axis, measure)
        raise ValueError(measure)

    def _measure_name(self, axis: ComputeAxis, measure: Measure) -> str:
        return f"{self.name}_{axis.name}_{measure.value}"

    def derived_measure_substitutions(self, substitutions=None):
        """Resolve named discrete measures whose inputs are concrete."""
        substitutions = dict(substitutions or {})
        result = {}
        for axis in self.compute_axes:
            extent = self._concrete(axis.extent, substitutions)
            tile = self._concrete(axis.tile, substitutions)
            if extent is None or tile is None:
                continue
            num_tiles = ceil_div(extent, tile)
            num_tiles_symbol = self.measure_symbol(self.name, axis, Measure.num_tiles)
            if isinstance(num_tiles_symbol, expr.Expr):
                result[num_tiles_symbol] = num_tiles
                substitutions[num_tiles_symbol] = num_tiles

            programs = self.programs_for(axis.id)
            programs = self._concrete(programs, substitutions)
            if programs is None:
                continue
            tiles_per_program = ceil_div(num_tiles, programs)
            for measure, value in (
                (Measure.tiles_per_program, tiles_per_program),
                (Measure.span_tiles, tiles_per_program),
                (Measure.span, tiles_per_program * tile),
            ):
                symbol = self.measure_symbol(self.name, axis, measure)
                if isinstance(symbol, expr.Expr):
                    result[symbol] = value
                    substitutions[symbol] = value
        return result

    def measure_constraints(self, axis: Axis | AxisId):
        """Return lower-bound relationships for relaxed derived measures."""
        extent = self.measure(axis, Measure.extent)
        tile = self.measure(axis, Measure.tile)
        programs = self.measure(axis, Measure.programs)
        num_tiles = self.measure(axis, Measure.num_tiles)
        tiles_per_program = self.measure(axis, Measure.tiles_per_program)
        return (
            num_tiles * tile >= extent,
            tiles_per_program * programs >= num_tiles,
            num_tiles >= 1,
            tiles_per_program >= 1,
        )

    def resolve_measure(self, axis: Axis | AxisId, measure: Measure):
        """Resolve a derived measure using exact discrete ceil-div semantics."""
        value = self.measure(axis, measure)
        if isinstance(value, expr.Expr):
            resolved = value.subs(self.derived_measure_substitutions())
            if isinstance(resolved, expr.Expr):
                raise ValueError(f"measure {measure} is not concrete for {axis}")
            return resolved
        return value

    @property
    def task_grid(self):
        return tuple(a.programs for a in self.program_axes)

    @property
    def tasks(self):
        return prod(self.task_grid)
