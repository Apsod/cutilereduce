from __future__ import annotations
from dataclasses import dataclass, field, fields
from functools import cached_property
from math import prod, ceil
from fractions import Fraction

import sympy

from .config import Config
from .base import *

import cuda.tile as ct

D = TypeVar('D')

class Dim(str):
    __slots__ = ('_grid',)

    def __new__(cls, name: str, grid: BoundGrid):
        obj = super().__new__(cls, name)
        obj._grid = grid
        return obj

    @property
    def grid(self) -> BoundGrid:
        assert self._grid is not None
        """Read-only access to the grid."""
        return self._grid

    @property
    def grid_outer_index(self):
        assert self.outer
        return self.grid.outer.index(self)

    @property
    def grid_index(self):
        return self.grid.dims.index(self)

    @property
    def outer(self):
        return self in self.grid.outer

    @property
    def tiled(self):
        return self.outer

    @property
    def inner(self):
        return not self.tiled

    @property
    def batch(self):
        return self in self.grid.batch

    @property
    def fold(self):
        return self in self.grid.fold

    @property
    def tile_var(self):
        return sympy.Symbol(f'({self!s}.tile)')

    @property
    def total_var(self):
        return sympy.Symbol(f'({self!s}.total)')

    @property
    def group_var(self):
        return sympy.Symbol(f'({self!s}.group)')

    @property
    def span_exp(self):
        match self.outer:
            case True: return self.tile_var * self.group_var
            case False: return self.total_var

    @property
    def num_tiles_relaxed(self):
        return self.total_var / self.tile_var

    @property
    def num_tiles(self):
        return ceil(Fraction(self.total_var, self.tile_var))

    @property
    def tile_exp(self):
        match self.outer:
            case True: return self.tile_var
            case False: return self.total_var

    def __repr__(self):
        return f"{self.__class__.__name__}({super().__repr__()}, grid={self.grid})"

    def __str__(self):
        return f"{super().__str__()}"

    def bind(self, config: Config) -> ConcreteDim:
        return ConcreteDim(self, self._grid, config)

class ConcreteDim(Dim):
    __slots__ = ('_grid', '_config')

    def __new__(cls, name: str, grid: BoundGrid, config: Config):
        obj = super().__new__(cls, name, grid)
        obj._config = config
        return obj

    @property
    def tile_var(self):
        return self._config.tiling[self]

    @property
    def total_var(self):
        return self._config.total[self]

    @property
    def group_var(self):
        return self._config.get_grouping(self)

    @property
    def grouped(self):
        return self == self._config.group_dim


@dataclass(frozen=True)
class Dims(TupleSet[str]):
    def bind(self, grid: BoundGrid) -> BoundDims:
        return BoundDims(self.tmap(lambda name: Dim(name, grid=grid)))

    def concretize(self, grid: BoundGrid, config: Config) -> ConcreteDims:
        return ConcreteDims(self.tmap(lambda name: ConcreteDim(name, grid=grid, config=config)))

@dataclass(frozen=True)
class BaseDims[D: Dim](TupleSet[D]):
    @property
    def dims(self) -> tuple[D, ...]:
        return self.value

    @property
    def dimset(self):
        return self.set

    @property
    def outer(self):
        return self.subset(lambda x: x.outer)

    @property
    def inner(self):
        return self.subset(lambda x: x.inner)

    @property
    def batch(self):
        return self.subset(lambda x: x.batch)

    @property
    def fold(self):
        return self.subset(lambda x: x.fold)

    @property
    def tiled(self):
        return self.outer

    @property
    def tile_prod(self):
        return prod(x.tile_exp for x in self)

    @property
    def total_prod(self):
        return prod(x.total_var for x in self)

    @property
    def span_prod(self):
        return prod(x.span_exp for x in self)
    
    @property
    def base(self) -> Dims:
        return Dims(self.tmap(str))

    def __str__(self):
        return str(self.tmap(str))

@dataclass(frozen=True)
class BoundDims(BaseDims[Dim]):
    def concretize(self, config: Config) -> ConcreteDims:
        return ConcreteDims(self.tmap(lambda d: d.bind(config)))

@dataclass(frozen=True)
class ConcreteDims(BaseDims[ConcreteDim]):
    pass

@dataclass(frozen=True)
class Grid:
    input: Dims
    output: Dims
    batch: Dims
    fold: Dims

    @staticmethod
    def make(input: dict[str, Buffer], 
             output: dict[str, Buffer],
             batch: Dims,
             fold: Dims,
             config: Config = None,
             ):
        if config is None:
            input = Dims.union(*(v.spec for v in input.values()))
            output = Dims.union(*(v.spec for v in output.values()))
            return BoundGrid(grid=Grid(input, output, batch, fold))
        else:
            input = Dims.union(*(v.spec for v in input.values()))
            output = Dims.union(*(v.spec for v in output.values()))
            return ConcreteGrid(grid=Grid(input, output, batch, fold), config=config)


@dataclass(frozen=True, kw_only=True)
class BaseGrid[D: Dim]:
    grid: Grid
    CTYPE: ClassVar = None

    def __getattr__(self, name: str) -> BaseDims[D]:
        if name in field_names(self.grid):
            return self.bind_dims(getattr(self.grid, name))
        else:
            raise AttributeError(f'name {name} not in {field_names(self.grid)}')

    @cached_property
    def dim_map(self) -> dict[D, D]:
        return {x: x for x in self.dims}

    @cached_property
    def outer(self) -> BaseDims[D]:
        return self.batch | self.fold

    @cached_property
    def dims(self) -> BaseDims[D]:
        return self.input | self.output

    @cached_property
    def inner(self) -> BaseDims[D]:
        return self.dims - self.outer

    def check(self, simple=True) -> None:
        if simple:
            assert len(self.fold) == 1, f'{self.fold!s}'
        assert self.dims.is_superset(self.batch, self.fold)
        assert self.output.is_superset(self.batch)
        assert self.fold.is_disjoint(self.batch, self.output)

    def bind_dims(self, dims : Dims) -> BaseDims[D]:
        raise NotImplementedError()

    @property
    def base(self) -> Grid:
        return self.grid


@dataclass(frozen=True, kw_only=True)
class BoundGrid(BaseGrid[Dim]):
    CTYPE: ClassVar = BoundDims

    def bind_dims(self, dims : Dims) -> BoundDims:
        return dims.bind(self)

    def concretize(self, config: Config) -> ConcreteGrid:
        return ConcreteGrid(grid=self.grid, config=config)

@dataclass(frozen=True, kw_only=True)
class ConcreteGrid(BaseGrid[ConcreteDim]):
    config: Config
    CTYPE: ClassVar = ConcreteDims

    def bind_dims(self, dims: Dims) -> ConcreteDims:
        return dims.concretize(self, self.config)
    
    @property
    def group_dim(self) -> ConcreteDim:
        return self.dims.get(self.config.group_dim)
