from __future__ import annotations
from dataclasses import dataclass, field, fields
from functools import cached_property
from math import prod

import sympy

from .base import *

class Dim(str):
    __slots__ = ('_grid',)

    def __new__(cls, name: str, grid: BoundGrid = None):
        obj = super().__new__(cls, name)
        obj._grid = grid
        return obj

    @property
    def grid(self) -> BoundGrid:
        assert self._grid is not None
        """Read-only access to the grid."""
        return self._grid

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
    def tile_exp(self):
        match self.outer:
            case True: return self.tile_var
            case False: return self.total_var

    def __repr__(self):
        return f"{self.__class__.__name__}({super().__repr__()}, grid={self.grid})"

    def __str__(self):
        return f"{super().__str__()}"

@dataclass(frozen=True)
class Dims(TupleSet[str]):
    def bind(self, grid: BoundGrid) -> BoundDims:
        return BoundDims(self.tmap(lambda name: Dim(name, grid=grid)))

@dataclass(frozen=True)
class BoundDims(TupleSet[Dim]):

    @property
    def dims(self) -> tuple[Dim]:
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

    def __str__(self):
        return str(self.tmap(str))

@dataclass(frozen=True)
class Grid:
    input: Dims
    output: Dims
    batch: Dims
    fold: Dims

    @classmethod
    def make(cls, 
             input: dict[str, Buffer], 
             output: dict[str, Buffer],
             batch: Dims,
             fold: Dims):
        input = Dims.union(*(v.spec for v in input.values()))
        output = Dims.union(*(v.spec for v in output.values()))
        return BoundGrid(cls(input, output, batch, fold))

@dataclass(frozen=True)
class BoundGrid:
    grid: Grid

    def __getattr__(self, name):
        if name in field_names(self.grid):
            return getattr(self.grid, name).bind(self)
        else:
            raise AttributeError(f'name {name} not in {field_names(self.grid)}')

    def check(self, simple=True):
        if simple:
            assert len(self.fold) == 1, f'{self.fold!s}'
        assert self.dims.is_superset(self.batch, self.fold)
        assert self.output.is_superset(self.batch)
        assert self.fold.is_disjoint(self.batch, self.output)

    @cached_property
    def dim_map(self):
        return {x: x for x in self.dims}

    @cached_property
    def outer(self) -> BoundDims:
        return self.batch | self.fold

    @cached_property
    def dims(self) -> BoundDims:
        return self.input | self.output

    @cached_property
    def inner(self) -> BoundDims:
        return self.dims - self.outer
