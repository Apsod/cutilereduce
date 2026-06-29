from __future__ import annotations
from dataclasses import dataclass
from sympy import Min, Max
from math import prod

from .base import field_names, Phase
from .grid import BaseGrid, Dims, Dim, ConcreteDim, BoundGrid, ConcreteGrid

@dataclass(frozen=True)
class MatMul:
    b: Dims
    m: Dims  
    n: Dims
    k: Dims

    @classmethod
    def make(cls, string: str):
        return cls(*map(Dims.parse, string.split(':')))


@dataclass(frozen=True)
class BaseMatMul[D: Dim]:
    spec: MatMul
    grid: BaseGrid[D]

    def __getattr__(self, name):
        if name in field_names(self.spec):
            return self.grid.bind_dims(getattr(self.spec, name))
        else:
            raise AttributeError(f'name {name} not in {field_names(self.spec)}')

    @property
    def A(self):
        return self.b | self.m | self.k

    @property
    def B(self):
        return self.b | self.n | self.k

    @property
    def C(self):
        return self.b | self.m | self.n

    @property
    def all(self):
        return self.b | self.m | self.n | self.k

    @property
    def tile_work(self):
        return 2 * self.all.tile_prod

    @property
    def total_work(self):
        return 2 * self.all.total_prod

    @property
    def span_work(self):
        return 2 * self.all.span_prod

    @property
    def tile_efficiency(self):
        return prod([Min(1, getattr(self, n).tile_prod / 16) for n in 'mnk'])

@dataclass(frozen=True)
class BoundMatMul(BaseMatMul[Dim]):
    pass

@dataclass(frozen=True)
class ConcreteMatMul(BaseMatMul[ConcreteDim]):
    @property
    def tile_efficiency(self):
        return prod([min(1, getattr(self, n).tile_prod / 16) for n in 'mnk'])


        
@dataclass(frozen=True)
class Work:
    forward: list[MatMul]
    recompute: list[MatMul]

    @classmethod
    def make(cls, forward: list[str], recompute: list[str]):
        return cls(
                forward = [MatMul.make(m) for m in forward],
                recompute = [MatMul.make(m) for m in recompute],
        )

def bind_work(grid, work):
    if isinstance(grid, ConcreteGrid):
        return ConcreteWork(work, grid)
    elif isinstance(grid, BoundGrid):
        return BoundWork(work, grid)

@dataclass(frozen=True)
class BaseWork[D: Dim]:
    spec: Work
    grid: BaseGrid[D]

    def mmas(self, phase):
        match phase:
            case Phase.fwd: return self.forward
            case Phase.bwd: return self.forward + self.forward + self.recompute
            case _: assert False


@dataclass(frozen=True)
class BoundWork(BaseWork[Dim]):
    def __getattr__(self, name):
        if name in field_names(self.spec):
            return [BoundMatMul(x, self.grid) for x in getattr(self.spec, name)]
        else:
            raise AttributeError(f'name {name} not in {field_names(self.spec)}')

@dataclass(frozen=True)
class ConcreteWork(BaseWork[ConcreteDim]):
    def __getattr__(self, name):
        if name in field_names(self.spec):
            return [ConcreteMatMul(x, self.grid) for x in getattr(self.spec, name)]
        else:
            raise AttributeError(f'name {name} not in {field_names(self.spec)}')

