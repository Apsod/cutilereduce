from __future__ import annotations
from dataclasses import dataclass
from sympy import Min, Max
from math import prod

from .base import field_names, Phase
from .grid import BoundGrid, Dims

@dataclass(frozen=True)
class MatMul:
    b: Dims
    m: Dims  
    n: Dims
    k: Dims

    @classmethod
    def make(cls, string: str):
        return cls(*map(Dims.parse, string.split(':')))

    def bind(self, grid) -> BoundMatMul:
        return BoundMatMul(self, grid)

@dataclass(frozen=True)
class BoundMatMul:
    spec: MatMul
    grid: BoundGrid

    def __getattr__(self, name):
        if name in field_names(self.spec):
            return getattr(self.spec, name).bind(self.grid)
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
    def tile_efficiency_prod(self):
        return prod([Min(1, getattr(self, n).tile_prod / 16) for n in 'mnk'])

    @property
    def tile_efficiency_bottleneck(self):
        return Min(*[Min(1, getattr(self, n).tile_prod / 16) for n in 'mnk'])
        
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

    def bind(self, grid: BoundGrid) -> BoundWork:
        return BoundWork(self, grid)



@dataclass(frozen=True)
class BoundWork:
    spec: Work
    grid: BoundGrid

    def __getattr__(self, name):
        if name in field_names(self.spec):
            return [x.bind(self.grid) for x in getattr(self.spec, name)]
        else:
            raise AttributeError(f'name {name} not in {field_names(self.spec)}')

    def mmas(self, phase):
        match phase:
            case Phase.fwd: return self.forward
            case Phase.bwd: return self.forward + self.forward + self.recompute
