from __future__ import annotations
from dataclasses import dataclass
from enum import Enum

from sympy import Rational

from .base import *
from .grid import BoundGrid, BoundDims, Dims
from .variables import * 
from .typestuff import DType

class BufferDep(Enum):
    Batch = 'batch'
    Fold = 'fold'
    Batch_Fold = 'batch_fold'
    Constant = 'constant'

class BufferRole(Enum):
    Input = 'input'
    Output = 'output'
    Intermediate = 'intermediate'

@dataclass(frozen=True)
class Buffer:
    spec: Dims
    dtype: DType

    @classmethod
    def make(cls, string: str, dtype: DType) -> Self:
        return cls(Dims.parse(string), dtype)

    def bind(self, name: str, grid: BoundGrid, role: BufferRole) -> BoundBuffer:
        return BoundBuffer(
                name=name,
                grid=grid,
                spec=self.spec.bind(grid),
                role=role,
                dtype=self.dtype,
                )

@dataclass(frozen=True)
class BoundBuffer:
    name: str
    grid: BoundGrid
    spec: BoundDims
    role: BufferRole
    dtype: DType
    
    @property
    def is_output(self):
        return self.role == BufferRole.Output

    @property
    def is_input(self):
        return self.role == BufferRole.Input

    @property
    def is_intermediate(self):
        return self.role == BufferRole.Intermediate

    @property
    def dependency(self):
        match (bool(self.spec.batch), bool(self.spec.fold)):
            case (True, True): return BufferDep.Batch_Fold
            case (True, False): return BufferDep.Batch
            case (False, True): return BufferDep.Fold
            case (False, False): return BufferDep.Constant

    @property
    def contribution(self):
        return self.grid.outer | self.spec.inner

    @property
    def absent(self):
        return self.contribution - self.spec

    @property
    def bsize(self):
        return Rational(self.dtype.bitwidth, 8)

    @property
    def accessed_bytes(self):
        return self.bsize * self.contribution.total_prod / self.absent.span_prod

    @property
    def residual_multiplicity(self):
        return self.absent.total_prod / self.absent.span_prod

    @property
    def req_grad(self):
        return self.is_input

    def is_write(self, phase):
        match phase:
            case Phase.fwd: return self.is_output
            case Phase.bwd: return self.req_grad

    def is_read(self, phase):
        match phase:
            case Phase.fwd: return self.is_input
            case Phase.bwd: return self.is_input | self.is_output

    def traffic(self, phase):
        kind = 0
        if self.is_write(phase):
            kind = kind + WRITE
        if self.is_read(phase):
            kind = kind + READ
        return kind * self.accessed_bytes

    def contention(self, phase):
        match phase:
            case Phase.fwd: C = FWD_CONTENTION
            case Phase.bwd: C = BWD_CONTENTION
        if self.is_write(phase):
            return self.accessed_bytes * C(self.residual_multiplicity)
        else:
            return 0

    @property
    def numel(self):
        return self.spec.total_prod

    @property
    def total_bytes(self):
        return self.numel * self.bsize

    @property
    def tile_bytes(self):
        tile = self.spec.outer.tile_prod
        full = self.spec.inner.total_prod
        return self.bsize * tile * full

    def check(self):
        assert self.grid.dims.is_superset(self.spec)
        if self.is_output:
            assert self.spec.is_superset(self.grid.batch)
