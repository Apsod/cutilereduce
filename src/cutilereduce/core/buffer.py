from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from fractions import Fraction

import sympy


from .base import *
from .grid import BaseGrid, BaseDims, Dims, D, Dim, ConcreteDim, BoundGrid, ConcreteDim, ConcreteGrid
from .variables import * 
from .typestuff import DType
from .config import Config

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
    req_grad: bool

    @classmethod
    def make(cls, string: str, dtype: DType, req_grad=False) -> Self:
        return cls(
                spec=Dims.parse(string), 
                dtype=dtype, 
                req_grad=req_grad,
                )

    def generic_bind(self, name: str, grid: BoundGrid | ConcreteGrid, role: BufferRole):
        if isinstance(grid, BoundGrid):
            return self.bind(name, grid, role)
        elif isinstance(grid, ConcreteGrid):
            return self.concretize(name, grid, role)


    def bind(self, name: str, grid: BoundGrid, role: BufferRole) -> BoundBuffer:
        return BoundBuffer(
                name=name,
                grid=grid,
                spec=grid.bind_dims(self.spec),
                role=role,
                dtype=self.dtype,
                req_grad=self.req_grad,
                )

    def concretize(self, name: str, grid: ConcreteGrid, role: BufferRole) -> ConcreteBuffer:
        return ConcreteBuffer(
                name=name,
                grid=grid,
                spec=grid.bind_dims(self.spec),
                role=role,
                dtype=self.dtype,
                req_grad=self.req_grad,
                )


@dataclass(frozen=True)
class BaseBuffer[D: Dim]:
    name: str
    grid: BaseGrid[D]
    spec: BaseDims[D]
    role: BufferRole
    dtype: DType
    req_grad: bool
    
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
    def batch_load(self):
        return self.dependency == BufferDep.Batch_Fold

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
        return Fraction(self.dtype.bitwidth, 8)

    @property
    def accessed_bytes(self):
        return self.bsize * self.contribution.total_prod / self.absent.span_prod

    @property
    def residual_multiplicity(self):
        return self.absent.total_prod / self.absent.span_prod

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
            assert not self.req_grad
            assert self.spec.is_superset(self.grid.batch)

@dataclass(frozen=True)
class BoundBuffer(BaseBuffer[Dim]):
    pass

@dataclass(frozen=True)
class ConcreteBuffer(BaseBuffer[ConcreteDim]):
    def traffic(self, phase):
        kind = 0
        if self.is_write(phase):
            kind = kind + WRITE
        if self.is_read(phase):
            kind = kind + READ
        return self.grid.config._eval(kind * self.accessed_bytes)

    def contention(self, phase):
        match phase:
            case Phase.fwd: C = FWD_CONTENTION
            case Phase.bwd: C = BWD_CONTENTION
        if self.is_write(phase):
            return self.grid.config._eval(self.accessed_bytes * C(self.residual_multiplicity))
        else:
            return 0
