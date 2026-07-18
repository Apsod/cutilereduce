from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
from copy import replace
from typing import Self, Any

import torch

from .grid import BaseDims, Dims, D, Dim, BoundGrid, ConcreteDim, ConcreteGrid
from .typestuff import DType, to_torch_dtype
import cuda.tile as ct

class BufferDep(Enum):
    Batch = 'batch'
    Fold = 'fold'
    Batch_Fold = 'batch_fold'
    Constant = 'constant'

class BufferRole(Enum):
    Input = 'input'
    Output = 'output'
    Intermediate = 'intermediate'
    State = 'state'

@dataclass(frozen=True, kw_only=True)
class Buffer:
    spec: Dims
    dtype: DType
    req_grad: bool
    default: Any = None


    @classmethod
    def make(cls, string: str, dtype: DType, req_grad=False, default=None) -> Self:
        return cls(
                spec=Dims.parse(string), 
                dtype=dtype, 
                req_grad=req_grad,
                default=default,
                )

    def generic_bind(self, name: str, index: int, grid: BoundGrid | ConcreteGrid, role: BufferRole):
        if isinstance(grid, BoundGrid):
            return self.bind(name, index, grid, role)
        elif isinstance(grid, ConcreteGrid):
            return self.concretize(name, index, grid, role)


    def bind(self, name: str, index: int, grid: BoundGrid, role: BufferRole) -> BoundBuffer:
        return BoundBuffer(
                name=name,
                program_index=index,
                spec=grid.bind_dims(self.spec),
                role=role,
                dtype=self.dtype,
                req_grad=self.req_grad,
                default=self.default,
                )

    def concretize(self, name: str, index: int, grid: ConcreteGrid, role: BufferRole) -> ConcreteBuffer:
        return ConcreteBuffer(
                name=name,
                program_index=index,
                spec=grid.bind_dims(self.spec),
                role=role,
                dtype=self.dtype,
                req_grad=self.req_grad,
                default=self.default,
                )


@dataclass(frozen=True, kw_only=True)
class BaseBuffer[D: Dim]:
    name: str
    program_index: int
    spec: BaseDims[D]
    role: BufferRole
    dtype: DType
    default: None
    req_grad: bool

    @property
    def padding_mode(self):
        match self.default:
            case 0: return ct.PaddingMode.ZERO                      # noqa: E701
            case -0: return ct.PaddingMode.NEG_ZERO                 # noqa: E701
            case float('inf'): return ct.PaddingMode.POS_INF        # noqa: E701
            case float('-inf'): return ct.PaddingMode.NEG_INF       # noqa: E701
            case _: return ct.PaddingMode.UNDETERMINED              # noqa: E701

    @property
    def dims(self):
        return self.spec

    @property
    def torch_dtype(self):
        return to_torch_dtype(self.dtype)

    @property
    def base(self):
        return Buffer(
                spec=self.spec.base, 
                dtype=self.dtype, 
                req_grad=self.req_grad,
                default=self.default,
                )

    @property
    def grid(self):
        return self.spec[0].grid
    
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
            case (True, True): return BufferDep.Batch_Fold     # noqa: E701
            case (True, False): return BufferDep.Batch         # noqa: E701
            case (False, True): return BufferDep.Fold          # noqa: E701
            case (False, False): return BufferDep.Constant     # noqa: E701

    @property
    def contribution(self):
        return self.grid.outer | self.spec.inner

    @property
    def absent(self):
        return self.contribution - self.spec

    @property
    def in_loop(self):
        return self.grid.group_dim not in self.absent

    @property
    def bsize(self):
        return Fraction(self.dtype.bitwidth, 8)

    @property
    def accessed_elems(self):
        return self.contribution.total_prod / self.absent.span_prod

    @property
    def accessed_bytes(self):
        return self.bsize * self.accessed_elems

    @property
    def residual_multiplicity(self):
        return self.absent.total_prod / self.absent.span_prod

    @property
    def target_tiles(self):
        return self.spec.total_prod / self.spec.span_prod






    #def is_write(self, phase):
    #    match phase:
    #        case Phase.fwd: return self.is_output              # noqa: E701
    #        case Phase.bwd: return self.req_grad               # noqa: E701

    #def is_read(self, phase):
    #    match phase:
    #        case Phase.fwd: return self.is_input                     # noqa: E701
    #        case Phase.bwd: return self.is_input | self.is_output    # noqa: E701

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

    @property
    def grid_index(self):
        return self.spec.grid_index

    @property
    def buffer_shape(self):
        return self.spec.shape

    @property
    def tile_shape(self):
        return self.spec.tile_shape

    def check(self):
        assert self.grid.dims.is_superset(self.spec)
        if self.is_output:
            assert not self.req_grad
            assert self.spec.is_superset(self.grid.batch)
    
    @property
    def grad_buffer(self):
        assert self.req_grad
        return replace(self, dtype=ct.float32)

@dataclass(frozen=True)
class BoundBuffer(BaseBuffer[Dim]):
    pass

@dataclass(frozen=True)
class ConcreteBuffer(BaseBuffer[ConcreteDim]):

    def empty(self, device=None):
        return torch.empty(self.buffer_shape, device=device, requires_grad=self.req_grad, dtype=self.torch_dtype)

    def default(self, device=None):
        return torch.full(self.buffer_shape, self.default, device=device, requires_grad=self.req_grad, dtype=self.torch_dtype)

    def zeros(self, device=None):
        return torch.zeros(self.buffer_shape, device=device)

    @property
    def is_grouped(self):
        return any(d.grouped for d in self.dims)

@dataclass(frozen=True)
class BufferBundle[D: Dim]:
    values: tuple[BaseBuffer[D], ...]

    def index(self, b):
        return self.values.index(b)
    
    @property
    def base(self):
        return {
            b.name: b.base
            for b
            in self.values
         }

    def empty(self, device=None):
        return tuple(
                b.empty(device=device) for b in self.values
        )

    def default(self, device=None):
        return tuple(
                b.full(device=device) for b in self.values
        )

    def zeros(self, device=None):
        return tuple(
                b.zeros(device=device) for b in self.values
        )

    def __add__(self, other):
        return BufferBundle(self.values + other.values)

    @property
    def grad(self):
        return BufferBundle(tuple(
            b.grad_buffer for b in self.values if b.req_grad
        ))
    
    @property
    def batch(self):
        return BufferBundle(tuple(
            b for b in self.values if not b.in_loop
        ))

    @property
    def fold(self):
        return BufferBundle(tuple(
            b for b in self.values if b.in_loop
        ))

    def without(self, d):
        return BufferBundle(tuple(
            b for b in self.values if d in b.absent
        ))

    def filter(self, fun):
        return BufferBundle(tuple(
            b for b in self.values if fun(b)
        ))

    def check(self):
        for b in self.values:
            b.check()

    @property
    def total_bytes(self):
        return sum(b.total_bytes for b in self.values)

    @property
    def tile_bytes(self):
        return sum(b.tile_bytes for b in self.values)

    @property
    def accessed_bytes(self):
        return sum(b.accessed_bytes for b in self.values)

    def __len__(self) -> int:
        return len(self.values)

    def __contains__(self, val: T) -> bool:
        return val in self.values

    def __iter__(self):
        return iter(self.values)



