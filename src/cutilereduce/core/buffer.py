from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Self, Any
from copy import replace
from fractions import Fraction

from .axis import Axis, AxisId, Axes, axis_id
from .typestuff import DType, to_torch_dtype
from .utilities import TupleSet, prod, forward

import cuda.tile as ct
import sympy

class BufferRole(Enum):
    Input = "input"
    Output = "output"
    GradStorage = "grad_storage"

Input = BufferRole.Input
Output = BufferRole.Output
GradStorage = BufferRole.GradStorage

@dataclass(frozen=True, order=True)
class Internal:
    tags: tuple[str]

    def __init__(self, *tags: str):
        object.__setattr__(self, "tags", tuple(tags))

    def tag(self, tag: str) -> Self:
        return Internal(*self.tags, tag)


@dataclass(frozen=True, order=True)
class BufferId:
    role: BufferRole | Internal
    name: str

    def tag(self, tag: str): 
        assert isinstance(self.role, Internal)
        return replace(self, role = self.role.tag(tag))
    
    @property
    def as_grad(self):
        assert self.role == BufferRole.Input
        return replace(self, role=BufferRole.GradStorage)

@dataclass(frozen=True)
class ShapeData:
    bytes_per_elem: Fraction
    shape: tuple[int | sympy.Expr,...]

    @property
    def numel(self):
        return prod(self.shape)

    @property
    def bytes(self):
        return self.numel * self.bytes_per_elem

@dataclass(frozen=True)
class Buffer:
    id: BufferId
    axes: Axes
    dtype: DType
    req_grad: bool = False
    default: Any = None

    role = forward('id', 'role')

    def as_grad(self, dtype=ct.float32) -> Self:
        assert self.role == BufferRole.Input
        assert self.req_grad
        return replace(
            self,
            id=self.id.as_grad,
            dtype=dtype,
            req_grad=False,
            default=0,
        )

    def with_prefix_axes(self, group:str, axes: Axes) -> Self:
        return replace(
                self,
                id=self.id.tag(group),
                axes=axes | self.axes,
                req_grad=False
        )

    @property
    def torch_dtype(self):
        return to_torch_dtype(self.dtype)

    @property
    def bytes_per_elem(self):
        return Fraction(self.dtype.bitwidth, 8)

    @property
    def padding_mode(self):
        match self.default:
            case 0: return ct.PaddingMode.ZERO                      # noqa: E701
            case -0: return ct.PaddingMode.NEG_ZERO                 # noqa: E701
            case float('inf'): return ct.PaddingMode.POS_INF        # noqa: E701
            case float('-inf'): return ct.PaddingMode.NEG_INF       # noqa: E701
            case _: return ct.PaddingMode.UNDETERMINED              # noqa: E701

@dataclass(frozen=True, kw_only=True)
class BufferBundle(TupleSet[Buffer]):

    def as_grad(self, dtype=ct.float32):
        assert all(b.role == Input for b in self)
        return self.subset(lambda b: b.req_grad).map(lambda b: b.as_grad(dtype))

    @classmethod
    def make(cls, role: BufferRole, **buffers: BufferSpec) -> Self:
        return BufferBundle(
            values=tuple(
                spec.bind(BufferId(role, name)) for name, spec in buffers.items()
            )
        )

    @staticmethod
    def key(x):
        match x:
            case BufferId():
                return x
            case Buffer():
                return x.id
            case _:
                raise KeyError(f'{type(x)} not BufferId')
    
    @property
    def buffers(self) -> tuple[Buffer, ...]:
        return self.values

bundle_spec = BufferBundle.make

@dataclass(frozen=True)
class BufferSpec:
    axes: Axes
    dtype: DType
    req_grad: bool = False
    default: Any = None

    @classmethod
    def make(cls, axes: str, dtype: DType, req_grad=False, default=None):
        return cls(axes=Axes.make(axes), dtype=dtype, req_grad=req_grad, default=default)

    def bind(self, id: BufferId) -> Buffer:
        return Buffer(
            id=id,
            axes=self.axes,
            dtype=self.dtype,
            req_grad=self.req_grad,
            default=self.default,
        )

buffer_spec = BufferSpec.make
