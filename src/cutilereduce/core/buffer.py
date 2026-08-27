from dataclasses import dataclass
from enum import Enum
from typing import Self, Any
from copy import replace
from fractions import Fraction

from .axis import Axis, AxisId, Axes, axis_id
from .typestuff import DType, to_torch_dtype
from .utilities import TupleSet, prod

import cuda.tile as ct
import sympy

class BufferRole(Enum):
    Input = "input"
    Output = "output"
    GradStorage = "grad_storage"
    Execution = "execution"
    GradAccumulator = "grad_accumulator"

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
    name: str
    axes: Axes
    dtype: DType
    req_grad: bool = False
    default: Any = None

    def with_prefix_axes(self, group:str, axes: Axes) -> Self:
        return replace(
                self,
                name=f"{group}:{self.name}",
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
    role: BufferRole

    def mk_grad_bundle(self, dtype=ct.Float32) -> Self:
        assert self.role == BufferRole.Input
        return (
                replace(self, role=BufferRole.GradStorage)
                .subset(lambda x: x.req_grad)
                .map(lambda x: replace(x, dtype=dtype, default=0))
        )

    def with_prefix_axes(self, group: str, axes: Axes) -> Self:
        return self.map(lambda x: x.with_prefix_axes(group, axes))

    @staticmethod
    def key(x):
        match x:
            case Buffer():
                return x.name
            case str():
                return x
            case _:
                raise KeyError(f'{type(x)} not in Buffer | str')
    
    @property
    def buffers(self) -> tuple[Buffer, ...]:
        return self.values
