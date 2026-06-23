from __future__ import annotations
from enum import Enum
from dataclasses import dataclass, field
from collections import namedtuple
from types import SimpleNamespace
from typing import TypeVar, Self, Any
import sympy
from math import prod
from sympy import diff, symbols
from functools import reduce, wraps, cached_property
from copy import replace
import cuda.tile as ct

T = TypeVar('T')

READ, WRITE = sympy.symbols('READ WRITE')

other_default = {READ: 1, WRITE: 1}

class BufferDep(Enum):
    Batch = 'batch'
    Fold = 'fold'
    Batch_Fold = 'batch_fold'
    Constant = 'constant'

class BufferRole(Enum):
    Input = 'input'
    Output = 'output'
    Intermediate = 'intermediate'

    @classmethod
    def dual(cls, val):
        match val:
            case BufferRole.Input: return BufferRole.Output
            case BufferRole.Output: return BufferRole.Input
            case BufferRole.Intermediate: return BufferRole.Intermediate



def promote_type(func):
    @wraps(func)
    def wrapper(self, other):
        result = func(self, other)
        
        # Respect Python's fallback mechanism if NotImplemented is returned
        if result is NotImplemented:
            return NotImplemented
            
        # Determine the most specific class
        if isinstance(other, TupleSet) and issubclass(type(other), type(self)):
            target_cls = type(other)
        else:
            target_cls = type(self)
            
        # If the result isn't already the subclass, reconstruct it
        if type(result) is not target_cls:
            return target_cls(result.value)
        return result
        
    return wrapper

@dataclass(frozen=True)
class TupleSet[T]:
    value: tuple[T, ...]

    @classmethod
    def parse(cls, txt: str) -> Self:
        return cls.make(*txt.strip().split())

    @classmethod
    def make(cls, *values: T) -> Self:
        assert len(values) == len(set(values))
        return cls(tuple(values))

    @property
    def set(self) -> set[T]:
        return set(self.value)

    def __len__(self) -> int:
        return len(self.value)

    def __contains__(self, val: T) -> bool:
        return val in self.value

    def __iter__(self):
        return iter(self.value)
    
    @promote_type
    def __or__(self, other: Self) -> Self:
        return TupleSet(self.value + tuple(x for x in other if x not in self))

    @promote_type
    def __and__(self, other: Self) -> Self:
        return TupleSet(tuple(x for x in self if x in other))

    @promote_type
    def __sub__(self, other: Self) -> Self:
        return TupleSet(tuple(x for x in self if x not in other))

    def __xor__(self, other: Self) -> Self:
        return (self - other) | (other - self)
    
    def __add__(self, other: Self) -> Self:
        return self | other

    def __bool__(self) -> bool:
        return bool(self.value)

    def __eq__(self, other) -> bool:
        return sorted(self.value) == sorted(other.value)
    
    @classmethod
    def zero(cls):
        return cls.make()

    @classmethod
    def union(cls, *xs):
        return reduce(lambda a, b: a | b, xs, cls.zero())

    def is_superset(self, *sets):
        return all(not bool(s - self) for s in sets)

    def is_disjoint(self, *sets):
        return all(not bool(s & self) for s in sets)

    def tmap(self, f):
        return tuple(f(x) for x in self)

    def dmap(self, f):
        return {x: f(x) for x in self}
    
    def subset(self, keep):
        return type(self)(tuple(x for x in self if keep(x)))


class Dim(str):
    __slots__ = ('_grid',)

    def __new__(cls, name: str, grid: Grid = None):
        obj = super().__new__(cls, name)
        obj._grid = grid
        return obj

    @property
    def grid(self) -> 'Grid':
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
    def grouped(self):
        return self in self.grid.fold

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

    def __repr__(self):
        return f"{self.__class__.__name__}({super().__repr__()}, grid={self.grid})"

    def __str__(self):
        return f"{super().__str__()}"

@dataclass(frozen=True)
class BufferSpec:
    spec: DimSpec
    dtype: ct.DType

    @classmethod
    def make(cls, str, dtype):
        return cls(DimSpec.parse(str), dtype)

DimSpec = TupleSet[str]

@dataclass(frozen=True)
class DimInfo(TupleSet[Dim]):
    @property
    def dims(self) -> tuple[Dim]:
        return self.value

    @property
    def dimset(self):
        return self.set

    @property
    def grouped(self):
        return self.subset(lambda x: x.grouped)

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
        return prod(x.tile_var for x in self.tiled)

    @property
    def group_prod(self):
        return prod(x.group_var for x in self.grouped)

    @property
    def total_prod(self):
        return prod(x.total_var for x in self)

    @property
    def span_prod(self):
        return prod(x.span_exp for x in self)

    @classmethod
    def from_spec(cls, spec: DimSpec, grid: Grid | None =None):
        return cls(spec.tmap(lambda name: Dim(name=name, grid=grid)))

    def __str__(self):
        return str(self.tmap(str))

@dataclass(frozen=True)
class Grid:
    _input: DimSpec
    _output: DimSpec
    _batch: DimSpec
    _fold: DimSpec

    @classmethod
    def make(cls, 
             input: dict[str, BufferSpec], 
             output: dict[str, BufferSpec],
             batch: DimSpec,
             fold: DimSpec):
        input = DimSpec.union(*(v.spec for v in input.values()))
        output = DimSpec.union(*(v.spec for v in output.values()))
        return cls(input, output, batch, fold)

    def check(self, simple=True):
        if simple:
            assert len(self.fold) == 1, f'{self.fold!s}'
        assert self.dims.is_superset(self.batch, self.fold)
        assert self.output.is_superset(self.batch)
        assert self.fold.is_disjoint(self.batch, self.output)

    @cached_property
    def dim_map(self):
        return {x: x for x in self.dims}
        return _cache[d]

    @cached_property
    def input(self):
        return DimInfo.from_spec(self._input, self)

    @cached_property
    def output(self):
        return DimInfo.from_spec(self._output, self)

    @cached_property
    def batch(self):
        return DimInfo.from_spec(self._batch, self)

    @cached_property
    def fold(self):
        return DimInfo.from_spec(self._fold, self)

    @cached_property
    def outer(self) -> DimInfo:
        return self.batch | self.fold

    @cached_property
    def dims(self) -> DimInfo:
        return self.input | self.output

    @cached_property
    def inner(self) -> DimInfo:
        return self.dims - self.outer

@dataclass(frozen=True)
class BufferInfo:
    name: str
    grid: Grid
    spec: DimInfo
    role: BufferRole
    dtype: ct.DType
    
    @classmethod
    def from_spec(cls, name: str, spec: BufferSpec, grid: Grid, role: BufferRole):
        return cls(
                name=name,
                grid=grid,
                spec=DimInfo.from_spec(spec.spec, grid),
                role=role,
                dtype=spec.dtype,
                )

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
        return sympy.Rational(self.dtype.bitwidth, 8)

    @property
    def fwd_kind_var(self):
        match self.role:
            case BufferRole.Input: return READ
            case BufferRole.Output: return WRITE
            case BufferRole.Intermediate: return 0

    @property
    def bwd_kind_var(self):
        match self.role:
            case BufferRole.Input: return WRITE
            case BufferRole.Output: return READ
            case BufferRole.Intermediate: return 0

    @property
    def fwd_traffic(self):
        """
        Get the memory polynomial corresponding to this buffer, i.e. 
        the amount of reads given the tiling and grouping.
        The current polynomial assumes:
            - one (grouped) fold dimension
            - all inner dimensions are loaded in full
            - batch dimensions are not grouped
        """
        den = self.absent.span_prod
        num = self.contribution.total_prod
        return self.bsize * self.fwd_kind_var * num / den

    @property
    def bwd_traffic(self):
        recompute = self.fwd_traffic
        den = self.absent.span_prod
        num = self.contribution.total_prod
        return recompute + self.bsize * self.bwd_kind_var * num / den

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

@dataclass(frozen=True)
class Work:
    forward: sympy.Expr
    recompute: sympy.Expr

    @classmethod
    def make(cls, forward: str, recompute: str):
        return cls(
                sympy.sympify(forward),
                sympy.sympify(recompute),
        )

    @property
    def fwd(self):
        return self.forward

    @property
    def bwd(self):
        return 2 * self.forward + self.recompute

@dataclass(frozen=True)
class Meta:
    input: dict[str, BufferInfo]
    output: dict[str, BufferInfo]
    intermediate: tuple[BufferInfo]
    work: Work
    grid: Grid

    @classmethod
    def make(cls, 
             input: dict[str, BufferSpec], 
             output: dict[str, BufferSpec], 
             batch: DimSpec, 
             fold: DimSpec, 
             work: Work, 
             intermediate: list[BufferSpec] = None):

        if intermediate is None:
            intermediate = []

        grid = Grid.make(
                input = input,
                output = output,
                batch = batch,
                fold = fold,
                )

        ret = cls(
                input={k: BufferInfo.from_spec(name=k, grid=grid, spec=v, role=BufferRole.Input) for k, v in input.items()},
                output={k: BufferInfo.from_spec(name=k, grid=grid, spec=v, role=BufferRole.Output) for k, v in output.items()},
                intermediate=tuple(BufferInfo.from_spec(name=f'intermediate_{k}', grid=grid, spec=v, role=BufferRole.Intermediate) for k, v in enumerate(intermediate)),
                work = work,
                grid=grid,
                )
        ret.check()
        return ret


    @property
    def output_buffers(self):
        return (*self.output.values(),)

    @property
    def input_buffers(self):
        return (*self.input.values(),)

    @property
    def io_buffers(self):
        return (*self.input_buffers, *self.output_buffers)

    @property
    def fwd_traffic(self):
        return sum(v.fwd_traffic for v in self.io_buffers)

    @property
    def fwd_traffic_lower_bound(self):
        return sum(v.total_bytes for v in self.io_buffers)

    @property
    def fwd_traffic_ratio(self):
        return self.fwd_traffic / self.fwd_traffic_lower_bound

    @property
    def bwd_traffic(self):
        return sum(v.bwd_traffic for v in self.io_buffers)

    @property
    def bwd_traffic_lower_bound(self):
        return sum(v.total_bytes for v in self.io_buffers)

    @property
    def bwd_traffic_ratio(self):
        return self.bwd_traffic / self.bwd_traffic_lower_bound

    @property
    def fwd_tile_bytes(self):
        return sum(v.tile_bytes for v in self.fwd_buffers + self.intermediate)

    @property
    def fwd_total_work(self):
        return self.work.fwd.subs({d: d.total_var for d in self.grid.dims})

    @property
    def bwd_total_work(self):
        return self.work.bwd.subs({d: d.total_var for d in self.grid.dims})

    @property
    def fwd_local_work(self):
        return self.work.fwd.subs({d: d.span_exp for d in self.grid.dims})

    @property
    def bwd_local_work(self):
        return self.work.bwd.subs({d: d.span_exp for d in self.grid.dims})

    def check(self):
        self.grid.check()
        for b in self.intermediate:
            b.check()
        for b in self.io_buffers:
            b.check()

    def estimate(self, sizes={}, tiling={}, grouping={}, other={}):
        return Estimator(self, sizes, tiling, grouping, other)

a = Meta.make(
        input = dict(
            q = BufferSpec.make('l h g dq', ct.bfloat16),
            k = BufferSpec.make('r h dq', ct.bfloat16),
            v = BufferSpec.make('r h dv', ct.bfloat16)
            ),
        output = dict(
            m = BufferSpec.make('l h g', ct.float32),
            e = BufferSpec.make('l h g', ct.float32),
            v = BufferSpec.make('l h g dv', ct.float32),
            ),
        intermediate = [
            BufferSpec.make('l h g r', ct.float32),
            ],
        work = Work.make(
            forward='2 * (l * r * h * g * dq + l * r * h * g * dv)',
            recompute='2 * (l * r * h * g * dq)',
            ),
        batch = DimSpec.parse('l h g'),
        fold = DimSpec.parse('r'),
    )


def substitute(func):
    @wraps(func)
    def wrapper(self):
        return func(self).subs(self.subs)
        
    return wrapper

@dataclass(frozen=True)
class Estimator:
    meta: Meta
    _sizes: dict[str, int]
    _tiling: dict[str, int]
    _grouping: dict[str, Any]
    _other: dict[sympy.Symbol, int]

    def group_dim(self, dim: str, groups=1):
        grouping = {}
        for d in self.dims:
            if d == dim:
                grouping[d] = d.total_var / (d.tile_var * groups)
            else:
                grouping[d] = 1
        return replace(self, _grouping=grouping)

    @property
    def dims(self):
        return self.meta.grid.dims

    @property
    def dim_map(self):
        return self.meta.grid.dim_map

    @property
    def sizes(self):
        return {self.dim_map[k].total_var.name: v for k, v in self._sizes.items()}

    @property
    def tiling(self):
        return {self.dim_map[k].tile_var.name: v for k, v in self._tiling.items()}

    @property
    def grouping(self):
        return {self.dim_map[k].group_var.name: v for k, v in self._grouping.items()}

    @property
    def subs(self):
        return {**self.sizes, **self.tiling, **self.grouping, **self._other}

    @property
    @substitute
    def fwd_traffic(self):
        return self.meta.fwd_traffic

    @property
    @substitute
    def fwd_traffic_ratio(self):
        return self.meta.fwd_traffic_ratio

    @property
    @substitute
    def bwd_traffic(self):
        return self.meta.bwd_traffic

    @property
    @substitute
    def bwd_traffic_ratio(self):
        return self.meta.bwd_traffic_ratio

    @property
    @substitute
    def fwd_total_work(self):
        return self.meta.fwd_total_work

    @property
    @substitute
    def fwd_local_work(self):
        return self.meta.fwd_local_work

    @property
    @substitute
    def bwd_total_work(self):
        return self.meta.bwd_total_work

    @property
    @substitute
    def bwd_local_work(self):
        return self.meta.bwd_local_work

sizes = dict(
    l = 1024*128,
    h = 16,
    g = 16,
    dq = 1024,
    dv = 1024,
    r = 1024*128,
)

e = a.estimate(
        sizes=sizes,
        other={READ: 1, WRITE: 1},
        )

def section(title, width=20, fill='='):
    print(f'{ {title} :{fill}^20}')

print()
print('==========')
print(f'{e.group_dim('r').fwd_traffic=}')
print(f'{e.group_dim('r').fwd_traffic_ratio=}')
print(f'{e.group_dim('r').fwd_total_work=}')
print(f'{e.group_dim('r').fwd_local_work=}')
print(f'{e.group_dim('r').bwd_traffic=}')
print(f'{e.group_dim('r').bwd_traffic_ratio=}')
print(f'{e.group_dim('r').bwd_total_work=}')
print(f'{e.group_dim('r').bwd_local_work=}')
print('==========')
