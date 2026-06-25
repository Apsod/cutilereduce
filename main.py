from __future__ import annotations
from enum import Enum
from dataclasses import dataclass, field, fields
from collections import namedtuple
from types import SimpleNamespace
from typing import TypeVar, Self, Any
import sympy
from math import prod
from sympy import diff, symbols, Max, Rational, Min, Function, log, ceiling
from functools import reduce, wraps, cached_property
from copy import replace
import cuda.tile as ct
import numpy
import itertools
import polars as pl

T = TypeVar('T')

READ = sympy.Symbol('READ')
WRITE = sympy.Symbol('WRITE')
GROUPS = sympy.Symbol('GROUPS')
PEAK_FLOPS = sympy.Symbol('PEAK_FLOPS')
BANDWIDTH = sympy.Symbol('BANDWIDTH')
MAX_RESIDENCY = sympy.Symbol('MAX_RESIDENCY')
MIN_PARALLELISM = sympy.Symbol('MIN_PARALLELISM')
MIN_MMA_EFFICIENCY = sympy.Symbol('MIN_MMA_EFFICIENCY')
FWD_CONTENTION = Function('FWD_CONTENTION')
BWD_CONTENTION = Function('BWD_CONTENTION')

def cdiv(num, den):
    return ceiling(num / den)

def contention_penalty(alpha, cap, contention):
    return Min(cap, alpha * log(Max(1, contention), 2))

def atomic_add_penalty(contention):
    return contention_penalty(alpha=0.5, cap=8, contention=contention)

def spinlock_penalty(contention):
    return contention_penalty(alpha=2, cap=64, contention=contention)

def field_names(x):
    return tuple((f.name for f in fields(x)))

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
    def dual(cls, val: BufferRole):
        match val:
            case BufferRole.Input: return BufferRole.Output
            case BufferRole.Output: return BufferRole.Input
            case BufferRole.Intermediate: return BufferRole.Intermediate
            case _: assert False

def substitute(func):
    @wraps(func)
    def wrapper(self):
        return self.eval(func(self))
        
    return wrapper

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

    def __getitem__(self, ix):
        return self.value[ix]
    
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

    @property
    def tile_exp(self):
        match self.outer:
            case True: return self.tile_var
            case False: return self.total_var

    def __repr__(self):
        return f"{self.__class__.__name__}({super().__repr__()}, grid={self.grid})"

    def __str__(self):
        return f"{super().__str__()}"

#Dims = TupleSet[str]

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
        return cls(input, output, batch, fold)

    def bind(self) -> BoundGrid:
        return BoundGrid(self)

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

@dataclass(frozen=True)
class Buffer:
    spec: Dims
    dtype: ct.DType

    @classmethod
    def make(cls, string: str, dtype: ct.DType) -> Self:
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
    dtype: ct.DType
    
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
        match self.role:
            case BufferRole.Input: kind = READ
            case BufferRole.Output: kind = WRITE
            case BufferRole.Intermediate: kind = 0
        return kind * self.accessed_bytes

    @property
    def bwd_traffic(self):
        match self.role:
            case BufferRole.Input: kind = READ
            case BufferRole.Output: kind = READ
            case BufferRole.Intermediate: kind = 0
        if self.req_grad:
            kind = kind + WRITE
        return kind * self.accessed_bytes

    @property
    def fwd_contention(self):
        kind = 1 if self.is_output else 0
        return kind * self.accessed_bytes * FWD_CONTENTION(self.residual_multiplicity)

    @property
    def bwd_contention(self):
        kind = 1 if self.req_grad else 0
        return kind * self.accessed_bytes *  BWD_CONTENTION(self.residual_multiplicity)

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

    @property
    def fwd(self):
        return self.forward

    @property
    def bwd(self):
        return self.forward + self.forward + self.recompute

class Phase(Enum):
    fwd = 'forward'
    bwd = 'backward'

@dataclass(frozen=True)
class Meta:
    grid: BoundGrid
    input: dict[str, BoundBuffer]
    output: dict[str, BoundBuffer]
    intermediate: tuple[BoundBuffer]
    work: BoundWork
    phase: Phase
    group_dim: str


    @classmethod
    def make(cls, 
             input: dict[str, Buffer], 
             output: dict[str, Buffer], 
             batch: Dims, 
             fold: Dims, 
             work: Work, 
             intermediate: list[Buffer] = None):

        if intermediate is None:
            intermediate = []

        grid = Grid.make(
                input = input,
                output = output,
                batch = batch,
                fold = fold,
                )

        grid = grid.bind()

        ret = cls(
                input={k: v.bind(k, grid, BufferRole.Input) for k, v in input.items()},
                output={k: v.bind(k, grid, BufferRole.Output)for k, v in output.items()},
                intermediate=tuple(v.bind(f'intermediate_{i}', grid, BufferRole.Intermediate) for i, v in enumerate(intermediate)),
                grid=grid,
                work=work.bind(grid),
                phase = Phase.fwd,
                group_dim = fold[0],
                )
        ret.check()
        return ret

    def group(self, dim: str):
        return replace(self, group_dim=dim)

    @property
    def fwd(self):
        return replace(self, phase=Phase.fwd)

    @property
    def bwd(self):
        return replace(self, phase=Phase.bwd)
    
    @property
    def full_span(self):
        return {
            d.group_var.name: d.total_var / d.tile_var
            for d in self.grid.outer
        }

    @property
    def output_buffers(self):
        return (*self.output.values(),)

    @property
    def input_buffers(self):
        return (*self.input.values(),)

    @property
    def grad_buffers(self):
        return tuple(b for b in self.input_buffers if b.req_grad)

    @property
    def io_buffers(self):
        return (*self.input_buffers, *self.output_buffers)

    @property
    def read_buffers(self):
        match self.phase:
            case Phase.fwd: return self.input_buffers
            case Phase.bwd: return self.input_buffers + self.output_buffers

    @property
    def write_buffers(self):
        match self.phase:
            case Phase.fwd: return self.output_buffers
            case Phase.bwd: return self.grad_buffers 

    @property
    def grouped_buffers(self):
        return tuple(b for b in self.write_buffers if self.group_dim in b.absent)

    @property
    def contention_dims(self):
        match self.phase:
            case Phase.fwd: return BoundDims.union(*(b.absent for b in self.output_buffers))
            case Phase.bwd: return BoundDims.union(*(b.absent for b in self.grad_buffers))

    @property
    def mmas(self):
        match self.phase:
            case Phase.fwd: return self.work.fwd
            case Phase.bwd: return self.work.bwd

    def check(self):
        self.grid.check()
        for b in self.intermediate:
            b.check()
        for b in self.io_buffers:
            b.check()

    def estimate(self, sizes={}, tiling={}, grouping={}, symbols={}, functions={}):
        return Estimator(
                self, 
                sizes,
                tiling,
                grouping,
                DEFAULT_SYMBOLS | symbols, 
                DEFAULT_FUNCTIONS | functions,
                )

    ###################### SYMBOLIC QUANTITIES ####################### 
    
    @property
    def traffic(self):
        match self.phase:
            case Phase.fwd: return sum(v.fwd_traffic for v in self.io_buffers)
            case Phase.bwd: return sum(v.bwd_traffic for v in self.io_buffers)

    @property
    def effective_traffic(self):
        return self.traffic + self.contention

    @property
    def traffic_lower_bound(self):
        return self.traffic.subs(self.full_span)

    @property
    def traffic_ratio(self):
        return self.traffic / self.traffic_lower_bound

    @property
    def effective_traffic_ratio(self):
        return self.effective_traffic / self.traffic_lower_bound

    @property
    def tile_bytes(self):
        return sum(v.tile_bytes for v in self.io_buffers + self.intermediate)

    @property
    def residency_bytes(self):
        return sum(v.tile_bytes for v in self.grouped_buffers + self.intermediate)

    @property
    def contention(self):
        match self.phase:
            case Phase.fwd: return sum(v.fwd_contention for v in self.output_buffers)
            case Phase.bwd: return sum(v.bwd_contention for v in self.grad_buffers)

    @property
    def total_work(self):
        return sum(x.total_work for x in self.mmas)

    @property
    def effective_total_work(self):
        return sum(x.total_work / x.tile_efficiency_prod for x in self.mmas)

    @property
    def mma_efficiency(self):
        return self.total_work / self.effective_total_work

    @property
    def mma_penalty(self):
        return 1 / self.mma_efficiency

    @property
    def tile_work(self):
        return sum(x.tile_work for x in self.mmas)

    @property
    def effective_tile_work(self):
        return sum(x.tile_work / x.tile_efficiency_prod for x in self.mmas)

    @property
    def span_work(self):
        return sum(x.span_work for x in self.mmas)

    @property
    def effective_span_work(self):
        return sum(x.span_work / x.tile_efficiency_prod for x in self.mmas)

    @property
    def effective_roofline(self):
        return Max(self.effective_total_work / PEAK_FLOPS, self.effective_traffic / BANDWIDTH)

    @property
    def ridge(self):
        return PEAK_FLOPS / BANDWIDTH

    @property
    def arithmetic_intensity(self):
        return (self.total_work / self.traffic) / self.ridge

    @property
    def effective_arithmetic_intensity(self):
        return (self.effective_total_work / self.effective_traffic) / self.ridge

    @property
    def parallelism(self):
        return self.total_work / self.span_work

    @property
    def inverse_tile_work(self):
        return 1 / self.tile_work

    @property
    def inverse_parallelism(self):
        return 1 / self.parallelism

    @property
    def program_count(self):
        return self.grid.outer.total_prod / self.grid.outer.span_prod

    @property
    def residency_filter(self):
        return MAX_RESIDENCY - self.residency_bytes

    @property
    def parallelism_filter(self):
        return self.program_count - MIN_PARALLELISM

    @property
    def groups(self):
        return GROUPS

    @property
    def mma_efficiency_filter(self):
        return Min(*(x.tile_efficiency_bottleneck for x in self.mmas)) - MIN_MMA_EFFICIENCY

DEFAULT_SYMBOLS = {
        READ: 1, 
        WRITE: 1,
        }

DEFAULT_FUNCTIONS = {
        FWD_CONTENTION: spinlock_penalty,
        BWD_CONTENTION: atomic_add_penalty,
        }

a = Meta.make(
        input = dict(
            q = Buffer.make('l h g dq', ct.bfloat16),
            k = Buffer.make('r h dq', ct.bfloat16),
            v = Buffer.make('r h dv', ct.bfloat16)
            ),
        output = dict(
            m = Buffer.make('l h g', ct.float32),
            e = Buffer.make('l h g', ct.float32),
            v = Buffer.make('l h g dv', ct.float32),
            ),
        intermediate = [
            Buffer.make('l h g r', ct.float32),
            ],
        work = Work.make(
            forward=[
                'h : l g : r : dq', 
                'h : l g : dv : r'
                ],
            recompute=[
                'h : l g : r : dq',
                ]
            ),
        batch = Dims.parse('l h g'),
        fold = Dims.parse('r'),
    )

PARETO_METRICS = ['mma_penalty', 'effective_roofline']
FILTER_METRICS = ['parallelism_filter', 'residency_filter', 'mma_efficiency_filter']
OTHER_METRICS = ['residency_bytes', 'arithmetic_intensity']
SWEEP_METRICS = list(set(PARETO_METRICS) | set(FILTER_METRICS) | set(OTHER_METRICS))
PARETO_IX = [SWEEP_METRICS.index(n) for n in PARETO_METRICS]
FILTER_IX = [SWEEP_METRICS.index(n) for n in FILTER_METRICS]

def pareto_mask(metrics):
    x = metrics[:, PARETO_IX]
    keep = numpy.ones(x.shape[0], dtype=bool)
    for i in range(x.shape[0]):
        if not keep[i]:
            continue

        dominates_i = numpy.all(x <= x[i], axis=1) & numpy.any(x < x[i], axis=1)
        if numpy.any(dominates_i):
            keep[i] = False
            continue

        i_dominates = numpy.all(x[i] <= x, axis=1) & numpy.any(x[i] < x, axis=1)
        keep[i_dominates] = False
    return keep

@dataclass(frozen=True)
class Estimator:
    meta: Meta
    _sizes: dict[str, int]
    _tiling: dict[str, int]
    _grouping: dict[str, Any]
    _symbols: dict[sympy.Symbol, int]
    _functions: dict[sympy.Symbol, Any]

    def __getattr__(self, name):
        value = getattr(self.meta, name)
        if isinstance(value, sympy.Expr):
            return self.eval(value)
        else:
            raise AttributeError(f'{name} is not a symbolic quantity of meta')

    def group(self, dim: str):
        grouping = {}
        grouped = self.meta.group(dim)
        for d in grouped.grid.outer:
            if d == dim:
                grouping[d] = d.total_var / (d.tile_var * GROUPS)
            else:
                grouping[d] = 1
        return replace(self, meta=grouped, _grouping=grouping)

    @property
    def fwd(self):
        return replace(self, meta=self.meta.fwd)

    @property
    def bwd(self):
        return replace(self, meta=self.meta.bwd)

    def eval(self, expr):
        e = expr.subs(self.subs)
        for n, f in self._functions.items():
            e = e.replace(n, f)
        return e

    def sweep(self, groups: dict[str, list[int]] | None = None, tilings: dict[str, list[int]] | None = None):

        max_tile = 1024
        max_groups = 16
        
        if groups is None:
            groups = {}
            for k in self.meta.contention_dims:
                groups[k] = list(range(1, max_groups + 1))

        if tilings is None:
            tilings = {}
            for k in self.meta.grid.outer:
                p = 1
                tilings[k] = []
                while (p <= max_tile) and (p < self._sizes[k]*2):
                    tilings[k].append(p)
                    p = p * 2
                assert tilings[k]

        i2g = list(groups)
        args = [GROUPS] + [self.dim_map[k].tile_var for k in tilings]

        scratch = [0]

        
        def chunk_inner(g):
            grouped = self.group(g)
            expressions = tuple(getattr(grouped, n) for n in SWEEP_METRICS)
            f = sympy.lambdify(args, expressions, 'numpy', cse=True)
            for e in expressions:
                free = e.free_symbols - set(args)
                assert not free
            it = itertools.product([i2g.index(g)], groups[g], *tilings.values())
            while chunk := list(itertools.islice(it, 1024)):
                scratch[0] += len(chunk)
                configs = numpy.array(chunk, dtype=numpy.int64)
                metrics = f(*(configs[:, i].astype(numpy.float64) for i in range(1, configs.shape[1])))
                metrics = numpy.stack(metrics, axis=1, dtype=numpy.float64)

                mask = numpy.all(metrics[:, FILTER_IX] >= 0, axis=1)
                metrics = metrics[mask]
                configs = configs[mask]

                mask = pareto_mask(metrics)
                metrics = metrics[mask]
                configs = configs[mask]
                yield configs, metrics
        
        def chunk_all():
            for g in i2g:
                yield from chunk_inner(g)

        parts = chunk_all()

        f_cfg, f_mtr = next(parts)
        for cfg, mtr in parts:
            cfg = numpy.vstack((f_cfg, cfg))
            mtr = numpy.vstack((f_mtr, mtr))
            mask = pareto_mask(mtr)
            f_cfg = cfg[mask]
            f_mtr = mtr[mask]
        PARAMS = ['group_dim', 'num_groups', *tilings]
        df = pl.from_numpy(numpy.hstack((f_cfg, f_mtr)))

        df.columns = [*PARAMS, *SWEEP_METRICS]
        df = df.with_columns(pl.col(p).cast(pl.Int64) for p in PARAMS)
        df = df.with_columns(pl.col('group_dim').replace_strict({i: g for i, g, in enumerate(i2g)}))
        print(scratch[0])
        print(df.select(*PARAMS, *PARETO_METRICS, 'residency_bytes'))

    @property
    def dim_map(self):
        return self.meta.grid.dim_map

    @property
    def sizes(self):
        return {self.dim_map[k].total_var: v for k, v in self._sizes.items()}

    @property
    def tiling(self):
        return {self.dim_map[k].tile_var: v for k, v in self._tiling.items()}

    @property
    def grouping(self):
        return {self.dim_map[k].group_var: v for k, v in self._grouping.items()}

    @property
    def subs(self):
        return {**self.sizes, **self.tiling, **self.grouping, **self._symbols}

sizes = dict(
    l = 128,
    r = 1024,
    h = 8,
    g = 4,
    dq = 128,
    dv = 128,
)

a100 = {
        PEAK_FLOPS: 312e12,
        BANDWIDTH: 1.55e12,
        MAX_RESIDENCY: 64*1024,
        MIN_PARALLELISM: 108,
        MIN_MMA_EFFICIENCY: 0.5,
        }

e = a.estimate(
        sizes=sizes,
        symbols=a100,
        )

def bracket(string, lb=' ', rb=None):
    if rb is None:
        return f'{lb}{string}{lb}'
    else:
        return f'{lb}{string}{rb}'

@dataclass(frozen=True)
class Formatter:
    width: int = 30

    def sep(self):
        print('-'*self.width)

    def title(self, string):
        print(f'{bracket(string):#^{self.width}}')

    def section(self, string):
        print(f'{bracket(string):=^{self.width}}')

fmt = Formatter()

fmt.sep()
fmt.title('fwd')
e.fwd.sweep()
fmt.title('bwd')
e.bwd.sweep()

print()
