from dataclasses import dataclass
from typing import Any
import itertools

from copy import replace
from sympy import Function, Max, Min, Symbol, log
import sympy
import polars as pl
import numpy

from .base import kmap
from .grid import Dim
from .spec import Meta
from .variables import * 

def contention_penalty(alpha, cap, contention):
    return Min(cap, alpha * log(Max(1, contention), 2))

def atomic_add_penalty(contention):
    return contention_penalty(alpha=0.5, cap=8, contention=contention)

def spinlock_penalty(contention):
    return contention_penalty(alpha=2, cap=64, contention=contention)

DEFAULT_SYMBOLS = {
        READ: 1, 
        WRITE: 1,
        }

DEFAULT_FUNCTIONS = {
        FWD_CONTENTION: spinlock_penalty,
        BWD_CONTENTION: atomic_add_penalty,
        }

@dataclass(frozen=True)
class SweepMetrics:
    pareto: list[str]
    filter: list[str]
    other: list[str]
    
    @property
    def sweep(self):
        return sorted(
                set(self.pareto) |
                set(self.filter) |
                set(self.other) 
                )
    @property
    def pareto_ix(self):
        return [self.sweep.index(n) for n in self.pareto]

    @property
    def filter_ix(self):
        return [self.sweep.index(n) for n in self.filter]

DEFAULT_METRICS = SweepMetrics(
        pareto = ['mma_penalty', 'effective_roofline'],
        filter = ['parallelism_filter', 'residency_filter', 'mma_efficiency_filter', 'overgrouping_filter'],
        other = ['residency_bytes', 'arithmetic_intensity', 'tile_work', 'traffic'],
        )


def pareto_mask(metrics):
    keep = numpy.ones(metrics.shape[0], dtype=bool)
    for i in range(metrics.shape[0]):
        if not keep[i]:
            continue

        dominates_i = numpy.all(metrics <= metrics[i], axis=1) & numpy.any(metrics < metrics[i], axis=1)
        if numpy.any(dominates_i):
            keep[i] = False
            continue

        i_dominates = numpy.all(metrics[i] <= metrics, axis=1) & numpy.any(metrics[i] < metrics, axis=1)
        keep[i_dominates] = False
    return keep

@dataclass(frozen=True)
class Result:
    metrics: SweepMetrics
    params: list[str]
    df: pl.DataFrame

    @property
    def configs(self):
        return self.df.select(self.params)

    def get(self, *columns, with_params=True):
        cols = self.params + columns if with_params else columns
        return self.df.select(cols)

@dataclass(frozen=True)
class Estimator:
    meta: Meta
    metrics: SweepMetrics
    sizes: dict[Dim, int]
    symbols: dict[Symbol, int]
    functions: dict[Function, Any]
    grouping: dict[Dim, int] | None = None

    def __getattr__(self, name):
        value = getattr(self.meta, name)
        if isinstance(value, sympy.Expr):
            return self._eval(value)
        else:
            raise AttributeError(f'{name} is not a symbolic quantity of meta')


    @classmethod
    def make(
            cls, 
            meta: Meta, 
            sizes: dict[str, int], 
            metrics: SweepMetrics | None = None,
            symbols: dict[Symbol, int] = {},
            functions: dict[Symbol, Any] = {},
            ):
        
        return cls(
                meta = meta,
                metrics = DEFAULT_METRICS if metrics is None else metrics,
                sizes = {meta.grid.dim_map[k]: v for k, v in sizes.items()},
                symbols = DEFAULT_SYMBOLS | symbols,
                functions = DEFAULT_FUNCTIONS | functions,
                )

    def group(self, dim: str):
        return replace(self, meta=self.meta.group(dim))

    @property
    def fwd(self):
        return replace(self, meta=self.meta.fwd)

    @property
    def bwd(self):
        return replace(self, meta=self.meta.bwd)

    def _eval(self, expr):
        sizes = kmap(lambda k: k.total_var, self.sizes)
        # functions and grouping can introduce new symbols. Do them first
        # sizes and symbols are assumed to be constants
        for n, f in self.functions.items():
            expr = expr.replace(n, f)
        expr = expr.subs(self.meta.grouping)
        expr = expr.subs(sizes | self.symbols)
        return expr

    def chunked_eval(self, groups: dict[str, int], tilings: dict[str, int]):
        i2g = list(groups)
        args = [GROUPS] + [k.tile_var for k in tilings]

        def chunk_inner(g):
            grouped = self.group(g)
            expressions = tuple(getattr(grouped, n) for n in self.metrics.sweep)
            f = sympy.lambdify(args, expressions, 'numpy', cse=True)
            for e in expressions:
                free = e.free_symbols - set(args)
                assert not free, free
            it = itertools.product([i2g.index(g)], groups[g], *tilings.values())
            while chunk := list(itertools.islice(it, 1024)):
                configs = numpy.array(chunk, dtype=numpy.int64)
                metrics = f(*(configs[:, i].astype(numpy.float64) for i in range(1, configs.shape[1])))
                metrics = numpy.stack(metrics, axis=1, dtype=numpy.float64)
                yield configs, metrics


        for g in i2g:
            yield from chunk_inner(g)

    def filter(self, configs, metrics):
        mask = numpy.all(metrics[:, self.metrics.filter_ix] >= 0, axis=1)
        return configs[mask], metrics[mask]

    def pareto(self, configs, metrics):
        mask = pareto_mask(metrics[:, self.metrics.pareto_ix])
        return configs[mask], metrics[mask]

    def pareto_combine(self, a, b=None):
        if b is None:
            return a
        a_c, a_m = a
        b_c, b_m = b
        configs = numpy.vstack((a_c, b_c))
        metrics = numpy.vstack((a_m, b_m))
        return self.pareto(configs, metrics)

    def generate_configs(self, max_tile=1024, max_groups=16):
        groups = {}
        for k in self.meta.contention_dims:
            groups[k] = list(range(1, max_groups + 1))
            assert groups[k]

        tilings = {}
        for k in self.meta.grid.outer:
            p = 1
            tilings[k] = []
            while (p <= max_tile) and (p < self.sizes[k]*2):
                tilings[k].append(p)
                p = p * 2
            assert tilings[k]

        return groups, tilings

    def format_results(self, configs, metrics, groups, tilings):
        i2g = list(groups)
        params = ['group_dim', 'num_groups', *tilings]
        df = pl.from_numpy(numpy.hstack((configs, metrics)))
        df.columns = [*params, *self.metrics.sweep]
        df = df.with_columns(pl.col(p).cast(pl.Int64) for p in params)
        df = df.with_columns(pl.col('group_dim').replace_strict({i: g for i, g, in enumerate(i2g)}))
        return Result(params, self.metrics, df)

    def evaluate(self, groups: dict[str, list[int]], tilings: dict[str, list[int]], filter=True):
        parts = []
        for configs, metrics in self.chunked_eval(groups, tilings):
            if filter:
                parts.append(self.filter(configs, metrics))
            else:
                parts.append((configs, metrics))
        
        configs, metrics = zip(*parts)

        configs = numpy.vstack(configs)
        metrics = numpy.vstack(metrics)
        return self.format_results(configs, metrics, groups, tilings)

    def pareto_sweep(self, max_tile=1024, max_groups=16, filter=True):
        groups, tilings = self.generate_configs(max_tile, max_groups)

        frontier = None
        for configs, metrics in self.chunked_eval(groups, tilings):
            if filter:
                configs, metrics = self.filter(configs, metrics)
            frontier = self.pareto_combine(
                    self.pareto(configs, metrics), 
                    frontier
                    )

        configs, metrics = frontier

        return self.format_results(configs, metrics, groups, tilings)
