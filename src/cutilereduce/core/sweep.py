from __future__ import annotations 
from dataclasses import dataclass
from typing import Any, ClassVar
import itertools

from copy import replace
from sympy import Function, Max, Min, Symbol, log
import sympy
import polars as pl
import numpy

from .base import kmap, Phase
from .grid import Dim
from .spec import Spec
from .variables import READ, WRITE, GROUPS, MIN_MMA_EFFICIENCY, FWD_CONTENTION, BWD_CONTENTION
from .config import Config

def contention_penalty(alpha, cap, contention):
    return Min(cap, alpha * log(Max(1, contention), 2))

def atomic_add_penalty(contention):
    return contention_penalty(alpha=1, cap=16, contention=contention)

def spinlock_penalty(contention):
    return contention_penalty(alpha=2, cap=64, contention=contention)

def group_penalty(contention):
    return 0

DEFAULT_SYMBOLS = {
        READ: 1, 
        WRITE: 1,
        MIN_MMA_EFFICIENCY: 1,
        }

DEFAULT_FUNCTIONS = {
        FWD_CONTENTION: group_penalty,
        BWD_CONTENTION: atomic_add_penalty,
        }

def pareto_mask(metrics):
    keep = numpy.ones(metrics.shape[0], dtype=bool)
    for i in range(metrics.shape[0]):
        if not keep[i]:
            continue

        cmp_lt = metrics < metrics[i]
        cmp_eq = metrics == metrics[i]
        cmp_gt = metrics > metrics[i]

        dominates_i = numpy.all(cmp_lt | cmp_eq, axis=1) & numpy.any(cmp_lt, axis=1)
        i_dominates = numpy.all(cmp_gt | cmp_eq, axis=1) & numpy.any(cmp_gt, axis=1)
        keep[i_dominates] = False
        if numpy.any(dominates_i):
            keep[i] = False
    return keep

@dataclass(frozen=True)
class Sweep:
    attributes: list[str]
    filters: list[pl.Expr]
    paretos: list[pl.Expr]
    sort: list[pl.Expr]

    default: ClassVar[Sweep]

    def filter(self, df):
        if self.filters:
            return df.filter(self.filters)
        else:
            return df

    def pareto(self, df):
        if self.paretos:
            mask = pareto_mask(df.select(*self.paretos).cast(pl.Float64).to_numpy())
            return df.filter(mask)
        else:
            return df

    def pareto_combine(self, a, b=None):
        if b is None:
            return a
        else:
            return self.pareto(pl.concat((a, b)))

    def apply(self, estimator: Estimator):
        frontier = None
        for df in estimator.chunked_eval(self.attributes, *estimator.generate_configs()):
            df = self.filter(df)
            df = self.pareto(df)
            frontier = self.pareto_combine(df, frontier)
        if self.sort:
            frontier = frontier.sort(self.sort)
        return frontier

    def run_all(self, estimator: Estimator):
        return pl.concat((
            self.apply(estimator.fwd),
            self.apply(estimator.bwd),
            ))

Sweep.default = Sweep(
            attributes = [
                'estimated_time', 'compute_time', 'traffic_time', 
                'excess_storage_ratio', 'effective_tile_work',
                'mma_efficiency', 'resident_programs_per_sm', 'group_size', 
                'residency_bytes', 'groups', 
                'SM_utilization',
                ],
            filters = [
                pl.col('resident_programs_per_sm') >= 1,
                pl.col('group_size') >= 1,
                pl.col('mma_efficiency') == 1,
                pl.when(pl.col('cfg:phase') == 'forward').then(pl.col('groups') == 1).otherwise(pl.col('groups') >= 1),
                ],
            paretos = [
                'compute_time', 'traffic_time',
                ],
            sort = ['estimated_time', pl.col('SM_utilization').neg(), 'residency_bytes', pl.col('effective_tile_work').neg()]
            )


@dataclass(frozen=True)
class Estimator:
    meta: Spec
    sizes: dict[Dim, int]
    symbols: dict[Symbol, int]
    functions: dict[Function, Any]

    @classmethod
    def make(
            cls, 
            meta: Spec, 
            sizes: dict[str, int], 
            symbols: dict[Symbol, int] = {},
            functions: dict[Symbol, Any] = {},
            ):
        
        return cls(
                meta = meta,
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
        expr = expr.subs(self.meta.grouping) # Can produce tile_vars.
        expr = expr.subs(sizes | self.symbols)
        return expr

    def result2cfg(self, df):
        configs = df.select(pl.selectors.starts_with('cfg:')).rename(lambda x: x[4:])
        names = configs.columns 
        for row in configs.iter_rows():
            cfg = {k: v for k, v in zip(names, row)}
            yield Config(
                    phase = Phase(cfg['phase']),
                    group_dim = str(cfg['group']),
                    num_groups = cfg['num_groups'],
                    total = {str(k): v for k, v in self.sizes.items()},
                    tiling = {str(k): cfg[k] for k in self.meta.grid.outer},
                    symbols = self.symbols,
                    functions = self.functions,
                    )


    def chunked_eval(self, attributes: list[str], groups: dict[str, list[int]], tilings: dict[str, list[int]]):
        i2g = list(groups)
        args = [GROUPS] + [k.tile_var for k in tilings]
        params = ['num_groups', *tilings]

        def chunk_inner(g):
            grouped = self.group(g)
            expressions = tuple(grouped._eval(getattr(grouped.meta, n)) for n in attributes)
            f = sympy.lambdify(args, expressions, 'numpy', cse=True)
            for e in expressions:
                free = e.free_symbols - set(args)
                assert not free, free
            it = itertools.product(groups[g], *tilings.values())
            while chunk := list(itertools.islice(it, 1024)):
                cols = list(zip(*chunk))
                configs = pl.DataFrame({'cfg:phase': self.meta.phase, 'cfg:group': g, **{f'cfg:{k}': v for k,v in zip(params, cols)}})
                metrics = f(*(numpy.array(col, dtype=numpy.float128) for col in cols))
                metrics = pl.DataFrame({n: v.astype(numpy.float64) for n, v in zip(attributes, metrics)})
                yield pl.concat((configs, metrics), how='horizontal')

        for g in i2g:
            yield from chunk_inner(g)

    def generate_configs(self, max_tile=1024, max_groups=1):
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

    def evaluate(self, attributes: list[str], groups: dict[str, list[int]], tilings: dict[str, list[int]]):
        return pl.concat(self.chunked_eval(attributes, groups, tilings))
