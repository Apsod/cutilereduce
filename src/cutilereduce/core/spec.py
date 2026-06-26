from dataclasses import dataclass
from enum import Enum
from copy import replace
from math import prod

from sympy import Rational, Max, Min

from .grid import BoundGrid, BoundDims, Dims, Grid, Dim
from .buffer import BoundBuffer, Buffer, BufferRole, Phase
from .work import BoundWork, Work
from .variables import *

@dataclass(frozen=True)
class Meta:
    grid: BoundGrid
    input: dict[str, BoundBuffer]
    output: dict[str, BoundBuffer]
    intermediate: tuple[BoundBuffer]
    work: BoundWork
    phase: Phase
    group_dim: Dim

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

        ret = cls(
                input={k: v.bind(k, grid, BufferRole.Input) for k, v in input.items()},
                output={k: v.bind(k, grid, BufferRole.Output)for k, v in output.items()},
                intermediate=tuple(v.bind(f'intermediate_{i}', grid, BufferRole.Intermediate) for i, v in enumerate(intermediate)),
                grid=grid,
                work=work.bind(grid),
                phase = Phase.fwd,
                group_dim = grid.fold[0],
                )
        ret.check()
        return ret

    def group(self, dim: str | None = None):
        if dim is None:
            dim = self.contention_dims[0]
        elif dim not in self.contention_dims:
            raise ValueError(f'dim {dim} not in contention dims: {self.contention_dims}')
        return replace(self, group_dim=dim)

    @property
    def fwd(self):
        return replace(self, phase=Phase.fwd).group()

    @property
    def bwd(self):
        return replace(self, phase=Phase.bwd).group()
    
    @property
    def full_span(self):
        return {
            d.group_var: d.total_var / d.tile_var
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
        return tuple(b for b in self.io_buffers if b.is_read(self.phase))

    @property
    def write_buffers(self):
        return tuple(b for b in self.io_buffers if  b.is_write(self.phase))

    @property
    def grouped_buffers(self):
        return tuple(b for b in self.write_buffers if self.group_dim in b.absent)

    @property
    def contention_dims(self):
        return BoundDims.union(*(b.absent for b in self.write_buffers))

    @property
    def mmas(self):
        return self.work.mmas(self.phase)

    def check(self):
        self.grid.check()
        for b in self.intermediate:
            b.check()
        for b in self.io_buffers:
            b.check()

    @property
    def grouping(self):
        grouping = {}
        for d in self.grid.outer:
            if d == self.group_dim:
                grouping[d.group_var] = d.total_var / (d.tile_var * GROUPS)
            else:
                grouping[d.group_var] = 1
        return grouping

    ###################### SYMBOLIC QUANTITIES ####################### 
    
    @property
    def traffic(self):
        return sum(v.traffic(self.phase) for v in self.io_buffers)

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
        return sum(v.contention(self.phase) for v in self.io_buffers)

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
    def overgrouping_filter(self):
        return self.group_dim.group_var - 1

    @property
    def mma_efficiency_filter(self):
        return Min(*(x.tile_efficiency_bottleneck for x in self.mmas)) - MIN_MMA_EFFICIENCY
