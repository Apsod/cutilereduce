from dataclasses import dataclass
from enum import Enum
from copy import replace
from math import prod

from sympy import Rational, Max, Min, floor
import sympy

from .grid import BaseGrid, Dims, Grid, Dim, ConcreteDim
from .buffer import BaseBuffer, Buffer, BufferRole, Phase
from .work import BaseWork, Work, bind_work
from .config import Config
from .variables import *

@dataclass(frozen=True, kw_only=True)
class BaseSpec[D: Dim]:
    grid: BaseGrid[D]
    input: dict[str, BaseBuffer[D]]
    output: dict[str, BaseBuffer[D]]
    intermediate: tuple[BaseBuffer[D]]
    work: BaseWork[D]
    phase: Phase

    @classmethod
    def _mkhelp(
            cls,
            grid: BaseGrid[D],
            input: dict[str, Buffer], 
            output: dict[str, Buffer], 
            work: Work, 
            intermediate: list[Buffer] = None):
        input = {k: v.generic_bind(k, grid, BufferRole.Input) for k, v in input.items()}
        output = {k: v.generic_bind(k, grid, BufferRole.Output) for k, v in output.items()}
        work = bind_work(grid, work)
        intermediate=tuple(v.generic_bind(f'intermediate_{i}', grid, BufferRole.Intermediate) for i, v in enumerate(intermediate))
        return dict(
                input=input,
                output=output,
                work=work,
                intermediate=intermediate
                )

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
    def mmas(self):
        return self.work.mmas(self.phase)

    @property
    def contention_dims(self):
        return self.grid.CTYPE.union(*(b.absent for b in self.write_buffers))

    def check(self):
        self.grid.check()
        for b in self.intermediate:
            b.check()
        for b in self.io_buffers:
            b.check()

    @property
    def grouped_buffers(self):
        return tuple(b for b in self.write_buffers if self.group_dim in b.absent)
    
    ################# QUANTITIES #################

    @property
    def traffic(self):
        return sum(v.traffic(self.phase) for v in self.io_buffers)

    @property
    def effective_traffic(self):
        return self.traffic + self.contention

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
        return sum(x.total_work / x.tile_efficiency for x in self.mmas)

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
    def span_work(self):
        return sum(x.span_work for x in self.mmas)

    @property
    def arithmetic_intensity(self):
        return (self.total_work / self.traffic) / self.ridge

    @property
    def effective_intensity_ratio(self):
        return (self.effective_total_work / self.effective_traffic) / self.ridge

    @property
    def parallelism(self):
        return self.total_work / self.span_work

    @property
    def program_count(self):
        return self.grid.outer.total_prod / self.grid.outer.span_prod

    @property
    def residency_penalty(self):
        return self.resident_programs_per_sm / floor(self.resident_programs_per_sm)

    @property
    def estimated_time(self):
        return Max(self.effective_total_work / PEAK_FLOPS, self.effective_traffic / BANDWIDTH)

    @property
    def ridge(self):
        return PEAK_FLOPS / BANDWIDTH
    
    @property
    def resident_programs(self):
        return SM_COUNT * self.resident_programs_per_sm

    @property
    def resident_programs_per_sm(self):
        return Min(MAX_PROGRAMS_PER_SM, SMEM_PER_SM / self.residency_bytes)

    @property
    def active_programs(self):
        return Min(self.program_count, self.resident_programs)

    @property
    def groups(self):
        return GROUPS

    @property
    def group_size(self):
        return self.group_dim.group_var

@dataclass(frozen=True, kw_only=True)
class Spec(BaseSpec[Dim]):
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

        kwargs = cls._mkhelp(grid=grid, input=input, output=output, intermediate=intermediate, work=work)

        ret = cls(
                grid=grid,
                phase = Phase.fwd,
                group_dim = grid.fold[0],
                **kwargs
                )

        ret.check()
        return ret

    def group(self, dim: str | None = None):
        if dim is None:
            dim = self.contention_dims[0]
        else:
            dim = self.contention_dims.get(dim)
        return replace(self, group_dim=dim)

    @property
    def fwd(self):
        return replace(self, phase=Phase.fwd).group()

    @property
    def bwd(self):
        return replace(self, phase=Phase.bwd).group()

    @property
    def traffic_lower_bound(self):
        return self.traffic.subs(self.full_span)

    @property
    def grouping(self):
        grouping = {}
        for d in self.grid.outer:
            if d == self.group_dim:
                grouping[d.group_var] = d.total_var / (d.tile_var * GROUPS)
            else:
                grouping[d.group_var] = 1
        return grouping

    @property
    def traffic_ratio(self):
        return self.traffic / self.traffic_lower_bound
    
    @property
    def full_span(self):
        return {
            d.group_var: d.total_var / d.tile_var
            for d in self.grid.outer
        }


@dataclass(frozen=True, kw_only=True)
class ConcreteSpec(BaseSpec[ConcreteDim]):
    @classmethod
    def make(cls, 
             input: dict[str, Buffer], 
             output: dict[str, Buffer], 
             batch: Dims, 
             fold: Dims, 
             work: Work, 
             config: Config,
             intermediate: list[Buffer] = None,
             ):

        if intermediate is None:
            intermediate = []

        grid = Grid.make(
                input = input,
                output = output,
                batch = batch,
                fold = fold,
                ).concretize(config)

        kwargs = cls._mkhelp(grid=grid, input=input, output=output, intermediate=intermediate, work=work)

        return cls(
                grid=grid,
                phase=config.phase,
                **kwargs,
                )
    
    def _eval(self, expr):
        return self.config._eval(expr)

    @property
    def config(self):
        return self.grid.config

    @property
    def group_dim(self):
        return self.grid.dim.get(self.config.group_dim)

    @property
    def estimated_time(self):
        return self._eval(Max(self.effective_total_work / PEAK_FLOPS, self.effective_traffic / BANDWIDTH))

    @property
    def ridge(self):
        return self._eval(PEAK_FLOPS / BANDWIDTH)
    
    @property
    def resident_programs(self):
        return self._eval(SM_COUNT * self.resident_programs_per_sm)

    @property
    def resident_programs_per_sm(self):
        return self._eval(Min(MAX_PROGRAMS_PER_SM, SMEM_PER_SM / self.residency_bytes))

    @property
    def active_programs(self):
        return self._eval(Min(self.program_count, self.resident_programs))

    @property
    def groups(self):
        return self._eval(GROUPS)

