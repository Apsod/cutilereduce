from dataclasses import dataclass
from copy import replace

import sympy
from sympy import Max, Min

import cuda.tile as ct

from .base import Phase
from .grid import BaseGrid, Dims, Grid, Dim, ConcreteDim
from .buffer import BaseBuffer, Buffer, BufferRole
from .work import BaseWork, Work, bind_work
from .config import Config
from .variables import READ, WRITE, GROUPS, PEAK_FLOPS, BANDWIDTH, FWD_CONTENTION, BWD_CONTENTION, SM_COUNT, MAX_PROGRAMS_PER_SM, SMEM_PER_SM

@dataclass(frozen=True, kw_only=True)
class BaseSpec[D: Dim]:
    grid: BaseGrid[D]
    input: dict[str, BaseBuffer[D]]
    execution: dict[str, BaseBuffer[D]]
    output: dict[str, BaseBuffer[D]]
    intermediate: tuple[BaseBuffer[D]]
    work: BaseWork[D]
    phase: Phase

    @classmethod
    def _mkhelp(
            cls,
            grid: BaseGrid[D],
            input: dict[str, Buffer], 
            execution: dict[str, Buffer], 
            output: dict[str, Buffer], 
            work: Work, 
            intermediate: list[Buffer] = None):
        input = {k: v.generic_bind(k, i, grid, BufferRole.Input) for (i, (k, v)) in enumerate(input.items())}
        execution = {k: v.generic_bind(k, i, grid, BufferRole.Intermediate) for (i, (k, v)) in enumerate(execution.items())}
        output = {k: v.generic_bind(k, i, grid, BufferRole.Output) for (i, (k, v)) in enumerate(output.items())}
        intermediate = tuple(v.generic_bind(f'intermediate:{i}', i, grid, BufferRole.Intermediate) for i, v in enumerate(intermediate))

        return dict(
                input=input,
                execution=execution,
                output=output,
                work=bind_work(grid, work),
                intermediate=intermediate
                )


    @property
    def output_buffers(self):
        return (*self.output.values(),)

    @property
    def input_buffers(self):
        return (*self.input.values(),)

    @property
    def execution_buffers(self):
        return (*self.execution.values(),)

    @property
    def intermediate_buffers(self):
        return (*self.execution.values(), *self.intermediate)

    @property
    def output_storage(self):
        return sum(b.total_bytes for b in self.output_buffers)

    @property
    def checkpoint_storage(self):
        return sum((self.groups - 1) * b.total_bytes for b in self.output_buffers)

    @property
    def input_storage(self):
        return sum(b.total_bytes for b in self.input_buffers)

    @property
    def excess_storage_ratio(self):
        base = self.input_storage + self.output_storage
        extra = self.checkpoint_storage
        return extra / base

    @property
    def grad_buffers(self):
        return tuple(replace(b, dtype=ct.float32, default=0) for b in self.input_buffers if b.req_grad)


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

    @property
    def batch_input_buffers(self):
        return tuple(b for b in self.input_buffers if self.group_dim in b.absent)

    @property
    def fold_input_buffers(self):
        return tuple(b for b in self.input_buffers if self.group_dim in b.dims)

    @property
    def batch_grad_buffers(self):
        return tuple(b for b in self.grad_buffers if self.group_dim in b.absent)

    @property
    def fold_grad_buffers(self):
        return tuple(b for b in self.grad_buffers if self.group_dim in b.dims)

    @property
    def batch_buffers(self):
        return tuple(b for b in self.read_buffers if self.group_dim in b.absent)

    @property
    def fold_buffers(self):
        return tuple(b for b in self.read_buffers if self.group_dim in b.dims)
    
    ################# QUANTITIES #################

    @property
    def C(self):
        match self.phase:
            case Phase.fwd: return self.FWD_CONTENTION # noqa: E701
            case Phase.bwd: return self.BWD_CONTENTION # noqa: E701


    def buffer_contention(self, b):
        if not b.is_write(self.phase):
            return 0
        R = b.residual_multiplicity
        P = self.program_count
        A = self.active_programs
        active_multiplicity = Min(R, 1 + (A - 1) * (R - 1) / Max(1, P-1))
        return b.accessed_bytes * self.C(active_multiplicity)

    def buffer_traffic(self, b):
        kind = 0
        if b.is_write(self.phase):
            kind += self.WRITE
        if b.is_read(self.phase):
            kind += self.READ
        return kind * b.accessed_bytes

    @property
    def traffic(self):
        return sum(self.buffer_traffic(v) for v in self.io_buffers)

    @property
    def effective_traffic(self):
        return self.traffic + self.contention

    @property
    def tile_bytes(self):
        return sum(v.tile_bytes for v in self.io_buffers + self.intermediate)

    @property
    def residency_bytes(self):
        base = sum(v.tile_bytes for v in self.intermediate + self.batch_buffers)

        if self.phase == Phase.bwd:
            base += sum(v.tile_bytes for v in self.grad_buffers if self.group_dim in
            v.absent)

        return base


    @property
    def contention(self):
        return sum(self.buffer_contention(v) for v in self.io_buffers)

    @property
    def expected_group_active_programs(self):
        return 1 + (self.active_programs - 1) / self.groups

    @property
    def total_work(self):
        return sum(x.total_work for x in self.mmas)

    @property
    def smem_utilization(self):
        resident_smem_per_sm = self.resident_programs_per_sm * self.residency_bytes
        return resident_smem_per_sm / self.SMEM_PER_SM

    @property
    def effective_total_work(self):
        return sum(x.total_work / x.tile_efficiency for x in self.mmas)

    @property
    def effective_tile_work(self):
        return sum(x.tile_work / x.tile_efficiency for x in self.mmas)

    @property
    def compute_time(self):
        return self.effective_total_work / self.effective_peak_flops

    @property
    def traffic_time(self):
        return self.effective_traffic / self.effective_bandwidth

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
    def estimated_time(self):
        return Max(self.compute_time, self.traffic_time)

    @property
    def ridge(self):
        return self.PEAK_FLOPS / self.BANDWIDTH
    
    @property
    def resident_programs(self):
        return self.SM_COUNT * self.resident_programs_per_sm

    @property
    def resident_programs_per_sm(self):
        return Min(self.MAX_PROGRAMS_PER_SM, self.SMEM_PER_SM // self.residency_bytes)

    @property
    def active_programs(self):
        return Min(self.program_count, self.resident_programs)

    @property
    def SM_utilization(self):
        return Min(1, self.active_programs / self.SM_COUNT) 

    @property
    def effective_peak_flops(self):
        return self.PEAK_FLOPS * self.SM_utilization

    @property
    def effective_bandwidth(self):
        return self.BANDWIDTH * self.SM_utilization


    @property
    def groups(self):
        return self.GROUPS

    @property
    def group_size(self):
        return self.group_dim.group_var

    #########################

    @property
    def PEAK_FLOPS(self):
        return PEAK_FLOPS

    @property
    def BANDWIDTH(self):
        return BANDWIDTH

    @property
    def SM_COUNT(self):
        return SM_COUNT

    @property
    def MAX_PROGRAMS_PER_SM(self):
        return MAX_PROGRAMS_PER_SM

    @property
    def SMEM_PER_SM(self):
        return SMEM_PER_SM

    @property
    def READ(self):
        return READ

    @property
    def WRITE(self):
        return WRITE

    @property
    def GROUPS(self):
        return GROUPS

    @property
    def FWD_CONTENTION(self):
        return FWD_CONTENTION

    @property
    def BWD_CONTENTION(self):
        return BWD_CONTENTION


@dataclass(frozen=True, kw_only=True)
class Spec(BaseSpec[Dim]):
    group_dim: Dim

    @classmethod
    def make(cls, 
             input: dict[str, Buffer], 
             execution: dict[str, Buffer],
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

        kwargs = cls._mkhelp(grid=grid, input=input, execution=execution, output=output, intermediate=intermediate, work=work)

        ret = cls(
                grid=grid,
                phase = Phase.fwd,
                group_dim = grid.fold[0],
                **kwargs
                )

        ret.check()
        return ret

    def concretize(self, config: Config):
        input = {k: v.base for k, v in self.input.items()}
        execution = {k: v.base for k, v in self.execution.items()}
        output = {k: v.base for k, v in self.output.items()}
        batch = self.grid.base.batch
        fold = self.grid.base.fold
        work = self.work.base
        intermediate = [b.base for b in self.intermediate]
        return ConcreteSpec.make(
                input,
                execution,
                output,
                batch,
                fold,
                work,
                config,
                intermediate,
                )


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
             execution: dict[str, Buffer], 
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
                config = config,
                )

        kwargs = cls._mkhelp(grid=grid, input=input, execution=execution, output=output, intermediate=intermediate, work=work)

        return cls(
                grid=grid,
                phase=config.phase,
                **kwargs,
                )

    @property
    def config(self):
        return self.grid.config
    
    def _eval(self, expr):
        return self.config._eval(expr)

    @property
    def group_dim(self):
        return self.grid.group_dim

    @property
    def groups(self):
        return self.config.num_groups

    def eval(self, name):
        value = getattr(self, name)
        if isinstance(value, sympy.Expr):
            return self._eval(value)
        else:
            return value
