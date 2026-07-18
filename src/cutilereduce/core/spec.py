from dataclasses import dataclass
from copy import replace

import sympy
from sympy import Max, Min, Piecewise


from .base import Phase
from .grid import BaseGrid, Dims, Grid, Dim, ConcreteDim
from .buffer import Buffer, BufferRole, BufferBundle
from .work import BaseWork, Work, bind_work
from .config import Config
from .variables import READ, WRITE, GROUPS, PEAK_FLOPS, BANDWIDTH, FWD_CONTENTION, BWD_CONTENTION, SM_COUNT, MAX_PROGRAMS_PER_SM, SMEM_PER_SM, ATOMIC_ADD

@dataclass(frozen=True, kw_only=True)
class BaseSpec[D: Dim]:
    grid: BaseGrid[D]
    input: BufferBundle[D]
    execution: BufferBundle[D]
    output: BufferBundle[D]
    grad_accumulator : BufferBundle[D]
    intermediate: BufferBundle[D]
    work: BaseWork[D]
    phase: Phase

    @classmethod
    def _mkhelp(
            cls,
            grid: BaseGrid[D],
            input: dict[str, Buffer], 
            execution: dict[str, Buffer], 
            output: dict[str, Buffer], 
            grad_accumulator: dict[str, Buffer],
            work: Work, 
            intermediate: list[Buffer] = None):
        input = BufferBundle(tuple(
            v.generic_bind(k, i, grid, BufferRole.Input) 
            for (i, (k, v)) 
            in enumerate(input.items())
            ))
        execution = BufferBundle(tuple(
            v.generic_bind(k, i, grid, BufferRole.State) 
            for (i, (k, v)) 
            in enumerate(execution.items())
            ))
        output = BufferBundle(tuple(
            v.generic_bind(k, i, grid, BufferRole.Output) 
            for (i, (k, v)) 
            in enumerate(output.items())
            ))
        grad_accumulator = BufferBundle(tuple(
            v.generic_bind(k, i, grid, BufferRole.State) 
            for (i, (k, v)) 
            in enumerate(grad_accumulator.items())
            ))

        intermediate = BufferBundle(tuple(
            v.generic_bind(f'intermediate:{i}', i, grid, BufferRole.Intermediate)
            for i, v 
            in enumerate(intermediate)
            ))

        return dict(
                input=input,
                execution=execution,
                output=output,
                grad_accumulator=grad_accumulator,
                work=bind_work(grid, work),
                intermediate=intermediate
                )

    @property
    def group_dim(self):
        return self.grid.group_dim

    @property
    def output_storage(self):
        return self.output.total_bytes

    @property
    def checkpoint_storage(self):
        return self.output.total_bytes * (self.groups - 1)

    @property
    def input_storage(self):
        return self.input.total_bytes

    @property
    def excess_storage_ratio(self):
        base = self.input_storage + self.output_storage
        extra = self.checkpoint_storage
        return extra / base

    @property
    def grad(self):
        return self.input.grad

    @property
    def read_buffers(self):
        read = self.input
        if self.phase == Phase.bwd:
            read = read + self.output + self.output
        return read

    @property
    def write_buffers(self):
        match self.phase:
            case Phase.fwd:
                return self.output
            case Phase.bwd:
                return self.grad

    @property
    def mmas(self):
        return self.work.mmas(self.phase)

    @property
    def contention_dims(self):
        return self.grid.CTYPE.union(*(b.absent for b in self.write_buffers))

    def check(self):
        self.grid.check()
        bundles = [self.intermediate, self.output, self.execution, self.grad_accumulator]
        for bundle in bundles:
            bundle.check()

    @property
    def state(self):
        match self.phase:
            case Phase.fwd:
                return self.execution
            case Phase.bwd:
                return self.grad_accumulator

    @property
    def streamed_resident_buffers(self):
        buffers = self.input.fold + self.state.fold
        match self.phase:
            case Phase.bwd:
                buffers += self.grad.fold
        return buffers

    @property
    def persistent_resident_buffers(self):
        buffers = self.input.batch + self.state.batch + self.intermediate
        match self.phase:
            case Phase.bwd:
                buffers += self.grad.batch
        return buffers

    @property
    def resident_buffers(self):
        return self.persistent_resident_buffers + self.streamed_resident_buffers

    ################# QUANTITIES #################

    @property
    def C(self):
        match self.phase:
            case Phase.fwd: return self.FWD_CONTENTION # noqa: E701
            case Phase.bwd: return self.BWD_CONTENTION # noqa: E701


    def buffer_contention(self, b):
        writers_per_tile = b.residual_multiplicity
        active_writers_per_tile = self.active_programs / b.target_tiles
        active_multiplicity = Min(writers_per_tile, Max(1, active_writers_per_tile))
        return b.accessed_bytes * self.C(active_multiplicity)

    @property
    def atomic_add_penalty(self):
        match self.phase:
            case Phase.fwd:
                return self.ATOMIC_ADD * 0 # sympy hack
            case Phase.bwd:
                return self.ATOMIC_ADD * sum(
                        b.accessed_bytes * Piecewise((0, b.residual_multiplicity <= 1), (1, True))
                        for b in self.write_buffers
                        )
    
    @property
    def atomic_ops(self):
        return (b.accessed_elems for b in self.write_buffers if b.residual_multiplicity > 1)


    @property
    def traffic(self):
        return (
                self.READ * sum(b.accessed_bytes for b in self.read_buffers) +
                self.WRITE * sum(b.accessed_bytes for b in self.write_buffers)
                )

    @property
    def effective_traffic(self):
        return self.traffic + self.atomic_add_penalty + self.contention

    @property
    def residency_bytes(self):
        return sum(b.tile_bytes for b in self.resident_buffers)

    @property
    def pipeline_bytes(self):
        return sum(b.tile_bytes for b in self.streamed_resident_buffers)

    @property
    def pipeline_stage_capacity(self):
        spare = self.SMEM_PER_SM - self.resident_programs_per_sm * self.residency_bytes
        denom = self.resident_programs_per_sm * Max(1, self.pipeline_bytes)
        slack = spare / denom
        return Piecewise(
                (0, self.pipeline_bytes <= 0),
                (slack, True),
                )
    @property
    def pipeline_factor(self):
        return Min(1, self.pipeline_stage_capacity * self.stream_compute_cover)

    @property
    def contention(self):
        return sum(self.buffer_contention(b) for b in self.write_buffers)

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
    def nonhiding_traffic(self):
        traffic = 0
        traffic += self.READ * self.read_buffers.batch.accessed_bytes
        traffic += self.WRITE * self.write_buffers.accessed_bytes
        #traffic += self.contention
        return traffic

    @property
    def streamed_traffic(self):
        return self.READ * self.read_buffers.fold.accessed_bytes

    @property
    def streamed_bytes_per_tile(self):
        return self.read_buffers.fold.tile_bytes
    
    @property
    def stream_compute_cover(self):
        return (self.effective_tile_work / Max(1, self.streamed_bytes_per_tile)) / self.ridge

    @property
    def traffic_time(self):
        return self.effective_traffic / self.effective_bandwidth

    @property
    def mma_efficiency(self):
        return self.total_work / self.effective_total_work

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
        nonh_time = self.nonhiding_traffic / self.effective_bandwidth
        stream_time = self.streamed_traffic / self.effective_bandwidth
        return nonh_time + stream_time + Max(0, self.compute_time - stream_time * self.pipeline_factor)

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
    def smem_budget(self):
        return self.SMEM_PER_SM / self.resident_programs_per_sm

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
    def ATOMIC_ADD(self):
        return ATOMIC_ADD

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
    @classmethod
    def make(cls, 
             input: dict[str, Buffer], 
             execution: dict[str, Buffer],
             output: dict[str, Buffer], 
             grad_accumulator: dict[str, Buffer],
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
                scan = Dims.empty()
                )

        kwargs = cls._mkhelp(grid=grid, input=input, execution=execution, output=output, grad_accumulator=grad_accumulator, intermediate=intermediate, work=work)

        ret = cls(
                grid=grid,
                phase = Phase.fwd,
                **kwargs
                )

        ret.check()
        return ret

    def concretize(self, config: Config):
        input = self.input.base
        execution = self.execution.base
        output = self.output.base
        grad_accumulator = self.grad_accumulator.base
        batch = self.grid.base.batch
        fold = self.grid.base.fold
        work = self.work.base
        intermediate = [b.base for b in self.intermediate]
        return ConcreteSpec.make(
                input=input,
                execution=execution,
                output=output,
                grad_accumulator=grad_accumulator,
                batch=batch,
                fold=fold,
                work=work,
                config=config,
                intermediate=intermediate,
                )


    def group(self, dim: str | None = None):
        if dim is None:
            dim = self.contention_dims[0]
        else:
            dim = self.contention_dims.get(dim)
        return replace(self, grid=self.grid.group(dim))

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
             grad_accumulator: dict[str, Buffer],
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
                scan = Dims.empty(),
                config = config,
                )

        kwargs = cls._mkhelp(grid=grid, input=input, execution=execution, output=output, grad_accumulator=grad_accumulator, intermediate=intermediate, work=work)

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
    def groups(self):
        return self.config.num_groups

    def eval(self, name):
        value = getattr(self, name)
        if isinstance(value, sympy.Expr):
            return self._eval(value)
        else:
            return value

    @property
    def heuristic_layout(self):
        read = {}
        write = {}
        for d in self.grid.outer:
            read[d] = self.read_buffers.without(d).accessed_bytes
            write[d] = (
                self.write_buffers.without(d)
                .filter(lambda b: b.residual_multiplicity > 1)
                .accessed_bytes
            )
        alpha = 0.5
        return tuple((*sorted(self.grid.outer, key=lambda d: read[d] - alpha * write[d]), *self.grid.inner))
        #alphas = [0.5]
        #for alpha in alphas:
        #    optimals.add(tuple(str(d) for d in sorted(self.grid.outer, key=lambda d: read[d] - write[d]*alpha)))
        #return optimals


