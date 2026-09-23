from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable
from fractions import Fraction

from . import symbolic as expr
from .symbolic import Max, Min, Piecewise, SaturateSub, floor

from .stage_buffer import KernelBuffers
from .stage_domain import StageDomain
from .measure import Measure
from .variables import (
    BANDWIDTH,
    PEAK_TENSOR_FLOPS,
    READ,
    SM_COUNT,
    SMEM_PER_SM,
    MAX_PROGRAMS_PER_SM,
    PIPELINE_EFFICIENCY,
    WRITE,
    normal_write,
)
from .work import WorkModel


@dataclass(frozen=True)
class KernelStage:
    name: str
    domain: StageDomain
    buffers: KernelBuffers
    work: WorkModel = WorkModel()
    write_model: Callable = normal_write

    @property
    def read_buffers(self) -> KernelBuffers:
        return self.buffers.read

    @property
    def write_buffers(self) -> KernelBuffers:
        return self.buffers.write

    @property
    def resident_buffers(self) -> KernelBuffers:
        return self.buffers.resident

    @property
    def streamed_read_buffers(self) -> KernelBuffers:
        return self.buffers.read.streamed

    @property
    def persistent_read_buffers(self) -> KernelBuffers:
        return self.buffers.read.persistent

    @property
    def streamed_resident_buffers(self) -> KernelBuffers:
        return self.buffers.resident.streamed

    @property
    def persistent_resident_buffers(self) -> KernelBuffers:
        return self.buffers.resident.persistent

    @property
    def program_count(self):
        return self.domain.tasks

    @property
    def cost(self) -> KernelStageCost:
        return KernelStageCost(self)


@dataclass(frozen=True)
class KernelStageCost:
    stage: KernelStage

    @property
    def nonhiding_traffic(self):
        return (
            READ * self.stage.persistent_read_buffers.accessed_bytes
            + WRITE * self.write_traffic
        )

    @property
    def streamed_traffic(self):
        return READ * self.stage.streamed_read_buffers.accessed_bytes

    @property
    def traffic(self):
        return self.nonhiding_traffic + self.streamed_traffic

    @property
    def effective_traffic(self):
        return self.traffic

    def write_multiplicity(self, buffer):
        writers_per_target = buffer.residual_multiplicity
        active_writers_per_target = self.active_programs / buffer.target_partitions
        return Min(writers_per_target, Max(1, active_writers_per_target))

    def write_bytes(self, buffer):
        return buffer.accessed_bytes * self.stage.write_model(
            buffer.residual_multiplicity,
            self.write_multiplicity(buffer),
        )

    @property
    def write_traffic(self):
        return sum(self.write_bytes(b) for b in self.stage.write_buffers)

    @property
    def materialized_storage_bytes(self):
        return self.stage.write_buffers.materialized.total_bytes

    @property
    def ordinary_storage_bytes(self):
        return self.stage.buffers.ordinary.total_bytes

    @property
    def partial_storage_ratio(self):
        return self.materialized_storage_bytes / Max(1, self.ordinary_storage_bytes)

    @property
    def residency_bytes(self):
        return self.stage.resident_buffers.tile_bytes

    @property
    def pipeline_bytes(self):
        return self.stage.streamed_resident_buffers.tile_bytes

    @property
    def pipeline_stage_capacity(self):
        spare = SaturateSub(
            SMEM_PER_SM,
            self._resident_programs_per_sm_measure * self.residency_bytes,
        )
        denom = self._resident_programs_per_sm_measure * Max(1, self.pipeline_bytes)
        slack = spare / denom
        return Piecewise((0, self.pipeline_bytes <= 0), (slack, True))

    @property
    def loop_tiles(self):
        loop_axis = self.stage.domain.loop_axis
        return (
            1
            if loop_axis is None
            else self.stage.domain.measure(loop_axis.id, Measure.span_tiles)
        )

    @property
    def average_pipeline_stages(self):
        loop_tiles = self.loop_tiles
        capacity = Min(self.pipeline_stage_capacity, loop_tiles)
        return Max(
            0,
            capacity / Max(1, loop_tiles) * (loop_tiles - (capacity + 1) / 2),
        )

    @property
    def pipeline_factor(self):
        return PIPELINE_EFFICIENCY * Min(
            1,
            self.average_pipeline_stages * self.stream_compute_cover,
        )

    @property
    def total_work(self):
        return self.stage.work.work(self.stage.domain, Measure.extent)

    @property
    def tile_work(self):
        return self.stage.work.work(self.stage.domain, Measure.tile)

    @property
    def span_work(self):
        return self.stage.work.work(self.stage.domain, Measure.span)

    @property
    def effective_total_work(self):
        return self.stage.work.effective_work(self.stage.domain, Measure.extent)

    @property
    def effective_tile_work(self):
        return self.stage.work.effective_work(self.stage.domain, Measure.tile)

    @property
    def work_efficiency(self):
        return self.total_work / Max(1, self.effective_total_work)

    @property
    def mma_efficiency(self):
        return self.work_efficiency

    @property
    def compute_time(self):
        return self.effective_total_work / self.effective_peak_flops

    @property
    def model_flops_per_second(self):
        return self.total_work / self.estimated_time

    @property
    def model_flops_utilization(self):
        return self.model_flops_per_second / PEAK_TENSOR_FLOPS

    @property
    def streamed_bytes_per_tile(self):
        return self.stage.streamed_read_buffers.tile_bytes

    @property
    def stream_compute_cover(self):
        return (
            self.effective_tile_work / Max(1, self.streamed_bytes_per_tile)
        ) / self.ridge

    @property
    def traffic_time(self):
        return self.effective_traffic / self.effective_bandwidth

    @property
    def estimated_time(self):
        nonh_time = self.nonhiding_traffic / self.effective_bandwidth
        stream_time = self.streamed_traffic / self.effective_bandwidth
        return (
            nonh_time
            + stream_time
            + SaturateSub(
                self.compute_time,
                stream_time * self.pipeline_factor,
            )
        )

    @property
    def estimated_time_cse(self):
        """Named DAG form of estimated time for inspection or evaluation."""
        return expr.cse(self.estimated_time)

    @property
    def ridge(self):
        return PEAK_TENSOR_FLOPS / BANDWIDTH

    @property
    def resident_programs(self):
        return SM_COUNT * self._resident_programs_per_sm_measure

    @property
    def resident_programs_per_sm(self):
        """Exact occupancy metric, retained for direct stage inspection."""
        capacity = floor(SMEM_PER_SM / Max(1, self.residency_bytes))
        return Min(MAX_PROGRAMS_PER_SM, capacity)

    @property
    def _resident_programs_per_sm_measure(self):
        if not self.stage.domain.resolved:
            return self.resident_programs_per_sm_symbol
        return self.resident_programs_per_sm

    @property
    def resident_programs_per_sm_symbol(self):
        return expr.Symbol(f"{self.stage.name}_resident_programs_per_sm")

    def derived_measure_substitutions(self, substitutions=None):
        """Resolve stage-local derived measures with concrete inputs."""
        def integer(value):
            if type(value) is int:
                return value
            if isinstance(value, Fraction) and value.denominator == 1:
                return value.numerator
            return None

        substitutions = dict(substitutions or {})
        result = self.stage.domain.derived_measure_substitutions(substitutions)
        substitutions.update(result)

        residency = self.residency_bytes
        if isinstance(residency, expr.Expr):
            residency = residency.subs(substitutions)
        smem = substitutions.get(SMEM_PER_SM, substitutions.get(str(SMEM_PER_SM)))
        max_programs = substitutions.get(
            MAX_PROGRAMS_PER_SM,
            substitutions.get(str(MAX_PROGRAMS_PER_SM)),
        )
        if isinstance(smem, expr.Expr):
            smem = smem.subs(substitutions)
        if isinstance(max_programs, expr.Expr):
            max_programs = max_programs.subs(substitutions)
        residency = integer(residency)
        smem = integer(smem)
        max_programs = integer(max_programs)
        if residency is not None and smem is not None and max_programs is not None:
            result[self.resident_programs_per_sm_symbol] = min(
                max_programs,
                smem // max(1, residency),
            )
        return result

    @property
    def active_programs(self):
        return Min(self.stage.program_count, self.resident_programs)

    @property
    def sm_utilization(self):
        return Min(1, self.active_programs / SM_COUNT)

    @property
    def effective_peak_flops(self):
        return PEAK_TENSOR_FLOPS * self.sm_utilization

    @property
    def effective_bandwidth(self):
        return BANDWIDTH * self.sm_utilization
