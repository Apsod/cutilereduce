from __future__ import annotations

from dataclasses import dataclass

import cuda.tile as ct

from cutilereduce.core.axis import Axis
from cutilereduce.core.buffer import BufferBundle
from cutilereduce.core.kernel_stage import KernelStage
from cutilereduce.core.stage_buffer import BufferStorage
from cutilereduce.core.stage_domain import ProgramAxis, ProgramAxes, StageDomain
from cutilereduce.stages.base import BufferUse, BuiltStage, StageKind, StageSchedule, bind_buffer_uses
from cutilereduce.stages.codegen import StageFunctions, identity, make_buffer_helper, stage_grid_info
from cutilereduce.stages.map_fold import batch_program_axes, fold_compute_axes


@dataclass(frozen=True)
class Fold:
    spec: object
    schedule: StageSchedule
    partition_axis: Axis
    partials: BufferBundle

    def build(self) -> BuiltStage:
        compute_axes = fold_compute_axes(
            self.spec,
            self.schedule,
            fold_axis=self.partition_axis,
            fold_extent=self.schedule.extent(self.partition_axis),
            fold_tile=self.schedule.tile(self.partition_axis),
        )
        program_axes = batch_program_axes(self.schedule, compute_axes)
        program_axes.append(ProgramAxis(
            axis=self.partition_axis,
            source=self.partition_axis.id,
            programs=self.schedule.program(self.partition_axis, 1),
        ))
        domain = StageDomain(
            name="fold",
            compute_axes=compute_axes,
            program_axes=ProgramAxes(values=tuple(program_axes)),
            loop=self.schedule.loop or self.partition_axis.id,
        )
        buffers = bind_buffer_uses(domain, (
            BufferUse.read_resident(self.partials, BufferStorage.Materialized),
            BufferUse.write(self.spec.output),
        ))
        return BuiltStage(
            kind=StageKind.Fold,
            stage=KernelStage("fold", domain, buffers, self.spec.combine_work),
            partials=self.partials,
            partition_axis=self.partition_axis,
            compiler=compile_fold_stage,
        )


def compile_fold_stage(stage: BuiltStage, functions: StageFunctions):
    if stage.kind != StageKind.Fold:
        raise ValueError(f"expected fold stage, got {stage.kind}")
    grid = stage_grid_info(stage)
    read = make_buffer_helper(stage.stage.read_buffers)
    write = make_buffer_helper(stage.stage.write_buffers)
    combine = functions.combine
    to_semantic = functions.to_semantic or identity
    to_output = functions.to_output or identity
    if combine is None:
        raise ValueError("fold stage requires combine function")

    @ct.kernel
    def kernel(read_buffers, write_buffers):
        tid, program_tid = grid.init()
        stage_tid = tid + program_tid
        loop_offset, loop_size = grid.loop_offset_and_size(program_tid)
        loop_tid = grid.set_loop_index(tid, loop_offset)
        acc = read.load(loop_tid + program_tid, read_buffers)
        for i in range(1, loop_size):
            loop_tid = grid.set_loop_index(tid, loop_offset + i)
            acc = combine(*acc, *read.load(loop_tid + program_tid, read_buffers))
        out = to_output(*to_semantic(*acc))
        write.store(stage_tid, write_buffers, out)

    return kernel


__all__ = [
    "Fold",
    "compile_fold_stage",
]
