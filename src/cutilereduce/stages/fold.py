from __future__ import annotations

from dataclasses import dataclass

import cuda.tile as ct

from cutilereduce.core.axis import Axis
from cutilereduce.core.buffer import BufferBundle
from cutilereduce.core.kernel_stage import KernelStage
from cutilereduce.core.stage_buffer import BufferStorage
from cutilereduce.core.stage_domain import ProgramAxis, ProgramAxes, StageDomain
from cutilereduce.stages.base import BufferUse, BuiltStage, StageSchedule, bind_buffer_uses
from cutilereduce.stages.codegen import StageFunctions, identity, make_buffer_helper, stage_grid_info
from cutilereduce.stages.map_fold import batch_program_axes, fold_compute_axes


@dataclass(frozen=True)
class Fold:
    spec: object
    schedule: StageSchedule
    partition_axis: Axis
    partials: BufferBundle
    combine: object | None = None
    initial: object | None = None
    to_semantic: object | None = None

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
            stage=KernelStage("fold", domain, buffers, self.spec.combine_work),
            partials=self.partials,
            partition_axis=self.partition_axis,
            compiler=lambda stage, functions: compile_fold_stage(
                stage,
                functions,
                combine=self.combine,
                initial=self.initial,
                to_semantic=self.to_semantic,
            ),
        )


def make_fold_program(stage: BuiltStage, combine, initial=None):
    grid = stage_grid_info(stage)
    read_buffers = tuple(stage.stage.read_buffers)
    load = make_buffer_helper(read_buffers).load
    if combine is None:
        raise ValueError("fold stage requires combine function")

    if stage.partition_axis is None:
        raise ValueError("fold stage has no partition axis")
    fold_axis = stage.partition_axis.id
    fold_positions = tuple(
        tuple(axis.id for axis in buffer.storage_axes).index(fold_axis)
        for buffer in read_buffers
    )
    if len(set(fold_positions)) != 1:
        raise ValueError(f"fold axis has inconsistent tile positions: {fold_positions}")
    tile_axis = fold_positions[0]
    fold_tile = int(stage.domain.get(fold_axis).tile)
    reduced_shapes = tuple(
        info.tile.shape[:tile_axis] + info.tile.shape[tile_axis + 1:]
        for info in read_buffers
    )
    identities = tuple(buffer.default for buffer in read_buffers)

    if fold_tile > 1:
        def load_carrier(tid, buffers):
            factors = load(tid, buffers)
            return ct.reduce(
                factors,
                axis=tile_axis,
                func=combine,
                identity=identities,
            )
    else:
        def load_carrier(tid, buffers):
            factors = load(tid, buffers)
            ret = ()
            for i, shape in ct.static_iter(enumerate(reduced_shapes)):
                ret += (ct.reshape(factors[i], shape),)
            return ret

    @ct.function
    def loop_body(loop_tid, loop_stage_tid, acc, read_buffers):
        return combine(*acc, *load_carrier(loop_stage_tid, read_buffers))


    if initial is None:
        loop = grid.loop_with_tail(loop_body, start=1)
        @ct.function
        def fold_program(tid, program_tid, read_buffers):
            loop_offset, _, _ = grid.loop_offset_and_size(program_tid)
            loop_tid = grid.set_loop_index(tid, loop_offset)
            acc = load_carrier(loop_tid + program_tid, read_buffers)
            return loop(tid, program_tid, acc, read_buffers)
    else:
        loop = grid.loop_with_tail(loop_body, start=0)

        @ct.function
        def fold_program(tid, program_tid, read_buffers):
            acc = initial()
            return loop(tid, program_tid, acc, read_buffers)

    return fold_program


def compile_fold_stage(
        stage: BuiltStage,
        functions: StageFunctions | None = None,
        *,
        combine=None,
        initial=None,
        to_semantic=None,
        ):
    grid = stage_grid_info(stage)
    write = make_buffer_helper(stage.stage.write_buffers)
    if functions is not None:
        combine = combine or functions.combine
        to_semantic = to_semantic or functions.to_semantic
    to_semantic = to_semantic or identity
    fold_program = make_fold_program(stage, combine, initial)

    @ct.kernel
    def kernel(read_buffers, write_buffers):
        tid, program_tid = grid.init()
        stage_tid = tid + program_tid
        acc = fold_program(tid, program_tid, read_buffers)
        out = to_semantic(*acc)
        write.store(stage_tid, write_buffers, out)

    return kernel


__all__ = [
    "Fold",
    "compile_fold_stage",
    "make_fold_program",
]
