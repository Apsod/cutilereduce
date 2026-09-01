from __future__ import annotations

from dataclasses import dataclass

import cuda.tile as ct

from cutilereduce.core.axis import Axis, Axes
from cutilereduce.core.buffer import BufferBundle, Internal
from cutilereduce.core.kernel_stage import KernelStage
from cutilereduce.core.stage_buffer import BufferStorage
from cutilereduce.core.stage_domain import (
    AxisRole,
    ProgramAxis,
    ProgramAxes,
    StageAxis,
    StageAxes,
    StageDomain,
)
from cutilereduce.core.utilities import ceil_div
from cutilereduce.stages.base import BufferUse, BuiltStage, StageSchedule, bind_buffer_uses
from cutilereduce.stages.codegen import StageFunctions, identity, make_buffer_helper, make_buffer_split, stage_grid_info


def stage_axis(schedule: StageSchedule, axis: Axis, role: AxisRole) -> StageAxis:
    return StageAxis(
        axis=axis,
        role=role,
        extent=schedule.extent(axis),
        tile=schedule.tile(axis),
    )


def fold_compute_axes(
        spec,
        schedule: StageSchedule,
        *,
        fold_axis: Axis,
        fold_extent,
        fold_tile,
        fold_role: AxisRole = AxisRole.Fold,
        ) -> StageAxes:
    axes = []
    for axis in spec.axes:
        if axis == spec.fold:
            continue
        role = AxisRole.Batch if axis in spec.batch else AxisRole.Inner
        axes.append(stage_axis(schedule, axis, role))
    axes.append(StageAxis(
        axis=fold_axis,
        role=fold_role,
        extent=fold_extent,
        tile=fold_tile,
    ))
    return StageAxes(values=tuple(axes))


def batch_program_axes(schedule: StageSchedule, compute_axes: StageAxes) -> list[ProgramAxis]:
    return [
        ProgramAxis(
            axis=a.axis,
            source=a.id,
            programs=schedule.program(a.axis, ceil_div(a.extent, a.tile)),
        )
        for a in compute_axes.outer
        if a.role == AxisRole.Batch
    ]


def partial_buffers(spec, partition_axis: Axis, partial_tag: str = "partial") -> BufferBundle:
    return spec.execution.map(
        lambda b: b.with_prefix_axes(partial_tag, Axes(values=(partition_axis,)))
    )


@dataclass(frozen=True)
class MapFold:
    spec: object
    schedule: StageSchedule
    map_reduce: object | None = None
    combine: object | None = None
    initial: object | None = None
    to_semantic: object | None = None

    def build(self) -> BuiltStage:
        compute_axes = fold_compute_axes(
            self.spec,
            self.schedule,
            fold_axis=self.spec.fold,
            fold_extent=self.schedule.extent(self.spec.fold),
            fold_tile=self.schedule.tile(self.spec.fold),
        )
        domain = StageDomain(
            name="map_fold",
            compute_axes=compute_axes,
            program_axes=ProgramAxes(values=tuple(batch_program_axes(self.schedule, compute_axes))),
            loop=self.schedule.loop or self.spec.fold.id,
        )
        buffers = bind_buffer_uses(domain, (
            BufferUse.read_resident(self.spec.input),
            BufferUse.resident(self.spec.execution),
            BufferUse.resident(self.spec.map_intermediate),
            BufferUse.write(self.spec.output),
        ))
        return BuiltStage(
            stage=KernelStage("map_fold", domain, buffers, self.spec.map_fold_work),
            compiler=lambda stage, functions: compile_map_fold_stage(
                stage,
                functions,
                map_reduce=self.map_reduce,
                combine=self.combine,
                initial=self.initial,
                to_semantic=self.to_semantic,
            ),
        )


@dataclass(frozen=True)
class MapFoldPartial:
    spec: object
    schedule: StageSchedule
    partition_axis: Axis
    partials: BufferBundle
    map_reduce: object | None = None
    combine: object | None = None
    initial: object | None = None

    @classmethod
    def make(
            cls,
            spec,
            schedule: StageSchedule,
            *,
            partial_tag: str = "partial",
            map_reduce=None,
            combine=None,
            initial=None,
            ) -> MapFoldPartial:
        partition_axis = spec.fold.partition_axis
        return cls(
            spec=spec,
            schedule=schedule,
            partition_axis=partition_axis,
            partials=partial_buffers(spec, partition_axis, partial_tag),
            map_reduce=map_reduce,
            combine=combine,
            initial=initial,
        )

    @property
    def partition_count(self):
        return self.schedule.program(self.partition_axis)

    def build(self) -> BuiltStage:
        compute_axes = fold_compute_axes(
            self.spec,
            self.schedule,
            fold_axis=self.spec.fold,
            fold_extent=self.schedule.extent(self.spec.fold),
            fold_tile=self.schedule.tile(self.spec.fold),
        )
        program_axes = batch_program_axes(self.schedule, compute_axes)
        program_axes.append(ProgramAxis(
            axis=self.partition_axis,
            source=self.spec.fold.id,
            programs=self.partition_count,
        ))
        domain = StageDomain(
            name="map_fold_partial",
            compute_axes=compute_axes,
            program_axes=ProgramAxes(values=tuple(program_axes)),
            loop=self.schedule.loop or self.spec.fold.id,
        )
        buffers = bind_buffer_uses(domain, (
            BufferUse.read_resident(self.spec.input),
            BufferUse.resident(self.spec.execution),
            BufferUse.resident(self.spec.map_intermediate),
            BufferUse.write(
                self.partials,
                BufferStorage.Materialized,
                axis_map={self.partition_axis.id: self.spec.fold.id},
            ),
        ))
        return BuiltStage(
            stage=KernelStage("map_fold_partial", domain, buffers, self.spec.map_fold_work),
            partials=self.partials,
            partition_axis=self.partition_axis,
            compiler=lambda stage, functions: compile_map_fold_partial_stage(
                stage,
                functions,
                map_reduce=self.map_reduce,
                combine=self.combine,
                initial=self.initial,
            ),
        )


def make_map_fold_program(stage: BuiltStage, map_reduce, combine, initial=None):
    grid = stage_grid_info(stage)
    execution = make_buffer_helper(tuple(
        buffer for buffer in stage.stage.resident_buffers.intermediate
        if isinstance(buffer.id.role, Internal)
        and "execution" in buffer.id.role.tags
    ))
    if map_reduce is None or combine is None:
        raise ValueError("map-fold stage requires map_reduce and combine functions")
    read_split = make_buffer_split(
        stage.stage.read_buffers,
        stage.stage.read_buffers.persistent,
        stage.stage.read_buffers.streamed,
    )
    split = read_split.functions
    read_persistent = make_buffer_helper(read_split.left_buffers)
    read_streamed = make_buffer_helper(read_split.right_buffers)

    @ct.function
    def loop_body(loop_tid, loop_stage_tid, acc, persistent_inputs, streamed_views):
        streamed_inputs = streamed_views.load(loop_stage_tid)
        inputs = split.merge(persistent_inputs, streamed_inputs)
        local = map_reduce(grid.tid_info(loop_tid), *inputs)
        return combine(*acc, *local)

    loop = grid.loop_with_tail(loop_body)

    @ct.function
    def map_fold_program(tid, program_tid, read_buffers):
        persistent_buffers = split.left(read_buffers)
        streamed_buffers = split.right(read_buffers)

        persistent_inputs = read_persistent.load(tid + program_tid, persistent_buffers)
        streamed_views = read_streamed.view(streamed_buffers)

        if initial is None:
            acc = execution.init()
        else:
            acc = initial()
        return loop(tid, program_tid, acc, persistent_inputs, streamed_views)

    return map_fold_program


def _compile_map_fold_like_stage(
        stage: BuiltStage,
        functions: StageFunctions | None,
        *,
        write_carrier: bool,
        map_reduce=None,
        combine=None,
        initial=None,
        to_semantic=None,
        ):
    grid = stage_grid_info(stage)
    write = make_buffer_helper(stage.stage.write_buffers)
    if functions is not None:
        map_reduce = map_reduce or functions.map_reduce
        combine = combine or functions.combine
        to_semantic = to_semantic or functions.to_semantic
    to_semantic = to_semantic or identity
    map_fold_program = make_map_fold_program(
        stage,
        map_reduce,
        combine,
        initial,
    )

    @ct.kernel
    def kernel(read_buffers, write_buffers):
        tid, program_tid = grid.init()
        stage_tid = tid + program_tid

        acc = map_fold_program(tid, program_tid, read_buffers)

        out = acc if write_carrier else to_semantic(*acc)
        write.store(stage_tid, write_buffers, out)

    return kernel


def compile_map_fold_stage(stage: BuiltStage, functions=None, **kwargs):
    return _compile_map_fold_like_stage(stage, functions, write_carrier=False, **kwargs)


def compile_map_fold_partial_stage(stage: BuiltStage, functions=None, **kwargs):
    return _compile_map_fold_like_stage(stage, functions, write_carrier=True, **kwargs)


__all__ = [
    "MapFold",
    "MapFoldPartial",
    "batch_program_axes",
    "compile_map_fold_stage",
    "compile_map_fold_partial_stage",
    "fold_compute_axes",
    "make_map_fold_program",
    "partial_buffers",
    "stage_axis",
]
