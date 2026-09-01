from __future__ import annotations

from dataclasses import dataclass

import cuda.tile as ct

from cutilereduce.core.axis import Axis, axis_id
from cutilereduce.core.buffer import BufferBundle, BufferRole, Input, Internal
from cutilereduce.core.kernel_stage import KernelStage
from cutilereduce.core.stage_buffer import BufferStorage
from cutilereduce.core.stage_domain import ProgramAxis, ProgramAxes, StageDomain
from cutilereduce.core.utilities import ceil_div
from cutilereduce.core.variables import atomic_add_write
from cutilereduce.stages.base import BufferUse, BuiltStage, StageSchedule, bind_buffer_uses
from cutilereduce.stages.codegen import StageFunctions, make_buffer_helper, make_buffer_project, make_buffer_split, stage_grid_info
from cutilereduce.stages.map_fold import batch_program_axes, fold_compute_axes


@dataclass(frozen=True)
class RecomputeFinalizeGradWrite:
    spec: object
    schedule: StageSchedule
    global_buffers: BufferBundle
    output_grad: BufferBundle
    grad_storage: BufferBundle | None = None
    partition_axis: Axis | None = None

    def build(self) -> BuiltStage:
        grad_storage = self.grad_storage or self.spec.grad_storage
        compute_axes = fold_compute_axes(
            self.spec,
            self.schedule,
            fold_axis=self.spec.fold,
            fold_extent=self.schedule.extent(self.spec.fold),
            fold_tile=self.schedule.tile(self.spec.fold),
        )
        loop_axis = self.schedule.loop or self.spec.fold.id
        program_axes = [
            ProgramAxis(
                axis=a.axis,
                source=a.id,
                programs=self.schedule.program(a.axis, ceil_div(a.extent, a.tile)),
            )
            for a in compute_axes.outer
            if a.id != axis_id(loop_axis)
        ]
        if self.partition_axis is not None:
            program_axes.append(ProgramAxis(
                axis=self.partition_axis,
                source=self.partition_axis.source,
                programs=self.schedule.program(self.partition_axis, 1),
            ))
        domain = StageDomain(
            name="recompute_finalize_grad_write",
            compute_axes=compute_axes,
            program_axes=ProgramAxes(values=tuple(program_axes)),
            loop=loop_axis,
        )
        buffers = bind_buffer_uses(domain, (
            BufferUse.read_resident(self.spec.input),
            BufferUse.read_resident(self.global_buffers),
            BufferUse.read_resident(self.output_grad),
            BufferUse.resident(self.spec.execution),
            BufferUse.resident(self.spec.finalize_intermediate),
            BufferUse.write(grad_storage),
        ))
        return BuiltStage(
            stage=KernelStage(
                "recompute_finalize_grad_write",
                domain,
                buffers,
                self.spec.backward_work or self.spec.map_fold_work,
                write_model=atomic_add_write,
            ),
            partition_axis=self.partition_axis,
            compiler=compile_recompute_finalize_grad_write_stage,
        )


@dataclass(frozen=True)
class RecomputeFoldFinalizeGradWrite:
    spec: object
    schedule: StageSchedule
    global_buffers: BufferBundle
    output_grad: BufferBundle
    grad_storage: BufferBundle | None = None

    def build(self) -> BuiltStage:
        grad_storage = self.grad_storage or self.spec.grad_storage
        compute_axes = fold_compute_axes(
            self.spec,
            self.schedule,
            fold_axis=self.spec.fold,
            fold_extent=self.schedule.extent(self.spec.fold),
            fold_tile=self.schedule.tile(self.spec.fold),
        )
        program_axes = batch_program_axes(self.schedule, compute_axes)
        domain = StageDomain(
            name="recompute_fold_finalize_grad_write",
            compute_axes=compute_axes,
            program_axes=ProgramAxes(values=tuple(program_axes)),
            loop=self.schedule.loop or self.spec.fold.id,
        )
        uses = [
            BufferUse.read_resident(self.spec.input),
            BufferUse.read_resident(self.global_buffers),
            BufferUse.read_resident(self.output_grad),
            BufferUse.resident(self.spec.execution),
            BufferUse.resident(self.spec.finalize_intermediate),
            BufferUse.write(grad_storage),
        ]
        buffers = bind_buffer_uses(domain, tuple(uses))
        return BuiltStage(
            stage=KernelStage(
                "recompute_fold_finalize_grad_write",
                domain,
                buffers,
                self.spec.backward_work or self.spec.map_fold_work,
                write_model=atomic_add_write,
            ),
            compiler=compile_recompute_fold_finalize_grad_write_stage,
        )


@dataclass(frozen=True)
class RecomputePrefixFoldFinalizeGradWrite:
    spec: object
    schedule: StageSchedule
    global_buffers: BufferBundle
    output_grad: BufferBundle
    prefix: BufferBundle
    prefix_axis: Axis
    grad_storage: BufferBundle | None = None

    def build(self) -> BuiltStage:
        grad_storage = self.grad_storage or self.spec.grad_storage
        compute_axes = fold_compute_axes(
            self.spec,
            self.schedule,
            fold_axis=self.spec.fold,
            fold_extent=self.schedule.extent(self.spec.fold),
            fold_tile=self.schedule.tile(self.spec.fold),
        )
        program_axes = batch_program_axes(self.schedule, compute_axes)
        program_axes.append(ProgramAxis(
            axis=self.prefix_axis,
            source=self.spec.fold.id,
            programs=self.schedule.program(self.prefix_axis, 1),
        ))
        domain = StageDomain(
            name="recompute_prefix_fold_finalize_grad_write",
            compute_axes=compute_axes,
            program_axes=ProgramAxes(values=tuple(program_axes)),
            loop=self.schedule.loop or self.spec.fold.id,
        )
        buffers = bind_buffer_uses(domain, (
            BufferUse.read_resident(self.spec.input),
            BufferUse.read_resident(self.global_buffers),
            BufferUse.read_resident(self.output_grad),
            BufferUse.read_resident(
                self.prefix,
                BufferStorage.Materialized,
                axis_map={self.prefix_axis.id: self.spec.fold.id},
            ),
            BufferUse.resident(self.spec.execution),
            BufferUse.resident(self.spec.finalize_intermediate),
            BufferUse.write(grad_storage),
        ))
        return BuiltStage(
            stage=KernelStage(
                "recompute_prefix_fold_finalize_grad_write",
                domain,
                buffers,
                self.spec.backward_work or self.spec.map_fold_work,
                write_model=atomic_add_write,
            ),
            checkpoints=self.prefix,
            partition_axis=self.prefix_axis,
            compiler=compile_recompute_prefix_fold_finalize_grad_write_stage,
        )


def _is_output_grad(buffer) -> bool:
    return buffer.role == BufferRole.OutputGrad


def _is_global_read(buffer) -> bool:
    return buffer.role != Input and not _is_output_grad(buffer)


def compile_recompute_finalize_grad_write_stage(stage: BuiltStage, functions: StageFunctions):
    grid = stage_grid_info(stage)
    embed_buffers = tuple(
        b for b in stage.stage.read_buffers
        if _is_global_read(b) or _is_output_grad(b)
    )
    embed_is_persistent = all(b.is_persistent for b in embed_buffers)
    read_split = make_buffer_split(
        stage.stage.read_buffers,
        stage.stage.read_buffers.persistent,
        stage.stage.read_buffers.streamed,
    )
    write_split = make_buffer_split(
        stage.stage.write_buffers,
        stage.stage.write_buffers.persistent,
        stage.stage.write_buffers.streamed,
    )
    read_split_functions = read_split.functions
    write_split_functions = write_split.functions
    read_persistent = make_buffer_helper(read_split.left_buffers)
    read_streamed = make_buffer_helper(read_split.right_buffers)
    write_persistent = make_buffer_helper(write_split.left_buffers)
    write_streamed = make_buffer_helper(write_split.right_buffers)
    input_project = make_buffer_project(
        stage.stage.read_buffers,
        tuple(b for b in stage.stage.read_buffers if b.role == BufferRole.Input),
    )
    persistent_global_project = make_buffer_project(
        read_split.left_buffers,
        tuple(b for b in read_split.left_buffers if _is_global_read(b)),
    )
    persistent_output_grad_project = make_buffer_project(
        read_split.left_buffers,
        tuple(b for b in read_split.left_buffers if _is_output_grad(b)),
    )
    dynamic_global_project = make_buffer_project(
        stage.stage.read_buffers,
        tuple(b for b in stage.stage.read_buffers if _is_global_read(b)),
    )
    dynamic_output_grad_project = make_buffer_project(
        stage.stage.read_buffers,
        tuple(b for b in stage.stage.read_buffers if _is_output_grad(b)),
    )
    finalize = functions.finalize
    embed = functions.embed
    if finalize is None or embed is None:
        raise ValueError("recompute-finalize-grad-write stage requires embed and finalize functions")

    if embed_is_persistent:
        @ct.function
        def loop_body(loop_tid, loop_stage_tid, persistent_grads, persistent_reads, streamed_views, streamed_grad_views, embedded):
            streamed_reads = streamed_views.load(loop_stage_tid)
            read_tiles = read_split_functions.merge(persistent_reads, streamed_reads)
            input_tiles = input_project(read_tiles)
            streamed_grads = write_streamed.init()
            grad_tiles = write_split_functions.merge(persistent_grads, streamed_grads)
            grads = finalize(grid.tid_info(loop_tid), *input_tiles, *grad_tiles, *embedded)
            persistent_grads = write_split_functions.left(grads)
            streamed_grads = write_split_functions.right(grads)
            streamed_grad_views.store_add(loop_stage_tid, streamed_grads)
            return persistent_grads

        loop = grid.loop_with_tail(loop_body)
    else:
        @ct.function
        def loop_body(loop_tid, loop_stage_tid, persistent_grads, persistent_reads, streamed_views, streamed_grad_views):
            streamed_reads = streamed_views.load(loop_stage_tid)
            read_tiles = read_split_functions.merge(persistent_reads, streamed_reads)
            input_tiles = input_project(read_tiles)
            streamed_grads = write_streamed.init()
            grad_tiles = write_split_functions.merge(persistent_grads, streamed_grads)
            embedded = embed(*dynamic_global_project(read_tiles), *dynamic_output_grad_project(read_tiles))
            grads = finalize(grid.tid_info(loop_tid), *input_tiles, *grad_tiles, *embedded)
            persistent_grads = write_split_functions.left(grads)
            streamed_grads = write_split_functions.right(grads)
            streamed_grad_views.store_add(loop_stage_tid, streamed_grads)
            return persistent_grads

        loop = grid.loop_with_tail(loop_body)

    @ct.kernel
    def kernel(read_buffers, write_buffers):
        tid, program_tid = grid.init()
        stage_tid = tid + program_tid
        persistent_read_buffers = read_split_functions.left(read_buffers)
        streamed_read_buffers = read_split_functions.right(read_buffers)
        persistent_write_buffers = write_split_functions.left(write_buffers)
        streamed_write_buffers = write_split_functions.right(write_buffers)
        persistent_reads = read_persistent.load(stage_tid, persistent_read_buffers)
        streamed_views = read_streamed.view(streamed_read_buffers)
        persistent_grad_views = write_persistent.view(persistent_write_buffers)
        streamed_grad_views = write_streamed.view(streamed_write_buffers)
        persistent_grads = write_persistent.init()
        if embed_is_persistent:
            embedded = embed(
                *persistent_global_project(persistent_reads),
                *persistent_output_grad_project(persistent_reads),
            )
            persistent_grads = loop(
                tid,
                program_tid,
                persistent_grads,
                persistent_reads,
                streamed_views,
                streamed_grad_views,
                embedded,
            )
        else:
            persistent_grads = loop(
                tid,
                program_tid,
                persistent_grads,
                persistent_reads,
                streamed_views,
                streamed_grad_views,
            )
        persistent_grad_views.store_add(stage_tid, persistent_grads)

    return kernel


def compile_recompute_fold_finalize_grad_write_stage(stage: BuiltStage, functions):
    return _compile_ordered_fold_grad_write_stage(stage, functions, read_prefix=False)


def compile_recompute_prefix_fold_finalize_grad_write_stage(stage: BuiltStage, functions):
    return _compile_ordered_fold_grad_write_stage(stage, functions, read_prefix=True)


def _is_prefix_buffer(stage: BuiltStage, buffer) -> bool:
    return stage.checkpoints is not None and buffer.id in stage.checkpoints


def _compile_ordered_fold_grad_write_stage(stage: BuiltStage, functions: StageFunctions, *, read_prefix: bool):
    grid = stage_grid_info(stage)
    # General-fold finalize is stateful: it consumes the exclusive fold state
    # before this tile and returns (gradients, inclusive state after the tile).
    # map_reduce_backward remains as a compatibility alias.
    map_reduce_backward = functions.finalize or functions.map_reduce_backward
    embed = functions.embed
    if map_reduce_backward is None or embed is None:
        raise ValueError("ordered fold backward stage requires embed and stateful finalize functions")

    ordinary_read_buffers = tuple(
        b for b in stage.stage.read_buffers
        if not _is_prefix_buffer(stage, b)
    )
    prefix_buffers = tuple(
        b for b in stage.stage.read_buffers
        if _is_prefix_buffer(stage, b)
    )
    if read_prefix and not prefix_buffers:
        raise ValueError("prefix ordered fold backward stage requires prefix read buffers")
    if not read_prefix and prefix_buffers:
        raise ValueError("non-prefix ordered fold backward stage must not have prefix read buffers")

    ordinary_read_project = make_buffer_project(stage.stage.read_buffers, ordinary_read_buffers)
    prefix_project = make_buffer_project(stage.stage.read_buffers, prefix_buffers)
    read_split = make_buffer_split(
        ordinary_read_buffers,
        tuple(b for b in ordinary_read_buffers if b.is_persistent),
        tuple(b for b in ordinary_read_buffers if b.is_streamed),
    )
    write_split = make_buffer_split(
        stage.stage.write_buffers,
        stage.stage.write_buffers.persistent,
        stage.stage.write_buffers.streamed,
    )
    read_split_functions = read_split.functions
    write_split_functions = write_split.functions
    read_persistent = make_buffer_helper(read_split.left_buffers)
    read_streamed = make_buffer_helper(read_split.right_buffers)
    write_persistent = make_buffer_helper(write_split.left_buffers)
    write_streamed = make_buffer_helper(write_split.right_buffers)
    execution = make_buffer_helper(tuple(
        buffer for buffer in stage.stage.resident_buffers.intermediate
        if isinstance(buffer.id.role, Internal)
        and "execution" in buffer.id.role.tags
    ))
    prefix_helper = make_buffer_helper(prefix_buffers)
    input_project = make_buffer_project(
        ordinary_read_buffers,
        tuple(b for b in ordinary_read_buffers if b.role == BufferRole.Input),
    )
    embed_buffers = tuple(
        b for b in ordinary_read_buffers
        if _is_global_read(b) or _is_output_grad(b)
    )
    embed_is_persistent = all(b.is_persistent for b in embed_buffers)
    persistent_global_project = make_buffer_project(
        read_split.left_buffers,
        tuple(b for b in read_split.left_buffers if _is_global_read(b)),
    )
    persistent_output_grad_project = make_buffer_project(
        read_split.left_buffers,
        tuple(b for b in read_split.left_buffers if _is_output_grad(b)),
    )
    dynamic_global_project = make_buffer_project(
        ordinary_read_buffers,
        tuple(b for b in ordinary_read_buffers if _is_global_read(b)),
    )
    dynamic_output_grad_project = make_buffer_project(
        ordinary_read_buffers,
        tuple(b for b in ordinary_read_buffers if _is_output_grad(b)),
    )

    if embed_is_persistent:
        @ct.function
        def loop_body(loop_tid, loop_stage_tid, carry, persistent_reads, streamed_views, streamed_grad_views, embedded):
            persistent_grads, prefix = carry
            streamed_reads = streamed_views.load(loop_stage_tid)
            read_tiles = read_split_functions.merge(persistent_reads, streamed_reads)
            input_tiles = input_project(read_tiles)
            streamed_grads = write_streamed.init()
            grad_tiles = write_split_functions.merge(persistent_grads, streamed_grads)
            grad_tiles, prefix = map_reduce_backward(
                grid.tid_info(loop_tid),
                *input_tiles,
                *grad_tiles,
                *embedded,
                *prefix,
            )
            persistent_grads = write_split_functions.left(grad_tiles)
            streamed_grads = write_split_functions.right(grad_tiles)
            streamed_grad_views.store_add(loop_stage_tid, streamed_grads)
            return persistent_grads, prefix

        loop = grid.loop_with_tail(loop_body)
    else:
        @ct.function
        def loop_body(loop_tid, loop_stage_tid, carry, persistent_reads, streamed_views, streamed_grad_views):
            persistent_grads, prefix = carry
            streamed_reads = streamed_views.load(loop_stage_tid)
            read_tiles = read_split_functions.merge(persistent_reads, streamed_reads)
            input_tiles = input_project(read_tiles)
            streamed_grads = write_streamed.init()
            grad_tiles = write_split_functions.merge(persistent_grads, streamed_grads)
            embedded = embed(*dynamic_global_project(read_tiles), *dynamic_output_grad_project(read_tiles))
            grad_tiles, prefix = map_reduce_backward(
                grid.tid_info(loop_tid),
                *input_tiles,
                *grad_tiles,
                *embedded,
                *prefix,
            )
            persistent_grads = write_split_functions.left(grad_tiles)
            streamed_grads = write_split_functions.right(grad_tiles)
            streamed_grad_views.store_add(loop_stage_tid, streamed_grads)
            return persistent_grads, prefix

        loop = grid.loop_with_tail(loop_body)

    @ct.kernel
    def kernel(read_buffers, write_buffers):
        tid, program_tid = grid.init()
        stage_tid = tid + program_tid
        ordinary_read_args = ordinary_read_project(read_buffers)
        persistent_read_buffers = read_split_functions.left(ordinary_read_args)
        streamed_read_buffers = read_split_functions.right(ordinary_read_args)
        persistent_write_buffers = write_split_functions.left(write_buffers)
        streamed_write_buffers = write_split_functions.right(write_buffers)
        persistent_reads = read_persistent.load(stage_tid, persistent_read_buffers)
        streamed_views = read_streamed.view(streamed_read_buffers)
        persistent_grad_views = write_persistent.view(persistent_write_buffers)
        streamed_grad_views = write_streamed.view(streamed_write_buffers)
        persistent_grads = write_persistent.init()
        if read_prefix:
            prefix = execution.reshape(
                prefix_helper.load(stage_tid, prefix_project(read_buffers))
            )
        else:
            prefix = execution.init()
        carry = (persistent_grads, prefix)
        if embed_is_persistent:
            embedded = embed(
                *persistent_global_project(persistent_reads),
                *persistent_output_grad_project(persistent_reads),
            )
            persistent_grads, _ = loop(
                tid,
                program_tid,
                carry,
                persistent_reads,
                streamed_views,
                streamed_grad_views,
                embedded,
            )
        else:
            persistent_grads, _ = loop(
                tid,
                program_tid,
                carry,
                persistent_reads,
                streamed_views,
                streamed_grad_views,
            )
        persistent_grad_views.store_add(stage_tid, persistent_grads)

    return kernel


__all__ = [
    "RecomputeFinalizeGradWrite",
    "RecomputeFoldFinalizeGradWrite",
    "RecomputePrefixFoldFinalizeGradWrite",
    "compile_recompute_finalize_grad_write_stage",
    "compile_recompute_fold_finalize_grad_write_stage",
    "compile_recompute_prefix_fold_finalize_grad_write_stage",
]
