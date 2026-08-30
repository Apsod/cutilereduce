from __future__ import annotations

from dataclasses import dataclass

from cutilereduce.core.axis import Axis
from cutilereduce.core.buffer import BufferBundle
from cutilereduce.core.kernel_stage import KernelStage
from cutilereduce.core.stage_buffer import BufferStorage
from cutilereduce.core.stage_domain import ProgramAxis, ProgramAxes, StageDomain
from cutilereduce.core.variables import atomic_add_write
from cutilereduce.stages.base import BufferUse, BuiltStage, StageKind, StageSchedule, bind_buffer_uses
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
        program_axes = batch_program_axes(self.schedule, compute_axes)
        if self.partition_axis is not None:
            program_axes.append(ProgramAxis(
                axis=self.partition_axis,
                source=self.spec.fold.id,
                programs=self.schedule.program(self.partition_axis, 1),
            ))
        domain = StageDomain(
            name="recompute_finalize_grad_write",
            compute_axes=compute_axes,
            program_axes=ProgramAxes(values=tuple(program_axes)),
            loop=self.schedule.loop or self.spec.fold.id,
        )
        buffers = bind_buffer_uses(domain, (
            BufferUse.read_resident(self.spec.input),
            BufferUse.read_resident(self.global_buffers),
            BufferUse.read_resident(self.output_grad),
            BufferUse.resident(self.spec.execution),
            BufferUse.write(grad_storage),
        ))
        return BuiltStage(
            kind=StageKind.RecomputeFinalizeGradWrite,
            stage=KernelStage(
                "recompute_finalize_grad_write",
                domain,
                buffers,
                self.spec.map_fold_work,
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
    prefix: BufferBundle | None = None
    grad_storage: BufferBundle | None = None
    prefix_axis: Axis | None = None

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
        if self.prefix_axis is not None:
            program_axes.append(ProgramAxis(
                axis=self.prefix_axis,
                source=self.prefix_axis.id,
                programs=self.schedule.program(self.prefix_axis, 1),
            ))
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
            BufferUse.write(grad_storage),
        ]
        if self.prefix is not None:
            uses.append(BufferUse.read_resident(
                self.prefix,
                BufferStorage.Materialized,
                axis_map={self.prefix_axis.id: self.spec.fold.id} if self.prefix_axis is not None else {},
            ))
        buffers = bind_buffer_uses(domain, tuple(uses))
        return BuiltStage(
            kind=StageKind.RecomputeFoldFinalizeGradWrite,
            stage=KernelStage(
                "recompute_fold_finalize_grad_write",
                domain,
                buffers,
                self.spec.map_fold_work,
                write_model=atomic_add_write,
            ),
            checkpoints=self.prefix,
            partition_axis=self.prefix_axis,
            compiler=compile_recompute_fold_finalize_grad_write_stage,
        )


def compile_recompute_finalize_grad_write_stage(stage: BuiltStage, functions):
    if stage.kind != StageKind.RecomputeFinalizeGradWrite:
        raise ValueError(f"expected recompute-finalize-grad-write stage, got {stage.kind}")
    raise NotImplementedError("cutile recompute-finalize-grad-write codegen is not implemented yet")


def compile_recompute_fold_finalize_grad_write_stage(stage: BuiltStage, functions):
    if stage.kind != StageKind.RecomputeFoldFinalizeGradWrite:
        raise ValueError(f"expected recompute-fold-finalize-grad-write stage, got {stage.kind}")
    raise NotImplementedError("cutile recompute-fold-finalize-grad-write codegen is not implemented yet")


__all__ = [
    "RecomputeFinalizeGradWrite",
    "RecomputeFoldFinalizeGradWrite",
    "compile_recompute_finalize_grad_write_stage",
    "compile_recompute_fold_finalize_grad_write_stage",
]
