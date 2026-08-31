from __future__ import annotations

from copy import replace
from dataclasses import dataclass

from cutilereduce.core.axis import Axis
from cutilereduce.core.buffer import BufferBundle
from cutilereduce.core.kernel_stage import KernelStage
from cutilereduce.core.stage_buffer import BufferStorage
from cutilereduce.core.stage_domain import AxisRole, ProgramAxis, ProgramAxes, StageDomain
from cutilereduce.stages.base import BufferUse, BuiltStage, StageSchedule, bind_buffer_uses
from cutilereduce.stages.map_fold import batch_program_axes, fold_compute_axes


def tag_buffers(bundle: BufferBundle, tag: str) -> BufferBundle:
    return bundle.map(lambda b: replace(b, id=b.id.tag(tag)))


@dataclass(frozen=True)
class Scan:
    spec: object
    schedule: StageSchedule
    scan_axis: Axis
    inputs: BufferBundle
    checkpoints: BufferBundle | None = None
    write_checkpoints: bool = True

    @classmethod
    def make(
            cls,
            spec,
            schedule: StageSchedule,
            *,
            scan_axis: Axis,
            inputs: BufferBundle,
            checkpoint_tag: str = "checkpoint",
            write_checkpoints: bool = True,
            ) -> Scan:
        checkpoints = tag_buffers(inputs, checkpoint_tag) if write_checkpoints else None
        return cls(
            spec=spec,
            schedule=schedule,
            scan_axis=scan_axis,
            inputs=inputs,
            checkpoints=checkpoints,
            write_checkpoints=write_checkpoints,
        )

    def build(self) -> BuiltStage:
        compute_axes = fold_compute_axes(
            self.spec,
            self.schedule,
            fold_axis=self.scan_axis,
            fold_extent=self.schedule.extent(self.scan_axis),
            fold_tile=self.schedule.tile(self.scan_axis),
            fold_role=AxisRole.Scan,
        )
        program_axes = batch_program_axes(self.schedule, compute_axes)
        program_axes.append(ProgramAxis(
            axis=self.scan_axis,
            source=self.scan_axis.id,
            programs=self.schedule.program(self.scan_axis, 1),
        ))
        domain = StageDomain(
            name="scan",
            compute_axes=compute_axes,
            program_axes=ProgramAxes(values=tuple(program_axes)),
            loop=self.schedule.loop or self.scan_axis.id,
        )
        uses = [
            BufferUse.read_resident(self.inputs, BufferStorage.Materialized),
            BufferUse.write(self.spec.output),
        ]
        if self.checkpoints is not None:
            uses.append(BufferUse.write(self.checkpoints, BufferStorage.Materialized))
        buffers = bind_buffer_uses(domain, tuple(uses))
        return BuiltStage(
            stage=KernelStage("scan", domain, buffers, self.spec.combine_work),
            partials=self.inputs,
            partition_axis=self.scan_axis,
            checkpoints=self.checkpoints,
            compiler=compile_scan_stage,
        )


def compile_scan_stage(stage: BuiltStage, functions):
    raise NotImplementedError("cutile scan codegen is not implemented for the new stage core yet")


__all__ = [
    "Scan",
    "compile_scan_stage",
    "tag_buffers",
]
