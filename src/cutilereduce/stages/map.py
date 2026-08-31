from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from cutilereduce.core.axis import AxisId, Axes
from cutilereduce.core.kernel_stage import KernelStage
from cutilereduce.core.stage_domain import AxisRole, ProgramAxis, ProgramAxes, StageAxis, StageAxes, StageDomain
from cutilereduce.core.utilities import ceil_div
from cutilereduce.core.work import WorkModel
from cutilereduce.stages.base import BufferUse, BuiltStage, StageSchedule, bind_buffer_uses


@dataclass(frozen=True)
class Map:
    name: str
    schedule: StageSchedule
    axes: Axes
    buffer_uses: tuple[BufferUse, ...]
    work: WorkModel = WorkModel()
    roles: Mapping[AxisId, AxisRole] | None = None

    def build(self) -> BuiltStage:
        roles = self.roles or {}
        compute_axes = StageAxes(values=tuple(
            StageAxis(
                axis=axis,
                role=roles.get(axis.id, AxisRole.Batch),
                extent=self.schedule.extent(axis),
                tile=self.schedule.tile(axis),
            )
            for axis in self.axes
        ))
        program_axes = ProgramAxes(values=tuple(
            ProgramAxis(
                axis=axis.axis,
                source=axis.id,
                programs=self.schedule.program(axis.axis, ceil_div(axis.extent, axis.tile)),
            )
            for axis in compute_axes.outer
        ))
        domain = StageDomain(
            name=self.name,
            compute_axes=compute_axes,
            program_axes=program_axes,
            loop=None,
        )
        return BuiltStage(
            stage=KernelStage(self.name, domain, bind_buffer_uses(domain, self.buffer_uses), self.work),
            compiler=compile_map_stage,
        )


def compile_map_stage(stage: BuiltStage, functions):
    raise NotImplementedError("cutile map codegen is not implemented for the new stage core yet")


__all__ = [
    "Map",
    "compile_map_stage",
]
