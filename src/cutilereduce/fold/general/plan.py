from __future__ import annotations

from cutilereduce.fold.plan import FoldPlan, FoldSpec, StageSchedule
from cutilereduce.stages import MapFold, MapFoldPartial, Scan


def full_fold_plan(spec: FoldSpec, schedule: StageSchedule) -> FoldPlan:
    return FoldPlan.make(spec, (MapFold(spec, schedule).build(),))


def partial_fold_plan(
        spec: FoldSpec,
        partial_schedule: StageSchedule,
        scan_schedule: StageSchedule,
        *,
        partial_tag: str = "partial",
        checkpoint_tag: str = "checkpoint",
        write_checkpoints: bool = True,
        ) -> FoldPlan:
    partial = MapFoldPartial.make(spec, partial_schedule, partial_tag=partial_tag)
    scan = Scan.make(
        spec,
        scan_schedule,
        scan_axis=partial.partition_axis,
        inputs=partial.partials,
        checkpoint_tag=checkpoint_tag,
        write_checkpoints=write_checkpoints,
    )
    return FoldPlan.make(spec, (partial.build(), scan.build()))


__all__ = [
    "full_fold_plan",
    "partial_fold_plan",
]
