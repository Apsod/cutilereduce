from __future__ import annotations

from cutilereduce.fold.plan import FoldPlan, FoldSpec, StageSchedule
from cutilereduce.stages import Fold, MapFold, MapFoldPartial, partial_buffers

FullFold = MapFold
PartialFold = MapFoldPartial
FoldPartial = Fold


def full_fold_plan(spec: FoldSpec, schedule: StageSchedule) -> FoldPlan:
    return FoldPlan.make(spec, (MapFold(spec, schedule).build(),))


def partial_fold_plan(
        spec: FoldSpec,
        partial_schedule: StageSchedule,
        combine_schedule: StageSchedule,
        *,
        partial_tag: str = "partial",
        ) -> FoldPlan:
    partial = MapFoldPartial.make(spec, partial_schedule, partial_tag=partial_tag)
    combine = Fold(
        spec=spec,
        schedule=combine_schedule,
        partition_axis=partial.partition_axis,
        partials=partial.partials,
    )
    return FoldPlan.make(spec, (partial.build(), combine.build()))


__all__ = [
    "FoldPartial",
    "FullFold",
    "PartialFold",
    "full_fold_plan",
    "partial_buffers",
    "partial_fold_plan",
]
