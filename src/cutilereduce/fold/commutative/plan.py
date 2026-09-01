from __future__ import annotations

import cuda.tile as ct

from cutilereduce.core.axis import Axis
from cutilereduce.core.buffer import BufferBundle
from cutilereduce.fold.plan import FoldPlan, FoldSpec, StageSchedule
from cutilereduce.stages import Fold, MapFold, MapFoldPartial, RecomputeFinalizeGradWrite, partial_buffers

FullFold = MapFold
PartialFold = MapFoldPartial


def output_grad_buffers(spec: FoldSpec, tag: str = "output_grad") -> BufferBundle:
    del tag
    return spec.output.as_output_grad(dtype=ct.float32)


def commutative_backward_stage(
        spec: FoldSpec,
        schedule: StageSchedule,
        *,
        global_buffers: BufferBundle | None = None,
        output_grad: BufferBundle | None = None,
        partition_axis: Axis | None = None,
        ):
    return RecomputeFinalizeGradWrite(
        spec=spec,
        schedule=schedule,
        global_buffers=global_buffers or spec.output,
        output_grad=output_grad or output_grad_buffers(spec),
        partition_axis=partition_axis,
    ).build()


def full_fold_plan(
        spec: FoldSpec,
        schedule: StageSchedule,
        *,
        backward_schedule: StageSchedule | None = None,
        ) -> FoldPlan:
    backward = (
        ()
        if backward_schedule is None
        else (commutative_backward_stage(spec, backward_schedule),)
    )
    return FoldPlan.make(spec, (MapFold(spec, schedule).build(),), backward)


def partial_fold_plan(
        spec: FoldSpec,
        partial_schedule: StageSchedule,
        combine_schedule: StageSchedule,
        *,
        partial_tag: str = "partial",
        backward_schedule: StageSchedule | None = None,
        ) -> FoldPlan:
    partial = MapFoldPartial.make(spec, partial_schedule, partial_tag=partial_tag)
    combine = Fold(
        spec=spec,
        schedule=combine_schedule,
        partition_axis=partial.partition_axis,
        partials=partial.partials,
    )
    backward = (
        ()
        if backward_schedule is None
        else (commutative_backward_stage(spec, backward_schedule),)
    )
    return FoldPlan.make(spec, (partial.build(), combine.build()), backward)


__all__ = [
    "FullFold",
    "PartialFold",
    "commutative_backward_stage",
    "full_fold_plan",
    "output_grad_buffers",
    "partial_buffers",
    "partial_fold_plan",
]
