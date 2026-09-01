from __future__ import annotations

import cuda.tile as ct

from cutilereduce.fold.plan import FoldPlan, FoldSpec, StageSchedule
from cutilereduce.stages import (
    MapFold,
    MapFoldPartial,
    RecomputeFoldFinalizeGradWrite,
    RecomputePrefixFoldFinalizeGradWrite,
    Scan,
)


def output_grad_buffers(spec: FoldSpec):
    return spec.output.as_output_grad(dtype=ct.float32)


def full_recompute_backward_stage(spec, schedule, *, global_buffers=None, output_grad=None):
    return RecomputeFoldFinalizeGradWrite(
        spec,
        schedule,
        global_buffers=global_buffers or spec.output,
        output_grad=output_grad or output_grad_buffers(spec),
    ).build()


def prefix_recompute_backward_stage(spec, schedule, *, checkpoints, partition_axis, global_buffers=None, output_grad=None):
    return RecomputePrefixFoldFinalizeGradWrite(
        spec,
        schedule,
        global_buffers=global_buffers or spec.output,
        output_grad=output_grad or output_grad_buffers(spec),
        prefix=checkpoints,
        prefix_axis=partition_axis,
    ).build()


def full_fold_plan(spec: FoldSpec, schedule: StageSchedule, *, backward_schedule=None) -> FoldPlan:
    backward = () if backward_schedule is None else (
        full_recompute_backward_stage(spec, backward_schedule),
    )
    return FoldPlan.make(spec, (MapFold(spec, schedule).build(),), backward)


def partial_fold_plan(
        spec: FoldSpec,
        partial_schedule: StageSchedule,
        scan_schedule: StageSchedule,
        *,
        partial_tag: str = "partial",
        backward_schedule: StageSchedule | None = None,
        checkpointed_backward: bool = False,
        ) -> FoldPlan:
    partial = MapFoldPartial.make(spec, partial_schedule, partial_tag=partial_tag)
    scan = Scan.make(
        spec,
        scan_schedule,
        scan_axis=partial.partition_axis,
        inputs=partial.partials,
        outputs=spec.output,
        exclusive=True,
    )
    backward = ()
    if backward_schedule is not None:
        backward = (
            prefix_recompute_backward_stage(
                spec,
                backward_schedule,
                checkpoints=scan.carriers,
                partition_axis=partial.partition_axis,
            ) if checkpointed_backward else full_recompute_backward_stage(
                spec, backward_schedule,
            ),
        )
    return FoldPlan.make(spec, (partial.build(), scan.build()), backward)


__all__ = [
    "full_fold_plan",
    "full_recompute_backward_stage",
    "output_grad_buffers",
    "partial_fold_plan",
    "prefix_recompute_backward_stage",
]
