from __future__ import annotations

from collections import Counter

from cutilereduce.fold.plan import FoldPlan
from cutilereduce.fold.tune import stage_key, tune_fold_stages

from .sweep import (
    commutative_backward_stage_from_config,
    commutative_fold_plan_from_config,
    sweep_commutative_backward,
    sweep_commutative_fold,
)


def tune_commutative_fold_plan(
        spec,
        sizes,
        candidate_count,
        timeout,
        *,
        functions,
        hardware,
        quiet_tuning=False,
        ):
    print("analytical sweep", flush=True)
    swept = sweep_commutative_fold(spec, sizes=sizes, hardware=hardware, max_tile=128)
    backward_swept = sweep_commutative_backward(
        spec,
        sizes=sizes,
        hardware=hardware,
        max_tile=128,
    )
    if swept.is_empty() or backward_swept.is_empty():
        raise RuntimeError("commutative fold sweep produced no candidates")

    forward_candidates = []
    forward_keys = set()
    for row in swept.iter_rows(named=True):
        plan = commutative_fold_plan_from_config(
            spec,
            sizes=sizes,
            config=row,
        )
        key = tuple(stage_key(stage) for stage in plan.forward)
        if key not in forward_keys:
            forward_keys.add(key)
            forward_candidates.append(plan)
        if len(forward_candidates) == candidate_count:
            break
    backward_candidates_by_loop = {
        axis: [] for axis in sorted(set(backward_swept["loop_axis"].to_list()))
    }
    backward_keys_by_loop = {axis: set() for axis in backward_candidates_by_loop}
    for row in backward_swept.iter_rows(named=True):
        loop_axis = row["loop_axis"]
        candidates = backward_candidates_by_loop[loop_axis]
        keys = backward_keys_by_loop[loop_axis]
        if len(candidates) == candidate_count:
            continue
        stage = commutative_backward_stage_from_config(spec, sizes=sizes, config=row)
        key = stage_key(stage)
        if key not in keys:
            keys.add(key)
            candidates.append(stage)
    backward_candidates = [
        stage
        for candidates in backward_candidates_by_loop.values()
        for stage in candidates
    ]

    forward_stages = tuple(stage for plan in forward_candidates for stage in plan.forward)
    backward_stages = tuple(backward_candidates)
    forward_plan_counts = Counter(
        "single" if len(plan.forward) == 1 else "partial"
        for plan in forward_candidates
    )
    unique_forward_stages = {
        stage_key(stage): stage for stage in forward_stages
    }
    forward_kernel_counts = Counter(
        stage.stage.name for stage in unique_forward_stages.values()
    )
    plan_summary = ", ".join(
        f"{count} {kind}" for kind, count in forward_plan_counts.items()
    )
    kernel_summary = ", ".join(
        f"{count} {name}" for name, count in forward_kernel_counts.items()
    )
    print(
        f"empirical forward-stage tuning "
        f"({len(forward_candidates)} plans: {plan_summary}; "
        f"{len(unique_forward_stages)} unique kernels: {kernel_summary})",
        flush=True,
    )
    forward_stage_times = tune_fold_stages(
        forward_stages,
        functions,
        timeout=timeout,
        quiet=quiet_tuning,
    )
    print(
        f"empirical backward-stage tuning "
        f"({len(backward_candidates)} configurations, "
        f"{len({stage_key(stage) for stage in backward_stages})} unique kernels)",
        flush=True,
    )
    backward_stage_times = tune_fold_stages(
        backward_stages,
        functions,
        timeout=timeout,
        quiet=quiet_tuning,
    )
    viable_forward = tuple(
        plan for plan in forward_candidates
        if all(stage_key(stage) in forward_stage_times for stage in plan.forward)
    )
    viable_backward = tuple(
        stage for stage in backward_candidates
        if stage_key(stage) in backward_stage_times
    )
    if not viable_forward or not viable_backward:
        raise RuntimeError("empirical tuning left no complete commutative fold plan")
    best_forward = min(
        viable_forward,
        key=lambda plan: sum(forward_stage_times[stage_key(stage)] for stage in plan.forward),
    )
    best_backward = min(
        viable_backward,
        key=lambda stage: backward_stage_times[stage_key(stage)],
    )
    plan = FoldPlan.make(spec, best_forward.forward, (best_backward,))

    print("isolated stage timings")
    for stage in plan.forward:
        tiles = ", ".join(f"{axis.name}={axis.tile}" for axis in stage.stage.domain.compute_axes)
        print(f"  forward {stage.stage.name} ({tiles}): {forward_stage_times[stage_key(stage)]:.1f} us")
    backward_stage = plan.backward[0]
    tiles = ", ".join(f"{axis.name}={axis.tile}" for axis in backward_stage.stage.domain.compute_axes)
    loop_axis = backward_stage.stage.domain.loop_axis
    loop_program_axis = backward_stage.stage.domain.program_axis_for(loop_axis)
    groups = 1 if loop_program_axis is None else loop_program_axis.programs
    print(
        f"  backward {backward_stage.stage.name} "
        f"({tiles}; loop={loop_axis.name}, groups={groups}): "
        f"{backward_stage_times[stage_key(backward_stage)]:.1f} us"
    )
    return plan



__all__ = ["tune_commutative_fold_plan"]
