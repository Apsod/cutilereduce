from __future__ import annotations

from cutilereduce.fold.plan import FoldPlan
from cutilereduce.fold.tune import stage_key, tune_fold_stages

from .sweep import (
    general_backward_stage_from_config,
    general_fold_plan_from_config,
    sweep_general_backward,
    sweep_general_fold,
)


def tune_general_fold_plan(
        spec,
        sizes,
        candidate_count,
        timeout,
        *,
        functions,
        hardware,
        max_tile=128,
        max_partition_count=4,
        quiet_tuning=False,
        backward=False,
        ):
    print("analytical general-fold sweep", flush=True)
    swept = sweep_general_fold(
        spec,
        sizes=sizes,
        hardware=hardware,
        max_tile=max_tile,
        max_partition_count=max_partition_count,
    )
    if swept.is_empty():
        raise RuntimeError("general fold sweep produced no candidates")

    candidates = []
    keys = set()
    for row in swept.iter_rows(named=True):
        plan = general_fold_plan_from_config(spec, sizes=sizes, config=row)
        key = tuple(stage_key(stage) for stage in plan.forward)
        if key not in keys:
            keys.add(key)
            candidates.append(plan)
        if len(candidates) == candidate_count:
            break

    stages = tuple(stage for plan in candidates for stage in plan.forward)
    unique_count = len({stage_key(stage) for stage in stages})
    print(
        f"empirical general forward-stage tuning "
        f"({len(candidates)} configurations, {unique_count} unique kernels)",
        flush=True,
    )
    times = tune_fold_stages(
        stages,
        functions,
        timeout=timeout,
        quiet=quiet_tuning,
    )
    viable = tuple(
        plan for plan in candidates
        if all(stage_key(stage) in times for stage in plan.forward)
    )
    if not viable:
        raise RuntimeError("empirical tuning left no complete general fold plan")
    plan = min(
        viable,
        key=lambda current: sum(times[stage_key(stage)] for stage in current.forward),
    )

    print("isolated stage timings")
    for stage in plan.forward:
        tiles = ", ".join(
            f"{axis.name}={axis.tile}" for axis in stage.stage.domain.compute_axes
        )
        print(
            f"  forward {stage.stage.name} ({tiles}): "
            f"{times[stage_key(stage)]:.1f} us"
        )
    if not backward:
        return plan

    print("analytical general backward sweep", flush=True)
    backward_swept = sweep_general_backward(
        spec,
        sizes=sizes,
        forward_plan=plan,
        hardware=hardware,
        max_tile=max_tile,
    )
    candidates = []
    counts = {}
    keys = set()
    for row in backward_swept.iter_rows(named=True):
        path = row["backward_path"]
        if counts.get(path, 0) >= candidate_count:
            continue
        stage = general_backward_stage_from_config(
            spec, sizes=sizes, forward_plan=plan, config=row,
        )
        key = stage_key(stage)
        if key in keys:
            continue
        keys.add(key)
        candidates.append((path, stage))
        counts[path] = counts.get(path, 0) + 1
    if not candidates:
        raise RuntimeError("general backward sweep produced no candidates")
    print(
        "empirical general backward-stage tuning ("
        + ", ".join(f"{count} {path}" for path, count in counts.items())
        + ")",
        flush=True,
    )
    backward_times = tune_fold_stages(
        tuple(stage for _, stage in candidates),
        functions,
        timeout=timeout,
        quiet=quiet_tuning,
    )
    viable_backward = tuple(
        (path, stage) for path, stage in candidates
        if stage_key(stage) in backward_times
    )
    if not viable_backward:
        raise RuntimeError("empirical tuning left no general backward stage")
    backward_path, backward_stage = min(
        viable_backward,
        key=lambda item: backward_times[stage_key(item[1])],
    )
    tiles = ", ".join(
        f"{axis.name}={axis.tile}"
        for axis in backward_stage.stage.domain.compute_axes
    )
    print(
        f"  backward {backward_path} ({tiles}): "
        f"{backward_times[stage_key(backward_stage)]:.1f} us"
    )
    return FoldPlan.make(spec, plan.forward, (backward_stage,))


__all__ = ["tune_general_fold_plan"]
