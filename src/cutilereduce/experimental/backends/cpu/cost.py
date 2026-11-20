"""Benchmark summaries derived from the shared execution-IR cost analysis.

Schedule selection belongs to the shared analytical sweep, not this module.
"""

from dataclasses import dataclass

from cutilereduce.experimental.planning.analysis import PlanningProblem
from cutilereduce.experimental.planning.execution_cost import analyze_bound_plan
from cutilereduce.experimental.planning.ssa import lower_backward_stage
from cutilereduce.experimental.planning.scheduled import schedule_backward_program
from .tuning import cpu_analytical_target


@dataclass(frozen=True)
class CpuPlanCost:
    operations: int
    materialized_bytes: int
    parallel_tasks: int
    serial_steps: int


def estimate_cpu_plan(problem, scheduled, *, backward_program=None):
    """Summarize a concrete plan without a competing selection heuristic."""
    backward = scheduled if scheduled.backward else None
    gradient = backward_program
    if backward is not None and gradient is None:
        stage = backward.backward[-1]
        outer = stage.bindings[0].expression
        fold = outer.body
        gradient = schedule_backward_program(problem, dict(scheduled.sizes),
            loop_axis=fold.iteration.logical.axis,
            tiles=dict(stage.tile_sizes), loop_programs=outer.iteration.count)
    planning = PlanningProblem(problem, cpu_analytical_target(scheduled.target), dict(scheduled.sizes))
    cost = analyze_bound_plan(planning, scheduled, backward=backward,
                              gradient_stage=lower_backward_stage(gradient) if gradient else None)
    return CpuPlanCost(int(cost.work.total), int(sum(s.allocated_bytes for s in cost.stages)),
                       int(max(s.parallel_tasks for s in cost.stages)),
                       int(sum(s.serial_steps for s in cost.stages)))
