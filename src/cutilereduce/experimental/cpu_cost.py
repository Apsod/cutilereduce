"""Simple analytical costs for experimental CPU schedules."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .cpu import (
    CpuExpr,
    CpuLeaf,
    CpuLoop,
    CpuLoopKind,
    CpuScheduleConfig,
    CpuScheduledPlan,
    CpuTarget,
    schedule_cpu,
)
from .ir import ContractionAxes, TensorRegion
from .operator import FoldProgram
from .planning import MapFoldRegion, Reference, RegionCall
from .planning import ForwardScheduleTemplate


_DTYPE_BYTES = {
    "bool": 1,
    "int8": 1,
    "float16": 2,
    "bfloat16": 2,
    "int16": 2,
    "float32": 4,
    "int32": 4,
    "float64": 8,
    "int64": 8,
}


@dataclass(frozen=True)
class CpuPlanCost:
    operations: int
    materialized_bytes: int
    parallel_tasks: int
    serial_steps: int

    def score(self, *, workers: int, vector_bytes: int) -> float:
        parallelism = max(1, min(workers, self.parallel_tasks))
        vector_lanes = max(1, vector_bytes // 4)
        compute = self.operations / parallelism / vector_lanes
        return compute + self.materialized_bytes / 16.0 + self.serial_steps


@dataclass(frozen=True)
class CpuPlanChoice:
    plan: ForwardScheduleTemplate
    config: CpuScheduleConfig
    scheduled: CpuScheduledPlan
    cost: CpuPlanCost


def _elements(axes: tuple[str, ...], sizes: dict[str, int]) -> int:
    return math.prod(sizes[axis] for axis in axes)


def estimate_region_work(region: TensorRegion, sizes: dict[str, int]) -> int:
    """Count scalar operations represented by one region invocation."""
    work = 0
    for node in region.nodes:
        elements = _elements(node.axes, sizes)
        if node.op in {"argument", "constant", "zero", "full", "cast", "stop_gradient"}:
            continue
        if node.op in {"contract", "contract_accumulate"}:
            partition = node.attribute("contraction_axes")
            if not isinstance(partition, ContractionAxes):
                raise TypeError("contract node has invalid axis metadata")
            reduction = _elements(partition.reduce, sizes)
            work += 2 * elements * reduction
        elif node.op.startswith("reduce_"):
            over = node.attribute("over")
            work += elements * _elements(over, sizes)
        elif node.op in {"exp", "log", "log1p", "expm1", "sin", "cos", "tanh", "sigmoid", "sqrt", "rsqrt"}:
            work += 4 * elements
        elif node.op == "powi":
            from .ops import multiplication_chain
            exponent = node.attribute("exponent")
            work += elements * (len(multiplication_chain(abs(exponent))) + (exponent < 0) if exponent else 0)
        else:
            work += elements
    return work


def _region(problem: FoldProgram, leaf: CpuLeaf) -> TensorRegion | None:
    expression = leaf.expression
    if isinstance(expression, MapFoldRegion):
        return problem.map_fold
    if isinstance(expression, RegionCall):
        value = getattr(problem, expression.region)
        return value if isinstance(value, TensorRegion) else None
    if isinstance(expression, Reference):
        return None
    raise TypeError(f"unknown CPU leaf: {expression!r}")


def _expr_cost(
    problem: FoldProgram,
    expression: CpuExpr,
    sizes: dict[str, int],
) -> tuple[int, int, int]:
    if isinstance(expression, CpuLeaf):
        region = _region(problem, expression)
        return (0 if region is None else estimate_region_work(region, sizes), 1, 1)
    if not isinstance(expression, CpuLoop):
        raise TypeError(f"unknown CPU expression: {expression!r}")
    local_sizes = dict(sizes)
    local_sizes[expression.iteration.logical.axis] = expression.iteration.grain
    work, tasks, serial = _expr_cost(problem, expression.body, local_sizes)
    count = expression.iteration.count
    work *= count
    if expression.operation == "carrier_fold" and count > 1:
        work += (count - 1) * estimate_region_work(
            problem.execution_combine, sizes
        )
    if expression.iteration.loop is CpuLoopKind.PARALLEL:
        tasks *= count
    else:
        serial *= count
    return work, tasks, serial


def _result_bytes(
    region: TensorRegion, sizes: dict[str, int], *, multiplicity: int = 1
) -> int:
    return multiplicity * sum(
        _elements(region.nodes[node_id].axes, sizes)
        * _DTYPE_BYTES[region.nodes[node_id].dtype]
        for _, node_id in region.results
    )


def estimate_cpu_plan(
    problem: FoldProgram, scheduled: CpuScheduledPlan
) -> CpuPlanCost:
    sizes = dict(scheduled.sizes)
    operations = 0
    materialized = 0
    tasks = 1
    serial = 1
    for stage in (*scheduled.forward, *scheduled.backward):
        for binding in stage.bindings:
            work, local_tasks, local_serial = _expr_cost(
                problem, binding.expression, sizes
            )
            operations += work
            tasks = max(tasks, local_tasks)
            serial += local_serial
            if binding.storage.value != "materialized":
                continue
            if binding.name in {"global_execution", "partial_execution", "prefix_execution"}:
                multiplicity = 1
                if binding.name != "global_execution" and isinstance(
                    binding.expression, CpuLoop
                ):
                    multiplicity = binding.expression.iteration.count
                materialized += _result_bytes(
                    problem.map_fold, sizes, multiplicity=multiplicity
                )
            elif binding.name == "output":
                materialized += _result_bytes(problem.project_output, sizes)
    return CpuPlanCost(operations, materialized, tasks, serial)


def select_cpu_plan(
    problem: FoldProgram,
    plans: tuple[ForwardScheduleTemplate, ...],
    sizes: dict[str, int],
    *,
    target: CpuTarget | None = None,
) -> CpuPlanChoice:
    """Rank a compact deterministic set of CPU schedules analytically."""
    if not plans:
        raise ValueError("cannot select from an empty plan set")
    if len(problem.spec.fold_axes) != 1:
        raise NotImplementedError("initial CPU selection requires one fold axis")
    target = target or CpuTarget.detect()
    axis = problem.spec.fold_axes[0]
    extent = sizes[axis]
    chunks = tuple(
        sorted({extent, min(extent, 128), min(extent, 64), min(extent, 32)})
    )
    choices = []
    for plan in plans:
        partition_counts = (
            (1,)
            if plan.name in {"online", "fused_tree"}
            else tuple(sorted({1, min(extent, target.worker_count)}))
        )
        for chunk in chunks:
            for partitions in partition_counts:
                config = CpuScheduleConfig(
                    chunk_sizes=((axis, chunk),),
                    partition_count=partitions,
                )
                scheduled = schedule_cpu(
                    problem, plan, sizes, target=target, config=config
                )
                cost = estimate_cpu_plan(problem, scheduled)
                choices.append(CpuPlanChoice(plan, config, scheduled, cost))
    return min(
        choices,
        key=lambda choice: choice.cost.score(
            workers=target.worker_count,
            vector_bytes=target.vector_bytes,
        ),
    )


__all__ = [
    "CpuPlanChoice", "CpuPlanCost", "estimate_cpu_plan",
    "estimate_region_work", "select_cpu_plan",
]
