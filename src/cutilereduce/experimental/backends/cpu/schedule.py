"""CPU scheduling for hardware-independent experimental fold plans."""

from __future__ import annotations

import os
import platform
from dataclasses import dataclass
from pathlib import Path

from cutilereduce.experimental.compiler.operator import FoldProgram
from cutilereduce.experimental.planning.templates import (
    BackwardScheduleTemplate,
    CarrierFold,
    CarrierScan,
    FoldTopology,
    ForwardScheduleTemplate,
    GradientFold,
    Iteration,
    MapFoldRegion,
    ParallelMap,
    PlanBinding,
    PlanExpr,
    Reference,
    RegionCall,
    TrainingScheduleTemplate,
)


from cutilereduce.experimental.planning.bound import (
    BoundLoopKind as CpuLoopKind,
    BoundStorage as CpuStorage,
    BoundIteration as CpuIteration,
    BoundExpr as CpuExpr,
    BoundLeaf as CpuLeaf,
    BoundLoop as CpuLoop,
    BoundBinding as CpuBinding,
    BoundStage as CpuStage,
    BoundPlan as CpuScheduledPlan,
)


def _cache_size(path: str) -> int | None:
    try:
        value = Path(path).read_text().strip().upper()
    except OSError:
        return None
    multiplier = 1024 if value.endswith("K") else 1024 * 1024
    if value[-1:] in {"K", "M"}:
        value = value[:-1]
    try:
        return int(value) * multiplier
    except ValueError:
        return None


@dataclass(frozen=True)
class CpuTarget:
    architecture: str
    worker_count: int
    vector_bytes: int
    cache_line_bytes: int
    l1_data_bytes: int | None = None
    l2_bytes: int | None = None
    l3_bytes: int | None = None

    @classmethod
    def detect(cls) -> CpuTarget:
        architecture = platform.machine().lower()
        vector_bytes = 16
        if architecture in {"x86_64", "amd64"}:
            try:
                import torch

                capability = torch.backends.cpu.get_cpu_capability()
            except (AttributeError, ImportError):
                capability = "DEFAULT"
            vector_bytes = {
                "AVX512": 64,
                "AVX2": 32,
                "VSX": 16,
                "Z VECTOR": 16,
                "DEFAULT": 16,
            }.get(capability, 16)
        return cls(
            architecture=architecture,
            worker_count=max(1, os.cpu_count() or 1),
            vector_bytes=vector_bytes,
            cache_line_bytes=64,
            l1_data_bytes=_cache_size("/sys/devices/system/cpu/cpu0/cache/index0/size"),
            l2_bytes=_cache_size("/sys/devices/system/cpu/cpu0/cache/index2/size"),
            l3_bytes=_cache_size("/sys/devices/system/cpu/cpu0/cache/index3/size"),
        )


@dataclass(frozen=True)
class CpuStageConfig:
    """Physical tensor tiles and scratch budget for one CPU plan stage."""

    name: str
    tile_sizes: tuple[tuple[str, int], ...] = ()
    scratch_bytes: int | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("CPU stage configuration requires a name")
        if len(dict(self.tile_sizes)) != len(self.tile_sizes):
            raise ValueError("CPU stage tile axes must be unique")
        if any(not axis for axis, _ in self.tile_sizes):
            raise ValueError("CPU stage tile axes must be named")
        if any(type(value) is not int or value <= 0 for _, value in self.tile_sizes):
            raise ValueError("CPU stage tile sizes must be positive integers")
        if self.scratch_bytes is not None and self.scratch_bytes <= 0:
            raise ValueError("CPU stage scratch budget must be positive")


@dataclass(frozen=True)
class CpuScheduleConfig:
    chunk_sizes: tuple[tuple[str, int], ...] = ()
    partition_count: int | None = None
    scratch_bytes_per_worker: int | None = None
    stages: tuple[CpuStageConfig, ...] = ()

    def __post_init__(self) -> None:
        if len(dict(self.chunk_sizes)) != len(self.chunk_sizes):
            raise ValueError("CPU chunk axes must be unique")
        if self.scratch_bytes_per_worker is not None and (
            self.scratch_bytes_per_worker <= 0
        ):
            raise ValueError("CPU scratch budget must be positive")
        names = tuple(stage.name for stage in self.stages)
        if len(set(names)) != len(names):
            raise ValueError("CPU stage configurations must have unique names")

    def chunk_size(self, axis: str, extent: int) -> int:
        configured = dict(self.chunk_sizes).get(axis)
        if configured is None:
            return min(extent, 64)
        if configured <= 0:
            raise ValueError("CPU chunk sizes must be positive")
        return min(extent, configured)

    def stage(self, name: str) -> CpuStageConfig | None:
        return next((stage for stage in self.stages if stage.name == name), None)



def _iteration(
    logical: Iteration,
    *,
    sizes: dict[str, int],
    target: CpuTarget,
    config: CpuScheduleConfig,
    requested: CpuLoopKind,
    parallel_active: bool,
) -> CpuIteration:
    extent = sizes[logical.axis]
    if logical.kind == "chunks":
        grain = config.chunk_size(logical.axis, extent)
        count = (extent + grain - 1) // grain
    else:
        count = config.partition_count or min(target.worker_count, extent)
        if count <= 0:
            raise ValueError("CPU partition count must be positive")
        count = min(count, extent)
        grain = (extent + count - 1) // count
    loop = requested
    if requested is CpuLoopKind.PARALLEL and parallel_active:
        loop = CpuLoopKind.SERIAL
    return CpuIteration(logical, extent, grain, count, loop)


def _schedule_expr(
    expression: PlanExpr,
    *,
    sizes: dict[str, int],
    target: CpuTarget,
    config: CpuScheduleConfig,
    parallel_active: bool = False,
) -> CpuExpr:
    if isinstance(expression, (MapFoldRegion, RegionCall, Reference)):
        return CpuLeaf(expression)
    if isinstance(expression, ParallelMap):
        iteration = _iteration(
            expression.iteration,
            sizes=sizes,
            target=target,
            config=config,
            requested=CpuLoopKind.PARALLEL,
            parallel_active=parallel_active,
        )
        return CpuLoop(
            iteration,
            _schedule_expr(
                expression.body,
                sizes=sizes,
                target=target,
                config=config,
                parallel_active=parallel_active
                or iteration.loop is CpuLoopKind.PARALLEL,
            ),
            "map",
        )
    if isinstance(expression, CarrierFold):
        requested = (
            CpuLoopKind.SERIAL
            if expression.topology is FoldTopology.ONLINE
            else CpuLoopKind.PARALLEL
        )
        iteration = _iteration(
            expression.iteration,
            sizes=sizes,
            target=target,
            config=config,
            requested=requested,
            parallel_active=parallel_active,
        )
        return CpuLoop(
            iteration,
            _schedule_expr(
                expression.body,
                sizes=sizes,
                target=target,
                config=config,
                parallel_active=parallel_active
                or iteration.loop is CpuLoopKind.PARALLEL,
            ),
            "carrier_fold",
            expression.topology,
            expression.reorder,
        )
    if isinstance(expression, CarrierScan):
        iteration = _iteration(
            expression.iteration,
            sizes=sizes,
            target=target,
            config=config,
            requested=CpuLoopKind.SERIAL,
            parallel_active=parallel_active,
        )
        return CpuLoop(
            iteration,
            _schedule_expr(
                expression.body,
                sizes=sizes,
                target=target,
                config=config,
                parallel_active=parallel_active,
            ),
            "exclusive_scan" if expression.exclusive else "inclusive_scan",
        )
    if isinstance(expression, GradientFold):
        iteration = _iteration(
            expression.iteration,
            sizes=sizes,
            target=target,
            config=config,
            requested=CpuLoopKind.SERIAL,
            parallel_active=parallel_active,
        )
        return CpuLoop(
            iteration,
            _schedule_expr(
                expression.body,
                sizes=sizes,
                target=target,
                config=config,
                parallel_active=parallel_active,
            ),
            "gradient_fold",
            expression.topology,
        )
    raise TypeError(f"unsupported plan expression: {expression!r}")


def _schedule_stages(
    stages,
    *,
    sizes: dict[str, int],
    target: CpuTarget,
    config: CpuScheduleConfig,
) -> tuple[CpuStage, ...]:
    result = []
    for stage in stages:
        stage_config = config.stage(stage.name)
        configured_tiles = dict(() if stage_config is None else stage_config.tile_sizes)
        unknown = set(configured_tiles) - set(sizes)
        if unknown:
            raise ValueError(
                f"CPU stage {stage.name!r} has unknown tile axes: {sorted(unknown)}"
            )
        tile_sizes = tuple(
            (
                axis,
                min(
                    extent,
                    configured_tiles.get(axis, min(extent, 64)),
                ),
            )
            for axis, extent in sizes.items()
        )
        default_scratch = (
            config.scratch_bytes_per_worker
            if config.scratch_bytes_per_worker is not None
            else min(
                1024 * 1024,
                max(64 * 1024, (target.l2_bytes * 3) // 4),
            )
            if target.l2_bytes is not None
            else 256 * 1024
        )
        scratch_bytes = (
            stage_config.scratch_bytes
            if stage_config is not None and stage_config.scratch_bytes is not None
            else default_scratch
        )
        bindings = []
        for binding in stage.bindings:
            if not isinstance(binding, PlanBinding):
                raise TypeError("invalid plan binding")
            storage = CpuStorage.MATERIALIZED if binding.persist else CpuStorage.PRIVATE
            bindings.append(
                CpuBinding(
                    binding.name,
                    _schedule_expr(
                        binding.expression,
                        sizes=sizes,
                        target=target,
                        config=config,
                    ),
                    storage,
                )
            )
        result.append(
            CpuStage(
                stage.name,
                tuple(bindings),
                tile_sizes=tile_sizes,
                scratch_bytes=scratch_bytes,
            )
        )
    return tuple(result)


def schedule_cpu(
    problem: FoldProgram,
    plan: ForwardScheduleTemplate | BackwardScheduleTemplate | TrainingScheduleTemplate,
    sizes: dict[str, int],
    *,
    target: CpuTarget | None = None,
    config: CpuScheduleConfig | None = None,
) -> CpuScheduledPlan:
    """Bind an abstract plan to concrete CPU loops and storage scopes."""
    expected = set(problem.spec.axes)
    if set(sizes) != expected:
        raise ValueError(
            f"sizes must contain exactly {sorted(expected)}, got {sorted(sizes)}"
        )
    if any(type(value) is not int or value <= 0 for value in sizes.values()):
        raise ValueError("all CPU extents must be positive integers")
    target = target or CpuTarget.detect()
    config = config or CpuScheduleConfig()
    forward_plan = (
        plan.forward
        if isinstance(plan, TrainingScheduleTemplate)
        else (plan if isinstance(plan, ForwardScheduleTemplate) else None)
    )
    backward_plan = (
        plan.backward
        if isinstance(plan, TrainingScheduleTemplate)
        else (plan if isinstance(plan, BackwardScheduleTemplate) else None)
    )
    available_stages = {
        stage.name
        for selected_plan in (forward_plan, backward_plan)
        if selected_plan is not None
        for stage in selected_plan.stages
    }
    unknown_stages = {stage.name for stage in config.stages} - available_stages
    if unknown_stages:
        raise ValueError(
            f"CPU configuration has unknown stages: {sorted(unknown_stages)}"
        )
    forward = (
        ()
        if forward_plan is None
        else _schedule_stages(
            forward_plan.stages,
            sizes=sizes,
            target=target,
            config=config,
        )
    )
    backward = (
        ()
        if backward_plan is None
        else _schedule_stages(
            backward_plan.stages,
            sizes=sizes,
            target=target,
            config=config,
        )
    )
    return CpuScheduledPlan(
        plan,
        tuple((axis, sizes[axis]) for axis in problem.spec.axes),
        target,
        forward,
        backward,
    )


def format_cpu_schedule(plan: CpuScheduledPlan) -> str:
    lines = [f"cpu[{plan.target.architecture}, workers={plan.target.worker_count}] {{"]
    for direction, stages in (("forward", plan.forward), ("backward", plan.backward)):
        if not stages:
            continue
        lines.append(f"  {direction} {{")
        for stage in stages:
            lines.append(f"    stage {stage.name} {{")
            tiles = ", ".join(f"{axis}={tile}" for axis, tile in stage.tile_sizes)
            lines.append(f"      tiles[{tiles}] scratch<={stage.scratch_bytes} bytes")
            for binding in stage.bindings:
                lines.append(f"      {binding.name} storage={binding.storage.value}")
                lines.extend(_format_cpu_expr(binding.expression, 8))
            lines.append("    }")
        lines.append("  }")
    lines.append("}")
    return "\n".join(lines)


def _format_cpu_expr(expression: CpuExpr, indent: int) -> list[str]:
    prefix = " " * indent
    if isinstance(expression, CpuLeaf):
        return [f"{prefix}{type(expression.expression).__name__}"]
    if isinstance(expression, CpuLoop):
        loop = expression.iteration
        line = (
            f"{prefix}{loop.loop.value} {expression.operation} "
            f"{loop.logical.index}: extent={loop.extent}, grain={loop.grain}, "
            f"count={loop.count}"
        )
        return [line, *_format_cpu_expr(expression.body, indent + 2)]
    raise TypeError(f"unknown CPU expression: {expression!r}")


__all__ = [
    "CpuBinding",
    "CpuExpr",
    "CpuIteration",
    "CpuLeaf",
    "CpuLoop",
    "CpuLoopKind",
    "CpuScheduleConfig",
    "CpuScheduledPlan",
    "CpuStage",
    "CpuStageConfig",
    "CpuStorage",
    "CpuTarget",
    "format_cpu_schedule",
    "schedule_cpu",
]
