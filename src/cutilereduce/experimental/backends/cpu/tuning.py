"""CPU adapters for the common analytical and empirical search pipeline."""

from __future__ import annotations

from typing import Any

from cutilereduce.experimental.planning.analysis import BackendModel, PlanningProblem
from cutilereduce.experimental.planning.templates import build_schedule_space
from .schedule import CpuScheduleConfig, CpuStageConfig, CpuTarget
from .analysis import CpuCostModel
from .realization import CpuRegionRealizations


def cpu_analytical_target(target: CpuTarget, **model_overrides) -> BackendModel:
    """Back CPU planning with the emitter's realization and explicit rate priors.

    Capacities come from the supplied target, never implicitly detected here.
    Per-core rates and cache slowdown coefficients are generic uncalibrated
    defaults; callers can override CpuCostModel fields and inspect the metadata.
    """
    from dataclasses import asdict

    lanes = max(1, target.vector_bytes // 4)
    parameters: dict[str, Any] = dict(workers=target.worker_count, vector_lanes=lanes,
                      l2_bytes=target.l2_bytes, l3_bytes=target.l3_bytes,
                      gemm_flops_per_second=lanes * 8e9,
                      scalar_ops_per_second=lanes * 2e9)
    model = CpuCostModel(**(parameters | model_overrides))
    def realize(problem, candidate):
        from .backend import CpuBackend
        from cutilereduce.experimental.planning.backend import BackendProgram
        prepared = BackendProgram(problem.program, target, problem.target,
            build_schedule_space(problem.program), tuple(sorted(problem.sizes.items())))
        return CpuBackend().realize(prepared, candidate)

    return BackendModel(
        "cpu", workers=model.workers, hardware_identity=asdict(target),
        contraction_flops_per_second=model.gemm_flops_per_second * model.workers,
        scalar_ops_per_second=model.scalar_ops_per_second * model.workers,
        transcendental_ops_per_second=model.transcendental_ops_per_second * model.workers,
        bytes_per_second=model.memory_bytes_per_second,
        region_analyzer=CpuRegionRealizations().analyze, stage_estimator=model.estimate_stage,
        realization_factory=realize,
        model_metadata={"name": "cpu_realization_v1", "assumptions": asdict(model)},
    )


def candidate_cpu_configs(problem, candidate):
    """Translate concrete search decisions to the existing native CPU scheduler."""
    sizes = dict(problem.sizes)
    parameters = dict(candidate.parameters)
    fold = problem.program.spec.fold_axes[0]
    space = build_schedule_space(problem.program)
    plan = next(p for p in space.forward if p.name == candidate.family)
    tile_axes = (*problem.program.spec.batch_axes, fold)

    def tiles(prefix):
        return tuple((a, parameters.get(f"{prefix}.{a}", parameters.get(f"tile.{a}", sizes[a])))
                     for a in tile_axes)

    forward = CpuScheduleConfig(
        chunk_sizes=((fold, parameters[f"tile.{fold}"]),),
        partition_count=parameters.get("partition_count", 1),
        stages=tuple(CpuStageConfig(stage.name, tiles("tile")) for stage in plan.stages),
    )
    backward = CpuScheduleConfig(
        chunk_sizes=((fold, dict(tiles("backward_tile"))[fold]),),
        partition_count=parameters.get("partition_count", 1),
        stages=tuple(CpuStageConfig(stage.name, tiles("embed_tile" if stage.name == "embed_output_vjp" else "backward_tile"))
                     for stage in space.backward[0].stages) if candidate.backward else (),
    )
    return plan, forward, backward


def compile_cpu_candidate(operator, problem, candidate, *, target=None):
    from .backend import CpuBackend
    from cutilereduce.experimental.planning.backend import BackendProgram

    estimate = problem.evaluate(candidate)
    if not estimate.accepted:
        raise ValueError(str(estimate.rejection_reasons))
    if estimate.implementation is not None:
        return CpuBackend().compile(estimate.implementation)
    target = target or CpuTarget.detect()
    prepared = BackendProgram(problem.program, target, problem.target,
                              build_schedule_space(problem.program),
                              tuple(sorted(problem.sizes.items())))
    backend = CpuBackend()
    return backend.compile(backend.realize(prepared, candidate))



def tune_cpu(operator, sizes, inputs, *, reference, target=None, analytical_target=None,
             backward=False, candidate_count=20, sweep=None, warmup=3, iterations=20,
             rtol=1e-4, atol=1e-5):
    from .backend import CpuBackend
    from cutilereduce.experimental.planning.backend import BackendProgram
    from cutilereduce.experimental.planning.driver import AnalysisReport, tune_report
    target = target or CpuTarget.detect()
    problem = PlanningProblem(operator.lower(backward=backward),
                              analytical_target or cpu_analytical_target(target), sizes)
    if problem.target.backend != "cpu":
        raise ValueError("tune_cpu requires a CPU analytical target")
    backend = CpuBackend()
    prepared = BackendProgram(problem.program, target, problem.target,
        build_schedule_space(problem.program), tuple(sorted(sizes.items())))
    sweep = problem.sweep(backward=backward) if sweep is None else sweep
    report = AnalysisReport(backend, prepared, problem, sweep)
    return tune_report(report, inputs, reference=reference, execution=backend.execution_adapter(),
                       backward=backward, candidate_count=candidate_count,
                       warmup=warmup, iterations=iterations, rtol=rtol, atol=atol)
