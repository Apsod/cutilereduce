"""CPU adapters for the common analytical and empirical search pipeline."""

from __future__ import annotations

from typing import Any

from cutilereduce.experimental.planning.analysis import AnalyticalTarget, PlanningProblem
from cutilereduce.experimental.planning.templates import build_schedule_space
from cutilereduce.experimental.planning.tuning import tune_candidates
from .schedule import CpuScheduleConfig, CpuStageConfig, CpuTarget
from .analysis import CpuCostModel, analyze_cpu_region


def cpu_analytical_target(target: CpuTarget, **model_overrides) -> AnalyticalTarget:
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
    return AnalyticalTarget(
        "cpu", workers=model.workers,
        contraction_flops_per_second=model.gemm_flops_per_second * model.workers,
        scalar_ops_per_second=model.scalar_ops_per_second * model.workers,
        transcendental_ops_per_second=model.transcendental_ops_per_second * model.workers,
        bytes_per_second=model.memory_bytes_per_second,
        region_analyzer=analyze_cpu_region, stage_estimator=model.estimate_stage,
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
    from .runtime import compile_cpu
    from cutilereduce.experimental.planning.scheduled import schedule_backward_program

    estimate = problem.evaluate(candidate)
    if not estimate.accepted:
        raise ValueError(str(estimate.rejection_reasons))
    plan, forward, backward = candidate_cpu_configs(problem, candidate)
    scheduled_backward = None
    if candidate.backward and not problem.program.requires_prefix:
        parameters = dict(candidate.parameters)
        tiles = {a: parameters.get(f"backward_tile.{a}", parameters.get(f"tile.{a}", size))
                 for a, size in problem.sizes.items()}
        scheduled_backward = schedule_backward_program(
            problem.program, dict(problem.sizes), loop_axis=candidate.loop_axis or operator.spec.fold_axes[0],
            tiles=tiles, loop_programs=parameters.get("loop_programs", 1),
        )
    return compile_cpu(operator, dict(problem.sizes), plan=plan, schedule=forward,
                       backward=candidate.backward, backward_schedule=backward,
                       backward_program=scheduled_backward, target=target)


def validate_callable(executable, inputs, reference, *, backward=False, rtol=1e-4, atol=1e-5):
    import torch

    actual_inputs = tuple(x.detach().clone().requires_grad_(x.requires_grad) for x in inputs)
    reference_inputs = tuple(x.detach().clone().requires_grad_(x.requires_grad) for x in inputs)
    actual = executable(*actual_inputs)
    expected = reference(*reference_inputs)
    actual = actual if isinstance(actual, tuple) else (actual,)
    expected = expected if isinstance(expected, tuple) else (expected,)
    if len(actual) != len(expected):
        raise AssertionError("reference output arity mismatch")
    for a, b in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, b, rtol=rtol, atol=atol, check_dtype=False)
    if backward:
        lhs = tuple(x for x in actual_inputs if x.requires_grad)
        rhs = tuple(x for x in reference_inputs if x.requires_grad)
        if not lhs:
            raise ValueError("backward validation needs differentiable inputs")
        cotangents = tuple(torch.ones_like(x) for x in actual)
        ag = torch.autograd.grad(actual, lhs, cotangents, allow_unused=True)
        bg = torch.autograd.grad(expected, rhs, tuple(g.to(x.dtype) for g, x in zip(cotangents, expected, strict=True)), allow_unused=True)
        for x, a, b in zip(lhs, ag, bg, strict=True):
            torch.testing.assert_close(torch.zeros_like(x) if a is None else a,
                                       torch.zeros_like(x) if b is None else b,
                                       rtol=rtol, atol=atol, check_dtype=False)


def tune_cpu(operator, sizes, inputs, *, reference, target=None, analytical_target=None,
             backward=False, candidate_count=20, sweep=None, warmup=3, iterations=20,
             rtol=1e-4, atol=1e-5):
    from .benchmark import benchmark_cpu, benchmark_cpu_training

    target = target or CpuTarget.detect()
    problem = PlanningProblem(operator.lower(backward=backward),
                              analytical_target or cpu_analytical_target(target), sizes)
    if problem.target.backend != "cpu":
        raise ValueError("tune_cpu requires a CPU analytical target")

    def measure(executable):
        if backward:
            return benchmark_cpu_training(executable, inputs, warmup=warmup,
                                          iterations=iterations).full_median_seconds
        return benchmark_cpu(executable, inputs, warmup=warmup,
                             iterations=iterations).median_seconds

    return tune_candidates(
        problem, compile_candidate=lambda c: compile_cpu_candidate(operator, problem, c, target=target),
        validate=lambda f: validate_callable(f, inputs, reference, backward=backward, rtol=rtol, atol=atol),
        measure=measure, sweep=sweep, backward=backward, candidate_count=candidate_count,
    )
